"""Memory tiers, retrieval, and the memory stream.

Five separable things live here, because they answer different questions:

  WorkingMemory    what the current task needs to see. Bounded, prioritized,
                   pinned items survive eviction. This is the "short-term" /
                   "working" tier and also what "in-context memory" means in
                   practice: whatever render() emits is what goes in the prompt.
  LongTermMemory   what survived the task. Append-only JSONL, one record per
                   line, searchable by keyword and by vector.
  VectorStore      semantic recall over those records. Ships a dependency-free
                   hashed embedder (blake2b over words and char 3-grams) so
                   the agent works offline; pass a real embedder function to
                   swap in an API or local model.
  GraphRAG         entity/relation recall: extract triples, expand k hops,
                   and summarise communities. Useful when the answer depends
                   on how facts connect, not on which chunk is closest.
  MemoryStream     a Generative-Agents-style append-only event log scored by
                   recency x importance x relevance, which emits reflections
                   once accumulated importance crosses a threshold.

Record shape in ``agentic_memory.jsonl`` (one JSON object per line, UTC epoch
seconds — no local-time ambiguity):

    {"id": "a1b2c3d4e5f6a7b8", "ts": 1770000000.0, "kind": "fact",
     "text": "tests run with pytest -q", "importance": 0.6,
     "tags": ["project"], "vector": [0.01, -0.02]}

The vector is stored so a reload is instant and does not depend on the
embedder staying byte-identical across versions.
"""

from __future__ import annotations

import json
import math
import re
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from agentic.base import cosine, extract_json, stable_hash, truncate

EMBED_DIM = 512
_WORD = re.compile(r"[A-Za-z][A-Za-z0-9_]+|[一-鿿]")


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

def hashed_embed(text: str, dim: int = EMBED_DIM) -> List[float]:
    """Deterministic, dependency-free embedding via feature hashing.

    Words, lowercased char 3-grams, and CJK characters all contribute. This is
    a bag-of-features proxy, not a semantic model: it matches paraphrases that
    share vocabulary and misses synonyms. Good enough for a local recall tier,
    and swappable — see ``VectorStore(embedder=...)``.
    """
    vector = [0.0] * dim
    text = (text or "").lower()
    grams: List[str] = []
    for token in _WORD.findall(text):
        grams.append("w:" + token)
        if len(token) > 3 and token.isascii():
            for i in range(len(token) - 2):
                grams.append("g:" + token[i : i + 3])
    if not grams and text.strip():
        grams = ["c:" + text.strip()[:40]]
    for gram in grams:
        digest = stable_hash(gram)
        bucket = int(digest[:8], 16) % dim
        sign = 1.0 if int(digest[8:10], 16) % 2 == 0 else -1.0
        vector[bucket] += sign
    norm = math.sqrt(sum(v * v for v in vector))
    if norm > 0:
        vector = [v / norm for v in vector]
    return vector


Embedder = Callable[[str], Sequence[float]]


# ---------------------------------------------------------------------------
# Working memory (short-term / in-context)
# ---------------------------------------------------------------------------

@dataclass
class WorkingItem:
    """One slot in working memory."""

    key: str
    text: str
    importance: float = 0.5
    pinned: bool = False


class WorkingMemory:
    """Bounded scratchpad with pinning — the short-term tier.

    When over capacity, unpinned items with the lowest importance are evicted
    first; ties break oldest-first. Pinned items never evict, so the task
    statement and the user's hard constraints cannot be pushed out by chatter.
    """

    def __init__(self, capacity: int = 20) -> None:
        self.capacity = max(1, capacity)
        self._items: Dict[str, WorkingItem] = {}
        self._order: List[str] = []

    def put(self, key: str, text: str, importance: float = 0.5, pinned: bool = False) -> None:
        """Insert or update an item, evicting if over capacity."""
        self._items[key] = WorkingItem(key, text.strip(), importance, pinned)
        if key not in self._order:
            self._order.append(key)
        self._evict()

    def get(self, key: str) -> Optional[str]:
        item = self._items.get(key)
        return item.text if item else None

    def drop(self, key: str) -> bool:
        if key in self._items:
            del self._items[key]
            self._order = [k for k in self._order if k != key]
            return True
        return False

    def _evict(self) -> None:
        while len(self._items) > self.capacity:
            evictable = [k for k in self._order if not self._items[k].pinned]
            if not evictable:
                return  # everything pinned: stay over capacity rather than lose it
            victim = min(
                evictable, key=lambda k: (self._items[k].importance, self._order.index(k))
            )
            self.drop(victim)

    def items(self) -> List[WorkingItem]:
        return [self._items[k] for k in self._order if k in self._items]

    def clear(self) -> None:
        self._items.clear()
        self._order.clear()

    def render(self, max_chars: int = 6000) -> str:
        """Prompt-ready block, highest importance first."""
        if not self._items:
            return ""
        ordered = sorted(self.items(), key=lambda i: (-i.importance, not i.pinned))
        blocks: List[str] = []
        used = 0
        for item in ordered:
            line = f"- {item.text}"
            if used + len(line) > max_chars:
                break
            blocks.append(line)
            used += len(line) + 1
        return "\n".join(blocks)


# ---------------------------------------------------------------------------
# Long-term memory + vector index
# ---------------------------------------------------------------------------

@dataclass
class MemoryRecord:
    """One durable memory."""

    id: str
    text: str
    kind: str = "fact"
    importance: float = 0.5
    tags: List[str] = field(default_factory=list)
    ts: float = field(default_factory=time.time)
    vector: List[float] = field(default_factory=list)

    def to_json(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "ts": self.ts,
            "kind": self.kind,
            "text": self.text,
            "importance": self.importance,
            "tags": self.tags,
            "vector": self.vector,
        }

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> "MemoryRecord":
        return cls(
            id=str(data.get("id") or stable_hash(str(data.get("text", "")))),
            text=str(data.get("text") or ""),
            kind=str(data.get("kind") or "fact"),
            importance=float(data.get("importance") or 0.5),
            tags=[str(t) for t in data.get("tags") or []],
            ts=float(data.get("ts") or 0.0),
            vector=[float(v) for v in data.get("vector") or []],
        )


class VectorStore:
    """Brute-force cosine index over MemoryRecords.

    Small collections (thousands) don't need FAISS; a linear scan in pure
    Python is milliseconds and has no build step. ``embedder`` defaults to the
    hashed one — inject a real model here when you have one:

        VectorStore(embedder=lambda t: my_embeddings(t))
    """

    def __init__(self, embedder: Optional[Embedder] = None) -> None:
        self.embedder: Embedder = embedder or hashed_embed
        self._records: Dict[str, MemoryRecord] = {}

    def add(self, record: MemoryRecord) -> MemoryRecord:
        if not record.vector:
            record.vector = [float(v) for v in self.embedder(record.text)]
        self._records[record.id] = record
        return record

    def remove(self, record_id: str) -> bool:
        return self._records.pop(record_id, None) is not None

    def __len__(self) -> int:
        return len(self._records)

    def records(self) -> List[MemoryRecord]:
        return list(self._records.values())

    def search(
        self, query: str, k: int = 5, min_score: float = 0.05
    ) -> List[Tuple[MemoryRecord, float]]:
        if not self._records:
            return []
        qvec = self.embedder(query)
        scored = []
        for record in self._records.values():
            score = cosine(qvec, record.vector)
            if score >= min_score:
                scored.append((record, score))
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[: max(1, k)]


class LongTermMemory:
    """Append-only JSONL memory with keyword and vector recall.

    Compaction is not implemented on purpose: this tier is meant to grow and
    be pruned by an explicit call, not silently rewritten. Use ``prune`` with
    a predicate (e.g. importance below a floor).
    """

    def __init__(self, path: Path | str, embedder: Optional[Embedder] = None) -> None:
        self.path = Path(path)
        self.store = VectorStore(embedder)
        self._load()

    def _load(self) -> None:
        if not self.path.is_file():
            return
        try:
            with self.path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        self.store.add(MemoryRecord.from_json(json.loads(line)))
                    except (json.JSONDecodeError, TypeError, ValueError):
                        continue  # skip a corrupt line rather than drop the file
        except OSError:
            pass

    def remember(
        self,
        text: str,
        kind: str = "fact",
        importance: float = 0.5,
        tags: Optional[Sequence[str]] = None,
    ) -> MemoryRecord:
        """Store a memory; identical text is deduplicated by content hash."""
        text = (text or "").strip()
        record = MemoryRecord(
            id=stable_hash(text),
            text=text,
            kind=kind,
            importance=max(0.0, min(1.0, importance)),
            tags=list(tags or []),
        )
        self.store.add(record)
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record.to_json(), ensure_ascii=False) + "\n")
        except OSError:
            pass  # memory is best-effort, never fatal
        return record

    def recall(
        self, query: str, k: int = 5, kind: Optional[str] = None
    ) -> List[MemoryRecord]:
        hits = self.store.search(query, k=k * 3 if kind else k)
        if kind:
            hits = [pair for pair in hits if pair[0].kind == kind]
        return [record for record, _ in hits[:k]]

    def keyword(self, query: str, k: int = 10) -> List[MemoryRecord]:
        """Literal token match — the complement to vector recall."""
        terms = [t for t in re.split(r"\W+", query.lower()) if len(t) > 2]
        if not terms:
            return []
        scored = []
        for record in self.store.records():
            haystack = record.text.lower()
            hits = sum(1 for term in terms if term in haystack)
            if hits:
                scored.append((hits, record))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [record for _, record in scored[:k]]

    def hybrid(self, query: str, k: int = 5) -> List[MemoryRecord]:
        """Vector and keyword results interleaved, deduplicated.

        Interleaving instead of concatenating keeps exact-match hits from being
        buried by a long tail of weak vector neighbours.
        """
        vector_hits = self.recall(query, k=k)
        keyword_hits = self.keyword(query, k=k)
        out: List[MemoryRecord] = []
        seen = set()
        for pair in zip(vector_hits, keyword_hits):
            for record in pair:
                if record.id not in seen:
                    seen.add(record.id)
                    out.append(record)
        for record in vector_hits + keyword_hits:
            if len(out) >= k:
                break
            if record.id not in seen:
                seen.add(record.id)
                out.append(record)
        return out[:k]

    def prune(self, keep: Callable[[MemoryRecord], bool]) -> int:
        """Rewrite the file keeping records where ``keep`` is true; returns dropped count."""
        kept = [r for r in self.store.records() if keep(r)]
        dropped = len(self.store.records()) - len(kept)
        if dropped == 0:
            return 0
        self.store = VectorStore(self.store.embedder)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as handle:
            for record in kept:
                self.store.add(record)
                handle.write(json.dumps(record.to_json(), ensure_ascii=False) + "\n")
        return dropped

    def render(self, records: Sequence[MemoryRecord], max_chars: int = 4000) -> str:
        lines: List[str] = []
        used = 0
        for record in records:
            line = f"- [{record.kind}] {record.text}"
            if used + len(line) > max_chars:
                break
            lines.append(line)
            used += len(line) + 1
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Graph RAG
# ---------------------------------------------------------------------------

TRIPLE_SYSTEM = (
    "Extract facts as a knowledge graph. For the text, output ONLY a JSON array "
    'of triples: [{{"subject": "...", "relation": "...", "object": "..."}}]. '
    "Use short noun phrases. Skip anything not stated. At most {limit} triples."
)

COMMUNITY_SYSTEM = (
    "Summarize this cluster of related facts in two sentences, naming the "
    "entities and what connects them. Output the summary only."
)


@dataclass
class Triple:
    subject: str
    relation: str
    object: str
    source: str = ""


class GraphRAG:
    """Entity graph built from text, queried by hop expansion.

    Extraction prefers the model (structured triples) and falls back to a
    regex over "<subject> is/are/has/was <object>" sentences when the model is
    unavailable or returns junk, so ingestion never blocks.

    Query flow: find seed nodes matching the query, expand ``hops`` outward
    with a decay per hop, then render the connecting triples as context. This
    answers "how is X related to Y" better than chunk retrieval, which cannot
    see across documents.
    """

    def __init__(self) -> None:
        self.triples: List[Triple] = []
        self._adjacency: Dict[str, List[Triple]] = {}
        self._parent: Dict[str, str] = {}  # union-find parents

    # -- construction ------------------------------------------------------

    def add_triple(self, subject: str, relation: str, obj: str, source: str = "") -> None:
        s, o = _norm(subject), _norm(obj)
        if not s or not o or s == o:
            return
        triple = Triple(subject.strip(), relation.strip() or "related_to", obj.strip(), source)
        self.triples.append(triple)
        self._adjacency.setdefault(s, []).append(triple)
        self._adjacency.setdefault(o, []).append(triple)
        self._union(s, o)

    def ingest(self, model: Any, text: str, source: str = "", limit: int = 20) -> int:
        """Extract triples from text with the model; regex fallback. Returns count."""
        added = 0
        reply = ""
        try:
            reply = str(
                model.chat(
                    [
                        {"role": "system", "content": TRIPLE_SYSTEM.format(limit=limit)},
                        {"role": "user", "content": truncate(text, 8000)},
                    ]
                ).get("content")
                or ""
            )
        except Exception:
            reply = ""
        parsed = extract_json(reply)
        if isinstance(parsed, list):
            for item in parsed:
                if not isinstance(item, dict):
                    continue
                subject = item.get("subject") or item.get("s")
                relation = item.get("relation") or item.get("p") or "related_to"
                obj = item.get("object") or item.get("o")
                if subject and obj:
                    self.add_triple(str(subject), str(relation), str(obj), source)
                    added += 1
        if added == 0:
            for match in re.finditer(
                r"([A-Z][\w\- ]{1,40}?)\s+(is|are|was|were|has|have|uses|depends on|requires)"
                r"\s+([^.;\n]{2,80})",
                text,
            ):
                self.add_triple(match.group(1), match.group(2), match.group(3), source)
                added += 1
        return added

    def _union(self, a: str, b: str) -> None:
        root_a, root_b = self._find(a), self._find(b)
        if root_a != root_b:
            self._parent[root_b] = root_a

    def _find(self, node: str) -> str:
        self._parent.setdefault(node, node)
        while self._parent[node] != node:
            self._parent[node] = self._parent[self._parent[node]]
            node = self._parent[node]
        return node

    # -- query -------------------------------------------------------------

    def seeds(self, query: str, limit: int = 5) -> List[str]:
        terms = set(re.findall(r"[a-z][\w\-]+|[一-鿿]+", query.lower()))
        scored = []
        for node in self._adjacency:
            node_terms = set(re.findall(r"[a-z][\w\-]+|[一-鿿]+", node.lower()))
            hits = len(terms & node_terms)
            if hits:
                scored.append((hits, node))
        scored.sort(key=lambda pair: (-pair[0], pair[1]))
        return [node for _, node in scored[:limit]]

    def expand(
        self, start: str, hops: int = 2, decay: float = 0.6
    ) -> List[Tuple[Triple, float]]:
        """Breadth-first expansion with per-hop score decay."""
        key = _norm(start)
        if key not in self._adjacency:
            return []
        results: Dict[int, Tuple[Triple, float]] = {}
        visited = {key}
        frontier: List[Tuple[str, float]] = [(key, 1.0)]
        for _ in range(max(1, hops)):
            next_frontier: List[Tuple[str, float]] = []
            for node, weight in frontier:
                for triple in self._adjacency.get(node, []):
                    marker = id(triple)
                    if marker not in results or weight > results[marker][1]:
                        results[marker] = (triple, weight)
                    other = (
                        _norm(triple.object)
                        if _norm(triple.subject) == node
                        else _norm(triple.subject)
                    )
                    if other not in visited:
                        visited.add(other)
                        next_frontier.append((other, weight * decay))
            frontier = next_frontier
            if not frontier:
                break
        ranked = sorted(results.values(), key=lambda pair: pair[1], reverse=True)
        return ranked

    def query(self, query: str, hops: int = 2, k: int = 10) -> List[Tuple[Triple, float]]:
        combined: Dict[int, Tuple[Triple, float]] = {}
        for seed in self.seeds(query):
            for triple, weight in self.expand(seed, hops=hops):
                marker = id(triple)
                if marker not in combined or weight > combined[marker][1]:
                    combined[marker] = (triple, weight)
        ranked = sorted(combined.values(), key=lambda pair: pair[1], reverse=True)
        return ranked[:k]

    def render(self, results: Sequence[Tuple[Triple, float]], max_chars: int = 3000) -> str:
        lines: List[str] = []
        used = 0
        for triple, weight in results:
            line = f"- {triple.subject} --{triple.relation}--> {triple.object} ({weight:.2f})"
            if used + len(line) > max_chars:
                break
            lines.append(line)
            used += len(line) + 1
        return "\n".join(lines)

    def communities(self) -> List[List[str]]:
        """Connected components via union-find — cheap 'communities'."""
        groups: Dict[str, List[str]] = {}
        for node in self._adjacency:
            groups.setdefault(self._find(node), []).append(node)
        return [sorted(members) for members in groups.values() if len(members) > 1]

    def summarize_communities(self, model: Any, max_communities: int = 5) -> List[str]:
        """One model summary per community, for a global view of the graph."""
        summaries: List[str] = []
        for members in self.communities()[:max_communities]:
            member_set = {_norm(m) for m in members}
            facts = [
                f"{t.subject} --{t.relation}--> {t.object}"
                for t in self.triples
                if _norm(t.subject) in member_set
            ][:15]
            if not facts:
                continue
            try:
                reply = model.chat(
                    [
                        {"role": "system", "content": COMMUNITY_SYSTEM},
                        {"role": "user", "content": "\n".join(facts)},
                    ]
                )
                summaries.append(str(reply.get("content") or "").strip())
            except Exception:
                continue
        return [s for s in summaries if s]


def _norm(text: str) -> str:
    """Normalise an entity name for adjacency keys."""
    return unicodedata.normalize("NFKC", (text or "").strip().lower())


# ---------------------------------------------------------------------------
# Memory stream (Generative Agents)
# ---------------------------------------------------------------------------

@dataclass
class StreamEntry:
    """One observation in the stream."""

    id: str
    text: str
    ts: float
    importance: float
    kind: str = "observation"
    seq: int = 0
    vector: List[float] = field(default_factory=list)


class MemoryStream:
    """Append-only log with recency/importance/relevance retrieval.

    Retrieval score is the Generative Agents combination:

        score = w_r * 0.99^hours_since + w_i * importance + w_v * cosine(query)

    with weights summing to 1. Reflections are appended as their own entries
    once the accumulated importance of unreflected entries passes a threshold,
    which is how the stream keeps producing higher-level observations instead
    of only raw events.
    """

    def __init__(
        self,
        embedder: Optional[Embedder] = None,
        reflection_threshold: float = 8.0,
        recency_weight: float = 0.5,
        importance_weight: float = 0.3,
        relevance_weight: float = 0.2,
    ) -> None:
        self.embedder: Embedder = embedder or hashed_embed
        self.reflection_threshold = max(0.1, reflection_threshold)
        self.weights = (recency_weight, importance_weight, relevance_weight)
        self._entries: List[StreamEntry] = []
        self._unreflected_importance = 0.0
        self._seq = 0

    def add(
        self,
        text: str,
        importance: float = 0.5,
        kind: str = "observation",
        now: Optional[float] = None,
    ) -> StreamEntry:
        text = (text or "").strip()
        self._seq += 1
        entry = StreamEntry(
            id=stable_hash(f"{self._seq}|{text}"),
            text=text,
            ts=now if now is not None else time.time(),
            importance=max(0.0, min(1.0, importance)),
            kind=kind,
            seq=self._seq,
            vector=[float(v) for v in self.embedder(text)],
        )
        self._entries.append(entry)
        if kind == "observation":
            self._unreflected_importance += entry.importance
        return entry

    def __len__(self) -> int:
        return len(self._entries)

    def should_reflect(self) -> bool:
        return self._unreflected_importance >= self.reflection_threshold

    def mark_reflected(self) -> None:
        self._unreflected_importance = 0.0

    def retrieve(
        self, query: str, k: int = 5, now: Optional[float] = None
    ) -> List[Tuple[StreamEntry, float]]:
        if not self._entries:
            return []
        now = now if now is not None else time.time()
        qvec = self.embedder(query)
        recency_w, importance_w, relevance_w = self.weights
        scored = []
        for entry in self._entries:
            hours = max(0.0, (now - entry.ts) / 3600.0)
            recency = 0.99 ** hours
            relevance = cosine(qvec, entry.vector)
            score = (
                recency_w * recency
                + importance_w * entry.importance
                + relevance_w * relevance
            )
            scored.append((entry, score))
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[: max(1, k)]

    def recent(self, n: int = 10) -> List[StreamEntry]:
        return list(self._entries[-max(1, n) :])

    def reflect(self, model: Any, k: int = 12) -> List[StreamEntry]:
        """Ask for higher-level insights over the most important entries.

        Returns the reflection entries appended. This makes exactly one
        reflection pass; the caller decides whether to loop.
        """
        if not self._entries:
            return []
        important = sorted(self._entries, key=lambda e: e.importance, reverse=True)[:k]
        transcript = "\n".join(f"- [{e.kind}] {e.text}" for e in important)
        try:
            reply = model.chat(
                [
                    {
                        "role": "system",
                        "content": (
                            "You maintain long-term memory. From these observations, "
                            "write 1-3 higher-level insights that will help future "
                            "work: patterns, preferences, recurring failures. One per "
                            "line, no numbering. Reply NONE if nothing generalizes."
                        ),
                    },
                    {"role": "user", "content": transcript},
                ]
            )
            text = str(reply.get("content") or "").strip()
        except Exception:
            return []
        if not text or text.strip().upper() == "NONE":
            self.mark_reflected()
            return []
        added: List[StreamEntry] = []
        for line in text.splitlines():
            line = line.strip().lstrip("-*• ").strip()
            if line and line.upper() != "NONE":
                added.append(self.add(line, importance=0.8, kind="reflection"))
        self.mark_reflected()
        return added

    def render(self, entries: Sequence[StreamEntry], max_chars: int = 3000) -> str:
        lines: List[str] = []
        used = 0
        for entry in entries:
            line = f"- ({entry.kind}) {entry.text}"
            if used + len(line) > max_chars:
                break
            lines.append(line)
            used += len(line) + 1
        return "\n".join(lines)

    def to_json(self) -> List[Dict[str, Any]]:
        return [
            {
                "id": e.id,
                "ts": e.ts,
                "kind": e.kind,
                "text": e.text,
                "importance": e.importance,
                "vector": e.vector,
            }
            for e in self._entries
        ]

    def save(self, path: Path | str) -> None:
        path = Path(path)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(self.to_json(), ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except OSError:
            pass

    @classmethod
    def load(cls, path: Path | str, embedder: Optional[Embedder] = None) -> "MemoryStream":
        stream = cls(embedder)
        path = Path(path)
        if not path.is_file():
            return stream
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return stream
        if isinstance(data, list):
            for item in data:
                if not isinstance(item, dict):
                    continue
                entry = stream.add(
                    str(item.get("text") or ""),
                    importance=float(item.get("importance") or 0.5),
                    kind=str(item.get("kind") or "observation"),
                    now=float(item.get("ts") or time.time()),
                )
                if isinstance(item.get("vector"), list) and item["vector"]:
                    entry.vector = [float(v) for v in item["vector"]]
        return stream
