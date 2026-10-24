"""Search over thoughts: Tree-of-Thoughts, Graph-of-Thoughts, MCTS.

The strategies in ``agentic.reasoning`` are linear: one chain, maybe retried.
The ones here treat reasoning as a search problem, generating several partial
solutions, scoring them, and spending more compute on the branches that look
promising.

  tree_of_thoughts   breadth-first expansion + beam pruning. Each level
                     generates N candidate next thoughts per node, scores
                     them in one batched judge call, keeps the best B.
  graph_of_thoughts  nodes plus edges, where branches can be merged
                     (aggregate) or locally improved (refine) as well as
                     expanded — this is what makes it a graph, not a tree.
  mcts               Monte-Carlo Tree Search with UCT selection, expansion,
                     LLM-scored rollout and backpropagation.

All scoring uses the model itself as the evaluator (LLM-as-judge), which is
the standard trade-off: no reward model needed, but each node costs a call.
Budget knobs (breadth, depth, max_nodes, simulations) bound the spend, and
every function degrades gracefully when the model returns unparseable text by
falling back to positional scores instead of raising.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from agentic.base import (
    TraceEvent,
    ask,
    extract_final,
    extract_json,
    parse_score,
    stable_hash,
    truncate,
)


@dataclass
class ThoughtNode:
    """One partial solution in a search tree or graph."""

    id: str
    content: str
    score: float = 0.0
    parent: Optional[str] = None
    depth: int = 0
    visits: int = 0
    value: float = 0.0
    children: List[str] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchResult:
    """Best solution found, plus the structure that produced it."""

    answer: str
    best_path: List[ThoughtNode] = field(default_factory=list)
    all_nodes: List[ThoughtNode] = field(default_factory=list)
    trace: List[TraceEvent] = field(default_factory=list)
    score: float = 0.0
    calls: int = 0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "answer": self.answer,
            "score": round(self.score, 3),
            "nodes": len(self.all_nodes),
            "calls": self.calls,
            "path": [truncate(n.content, 200) for n in self.best_path],
            "trace": [
                {"step": e.step, "kind": e.kind, "detail": e.detail, "score": e.score}
                for e in self.trace
            ],
        }


GENERATE_SYSTEM = (
    "You are expanding a reasoning tree. Given the problem and the partial "
    "reasoning so far, propose {count} DIFFERENT next steps. Each must advance "
    "toward the solution in a materially different way. Output only a JSON "
    'array of strings: ["step one", "step two"]'
)

SCORE_SYSTEM = (
    "You score partial solutions. For each candidate, judge how likely it is to "
    "lead to a correct, complete solution of the problem. Output only a JSON "
    'object mapping the candidate label to a number 0-10: {"1": 7, "2": 3}'
)

AGGREGATE_SYSTEM = (
    "You merge several partial solutions to one problem into a single improved "
    "solution. Keep what is correct in each, drop contradictions, and resolve "
    "gaps. Output the merged step only."
)

REFINE_SYSTEM = (
    "You improve one partial solution. Remove errors, add missing steps, make "
    "it more concrete. Output the improved step only."
)

FINALIZE_SYSTEM = (
    "Turn the winning partial solution into the final deliverable for the task. "
    "Do not mention the tree, graph, or search. End with 'FINAL ANSWER:'."
)


def _score_batch(model: Any, task: str, candidates: Sequence[str]) -> Tuple[List[float], int]:
    """Score candidates 0-1 with one model call; positional default on failure."""
    if not candidates:
        return [], 0
    listing = "\n\n".join(f"CANDIDATE {i + 1}:\n{c}" for i, c in enumerate(candidates))
    reply = ask(model, SCORE_SYSTEM, f"PROBLEM:\n{task}\n\n{listing}")
    parsed = extract_json(reply)
    scores: List[float] = []
    if isinstance(parsed, dict):
        for index in range(len(candidates)):
            raw = parsed.get(str(index + 1), parsed.get(index + 1))
            try:
                scores.append(max(0.0, min(1.0, float(raw) / 10.0)))
            except (TypeError, ValueError):
                scores.append(parse_score(reply, default=0.5))
    elif isinstance(parsed, list):
        for index in range(len(candidates)):
            try:
                raw = float(parsed[index])
                scores.append(max(0.0, min(1.0, raw / 10.0 if raw > 1 else raw)))
            except (TypeError, ValueError, IndexError):
                scores.append(0.5)
    else:
        scores = [parse_score(reply, default=0.5)] * len(candidates)
    return scores, 1


def _expand(model: Any, task: str, node: ThoughtNode, count: int) -> Tuple[List[str], int]:
    """Ask for ``count`` distinct continuations of one node."""
    context = (
        f"PARTIAL REASONING:\n{node.content}"
        if node.parent
        else "This is the first step of the solution."
    )
    reply = ask(
        model,
        GENERATE_SYSTEM.format(count=count),
        f"PROBLEM:\n{task}\n\n{context}",
    )
    parsed = extract_json(reply)
    steps: List[str] = []
    if isinstance(parsed, list):
        steps = [str(item).strip() for item in parsed if str(item).strip()]
    elif isinstance(parsed, dict):
        steps = [str(v).strip() for v in parsed.values() if str(v).strip()]
    if not steps and reply:
        # Unstructured reply: split on non-empty lines as a last resort so the
        # search still has branches to explore.
        steps = [line.strip() for line in reply.splitlines() if len(line.strip()) > 20]
    return steps[:count], 1


def _path_of(nodes: Dict[str, ThoughtNode], node_id: str) -> List[ThoughtNode]:
    """Root-to-node lineage, guarding against cycles from merged nodes."""
    path: List[ThoughtNode] = []
    seen = set()
    while node_id and node_id in nodes and node_id not in seen:
        seen.add(node_id)
        node = nodes[node_id]
        path.append(node)
        node_id = node.parent or ""
    return list(reversed(path))


def _finalize(model: Any, task: str, raw: str) -> Tuple[str, int]:
    """Turn the best partial solution into prose; 1 model call, or 0 if empty."""
    if not raw.strip():
        return raw, 0
    reply = ask(model, FINALIZE_SYSTEM, f"TASK:\n{task}\n\nPARTIAL SOLUTION:\n{raw}")
    return extract_final(reply, fallback=raw), 1


# ---------------------------------------------------------------------------
# Tree of Thoughts
# ---------------------------------------------------------------------------

def tree_of_thoughts(
    model: Any,
    task: str,
    breadth: int = 3,
    depth: int = 3,
    beam_width: int = 2,
    synthesize: bool = True,
) -> SearchResult:
    """Beam search over generated thoughts.

    Level by level, every node in the beam spawns ``breadth`` continuations;
    all of them are scored in a single batched judge call, and only the best
    ``beam_width`` survive to the next level. Search stops early when the best
    branch scores >= 0.95. With ``synthesize=True`` the winning path is turned
    into readable prose, which usually reads better than the raw chain.
    """
    breadth, depth, beam_width = max(1, breadth), max(1, depth), max(1, beam_width)
    calls = 0
    trace: List[TraceEvent] = []
    nodes: Dict[str, ThoughtNode] = {
        "root": ThoughtNode("root", f"PROBLEM:\n{task}", depth=0)
    }
    beam = ["root"]

    for level in range(1, depth + 1):
        child_ids: List[str] = []
        for parent_id in beam:
            parent = nodes[parent_id]
            steps, used = _expand(model, task, parent, breadth)
            calls += used
            for step in steps:
                child_id = stable_hash(f"{parent_id}|{step}")
                if child_id in nodes:
                    continue
                nodes[child_id] = ThoughtNode(
                    child_id,
                    (parent.content + "\n" + step) if parent.parent else step,
                    parent=parent_id,
                    depth=level,
                )
                parent.children.append(child_id)
                child_ids.append(child_id)
        if not child_ids:
            trace.append(TraceEvent(level, "expand", "no candidates generated; stopping"))
            break

        scores, used = _score_batch(model, task, [nodes[c].content for c in child_ids])
        calls += used
        for child_id, score in zip(child_ids, scores):
            nodes[child_id].score = score
        ranked = sorted(child_ids, key=lambda c: nodes[c].score, reverse=True)
        beam = ranked[:beam_width]
        trace.append(
            TraceEvent(
                level,
                "level",
                f"{len(child_ids)} candidates, best={nodes[beam[0]].score:.2f}",
                nodes[beam[0]].score,
            )
        )
        if nodes[beam[0]].score >= 0.95:
            break

    best_id = max(beam, key=lambda c: nodes[c].score) if beam else "root"
    path = _path_of(nodes, best_id)
    raw = nodes[best_id].content if best_id != "root" else ""
    answer = raw
    if synthesize:
        answer, used = _finalize(model, task, raw)
        calls += used
        if used:
            trace.append(TraceEvent(len(trace) + 1, "synthesize", truncate(answer, 400)))

    return SearchResult(
        answer=answer,
        best_path=path,
        all_nodes=list(nodes.values()),
        trace=trace,
        score=nodes[best_id].score if best_id in nodes else 0.0,
        calls=calls,
    )


# ---------------------------------------------------------------------------
# Graph of Thoughts
# ---------------------------------------------------------------------------

def graph_of_thoughts(
    model: Any,
    task: str,
    expand_rounds: int = 2,
    breadth: int = 2,
    aggregate_every: int = 2,
    refine_top: int = 1,
    max_nodes: int = 12,
    synthesize: bool = True,
) -> SearchResult:
    """Nodes, edges, and two operations a tree cannot express.

    Beyond expansion, GoT supports:
      * aggregate — merge two strong branches into one node with edges from
        both parents, which recovers when each branch got half the answer;
      * refine — take a current best node and ask for an improved version,
        spending compute where it is most likely to pay off.

    Each node's ``meta["op"]``/``meta["from"]`` records provenance, so the
    final answer's lineage is inspectable.
    """
    trace: List[TraceEvent] = []
    calls = 0
    nodes: Dict[str, ThoughtNode] = {
        "root": ThoughtNode("root", f"PROBLEM:\n{task}", depth=0)
    }
    frontier = ["root"]
    breadth = max(1, breadth)

    for round_index in range(1, max(1, expand_rounds) + 1):
        new_ids: List[str] = []
        for parent_id in frontier:
            if len(nodes) >= max_nodes:
                break
            parent = nodes[parent_id]
            steps, used = _expand(model, task, parent, breadth)
            calls += used
            for step in steps:
                child_id = stable_hash(f"{parent_id}|{step}")
                if child_id in nodes:
                    continue
                nodes[child_id] = ThoughtNode(
                    child_id,
                    (parent.content + "\n" + step) if parent.parent else step,
                    parent=parent_id,
                    depth=parent.depth + 1,
                    meta={"op": "generate", "from": [parent_id]},
                )
                parent.children.append(child_id)
                new_ids.append(child_id)
        if not new_ids:
            break
        scores, used = _score_batch(model, task, [nodes[n].content for n in new_ids])
        calls += used
        for node_id, score in zip(new_ids, scores):
            nodes[node_id].score = score
        new_ids.sort(key=lambda n: nodes[n].score, reverse=True)
        trace.append(
            TraceEvent(
                round_index,
                "expand",
                f"{len(new_ids)} new, best={nodes[new_ids[0]].score:.2f}",
                nodes[new_ids[0]].score,
            )
        )

        if len(new_ids) >= aggregate_every and len(nodes) < max_nodes:
            pair = new_ids[:aggregate_every]
            merged = ask(
                model,
                AGGREGATE_SYSTEM,
                f"PROBLEM:\n{task}\n\n"
                + "\n\n".join(
                    f"SOLUTION {i + 1}:\n{nodes[n].content}" for i, n in enumerate(pair)
                ),
            )
            calls += 1
            merged_id = stable_hash(f"agg|{'|'.join(pair)}")
            if merged.strip() and merged_id not in nodes:
                nodes[merged_id] = ThoughtNode(
                    merged_id,
                    merged.strip(),
                    parent=pair[0],
                    depth=max(nodes[p].depth for p in pair),
                    meta={"op": "aggregate", "from": list(pair)},
                )
                merged_scores, used = _score_batch(model, task, [merged.strip()])
                calls += used
                nodes[merged_id].score = merged_scores[0] if merged_scores else 0.0
                new_ids.insert(0, merged_id)
                trace.append(
                    TraceEvent(round_index, "aggregate", f"merged {len(pair)} branches",
                               nodes[merged_id].score)
                )

        for node_id in new_ids[: max(0, refine_top)]:
            if len(nodes) >= max_nodes:
                break
            improved = ask(
                model,
                REFINE_SYSTEM,
                f"PROBLEM:\n{task}\n\nPARTIAL SOLUTION:\n{nodes[node_id].content}",
            )
            calls += 1
            refined_id = stable_hash(f"ref|{node_id}|{improved}")
            if not improved.strip() or refined_id in nodes:
                continue
            nodes[refined_id] = ThoughtNode(
                refined_id,
                improved.strip(),
                parent=node_id,
                depth=nodes[node_id].depth + 1,
                meta={"op": "refine", "from": [node_id]},
            )
            refined_scores, used = _score_batch(model, task, [improved.strip()])
            calls += used
            nodes[refined_id].score = (
                refined_scores[0] if refined_scores else nodes[node_id].score
            )
            new_ids.append(refined_id)
            trace.append(
                TraceEvent(round_index, "refine", f"refined {node_id[:8]}", nodes[refined_id].score)
            )

        frontier = new_ids[: max(1, breadth * 2)]

    best_id = max(
        (n for n in nodes if n != "root"), key=lambda n: nodes[n].score, default="root"
    )
    path = _path_of(nodes, best_id)
    raw = nodes[best_id].content if best_id != "root" else ""
    answer = raw
    if synthesize:
        answer, used = _finalize(model, task, raw)
        calls += used
        if used:
            trace.append(TraceEvent(len(trace) + 1, "synthesize", truncate(answer, 400)))
    return SearchResult(
        answer=answer,
        best_path=path,
        all_nodes=list(nodes.values()),
        trace=trace,
        score=nodes[best_id].score if best_id in nodes else 0.0,
        calls=calls,
    )


# ---------------------------------------------------------------------------
# Monte-Carlo Tree Search
# ---------------------------------------------------------------------------

def mcts(
    model: Any,
    task: str,
    simulations: int = 8,
    expand_width: int = 2,
    exploration: float = 1.4,
    max_depth: int = 3,
    prune_below: float = 0.3,
) -> SearchResult:
    """UCT tree search where the model both expands nodes and scores rollouts.

    Selection walks the tree by UCT: exploit the child with the highest mean
    value, explore children with few visits. The chosen node is expanded into
    ``expand_width`` continuations, one of which is evaluated (the "rollout" —
    a direct judge call rather than a random playout, since LLM rollouts are
    too expensive to randomise), and the score is backed up to the root.

    ``exploration`` is the UCB constant: higher explores more branches, lower
    commits to the current best. ``prune_below`` stops descending into a child
    whose mean value is that low while siblings exist, which keeps a bad first
    branch from eating the whole budget.
    """
    simulations = max(1, simulations)
    exploration_constant = max(0.0, exploration)
    trace: List[TraceEvent] = []
    calls = 0
    nodes: Dict[str, ThoughtNode] = {
        "root": ThoughtNode("root", f"PROBLEM:\n{task}", depth=0)
    }

    def uct(child: ThoughtNode, parent_visits: int) -> float:
        if child.visits == 0:
            return float("inf")  # unvisited children get tried first
        exploitation = child.value / child.visits
        exploration_bonus = exploration_constant * math.sqrt(
            math.log(max(1, parent_visits)) / child.visits
        )
        return exploitation + exploration_bonus

    for sim in range(1, simulations + 1):
        # --- selection ---
        node_id = "root"
        path = [node_id]
        while nodes[node_id].children and nodes[node_id].depth < max_depth:
            children = [nodes[c] for c in nodes[node_id].children]
            best = max(children, key=lambda c: uct(c, nodes[node_id].visits))
            if (
                len(children) > 1
                and best.visits > 0
                and best.value / best.visits < prune_below
            ):
                break
            node_id = best.id
            path.append(node_id)
            if nodes[node_id].visits == 0:
                break

        current = nodes[node_id]

        # --- expansion ---
        if current.depth < max_depth and len(nodes) < simulations * expand_width + 1:
            steps, used = _expand(model, task, current, expand_width)
            calls += used
            for step in steps:
                child_id = stable_hash(f"mcts|{node_id}|{step}")
                if child_id in nodes:
                    continue
                nodes[child_id] = ThoughtNode(
                    child_id,
                    (current.content + "\n" + step) if current.parent else step,
                    parent=node_id,
                    depth=current.depth + 1,
                )
                current.children.append(child_id)
            if current.children:
                fresh = [c for c in current.children if nodes[c].visits == 0]
                node_id = fresh[0] if fresh else current.children[0]
                path.append(node_id)
                current = nodes[node_id]

        # --- rollout / evaluation ---
        if current.id == "root":
            score = 0.0
        else:
            scored, used = _score_batch(model, task, [current.content])
            calls += used
            score = scored[0] if scored else 0.0

        # --- backpropagation ---
        for visited in path:
            nodes[visited].visits += 1
            nodes[visited].value += score
        trace.append(
            TraceEvent(sim, "simulation", f"depth={current.depth} nodes={len(nodes)}", score)
        )

    scored_nodes = [n for n in nodes.values() if n.id != "root" and n.visits > 0]
    best = max(scored_nodes, key=lambda n: n.value / n.visits, default=None)
    if best is None:
        return SearchResult("", [], list(nodes.values()), trace, 0.0, calls)

    best_path = _path_of(nodes, best.id)
    answer, used = _finalize(model, task, best.content)
    calls += used
    return SearchResult(
        answer=answer,
        best_path=best_path,
        all_nodes=list(nodes.values()),
        trace=trace,
        score=best.value / best.visits,
        calls=calls,
    )


# ---------------------------------------------------------------------------
# Dispatch helper used by the suite
# ---------------------------------------------------------------------------

SEARCH_STRATEGIES: Dict[str, Callable[..., SearchResult]] = {
    "tot": tree_of_thoughts,
    "tree-of-thoughts": tree_of_thoughts,
    "got": graph_of_thoughts,
    "graph-of-thoughts": graph_of_thoughts,
    "mcts": mcts,
}

_SEARCH_KWARGS = {
    "breadth", "depth", "beam_width", "synthesize", "expand_rounds",
    "aggregate_every", "refine_top", "max_nodes", "simulations",
    "expand_width", "exploration", "max_depth", "prune_below",
}


def run_search(name: str, model: Any, task: str, **kwargs: Any) -> SearchResult:
    """Call one of the search strategies by name (used by the think tool)."""
    key = (name or "").strip().lower()
    if key not in SEARCH_STRATEGIES:
        raise ValueError(
            f"Unknown search strategy '{name}'. "
            f"Choose one of: {', '.join(sorted(SEARCH_STRATEGIES))}"
        )
    filtered = {k: v for k, v in kwargs.items() if k in _SEARCH_KWARGS}
    return SEARCH_STRATEGIES[key](model, task, **filtered)
