"""Alignment & self-improvement layer: SFT, RLHF/RLAIF, DPO/PPO, bootstrapping.

This module turns agent trajectories into training data and runs the model-side
loops that improve an agent over time. Everything is *model-agnostic*: a
"model" is anything accepted by :func:`agentic.base.ask` (an object with
``chat(messages)`` or a bare callable). No trainer framework is pulled in - the
job here is to produce the datasets (JSONL in the standard SFT / preference
shapes) and the reward/advantage signals those trainers consume, plus the
self-play loop that generates and filters fresh examples.

Pieces
------
- Instruction fine-tuning (SFT): collect (instruction, context, output)
  trajectories and export instruction-tuning JSONL.
- RLHF / RLAIF: an AI-feedback :class:`RewardModel` scores a response against a
  rubric; pairwise preference labelling builds (chosen, rejected) pairs without
  a human in the loop (RLAIF) or around human labels (RLHF).
- DPO: export preference pairs in the ``{prompt, chosen, rejected}`` shape DPO
  trainers expect.
- PPO: :func:`compute_advantages` turns per-step rewards into GAE advantages
  and returns-to-go, the signal a PPO update consumes.
- Bootstrapping / self-improvement (STaR-style): propose task, solve, verify,
  keep only verified trajectories, feed them back as SFT data.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

from agentic.base import ask, extract_final, parse_score, stable_hash, truncate

Message = Dict[str, Any]

# A verifier returns True when an answer is acceptable (tests pass, output
# matches, judge approves...). A solver maps a task string to an answer string.
Verifier = Callable[[str, str], bool]
Solver = Callable[[str], str]


# ---------------------------------------------------------------------------
# Trajectory record - the common currency of every exporter below
# ---------------------------------------------------------------------------


@dataclass
class Trajectory:
    """One instruction attempt with its reasoning/tool steps and outcome."""

    instruction: str
    answer: str
    context: str = ""
    steps: List[Dict[str, Any]] = field(default_factory=list)
    reward: float = 0.0
    verified: Optional[bool] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def id(self) -> str:
        return stable_hash(self.instruction + "\x00" + self.answer)

    def to_sft(self) -> Dict[str, Any]:
        """Instruction-tuning row (Alpaca-style instruction/input/output)."""
        return {
            "instruction": self.instruction,
            "input": self.context,
            "output": self.answer,
            "steps": self.steps,
            "reward": round(self.reward, 4),
        }

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _write_jsonl(rows: Iterable[Dict[str, Any]], path: Path | str) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


# ---------------------------------------------------------------------------
# Instruction fine-tuning (SFT)
# ---------------------------------------------------------------------------


def trajectories_from_history(
    history: Sequence[Message],
    *,
    context: str = "",
) -> List[Trajectory]:
    """Pull (user -> assistant) instruction/answer pairs from a chat history.

    Tool-call turns (assistant messages with no plain text, plus their tool
    results) are folded into ``steps`` of the trajectory they precede, so a
    multi-tool answer stays one training example rather than fragmenting.
    """
    trajectories: List[Trajectory] = []
    pending_instruction: Optional[str] = None
    pending_steps: List[Dict[str, Any]] = []

    for msg in history:
        role = msg.get("role")
        content = msg.get("content")
        text = content.strip() if isinstance(content, str) else ""
        if role == "user" and text:
            pending_instruction = text
            pending_steps = []
        elif role == "tool":
            pending_steps.append({"role": "tool", "content": truncate(text, 2000)})
        elif role == "assistant":
            if msg.get("tool_calls"):
                pending_steps.append({"role": "assistant", "tool_calls": msg["tool_calls"]})
                continue
            if pending_instruction and text:
                trajectories.append(
                    Trajectory(
                        instruction=pending_instruction,
                        answer=text,
                        context=context,
                        steps=list(pending_steps),
                    )
                )
                pending_instruction = None
                pending_steps = []
    return trajectories


def export_sft(trajectories: Sequence[Trajectory], path: Path | str) -> int:
    """Write instruction-tuning JSONL. Returns the number of rows written."""
    return _write_jsonl((t.to_sft() for t in trajectories), path)


def export_sft_from_sessions(
    sessions_dir: Path | str,
    out_path: Path | str,
    *,
    min_reward: float = 0.0,
) -> Dict[str, Any]:
    """Harvest every saved session's history into one SFT file.

    Sessions are JSON files holding a ``history`` list (the project's own
    ``.agent_sessions`` format). Only trajectories with reward >= ``min_reward``
    are kept, so a reward-annotated corpus can be filtered to its best slice.
    """
    sessions_dir = Path(sessions_dir)
    rows: List[Dict[str, Any]] = []
    files = 0
    for file in sorted(sessions_dir.glob("*.json")):
        try:
            data = json.loads(file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        history = data.get("history") if isinstance(data, dict) else None
        if not isinstance(history, list):
            continue
        files += 1
        for traj in trajectories_from_history(history):
            if traj.reward >= min_reward:
                rows.append(traj.to_sft())
    written = _write_jsonl(rows, out_path)
    return {"sessions": files, "examples": written, "path": str(out_path)}


# ---------------------------------------------------------------------------
# RLHF / RLAIF - AI-feedback reward model
# ---------------------------------------------------------------------------

REWARD_SYSTEM = (
    "You are a strict evaluator scoring an AI assistant's response. Judge it on "
    "correctness, completeness, safety, and how well it follows the request. "
    "Respond with a single line 'SCORE: <0..1>' where 1 is excellent and 0 is "
    "useless or wrong, optionally followed by one sentence of justification."
)

PREFERENCE_SYSTEM = (
    "You compare two candidate responses (A and B) to the same request. Decide "
    "which is better on correctness, helpfulness, and safety. Answer with a "
    "single line 'WINNER: A' or 'WINNER: B', then one sentence of reasoning. "
    "Never answer 'tie'."
)


@dataclass
class RewardModel:
    """LLM-as-judge scalar reward used for RLAIF and PPO shaping.

    ``rubric`` is appended to the base system prompt so a caller can bias the
    judge toward a project's own bar (e.g. "penalise answers that skip tests").
    """

    model: Any
    rubric: str = ""

    def score(self, instruction: str, answer: str, *, context: str = "") -> float:
        system = REWARD_SYSTEM + (f"\nExtra rubric: {self.rubric}" if self.rubric else "")
        user = (
            f"REQUEST:\n{truncate(instruction, 4000)}\n\n"
            + (f"CONTEXT:\n{truncate(context, 2000)}\n\n" if context else "")
            + f"RESPONSE:\n{truncate(answer, 6000)}"
        )
        return parse_score(ask(self.model, system, user), default=0.5)

    def prefer(self, instruction: str, answer_a: str, answer_b: str) -> int:
        """Return +1 if A is better, -1 if B is better (pairwise AI feedback)."""
        user = (
            f"REQUEST:\n{truncate(instruction, 4000)}\n\n"
            f"RESPONSE A:\n{truncate(answer_a, 4000)}\n\n"
            f"RESPONSE B:\n{truncate(answer_b, 4000)}"
        )
        reply = ask(self.model, PREFERENCE_SYSTEM, user).upper()
        marker = reply.split("WINNER:", 1)[-1].strip() if "WINNER:" in reply else reply
        return -1 if marker[:1] == "B" else 1

    def annotate(self, trajectories: Sequence[Trajectory]) -> List[Trajectory]:
        """Fill each trajectory's ``reward`` in place (RLAIF labelling)."""
        for traj in trajectories:
            traj.reward = self.score(traj.instruction, traj.answer, context=traj.context)
        return list(trajectories)


# ---------------------------------------------------------------------------
# DPO - preference pair construction & export
# ---------------------------------------------------------------------------


@dataclass
class PreferencePair:
    prompt: str
    chosen: str
    rejected: str
    margin: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt": self.prompt,
            "chosen": self.chosen,
            "rejected": self.rejected,
            "margin": round(self.margin, 4),
        }


def build_preference_pairs(
    instruction: str,
    candidates: Sequence[str],
    reward: RewardModel,
    *,
    max_pairs: int = 3,
) -> List[PreferencePair]:
    """Score N candidates once, then pair best-vs-worst by reward margin.

    One reward call per candidate (not per pair) keeps labelling cheap; pairs
    are formed by sorting on the scalar reward and matching the top against the
    bottom. ``margin`` (reward gap) lets a trainer drop near-ties.
    """
    scored = sorted(
        ((reward.score(instruction, candidate), candidate) for candidate in candidates if candidate),
        key=lambda item: item[0],
        reverse=True,
    )
    pairs: List[PreferencePair] = []
    lo, hi = 0, len(scored) - 1
    while lo < hi and len(pairs) < max_pairs:
        high_score, chosen = scored[lo]
        low_score, rejected = scored[hi]
        if chosen != rejected and high_score > low_score:
            pairs.append(PreferencePair(instruction, chosen, rejected, high_score - low_score))
        lo += 1
        hi -= 1
    return pairs


def export_dpo(pairs: Sequence[PreferencePair], path: Path | str) -> int:
    """Write preference JSONL in the {prompt, chosen, rejected} DPO shape."""
    return _write_jsonl((pair.to_dict() for pair in pairs), path)


# ---------------------------------------------------------------------------
# PPO - advantage estimation (GAE)
# ---------------------------------------------------------------------------


def compute_advantages(
    rewards: Sequence[float],
    values: Optional[Sequence[float]] = None,
    *,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> Dict[str, List[float]]:
    """Generalised Advantage Estimation over one trajectory's step rewards.

    Returns ``advantages`` (the PPO policy-gradient weight per step) and
    ``returns`` (advantage + baseline value, the critic target). With no
    ``values`` the baseline is zero, so advantages reduce to discounted
    returns-to-go - still a usable, if higher-variance, PPO signal.
    """
    n = len(rewards)
    if n == 0:
        return {"advantages": [], "returns": []}
    baseline = list(values) if values is not None else [0.0] * n
    if len(baseline) != n:
        raise ValueError("values must match rewards length")

    advantages = [0.0] * n
    gae = 0.0
    for t in range(n - 1, -1, -1):
        next_value = baseline[t + 1] if t + 1 < n else 0.0
        delta = rewards[t] + gamma * next_value - baseline[t]
        gae = delta + gamma * lam * gae
        advantages[t] = gae
    returns = [advantages[t] + baseline[t] for t in range(n)]
    return {"advantages": advantages, "returns": returns}


# ---------------------------------------------------------------------------
# Bootstrapping / self-improvement (STaR-style self-play)
# ---------------------------------------------------------------------------

TASK_SYSTEM = (
    "You design challenging but solvable practice tasks for an autonomous "
    "software agent. Each task must be concrete, self-contained, and verifiable "
    "from its output alone. Reply with the task statement only - no preamble."
)


@dataclass
class BootstrapReport:
    attempted: int = 0
    kept: int = 0
    trajectories: List[Trajectory] = field(default_factory=list)

    @property
    def yield_rate(self) -> float:
        return self.kept / self.attempted if self.attempted else 0.0


class SelfImprover:
    """Propose -> solve -> verify -> keep loop that mines its own SFT data.

    ``solver(task) -> answer`` runs the agent; ``verifier(task, answer) -> bool``
    is the gate (unit tests, a checker, or a reward threshold). Only verified
    trajectories are retained, which is the property that lets the resulting
    SFT set raise the policy instead of reinforcing its mistakes.
    """

    def __init__(
        self,
        model: Any,
        solver: Solver,
        verifier: Verifier,
        *,
        reward: Optional[RewardModel] = None,
    ) -> None:
        self.model = model
        self.solver = solver
        self.verifier = verifier
        self.reward = reward

    def propose(self, topic: str) -> str:
        return ask(self.model, TASK_SYSTEM, f"Topic: {topic}\nWrite one task.").strip()

    def attempt(self, task: str) -> Trajectory:
        raw = self.solver(task)
        answer = extract_final(raw, fallback=raw)
        ok = bool(self.verifier(task, answer))
        reward = (
            self.reward.score(task, answer) if self.reward is not None else (1.0 if ok else 0.0)
        )
        return Trajectory(
            instruction=task,
            answer=answer,
            reward=reward,
            verified=ok,
            meta={"ts": time.time()},
        )

    def run(self, topics: Sequence[str], *, per_topic: int = 3) -> BootstrapReport:
        report = BootstrapReport()
        for topic in topics:
            for _ in range(max(1, per_topic)):
                task = self.propose(topic)
                if not task:
                    continue
                traj = self.attempt(task)
                report.attempted += 1
                if traj.verified:
                    report.kept += 1
                    report.trajectories.append(traj)
        return report


def bootstrap_to_sft(
    model: Any,
    solver: Solver,
    verifier: Verifier,
    topics: Sequence[str],
    out_path: Path | str,
    *,
    per_topic: int = 3,
    reward: Optional[RewardModel] = None,
) -> Dict[str, Any]:
    """End-to-end: self-play a batch, export the verified wins as SFT data."""
    improver = SelfImprover(model, solver, verifier, reward=reward)
    report = improver.run(topics, per_topic=per_topic)
    written = export_sft(report.trajectories, out_path)
    return {
        "attempted": report.attempted,
        "kept": report.kept,
        "yield_rate": round(report.yield_rate, 3),
        "written": written,
        "path": str(out_path),
    }
