"""SOP workflows and hierarchical manager-worker execution.

  SOPRunner   a declarative standard operating procedure: ordered steps, each
              with a prompt template, an optional validator, retries, and a
              skip condition. The run is checkpointed to JSON after every step
              so a long procedure can resume after a crash rather than restart.

  HierarchicalOrchestrator
              a manager decomposes the goal into subtasks with dependencies,
              worker callables execute the ready ones in parallel, and the
              manager reviews results and may issue a second round. This is
              the manager-worker pattern: cheap breadth from workers, coherence
              from a single planner who sees everything.

Both take plain callables for the actual work, so they compose with an Agent
(``lambda prompt: agent.ask(prompt)``), with a model, or with fakes in tests.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from agentic.base import TraceEvent, ask, extract_json, truncate

WorkerFn = Callable[[str, Dict[str, str]], str]


# ---------------------------------------------------------------------------
# SOP workflow
# ---------------------------------------------------------------------------

@dataclass
class SOPStep:
    """One procedure step.

    ``template`` may reference earlier outputs with ``{step_name}`` placeholders
    and the original goal with ``{goal}``. ``validate(output) -> (ok, reason)``
    decides whether the step succeeded; on failure the step retries up to
    ``retries`` times, appending the validator's reason to the prompt.
    ``skip_if(state) -> bool`` short-circuits a step entirely.
    """

    name: str
    template: str
    retries: int = 1
    validate: Optional[Callable[[str], Tuple[bool, str]]] = None
    skip_if: Optional[Callable[[Dict[str, str]], bool]] = None
    required: bool = True


@dataclass
class SOPResult:
    ok: bool
    outputs: Dict[str, str] = field(default_factory=dict)
    trace: List[TraceEvent] = field(default_factory=list)
    failed_step: str = ""

    def render(self) -> str:
        lines = [f"{'ok' if self.ok else 'failed'} — {len(self.outputs)} step(s)"]
        for name, output in self.outputs.items():
            lines.append(f"\n## {name}\n{truncate(output, 1200)}")
        if not self.ok:
            lines.append(f"\nFailed at step: {self.failed_step}")
        return "\n".join(lines)


class SOPRunner:
    """Run ordered steps with retries, validation, and a checkpoint file.

    ``run_fn(prompt) -> str`` performs one step. The checkpoint stores the
    goal and each completed output, so ``resume=True`` on an identical goal
    skips the steps already done. Nothing here retries a *successful* step:
    re-running a procedure that writes files must stay deliberate.
    """

    def __init__(
        self,
        run_fn: Callable[[str], str],
        checkpoint: Optional[Path | str] = None,
    ) -> None:
        self.run_fn = run_fn
        self.checkpoint = Path(checkpoint) if checkpoint else None

    def _load_checkpoint(self, goal: str) -> Dict[str, str]:
        if not self.checkpoint or not self.checkpoint.is_file():
            return {}
        try:
            data = json.loads(self.checkpoint.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        if not isinstance(data, dict) or data.get("goal") != goal:
            return {}
        outputs = data.get("outputs")
        return {str(k): str(v) for k, v in outputs.items()} if isinstance(outputs, dict) else {}

    def _save_checkpoint(self, goal: str, outputs: Dict[str, str]) -> None:
        if not self.checkpoint:
            return
        try:
            self.checkpoint.parent.mkdir(parents=True, exist_ok=True)
            self.checkpoint.write_text(
                json.dumps(
                    {"goal": goal, "outputs": outputs, "updated": time.time()},
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
        except OSError:
            pass

    def run(
        self, goal: str, steps: Sequence[SOPStep], resume: bool = False
    ) -> SOPResult:
        outputs = self._load_checkpoint(goal) if resume else {}
        trace: List[TraceEvent] = []
        context: Dict[str, str] = {"goal": goal, **outputs}

        for index, step in enumerate(steps, start=1):
            if step.name in outputs:
                trace.append(TraceEvent(index, step.name, "skipped (checkpoint)"))
                continue
            if step.skip_if and step.skip_if(context):
                trace.append(TraceEvent(index, step.name, "skipped (condition)"))
                continue
            try:
                prompt = step.template.format(**{**context, "goal": goal})
            except KeyError as exc:
                # A missing placeholder is a workflow bug, not a model failure.
                outputs[step.name] = f"Template error: missing {exc}"
                if step.required:
                    return SOPResult(False, outputs, trace, step.name)
                continue

            problem = ""
            succeeded = False
            output = ""
            for attempt in range(1, max(1, step.retries) + 1):
                attempt_prompt = prompt
                if problem:
                    attempt_prompt = (
                        f"{prompt}\n\nPrevious attempt was rejected: {problem}\n"
                        "Fix that and try again."
                    )
                try:
                    output = self.run_fn(attempt_prompt) or ""
                except Exception as exc:
                    problem = f"step raised: {exc}"
                    output = ""
                if output and step.validate:
                    ok, reason = step.validate(output)
                    if not ok:
                        problem = reason or "validation failed"
                        output = ""
                if output:
                    succeeded = True
                trace.append(
                    TraceEvent(
                        index,
                        f"{step.name} attempt {attempt}",
                        truncate(problem or output, 400),
                        1.0 if succeeded else 0.0,
                    )
                )
                if succeeded:
                    break
            if not succeeded:
                if step.required:
                    return SOPResult(False, outputs, trace, step.name)
                outputs[step.name] = ""
                continue
            outputs[step.name] = output
            context[step.name] = output
            self._save_checkpoint(goal, outputs)

        return SOPResult(True, outputs, trace)


# ---------------------------------------------------------------------------
# Hierarchical manager-worker
# ---------------------------------------------------------------------------

MANAGER_PLAN_SYSTEM = (
    "You are a manager. Decompose the goal into at most {limit} subtasks for "
    "workers. Each subtask must be independently executable and concrete. "
    'Output ONLY JSON: [{{"id": "1", "task": "...", "depends_on": []}}]'
)

MANAGER_REVIEW_SYSTEM = (
    "You review finished worker output against the goal. Reply with exactly "
    'JSON: {"complete": true|false, "gaps": ["..."], "next": ["..."]}. '
    "If anything essential is missing, set complete false and put the missing "
    "work in next."
)

MANAGER_SYNTH_SYSTEM = (
    "Synthesize the worker outputs into one coherent answer for the goal. Do not "
    "mention workers or subtasks. End with 'FINAL ANSWER:'."
)


@dataclass
class Subtask:
    id: str
    task: str
    depends_on: List[str] = field(default_factory=list)
    result: str = ""
    ok: bool = False


@dataclass
class TeamResult:
    answer: str
    subtasks: List[Subtask] = field(default_factory=list)
    trace: List[TraceEvent] = field(default_factory=list)
    rounds: int = 0

    def render(self) -> str:
        lines = [f"team run: {len(self.subtasks)} subtask(s), {self.rounds} round(s)"]
        for sub in self.subtasks:
            mark = "x" if sub.ok else " "
            lines.append(f"- [{mark}] {sub.id} {sub.task} -> {truncate(sub.result, 160)}")
        lines.append(f"\n## Answer\n{self.answer}")
        return "\n".join(lines)


class HierarchicalOrchestrator:
    """Manager plans, workers execute in dependency order, manager reviews.

    ``worker_fn(task, upstream_results) -> str`` is whatever actually does the
    work; ``model`` is the manager. Ready subtasks run in parallel up to
    ``max_workers``. A subtask whose dependency failed is marked failed rather
    than run against missing input. After the first round the manager either
    accepts, or emits follow-up subtasks for one more round (``max_rounds``
    bounds this so a manager that never accepts cannot loop forever).
    """

    def __init__(
        self,
        model: Any,
        worker_fn: WorkerFn,
        max_workers: int = 4,
        max_rounds: int = 2,
        max_subtasks: int = 8,
    ) -> None:
        self.model = model
        self.worker_fn = worker_fn
        self.max_workers = max(1, max_workers)
        self.max_rounds = max(1, max_rounds)
        self.max_subtasks = max(1, max_subtasks)

    # -- planning ---------------------------------------------------------

    def plan(self, goal: str, context: str = "") -> List[Subtask]:
        reply = ask(
            self.model,
            MANAGER_PLAN_SYSTEM.format(limit=self.max_subtasks),
            f"GOAL:\n{goal}\n\nCONTEXT:\n{context}".strip(),
        )
        parsed = extract_json(reply)
        subtasks: List[Subtask] = []
        if isinstance(parsed, list):
            for index, item in enumerate(parsed[: self.max_subtasks], start=1):
                if not isinstance(item, dict):
                    continue
                task_text = str(item.get("task") or item.get("description") or "").strip()
                if not task_text:
                    continue
                deps = item.get("depends_on") or item.get("dependencies") or []
                subtasks.append(
                    Subtask(
                        id=str(item.get("id") or index),
                        task=task_text,
                        depends_on=[str(d) for d in deps] if isinstance(deps, list) else [],
                    )
                )
        if not subtasks:
            # Fallback: a single worker does the whole goal. Better than
            # failing because the manager did not emit JSON.
            subtasks = [Subtask(id="1", task=goal)]
        return subtasks

    # -- execution --------------------------------------------------------

    def _run_wave(self, subtasks: List[Subtask], done: Dict[str, Subtask]) -> None:
        """Execute ready subtasks in parallel; repeat until no progress."""
        pending = list(subtasks)
        while pending:
            ready = [
                sub for sub in pending if all(dep in done for dep in sub.depends_on)
            ]
            blocked = [
                sub
                for sub in pending
                if any(dep in done and not done[dep].ok for dep in sub.depends_on)
            ]
            for sub in blocked:
                sub.ok = False
                sub.result = "Skipped: a dependency failed."
                done[sub.id] = sub
                pending.remove(sub)
            if not ready:
                # Remaining subtasks reference ids that never appear: run them
                # anyway rather than deadlock, since ids are model-authored.
                ready = pending[: self.max_workers]
            ready = ready[: self.max_workers]
            upstream = {sub.id: sub.result for sub in done.values() if sub.ok}

            def work(sub: Subtask) -> Tuple[Subtask, str, bool]:
                try:
                    output = self.worker_fn(sub.task, upstream)
                except Exception as exc:
                    return sub, f"Worker error: {exc}", False
                return sub, output or "", bool(output)

            with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
                for sub, output, ok in pool.map(work, ready):
                    sub.result = output
                    sub.ok = ok
                    done[sub.id] = sub
                    pending.remove(sub)

    def _review(self, goal: str, subtasks: List[Subtask]) -> Tuple[bool, List[str]]:
        listing = "\n\n".join(
            f"SUBTASK {sub.id}: {sub.task}\nRESULT: {truncate(sub.result, 1200)}"
            for sub in subtasks
        )
        reply = ask(self.model, MANAGER_REVIEW_SYSTEM, f"GOAL:\n{goal}\n\n{listing}")
        parsed = extract_json(reply)
        if isinstance(parsed, dict):
            complete = bool(parsed.get("complete"))
            gaps = parsed.get("next") or parsed.get("gaps") or []
            return complete, [str(g) for g in gaps if str(g).strip()]
        # Unparseable review: treat as complete so the run always terminates.
        return True, []

    def run(self, goal: str, context: str = "") -> TeamResult:
        trace: List[TraceEvent] = []
        subtasks = self.plan(goal, context)
        trace.append(TraceEvent(1, "plan", f"{len(subtasks)} subtask(s)"))
        done: Dict[str, Subtask] = {}
        all_subtasks: List[Subtask] = list(subtasks)
        round_index = 0

        while round_index < self.max_rounds:
            round_index += 1
            self._run_wave(subtasks, done)
            for sub in subtasks:
                trace.append(
                    TraceEvent(
                        round_index + 1,
                        "worker",
                        f"{sub.id}: {truncate(sub.task, 120)} -> {truncate(sub.result, 200)}",
                        1.0 if sub.ok else 0.0,
                    )
                )
            complete, follow_ups = self._review(goal, all_subtasks)
            trace.append(TraceEvent(round_index + 2, "review", f"complete={complete}"))
            if complete or not follow_ups:
                break
            base = len(all_subtasks)
            subtasks = [
                Subtask(id=str(base + index + 1), task=text)
                for index, text in enumerate(follow_ups[: self.max_subtasks])
            ]
            all_subtasks.extend(subtasks)

        listing = "\n\n".join(
            f"{sub.task}\n{sub.result}" for sub in all_subtasks if sub.ok
        )
        synthesized = ask(
            self.model,
            MANAGER_SYNTH_SYSTEM,
            f"GOAL:\n{goal}\n\nWORKER OUTPUTS:\n{listing}",
        )
        answer = synthesized.strip() or listing
        return TeamResult(answer, all_subtasks, trace, round_index)
