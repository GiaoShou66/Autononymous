"""Inference-time techniques: spend extra model calls to get a better answer.

reflect()    - draft -> critique -> revise, repeated until the critic approves
               or the round budget runs out.
best_of_n()  - sample N independent answers, then have the model pick the
               strongest one.

Both work on any object with a `chat(messages)` method returning a dict with
a "content" key (ModelClient, or a stub in tests). Neither uses tools: they
refine text answers, they don't act on the project.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

Message = Dict[str, Any]

CRITIC_SYSTEM = (
    "You are a strict reviewer. You are given a task and a draft answer. "
    "Find concrete, real problems: factual errors, bugs, missing requirements, "
    "unsafe or misleading claims. Ignore style preferences. If the draft has no "
    "real problems, reply with exactly the single word APPROVED. Otherwise reply "
    "with a short bullet list of the specific problems and how to fix each."
)

REVISER_SYSTEM = (
    "You are revising a draft answer. You are given the task, the draft, and a "
    "reviewer's list of problems. Produce the corrected final answer only, with "
    "no preamble and no mention of the review."
)

JUDGE_SYSTEM = (
    "You are choosing the best of several candidate answers to a task. Prefer "
    "the answer that is most correct and complete. Reply with only the number "
    "of the best candidate."
)


def _text(response: Message) -> str:
    return str(response.get("content") or "").strip()


def reflect(model: Any, task: str, answer: str, rounds: int = 1) -> Tuple[str, List[str]]:
    """Critique and revise `answer` up to `rounds` times.

    Returns (final_answer, critiques). `critiques` holds each non-approving
    critique in order; an empty list means the first review approved as-is.
    """
    critiques: List[str] = []
    for _ in range(max(0, rounds)):
        critique = _text(
            model.chat(
                [
                    {"role": "system", "content": CRITIC_SYSTEM},
                    {"role": "user", "content": f"TASK:\n{task}\n\nDRAFT:\n{answer}"},
                ]
            )
        )
        if not critique or critique.upper().startswith("APPROVED"):
            break
        critiques.append(critique)
        revised = _text(
            model.chat(
                [
                    {"role": "system", "content": REVISER_SYSTEM},
                    {
                        "role": "user",
                        "content": (
                            f"TASK:\n{task}\n\nDRAFT:\n{answer}\n\nPROBLEMS:\n{critique}"
                        ),
                    },
                ]
            )
        )
        if revised:
            answer = revised
    return answer, critiques


def best_of_n(
    model: Any, messages: List[Message], n: int = 3
) -> Tuple[str, List[str], int]:
    """Sample `n` answers to `messages` and pick the best.

    Returns (best_answer, all_candidates, chosen_index). With n <= 1 this is
    a single plain call. If the judge's reply has no usable number, the first
    candidate wins.
    """
    n = max(1, n)
    candidates = [c for c in (_text(model.chat(messages)) for _ in range(n)) if c]
    if not candidates:
        return "", [], 0
    if len(candidates) == 1:
        return candidates[0], candidates, 0

    task = next(
        (str(m.get("content") or "") for m in reversed(messages) if m.get("role") == "user"),
        "",
    )
    listing = "\n\n".join(f"CANDIDATE {i + 1}:\n{c}" for i, c in enumerate(candidates))
    verdict = _text(
        model.chat(
            [
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": f"TASK:\n{task}\n\n{listing}"},
            ]
        )
    )
    match = re.search(r"\d+", verdict)
    chosen = int(match.group()) - 1 if match else 0
    if not 0 <= chosen < len(candidates):
        chosen = 0
    return candidates[chosen], candidates, chosen
