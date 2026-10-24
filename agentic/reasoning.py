"""Prompt-level reasoning strategies.

These spend extra model calls to get a better answer out of a fixed model:

  chain_of_thought   scratchpad reasoning; with samples>1 it also does
                     self-consistency (sample k traces, majority-vote the
                     final answers).
  plan_and_solve     two-stage: write a numbered plan first, then execute it,
                     then synthesise the deliverable. Optionally verify.
  react              interleaved Thought / Action / Observation loop that can
                     call real tools (function calling / tool use).
  self_ask           follow-up-question decomposition for multi-hop questions.
  debate             several independent answers, N rounds of mutual critique,
                     then a judge picks or a consensus emerges.
  reflexion          attempt -> evaluate -> verbal reflection -> retry.

Every function takes any object exposing ``chat(messages, tools=None)`` and
returns a :class:`StrategyResult` so the caller keeps the trace, not just the
text. Nothing here mutates project files: acting on the repo is the Agent's
job, and ReAct reaches it through the ``run_tool`` callable you pass in.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from agentic.base import (
    TraceEvent,
    ask,
    complete,
    extract_final,
    extract_json,
    majority_vote,
    parse_score,
    truncate,
)


@dataclass
class StrategyResult:
    """Outcome of any reasoning strategy."""

    answer: str
    trace: List[TraceEvent] = field(default_factory=list)
    candidates: List[str] = field(default_factory=list)
    scores: List[float] = field(default_factory=list)
    steps: int = 0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "answer": self.answer,
            "steps": self.steps,
            "candidates": len(self.candidates),
            "scores": self.scores,
            "trace": [
                {"step": e.step, "kind": e.kind, "detail": e.detail, "score": e.score}
                for e in self.trace
            ],
        }


# ---------------------------------------------------------------------------
# Chain of Thought (+ self-consistency)
# ---------------------------------------------------------------------------

COT_SYSTEM = (
    "You are a careful reasoner. Work through the problem step by step in a "
    "scratchpad: state assumptions, check arithmetic or logic, and consider at "
    "least one alternative before committing. Then give the answer on its own "
    "line starting with 'FINAL ANSWER:'."
)


def chain_of_thought(
    model: Any,
    task: str,
    samples: int = 1,
    context: str = "",
) -> StrategyResult:
    """Scratchpad reasoning; ``samples > 1`` adds self-consistency voting.

    With one sample this is plain CoT. With more, each sample is independent
    (a fresh conversation, no shared state), the final answers are grouped by
    normalised text, and the largest group wins. The agreement share lands in
    ``scores`` so a caller can see how confident the vote was.
    """
    samples = max(1, samples)
    prompt = f"{task}\n\n{context}".strip()
    trace: List[TraceEvent] = []
    candidates: List[str] = []
    for index in range(samples):
        reply = ask(model, COT_SYSTEM, prompt)
        answer = extract_final(reply, fallback=reply)
        candidates.append(answer)
        trace.append(TraceEvent(index + 1, "cot-sample", truncate(reply, 400)))
    if samples == 1:
        return StrategyResult(candidates[0] if candidates else "", trace, candidates, [], 1)

    finals = [extract_final(c) for c in candidates]
    winner, votes = majority_vote(finals)
    # Prefer the winning variant's full text (the extract can clip detail).
    chosen_full = next((c for c, f in zip(candidates, finals) if f == winner), winner)
    trace.append(
        TraceEvent(
            samples + 1,
            "self-consistency",
            f"{votes}/{samples} agreement",
            votes / samples,
        )
    )
    return StrategyResult(
        chosen_full or winner, trace, candidates, [votes / samples] * samples, samples + 1
    )


# ---------------------------------------------------------------------------
# Plan and Solve
# ---------------------------------------------------------------------------

PLAN_SYSTEM = (
    "You are a planner. Break the task into a short numbered list of concrete "
    "steps (5 or fewer) that a competent executor can follow in order. Each "
    "step must name the action, not the goal. Output only the numbered list."
)

SOLVE_SYSTEM = (
    "You are executing a plan. Follow the steps in order, showing the result "
    "of each step briefly as you go. If a step cannot be completed, say so and "
    "adjust. End with 'FINAL ANSWER:' followed by the deliverable."
)

VERIFY_SYSTEM = (
    "You review a deliverable against its task and plan. List only concrete "
    "defects that would make it wrong or incomplete. If there are none, reply "
    "with exactly APPROVED."
)


def plan_and_solve(
    model: Any,
    task: str,
    verify: bool = True,
    context: str = "",
) -> StrategyResult:
    """Plan first, then execute the plan, then optionally verify and fix.

    Plan-and-Solve beats plain CoT on multi-step tasks because the plan acts
    as a checklist the second call must honour; the verify pass catches the
    common failure where a step was skipped silently.
    """
    trace: List[TraceEvent] = []
    plan_text = ask(model, PLAN_SYSTEM, f"{task}\n\n{context}".strip())
    if not plan_text:
        # Fall back to CoT rather than running an empty plan.
        result = chain_of_thought(model, task, context=context)
        result.trace.insert(0, TraceEvent(0, "plan", "planner returned nothing; fell back to CoT"))
        return result
    trace.append(TraceEvent(1, "plan", truncate(plan_text, 600)))

    solved = ask(
        model,
        SOLVE_SYSTEM,
        f"TASK:\n{task}\n\nPLAN:\n{plan_text}\n\n{context}".strip(),
    )
    answer = extract_final(solved, fallback=solved)
    trace.append(TraceEvent(2, "execute", truncate(solved, 600)))

    if verify:
        critique = ask(
            model,
            VERIFY_SYSTEM,
            f"TASK:\n{task}\n\nPLAN:\n{plan_text}\n\nDELIVERABLE:\n{answer}",
        )
        trace.append(TraceEvent(3, "verify", truncate(critique, 600)))
        if critique and not critique.upper().startswith("APPROVED"):
            fixed = ask(
                model,
                SOLVE_SYSTEM,
                (
                    f"TASK:\n{task}\n\nPLAN:\n{plan_text}\n\n"
                    f"DRAFT:\n{answer}\n\nDEFECTS TO FIX:\n{critique}\n\n"
                    "Produce the corrected deliverable only."
                ),
            )
            if fixed.strip():
                answer = extract_final(fixed, fallback=fixed)
                trace.append(TraceEvent(4, "revise", truncate(fixed, 600)))

    return StrategyResult(answer, trace, [solved, answer], [], len(trace))


# ---------------------------------------------------------------------------
# ReAct (reason + act, with real tool calls)
# ---------------------------------------------------------------------------

REACT_SYSTEM = (
    "You solve tasks by interleaving reasoning and tool use. On every turn "
    "output exactly one of these two shapes:\n\n"
    "Thought: <your reasoning>\n"
    "Action: <tool name>\n"
    "Action Input: <JSON object of arguments>\n\n"
    "or, when the task is complete:\n\n"
    "Thought: <why you are done>\n"
    "Final Answer: <the answer>\n\n"
    "Use only the tools listed. Never invent tool names. Keep observations in "
    "mind and do not repeat a call that already failed the same way."
)


def _parse_react(reply: str) -> Tuple[Optional[str], Dict[str, Any], Optional[str]]:
    """Split a ReAct turn into (tool_name, arguments, final_answer).

    Exactly one of the first two return values is meaningful: final_answer is
    set when the model declared completion. Arguments default to {} when the
    model writes something that is not valid JSON.
    """
    text = reply or ""
    lowered = text.lower()
    final_index = lowered.rfind("final answer:")
    if final_index != -1:
        final = text[final_index + len("final answer:") :].strip()
        if final:
            return None, {}, final
    action_index = lowered.find("action:")
    if action_index == -1:
        return None, {}, None
    remainder = text[action_index + len("action:") :]
    lines = remainder.splitlines()
    if not lines:
        return None, {}, None
    name = lines[0].strip().strip("`").strip()
    if not name or name.lower().startswith("final"):
        return None, {}, None
    rest = "\n".join(lines[1:])
    args: Dict[str, Any] = {}
    input_index = rest.lower().find("action input:")
    if input_index != -1:
        raw = rest[input_index + len("action input:") :].strip()
        parsed = extract_json(raw)
        if isinstance(parsed, dict):
            args = parsed
        elif isinstance(parsed, str):
            args = {"input": parsed}
        elif raw:
            args = {"input": raw.splitlines()[0].strip()}
    return name, args, None


def react(
    model: Any,
    task: str,
    tool_descriptions: Dict[str, str],
    run_tool: Callable[[str, Dict[str, Any]], str],
    max_steps: int = 8,
    context: str = "",
) -> StrategyResult:
    """Thought -> Action -> Observation loop until a final answer.

    ``tool_descriptions`` is what the model is told it may call; ``run_tool``
    is the only way it touches the world. A hallucinated tool name becomes an
    observation telling it the name is unknown, which is usually enough for the
    next turn to correct course — no exception is raised into the loop.
    """
    catalog = "\n".join(f"- {name}: {desc}" for name, desc in tool_descriptions.items())
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": f"{REACT_SYSTEM}\n\nAvailable tools:\n{catalog}"},
        {"role": "user", "content": f"TASK:\n{task}\n\n{context}".strip()},
    ]
    trace: List[TraceEvent] = []
    observations: List[str] = []

    for step in range(1, max_steps + 1):
        reply = complete(model, messages)
        if not reply:
            trace.append(TraceEvent(step, "react", "empty reply; stopping"))
            break
        name, args, final = _parse_react(reply)
        trace.append(TraceEvent(step, "thought", truncate(reply, 500)))

        if final is not None:
            return StrategyResult(final, trace, observations, [], step)
        if not name:
            # No parseable action and no final answer: ask for one more turn
            # with the shape spelled out, then give up rather than loop.
            messages.append({"role": "assistant", "content": reply})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Your reply had neither 'Action:' nor 'Final Answer:'. "
                        "Reply again in the required shape."
                    ),
                }
            )
            continue
        if name not in tool_descriptions:
            observation = f"Unknown tool '{name}'. Available: {', '.join(tool_descriptions)}."
        else:
            try:
                observation = run_tool(name, args)
            except Exception as exc:  # tool failure is data, not a crash
                observation = f"Tool '{name}' raised: {exc}"
        observation = truncate(str(observation), 4000)
        observations.append(f"{name}({json.dumps(args, ensure_ascii=False)}) -> {observation}")
        trace.append(
            TraceEvent(step, "observation", truncate(observation, 400), meta={"tool": name})
        )
        messages.append({"role": "assistant", "content": reply})
        messages.append({"role": "user", "content": f"Observation:\n{observation}"})

    # Step budget exhausted: ask for a best-effort answer from what was seen.
    summary = ask(
        model,
        "You could not finish the tool loop. Using only the observations below, "
        "give the best answer you can, or state plainly what is still missing.",
        f"TASK:\n{task}\n\nOBSERVATIONS:\n" + "\n\n".join(observations[-10:]),
    )
    trace.append(TraceEvent(max_steps + 1, "forced-answer", truncate(summary, 400)))
    return StrategyResult(
        extract_final(summary, fallback=summary), trace, observations, [], max_steps + 1
    )


# ---------------------------------------------------------------------------
# Self-Ask
# ---------------------------------------------------------------------------

def self_ask(model: Any, task: str, max_hops: int = 4) -> StrategyResult:
    """Decompose by asking and answering sub-questions before the final answer.

    Useful for multi-hop questions (entity chains, comparisons) where one-shot
    answering tends to skip a link. Each sub-question is answered in the same
    growing transcript so later hops can use earlier facts.
    """
    trace: List[TraceEvent] = []
    transcript = f"QUESTION: {task}\n"
    for hop in range(1, max_hops + 1):
        follow_up = ask(
            model,
            "You decompose a question into sub-questions. If more information is "
            "needed, reply with exactly one line: 'FOLLOW-UP: <question>'. "
            "If everything is answered, reply with exactly 'DONE'.",
            transcript,
        )
        if not follow_up or follow_up.strip().upper().startswith("DONE"):
            break
        question = follow_up.strip()
        if question.upper().startswith("FOLLOW-UP:"):
            question = question.split(":", 1)[1].strip()
        if not question:
            break
        answer = ask(
            model,
            "Answer the sub-question in one or two sentences, using the transcript "
            "where relevant. Say 'unknown' if it cannot be answered.",
            transcript + f"\nSUB-QUESTION: {question}\n",
        )
        trace.append(TraceEvent(hop, "sub-question", truncate(question, 300)))
        trace.append(TraceEvent(hop, "sub-answer", truncate(answer, 300)))
        transcript += f"\nSUB-QUESTION: {question}\nSUB-ANSWER: {answer}\n"

    final = ask(
        model,
        "Answer the original question using the established facts. "
        "End with 'FINAL ANSWER:' on its own line.",
        transcript,
    )
    trace.append(TraceEvent(len(trace) + 1, "final", truncate(final, 500)))
    return StrategyResult(extract_final(final, fallback=final), trace, [], [], len(trace))


# ---------------------------------------------------------------------------
# Debate and consensus
# ---------------------------------------------------------------------------

DEBATE_PERSONA = (
    "You are debater {index}. Answer the question on your own first, then defend "
    "or revise your position based only on evidence and reasoning — not on "
    "majority pressure. End with 'FINAL ANSWER:' and your position."
)

JUDGE_DEBATE_SYSTEM = (
    "You judge a debate between independent answers. Weigh the arguments and "
    "the evidence, not the number of speakers. Reply with a single line: "
    "'WINNER: <number>' followed by one sentence of justification."
)


@dataclass
class DebateResult(StrategyResult):
    """Debate adds a machine-readable consensus ratio to StrategyResult."""

    agreement: float = 0.0
    winner: int = 0
    positions: List[str] = field(default_factory=list)


def debate(
    model: Any,
    task: str,
    debaters: int = 3,
    rounds: int = 2,
    judge_model: Any = None,
    context: str = "",
) -> DebateResult:
    """Independent answers -> peer critique rounds -> judge.

    Round 0 samples each debater independently so the pool is diverse. Later
    rounds show every debater the others' current positions and let them
    revise. ``agreement`` is the share of debaters whose final answer matches
    the plurality; it is reported rather than enforced, so a genuinely split
    debate still produces a judge-selected answer with a low agreement score.
    """
    debaters = max(2, debaters)
    rounds = max(0, rounds)
    judge = judge_model or model
    trace: List[TraceEvent] = []
    positions: List[str] = []

    for index in range(1, debaters + 1):
        reply = ask(
            model,
            DEBATE_PERSONA.format(index=index),
            f"QUESTION:\n{task}\n\n{context}".strip(),
        )
        positions.append(extract_final(reply, fallback=reply))
        trace.append(TraceEvent(index, "opening", truncate(reply, 400)))

    for round_index in range(1, rounds + 1):
        revised: List[str] = []
        for index in range(debaters):
            others = [
                f"Debater {j + 1}: {positions[j]}" for j in range(debaters) if j != index
            ]
            prompt = (
                f"QUESTION:\n{task}\n\nYOUR CURRENT POSITION:\n{positions[index]}\n\n"
                "OTHER POSITIONS:\n"
                + "\n\n".join(others)
                + "\n\nRevise or defend. Be concrete about any factual disagreement."
            )
            reply = ask(model, DEBATE_PERSONA.format(index=index + 1), prompt)
            revised.append(extract_final(reply, fallback=reply))
            trace.append(
                TraceEvent(
                    debaters + (round_index - 1) * debaters + index + 1,
                    f"round-{round_index}",
                    truncate(reply, 400),
                )
            )
        positions = revised

    winner_answer, votes = majority_vote(positions)
    agreement = votes / max(1, debaters)
    listing = "\n\n".join(f"DEBATER {i + 1}:\n{p}" for i, p in enumerate(positions))
    verdict = ask(judge, JUDGE_DEBATE_SYSTEM, f"QUESTION:\n{task}\n\n{listing}")
    winner = 1
    for token in verdict.replace(":", " ").split():
        if token.isdigit():
            candidate = int(token)
            if 1 <= candidate <= len(positions):
                winner = candidate
                break
    trace.append(TraceEvent(len(trace) + 1, "judgement", truncate(verdict, 400), agreement))

    # Judge wins on a split; a clear consensus overrides a stray judge pick.
    chosen = positions[winner - 1] if agreement < 0.7 else winner_answer
    return DebateResult(
        answer=chosen,
        trace=trace,
        candidates=positions,
        scores=[agreement] * len(positions),
        steps=len(trace),
        agreement=agreement,
        winner=winner,
        positions=positions,
    )


# ---------------------------------------------------------------------------
# Reflexion (self-reflection across attempts)
# ---------------------------------------------------------------------------

EVALUATE_SYSTEM = (
    "You evaluate an attempt against its task. Give a 0-10 score for whether "
    "the attempt would satisfy the task as written, then list concrete defects. "
    "Format:\nSCORE: <n>\nDEFECTS:\n- ..."
)

REFLECT_SYSTEM = (
    "You are the agent that just failed. Write a short, specific reflection: "
    "what went wrong, and the rule you will follow next attempt. Do not restate "
    "the task. Two or three sentences."
)


@dataclass
class ReflexionResult(StrategyResult):
    """Reflexion keeps every attempt plus the lessons learned between them."""

    reflections: List[str] = field(default_factory=list)
    best_score: float = 0.0


def reflexion(
    model: Any,
    task: str,
    attempt_fn: Callable[[str, List[str]], str],
    max_attempts: int = 3,
    threshold: float = 0.8,
    context: str = "",
) -> ReflexionResult:
    """Attempt, evaluate, reflect, retry — keeping the best-scoring attempt.

    ``attempt_fn(task, reflections) -> answer`` performs one attempt. Keeping
    the reflection text in the retry prompt (verbal reinforcement) is what
    distinguishes this from a plain retry loop: the failure mode from attempt
    n is named explicitly to attempt n+1.
    """
    reflections: List[str] = []
    trace: List[TraceEvent] = []
    best_answer = ""
    best_score = -1.0
    attempts: List[str] = []

    for index in range(1, max(1, max_attempts) + 1):
        answer = attempt_fn(task, list(reflections))
        attempts.append(answer)
        evaluation = ask(
            model,
            EVALUATE_SYSTEM,
            f"TASK:\n{task}\n\nATTEMPT:\n{truncate(answer, 6000)}",
        )
        score = parse_score(evaluation, default=0.5)
        trace.append(TraceEvent(index, "attempt", truncate(answer, 400), score))
        if score > best_score:
            best_score, best_answer = score, answer
        if score >= threshold:
            break
        reflection = ask(
            model,
            REFLECT_SYSTEM,
            f"TASK:\n{task}\n\nATTEMPT:\n{truncate(answer, 3000)}\n\n"
            f"EVALUATION:\n{evaluation}",
        )
        if reflection and reflection.upper() != "NONE":
            reflections.append(reflection.strip())
            trace.append(TraceEvent(index, "reflection", truncate(reflection, 400)))

    trace.append(TraceEvent(len(trace) + 1, "chosen", f"best score {best_score:.2f}", best_score))
    return ReflexionResult(
        answer=best_answer,
        trace=trace,
        candidates=attempts,
        scores=[best_score],
        steps=len(trace),
        reflections=reflections,
        best_score=max(0.0, best_score),
    )
