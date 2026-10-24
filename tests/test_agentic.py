"""Tests for the agentic technique package: reasoning, search, memory, tools,
orchestration, and training.

Models are stubbed callables, so the suite is offline and deterministic. The
goal is to pin each technique's contract: right result type, budget knobs
honoured, and no silent swallowing of malformed model output.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import agentic
from agentic import (
    CodeInterpreter,
    GraphRAG,
    HierarchicalOrchestrator,
    LongTermMemory,
    PreferencePair,
    RewardModel,
    SelfImprover,
    SOPRunner,
    SOPStep,
    Trajectory,
    WebBrowser,
    WorkingMemory,
    build_preference_pairs,
    chain_of_thought,
    compute_advantages,
    export_dpo,
    export_sft,
    graph_of_thoughts,
    mcts,
    plan_and_solve,
    react,
    reflexion,
    run_search,
    run_technique,
    self_ask,
    trajectories_from_history,
    tree_of_thoughts,
)

# ---------------------------------------------------------------------------
# Stub model
# ---------------------------------------------------------------------------


class ScriptedModel:
    """Returns canned replies in order, then repeats the last one."""

    def __init__(self, replies):
        self.replies = list(replies) if isinstance(replies, list) else [replies]
        self.calls = 0

    def chat(self, messages, tools=None):
        index = min(self.calls, len(self.replies) - 1)
        self.calls += 1
        reply = self.replies[index]
        return {"content": reply(messages) if callable(reply) else reply}


def test_package_exports_every_requested_technique():
    expected = {
        "chain_of_thought", "plan_and_solve", "react", "self_ask", "debate", "reflexion",
        "tree_of_thoughts", "graph_of_thoughts", "mcts", "run_search", "run_technique",
        "WorkingMemory", "LongTermMemory", "VectorStore", "GraphRAG", "MemoryStream",
        "CodeInterpreter", "WebBrowser", "ComputerUse",
        "SOPRunner", "HierarchicalOrchestrator",
        "Trajectory", "RewardModel", "SelfImprover", "export_sft", "export_dpo",
        "compute_advantages", "bootstrap_to_sft",
    }
    assert expected <= set(agentic.__all__)
    families = agentic.available_techniques()
    assert {"cot", "react", "reflexion"} <= set(families["reasoning"])
    assert {"tot", "got", "mcts"} <= set(families["search"])


# ---------------------------------------------------------------------------
# Reasoning
# ---------------------------------------------------------------------------


def test_cot_extracts_final_answer():
    result = chain_of_thought(ScriptedModel("Step 1: think.\nFINAL ANSWER: 42"), "6*7?")
    assert result.answer == "42"


def test_cot_self_consistency_picks_majority():
    model = ScriptedModel(["FINAL ANSWER: 42", "FINAL ANSWER: 41", "FINAL ANSWER: 42"])
    result = chain_of_thought(model, "6*7?", samples=3)
    assert result.answer == "42"


def test_plan_and_solve_produces_answer():
    model = ScriptedModel(["Plan:\n1. read file\n2. patch bug", "FINAL ANSWER: patched"])
    result = plan_and_solve(model, "fix the bug", verify=False)
    assert result.answer == "patched"


def test_plan_and_solve_falls_back_to_cot_on_empty_plan():
    model = ScriptedModel(["", "FINAL ANSWER: fallback"])
    result = plan_and_solve(model, "task", verify=False)
    assert result.answer == "fallback"


def test_self_ask_returns_answer():
    model = ScriptedModel(
        [
            "Follow-up: what is 6*7?\nIntermediate answer: 42",
            "FINAL ANSWER: 42",
        ]
    )
    result = self_ask(model, "what is 6*7", max_hops=2)
    assert result.answer


def test_react_stops_at_final_answer():
    model = ScriptedModel(
        [
            'Thought: need the file\nAction: read_file\nAction Input: {"path": "a.py"}',
            "Thought: done\nFinal Answer: the file is fine",
        ]
    )
    seen = []

    def run_tool(name, arguments):
        seen.append((name, arguments))
        return "print('hi')"

    result = react(
        model,
        "check a.py",
        tool_descriptions={"read_file": "read a file by path"},
        run_tool=run_tool,
        max_steps=4,
    )
    assert "fine" in result.answer
    assert seen == [("read_file", {"path": "a.py"})]


def test_reflexion_stops_once_threshold_met():
    model = ScriptedModel(["SCORE: 0.95 - good"])
    result = reflexion(
        model,
        "solve it",
        attempt_fn=lambda task, reflections: "an attempt",
        max_attempts=3,
        threshold=0.8,
    )
    assert result.answer == "an attempt"
    assert result.best_score >= 0.8
    assert model.calls == 1


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


class SearchModel:
    """Prompt-shaped stub: each search stage gets the JSON its parser expects.

    Search strategies ask three different questions (expand / score /
    finalize). Dispatching on the system prompt keeps one stub valid for ToT,
    GoT, and MCTS without the test knowing their internal call order.
    """

    def __init__(self):
        self.calls = 0
        self._step = 0

    def chat(self, messages, tools=None):
        self.calls += 1
        system = messages[0].get("content", "")
        if "expanding a reasoning tree" in system or "DIFFERENT next steps" in system:
            self._step += 1
            return {"content": json.dumps([f"step A{self._step}", f"step B{self._step}"])}
        if "You score partial solutions" in system:
            return {"content": json.dumps({"1": 9, "2": 4})}
        if "merge" in system or "improve one partial solution" in system:
            return {"content": "merged step"}
        return {"content": "FINAL ANSWER: the answer"}


def test_tree_of_thoughts_returns_answer():
    result = tree_of_thoughts(SearchModel(), "pick an approach", breadth=2, depth=2)
    assert result.answer
    assert len(result.all_nodes) > 1
    assert result.best_path


def test_graph_of_thoughts_returns_answer():
    result = graph_of_thoughts(SearchModel(), "combine ideas", expand_rounds=1, breadth=2)
    assert result.answer


def test_mcts_returns_answer():
    result = mcts(SearchModel(), "find a move", simulations=3)
    assert result.answer


def test_tree_of_thoughts_stops_early_on_high_score():
    """A 0.9/1.0 branch at depth 1 must not keep expanding to the depth limit."""
    model = SearchModel()
    result = tree_of_thoughts(model, "easy task", breadth=2, depth=4, beam_width=1)
    assert result.answer
    expands = sum(1 for event in result.trace if event.kind == "expand")
    assert expands < 4


def test_run_search_rejects_unknown_strategy():
    with pytest.raises(ValueError):
        run_search("nope", ScriptedModel("x"), "task")


def test_run_technique_dispatches_both_families():
    assert run_technique("cot", ScriptedModel("FINAL ANSWER: ok"), "t").answer == "ok"
    assert run_technique("tot", SearchModel(), "t").answer
    with pytest.raises(ValueError):
        run_technique("bogus", ScriptedModel("x"), "t")


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------


def test_working_memory_keeps_pinned_item_under_pressure():
    memory = WorkingMemory(capacity=2)
    memory.put("goal", "ship the fix", importance=1.0, pinned=True)
    memory.put("note", "low value", importance=0.1)
    memory.put("note2", "medium value", importance=0.5)
    assert memory.get("goal") == "ship the fix"
    assert len(memory.items()) <= 2


def test_long_term_memory_survives_reload(tmp_path):
    path = tmp_path / "memory.jsonl"
    memory = LongTermMemory(path)
    memory.remember("the build uses uv", importance=0.8)
    memory.remember("tests run with pytest", importance=0.6)
    hits = LongTermMemory(path).recall("uv build", k=1)
    assert hits and "uv" in hits[0].text


def test_graph_rag_extracts_triples_and_renders_context():
    graph = GraphRAG()
    graph.ingest(ScriptedModel("not json"), "The Agent uses the ModelClient.")
    results = graph.query("Agent", hops=2)
    assert results
    assert "ModelClient" in graph.render(results)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


def test_code_interpreter_keeps_state_between_calls():
    interpreter = CodeInterpreter()
    try:
        first = interpreter.run("x = 41")
        second = interpreter.run("print(x + 1)")
        assert first.ok and second.ok
        assert "42" in second.output
    finally:
        interpreter.close()


def test_web_browser_research_uses_pluggable_search_and_fetch():
    browser = WebBrowser(
        search_fn=lambda query, limit: "1. Rust Book\nURL: https://example.test/a",
        fetch_fn=lambda url: "Rust ownership means one owner per value.\n" * 20,
    )
    context = browser.research("rust ownership", max_results=1, top_per_page=1)
    assert "ownership" in context.lower()
    assert "https://example.test/a" in context


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def test_sop_runner_executes_steps_in_order():
    outputs = []
    runner = SOPRunner(run_fn=lambda prompt: outputs.append(prompt) or f"done:{len(outputs)}")
    result = runner.run(
        "ship feature",
        [
            SOPStep("plan", "plan for {goal}"),
            SOPStep("implement", "implement using {plan}"),
        ],
    )
    assert result.ok
    assert list(result.outputs) == ["plan", "implement"]
    assert "done:1" in outputs[1]


def test_sop_runner_retries_failed_step():
    calls = {"n": 0}

    def run_fn(prompt):
        calls["n"] += 1
        return "bad" if calls["n"] == 1 else "good"

    runner = SOPRunner(run_fn=run_fn)
    result = runner.run(
        "goal",
        [SOPStep("check", "do it", retries=2, validate=lambda out: (out == "good", "not good"))],
    )
    assert result.ok
    assert calls["n"] == 2


def test_hierarchical_orchestrator_runs_workers():
    manager_replies = [
        (
            '[{"id": "a", "task": "sub one", "depends_on": []},'
            ' {"id": "b", "task": "sub two", "depends_on": ["a"]}]'
        ),
        "APPROVED: all good",
    ]
    result = HierarchicalOrchestrator(
        ScriptedModel(manager_replies),
        worker_fn=lambda task, upstream: f"did {task}",
        max_workers=2,
    ).run("ship it")
    assert result.answer
    assert {sub.id for sub in result.subtasks} == {"a", "b"}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def test_trajectories_from_history_folds_tool_calls():
    history = [
        {"role": "user", "content": "fix the bug"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "1"}]},
        {"role": "tool", "content": "file contents"},
        {"role": "assistant", "content": "fixed it"},
    ]
    trajectories = trajectories_from_history(history)
    assert len(trajectories) == 1
    assert trajectories[0].answer == "fixed it"
    assert len(trajectories[0].steps) == 2


def test_export_sft_writes_standard_rows(tmp_path):
    out = tmp_path / "sft.jsonl"
    written = export_sft(
        [Trajectory(instruction="q", answer="a"), Trajectory(instruction="q2", answer="a2")], out
    )
    assert written == 2
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert set(rows[0]) == {"instruction", "input", "output", "steps", "reward"}


def test_reward_model_parses_score_and_prefers_b():
    reward = RewardModel(ScriptedModel(["SCORE: 0.75 - solid", "WINNER: B because it is safer"]))
    assert reward.score("q", "a") == pytest.approx(0.75)
    assert reward.prefer("q", "a", "b") == -1


def test_build_preference_pairs_pairs_best_against_worst():
    scores = iter(["SCORE: 0.2", "SCORE: 0.9", "SCORE: 0.5"])
    reward = RewardModel(ScriptedModel([lambda messages: next(scores, "SCORE: 0.5")]))
    pairs = build_preference_pairs("q", ["bad", "good", "mid"], reward)
    assert pairs[0].chosen == "good"
    assert pairs[0].rejected == "bad"
    assert pairs[0].margin > 0


def test_export_dpo_writes_prompt_chosen_rejected(tmp_path):
    out = tmp_path / "dpo.jsonl"
    export_dpo([PreferencePair("q", "good", "bad", 0.4)], out)
    row = json.loads(out.read_text(encoding="utf-8").strip())
    assert (row["prompt"], row["chosen"], row["rejected"]) == ("q", "good", "bad")


def test_compute_advantages_reduces_to_returns_without_baseline():
    assert compute_advantages([1.0, 1.0], gamma=1.0, lam=1.0)["advantages"] == [2.0, 1.0]


def test_compute_advantages_rejects_mismatched_baseline():
    with pytest.raises(ValueError):
        compute_advantages([1.0, 2.0], values=[0.0])


def test_self_improver_keeps_only_verified_trajectories():
    improver = SelfImprover(
        ScriptedModel(["write a parser task", "write a cache task"]),
        solver=lambda task: "FINAL ANSWER: " + task,
        verifier=lambda task, answer: "parser" in task,
    )
    report = improver.run(["compilers"], per_topic=2)
    assert report.attempted == 2
    assert report.kept == 1
    assert report.trajectories[0].reward == 1.0
