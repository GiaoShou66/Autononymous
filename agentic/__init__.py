"""Agentic technique library: reasoning, search, memory, tools, orchestration, training.

Every technique in the package is model-agnostic. A "model" is any object with
``chat(messages)`` (the project's ``ModelClient``) or any callable taking a
message list and returning text, so the same code runs against a live model or
a stub in tests.

Where each requested technique lives:

Reasoning (``agentic.reasoning``)
    chain_of_thought ......... CoT, with self-consistency sampling
    plan_and_solve ........... Plan-and-Solve prompting
    react .................... ReAct (thought/action/observation loop)
    self_ask ................. Self-Ask decomposition
    debate ................... Debate & consensus
    reflexion ................ Self-reflection / Reflexion
    run_strategy ............. dispatch by name

Search (``agentic.search``)
    tree_of_thoughts ......... ToT
    graph_of_thoughts ........ GoT
    mcts ..................... MCTS
    run_search ............... dispatch by name

Memory (``agentic.memory_store``)
    WorkingMemory ............ short-term / working memory
    LongTermMemory + VectorStore ... long-term memory, Vector RAG
    GraphRAG ................. Graph RAG
    MemoryStream ............. Generative-Agents memory stream

Tools (``agentic.tools_ext``)
    CodeInterpreter .......... code interpreter
    WebBrowser ............... web browsing / RAG search
    ComputerUse .............. computer use / GUI automation

Orchestration (``agentic.orchestration``)
    SOPRunner ................ SOP workflow
    HierarchicalOrchestrator . manager-worker hierarchy

Training (``agentic.training``)
    Trajectory, export_sft ........... instruction fine-tuning
    RewardModel ...................... RLHF / RLAIF feedback
    build_preference_pairs, export_dpo ... DPO
    compute_advantages ............... PPO (GAE)
    SelfImprover ..................... bootstrapping / self-improvement
"""

from __future__ import annotations

from typing import Any

from agentic.base import (
    TraceEvent,
    ask,
    complete,
    estimate_tokens,
    extract_final,
    extract_json,
    parse_score,
    render_trace,
    truncate,
)
from agentic.memory_store import (
    GraphRAG,
    LongTermMemory,
    MemoryRecord,
    MemoryStream,
    VectorStore,
    WorkingMemory,
)
from agentic.orchestration import (
    HierarchicalOrchestrator,
    SOPResult,
    SOPRunner,
    SOPStep,
    TeamResult,
)
from agentic.reasoning import (
    DebateResult,
    ReflexionResult,
    StrategyResult,
    chain_of_thought,
    debate,
    plan_and_solve,
    react,
    reflexion,
    self_ask,
)
from agentic.search import (
    SearchResult,
    graph_of_thoughts,
    mcts,
    run_search,
    tree_of_thoughts,
)
from agentic.tools_ext import (
    CodeInterpreter,
    CodeResult,
    ComputerUse,
    WebBrowser,
    format_code_result,
)
from agentic.training import (
    BootstrapReport,
    PreferencePair,
    RewardModel,
    SelfImprover,
    Trajectory,
    bootstrap_to_sft,
    build_preference_pairs,
    compute_advantages,
    export_dpo,
    export_sft,
    export_sft_from_sessions,
    trajectories_from_history,
)

__all__ = [
    # base
    "TraceEvent", "ask", "complete", "estimate_tokens", "extract_final",
    "extract_json", "parse_score", "render_trace", "truncate",
    # reasoning
    "StrategyResult", "DebateResult", "ReflexionResult",
    "chain_of_thought", "plan_and_solve", "react", "self_ask", "debate", "reflexion",
    # search
    "SearchResult", "tree_of_thoughts", "graph_of_thoughts", "mcts", "run_search",
    # memory
    "WorkingMemory", "LongTermMemory", "MemoryRecord", "VectorStore", "GraphRAG", "MemoryStream",
    # tools
    "CodeInterpreter", "CodeResult", "format_code_result", "WebBrowser", "ComputerUse",
    # orchestration
    "SOPStep", "SOPResult", "SOPRunner", "TeamResult", "HierarchicalOrchestrator",
    # training
    "Trajectory", "RewardModel", "PreferencePair", "BootstrapReport", "SelfImprover",
    "export_sft", "export_sft_from_sessions", "trajectories_from_history",
    "build_preference_pairs", "export_dpo", "compute_advantages", "bootstrap_to_sft",
    # dispatch
    "REASONING_STRATEGIES", "SEARCH_STRATEGIES", "available_techniques",
    "run_strategy", "run_technique",
]


# ---------------------------------------------------------------------------
# Unified dispatch
# ---------------------------------------------------------------------------

REASONING_STRATEGIES = {
    "cot": chain_of_thought,
    "chain-of-thought": chain_of_thought,
    "plan-and-solve": plan_and_solve,
    "pas": plan_and_solve,
    "react": react,
    "self-ask": self_ask,
    "debate": debate,
    "reflexion": reflexion,
    "self-reflection": reflexion,
}

SEARCH_STRATEGIES = {
    "tot": tree_of_thoughts,
    "tree-of-thoughts": tree_of_thoughts,
    "got": graph_of_thoughts,
    "graph-of-thoughts": graph_of_thoughts,
    "mcts": mcts,
}


def available_techniques() -> dict[str, list[str]]:
    """Names accepted by :func:`run_technique`, grouped by family."""
    return {
        "reasoning": sorted(REASONING_STRATEGIES),
        "search": sorted(SEARCH_STRATEGIES),
    }


def run_strategy(name: str, model: Any, task: str, **kwargs: Any) -> StrategyResult:
    """Run a reasoning strategy by name; search names are rejected here."""
    key = (name or "").strip().lower()
    if key in SEARCH_STRATEGIES:
        raise ValueError(f"'{key}' is a search strategy - use run_search or run_technique")
    if key not in REASONING_STRATEGIES:
        raise ValueError(
            f"Unknown reasoning strategy '{name}'. "
            f"Choose one of: {', '.join(sorted(REASONING_STRATEGIES))}"
        )
    return REASONING_STRATEGIES[key](model, task, **kwargs)


def run_technique(name: str, model: Any, task: str, **kwargs: Any):
    """Dispatch to a reasoning or search technique by name.

    Returns whatever the underlying function returns (a ``StrategyResult`` or
    ``SearchResult``). Unknown keywords are dropped per-strategy so a single
    caller can pass a superset of options.
    """
    key = (name or "").strip().lower()
    if key in SEARCH_STRATEGIES:
        return run_search(key, model, task, **kwargs)
    if key in REASONING_STRATEGIES:
        fn = REASONING_STRATEGIES[key]
        accepted = fn.__code__.co_varnames[: fn.__code__.co_argcount]
        filtered = {k: v for k, v in kwargs.items() if k in accepted}
        return fn(model, task, **filtered)
    raise ValueError(
        f"Unknown technique '{name}'. Available: "
        f"{', '.join(sorted({**REASONING_STRATEGIES, **SEARCH_STRATEGIES}))}"
    )
