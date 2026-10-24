"""Tool registry for the ToolDispatcher.

This module provides a registry of tool functions that can be
registered with the ToolDispatcher. The tools are wrappers around
the existing Agent methods to maintain backward compatibility.

In future phases, the actual tool implementations will be moved here
as well, but for Phase 1 we only wrap the existing methods.
"""

from __future__ import annotations

from typing import Callable, Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from agent import Agent


def create_tool_registry(agent: Any) -> Dict[str, Callable]:
    """
    Create a tool registry from an Agent instance.

    This extracts tool methods from the Agent and registers them
    with the ToolDispatcher.

    Args:
        agent: Agent instance with tool methods

    Returns:
        Dictionary mapping tool names to callable functions
    """
    tools: Dict[str, Callable] = {}

    # File operations
    tools["read_file"] = lambda **kwargs: agent._execute_tool_inner("read_file", kwargs)
    tools["write_file"] = lambda **kwargs: agent._execute_tool_inner("write_file", kwargs)
    tools["apply_patch"] = lambda **kwargs: agent._execute_tool_inner("apply_patch", kwargs)
    tools["delete_file"] = lambda **kwargs: agent._execute_tool_inner("delete_file", kwargs)
    tools["delete_code"] = lambda **kwargs: agent._execute_tool_inner("delete_code", kwargs)
    tools["add_file"] = lambda **kwargs: agent._execute_tool_inner("add_file", kwargs)
    tools["create_directory"] = lambda **kwargs: agent._execute_tool_inner("create_directory", kwargs)

    # Search and navigation
    tools["list_files"] = lambda **kwargs: agent._execute_tool_inner("list_files", kwargs)
    tools["read_lines"] = lambda **kwargs: agent._execute_tool_inner("read_lines", kwargs)
    tools["grep_search"] = lambda **kwargs: agent._execute_tool_inner("grep_search", kwargs)
    tools["get_outline"] = lambda **kwargs: agent._execute_tool_inner("get_outline", kwargs)
    tools["search_codebase"] = lambda **kwargs: agent._execute_tool_inner("search_codebase", kwargs)
    tools["search_web"] = lambda **kwargs: agent._execute_tool_inner("search_web", kwargs)

    # Network and external
    tools["fetch_url"] = lambda **kwargs: agent._execute_tool_inner("fetch_url", kwargs)
    tools["http_request"] = lambda **kwargs: agent._execute_tool_inner("http_request", kwargs)

    # Command execution
    tools["run_command"] = lambda **kwargs: agent._execute_tool_inner("run_command", kwargs)

    # Media analysis
    tools["analyze_media"] = lambda **kwargs: agent._execute_tool_inner("analyze_media", kwargs)
    tools["describe_image"] = lambda **kwargs: agent._execute_tool_inner("describe_image", kwargs)
    tools["analyze_video"] = lambda **kwargs: agent._execute_tool_inner("analyze_video", kwargs)
    tools["analyze_data"] = lambda **kwargs: agent._execute_tool_inner("analyze_data", kwargs)

    # Git operations
    tools["git_status"] = lambda **kwargs: agent._execute_tool_inner("git_status", kwargs)
    tools["git_diff"] = lambda **kwargs: agent._execute_tool_inner("git_diff", kwargs)
    tools["git_commit"] = lambda **kwargs: agent._execute_tool_inner("git_commit", kwargs)

    # Validation
    tools["validate_syntax"] = lambda **kwargs: agent._execute_tool_inner("validate_syntax", kwargs)

    # Memory and tasks
    tools["remember"] = lambda **kwargs: agent._execute_tool_inner("remember", kwargs)
    tools["add_task"] = lambda **kwargs: agent._execute_tool_inner("add_task", kwargs)
    tools["update_task"] = lambda **kwargs: agent._execute_tool_inner("update_task", kwargs)
    tools["list_tasks"] = lambda **kwargs: agent._execute_tool_inner("list_tasks", kwargs)

    # Agent operations
    tools["spawn_agent"] = lambda **kwargs: agent._execute_tool_inner("spawn_agent", kwargs)
    tools["ask_user_question"] = lambda **kwargs: agent._execute_tool_inner("ask_user_question", kwargs)

    # Change management
    tools["undo_change"] = lambda **kwargs: agent._execute_tool_inner("undo_change", kwargs)

    # Skill operations
    tools["list_skill_files"] = lambda **kwargs: agent._execute_tool_inner("list_skill_files", kwargs)
    tools["read_skill_file"] = lambda **kwargs: agent._execute_tool_inner("read_skill_file", kwargs)

    # Agentic techniques (reasoning, code, research, memory, orchestration, training)
    tools["think"] = lambda **kwargs: agent._execute_tool_inner("think", kwargs)
    tools["run_python"] = lambda **kwargs: agent._execute_tool_inner("run_python", kwargs)
    tools["web_research"] = lambda **kwargs: agent._execute_tool_inner("web_research", kwargs)
    tools["recall"] = lambda **kwargs: agent._execute_tool_inner("recall", kwargs)
    tools["run_sop"] = lambda **kwargs: agent._execute_tool_inner("run_sop", kwargs)
    tools["orchestrate"] = lambda **kwargs: agent._execute_tool_inner("orchestrate", kwargs)
    tools["computer_use"] = lambda **kwargs: agent._execute_tool_inner("computer_use", kwargs)
    tools["graph_recall"] = lambda **kwargs: agent._execute_tool_inner("graph_recall", kwargs)
    tools["self_improve"] = lambda **kwargs: agent._execute_tool_inner("self_improve", kwargs)
    tools["export_training_data"] = lambda **kwargs: agent._execute_tool_inner(
        "export_training_data", kwargs
    )

    return tools
