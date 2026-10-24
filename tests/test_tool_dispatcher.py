"""Unit tests for ToolDispatcher (Phase 1 refactoring)."""

import pytest
from core.tool_dispatcher import ToolDispatcher, ToolContext, ToolResult
from core.exceptions import ToolExecutionError


def test_dispatch_success():
    """Test successful tool dispatch."""
    dispatcher = ToolDispatcher({"echo": lambda x: x})
    ctx = ToolContext("echo", {"x": "hello"}, "test", [])
    result = dispatcher.dispatch(ctx)

    assert result.success
    assert result.output == "hello"
    assert result.error is None
    assert result.execution_time_ms >= 0


def test_dispatch_unknown_tool():
    """Test dispatch with unknown tool name."""
    dispatcher = ToolDispatcher({})
    ctx = ToolContext("unknown", {}, "test", [])
    result = dispatcher.dispatch(ctx)

    assert not result.success
    assert "Unknown tool" in result.error
    assert result.output == ""


def test_dispatch_with_exception():
    """Test dispatch when tool raises an exception."""
    def failing_tool(**kwargs):
        raise ValueError("Test error")

    dispatcher = ToolDispatcher({"fail": failing_tool})
    ctx = ToolContext("fail", {}, "test", [])
    result = dispatcher.dispatch(ctx)

    assert not result.success
    assert "Test error" in result.error
    assert result.output == ""


def test_middleware_execution():
    """Test that middleware is executed in the chain."""
    call_log = []

    def middleware1(ctx, next):
        call_log.append("middleware1")
        return next(ctx)

    def middleware2(ctx, next):
        call_log.append("middleware2")
        return next(ctx)

    dispatcher = ToolDispatcher({"echo": lambda x: x})
    dispatcher.add_middleware(middleware1)
    dispatcher.add_middleware(middleware2)

    ctx = ToolContext("echo", {"x": "test"}, "test", [])
    result = dispatcher.dispatch(ctx)

    assert result.success
    assert call_log == ["middleware1", "middleware2"]


def test_middleware_short_circuit():
    """Test that middleware can short-circuit execution."""
    def blocking_middleware(ctx, next):
        return ToolResult(success=False, error="Blocked by middleware", output="")

    dispatcher = ToolDispatcher({"should_not_run": lambda: "fail"})
    dispatcher.add_middleware(blocking_middleware)

    ctx = ToolContext("should_not_run", {}, "test", [])
    result = dispatcher.dispatch(ctx)

    assert not result.success
    assert result.error == "Blocked by middleware"


def test_list_tools():
    """Test listing registered tools."""
    dispatcher = ToolDispatcher({"tool1": lambda: 1, "tool2": lambda: 2})
    tools = dispatcher.list_tools()

    assert set(tools) == {"tool1", "tool2"}


def test_has_tool():
    """Test checking if a tool is registered."""
    dispatcher = ToolDispatcher({"tool1": lambda: 1})

    assert dispatcher.has_tool("tool1")
    assert not dispatcher.has_tool("tool2")


def test_execution_time_tracking():
    """Test that execution time is tracked."""
    import time

    def slow_tool(**kwargs):
        time.sleep(0.01)
        return "done"

    dispatcher = ToolDispatcher({"slow": slow_tool})
    ctx = ToolContext("slow", {}, "test", [])
    result = dispatcher.dispatch(ctx)

    assert result.success
    assert result.execution_time_ms >= 10  # At least 10ms


def test_tool_with_none_output():
    """Test tool that returns None."""
    dispatcher = ToolDispatcher({"none_tool": lambda: None})
    ctx = ToolContext("none_tool", {}, "test", [])
    result = dispatcher.dispatch(ctx)

    assert result.success
    assert result.output == ""


def test_tool_context_metadata():
    """Test that metadata is preserved in context."""
    def metadata_tool(**kwargs):
        return "ok"

    dispatcher = ToolDispatcher({"meta": metadata_tool})
    ctx = ToolContext(
        "meta",
        {},
        "test",
        [],
        metadata={"custom": "value"}
    )
    result = dispatcher.dispatch(ctx)

    assert result.success
    # Tool dispatcher adds tool name to metadata
    assert "tool" in result.metadata


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
