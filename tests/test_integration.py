"""Integration test to verify Agent still works with ToolDispatcher."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import Agent
from config import load_config
from core.tool_dispatcher import ToolDispatcher


def test_agent_initialization():
    """Test that Agent initializes with ToolDispatcher."""
    config = load_config()
    # Create a minimal Agent for testing
    from models import ModelClient
    from tools import ProjectTools
    from pathlib import Path

    # This is a minimal test - we'll just check the dispatcher exists
    # In a real test, we'd need a proper model client
    try:
        agent = Agent(
            model=None,  # Would need real model for full test
            tools=ProjectTools(project_root=Path.cwd()),
        )
        assert agent.tool_dispatcher is not None
        assert isinstance(agent.tool_dispatcher, ToolDispatcher)
        print("✓ Agent initializes with ToolDispatcher")
    except Exception as e:
        print(f"✗ Agent initialization failed: {e}")
        raise


def test_tool_dispatcher_integration():
    """Test that the dispatcher has the expected tools."""
    from core.tools_registry import create_tool_registry

    # Mock agent with minimal methods
    class MockAgent:
        def _execute_tool_inner(self, name, kwargs):
            return f"Mocked: {name}"

    mock_agent = MockAgent()
    registry = create_tool_registry(mock_agent)

    # Check that key tools are registered
    expected_tools = {
        "read_file",
        "write_file",
        "run_command",
        "search_codebase",
        "search_web",
    }

    registered_tools = set(registry.keys())
    assert expected_tools.issubset(registered_tools), f"Missing tools: {expected_tools - registered_tools}"
    print(f"✓ Tool registry has {len(registered_tools)} tools")


if __name__ == "__main__":
    print("Running integration tests...")
    test_tool_dispatcher_integration()
    print("\nAll integration tests passed!")
