"""Custom exception hierarchy for the Autononymous agent.

Provides structured error types for different failure modes:
- AgentError: Base exception for all agent-related errors
- ToolExecutionError: Errors during tool execution
- ModelInvocationError: Errors during model API calls
- ConfigurationError: Invalid configuration
"""

from __future__ import annotations


class AgentError(Exception):
    """Base exception for agent errors.

    All custom exceptions in the agent should inherit from this
    to allow for consistent error handling across the system.
    """

    pass


class ToolExecutionError(AgentError):
    """Raised when a tool fails to execute.

    Attributes:
        tool_name: Name of the tool that failed
        message: Detailed error message
    """

    def __init__(self, tool_name: str, message: str):
        self.tool_name = tool_name
        self.message = message
        super().__init__(f"Tool '{tool_name}' failed: {message}")


class ModelInvocationError(AgentError):
    """Raised when model call fails.

    This covers API errors, timeout issues, invalid responses,
    and any other failures in the model client layer.
    """

    pass


class ConfigurationError(AgentError):
    """Raised when configuration is invalid.

    Covers missing required settings, invalid values,
    or configuration parsing errors.
    """

    pass
