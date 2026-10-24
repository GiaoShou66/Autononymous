"""Tool dispatcher for routing and executing tool calls.

This module provides a centralized tool execution system with:
- Middleware chain for pre/post-processing
- Standardized error handling
- Execution time tracking
- Type-safe tool context and results
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, Any, Optional, List

from core.exceptions import ToolExecutionError
from core.logger import setup_logger

# Configure logger
logger = setup_logger(__name__)


@dataclass
class ToolContext:
    """Context for a tool invocation.

    Encapsulates all information needed to execute a tool call,
    including the tool name, arguments, and conversation context.

    Attributes:
        tool_name: Name of the tool to execute
        arguments: Keyword arguments to pass to the tool
        user_message: The original user message that triggered this tool call
        conversation_history: Full conversation history for context
        metadata: Additional metadata (optional)
    """

    tool_name: str
    arguments: Dict[str, Any]
    user_message: str
    conversation_history: List[Dict[str, Any]]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolResult:
    """Result of a tool execution.

    Standardized return value for all tool executions, providing
    consistent success/failure handling and metadata.

    Attributes:
        success: Whether the tool executed successfully
        output: String output from the tool (if successful)
        error: Error message (if failed)
        metadata: Additional execution metadata
        execution_time_ms: Time taken to execute in milliseconds
    """

    success: bool
    output: str
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    execution_time_ms: float = 0.0


class ToolDispatcher:
    """
    Routes and executes tool calls with middleware support.

    Single responsibility: tool scheduling and execution.

    The dispatcher maintains a registry of available tools and executes
    them through a middleware chain, allowing for cross-cutting concerns
    like logging, validation, and monitoring.

    Example:
        >>> dispatcher = ToolDispatcher({"echo": lambda x: x})
        >>> ctx = ToolContext("echo", {"x": "hello"}, "test", [])
        >>> result = dispatcher.dispatch(ctx)
        >>> assert result.success
        >>> assert result.output == "hello"
    """

    def __init__(self, tools: Dict[str, Callable]):
        """Initialize the dispatcher with a tool registry.

        Args:
            tools: Dictionary mapping tool names to callable functions
        """
        self.tools = tools
        self.middleware: List[Callable] = []
        logger.info(f"ToolDispatcher initialized with {len(tools)} tools")

    def add_middleware(self, middleware: Callable) -> None:
        """
        Add middleware to the execution chain.

        Middleware functions follow the pattern:
            (ctx: ToolContext, next: Callable) -> ToolResult

        They can modify the context, short-circuit execution,
        or wrap the next middleware in the chain.

        Args:
            middleware: Middleware function to add
        """
        self.middleware.append(middleware)
        logger.debug(f"Added middleware, total: {len(self.middleware)}")

    def dispatch(self, ctx: ToolContext) -> ToolResult:
        """
        Execute tool call through middleware chain.

        This is the main entry point for tool execution. It:
        1. Records start time
        2. Executes through middleware chain
        3. Records execution time
        4. Logs the result

        Args:
            ctx: Tool context with tool name and arguments

        Returns:
            ToolResult with success status, output, and metadata
        """
        logger.info(f"Dispatching tool: {ctx.tool_name}")
        start = time.time()

        def execute_next(idx: int, ctx: ToolContext) -> ToolResult:
            """Execute the next middleware in the chain."""
            if idx >= len(self.middleware):
                # End of chain: execute actual tool
                return self._execute_tool(ctx)
            # Execute current middleware, passing next as continuation
            return self.middleware[idx](ctx, lambda c: execute_next(idx + 1, c))

        try:
            result = execute_next(0, ctx)
            result.execution_time_ms = (time.time() - start) * 1000

            if result.success:
                logger.info(
                    f"Tool '{ctx.tool_name}' succeeded in {result.execution_time_ms:.2f}ms"
                )
            else:
                logger.error(f"Tool '{ctx.tool_name}' failed: {result.error}")

            return result
        except Exception as e:
            # Catch any unexpected errors in the middleware chain
            execution_time_ms = (time.time() - start) * 1000
            logger.exception(f"Unexpected error dispatching tool '{ctx.tool_name}'")
            return ToolResult(
                success=False,
                error=f"Dispatcher error: {str(e)}",
                output="",
                execution_time_ms=execution_time_ms,
            )

    def _execute_tool(self, ctx: ToolContext) -> ToolResult:
        """Execute the actual tool function.

        Args:
            ctx: Tool context

        Returns:
            ToolResult with execution outcome
        """
        if ctx.tool_name not in self.tools:
            error_msg = f"Unknown tool: {ctx.tool_name}"
            logger.error(error_msg)
            return ToolResult(success=False, error=error_msg, output="")

        try:
            tool_func = self.tools[ctx.tool_name]
            logger.debug(f"Executing tool '{ctx.tool_name}' with args: {ctx.arguments}")

            output = tool_func(**ctx.arguments)

            # Convert output to string if possible
            str_output = str(output) if output is not None else ""

            return ToolResult(
                success=True,
                output=str_output,
                metadata={"tool": ctx.tool_name},
            )
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Tool '{ctx.tool_name}' raised exception: {error_msg}")
            return ToolResult(success=False, error=error_msg, output="")

    def list_tools(self) -> List[str]:
        """Return list of registered tool names."""
        return list(self.tools.keys())

    def has_tool(self, tool_name: str) -> bool:
        """Check if a tool is registered."""
        return tool_name in self.tools
