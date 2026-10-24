"""Tool permission system for the Autononymous agent.

This module provides fine-grained permission control for tool execution,
including path restrictions, command whitelists, and resource limits.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Set, Optional, List
from core.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class ToolPermissions:
    """
    Permission configuration for tool execution.
    
    Attributes:
        allowed_paths: Set of paths where file operations are allowed
        allowed_commands: Set of command patterns that are whitelisted
        denied_patterns: Set of patterns that are explicitly denied
        max_file_size: Maximum file size in bytes for write operations
        restrict_to_project: Whether to restrict all operations to project root
    """
    allowed_paths: Set[Path] = field(default_factory=set)
    allowed_commands: Set[str] = field(default_factory=set)
    denied_patterns: Set[str] = field(default_factory=set)
    max_file_size: int = 10 * 1024 * 1024  # 10MB default
    restrict_to_project: bool = True
    
    # Malicious command patterns to detect
    MALICIOUS_PATTERNS: List[str] = field(
        default_factory=lambda: [
            r"rm\s+-rf\s+/",
            r":\(\)\{\:|\:&\}\;",
            r">\s*/dev/",
            r"mkfs\.",
            r"dd\s+if=",
            r"chmod\s+777",
            r"chown\s+-R",
        ]
    )
    
    def __post_init__(self) -> None:
        """Initialize allowed paths with project root if restrict_to_project is True."""
        if self.restrict_to_project and not self.allowed_paths:
            # Default to current working directory
            self.allowed_paths.add(Path.cwd())
    
    def is_path_allowed(self, path: Path) -> bool:
        """
        Check if a path is allowed for file operations.
        
        Args:
            path: Path to check
            
        Returns:
            True if path is allowed, False otherwise
        """
        if not self.restrict_to_project:
            return True
        
        path = path.resolve()
        for allowed in self.allowed_paths:
            allowed = allowed.resolve()
            try:
                path.relative_to(allowed)
                return True
            except ValueError:
                continue
        
        logger.warning(f"Path denied: {path} is outside allowed paths")
        return False
    
    def is_command_allowed(self, command: str) -> bool:
        """
        Check if a command is allowed for execution.
        
        Args:
            command: Command string to check
            
        Returns:
            True if command is allowed, False otherwise
        """
        # Check denied patterns first
        for pattern in self.denied_patterns:
            if re.search(pattern, command, re.IGNORECASE):
                logger.warning(f"Command denied by pattern: {pattern}")
                return False
        
        # Check malicious patterns
        for pattern in self.MALICIOUS_PATTERNS:
            if re.search(pattern, command, re.IGNORECASE):
                logger.warning(f"Malicious command detected: {command}")
                return False
        
        # If whitelist is empty, allow all (except denied patterns)
        if not self.allowed_commands:
            return True
        
        # Check whitelist
        for allowed in self.allowed_commands:
            if command.startswith(allowed):
                return True
        
        logger.warning(f"Command not in whitelist: {command}")
        return False
    
    def is_file_size_allowed(self, size: int) -> bool:
        """
        Check if a file size is within limits.
        
        Args:
            size: File size in bytes
            
        Returns:
            True if size is allowed, False otherwise
        """
        if size > self.max_file_size:
            logger.warning(f"File size {size} exceeds limit {self.max_file_size}")
            return False
        return True


class PermissionChecker:
    """
    Permission checker for tool execution.
    
    This class validates tool calls against permission rules before execution.
    """
    
    def __init__(self, permissions: Optional[ToolPermissions] = None):
        """
        Initialize permission checker.
        
        Args:
            permissions: Permission configuration. If None, uses default restrictive settings.
        """
        self.permissions = permissions or ToolPermissions()
        self.denied_count: int = 0
        self.allowed_count: int = 0
    
    def check_tool_call(
        self,
        tool_name: str,
        arguments: dict,
    ) -> tuple[bool, Optional[str]]:
        """
        Check if a tool call is allowed.
        
        Args:
            tool_name: Name of the tool being called
            arguments: Tool arguments
            
        Returns:
            Tuple of (is_allowed, denial_reason)
        """
        # Check file operations
        if tool_name in {"write_file", "read_file", "delete_file", "apply_patch"}:
            path = arguments.get("path", "")
            if path:
                if not self.permissions.is_path_allowed(Path(path)):
                    self.denied_count += 1
                    return False, f"Path '{path}' is outside allowed project directory"
        
        # Check command execution
        if tool_name == "run_command":
            command = arguments.get("command", "")
            if command:
                if not self.permissions.is_command_allowed(command):
                    self.denied_count += 1
                    return False, f"Command '{command}' is not allowed or appears malicious"
        
        # Check file size for write operations
        if tool_name in {"write_file", "apply_patch"}:
            content = arguments.get("content", "")
            if content:
                size = len(content.encode("utf-8"))
                if not self.permissions.is_file_size_allowed(size):
                    self.denied_count += 1
                    return False, f"Content size {size} bytes exceeds limit {self.permissions.max_file_size}"
        
        self.allowed_count += 1
        return True, None
    
    def get_stats(self) -> dict:
        """
        Get permission checker statistics.
        
        Returns:
            Dictionary with allowed_count and denied_count
        """
        return {
            "allowed": self.allowed_count,
            "denied": self.denied_count,
            "total": self.allowed_count + self.denied_count,
        }
