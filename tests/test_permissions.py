"""Unit tests for tool permission system (Phase 1 security fix)."""

import pytest
from pathlib import Path
from core.permissions import ToolPermissions, PermissionChecker


class TestToolPermissions:
    """Test ToolPermissions configuration and validation."""
    
    def test_default_restrict_to_project(self):
        """Test that restrict_to_project defaults to True."""
        perms = ToolPermissions()
        assert perms.restrict_to_project is True
        assert len(perms.allowed_paths) > 0
    
    def test_path_allowed_within_project(self, tmp_path):
        """Test that paths within project root are allowed."""
        perms = ToolPermissions(allowed_paths={tmp_path}, restrict_to_project=True)
        test_file = tmp_path / "test.txt"
        assert perms.is_path_allowed(test_file) is True
    
    def test_path_denied_outside_project(self, tmp_path):
        """Test that paths outside project root are denied."""
        perms = ToolPermissions(allowed_paths={tmp_path}, restrict_to_project=True)
        outside_path = Path("/etc/passwd")
        assert perms.is_path_allowed(outside_path) is False
    
    def test_path_allowed_when_restriction_disabled(self):
        """Test that all paths are allowed when restriction is disabled."""
        perms = ToolPermissions(restrict_to_project=False)
        assert perms.is_path_allowed(Path("/etc/passwd")) is True
    
    def test_command_whitelist_allows_matching(self):
        """Test that whitelisted commands are allowed."""
        perms = ToolPermissions(allowed_commands={"git", "npm", "python"})
        assert perms.is_command_allowed("git status") is True
        assert perms.is_command_allowed("npm install") is True
        assert perms.is_command_allowed("python script.py") is True
    
    def test_command_denied_not_in_whitelist(self):
        """Test that non-whitelisted commands are denied when whitelist is set."""
        perms = ToolPermissions(allowed_commands={"git", "npm"})
        assert perms.is_command_allowed("rm -rf /") is False
    
    def test_command_allowed_when_whitelist_empty(self):
        """Test that all commands are allowed when whitelist is empty."""
        perms = ToolPermissions(allowed_commands=set())
        assert perms.is_command_allowed("any command") is True
    
    def test_malicious_command_detection(self):
        """Test that malicious command patterns are detected."""
        perms = ToolPermissions()
        assert perms.is_command_allowed("rm -rf /") is False
        assert perms.is_command_allowed(":(){:|:&};:") is False
        assert perms.is_command_allowed("dd if=/dev/zero of=/dev/sda") is False
    
    def test_denied_pattern_blocks_command(self):
        """Test that denied patterns block commands."""
        perms = ToolPermissions(denied_patterns={r"sudo"})
        assert perms.is_command_allowed("sudo apt-get install") is False
    
    def test_file_size_allowed_within_limit(self):
        """Test that file sizes within limit are allowed."""
        perms = ToolPermissions(max_file_size=1024)
        assert perms.is_file_size_allowed(512) is True
        assert perms.is_file_size_allowed(1024) is True
    
    def test_file_size_denied_over_limit(self):
        """Test that file sizes over limit are denied."""
        perms = ToolPermissions(max_file_size=1024)
        assert perms.is_file_size_allowed(2048) is False


class TestPermissionChecker:
    """Test PermissionChecker tool call validation."""
    
    def test_allows_safe_write_file(self, tmp_path):
        """Test that safe write_file calls are allowed."""
        perms = ToolPermissions(allowed_paths={tmp_path}, restrict_to_project=True)
        checker = PermissionChecker(perms)
        
        is_allowed, reason = checker.check_tool_call(
            "write_file",
            {"path": str(tmp_path / "test.txt"), "content": "hello"},
        )
        assert is_allowed is True
        assert reason is None
    
    def test_denies_write_file_outside_project(self):
        """Test that write_file outside project is denied."""
        perms = ToolPermissions(allowed_paths={Path.cwd()}, restrict_to_project=True)
        checker = PermissionChecker(perms)
        
        is_allowed, reason = checker.check_tool_call(
            "write_file",
            {"path": "/etc/passwd", "content": "malicious"},
        )
        assert is_allowed is False
        assert "outside allowed project directory" in reason
    
    def test_denies_malicious_run_command(self):
        """Test that malicious run_command is denied."""
        perms = ToolPermissions()
        checker = PermissionChecker(perms)
        
        is_allowed, reason = checker.check_tool_call(
            "run_command",
            {"command": "rm -rf /"},
        )
        assert is_allowed is False
        assert "not allowed or appears malicious" in reason
    
    def test_denies_file_over_size_limit(self):
        """Test that write_file over size limit is denied."""
        perms = ToolPermissions(max_file_size=100)
        checker = PermissionChecker(perms)
        
        large_content = "x" * 200
        is_allowed, reason = checker.check_tool_call(
            "write_file",
            {"path": "test.txt", "content": large_content},
        )
        assert is_allowed is False
        assert "exceeds limit" in reason
    
    def test_allows_read_file_within_project(self, tmp_path):
        """Test that read_file within project is allowed."""
        perms = ToolPermissions(allowed_paths={tmp_path}, restrict_to_project=True)
        checker = PermissionChecker(perms)
        
        is_allowed, reason = checker.check_tool_call(
            "read_file",
            {"path": str(tmp_path / "test.txt")},
        )
        assert is_allowed is True
        assert reason is None
    
    def test_stats_tracking(self):
        """Test that permission checker tracks statistics."""
        perms = ToolPermissions(max_file_size=100)
        checker = PermissionChecker(perms)
        
        # Allow some calls
        checker.check_tool_call("write_file", {"path": "test.txt", "content": "small"})
        checker.check_tool_call("read_file", {"path": "test.txt"})
        
        # Deny some calls
        checker.check_tool_call("write_file", {"path": "test.txt", "content": "x" * 200})
        checker.check_tool_call("run_command", {"command": "rm -rf /"})
        
        stats = checker.get_stats()
        assert stats["allowed"] == 2
        assert stats["denied"] == 2
        assert stats["total"] == 4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
