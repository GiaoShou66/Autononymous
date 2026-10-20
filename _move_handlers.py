"""One-off surgery: relocate the autonomous command handlers.

Moves lines 2135-2196 (0-indexed slice [2135:2197]) from below the
custom-command catch-all to immediately above line 2120 (the
'# Custom slash commands' comment).
"""
from __future__ import annotations

path = "cli.py"
with open(path, encoding="utf-8") as fh:
    lines = fh.readlines()

AUTONOMOUS_START = 2135   # 0-indexed
AUTONOMOUS_END = 2197     # exclusive
INSERT_BEFORE = 2120      # 0-indexed

# Sanity-check the anchors
assert "Autonomous features" in lines[AUTONOMOUS_START], lines[AUTONOMOUS_START]
assert "Custom slash commands" in lines[INSERT_BEFORE], lines[INSERT_BEFORE]

block = lines[AUTONOMOUS_START:AUTONOMOUS_END]

# Lines 2132-2134 hold the "else:" + "message_to_send = _expand_file_mentions(...)"
# + a comment. Those must stay and run BEFORE the relocated block.
prelude = lines[AUTONOMOUS_START - 4:AUTONOMOUS_START]
assert prelude[0].strip() == "else:", prelude[0]
assert "_expand_file_mentions" in prelude[1], prelude[1]
assert preamble_check(prelude[2]), prelude[2] if False else True

remaining = lines[:INSERT_BEFORE] + lines[INSERT_BEFORE:AUTONOMOUS_START - 3] + lines[AUTONOMOUS_END:]
# Now re-insert: else-branch first, then the autonomous handlers.
new_lines = remaining[:INSERT_BEFORE] + prelude + block + remaining[INSERT_BEFORE:]

with open(path, "w", encoding="utf-8", newline="") as fh:
    fh.writelines(new_lines)

print(f"Moved {len(block)} lines; cli.py now has {len(new_lines)} lines")
