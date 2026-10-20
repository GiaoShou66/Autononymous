from __future__ import annotations

import difflib
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class DiffRow:
    line_no: Optional[int]
    kind: str  # "add" | "del" | "ctx" | "skip"
    text: str


@dataclass
class DiffResult:
    rows: List[DiffRow] = field(default_factory=list)
    added: int = 0
    removed: int = 0

    @property
    def summary(self) -> str:
        added_str = f"Added {self.added} line" if self.added == 1 else f"Added {self.added} lines"
        if self.added > 0 and self.removed > 0:
            removed_str = f"removed {self.removed} line" if self.removed == 1 else f"removed {self.removed} lines"
            return f"{added_str}, {removed_str}"
        if self.added > 0:
            return added_str
        if self.removed > 0:
            return f"Removed {self.removed} line" if self.removed == 1 else f"Removed {self.removed} lines"
        return "No lines changed"

    def to_dict(self) -> dict:
        return {
            "added": self.added,
            "removed": self.removed,
            "summary": self.summary,
            "rows": [{"line": r.line_no, "kind": r.kind, "text": r.text} for r in self.rows],
        }


def compute_diff(
    pre_text: str = "",
    post_text: str = "",
    old_snippet: Optional[str] = None,
    new_snippet: Optional[str] = None,
    old_start_line: int = 1,
    max_rows: int = 80,
) -> DiffResult:
    """
    Build a structured, line-numbered diff between two file states, same
    shape whether a whole file changed (pre_text/post_text) or a single
    apply_patch snippet changed within a larger file (old_snippet/new_snippet
    plus surrounding context from pre_text/post_text).
    """
    result = DiffResult()
    pre_lines = pre_text.splitlines()
    post_lines = post_text.splitlines()

    if old_snippet is not None and new_snippet is not None:
        if not pre_text and old_snippet:
            ctx_before_lines: List[str] = []
            ctx_before_start = old_start_line
        else:
            ctx_before_lines = pre_lines[max(0, old_start_line - 4):old_start_line - 1]
            ctx_before_start = old_start_line - len(ctx_before_lines)

        old_lines = old_snippet.splitlines()
        new_lines = new_snippet.splitlines()
        sm = difflib.SequenceMatcher(None, old_lines, new_lines)

        diff_rows: List[DiffRow] = []
        old_idx = old_start_line
        new_idx = old_start_line

        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == "equal":
                for line in new_lines[j1:j2]:
                    diff_rows.append(DiffRow(new_idx, "ctx", line))
                    old_idx += 1
                    new_idx += 1
            elif tag == "delete":
                for line in old_lines[i1:i2]:
                    diff_rows.append(DiffRow(old_idx, "del", line))
                    old_idx += 1
                    result.removed += 1
            elif tag == "insert":
                for line in new_lines[j1:j2]:
                    diff_rows.append(DiffRow(new_idx, "add", line))
                    new_idx += 1
                    result.added += 1
            elif tag == "replace":
                for line in old_lines[i1:i2]:
                    diff_rows.append(DiffRow(old_idx, "del", line))
                    old_idx += 1
                    result.removed += 1
                for line in new_lines[j1:j2]:
                    diff_rows.append(DiffRow(new_idx, "add", line))
                    new_idx += 1
                    result.added += 1

        ctx_after_start = new_idx
        ctx_after_lines = post_lines[ctx_after_start - 1:ctx_after_start + 2] if post_lines else []

        for i, line in enumerate(ctx_before_lines):
            result.rows.append(DiffRow(ctx_before_start + i, "ctx", line))
        result.rows.extend(diff_rows)
        for i, line in enumerate(ctx_after_lines):
            result.rows.append(DiffRow(ctx_after_start + i, "ctx", line))
        return result

    if not pre_text and post_text:
        # Brand new file.
        result.added = len(post_lines)
        if len(post_lines) <= 45:
            for idx, line in enumerate(post_lines, 1):
                result.rows.append(DiffRow(idx, "add", line))
        else:
            for idx in range(1, 31):
                result.rows.append(DiffRow(idx, "add", post_lines[idx - 1]))
            result.rows.append(DiffRow(None, "skip", f"... ({len(post_lines) - 35} more lines)"))
            for idx in range(len(post_lines) - 4, len(post_lines) + 1):
                result.rows.append(DiffRow(idx, "add", post_lines[idx - 1]))
        return result

    if not post_text and pre_text:
        # Whole file removed.
        result.removed = len(pre_lines)
        for idx, line in enumerate(pre_lines, 1):
            result.rows.append(DiffRow(idx, "del", line))
        return result

    # Full old-text vs new-text comparison.
    sm = difflib.SequenceMatcher(None, pre_lines, post_lines)
    old_idx = 1
    new_idx = 1
    hunks = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            old_idx += (i2 - i1)
            new_idx += (j2 - j1)
        else:
            hunk_del = [DiffRow(old_idx + k, "del", pre_lines[i1 + k]) for k in range(i2 - i1)]
            hunk_add = [DiffRow(new_idx + k, "add", post_lines[j1 + k]) for k in range(j2 - j1)]
            result.removed += len(hunk_del)
            result.added += len(hunk_add)

            b_start = max(0, i1 - 3)
            ctx_before = [
                DiffRow(new_idx - (i1 - b_start) + k, "ctx", pre_lines[b_start + k])
                for k in range(i1 - b_start)
            ]
            ctx_after = [
                DiffRow(new_idx + len(hunk_add) + k, "ctx", post_lines[j2 + k])
                for k in range(min(3, len(post_lines) - j2))
            ]
            hunks.append((ctx_before, hunk_del, hunk_add, ctx_after))
            old_idx += (i2 - i1)
            new_idx += (j2 - j1)

    for idx, (cb, hd, ha, ca) in enumerate(hunks):
        if idx > 0:
            result.rows.append(DiffRow(None, "skip", "..."))
        result.rows.extend(cb)
        result.rows.extend(hd)
        result.rows.extend(ha)
        result.rows.extend(ca)
        if len(result.rows) > max_rows:
            result.rows.append(DiffRow(None, "skip", "... (additional changes omitted)"))
            break

    return result
