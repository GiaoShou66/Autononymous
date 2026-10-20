"""Small dependency-free data profiler for CSV, TSV, and JSON datasets."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from statistics import mean, median
from typing import Any
from xml.sax.saxutils import escape


MAX_ROWS = 50_000


def _read_rows(source: Path) -> list[dict[str, Any]]:
    suffix = source.suffix.lower()
    if suffix in {".csv", ".tsv"}:
        with source.open("r", encoding="utf-8-sig", newline="") as handle:
            sample = handle.read(8192)
            handle.seek(0)
            delimiter = "\t" if suffix == ".tsv" else ","
            try:
                delimiter = csv.Sniffer().sniff(sample, delimiters=",\t;|").delimiter
            except csv.Error:
                pass
            reader = csv.DictReader(handle, delimiter=delimiter)
            return [dict(row) for _, row in zip(range(MAX_ROWS + 1), reader)]
    if suffix == ".json":
        value = json.loads(source.read_text(encoding="utf-8-sig"))
        if isinstance(value, list):
            if value and not all(isinstance(row, dict) for row in value):
                return [{"value": item} for item in value[: MAX_ROWS + 1]]
            return value[: MAX_ROWS + 1]
        if isinstance(value, dict):
            # A dict of equally sized arrays becomes a conventional table.
            arrays = {key: item for key, item in value.items() if isinstance(item, list)}
            if arrays and len(arrays) == len(value):
                keys = list(arrays)
                return [
                    {key: arrays[key][index] if index < len(arrays[key]) else None for key in keys}
                    for index in range(min(max(map(len, arrays.values()), default=0), MAX_ROWS + 1))
                ]
            return [value]
        raise ValueError("JSON must contain an object or an array.")
    raise ValueError("Supported dataset formats: .csv, .tsv, and .json.")


def _numeric(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = float(str(value).replace(",", "").strip())
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _write_svg_chart(values: list[float], column: str, target: Path) -> None:
    width, height, margin = 800, 420, 56
    low, high = min(values), max(values)
    span = high - low or 1.0
    points = []
    for index, value in enumerate(values):
        x = margin + index * (width - 2 * margin) / max(1, len(values) - 1)
        y = height - margin - (value - low) * (height - 2 * margin) / span
        points.append(f"{x:.1f},{y:.1f}")
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">'
        '<rect width="100%" height="100%" fill="#fff"/>'
        f'<text x="{margin}" y="32" font-family="sans-serif" font-size="20">'
        f'{escape(column)} — values by row</text>'
        f'<path d="M {margin} {height-margin} H {width-margin} M {margin} {margin} V {height-margin}" '
        'stroke="#555" fill="none"/>'
        f'<polyline points="{" ".join(points)}" fill="none" stroke="#356ae6" stroke-width="2"/>'
        f'<text x="8" y="{margin+4}" font-family="sans-serif" font-size="12">{high:g}</text>'
        f'<text x="8" y="{height-margin}" font-family="sans-serif" font-size="12">{low:g}</text>'
        '</svg>'
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(svg, encoding="utf-8")


def analyze_dataset(
    path: str | Path,
    chart_path: str | Path | None = None,
    max_preview_rows: int = 8,
) -> str:
    """Summarize shape, missingness, numeric columns, and sample rows."""
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        return f"[Data analysis] File not found: {source}"
    try:
        rows = _read_rows(source)
        if not rows:
            return f"Dataset: {source.name}\nRows: 0\nColumns: 0"
        truncated = len(rows) > MAX_ROWS
        rows = rows[:MAX_ROWS]
        columns = list(dict.fromkeys(key for row in rows for key in row))
        output = [
            f"Dataset: {source}",
            f"Rows analyzed: {len(rows)}" + (f" (capped at {MAX_ROWS})" if truncated else ""),
            f"Columns ({len(columns)}): {', '.join(str(column) for column in columns)}",
            "Column profile:",
        ]
        first_numeric: tuple[str, list[float]] | None = None
        for column in columns:
            values = [row.get(column) for row in rows]
            nonempty = [value for value in values if value not in (None, "")]
            nums = [number for value in nonempty if (number := _numeric(value)) is not None]
            output.append(
                f"- {column}: non-empty {len(nonempty)}/{len(rows)}, "
                f"missing {len(rows)-len(nonempty)}"
            )
            if nums and len(nums) == len(nonempty):
                output[-1] += (
                    f", numeric min {min(nums):g}, max {max(nums):g}, "
                    f"mean {mean(nums):g}, median {median(nums):g}"
                )
                if first_numeric is None:
                    first_numeric = (str(column), nums)
            elif nonempty:
                examples = list(dict.fromkeys(str(value) for value in nonempty))[:4]
                output[-1] += f", examples: {', '.join(examples)}"
        output.append("Sample rows:")
        output.extend(json.dumps(row, ensure_ascii=False, default=str) for row in rows[:max_preview_rows])
        if chart_path and first_numeric and len(first_numeric[1]) > 1:
            target = Path(chart_path).expanduser().resolve()
            if target.suffix.lower() != ".svg":
                target = target.with_suffix(".svg")
            _write_svg_chart(first_numeric[1], first_numeric[0], target)
            output.append(f"Chart: {target}")
        elif chart_path:
            output.append("Chart skipped: no numeric column with at least two values.")
        return "\n".join(output)
    except (OSError, UnicodeError, csv.Error, json.JSONDecodeError, ValueError) as exc:
        return f"[Data analysis error] {exc}"
