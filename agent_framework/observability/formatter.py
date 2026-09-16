from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union

from agent_framework.observability.trace import EventType, TraceEvent, WorkflowTrace

# Renders a WorkflowTrace for a person: a header, the timeline, the tool
# statistics. Everything is built as a string first (format_run_summary) so a
# test can assert on it and a caller can write it somewhere other than stdout.

INDENT = "  "
WIDTH = 120

# Column widths for the timeline table. Summary takes whatever is left.
_COLUMNS = (("Time", 9), ("Step Type", 14), ("Entity", 18), ("Duration", 8), ("Status", 10))
_SUMMARY_WIDTH = WIDTH - sum(width + 3 for _, width in _COLUMNS)


def _one_line(text: Any, limit: int) -> str:
    """Whitespace collapsed, then cut to `limit` characters."""
    flat = " ".join(("" if text is None else str(text)).split())
    if not flat:
        return "-"
    return flat if len(flat) <= limit else flat[: max(limit - 3, 0)] + "..."


def _fit(text: Any, width: int) -> str:
    return _one_line(text, width).ljust(width)


def _clock(timestamp: Optional[float]) -> str:
    if timestamp is None:
        return "-"
    try:
        return datetime.fromtimestamp(timestamp).strftime("%H:%M:%S.%f")[:-3]
    except (OverflowError, OSError, ValueError, TypeError):
        return "?"


def _seconds(value: Optional[float]) -> str:
    if value is None:
        return "-"
    if value < 0.001:
        return "<1ms"
    if value < 1:
        return f"{value * 1000:.0f}ms"
    return f"{value:.2f}s"


def _status_label(event: TraceEvent) -> str:
    if event.event_type is EventType.VERIFICATION:
        verdict = event.data.get("is_complete")
        return "-" if verdict is None else ("PASS" if verdict else "INCOMPLETE")
    status = event.status
    if status == "unauthorized":
        return "BLOCKED"
    return status.upper() if status != "-" else "-"


def _describe(event: TraceEvent, limit: int) -> str:
    """The one-line summary column, phrased per event kind."""
    data = event.data
    kind = event.event_type

    if kind is EventType.AGENT_CALL:
        if data.get("error"):
            return _one_line(f"error: {data['error']}", limit)
        return _one_line(f"{data.get('task') or ''} -> {data.get('output') or ''}", limit)

    if kind is EventType.TOOL_CALL:
        arguments = data.get("arguments") or {}
        call = f"{data.get('tool')}({', '.join(f'{k}={v!r}' for k, v in arguments.items())})"
        if data.get("error"):
            return _one_line(f"{call} : {data['error']}", limit)
        return _one_line(f"{call} -> {data.get('output')!r}", limit)

    if kind is EventType.ROUTE_DECISION:
        reason = f" ({data['reasoning']})" if data.get("reasoning") else ""
        return _one_line(f"-> {data.get('route')}{reason}", limit)

    if kind is EventType.DELEGATION:
        reason = f": {data['reason']}" if data.get("reason") else ""
        return _one_line(f"{data.get('from')} -> {data.get('to')}{reason}", limit)

    if kind is EventType.VERIFICATION:
        attempt = f"attempt {data['attempt']}: " if data.get("attempt") is not None else ""
        return _one_line(f"{attempt}{data.get('feedback') or ''}", limit)

    if kind is EventType.STATE_SNAPSHOT:
        keys = data.get("artifact_keys") or []
        label = f"{data['label']}: " if data.get("label") else ""
        return _one_line(f"{label}{len(keys)} artifact(s) {keys}", limit)

    return _one_line(data, limit)


def _entity(event: TraceEvent) -> str:
    return event.agent_name or event.workflow or "-"


# --- Sections ---


def _header(trace: WorkflowTrace, summary: dict[str, Any]) -> list[str]:
    lines = ["=" * WIDTH, f" RUN SUMMARY  {trace.run_id}", "=" * WIDTH]
    started = datetime.fromtimestamp(trace.start_time).strftime("%Y-%m-%d %H:%M:%S")
    lines.append(f"{INDENT}status:     {summary['status'].upper()}")
    lines.append(f"{INDENT}started:    {started}")
    lines.append(
        f"{INDENT}wall time:  {_seconds(summary['wall_time'])}"
        + ("" if trace.end_time is not None else "  (still running)")
    )
    counts = summary["counts"]
    lines.append(
        f"{INDENT}events:     {summary['event_count']} - "
        + ", ".join(f"{kind}={count}" for kind, count in counts.items() if count)
    )
    if trace.metadata:
        rendered = ", ".join(f"{key}={value!r}" for key, value in trace.metadata.items())
        lines.append(f"{INDENT}metadata:   {_one_line(rendered, WIDTH - 14)}")
    return lines


def _timeline(trace: WorkflowTrace) -> list[str]:
    lines = ["", "-" * WIDTH, " TIMELINE", "-" * WIDTH]
    head = " | ".join(title.ljust(width) for title, width in _COLUMNS) + " | Summary"
    lines.append(head)
    lines.append("-+-".join("-" * width for _, width in _COLUMNS) + "-+-" + "-" * _SUMMARY_WIDTH)

    events = list(trace.events)
    if not events:
        lines.append(f"{INDENT}(no events recorded)")
        return lines

    for event in events:
        offset = event.timestamp - trace.start_time
        cells = (
            _fit(f"+{offset:.3f}s", _COLUMNS[0][1]),
            _fit(event.event_type.value, _COLUMNS[1][1]),
            _fit(_entity(event), _COLUMNS[2][1]),
            _fit(_seconds(event.duration), _COLUMNS[3][1]),
            _fit(_status_label(event), _COLUMNS[4][1]),
        )
        lines.append(" | ".join(cells) + " | " + _describe(event, _SUMMARY_WIDTH))
    return lines


def _tool_stats(summary: dict[str, Any]) -> list[str]:
    tools = summary["tools"]
    lines = ["", "-" * WIDTH, " TOOL EXECUTION", "-" * WIDTH]
    if not tools["total_calls"]:
        lines.append(f"{INDENT}(no tool calls)")
        return lines

    lines.append(
        f"{INDENT}total calls: {tools['total_calls']}   "
        f"allowed: {tools['allowed']}   blocked: {tools['blocked']}   "
        f"failed: {tools['failed']}"
    )
    lines.append("")
    lines.append(f"{INDENT}{'tool':<20} {'calls':>6} {'allowed':>8} {'blocked':>8} {'failed':>7}")
    for name, entry in tools["by_tool"].items():
        lines.append(
            f"{INDENT}{name:<20} {entry['calls']:>6} {entry['allowed']:>8} "
            f"{entry['blocked']:>8} {entry['failed']:>7}"
        )
    return lines


def _agent_stats(summary: dict[str, Any]) -> list[str]:
    agents = summary["agents"]
    if not agents:
        return []
    lines = ["", "-" * WIDTH, " AGENT CALLS", "-" * WIDTH]
    lines.append(f"{INDENT}{'agent':<24} {'calls':>6} {'failed':>7} {'total time':>11} {'mean':>8}")
    for name, entry in agents.items():
        mean = entry["total_duration"] / entry["calls"] if entry["calls"] else 0.0
        lines.append(
            f"{INDENT}{name:<24} {entry['calls']:>6} {entry['failed']:>7} "
            f"{_seconds(entry['total_duration']):>11} {_seconds(mean):>8}"
        )
    return lines


def _errors(summary: dict[str, Any]) -> list[str]:
    errors = summary["errors"]
    if not errors:
        return []
    lines = ["", "-" * WIDTH, f" ERRORS AND REFUSALS ({len(errors)})", "-" * WIDTH]
    for item in errors:
        lines.append(
            f"{INDENT}[{item['event_type']}] {item['entity']}: "
            f"{_one_line(item['error'], WIDTH - 30)}"
        )
    return lines


def _footer(json_path: Optional[Union[str, Path]]) -> list[str]:
    lines = ["", "-" * WIDTH]
    if json_path is not None:
        lines.append(f"{INDENT}JSON trace: {Path(json_path).resolve()}")
    else:
        lines.append(f"{INDENT}JSON trace: not exported (call trace.export_json(path))")
    lines.append("=" * WIDTH)
    return lines


# --- Public API ---


def format_run_summary(
    trace: WorkflowTrace,
    json_path: Optional[Union[str, Path]] = None,
) -> str:
    """The run summary as one string. `json_path` is shown as the export location."""
    summary = trace.summary()
    lines: list[str] = []
    lines += _header(trace, summary)
    lines += _timeline(trace)
    lines += _tool_stats(summary)
    lines += _agent_stats(summary)
    lines += _errors(summary)
    lines += _footer(json_path)
    return "\n".join(lines)


def print_run_summary(
    trace: WorkflowTrace,
    json_path: Optional[Union[str, Path]] = None,
    export: bool = True,
) -> Optional[Path]:
    """Print the run summary; export the trace to `json_path` first when asked.

    Returns the path the JSON was written to, or None. Accepts a Tracer as
    well as a WorkflowTrace, so `print_run_summary(tracer)` reads naturally.
    """
    if hasattr(trace, "trace") and not isinstance(trace, WorkflowTrace):
        trace = trace.trace  # a Tracer

    exported: Optional[Path] = None
    if json_path is not None and export:
        trace.export_json(json_path)
        exported = Path(json_path)

    print()
    print(format_run_summary(trace, json_path=json_path if json_path is not None else None))
    return exported
