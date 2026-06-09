from __future__ import annotations

from typing import Any


def format_pmu_perf_status_markdown_lines(summary: dict[str, Any]) -> list[str]:
    """Render compact PMU/perf availability and provenance lines for summaries."""
    if not summary:
        return []
    lines: list[str] = []
    status = _format_named_summary_values(
        summary,
        (
            "hardware_counters_available",
            "perf_hotspots_available",
            "perf_sample_count",
            "perf_hotspot_error",
            "perf_data_path",
            "perf_script_path",
        ),
    )
    provenance = _format_named_summary_values(
        dict(summary.get("perf_hotspot_provenance") or {}),
        (
            "capture_scope",
            "warmup_completed",
            "record_started_after_warmup",
            "attached_pids",
            "measured_query_repetitions",
            "measured_batch_size",
            "source_line_decode",
        ),
    )
    if status is not None:
        lines.append(f"- PMU/perf status: {status}")
    if provenance is not None:
        lines.append(f"- PMU/perf provenance: {provenance}")
    return lines


def _format_named_summary_values(
    summary: dict[str, Any],
    keys: tuple[str, ...],
) -> str | None:
    pairs = [
        f"{key}={_truncate_evidence_value(str(summary[key]))}"
        for key in keys
        if key in summary and summary[key] not in (None, "", [], {})
    ]
    if not pairs:
        return None
    return "; ".join(pairs)


def _truncate_evidence_value(value: str, limit: int = 240) -> str:
    if len(value) <= limit:
        return value
    return value[: limit - 3] + "..."
