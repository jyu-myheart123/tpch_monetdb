from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


TRACE_SUMMARY_MAX_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True)
class TraceSample:
    profile_ns_by_name: dict[str, tuple[int, ...]] = field(default_factory=dict)
    count_by_name: dict[str, tuple[int, ...]] = field(default_factory=dict)


@dataclass(frozen=True)
class TraceHotspotSummary:
    query_id: str
    issue_class: str
    evidence_sufficient: bool
    top_profiles: tuple[tuple[str, int], ...]
    counters: dict[str, int]
    summary_text: str
    sampled_instantiations: tuple[str, ...] = ()
    sampled_count: int = 0
    omitted_count: int = 0
    vectorization_candidate: bool = False
    hardware_counter_summary: dict[str, Any] = field(default_factory=dict)
    compiler_vectorization_summary: dict[str, Any] = field(default_factory=dict)
    change_scope: str | None = None


def parse_trace_lines(lines: Iterable[str]) -> TraceSample:
    """Parse TRACE lines without requiring the whole file in memory."""
    profiles: dict[str, list[int]] = {}
    counts: dict[str, list[int]] = {}
    for line in lines:
        parts = line.strip().split()
        if len(parts) != 3:
            continue
        kind, name, raw_value = parts
        try:
            value = int(raw_value)
        except ValueError:
            continue
        if kind == "PROFILE":
            profiles.setdefault(name, []).append(value)
        elif kind == "COUNT":
            counts.setdefault(name, []).append(value)
    return TraceSample(
        profile_ns_by_name={key: tuple(values) for key, values in profiles.items()},
        count_by_name={key: tuple(values) for key, values in counts.items()},
    )


def parse_trace_log(text: str) -> TraceSample:
    """Parse TRACE text held in memory for tests and small inline samples."""
    return parse_trace_lines(text.splitlines())


def summarize_trace_sample(
    *,
    query_id: str,
    sample: TraceSample,
    instantiation_id: str | None = None,
    args_string: str | None = None,
    hardware_counter_summary: dict[str, Any] | None = None,
    compiler_vectorization_summary: dict[str, Any] | None = None,
) -> TraceHotspotSummary:
    """Build one query hotspot summary from TRACE scopes, counters, and perf evidence."""
    profile_totals = {
        name: sum(values)
        for name, values in sample.profile_ns_by_name.items()
    }
    top_profiles = tuple(
        sorted(profile_totals.items(), key=lambda item: item[1], reverse=True)[:8]
    )
    counters = {
        name: values[-1]
        for name, values in sample.count_by_name.items()
        if values
    }
    resolved_hardware_summary = (
        {} if hardware_counter_summary is None else dict(hardware_counter_summary)
    )
    resolved_vectorization_summary = (
        {}
        if compiler_vectorization_summary is None
        else dict(compiler_vectorization_summary)
    )
    derived_metrics = resolved_hardware_summary.get("derived_metrics", {})
    merged_counters = dict(counters)
    for key, value in derived_metrics.items():
        if isinstance(value, (int, float)):
            merged_counters[key] = value
    issue_class = classify_trace_issue(top_profiles, merged_counters)
    evidence_sufficient = len(top_profiles) >= 5 and issue_class != "evidence_insufficient"
    vectorization_candidate = bool(
        resolved_vectorization_summary.get("missed_loops")
        or resolved_vectorization_summary.get("vectorization_candidate")
    )
    return TraceHotspotSummary(
        query_id=query_id,
        issue_class=issue_class,
        evidence_sufficient=evidence_sufficient,
        top_profiles=top_profiles,
        counters=merged_counters,
        summary_text=render_trace_summary_text(
            query_id,
            top_profiles,
            merged_counters,
            issue_class,
            vectorization_candidate=vectorization_candidate,
            hardware_counter_summary=resolved_hardware_summary,
            change_scope=recommend_change_scope(query_id, issue_class),
        ),
        sampled_instantiations=() if instantiation_id is None else (instantiation_id,),
        sampled_count=0 if instantiation_id is None else 1,
        omitted_count=0,
        vectorization_candidate=vectorization_candidate,
        hardware_counter_summary=resolved_hardware_summary,
        compiler_vectorization_summary=resolved_vectorization_summary,
        change_scope=recommend_change_scope(query_id, issue_class),
    )


def summarize_trace_file(
    *,
    query_id: str,
    trace_path: Path,
    instantiation_id: str | None = None,
    args_string: str | None = None,
    hardware_counter_summary: dict[str, Any] | None = None,
    compiler_vectorization_summary: dict[str, Any] | None = None,
) -> TraceHotspotSummary:
    """Read tracing_output.log and summarize it for one query instantiation."""
    if not trace_path.exists():
        return TraceHotspotSummary(
            query_id=query_id,
            issue_class="evidence_insufficient",
            evidence_sufficient=False,
            top_profiles=(),
            counters={},
            summary_text=f"Query {query_id}: tracing_output.log missing.",
            sampled_instantiations=() if instantiation_id is None else (instantiation_id,),
            sampled_count=0,
            omitted_count=0,
        )
    trace_size = trace_path.stat().st_size
    if trace_size > TRACE_SUMMARY_MAX_BYTES:
        return TraceHotspotSummary(
            query_id=query_id,
            issue_class="evidence_insufficient",
            evidence_sufficient=False,
            top_profiles=(),
            counters={},
            summary_text=(
                f"Query {query_id}: tracing_output.log is {trace_size} bytes, "
                f"exceeding the safe summary limit of {TRACE_SUMMARY_MAX_BYTES} bytes. "
                "Generate a bounded query-only trace before hotspot diagnosis."
            ),
            sampled_instantiations=() if instantiation_id is None else (instantiation_id,),
            sampled_count=0,
            omitted_count=0,
            hardware_counter_summary={},
            compiler_vectorization_summary={},
        )
    with trace_path.open("r", encoding="utf-8", errors="replace") as handle:
        sample = parse_trace_lines(handle)
    return summarize_trace_sample(
        query_id=query_id,
        sample=sample,
        instantiation_id=instantiation_id,
        args_string=args_string,
        hardware_counter_summary=hardware_counter_summary,
        compiler_vectorization_summary=compiler_vectorization_summary,
    )


def merge_trace_summaries(
    *,
    query_id: str,
    summaries: list[TraceHotspotSummary],
    omitted_count: int,
) -> TraceHotspotSummary:
    """Merge multiple instantiation summaries into one query-level hotspot view."""
    if not summaries:
        return TraceHotspotSummary(
            query_id=query_id,
            issue_class="evidence_insufficient",
            evidence_sufficient=False,
            top_profiles=(),
            counters={},
            summary_text=f"Query {query_id}: no trace samples collected.",
            sampled_instantiations=(),
            sampled_count=0,
            omitted_count=omitted_count,
        )
    profile_totals: dict[str, int] = {}
    counter_max: dict[str, int] = {}
    sampled_instantiations: list[str] = []
    vectorization_candidate = False
    hardware_counter_summary: dict[str, Any] = {}
    compiler_vectorization_summary: dict[str, Any] = {}
    for summary in summaries:
        sampled_instantiations.extend(summary.sampled_instantiations)
        vectorization_candidate = (
            vectorization_candidate or summary.vectorization_candidate
        )
        for name, value in summary.top_profiles:
            profile_totals[name] = profile_totals.get(name, 0) + value
        for name, value in summary.counters.items():
            counter_max[name] = max(counter_max.get(name, 0), value)
        if summary.hardware_counter_summary:
            hardware_counter_summary = dict(summary.hardware_counter_summary)
        if summary.compiler_vectorization_summary:
            compiler_vectorization_summary = dict(
                summary.compiler_vectorization_summary
            )
    merged_top_profiles = tuple(
        sorted(profile_totals.items(), key=lambda item: item[1], reverse=True)[:8]
    )
    issue_class = classify_trace_issue(merged_top_profiles, counter_max)
    evidence_sufficient = (
        any(summary.evidence_sufficient for summary in summaries)
        and issue_class != "evidence_insufficient"
    )
    return TraceHotspotSummary(
        query_id=query_id,
        issue_class=issue_class,
        evidence_sufficient=evidence_sufficient,
        top_profiles=merged_top_profiles,
        counters=counter_max,
        summary_text=render_trace_summary_text(
            query_id,
            merged_top_profiles,
            counter_max,
            issue_class,
            vectorization_candidate=vectorization_candidate,
            hardware_counter_summary=hardware_counter_summary,
            change_scope=recommend_change_scope(query_id, issue_class),
        ),
        sampled_instantiations=tuple(sampled_instantiations),
        sampled_count=len(sampled_instantiations),
        omitted_count=omitted_count,
        vectorization_candidate=vectorization_candidate,
        hardware_counter_summary=hardware_counter_summary,
        compiler_vectorization_summary=compiler_vectorization_summary,
        change_scope=recommend_change_scope(query_id, issue_class),
    )


def classify_trace_issue(
    top_profiles: tuple[tuple[str, int], ...],
    counters: dict[str, Any],
) -> str:
    """Classify the dominant optimization issue from trace names and counters."""
    if len(top_profiles) < 5:
        return "evidence_insufficient"
    profile_names = {name for name, _value in top_profiles}
    rows_scanned = counters.get("rows_scanned", 0)
    output_rows = counters.get("query_output_rows", counters.get("rows_emitted", 0))
    cache_miss_rate = float(counters.get("cache_miss_rate", 0.0) or 0.0)
    llc_mpki = float(counters.get("llc_mpki", 0.0) or 0.0)
    dtlb_mpki = float(counters.get("dtlb_mpki", 0.0) or 0.0)
    branch_miss_rate = float(counters.get("branch_miss_rate", 0.0) or 0.0)
    if dtlb_mpki >= 5.0:
        return "tlb/allocation bound"
    if cache_miss_rate >= 0.05 or llc_mpki >= 10.0:
        return "cache/layout bound"
    if branch_miss_rate >= 0.03 or any("branch" in name for name in profile_names):
        return "branch/filter bound"
    if any("sort" in name or "merge" in name for name in profile_names):
        return "algorithm/suboptimal bound"
    if any("output" in name or "material" in name for name in profile_names):
        return "materialization/output bound"
    if rows_scanned > 0 and output_rows * 10 < rows_scanned:
        return "layout/access-path bound"
    if rows_scanned > 0 and output_rows > rows_scanned // 2:
        return "materialization/output bound"
    if any("scan" in name or "aggregate" in name or "bucket" in name for name in profile_names):
        return "kernel/compute bound"
    return "evidence_insufficient"


def render_trace_summary_text(
    query_id: str,
    top_profiles: tuple[tuple[str, int], ...],
    counters: dict[str, Any],
    issue_class: str,
    *,
    vectorization_candidate: bool = False,
    hardware_counter_summary: dict[str, Any] | None = None,
    change_scope: str | None = None,
) -> str:
    """Render trace, counter, and perf hotspot evidence for optimizer prompts."""
    lines = [
        f"Query {query_id}",
        f"Issue class: {issue_class}",
        f"Vectorization candidate: {vectorization_candidate}",
        f"Change scope: {change_scope or 'query'}",
        "Top profiles:",
    ]
    lines.extend(f"- {name}: {value} ns" for name, value in top_profiles)
    lines.append("Counters:")
    lines.extend(f"- {name}: {value}" for name, value in sorted(counters.items()))
    lines.extend(render_perf_hotspot_lines(hardware_counter_summary))
    return "\n".join(lines)


def render_perf_hotspot_lines(
    hardware_counter_summary: dict[str, Any] | None,
) -> list[str]:
    """Render perf hotspot evidence with sample counts and optional source lines."""
    if not hardware_counter_summary:
        return []
    top_symbols = hardware_counter_summary.get("perf_top_symbols", [])
    top_frames = hardware_counter_summary.get("perf_top_frames", [])
    top_source_lines = hardware_counter_summary.get("perf_top_source_lines", [])
    if not hardware_counter_summary.get("perf_hotspots_available") or not top_symbols:
        return []
    sample_count = int(hardware_counter_summary.get("perf_sample_count", 0) or 0)
    lines = ["Perf hotspots:", f"- samples: {sample_count}", "Perf top symbols:"]
    lines.extend(_render_perf_pairs(top_symbols, sample_count, "samples"))
    if top_source_lines:
        lines.append("Perf top source lines:")
        lines.extend(_render_perf_pairs(top_source_lines, sample_count, "samples"))
    if top_frames:
        lines.append("Perf top call-stack frames:")
        lines.extend(_render_perf_pairs(top_frames, sample_count, "frames"))
    return lines


def _render_perf_pairs(
    values: Any,
    sample_count: int,
    unit: str,
) -> list[str]:
    pairs = [
        (str(item[0]), int(item[1]))
        for item in values
        if isinstance(item, (list, tuple)) and len(item) >= 2
    ]
    return [
        f"- {name}: {count} {unit}{_format_perf_percentage(count, sample_count)}"
        for name, count in pairs[:8]
    ]


def _format_perf_percentage(count: int, sample_count: int) -> str:
    if sample_count <= 0:
        return ""
    return f" ({count * 100.0 / sample_count:.1f}%)"


def recommend_change_scope(query_id: str, issue_class: str) -> str:
    """Recommend whether an optimization should stay query-local or family-scoped."""
    family_queries = {"3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14"}
    if issue_class in {
        "cache/layout bound",
        "tlb/allocation bound",
        "algorithm/suboptimal bound",
        "kernel/compute bound",
    } and str(query_id) in family_queries:
        return "family"
    return "query"
