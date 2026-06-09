"""Optimization Run Summary 管理工具."""

import dataclasses
import json
import logging
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from tpch_monetdb.config import DEFAULT_TPCH_MONETDB_ARTIFACTS_DIR
from tpch_monetdb.benchmark.runtime_accounting import (
    KERNEL_RUNTIME_METRIC_KIND,
    OPTIMIZATION_RUNTIME_METRIC_KINDS,
)
from tpch_monetdb.utils.pipeline_evidence import MeasurementKind, MeasurementShapeStatus
from tpch_monetdb.utils.summary_gates import is_successful_large_data_run
from tpch_monetdb.utils.duration_format import (
    format_duration_ms,
    is_positive_finite_runtime_ms,
    safe_speedup,
)

logger = logging.getLogger(__name__)


def baseline_display_name_for_benchmark(benchmark: str) -> str:
    """Return the human-readable baseline name for one benchmark."""
    if str(benchmark).strip().lower() == "tpch":
        return "MonetDB"
    return "Baseline"


def baseline_engine_for_benchmark(benchmark: str) -> str:
    """Return the measurement-record engine name for one benchmark baseline."""
    if str(benchmark).strip().lower() == "tpch":
        return "monetdb"
    return "baseline"


class ValidationKernelReportStatus(str, Enum):
    """Final validator report availability status."""

    AVAILABLE = "available"
    MISSING_METRICS = "missing_metrics"
    CORRECTNESS_FAILED = "correctness_failed"
    INCOMPLETE_METRICS = "incomplete_metrics"


@dataclass(frozen=True)
class ValidationKernelReportRow:
    """One query row from the final validator no-CSV kernel report."""

    query_id: str
    implementation_ms: float
    baseline_ms: float
    speedup: float


@dataclass(frozen=True)
class ValidationKernelReport:
    """Structured final validator no-CSV kernel report."""

    conv_name: str
    status: ValidationKernelReportStatus
    query_ids: tuple[str, ...]
    scale_factor: Any = "unknown"
    rows: tuple[ValidationKernelReportRow, ...] = ()
    missing_metrics: tuple[str, ...] = ()
    total_implementation_ms: float | None = None
    total_baseline_ms: float | None = None
    total_speedup: float | None = None
    speedup_warn_threshold: float = 1.0

    def is_available(self) -> bool:
        """Return whether this report has complete validator metrics."""
        return self.status == ValidationKernelReportStatus.AVAILABLE


@dataclass
class OptimizationRunSummary:
    """Optimization run 成功摘要."""

    benchmark: str
    conv_name: str
    run_id: str
    query_list: list[str]
    is_bespoke_storage: bool
    start_snapshot_hash: str
    final_snapshot_hash: str
    best_runtime_ms_by_query: dict[str, float]
    final_runtime_ms_by_query: dict[str, float]
    final_correctness: bool
    completed_at: str
    conversation_json: str
    session_db_path: str
    success: bool
    baseline_runtime_ms_by_query: dict[str, float] = field(default_factory=dict)
    issue_class_by_query: dict[str, str] = field(default_factory=dict)
    wandb_primary_run_id: str | None = None
    wandb_final_run_id: str | None = None
    wandb_init_attempt_count: int = 0
    wandb_attempted_run_ids: list[str] = field(default_factory=list)
    wandb_retry_used: bool = False
    wandb_first_failure_excerpt: str | None = None
    failure_code: str | None = None
    failure_detail: str | None = None
    hotspot_summary_path: str | None = None
    stage_records: list[dict[str, Any]] = field(default_factory=list)
    global_regression_records: list[dict[str, Any]] = field(default_factory=list)
    global_optimization_candidates: list[dict[str, Any]] = field(default_factory=list)
    global_optimization_winner: dict[str, Any] = field(default_factory=dict)
    optimization_units: list[dict[str, Any]] = field(default_factory=list)
    unit_scores: dict[str, float] = field(default_factory=dict)
    control_artifact_hashes: dict[str, str] = field(default_factory=dict)
    control_artifacts_read_by_stage: dict[str, list[str]] = field(default_factory=dict)
    control_artifacts_injected_by_stage: dict[str, list[str]] = field(default_factory=dict)
    storage_plan_alignment: dict[str, Any] = field(default_factory=dict)
    todo_reconciliation: dict[str, Any] = field(default_factory=dict)
    change_scope: str | None = None
    stage_history: list[dict[str, Any]] = field(default_factory=list)
    completed_stage_summaries: list[dict[str, Any]] = field(default_factory=list)
    measurement_repetition: dict[str, Any] = field(default_factory=dict)
    hardware_counter_summary: dict[str, Any] = field(default_factory=dict)
    compiler_vectorization_summary: dict[str, Any] = field(default_factory=dict)
    workload_objective: dict[str, Any] = field(default_factory=dict)
    objective_failures: list[str] = field(default_factory=list)
    objective_failure_route: str | None = None
    measurement_records: list[dict[str, Any]] = field(default_factory=list)
    final_validation_metrics: dict[str, Any] = field(default_factory=dict)
    pipeline_evidence_ledger: dict[str, Any] = field(default_factory=dict)
    build_profile: str | None = None
    target_cpu: str | None = None
    hotspot_analysis_degraded: bool = False
    hotspot_analysis_failure_reason: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "OptimizationRunSummary":
        known = {f.name for f in dataclasses.fields(cls)}
        payload = {k: v for k, v in data.items() if k in known}
        payload["is_bespoke_storage"] = True
        return cls(**payload)


def render_bottleneck_report(
    summary: "OptimizationRunSummary",
    speedup_warn_threshold: float = 1.0,
) -> str:
    """Render a human/LLM-readable bottleneck report from an OptimizationRunSummary.

    Lists all queries with their speedup vs the active baseline, marking those
    below speedup_warn_threshold as ⚠ slow.

    Args:
        summary: The optimization run summary to report on.
        speedup_warn_threshold: Queries with speedup below this value are marked slow.

    Returns:
        Formatted markdown string suitable for prompt injection.
    """
    final_rt = summary.final_runtime_ms_by_query
    baseline_rt = summary.baseline_runtime_ms_by_query
    baseline_label = baseline_display_name_for_benchmark(summary.benchmark)
    query_list = sorted(summary.query_list)
    measurement_kind_by_query_engine = _validate_report_measurement_records(summary)

    rows: list[str] = []
    issue_classes = summary.issue_class_by_query or infer_issue_class_by_query(
        query_list=summary.query_list,
        final_runtime_ms_by_query=summary.final_runtime_ms_by_query,
        baseline_runtime_ms_by_query=summary.baseline_runtime_ms_by_query,
        final_correctness=summary.final_correctness,
        speedup_warn_threshold=speedup_warn_threshold,
    )

    rows.append(f"## Previous round ({summary.conv_name}) results")
    if not summary.success:
        rows.append(
            "Run status: failed; aggregate no-CSV kernel speedup is intentionally suppressed."
        )
    if summary.change_scope:
        rows.append(f"Change scope: `{summary.change_scope}`")
    if summary.measurement_repetition:
        rows.append(
            "Measurement repetition: "
            + json.dumps(summary.measurement_repetition, ensure_ascii=False, sort_keys=True)
        )
    if measurement_kind_by_query_engine:
        rows.append(
            "Measurement records: "
            + json.dumps(
                measurement_kind_by_query_engine,
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        shape_status_by_query_engine = _collect_report_measurement_shape_status(summary)
        rows.append(
            "Measurement shape status: "
            + json.dumps(
                shape_status_by_query_engine,
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        if _has_unknown_measurement_shape(shape_status_by_query_engine):
            rows.append(
                "Measurement shape warning: exact-instantiation row/output shape is unknown; "
                "do not interpret these speedups as fixed-SF or heavy-load conclusions."
            )
    if not summary.final_correctness:
        rows.append(
            "Correctness gate: previous round `final_correctness=false`; do not use this report as layout evidence."
        )
    rows.append(f"| Query | Impl no-CSV kernel ms | {baseline_label} ms | Kernel Speedup | Issue Class | Status |")
    rows.append("|-------|-----------------------|------------|----------------|-------------|--------|")

    slow_queries: list[str] = []
    for qid in query_list:
        impl_ms = final_rt.get(qid, float("inf"))
        base_ms = baseline_rt.get(qid, 0.0)
        speedup = safe_speedup(base_ms, impl_ms)
        if speedup is not None:
            status = "✅ good" if speedup >= speedup_warn_threshold else "⚠ slow"
            if speedup < speedup_warn_threshold:
                slow_queries.append(qid)
            issue_class = issue_classes.get(qid, "mixed")
            rows.append(
                f"| {qid} | {format_duration_ms(impl_ms)} | {format_duration_ms(base_ms)} | "
                f"{speedup:.2f}× | {issue_class} | {status} |"
            )
        else:
            issue_class = issue_classes.get(qid, "mixed")
            rows.append(
                f"| {qid} | {format_duration_ms(impl_ms)} | "
                f"{format_duration_ms(base_ms)} | n/a | {issue_class} | ❓ |"
            )

    if slow_queries:
        rows.append(f"Slow queries (<{speedup_warn_threshold:.1f}×): {', '.join(slow_queries)}")
    else:
        rows.append("No slow queries detected.")

    speedup_values = [
        speedup
        for qid in query_list
        for speedup in [safe_speedup(baseline_rt.get(qid), final_rt.get(qid))]
        if speedup is not None
    ]
    if summary.success and speedup_values:
        agg_speedup = math.prod(speedup_values) ** (1.0 / len(speedup_values))
        rows.append(f"Aggregate no-CSV kernel speedup: {agg_speedup:.2f}×")

    return "\n".join(rows)


def _sorted_query_ids(query_ids: list[str]) -> list[str]:
    """按查询编号排序，非数字 ID 保持字典序兜底."""
    return sorted(
        [str(qid) for qid in query_ids],
        key=lambda qid: (0, int(qid)) if qid.isdigit() else (1, qid),
    )


def _float_metric(metrics: dict[str, Any], key: str) -> float | None:
    """从 metrics 中读取正有限浮点值."""
    value = metrics.get(key)
    if not is_positive_finite_runtime_ms(value):
        return None
    return float(value)


def _validator_query_prefix(query_id: str) -> str:
    """返回 validator per-query metric 前缀."""
    return f"validation/query_{str(query_id).zfill(3)}"


def build_validation_kernel_report(
    metrics: dict[str, Any],
    query_ids: list[str],
    conv_name: str,
    speedup_warn_threshold: float = 1.0,
) -> ValidationKernelReport:
    """Build a structured report from final-validator no-CSV kernel metrics."""
    sorted_query_ids = tuple(_sorted_query_ids(query_ids))
    if not metrics:
        return ValidationKernelReport(
            conv_name=conv_name,
            status=ValidationKernelReportStatus.MISSING_METRICS,
            query_ids=sorted_query_ids,
            speedup_warn_threshold=speedup_warn_threshold,
        )
    if metrics.get("validation/correct") is False:
        return ValidationKernelReport(
            conv_name=conv_name,
            status=ValidationKernelReportStatus.CORRECTNESS_FAILED,
            query_ids=sorted_query_ids,
            scale_factor=metrics.get("validation/scale_factor", "unknown"),
            speedup_warn_threshold=speedup_warn_threshold,
        )

    missing: list[str] = []
    report_rows: list[ValidationKernelReportRow] = []
    for query_id in sorted_query_ids:
        prefix = _validator_query_prefix(query_id)
        metric_kind = str(metrics.get(f"{prefix}/runtime_metric_kind") or "")
        impl_ms = _float_metric(metrics, f"{prefix}/no_csv_runtime_ms")
        baseline_ms = _float_metric(metrics, f"{prefix}/baseline_runtime_ms")
        if metric_kind != KERNEL_RUNTIME_METRIC_KIND:
            missing.append(f"Q{query_id}:runtime_metric_kind={metric_kind or 'missing'}")
            continue
        if impl_ms is None:
            missing.append(f"Q{query_id}:no_csv_runtime_ms")
            continue
        if baseline_ms is None:
            missing.append(f"Q{query_id}:baseline_runtime_ms")
            continue
        speedup = safe_speedup(baseline_ms, impl_ms)
        if speedup is None:
            missing.append(f"Q{query_id}:kernel_speedup")
            continue
        report_rows.append(
            ValidationKernelReportRow(
                query_id=query_id,
                implementation_ms=impl_ms,
                baseline_ms=baseline_ms,
                speedup=speedup,
            )
        )

    if missing:
        return ValidationKernelReport(
            conv_name=conv_name,
            status=ValidationKernelReportStatus.INCOMPLETE_METRICS,
            query_ids=sorted_query_ids,
            scale_factor=metrics.get("validation/scale_factor", "unknown"),
            missing_metrics=tuple(missing),
            speedup_warn_threshold=speedup_warn_threshold,
        )
    total_impl_ms = _float_metric(metrics, "validation/total_no_csv_runtime_ms")
    total_baseline_ms = _float_metric(metrics, "validation/total_baseline_runtime_ms")
    total_speedup = _float_metric(metrics, "validation/no_csv_total_speedup")
    if total_speedup is None:
        total_speedup = _float_metric(metrics, "validation/total_kernel_speedup")
    if total_speedup is None and total_impl_ms is not None and total_baseline_ms is not None:
        total_speedup = safe_speedup(total_baseline_ms, total_impl_ms)
    return ValidationKernelReport(
        conv_name=conv_name,
        status=ValidationKernelReportStatus.AVAILABLE,
        query_ids=sorted_query_ids,
        scale_factor=metrics.get("validation/scale_factor", "unknown"),
        rows=tuple(report_rows),
        total_implementation_ms=total_impl_ms,
        total_baseline_ms=total_baseline_ms,
        total_speedup=total_speedup,
        speedup_warn_threshold=speedup_warn_threshold,
    )


def render_validation_kernel_report(
    report_or_metrics: ValidationKernelReport | dict[str, Any],
    query_ids: list[str] | None = None,
    conv_name: str = "",
    speedup_warn_threshold: float = 1.0,
) -> str:
    """Render the structured final-validator no-CSV kernel report as Markdown."""
    if isinstance(report_or_metrics, ValidationKernelReport):
        report = report_or_metrics
    else:
        if query_ids is None:
            raise ValueError("query_ids is required when rendering metrics directly")
        report = build_validation_kernel_report(
            report_or_metrics,
            query_ids,
            conv_name,
            speedup_warn_threshold=speedup_warn_threshold,
        )

    rows = [f"## Previous round ({report.conv_name}) validator results"]
    if report.status == ValidationKernelReportStatus.MISSING_METRICS:
        rows.append(
            "Official validator report unavailable: final validation metrics are missing."
        )
        return "\n".join(rows)
    if report.status == ValidationKernelReportStatus.CORRECTNESS_FAILED:
        rows.append(
            "Official validator report unavailable: final validator correctness failed."
        )
        return "\n".join(rows)
    if report.status == ValidationKernelReportStatus.INCOMPLETE_METRICS:
        rows.append(
            "Official validator report unavailable: missing or invalid metrics: "
            + ", ".join(report.missing_metrics)
        )
        rows.append(
            "Do not reuse diagnostic measurement tables as final validator evidence."
        )
        return "\n".join(rows)

    rows.append(f"Validator scale factor: `{report.scale_factor}`")
    rows.append(
        "| Query | Validator no-CSV kernel ms | Validator baseline ms | Kernel Speedup | Status |"
    )
    rows.append("|-------|-----------------------------|----------------------|----------------|--------|")
    slow_queries: list[str] = []
    for row in report.rows:
        status = "✅ good" if row.speedup >= report.speedup_warn_threshold else "⚠ slow"
        if row.speedup < report.speedup_warn_threshold:
            slow_queries.append(row.query_id)
        rows.append(
            f"| {row.query_id} | {format_duration_ms(row.implementation_ms)} | "
            f"{format_duration_ms(row.baseline_ms)} | {row.speedup:.2f}× | {status} |"
        )
    if slow_queries:
        rows.append(
            f"Slow queries (<{report.speedup_warn_threshold:.1f}×): "
            + ", ".join(slow_queries)
        )
    else:
        rows.append("No slow queries detected.")

    if report.total_implementation_ms is not None and report.total_baseline_ms is not None:
        rows.append(
            "Validator no-CSV kernel totals: "
            f"{format_duration_ms(report.total_implementation_ms)} (Generated TPC-H) vs "
            f"{format_duration_ms(report.total_baseline_ms)} (baseline)"
        )
    if report.total_speedup is not None:
        rows.append(f"Total validator no-CSV kernel speedup: {report.total_speedup:.2f}×")
    return "\n".join(rows)


def _validate_report_measurement_records(
    summary: "OptimizationRunSummary",
) -> dict[str, dict[str, str]]:
    """Require speedup rows to use compatible exact-instantiation records."""
    if not summary.measurement_records:
        return {}
    required_kind = MeasurementKind.EXACT_INSTANTIATION.value
    baseline_engine = baseline_engine_for_benchmark(summary.benchmark)
    by_query_engine: dict[str, dict[str, str]] = {}
    bad_records: list[str] = []
    duplicate_records: list[str] = []
    for record in summary.measurement_records:
        query_id = str(record.get("query_id") or "")
        engine = str(record.get("engine") or "")
        kind = str(record.get("measurement_kind") or "")
        if not query_id or not engine:
            bad_records.append(json.dumps(record, ensure_ascii=False, sort_keys=True))
            continue
        if kind != required_kind:
            bad_records.append(f"{query_id}:{engine}:{kind}")
            continue
        provenance = dict(record.get("provenance") or {})
        if engine == "generated_tpch":
            metric_kind = str(provenance.get("runtime_metric_kind") or "")
            if metric_kind not in OPTIMIZATION_RUNTIME_METRIC_KINDS:
                bad_records.append(
                    f"{query_id}:{engine}:runtime_metric_kind={metric_kind}"
                )
                continue
        engine_map = by_query_engine.setdefault(query_id, {})
        if engine in engine_map and engine_map[engine] != kind:
            duplicate_records.append(f"{query_id}:{engine}")
            continue
        engine_map[engine] = kind
    missing_records: list[str] = []
    for query_id in summary.query_list:
        if query_id in summary.final_runtime_ms_by_query and "generated_tpch" not in by_query_engine.get(query_id, {}):
            missing_records.append(f"{query_id}:generated_tpch")
        if query_id in summary.baseline_runtime_ms_by_query and baseline_engine not in by_query_engine.get(query_id, {}):
            missing_records.append(f"{query_id}:{baseline_engine}")
    if bad_records or duplicate_records or missing_records:
        raise ValueError(
            "Bottleneck report measurement provenance is inconsistent: "
            f"bad={bad_records}, duplicates={duplicate_records}, missing={missing_records}"
        )
    return by_query_engine


def _collect_report_measurement_shape_status(
    summary: "OptimizationRunSummary",
) -> dict[str, dict[str, str]]:
    """Collect measurement shape status for report annotations."""
    shape_by_query_engine: dict[str, dict[str, str]] = {}
    for record in summary.measurement_records:
        query_id = str(record.get("query_id") or "")
        engine = str(record.get("engine") or "")
        if not query_id or not engine:
            continue
        shape_status = str(
            record.get("measurement_shape_status")
            or MeasurementShapeStatus.UNKNOWN.value
        )
        shape_by_query_engine.setdefault(query_id, {})[engine] = shape_status
    return shape_by_query_engine


def _has_unknown_measurement_shape(
    shape_status_by_query_engine: dict[str, dict[str, str]],
) -> bool:
    """Return whether any report measurement lacks known row/output shape metadata."""
    unknown_values = {"", MeasurementShapeStatus.UNKNOWN.value}
    return any(
        status in unknown_values
        for engine_map in shape_status_by_query_engine.values()
        for status in engine_map.values()
    )


def infer_issue_class_by_query(
    *,
    query_list: list[str],
    final_runtime_ms_by_query: dict[str, float],
    baseline_runtime_ms_by_query: dict[str, float],
    final_correctness: bool,
    speedup_warn_threshold: float = 1.0,
) -> dict[str, str]:
    """Infer coarse per-query issue classes for outer-loop feedback reuse."""
    implementation_queries = {"6", "9", "13", "14"}
    layout_queries = {"15"}
    mixed_queries = {"8", "11", "12"}
    issue_class_by_query: dict[str, str] = {}
    for qid in sorted(query_list):
        impl_ms = final_runtime_ms_by_query.get(qid, float("inf"))
        base_ms = baseline_runtime_ms_by_query.get(qid, 0.0)
        if not final_correctness:
            issue_class_by_query[qid] = "correctness"
            continue
        speedup = safe_speedup(base_ms, impl_ms)
        if speedup is None:
            issue_class_by_query[qid] = "invalid_runtime"
            continue
        if speedup >= speedup_warn_threshold:
            issue_class_by_query[qid] = "stable"
            continue
        if qid in implementation_queries:
            issue_class_by_query[qid] = "implementation"
            continue
        if qid in layout_queries:
            issue_class_by_query[qid] = "layout"
            continue
        if qid in mixed_queries:
            issue_class_by_query[qid] = "mixed"
            continue
        issue_class_by_query[qid] = "mixed"
    return issue_class_by_query


def get_summary_dir(artifacts_dir: Path) -> Path:
    return artifacts_dir / "optimization_runs"


def _load_summary_file(file_path: Path) -> OptimizationRunSummary:
    with open(file_path) as f:
        data = json.load(f)
    return OptimizationRunSummary.from_dict(data)


def _load_latest_summary(conv_dir: Path) -> OptimizationRunSummary | None:
    latest_path = conv_dir / "latest.json"
    if not latest_path.exists():
        return None
    with open(latest_path) as f:
        latest_data = json.load(f)
    embedded_summary = latest_data.get("summary")
    if isinstance(embedded_summary, dict):
        return OptimizationRunSummary.from_dict(embedded_summary)
    latest_file = latest_data.get("latest_file")
    if not isinstance(latest_file, str) or not latest_file:
        raise KeyError("latest.json missing latest_file")
    return _load_summary_file(conv_dir / latest_file)


def write_optimization_run_summary(
    summary: OptimizationRunSummary,
    artifacts_dir: Path,
) -> Path:
    if not summary.final_snapshot_hash:
        raise ValueError("final_snapshot_hash cannot be empty")
    if summary.success and not summary.final_runtime_ms_by_query:
        raise ValueError("final_runtime_ms_by_query cannot be empty")
    summary.is_bespoke_storage = True

    summary_dir = get_summary_dir(artifacts_dir)
    conv_dir = summary_dir / summary.conv_name
    conv_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{summary.run_id}.json"

    file_path = conv_dir / filename
    with open(file_path, "w") as f:
        json.dump(summary.to_dict(), f, indent=2)

    latest_path = conv_dir / "latest.json"
    with open(latest_path, "w") as f:
        json.dump({
            "latest_file": filename,
            "summary": summary.to_dict(),
        }, f, indent=2)

    logger.info(f"Written optimization run summary to {file_path}")
    return file_path


def persist_successful_optimization_run(
    *,
    benchmark: str,
    conv_name: str,
    query_list: list[str],
    is_bespoke_storage: bool,
    start_snapshot_hash: str,
    final_snapshot_hash: str,
    best_runtime_ms_by_query: dict[str, float],
    final_runtime_ms_by_query: dict[str, float],
    final_correctness: bool,
    conversation_json_path: Path,
    session_db_path: Path,
    artifacts_dir: Path,
    baseline_runtime_ms_by_query: "Optional[dict[str, float]]" = None,
    issue_class_by_query: "Optional[dict[str, str]]" = None,
    final_validation_metrics: dict[str, Any] | None = None,
    wandb_result: "Optional[Any]" = None,
) -> Path:
    resolved_baseline = baseline_runtime_ms_by_query or {}
    resolved_issue_classes = issue_class_by_query or infer_issue_class_by_query(
        query_list=query_list,
        final_runtime_ms_by_query=final_runtime_ms_by_query,
        baseline_runtime_ms_by_query=resolved_baseline,
        final_correctness=final_correctness,
    )
    summary = OptimizationRunSummary(
        benchmark=benchmark,
        conv_name=conv_name,
        run_id=conv_name,
        query_list=query_list,
        is_bespoke_storage=True,
        start_snapshot_hash=start_snapshot_hash,
        final_snapshot_hash=final_snapshot_hash,
        best_runtime_ms_by_query=best_runtime_ms_by_query,
        final_runtime_ms_by_query=final_runtime_ms_by_query,
        final_correctness=final_correctness,
        completed_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        conversation_json=str(conversation_json_path),
        session_db_path=str(session_db_path),
        success=True,
        baseline_runtime_ms_by_query=resolved_baseline,
        issue_class_by_query=resolved_issue_classes,
        final_validation_metrics=final_validation_metrics or {},
        wandb_primary_run_id=wandb_result.primary_run_id if wandb_result is not None else None,
        wandb_final_run_id=wandb_result.final_run_id if wandb_result is not None else None,
        wandb_init_attempt_count=wandb_result.attempt_count if wandb_result is not None else 0,
        wandb_attempted_run_ids=wandb_result.attempted_run_ids if wandb_result is not None else [],
        wandb_retry_used=wandb_result.used_fallback if wandb_result is not None else False,
        wandb_first_failure_excerpt=wandb_result.first_failure_excerpt if wandb_result is not None else None,
    )
    file_path = write_optimization_run_summary(summary, artifacts_dir)
    logger.info(f"Optimization run completed successfully. Summary written to {file_path}")
    return file_path


def find_latest_successful_optimization_run(
    conv_name: str | None,
    query_list: list[str],
    benchmark: str = "tpch",
    artifacts_dir: Path | None = None,
) -> Optional[OptimizationRunSummary]:
    if artifacts_dir is None:
        artifacts_dir = Path(DEFAULT_TPCH_MONETDB_ARTIFACTS_DIR)

    summary_dir = get_summary_dir(artifacts_dir)
    if conv_name is None:
        conv_dirs = [path for path in summary_dir.iterdir() if path.is_dir()] if summary_dir.exists() else []
    else:
        conv_dirs = [summary_dir / conv_name]

    if not conv_dirs:
        logger.info(f"No optimization run directory found for {conv_name}")
        return None

    if conv_name is not None:
        conv_dir = conv_dirs[0]
        if conv_dir.exists():
            try:
                latest_summary = _load_latest_summary(conv_dir)
                if (
                    latest_summary is not None
                    and latest_summary.success
                    and latest_summary.benchmark == benchmark
                    and set(latest_summary.query_list) == set(query_list)
                    and is_successful_large_data_run(latest_summary)
                ):
                    return latest_summary
            except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                logger.warning(f"Failed to read latest summary pointer for {conv_name}: {exc}")

    summaries = []
    for conv_dir in conv_dirs:
        if not conv_dir.exists():
            continue
        for file_path in conv_dir.glob("*.json"):
            if file_path.name == "latest.json":
                continue
            try:
                summary = _load_summary_file(file_path)
                if (
                    summary.success
                    and summary.benchmark == benchmark
                    and set(summary.query_list) == set(query_list)
                    and is_successful_large_data_run(summary)
                ):
                    summaries.append(summary)
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                logger.warning(f"Failed to parse summary file {file_path}: {e}")
                continue

    if not summaries:
        logger.info(f"No compatible optimization run found for {conv_name}")
        return None

    summaries.sort(key=lambda s: s.completed_at, reverse=True)
    latest = summaries[0]
    logger.info(f"Found latest optimization run: {latest.conv_name} at {latest.completed_at}")
    return latest


def find_latest_optimization_run(
    conv_name: str | None,
    query_list: list[str],
    benchmark: str = "tpch",
    artifacts_dir: Path | None = None,
) -> Optional[OptimizationRunSummary]:
    if artifacts_dir is None:
        artifacts_dir = Path(DEFAULT_TPCH_MONETDB_ARTIFACTS_DIR)
    summary_dir = get_summary_dir(artifacts_dir)
    if conv_name is None:
        conv_dirs = [path for path in summary_dir.iterdir() if path.is_dir()] if summary_dir.exists() else []
    else:
        conv_dirs = [summary_dir / conv_name]
    if not conv_dirs:
        return None
    if conv_name is not None:
        conv_dir = conv_dirs[0]
        if conv_dir.exists():
            try:
                latest_summary = _load_latest_summary(conv_dir)
                if latest_summary is not None and latest_summary.benchmark == benchmark and set(latest_summary.query_list) == set(query_list):
                    return latest_summary
            except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
                pass
    summaries = []
    for conv_dir in conv_dirs:
        if not conv_dir.exists():
            continue
        for file_path in conv_dir.glob("*.json"):
            if file_path.name == "latest.json":
                continue
            try:
                summary = _load_summary_file(file_path)
                if summary.benchmark == benchmark and set(summary.query_list) == set(query_list):
                    summaries.append(summary)
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
    if not summaries:
        return None
    summaries.sort(key=lambda s: s.completed_at, reverse=True)
    return summaries[0]


def persist_optimization_run(
    *,
    benchmark: str,
    conv_name: str,
    query_list: list[str],
    is_bespoke_storage: bool,
    start_snapshot_hash: str,
    final_snapshot_hash: str,
    artifacts_dir: Path,
    success: bool,
    final_correctness: bool,
    best_runtime_ms_by_query: dict[str, float] | None = None,
    final_runtime_ms_by_query: dict[str, float] | None = None,
    baseline_runtime_ms_by_query: dict[str, float] | None = None,
    failure_code: str | None = None,
    failure_detail: str | None = None,
    hotspot_summary_path: Path | None = None,
    stage_records: list[dict[str, Any]] | None = None,
    global_regression_records: list[dict[str, Any]] | None = None,
    global_optimization_candidates: list[dict[str, Any]] | None = None,
    global_optimization_winner: dict[str, Any] | None = None,
    optimization_units: list[dict[str, Any]] | None = None,
    unit_scores: dict[str, float] | None = None,
    control_artifact_hashes: dict[str, str] | None = None,
    control_artifacts_read_by_stage: dict[str, list[str]] | None = None,
    control_artifacts_injected_by_stage: dict[str, list[str]] | None = None,
    storage_plan_alignment: dict[str, Any] | None = None,
    todo_reconciliation: dict[str, Any] | None = None,
    change_scope: str | None = None,
    stage_history: list[dict[str, Any]] | None = None,
    completed_stage_summaries: list[dict[str, Any]] | None = None,
    measurement_repetition: dict[str, Any] | None = None,
    hardware_counter_summary: dict[str, Any] | None = None,
    compiler_vectorization_summary: dict[str, Any] | None = None,
    workload_objective: dict[str, Any] | None = None,
    objective_failures: list[str] | None = None,
    objective_failure_route: str | None = None,
    measurement_records: list[dict[str, Any]] | None = None,
    final_validation_metrics: dict[str, Any] | None = None,
    pipeline_evidence_ledger: dict[str, Any] | None = None,
    build_profile: str | None = None,
    target_cpu: str | None = None,
    hotspot_analysis_degraded: bool = False,
    hotspot_analysis_failure_reason: str | None = None,
) -> Path:
    if not final_snapshot_hash:
        raise ValueError("final_snapshot_hash cannot be empty")
    summary = OptimizationRunSummary(
        benchmark=benchmark,
        conv_name=conv_name,
        run_id=conv_name,
        query_list=query_list,
        is_bespoke_storage=True,
        start_snapshot_hash=start_snapshot_hash,
        final_snapshot_hash=final_snapshot_hash,
        best_runtime_ms_by_query=best_runtime_ms_by_query or {},
        final_runtime_ms_by_query=final_runtime_ms_by_query or {},
        final_correctness=final_correctness,
        completed_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        conversation_json="",
        session_db_path="",
        success=success,
        baseline_runtime_ms_by_query=baseline_runtime_ms_by_query or {},
        issue_class_by_query={},
        failure_code=failure_code,
        failure_detail=failure_detail,
        hotspot_summary_path=str(hotspot_summary_path) if hotspot_summary_path else None,
        stage_records=stage_records or [],
        global_regression_records=global_regression_records or [],
        global_optimization_candidates=global_optimization_candidates or [],
        global_optimization_winner=global_optimization_winner or {},
        optimization_units=optimization_units or [],
        unit_scores=unit_scores or {},
        control_artifact_hashes=control_artifact_hashes or {},
        control_artifacts_read_by_stage=control_artifacts_read_by_stage or {},
        control_artifacts_injected_by_stage=control_artifacts_injected_by_stage or {},
        storage_plan_alignment=storage_plan_alignment or {},
        todo_reconciliation=todo_reconciliation or {},
        change_scope=change_scope,
        stage_history=stage_history or [],
        completed_stage_summaries=completed_stage_summaries or [],
        measurement_repetition=measurement_repetition or {},
        hardware_counter_summary=hardware_counter_summary or {},
        compiler_vectorization_summary=compiler_vectorization_summary or {},
        workload_objective=workload_objective or {},
        objective_failures=objective_failures or [],
        objective_failure_route=objective_failure_route,
        measurement_records=measurement_records or [],
        final_validation_metrics=final_validation_metrics or {},
        pipeline_evidence_ledger=pipeline_evidence_ledger or {},
        build_profile=build_profile,
        target_cpu=target_cpu,
        hotspot_analysis_degraded=hotspot_analysis_degraded,
        hotspot_analysis_failure_reason=hotspot_analysis_failure_reason,
    )
    return write_optimization_run_summary(summary, artifacts_dir)
