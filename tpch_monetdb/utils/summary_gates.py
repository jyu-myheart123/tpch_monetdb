from __future__ import annotations

import math
from typing import Any

from tpch_monetdb.utils.duration_format import is_positive_finite_runtime_ms
from tpch_monetdb.utils.pipeline_contracts import raise_pipeline_contract_error
from tpch_monetdb.utils.large_data_objectives import (
    collect_large_data_failures,
    is_large_data_success,
)
from tpch_monetdb.utils.pipeline_evidence import MeasurementKind, MeasurementShapeStatus

MEASUREMENT_AGGREGATION_GEOMEAN = "geomean"


def collect_measurable_failures(summary: Any) -> tuple[str, ...]:
    """Return the ordered failure codes that make a summary non-measurable."""
    failures: list[str] = []
    query_list = [str(qid) for qid in getattr(summary, "query_list", [])]
    runtime_map = dict(getattr(summary, "final_runtime_ms_by_query", {}) or {})
    baseline_map = dict(getattr(summary, "baseline_runtime_ms_by_query", {}) or {})
    control_hashes = dict(getattr(summary, "control_artifact_hashes", {}) or {})
    todo_reconciliation = dict(getattr(summary, "todo_reconciliation", {}) or {})
    storage_plan_alignment = dict(getattr(summary, "storage_plan_alignment", {}) or {})
    stage_history = list(getattr(summary, "stage_history", []) or [])
    measurement_repetition = dict(getattr(summary, "measurement_repetition", {}) or {})
    measurement_records = list(getattr(summary, "measurement_records", []) or [])

    if not getattr(summary, "success", False) or not getattr(
        summary, "final_correctness", False
    ):
        failures.append("SUMMARY_NOT_MEASURABLE")
    if not runtime_map:
        failures.append("SUMMARY_RUNTIME_MAP_EMPTY")
    else:
        for qid, runtime_ms in runtime_map.items():
            if not is_positive_finite_runtime_ms(runtime_ms):
                failures.append("SUMMARY_RUNTIME_INVALID")
                break
        if query_list and any(qid not in runtime_map for qid in query_list):
            failures.append("SUMMARY_RUNTIME_INVALID")
    if query_list:
        if any(qid not in baseline_map for qid in query_list):
            failures.append("SUMMARY_BASELINE_MISSING")
        else:
            for qid in query_list:
                runtime_ms = baseline_map[qid]
                if not is_positive_finite_runtime_ms(runtime_ms):
                    failures.append("SUMMARY_BASELINE_MISSING")
                    break
    if not control_hashes or not todo_reconciliation or not storage_plan_alignment:
        failures.append("SUMMARY_CONTROL_ARTIFACT_INCOMPLETE")
    if not stage_history:
        failures.append("SUMMARY_MEASUREMENT_INCOMPLETE")
    if not _has_valid_measurement_repetition(measurement_repetition, query_list):
        failures.append("SUMMARY_MEASUREMENT_INCOMPLETE")
    if _has_unknown_exact_measurement_shape(measurement_records):
        failures.append("SUMMARY_MEASUREMENT_SHAPE_UNKNOWN")
    return tuple(dict.fromkeys(failures))


def _has_valid_measurement_repetition(
    measurement_repetition: dict[str, Any],
    query_list: list[str],
) -> bool:
    """Validate repeated measurement evidence instead of accepting any non-empty dict."""
    if not measurement_repetition:
        return False
    required_fields = (
        "scale_factor",
        "query_ids",
        "repetitions",
        "sample_count",
        "aggregate_runtime_ms_samples",
        "aggregate_runtime_ms_median",
        "aggregate_runtime_ms_min",
        "aggregate_runtime_ms_max",
        "per_query_runtime_ms_samples",
        "aggregation_method",
        "source_command",
    )
    if any(field not in measurement_repetition for field in required_fields):
        return False
    repetitions = measurement_repetition.get("repetitions")
    sample_count = measurement_repetition.get("sample_count")
    if not isinstance(repetitions, int) or repetitions < 1:
        return False
    if sample_count != repetitions:
        return False
    samples = measurement_repetition.get("aggregate_runtime_ms_samples")
    if not isinstance(samples, list) or len(samples) != repetitions:
        return False
    if any(not _is_positive_finite_number(sample) for sample in samples):
        return False
    for field in (
        "aggregate_runtime_ms_median",
        "aggregate_runtime_ms_min",
        "aggregate_runtime_ms_max",
    ):
        if not _is_positive_finite_number(measurement_repetition.get(field)):
            return False
    measured_query_ids = measurement_repetition.get("query_ids")
    if not isinstance(measured_query_ids, list):
        return False
    if not measured_query_ids:
        return False
    if query_list and set(measured_query_ids) != set(query_list):
        return False
    per_query = measurement_repetition.get("per_query_runtime_ms_samples")
    if not isinstance(per_query, dict):
        return False
    aggregation_method = str(measurement_repetition.get("aggregation_method") or "")
    if aggregation_method != MEASUREMENT_AGGREGATION_GEOMEAN:
        return False
    for qid in measured_query_ids:
        values = per_query.get(qid)
        if not isinstance(values, list) or len(values) != repetitions:
            return False
        if any(not _is_positive_finite_number(value) for value in values):
            return False
    if not _aggregate_samples_match(samples, per_query, measured_query_ids, repetitions):
        return False
    source_command = measurement_repetition.get("source_command")
    return isinstance(source_command, str) and bool(source_command.strip())


def _has_unknown_exact_measurement_shape(
    measurement_records: list[Any],
) -> bool:
    """Return True when exact-instantiation measurements lack row/output shape."""
    for record in measurement_records:
        if not isinstance(record, dict):
            continue
        if not _measurement_kind_matches(
            record.get("measurement_kind"),
            MeasurementKind.EXACT_INSTANTIATION,
        ):
            continue
        if _is_unknown_measurement_shape_status(record.get("measurement_shape_status")):
            return True
    return False


def _measurement_kind_matches(value: Any, expected: MeasurementKind) -> bool:
    """Return whether a measurement kind value matches the expected enum."""
    if isinstance(value, MeasurementKind):
        return value == expected
    return str(value or "") == expected.value


def _is_unknown_measurement_shape_status(value: Any) -> bool:
    """Return whether a measurement shape status is missing or unknown."""
    if isinstance(value, MeasurementShapeStatus):
        return value == MeasurementShapeStatus.UNKNOWN
    normalized = str(value or "")
    return normalized in {"", MeasurementShapeStatus.UNKNOWN.value}


def _aggregate_samples_match(
    aggregate_samples: list[Any],
    per_query: dict[str, Any],
    measured_query_ids: list[Any],
    repetitions: int,
) -> bool:
    """Validate aggregate samples against per-query samples with geometric mean."""
    for sample_index in range(repetitions):
        per_query_values = [
            float(per_query[qid][sample_index])
            for qid in measured_query_ids
        ]
        expected = math.prod(per_query_values) ** (1.0 / len(per_query_values))
        if not math.isclose(
            float(aggregate_samples[sample_index]),
            expected,
            rel_tol=1e-6,
            abs_tol=1e-6,
        ):
            return False
    return True


def _is_positive_finite_number(value: Any) -> bool:
    """Return True when value is a positive finite int/float."""
    return is_positive_finite_runtime_ms(value)


def is_measurable_success(summary: Any) -> bool:
    """Return whether a summary is eligible for performance decisions."""
    return not collect_measurable_failures(summary)


def require_measurable_success(summary: Any, *, stage: str | None = None) -> None:
    """Raise a structured error when a summary is not measurable."""
    failures = collect_measurable_failures(summary)
    if not failures:
        return None
    raise_pipeline_contract_error(
        code=failures[0],
        message="Summary is not measurable: " + ", ".join(failures),
        stage=stage,
    )
    return None


def collect_success_failures(summary: Any) -> tuple[str, ...]:
    """Return all failures that block outer-loop success."""
    failures = [
        *collect_measurable_failures(summary),
        *collect_large_data_failures(summary),
    ]
    return tuple(dict.fromkeys(failures))


def is_successful_large_data_run(summary: Any) -> bool:
    """Return True only when measurable and large-data objective gates pass."""
    return is_measurable_success(summary) and is_large_data_success(summary)
