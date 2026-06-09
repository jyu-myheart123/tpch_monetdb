from dataclasses import dataclass
import math
from typing import Any, Iterable, Sequence

from tpch_monetdb.config import resolve_active_verify_scale_factors
from tpch_monetdb.tools.tpch.runtime_hygiene import classify_infra_failure


def _tail_text(text: str, max_chars: int = 1200) -> str:
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def _build_failure_detail(
    *,
    scale_factor: int,
    message: str,
    metrics: dict[str, Any] | None,
) -> str:
    parts = [f"scale_factor={scale_factor}"]
    if metrics is not None:
        detail = metrics.get("validation/failure_detail")
        if isinstance(detail, str) and detail:
            parts.append(detail)
    if message:
        parts.append(_tail_text(message))
    return "\n".join(parts)


@dataclass
class CorrectnessCheckSummary:
    success: bool
    message: str
    metrics: dict[str, Any] | None
    failed_scale_factor: int | None
    failure_code: str | None = None
    failure_detail: str | None = None


@dataclass(frozen=True)
class UnitValidationPlan:
    scope_query_ids: tuple[str, ...]
    scale_factors: tuple[int, ...]
    rollback_on_regression: bool


def aggregate_scope_runtime_seconds(runtime_by_query: dict[str, float]) -> float:
    """Aggregate a scope runtime map using geometric mean in seconds."""
    if not runtime_by_query:
        raise RuntimeError("runtime_by_query must not be empty")
    values = [runtime_by_query[qid] for qid in sorted(runtime_by_query)]
    for value in values:
        if not math.isfinite(value) or value <= 0:
            raise RuntimeError("runtime_by_query must contain positive finite values")
    return math.prod(values) ** (1.0 / len(values))


def validate_unit_correctness(
    run_tool: Any,
    scale_factors: Sequence[int],
    query_ids: Sequence[str],
    *,
    trace_mode: bool = False,
    optimize: bool = True,
    external_call: bool = True,
    fail_fast: bool = True,
    force_fresh_validation: bool = False,
) -> CorrectnessCheckSummary:
    """Validate correctness for a query unit or query batch."""
    return run_required_correctness_checks(
        run_tool,
        scale_factors,
        query_ids,
        trace_mode=trace_mode,
        optimize=optimize,
        external_call=external_call,
        fail_fast=fail_fast,
        force_fresh_validation=force_fresh_validation,
    )


def build_unit_validation_plan(
    *,
    query_id: str,
    scope_query_ids: Sequence[str] | None,
    written_files: Sequence[str],
    all_query_ids: Sequence[str],
    light_scale_factors: Sequence[int],
    full_scale_factors: Sequence[int],
) -> UnitValidationPlan:
    """Build the validation plan for one query unit or query stage."""
    resolved_scope_query_ids = (
        (query_id,) if scope_query_ids is None else tuple(str(item) for item in scope_query_ids)
    )
    touches_shared_runtime_scope = any(
        any(marker in path for marker in ("builder_impl", "loader_impl", "query_shared", "query_family_", "storage_plan", "args_parser"))
        for path in written_files
    )
    use_full_scope = touches_shared_runtime_scope or len(resolved_scope_query_ids) > 1
    return UnitValidationPlan(
        scope_query_ids=tuple(all_query_ids) if touches_shared_runtime_scope else resolved_scope_query_ids,
        scale_factors=tuple(full_scale_factors if use_full_scope else light_scale_factors),
        rollback_on_regression=True,
    )


def should_rollback_unit_regression(
    *,
    rt_before_s: float,
    rt_after_s: float,
    revert_on_regression: bool,
) -> bool:
    """Return whether a unit regression must be rolled back."""
    return bool(revert_on_regression and rt_after_s >= rt_before_s)


def required_validation_scale_factors(
    verify_sf_list: Iterable[int],
    benchmark_sf: int,
) -> list[int]:
    active_verify = resolve_active_verify_scale_factors(
        benchmark_sf=benchmark_sf,
        verify_sf_list=verify_sf_list,
    )
    return list(dict.fromkeys([*active_verify, int(benchmark_sf)]))


def run_required_correctness_checks(
    run_tool: Any,
    scale_factors: Sequence[int],
    query_ids: Sequence[str],
    *,
    trace_mode: bool = False,
    optimize: bool = True,
    external_call: bool = True,
    fail_fast: bool = True,
    force_fresh_validation: bool = False,
) -> CorrectnessCheckSummary:
    """Run correctness validation for every required scale factor."""
    last_message = ""
    last_metrics: dict[str, Any] | None = None
    qids = list(query_ids)
    failure_messages: list[str] = []
    failure_details: list[str] = []
    first_failure_sf: int | None = None
    first_failure_code: str | None = None
    first_failure_metrics: dict[str, Any] | None = None

    for sf in scale_factors:
        try:
            message, metrics = run_tool.run(
                scale_factor=sf,
                optimize=optimize,
                query_id=qids,
                trace_mode=trace_mode,
                external_call=external_call,
                force_fresh_validation=force_fresh_validation,
            )
        except Exception as exc:
            text = str(exc)
            infra_code = classify_infra_failure(text)
            failure_code = infra_code or "VALIDATION_EXCEPTION"
            failure_message = f"Validation raised for sf{sf}: {text}"
            failure_metrics = {
                "validation/correct": False,
                "validation/error": True,
                "validation/failure_code": failure_code,
                "validation/failure_detail": text,
            }
            failure_detail = f"scale_factor={sf}: {text}"
            if fail_fast:
                return CorrectnessCheckSummary(
                    success=False,
                    message=failure_message,
                    metrics=failure_metrics,
                    failed_scale_factor=sf,
                    failure_code=failure_code,
                    failure_detail=failure_detail,
                )
            if first_failure_sf is None:
                first_failure_sf = sf
                first_failure_code = failure_code
                first_failure_metrics = failure_metrics
            failure_messages.append(failure_message)
            failure_details.append(failure_detail)
            continue

        last_message = message
        last_metrics = metrics

        if metrics is None:
            infra_code = classify_infra_failure(message)
            failure_code = infra_code or "VALIDATION_NO_METRICS"
            failure_message = f"Validation returned no metrics for sf{sf}: {message}"
            failure_detail = _build_failure_detail(
                scale_factor=sf,
                message=f"run_tool returned no metrics\n{message}",
                metrics=None,
            )
            if fail_fast:
                return CorrectnessCheckSummary(
                    success=False,
                    message=failure_message,
                    metrics=None,
                    failed_scale_factor=sf,
                    failure_code=failure_code,
                    failure_detail=failure_detail,
                )
            if first_failure_sf is None:
                first_failure_sf = sf
                first_failure_code = failure_code
                first_failure_metrics = None
            failure_messages.append(failure_message)
            failure_details.append(failure_detail)
            continue

        if not metrics.get("validation/correct", False):
            infra_code = classify_infra_failure(message, metrics)
            failure_code = infra_code or "VALIDATION_INCORRECT"
            failure_message = f"Validation failed for sf{sf}: {message}"
            failure_detail = _build_failure_detail(
                scale_factor=sf,
                message=message if infra_code is not None else f"validation incorrect\n{message}",
                metrics=metrics,
            )
            if fail_fast:
                return CorrectnessCheckSummary(
                    success=False,
                    message=failure_message,
                    metrics=metrics,
                    failed_scale_factor=sf,
                    failure_code=failure_code,
                    failure_detail=failure_detail,
                )
            if first_failure_sf is None:
                first_failure_sf = sf
                first_failure_code = failure_code
                first_failure_metrics = metrics
            failure_messages.append(failure_message)
            failure_details.append(failure_detail)
            continue

    if failure_messages:
        return CorrectnessCheckSummary(
            success=False,
            message="\n".join(failure_messages),
            metrics=first_failure_metrics,
            failed_scale_factor=first_failure_sf,
            failure_code=first_failure_code,
            failure_detail="\n".join(failure_details),
        )

    return CorrectnessCheckSummary(
        success=True,
        message=last_message,
        metrics=last_metrics,
        failed_scale_factor=None,
    )
