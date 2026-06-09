from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Sequence


def build_instrumentation_prompt_metadata(
    qids: list[str],
    *,
    active_unit_id: str | None = None,
    active_unit_kind: str | None = None,
    active_unit_files: list[str] | None = None,
    hardware_counter_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {"active_query_ids": list(qids)}
    if active_unit_id is not None:
        metadata["active_unit_id"] = active_unit_id
    if active_unit_kind is not None:
        metadata["active_unit_kind"] = active_unit_kind
    if active_unit_files:
        metadata["active_unit_files"] = list(active_unit_files)
    if (
        active_unit_id is not None
        or active_unit_kind is not None
        or active_unit_files
    ):
        metadata["active_unit_query_ids"] = list(qids)
    if hardware_counter_summary:
        metadata["hardware_counter_summary"] = dict(hardware_counter_summary)
    return metadata


def build_instrumentation_feedback_prompt(qids: list[str], trace_mode: bool) -> str:
    mode = "enabled" if trace_mode else "disabled"
    return (
        "Validation failed after instrumentation changes with trace_mode="
        f"{mode} for queries {', '.join(qids)}. Fix instrumentation only. "
        "Do not change query semantics, storage layout, or unrelated logic. "
        "Once validation passes, stop immediately."
    )


def build_trace_evidence_feedback_prompt(
    qids: list[str],
    summary: TraceEvidenceSummary,
) -> str:
    return (
        f"Trace evidence insufficient for queries {', '.join(summary.insufficient_qids)}. "
        "Add more PROFILE scopes to distinguish scan/filter/aggregate/output sub-paths. "
        "Do not change query semantics, storage layout, or algorithm logic. "
        "Only modify instrumentation code. Once trace output meets minimum coverage, stop."
    )


@dataclass(frozen=True)
class InstrumentationPolicy:
    smoke_scale_factors: tuple[int, ...]
    full_scale_factors: tuple[int, ...]
    repair_attempts: int = 2
    batch_size: int = 3
    evidence_repair_attempts: int = 3


@dataclass(frozen=True)
class TraceEvidenceSummary:
    qids: tuple[str, ...]
    sufficient: bool
    message: str
    degraded: bool = False
    insufficient_qids: tuple[str, ...] = ()
    failure_code: str | None = None
    raw_execution_ok: bool = True
    trace_file_present: bool = True
    profile_count_by_query: dict[str, int] | None = None


def build_instrumentation_policy(
    required_validation_sf_list: Sequence[int],
) -> InstrumentationPolicy:
    full = tuple(required_validation_sf_list)
    if not full:
        raise RuntimeError("required_validation_sf_list must not be empty")
    return InstrumentationPolicy(
        smoke_scale_factors=(min(full),),
        full_scale_factors=full,
    )


async def check_instrumentation_smoke(
    *,
    qids: list[str],
    policy: InstrumentationPolicy,
    max_turns: int,
    check_correctness_fn: Callable[[list[str], bool, tuple[int, ...]], Awaitable[bool]],
    exec_fn: Callable[..., Awaitable[Any]],
) -> None:
    attempts = 0
    while not await check_correctness_fn(qids, False, policy.smoke_scale_factors):
        attempts += 1
        if attempts > policy.repair_attempts:
            raise RuntimeError(
                f"Instrumentation smoke correctness failed for qids={qids}"
            )
        await exec_fn(
            build_instrumentation_feedback_prompt(qids, trace_mode=False),
            "Fix Instrumentation Smoke",
            max_turns=max_turns,
            tool_profile="optimization_instrumentation",
            prompt_metadata=build_instrumentation_prompt_metadata(qids),
        )


async def check_trace_evidence_and_feedback(
    *,
    qids: list[str],
    policy: InstrumentationPolicy,
    summarize_trace_fn: Callable[[list[str]], TraceEvidenceSummary],
    exec_fn: Callable[..., Awaitable[Any]],
    max_turns: int,
) -> TraceEvidenceSummary:
    attempts = 0
    while True:
        summary = summarize_trace_fn(qids)
        if summary.sufficient:
            return summary
        attempts += 1
        if attempts > policy.evidence_repair_attempts:
            return TraceEvidenceSummary(
                qids=summary.qids,
                sufficient=False,
                message=summary.message,
                degraded=True,
                insufficient_qids=summary.insufficient_qids,
                failure_code=summary.failure_code,
                raw_execution_ok=summary.raw_execution_ok,
                trace_file_present=summary.trace_file_present,
                profile_count_by_query=summary.profile_count_by_query,
            )
        await exec_fn(
            build_trace_evidence_feedback_prompt(qids, summary),
            "Fix Trace Evidence Coverage",
            max_turns=max_turns,
            tool_profile="optimization_instrumentation",
            prompt_metadata=build_instrumentation_prompt_metadata(qids),
        )


async def check_trace_mode_smoke(
    *,
    qids: list[str],
    policy: InstrumentationPolicy,
    max_turns: int,
    check_correctness_fn: Callable[[list[str], bool, tuple[int, ...]], Awaitable[bool]],
    exec_fn: Callable[..., Awaitable[Any]],
) -> None:
    """Run a bounded trace-enabled correctness smoke gate after trace file setup."""
    attempts = 0
    smoke_qids = qids[: policy.batch_size]
    while not await check_correctness_fn(smoke_qids, True, policy.smoke_scale_factors):
        attempts += 1
        if attempts > policy.repair_attempts:
            raise RuntimeError(
                f"Instrumentation trace-mode smoke correctness failed for qids={smoke_qids}"
            )
        await exec_fn(
            build_instrumentation_feedback_prompt(smoke_qids, trace_mode=True),
            "Fix Trace-Mode Instrumentation Smoke",
            max_turns=max_turns,
            tool_profile="optimization_instrumentation",
            prompt_metadata=build_instrumentation_prompt_metadata(smoke_qids),
        )
    return None
