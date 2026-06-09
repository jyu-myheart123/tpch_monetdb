from __future__ import annotations

import subprocess
from dataclasses import dataclass

from tpch_monetdb.tools.tpch.runtime_hygiene import classify_infra_failure


CONTEXT_TOO_LARGE_MARKERS: tuple[str, ...] = (
    "[ERROR:CONTEXT_TOO_LARGE]",
    "413 Request Entity Too Large",
    "Request Entity Too Large",
    "context length",
    "context_length_exceeded",
    "maximum context length",
    "blocking threshold",
)

TRANSIENT_LLM_MARKERS: tuple[str, ...] = (
    "Connection error",
    "APIConnectionError",
    "APITimeoutError",
    "RateLimitError",
    "temporarily unavailable",
)

PERSISTENT_CONNECTION_MARKERS: tuple[str, ...] = (
    "Connection reset by peer",
    "Connection refused",
    "request body likely exceeds server proxy limit",
)

REACTIVE_COMPACT_FAILURE_CODES: frozenset[str] = frozenset({
    "CONTEXT_TOO_LARGE",
})

REACTIVE_COMPACT_MARKERS: tuple[str, ...] = (
    "prompt too long",
    "body too large",
    "request body likely exceeds server proxy limit",
    "request body too large",
    "request body exceeds",
    "request too large",
    "maximum context",
)


def classify_model_failure(text: str) -> str | None:
    if any(marker in text for marker in CONTEXT_TOO_LARGE_MARKERS):
        return "CONTEXT_TOO_LARGE"
    if any(marker in text for marker in PERSISTENT_CONNECTION_MARKERS):
        return "PERSISTENT_CONNECTION_ERROR"
    if any(marker in text for marker in TRANSIENT_LLM_MARKERS):
        return "TRANSIENT_LLM_FAILURE"
    return None


def should_reactive_compact(exc: BaseException) -> bool:
    """Return whether a prompt failure should trigger local compact and retry."""
    text = str(exc)
    failure_code = classify_model_failure(text)
    if failure_code in REACTIVE_COMPACT_FAILURE_CODES:
        return True
    lowered = text.lower()
    return any(marker in lowered for marker in REACTIVE_COMPACT_MARKERS)


@dataclass(frozen=True)
class OuterLoopSupervisorDecision:
    outcome: str
    action: str
    failure_code: str | None
    failure_detail: str
    should_retry: bool
    should_cleanup_runtime: bool


def classify_optimization_result(
    result: subprocess.CompletedProcess,
    *,
    summary_found: bool,
    retry_count: int,
    retry_budget: int,
) -> OuterLoopSupervisorDecision:
    """Classify an optimization subprocess result into the outer-loop action."""
    text = (result.stdout or "") + "\n" + (result.stderr or "")
    model_code = classify_model_failure(text)
    if model_code == "CONTEXT_TOO_LARGE":
        return OuterLoopSupervisorDecision(
            outcome="failed",
            action="failed",
            failure_code="CONTEXT_TOO_LARGE",
            failure_detail="model request/context exceeded provider limit",
            should_retry=False,
            should_cleanup_runtime=False,
        )
    if model_code == "PERSISTENT_CONNECTION_ERROR":
        return OuterLoopSupervisorDecision(
            outcome="failed",
            action="failed",
            failure_code="PERSISTENT_CONNECTION_ERROR",
            failure_detail="request body likely exceeds server proxy limit — do not retry",
            should_retry=False,
            should_cleanup_runtime=False,
        )
    if model_code == "TRANSIENT_LLM_FAILURE":
        return OuterLoopSupervisorDecision(
            outcome="failed",
            action="retry" if retry_count < retry_budget else "failed",
            failure_code="TRANSIENT_LLM_FAILURE",
            failure_detail="transient LLM/provider failure",
            should_retry=retry_count < retry_budget,
            should_cleanup_runtime=False,
        )
    infra_code = classify_infra_failure(text)
    if infra_code is not None:
        return OuterLoopSupervisorDecision(
            outcome="failed",
            action="retry" if retry_count < retry_budget else "failed",
            failure_code=infra_code,
            failure_detail=infra_code,
            should_retry=retry_count < retry_budget,
            should_cleanup_runtime=True,
        )
    if result.returncode != 0:
        return OuterLoopSupervisorDecision(
            outcome="failed",
            action="retry" if retry_count < retry_budget else "failed",
            failure_code="PHASE_RETRY_EXHAUSTED",
            failure_detail=f"returncode={result.returncode}",
            should_retry=retry_count < retry_budget,
            should_cleanup_runtime=False,
        )
    if not summary_found:
        return OuterLoopSupervisorDecision(
            outcome="failed",
            action="failed",
            failure_code="PHASE_SUMMARY_MISSING",
            failure_detail="optimization summary missing",
            should_retry=False,
            should_cleanup_runtime=False,
        )
    return OuterLoopSupervisorDecision(
        outcome="success",
        action="continue",
        failure_code=None,
        failure_detail="",
        should_retry=False,
        should_cleanup_runtime=False,
    )
