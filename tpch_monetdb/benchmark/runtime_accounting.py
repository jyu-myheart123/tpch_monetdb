"""Shared runtime accounting helpers for Generated TPC-H phase9.

Single source of truth for:
- runtime_schema_version constant
- Query timing parsing (precedence, fallback, index-mismatch failure)
- Ingest timing parsing
- Runtime sample normalization (median, first, lazy-build)
- Lazy-build suspicion gate
"""

import logging
import re
import statistics
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from tpch_monetdb.tools.tpch.runtime_hygiene import classify_infra_failure

logger = logging.getLogger(__name__)

RUNTIME_SCHEMA_VERSION = "phase14_no_csv_kernel_runtime_v1"
QUERY_RUNTIME_METRIC_KIND = "query_e2e_ms"
QUERY_MATERIALIZATION_RUNTIME_METRIC_KIND = "query_materialization_ms"
OFFICIAL_QUERY_RUNTIME_METRIC_KINDS = frozenset(
    {QUERY_RUNTIME_METRIC_KIND, QUERY_MATERIALIZATION_RUNTIME_METRIC_KIND}
)
KERNEL_RUNTIME_METRIC_KIND = "kernel_ms"
OPTIMIZATION_RUNTIME_METRIC_KIND = KERNEL_RUNTIME_METRIC_KIND
OPTIMIZATION_RUNTIME_METRIC_KINDS = frozenset({OPTIMIZATION_RUNTIME_METRIC_KIND})

LAZY_BUILD_RATIO_THRESHOLD = 3.0
LAZY_BUILD_ABSOLUTE_MS_THRESHOLD = 1.0

WARMUP_RUNS = 1
MEASURED_RUNS = 3

RUNTIME_COLD_START_MIN_TIMEOUT_S = 180
RUNTIME_COLD_START_MAX_TIMEOUT_S = 1800
RUNTIME_WARM_QUERY_MIN_TIMEOUT_S = 10
RUNTIME_WARM_QUERY_FALLBACK_TIMEOUT_S = 60
RUNTIME_WARM_QUERY_MAX_TIMEOUT_S = 600
RUNTIME_TRACE_MIN_TIMEOUT_S = 240
RUNTIME_PMU_MIN_TIMEOUT_S = 240


@dataclass(frozen=True)
class RuntimeTimeoutPolicy:
    cold_start_timeout_s: int
    warm_query_timeout_s: int
    trace_timeout_s: int
    pmu_timeout_s: int
    scale_factor: float
    num_queries: int
    baseline_runtime_ms: Optional[float] = None

    def to_provenance(self) -> dict[str, Any]:
        return {
            "cold_start_timeout_s": self.cold_start_timeout_s,
            "warm_query_timeout_s": self.warm_query_timeout_s,
            "trace_timeout_s": self.trace_timeout_s,
            "pmu_timeout_s": self.pmu_timeout_s,
            "scale_factor": self.scale_factor,
            "num_queries": self.num_queries,
            "baseline_runtime_ms": self.baseline_runtime_ms,
        }


@dataclass(frozen=True)
class RuntimeExecutionFailure:
    failure_code: str
    detail: str


class RuntimeExecutionFailureError(RuntimeError):
    def __init__(self, failure: RuntimeExecutionFailure) -> None:
        self.failure = failure
        super().__init__(f"[ERROR:{failure.failure_code}] {failure.detail}")
        return None


_EXIT_STATUS_RE = re.compile(r"exit_code:\s*(-?\d+)\s+signal:\s*(\d+)")


def build_runtime_timeout_policy(
    scale_factor: int | float,
    *,
    num_queries: int = 1,
    baseline_runtime_ms: Optional[float] = None,
) -> RuntimeTimeoutPolicy:
    """Build cold-start and warm-query timeout budgets for generated runtimes."""
    sf = max(float(scale_factor), 1.0)
    query_count = max(int(num_queries), 1)
    cold_estimate = int(60 + (sf * 15 * query_count))
    cold_timeout_s = min(
        RUNTIME_COLD_START_MAX_TIMEOUT_S,
        max(RUNTIME_COLD_START_MIN_TIMEOUT_S, cold_estimate),
    )
    if baseline_runtime_ms is not None and baseline_runtime_ms > 0:
        warm_estimate = int((baseline_runtime_ms * 50.0 / 1000.0) + 5.0)
    else:
        warm_estimate = RUNTIME_WARM_QUERY_FALLBACK_TIMEOUT_S
    warm_timeout_s = min(
        RUNTIME_WARM_QUERY_MAX_TIMEOUT_S,
        max(RUNTIME_WARM_QUERY_MIN_TIMEOUT_S, warm_estimate),
    )
    return RuntimeTimeoutPolicy(
        cold_start_timeout_s=cold_timeout_s,
        warm_query_timeout_s=warm_timeout_s,
        trace_timeout_s=max(RUNTIME_TRACE_MIN_TIMEOUT_S, cold_timeout_s),
        pmu_timeout_s=max(RUNTIME_PMU_MIN_TIMEOUT_S, warm_timeout_s * 2),
        scale_factor=sf,
        num_queries=query_count,
        baseline_runtime_ms=baseline_runtime_ms,
    )


def detect_runtime_execution_failure(
    response: str,
    stdout: str,
    stderr: str,
) -> RuntimeExecutionFailure | None:
    """Detect runner/child failures before any timing parser reads stale output."""
    text = "\n".join(part for part in (response, stdout, stderr) if part)
    failure_code = classify_infra_failure(text)
    if failure_code is not None:
        return RuntimeExecutionFailure(
            failure_code=failure_code,
            detail=_trim_runtime_failure_detail(text),
        )
    for exit_code, signal_code in _EXIT_STATUS_RE.findall(response or ""):
        if int(exit_code) != 0 or int(signal_code) != 0:
            return RuntimeExecutionFailure(
                failure_code="RUNNER_NONZERO_EXIT",
                detail=_trim_runtime_failure_detail(text),
            )
    return None


def raise_for_runtime_execution_failure(
    response: str,
    stdout: str,
    stderr: str,
) -> None:
    """Raise when runtime output contains infrastructure failure evidence."""
    failure = detect_runtime_execution_failure(response, stdout, stderr)
    if failure is not None:
        raise RuntimeExecutionFailureError(failure)
    return None


def _trim_runtime_failure_detail(text: str) -> str:
    detail = text.strip()
    if len(detail) <= 4000:
        return detail
    return detail[-4000:]


@dataclass
class QueryTimingResult:
    primary_runtime_ms: float
    kernel_runtime_ms: Optional[float]
    query_runtime_ms: Optional[float]
    fallback_reason: Optional[str]
    primary_metric_kind: str = QUERY_RUNTIME_METRIC_KIND


def _check_primary_metric_kind(primary_metric_kind: str) -> None:
    """Validate the requested primary timing metric."""
    if primary_metric_kind not in {
        QUERY_RUNTIME_METRIC_KIND,
        KERNEL_RUNTIME_METRIC_KIND,
    }:
        raise ValueError(
            f"Unsupported primary_metric_kind={primary_metric_kind!r}; "
            f"expected {QUERY_RUNTIME_METRIC_KIND!r} or {KERNEL_RUNTIME_METRIC_KIND!r}."
        )
    return None


@dataclass
class QuerySamples:
    measured_runs_ms: List[float]
    kernel_runs_ms: List[float] = field(default_factory=list)

    @property
    def first_query_ms(self) -> Optional[float]:
        return self.measured_runs_ms[0] if self.measured_runs_ms else None

    @property
    def median_query_ms(self) -> Optional[float]:
        if not self.measured_runs_ms:
            return None
        return statistics.median(self.measured_runs_ms)

    @property
    def first_kernel_ms(self) -> Optional[float]:
        return self.kernel_runs_ms[0] if self.kernel_runs_ms else None

    @property
    def median_kernel_ms(self) -> Optional[float]:
        if not self.kernel_runs_ms:
            return None
        return statistics.median(self.kernel_runs_ms)


@dataclass
class IngestTimingResult:
    ingest_ms: float
    load_ms: Optional[float]
    build_ms: Optional[float]


@dataclass(frozen=True)
class DerivedBespokeIngestMetrics:
    row_count: int
    metric_count: int
    rows_per_sec: float
    metrics_per_sec: float


def parse_query_timing(
    stdout: str,
    stderr: str,
    query_id: str,
    index: Optional[int] = None,
    primary_metric_kind: str = QUERY_RUNTIME_METRIC_KIND,
) -> QueryTimingResult:
    """Parse Generated TPC-H query timing from stdout+stderr.

    Query validation defaults to Query ms. Optimization can explicitly request
    kernel_ms, which must come from Execution ms/no-output measurement.
    When index is given, uses positional match; count mismatch raises ValueError.
    """
    _check_primary_metric_kind(primary_metric_kind)
    text = stdout + "\n" + stderr
    query_matches = re.findall(r"(\d+)\s*\|\s*Query ms:\s*([\d.]+)", text)
    kernel_matches = re.findall(r"(\d+)\s*\|\s*Execution ms:\s*([\d.]+)", text)

    fallback_reason: Optional[str] = None

    if index is not None:
        if primary_metric_kind == KERNEL_RUNTIME_METRIC_KIND:
            if index >= len(kernel_matches):
                raise ValueError(
                    f"No no-CSV kernel timing output found for query {query_id}[{index}]. "
                    f"Expected format: 'N | Execution ms: X'"
                )
            primary_ms = float(kernel_matches[index][1])
            kernel_ms = primary_ms
            query_ms = float(query_matches[index][1]) if index < len(query_matches) else None
        elif index < len(query_matches):
            primary_ms = float(query_matches[index][1])
            query_ms = primary_ms
            kernel_ms = float(kernel_matches[index][1]) if index < len(kernel_matches) else None
        elif index < len(kernel_matches):
            fallback_reason = "kernel_fallback"
            kernel_ms = float(kernel_matches[index][1])
            raise ValueError(
                f"Official Query ms missing for query {query_id}[{index}]; "
                f"Execution ms={kernel_ms:.6f} is diagnostic kernel_ms only."
            )
        else:
            raise ValueError(
                f"Timing index {index} out of range for query {query_id}. "
                f"Found {len(query_matches)} Query ms and "
                f"{len(kernel_matches)} Execution ms entries, "
                f"expected at least {index + 1}."
            )
    else:
        if primary_metric_kind == KERNEL_RUNTIME_METRIC_KIND:
            if not kernel_matches:
                raise ValueError(
                    f"No no-CSV kernel timing output found for query {query_id}. "
                    "Expected format: 'N | Execution ms: X'"
                )
            primary_ms = float(kernel_matches[0][1])
            kernel_ms = primary_ms
            query_ms = float(query_matches[0][1]) if query_matches else None
        elif query_matches:
            primary_ms = float(query_matches[0][1])
            query_ms = primary_ms
            kernel_ms = float(kernel_matches[0][1]) if kernel_matches else None
        elif kernel_matches:
            fallback_reason = "kernel_fallback"
            kernel_ms = float(kernel_matches[0][1])
            raise ValueError(
                f"Official Query ms missing for query {query_id}; "
                f"Execution ms={kernel_ms:.6f} is diagnostic kernel_ms only."
            )
        else:
            raise ValueError(
                f"No official timing output found for query {query_id}. "
                "Expected format: 'N | Query ms: X'"
            )

    return QueryTimingResult(
        primary_runtime_ms=primary_ms,
        kernel_runtime_ms=kernel_ms,
        query_runtime_ms=query_ms,
        fallback_reason=fallback_reason,
        primary_metric_kind=primary_metric_kind,
    )


def parse_query_timing_by_id(
    stdout: str,
    query_ids: List[str],
    primary_metric_kind: str = QUERY_RUNTIME_METRIC_KIND,
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, str]]:
    """Parse primary timings keyed by query_id string (validator path).

    Returns (primary_map, kernel_map, fallback_map).
    """
    _check_primary_metric_kind(primary_metric_kind)
    impl_map: Dict[str, float] = {}
    kernel_map: Dict[str, float] = {}
    fallback_map: Dict[str, str] = {}

    query_matches = re.findall(r"(\d+)\s*\|\s*Query ms:\s*([\d.]+)", stdout)
    kernel_matches = re.findall(r"(\d+)\s*\|\s*Execution ms:\s*([\d.]+)", stdout)

    query_by_run: Dict[int, float] = {int(r): float(t) for r, t in query_matches}
    kernel_by_run: Dict[int, float] = {int(r): float(t) for r, t in kernel_matches}

    for query_id in query_ids:
        try:
            qid_int = int(query_id)
        except (TypeError, ValueError):
            continue
        if primary_metric_kind == KERNEL_RUNTIME_METRIC_KIND:
            if qid_int in kernel_by_run:
                impl_map[query_id] = kernel_by_run[qid_int]
            elif qid_int in query_by_run:
                fallback_map[query_id] = "no_csv_kernel_runtime_missing"
                logger.warning(
                    "Generated TPC-H %s: no-CSV Execution ms missing; Query ms is full-CSV diagnostic only",
                    query_id,
                )
        else:
            if qid_int in query_by_run:
                impl_map[query_id] = query_by_run[qid_int]
            elif qid_int in kernel_by_run:
                fallback_map[query_id] = "official_query_runtime_missing"
                logger.warning(
                    "Generated TPC-H %s: Query ms missing; Execution ms is diagnostic kernel_ms only",
                    query_id,
                )
        if qid_int in kernel_by_run:
            kernel_map[query_id] = kernel_by_run[qid_int]

    return impl_map, kernel_map, fallback_map


def parse_query_timing_by_id_with_metric_kind(
    stdout: str,
    query_ids: List[str],
    primary_metric_kind: str = QUERY_RUNTIME_METRIC_KIND,
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, str], Dict[str, str]]:
    """Parse query timings and attach the metric kind of each primary runtime."""
    impl_map, kernel_map, fallback_map = parse_query_timing_by_id(
        stdout,
        query_ids,
        primary_metric_kind=primary_metric_kind,
    )
    metric_kind_map = {query_id: primary_metric_kind for query_id in impl_map}
    if primary_metric_kind == QUERY_RUNTIME_METRIC_KIND:
        metric_kind_map.update(
            {query_id: KERNEL_RUNTIME_METRIC_KIND for query_id in fallback_map}
        )
    return impl_map, kernel_map, fallback_map, metric_kind_map


def parse_ingest_timing(
    stdout: str,
    stderr: str,
) -> IngestTimingResult:
    """Parse Generated TPC-H ingest timing from stdout+stderr.

    Precedence: Ingest ms > Load ms + Build ms.
    """
    text = stdout + "\n" + stderr
    ingest_match = re.search(r"Ingest ms:\s*([\d.]+)", text)
    load_match = re.search(r"Load ms:\s*([\d.]+)", text)
    build_match = re.search(r"Build ms:\s*([\d.]+)", text)

    load_ms = float(load_match.group(1)) if load_match else None
    build_ms = float(build_match.group(1)) if build_match else None

    if ingest_match:
        ingest_ms = float(ingest_match.group(1))
    elif load_ms is not None and build_ms is not None:
        ingest_ms = load_ms + build_ms
    else:
        raise ValueError(
            "No ingest timing found. Expected 'Ingest ms: X' or 'Load ms: X' + 'Build ms: Y'"
        )

    return IngestTimingResult(ingest_ms=ingest_ms, load_ms=load_ms, build_ms=build_ms)


def parse_ingest_timing_from_text(
    text: str,
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Parse ingest timing from a single text (stdout or stderr, validator path).

    Returns (ingest_ms, load_ms, build_ms); ingest_ms is None if unparseable.
    """
    ingest_match = re.search(r"Ingest ms:\s*([\d.]+)", text)
    load_match = re.search(r"Load ms:\s*([\d.]+)", text)
    build_match = re.search(r"Build ms:\s*([\d.]+)", text)

    load_ms = float(load_match.group(1)) if load_match else None
    build_ms = float(build_match.group(1)) if build_match else None
    ingest_ms: Optional[float] = None

    if ingest_match:
        ingest_ms = float(ingest_match.group(1))
    elif load_ms is not None and build_ms is not None:
        ingest_ms = load_ms + build_ms

    return ingest_ms, load_ms, build_ms


def compute_median(values: List[float]) -> float:
    if not values:
        raise ValueError("Cannot compute median of empty list")
    return statistics.median(values)


INGEST_FORMAL_REQUIRED_BASELINE_FIELDS = frozenset([
    "baseline_ingest_ms",
    "baseline_ingest_rows_per_sec",
    "baseline_ingest_metrics_per_sec",
    "workers",
])

INGEST_FORMAL_REQUIRED_BESPOKE_FIELDS = frozenset([
    "generated_tpch_ingest_ms",
    "generated_tpch_ingest_rows_per_sec",
    "generated_tpch_ingest_metrics_per_sec",
])


def check_ingest_completeness(
    baseline_ingest_ms: Optional[float],
    baseline_ingest_rows_per_sec: Optional[float],
    baseline_ingest_metrics_per_sec: Optional[float],
    baseline_workers: Optional[int],
    bespoke_ingest_ms: Optional[float],
    bespoke_ingest_rows_per_sec: Optional[float],
    bespoke_ingest_metrics_per_sec: Optional[float],
) -> Tuple[bool, List[str]]:
    """Check that all required ingest fields are present and workers=1.

    Returns (is_complete, missing_or_invalid_fields).
    """
    missing: List[str] = []
    if baseline_ingest_ms is None:
        missing.append("baseline_ingest_ms")
    if baseline_ingest_rows_per_sec is None:
        missing.append("baseline_ingest_rows_per_sec")
    if baseline_ingest_metrics_per_sec is None:
        missing.append("baseline_ingest_metrics_per_sec")
    if baseline_workers is None:
        missing.append("workers")
    elif baseline_workers != 1:
        missing.append(f"workers={baseline_workers} (must be 1)")
    if bespoke_ingest_ms is None:
        missing.append("generated_tpch_ingest_ms")
    if bespoke_ingest_rows_per_sec is None:
        missing.append("generated_tpch_ingest_rows_per_sec")
    if bespoke_ingest_metrics_per_sec is None:
        missing.append("generated_tpch_ingest_metrics_per_sec")
    return len(missing) == 0, missing


def derive_bespoke_ingest_metrics(
    scale_factor: int,
    bespoke_ingest_ms: Optional[float],
) -> Tuple[Optional[DerivedBespokeIngestMetrics], List[str]]:
    """Return no derived ingest metrics after legacy ingest removal."""
    missing: List[str] = []
    if bespoke_ingest_ms is None:
        missing.append("generated_tpch_ingest_ms")
        return None, missing
    if bespoke_ingest_ms <= 0:
        missing.append(f"generated_tpch_ingest_ms={bespoke_ingest_ms} (must be > 0)")
        return None, missing
    _ = scale_factor
    missing.append("legacy ingest throughput derivation removed")
    return None, missing


def is_lazy_build_suspected(samples: QuerySamples) -> bool:
    """Return True if first measured run is suspiciously slower than median.

    Rules (either condition triggers):
    - first_query_ms > LAZY_BUILD_RATIO_THRESHOLD * median_query_ms
    - first_query_ms - median_query_ms > LAZY_BUILD_ABSOLUTE_MS_THRESHOLD
    """
    first = samples.first_query_ms
    median = samples.median_query_ms
    if first is None or median is None:
        return False
    if median <= 0:
        return False
    ratio_exceeded = first > LAZY_BUILD_RATIO_THRESHOLD * median
    absolute_exceeded = (first - median) > LAZY_BUILD_ABSOLUTE_MS_THRESHOLD
    return ratio_exceeded or absolute_exceeded
