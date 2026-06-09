from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from tpch_monetdb.utils.pipeline_contracts import raise_pipeline_contract_error

FORBIDDEN_FINAL_PATH_TOKENS: tuple[str, ...] = (
    "query_shared_instrumented",
    "_instrumented(",
    "execute_single_groupby_instrumented",
    "execute_cpu_max_groupby_instrumented",
    "execute_double_groupby_instrumented",
    "execute_high_cpu_instrumented",
)


class MeasurementKind(StrEnum):
    """Classify runtime measurements so incompatible benchmark evidence is not mixed."""

    FIXED_VALIDATION = "fixed_validation"
    EXACT_INSTANTIATION = "exact_instantiation"
    SCALE_SWEEP = "scale_sweep"
    REPETITION = "repetition"


class MeasurementShapeStatus(StrEnum):
    """Classify whether measurement row-shape metadata is available."""

    KNOWN = "known"
    ESTIMATED = "estimated"
    UNKNOWN = "unknown"


class EvidenceStatus(StrEnum):
    """Normalize one piece of query evidence into a machine-checkable status."""

    MISSING = "missing"
    PRESENT = "present"
    PASS = "pass"
    FAIL = "fail"
    DEFERRED = "deferred"


class PipelineEvidenceStage(StrEnum):
    """Identify which pipeline stage is consuming the evidence ledger."""

    BASE_PROMOTION = "base_promotion"
    OPTIMIZATION_FINAL = "optimization_final"


@dataclass(frozen=True)
class QueryMeasurementRecord:
    """Typed runtime measurement for one query, engine, and measurement kind."""

    query_id: str
    engine: str
    measurement_kind: MeasurementKind | str
    runtime_ms: float
    instantiation_id: str | None = None
    args_string: str | None = None
    scale_factor: int | None = None
    row_count: int | None = None
    output_row_count: int | None = None
    query_file_sha256: str | None = None
    measurement_shape_status: MeasurementShapeStatus | str = MeasurementShapeStatus.UNKNOWN
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the measurement record."""
        measurement_kind = (
            self.measurement_kind.value
            if isinstance(self.measurement_kind, MeasurementKind)
            else str(self.measurement_kind)
        )
        shape_status = (
            self.measurement_shape_status.value
            if isinstance(self.measurement_shape_status, MeasurementShapeStatus)
            else str(self.measurement_shape_status)
        )
        return {
            "query_id": self.query_id,
            "engine": self.engine,
            "measurement_kind": measurement_kind,
            "runtime_ms": self.runtime_ms,
            "instantiation_id": self.instantiation_id,
            "args_string": self.args_string,
            "scale_factor": self.scale_factor,
            "row_count": self.row_count,
            "output_row_count": self.output_row_count,
            "query_file_sha256": self.query_file_sha256,
            "measurement_shape_status": shape_status,
            "provenance": dict(self.provenance),
        }


@dataclass(frozen=True)
class QueryEvidence:
    """Host-owned evidence state for one query in the TPC-H MonetDB pipeline."""

    query_id: str
    is_critical: bool
    requires_vectorization: bool
    requires_pmu: bool
    deferred_obligations: tuple[str, ...] = ()
    vector_contract_status: EvidenceStatus = EvidenceStatus.MISSING
    correctness_status: EvidenceStatus = EvidenceStatus.MISSING
    final_path_status: EvidenceStatus = EvidenceStatus.MISSING
    vectorization_status: EvidenceStatus = EvidenceStatus.MISSING
    pmu_status: EvidenceStatus = EvidenceStatus.MISSING
    runtime_status: EvidenceStatus = EvidenceStatus.MISSING
    todo_status: EvidenceStatus = EvidenceStatus.MISSING
    failures: tuple[str, ...] = ()
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PipelineEvidenceLedger:
    """Single source of truth for TPC-H MonetDB objective and promotion decisions."""

    objective_id: str
    stage: PipelineEvidenceStage | str
    query_evidence: dict[str, QueryEvidence]
    measurement_records: tuple[QueryMeasurementRecord, ...] = ()
    failures: tuple[str, ...] = ()
    failure_route: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable ledger payload for summaries and audits."""
        return {
            "objective_id": self.objective_id,
            "stage": (
                self.stage.value
                if isinstance(self.stage, PipelineEvidenceStage)
                else str(self.stage)
            ),
            "query_evidence": {
                qid: {
                    "query_id": evidence.query_id,
                    "is_critical": evidence.is_critical,
                    "requires_vectorization": evidence.requires_vectorization,
                    "requires_pmu": evidence.requires_pmu,
                    "deferred_obligations": list(evidence.deferred_obligations),
                    "vector_contract_status": evidence.vector_contract_status.value,
                    "correctness_status": evidence.correctness_status.value,
                    "final_path_status": evidence.final_path_status.value,
                    "vectorization_status": evidence.vectorization_status.value,
                    "pmu_status": evidence.pmu_status.value,
                    "runtime_status": evidence.runtime_status.value,
                    "todo_status": evidence.todo_status.value,
                    "failures": list(evidence.failures),
                    "details": dict(evidence.details),
                }
                for qid, evidence in self.query_evidence.items()
            },
            "measurement_records": [
                record.to_dict() for record in self.measurement_records
            ],
            "failures": list(self.failures),
            "failure_route": self.failure_route,
        }


def build_pipeline_evidence_ledger(
    *,
    workspace_path: Path,
    workload_objective: dict[str, Any],
    stage: PipelineEvidenceStage | str = PipelineEvidenceStage.OPTIMIZATION_FINAL,
    final_runtime_ms_by_query: dict[str, float] | None = None,
    baseline_runtime_ms_by_query: dict[str, float] | None = None,
    compiler_vectorization_summary: dict[str, Any] | None = None,
    hardware_counter_summary: dict[str, Any] | None = None,
    todo_reconciliation: dict[str, Any] | None = None,
    measurement_records: tuple[QueryMeasurementRecord, ...] = (),
) -> PipelineEvidenceLedger:
    """Build the host-owned ledger from objective, code, benchmark, vector, and PMU evidence."""
    normalized_stage = _normalize_stage(stage)
    objective_id = str(workload_objective.get("objective_id") or "")
    query_ids = [str(qid) for qid in workload_objective.get("query_ids", [])]
    critical_query_ids = {
        str(qid) for qid in workload_objective.get("critical_query_ids", [])
    }
    target_map = {
        str(qid): dict(target or {})
        for qid, target in dict(
            workload_objective.get("critical_query_targets", {}) or {}
        ).items()
    }
    query_evidence: dict[str, QueryEvidence] = {}

    for query_id in query_ids:
        target = target_map.get(query_id, {})
        failures: list[str] = []
        details: dict[str, Any] = {}
        is_critical = query_id in critical_query_ids
        requires_vectorization = target.get("requires_vectorization") is True
        requires_pmu = target.get("requires_pmu") is True
        deferred_obligations = build_deferred_obligations(
            stage=normalized_stage,
            is_critical=is_critical,
            requires_vectorization=requires_vectorization,
            requires_pmu=requires_pmu,
        )
        vector_contract_status = evaluate_vector_contract_status(
            requires_vectorization=requires_vectorization,
            stage=normalized_stage,
            deferred_obligations=deferred_obligations,
            details=details,
        )

        final_path_status = evaluate_final_path_status(
            workspace_path=workspace_path,
            query_id=query_id,
            failures=failures,
            details=details,
        )
        runtime_status = evaluate_runtime_status(
            query_id=query_id,
            is_critical=is_critical,
            final_runtime_ms_by_query=final_runtime_ms_by_query or {},
            baseline_runtime_ms_by_query=baseline_runtime_ms_by_query or {},
            failures=failures,
            details=details,
        )
        vectorization_status = evaluate_vectorization_status(
            query_id=query_id,
            stage=normalized_stage,
            requires_vectorization=requires_vectorization,
            compiler_vectorization_summary=compiler_vectorization_summary or {},
            failures=failures,
            details=details,
        )
        pmu_status = evaluate_pmu_status(
            query_id=query_id,
            stage=normalized_stage,
            requires_pmu=requires_pmu,
            hardware_counter_summary=hardware_counter_summary or {},
            failures=failures,
            details=details,
        )
        todo_status = evaluate_todo_status(
            query_id=query_id,
            requires_vectorization=requires_vectorization,
            todo_reconciliation=todo_reconciliation or {},
            details=details,
        )

        evaluate_measurement_provenance(
            query_id=query_id,
            is_critical=is_critical,
            measurement_records=measurement_records,
            failures=failures,
            details=details,
        )

        query_evidence[query_id] = QueryEvidence(
            query_id=query_id,
            is_critical=is_critical,
            requires_vectorization=requires_vectorization,
            requires_pmu=requires_pmu,
            deferred_obligations=deferred_obligations,
            vector_contract_status=vector_contract_status,
            correctness_status=EvidenceStatus.PRESENT,
            final_path_status=final_path_status,
            vectorization_status=vectorization_status,
            pmu_status=pmu_status,
            runtime_status=runtime_status,
            todo_status=todo_status,
            failures=tuple(dict.fromkeys(failures)),
            details=details,
        )

    all_failures = tuple(
        dict.fromkeys(
            failure
            for evidence in query_evidence.values()
            for failure in evidence.failures
        )
    )
    return PipelineEvidenceLedger(
        objective_id=objective_id,
        stage=normalized_stage,
        query_evidence=query_evidence,
        measurement_records=measurement_records,
        failures=all_failures,
        failure_route=classify_ledger_failure_route(all_failures),
    )


def build_deferred_obligations(
    *,
    stage: PipelineEvidenceStage,
    is_critical: bool,
    requires_vectorization: bool,
    requires_pmu: bool,
) -> tuple[str, ...]:
    """Return proof obligations intentionally deferred from base to optimization."""
    if stage is not PipelineEvidenceStage.BASE_PROMOTION or not is_critical:
        return ()
    obligations: list[str] = []
    if requires_vectorization:
        obligations.append("VECTOR_PROOF_DEFERRED_TO_OPTIMIZATION")
    if requires_pmu:
        obligations.append("PMU_PROOF_DEFERRED_TO_OPTIMIZATION")
    return tuple(obligations)


def evaluate_vector_contract_status(
    *,
    requires_vectorization: bool,
    stage: PipelineEvidenceStage,
    deferred_obligations: tuple[str, ...],
    details: dict[str, Any],
) -> EvidenceStatus:
    """Record whether the objective contains a vectorization contract for this query."""
    if not requires_vectorization:
        return EvidenceStatus.PRESENT
    if (
        stage is PipelineEvidenceStage.BASE_PROMOTION
        and "VECTOR_PROOF_DEFERRED_TO_OPTIMIZATION" not in deferred_obligations
    ):
        details["vectorization_contract_missing_deferred_obligation"] = True
        return EvidenceStatus.FAIL
    details["vectorization_contract_registered"] = True
    return EvidenceStatus.PRESENT


def evaluate_final_path_status(
    *,
    workspace_path: Path,
    query_id: str,
    failures: list[str],
    details: dict[str, Any],
) -> EvidenceStatus:
    """Reject generated final query entrypoints that call instrumentation replacements."""
    query_cpp = workspace_path / f"query_q{query_id}.cpp"
    if not query_cpp.exists():
        details["final_path_missing"] = query_cpp.as_posix()
        return EvidenceStatus.MISSING
    text = query_cpp.read_text(encoding="utf-8")
    matches = [token for token in FORBIDDEN_FINAL_PATH_TOKENS if token in text]
    if matches:
        failures.append("FORBIDDEN_INSTRUMENTED_FINAL_PATH")
        details["forbidden_final_path_tokens"] = matches
        return EvidenceStatus.FAIL
    return EvidenceStatus.PASS


def evaluate_runtime_status(
    *,
    query_id: str,
    is_critical: bool,
    final_runtime_ms_by_query: dict[str, float],
    baseline_runtime_ms_by_query: dict[str, float],
    failures: list[str],
    details: dict[str, Any],
) -> EvidenceStatus:
    """Validate that critical queries carry positive finite final and baseline runtimes."""
    if query_id not in final_runtime_ms_by_query and query_id not in baseline_runtime_ms_by_query:
        return EvidenceStatus.MISSING
    final_ms = final_runtime_ms_by_query.get(query_id)
    baseline_ms = baseline_runtime_ms_by_query.get(query_id)
    if not _is_positive_finite_number(final_ms) or not _is_positive_finite_number(baseline_ms):
        if is_critical:
            failures.append("CRITICAL_QUERY_RUNTIME_MISSING")
            details.setdefault("critical_runtime_missing", []).append(query_id)
        return EvidenceStatus.FAIL
    return EvidenceStatus.PASS


def evaluate_vectorization_status(
    *,
    query_id: str,
    stage: PipelineEvidenceStage,
    requires_vectorization: bool,
    compiler_vectorization_summary: dict[str, Any],
    failures: list[str],
    details: dict[str, Any],
) -> EvidenceStatus:
    """Require vectorization evidence to map to the current query hot loop."""
    if not requires_vectorization:
        return EvidenceStatus.PRESENT
    if stage is PipelineEvidenceStage.BASE_PROMOTION:
        details.setdefault("vectorization_proof_deferred", []).append(query_id)
        return EvidenceStatus.DEFERRED
    query_summary = _summary_for_query(compiler_vectorization_summary, query_id)
    if query_summary.get("vectorization_applied") is not True:
        failures.append("VECTOR_HOT_LOOP_NOT_OPTIMIZED")
        details.setdefault("vectorization_missing", []).append(query_id)
        return EvidenceStatus.FAIL
    hot_loop_mapping = query_summary.get("hot_loop_mapping") or {}
    if not isinstance(hot_loop_mapping, dict) or hot_loop_mapping.get("status") != "matched":
        failures.append("VECTOR_HOT_LOOP_MAPPING_MISSING")
        details.setdefault("vectorization_unmapped", []).append(query_id)
        return EvidenceStatus.FAIL
    return EvidenceStatus.PASS


def evaluate_pmu_status(
    *,
    query_id: str,
    stage: PipelineEvidenceStage,
    requires_pmu: bool,
    hardware_counter_summary: dict[str, Any],
    failures: list[str],
    details: dict[str, Any],
) -> EvidenceStatus:
    """Require PMU evidence to be available and scoped to the query loop."""
    if not requires_pmu:
        return EvidenceStatus.PRESENT
    if stage is PipelineEvidenceStage.BASE_PROMOTION:
        details.setdefault("pmu_proof_deferred", []).append(query_id)
        return EvidenceStatus.DEFERRED
    query_summary = _summary_for_query(hardware_counter_summary, query_id)
    if query_summary.get("hardware_counters_available") is not True:
        failures.append("PMU_REQUIRED_BUT_MISSING")
        details.setdefault("pmu_missing", []).append(query_id)
        return EvidenceStatus.FAIL
    hotspot_provenance = dict(query_summary.get("perf_hotspot_provenance") or {})
    provenance_failure = collect_pmu_provenance_failure(
        query_id=query_id,
        hotspot_provenance=hotspot_provenance,
        details=details,
    )
    if provenance_failure is not None:
        failures.append(provenance_failure)
        return EvidenceStatus.FAIL
    if query_summary.get("perf_hotspots_available") is not True:
        failures.append("PMU_HOTSPOT_MISSING")
        details.setdefault("pmu_hotspot_missing", []).append(query_id)
        return EvidenceStatus.FAIL
    return EvidenceStatus.PASS


def collect_pmu_provenance_failure(
    *,
    query_id: str,
    hotspot_provenance: dict[str, Any],
    details: dict[str, Any],
) -> str | None:
    """Validate query-only PMU provenance using structured capture-boundary fields."""
    capture_scope = hotspot_provenance.get("capture_scope")
    if capture_scope != "query_loop_only":
        details.setdefault("pmu_wrong_scope", {})[query_id] = capture_scope
        return "PMU_CAPTURE_SCOPE_NOT_QUERY_ONLY"
    if hotspot_provenance.get("warmup_completed") is not True:
        details.setdefault("pmu_warmup_not_completed", []).append(query_id)
        return "PMU_WARMUP_NOT_COMPLETED"
    if hotspot_provenance.get("record_started_after_warmup") is not True:
        details.setdefault("pmu_record_not_after_warmup", []).append(query_id)
        return "PMU_RECORD_NOT_AFTER_WARMUP"
    measured_repetitions = hotspot_provenance.get("measured_query_repetitions")
    measured_batch_size = hotspot_provenance.get("measured_batch_size")
    if not _is_positive_int(measured_repetitions) and not _is_positive_int(measured_batch_size):
        details.setdefault("pmu_measured_query_missing", []).append(query_id)
        return "PMU_MEASURED_QUERY_MISSING"
    return None


def evaluate_todo_status(
    *,
    query_id: str,
    requires_vectorization: bool,
    todo_reconciliation: dict[str, Any],
    details: dict[str, Any],
) -> EvidenceStatus:
    """Expose TODO semantic status without using TODO as the authoritative gate."""
    semantic_items = dict(todo_reconciliation.get("semantic_items", {}) or {})
    if not semantic_items:
        return EvidenceStatus.MISSING
    if not requires_vectorization:
        return EvidenceStatus.PRESENT
    vector_items = [
        item for item in semantic_items.values()
        if item.get("category") == "vectorization"
        and query_id in str(item.get("content", ""))
    ]
    if not vector_items:
        return EvidenceStatus.MISSING
    details.setdefault("todo_vectorization_items", {})[query_id] = vector_items
    if all(item.get("status") == "completed" for item in vector_items):
        return EvidenceStatus.PASS
    return EvidenceStatus.FAIL


def evaluate_measurement_provenance(
    *,
    query_id: str,
    is_critical: bool,
    measurement_records: tuple[QueryMeasurementRecord, ...],
    failures: list[str],
    details: dict[str, Any],
) -> EvidenceStatus:
    """Require no-CSV optimization runtime provenance for measurement gate decisions."""
    from tpch_monetdb.benchmark.runtime_accounting import (
        OPTIMIZATION_RUNTIME_METRIC_KINDS,
    )

    bespoke_records = [
        record for record in measurement_records
        if record.engine == "generated_tpch" and record.query_id == query_id
    ]
    if not bespoke_records:
        if is_critical:
            failures.append("OPTIMIZATION_RUNTIME_MISSING")
            details.setdefault("optimization_runtime_missing", []).append(query_id)
            return EvidenceStatus.FAIL
        return EvidenceStatus.MISSING
    for record in bespoke_records:
        metric_kind = record.provenance.get("runtime_metric_kind", "")
        if metric_kind not in OPTIMIZATION_RUNTIME_METRIC_KINDS:
            failures.append("OPTIMIZATION_RUNTIME_MISSING")
            details.setdefault("optimization_runtime_metric_kind_invalid", {})[
                query_id
            ] = metric_kind
            return EvidenceStatus.FAIL
    return EvidenceStatus.PASS


def classify_ledger_failure_route(failures: tuple[str, ...]) -> str | None:
    """Map ledger failure codes to the next outer-loop route."""
    if not failures:
        return None
    route_by_failure = (
        (("STORAGE_PLAN", "CONTROL_ARTIFACT", "DATA_LAW"), "storage_plan"),
        (("VECTOR", "BASE_VECTORIZATION"), "vectorization"),
        (("PMU", "MEASUREMENT", "RUNTIME_METRIC_KIND_FALLBACK", "OFFICIAL_QUERY_RUNTIME", "OPTIMIZATION_RUNTIME"), "instrumentation"),
        (("FORBIDDEN_INSTRUMENTED", "FINAL_PATH"), "instrumentation"),
        (("CRITICAL_QUERY",), "optimization"),
    )
    for prefixes, route in route_by_failure:
        if any(failure.startswith(prefix) for prefix in prefixes for failure in failures):
            return route
    return "optimization"


def require_base_impl_promotable(ledger: PipelineEvidenceLedger) -> None:
    """Raise when critical base implementation evidence is not ready for optimization."""
    failures: list[str] = []
    for evidence in ledger.query_evidence.values():
        if not evidence.is_critical:
            continue
        if evidence.final_path_status is not EvidenceStatus.PASS:
            failures.append("BASE_FINAL_PATH_INVALID")
        if (
            evidence.requires_vectorization
            and evidence.vector_contract_status is not EvidenceStatus.PRESENT
        ):
            failures.append("BASE_VECTOR_CONTRACT_NOT_REGISTERED")
        if (
            evidence.requires_vectorization
            and "VECTOR_PROOF_DEFERRED_TO_OPTIMIZATION" not in evidence.deferred_obligations
        ):
            failures.append("BASE_VECTOR_PROOF_OBLIGATION_MISSING")
        if (
            evidence.requires_pmu
            and "PMU_PROOF_DEFERRED_TO_OPTIMIZATION" not in evidence.deferred_obligations
        ):
            failures.append("BASE_PMU_PROOF_OBLIGATION_MISSING")
    if failures:
        raise_pipeline_contract_error(
            code=failures[0],
            message=(
                "Base implementation is not promotable: "
                + ", ".join(dict.fromkeys(failures))
            ),
            stage="base_impl_promotion",
        )
    return None


def _normalize_stage(stage: PipelineEvidenceStage | str) -> PipelineEvidenceStage:
    """Return a PipelineEvidenceStage from enum or string input."""
    if isinstance(stage, PipelineEvidenceStage):
        return stage
    return PipelineEvidenceStage(str(stage))


def _summary_for_query(summary_by_query: dict[str, Any], query_id: str) -> dict[str, Any]:
    """Return a query-specific summary from flat, numeric, or Q-prefixed maps."""
    if query_id in summary_by_query and isinstance(summary_by_query[query_id], dict):
        return dict(summary_by_query[query_id])
    prefixed = f"Q{query_id}"
    if prefixed in summary_by_query and isinstance(summary_by_query[prefixed], dict):
        return dict(summary_by_query[prefixed])
    return summary_by_query if isinstance(summary_by_query, dict) else {}


def _is_positive_finite_number(value: Any) -> bool:
    """Return True when value is a positive finite int or float."""
    return isinstance(value, (int, float)) and math.isfinite(value) and value > 0


def _is_positive_int(value: Any) -> bool:
    """Return True when value is a positive integer-like count."""
    return isinstance(value, int) and not isinstance(value, bool) and value > 0
