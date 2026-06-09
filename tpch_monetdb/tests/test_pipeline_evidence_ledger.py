from pathlib import Path

from tpch_monetdb.benchmark.runtime_accounting import (
    KERNEL_RUNTIME_METRIC_KIND,
    QUERY_RUNTIME_METRIC_KIND,
)
from tpch_monetdb.utils.pipeline_evidence import (
    EvidenceStatus,
    MeasurementShapeStatus,
    PipelineEvidenceStage,
    QueryMeasurementRecord,
    build_pipeline_evidence_ledger,
    evaluate_measurement_provenance,
    require_base_impl_promotable,
)


def _objective() -> dict:
    """Build a critical-query objective fixture for evidence-ledger tests."""
    return {
        "objective_id": "obj-test",
        "query_ids": ["8"],
        "critical_query_ids": ["8"],
        "critical_query_targets": {
            "8": {
                "min_speedup_vs_baseline": 1.0,
                "requires_vectorization": True,
                "requires_pmu": True,
            }
        },
    }


def _query_only_pmu_provenance() -> dict:
    """Build structured query-only PMU provenance for ledger tests."""
    return {
        "capture_scope": "query_loop_only",
        "warmup_completed": True,
        "record_started_after_warmup": True,
        "measured_query_repetitions": 3,
        "measured_batch_size": 3,
    }


def test_pipeline_evidence_ledger_rejects_final_instrumented_path(tmp_path: Path) -> None:
    """Final query entrypoints must not call instrumentation-only implementations."""
    (tmp_path / "query_q8.cpp").write_text(
        "void execute_q8() { execute_cpu_max_groupby_instrumented(); }\n",
        encoding="utf-8",
    )

    ledger = build_pipeline_evidence_ledger(
        workspace_path=tmp_path,
        workload_objective=_objective(),
        final_runtime_ms_by_query={"8": 1.0},
        baseline_runtime_ms_by_query={"8": 2.0},
        compiler_vectorization_summary={
            "8": {
                "vectorization_applied": True,
                "hot_loop_mapping": {"status": "matched"},
            }
        },
        hardware_counter_summary={
            "8": {
                "hardware_counters_available": True,
                "perf_hotspots_available": True,
                "perf_hotspot_provenance": _query_only_pmu_provenance(),
            }
        },
        measurement_records=(
            QueryMeasurementRecord(
                query_id="8",
                engine="generated_tpch",
                measurement_kind="exact_instantiation",
                runtime_ms=1.0,
                measurement_shape_status=MeasurementShapeStatus.UNKNOWN,
            ),
        ),
    )

    assert "FORBIDDEN_INSTRUMENTED_FINAL_PATH" in ledger.failures
    assert ledger.failure_route == "instrumentation"


def test_pipeline_evidence_ledger_requires_query_only_pmu_scope(tmp_path: Path) -> None:
    """PMU evidence must carry query-loop-only provenance."""
    (tmp_path / "query_q8.cpp").write_text(
        "void execute_q8() {}\n",
        encoding="utf-8",
    )

    ledger = build_pipeline_evidence_ledger(
        workspace_path=tmp_path,
        workload_objective=_objective(),
        final_runtime_ms_by_query={"8": 1.0},
        baseline_runtime_ms_by_query={"8": 2.0},
        compiler_vectorization_summary={
            "8": {
                "vectorization_applied": True,
                "hot_loop_mapping": {"status": "matched"},
            }
        },
        hardware_counter_summary={
            "8": {
                "hardware_counters_available": True,
                "perf_hotspots_available": True,
                "perf_hotspot_provenance": {"capture_scope": "query_batch_file"},
            }
        },
    )

    assert "PMU_CAPTURE_SCOPE_NOT_QUERY_ONLY" in ledger.failures
    assert ledger.failure_route == "instrumentation"


def test_require_base_impl_promotable_defers_vectorization_proof(tmp_path: Path) -> None:
    """Base promotion records vector contracts without requiring optimization proof."""
    (tmp_path / "query_q8.cpp").write_text(
        "void execute_q8() {}\n",
        encoding="utf-8",
    )
    ledger = build_pipeline_evidence_ledger(
        workspace_path=tmp_path,
        workload_objective=_objective(),
        stage=PipelineEvidenceStage.BASE_PROMOTION,
        hardware_counter_summary={},
    )

    require_base_impl_promotable(ledger)
    evidence = ledger.query_evidence["8"]
    assert evidence.vector_contract_status is EvidenceStatus.PRESENT
    assert evidence.vectorization_status is EvidenceStatus.DEFERRED
    assert evidence.pmu_status is EvidenceStatus.DEFERRED
    assert "VECTOR_PROOF_DEFERRED_TO_OPTIMIZATION" in evidence.deferred_obligations
    assert "PMU_PROOF_DEFERRED_TO_OPTIMIZATION" in evidence.deferred_obligations
    return None


def test_optimization_final_still_requires_vectorization_proof(tmp_path: Path) -> None:
    """Optimization-final ledger must reject missing vectorization proof."""
    (tmp_path / "query_q8.cpp").write_text(
        "void execute_q8() {}\n",
        encoding="utf-8",
    )
    ledger = build_pipeline_evidence_ledger(
        workspace_path=tmp_path,
        workload_objective=_objective(),
        stage=PipelineEvidenceStage.OPTIMIZATION_FINAL,
        compiler_vectorization_summary={
            "8": {
                "vectorization_applied": False,
                "hot_loop_mapping": {"status": "missing"},
            }
        },
        hardware_counter_summary={
            "8": {
                "hardware_counters_available": True,
                "perf_hotspots_available": True,
                "perf_hotspot_provenance": _query_only_pmu_provenance(),
            }
        },
    )

    assert "VECTOR_HOT_LOOP_NOT_OPTIMIZED" in ledger.failures
    assert ledger.failure_route == "vectorization"
    return None


def test_optimization_final_rejects_query_only_pmu_without_warmup_boundary(
    tmp_path: Path,
) -> None:
    """Query-only PMU evidence must include structured warmup and measured-window fields."""
    (tmp_path / "query_q8.cpp").write_text(
        "void execute_q8() {}\n",
        encoding="utf-8",
    )

    ledger = build_pipeline_evidence_ledger(
        workspace_path=tmp_path,
        workload_objective=_objective(),
        stage=PipelineEvidenceStage.OPTIMIZATION_FINAL,
        final_runtime_ms_by_query={"8": 1.0},
        baseline_runtime_ms_by_query={"8": 2.0},
        compiler_vectorization_summary={
            "8": {
                "vectorization_applied": True,
                "hot_loop_mapping": {"status": "matched"},
            }
        },
        hardware_counter_summary={
            "8": {
                "hardware_counters_available": True,
                "perf_hotspots_available": True,
                "perf_hotspot_provenance": {
                    "capture_scope": "query_loop_only",
                    "record_started_after_warmup": True,
                    "measured_query_repetitions": 3,
                },
            }
        },
    )

    assert "PMU_WARMUP_NOT_COMPLETED" in ledger.failures
    assert ledger.failure_route == "instrumentation"
    return None


def test_measurement_provenance_accepts_no_csv_kernel_runtime(tmp_path: Path) -> None:
    (tmp_path / "query_q8.cpp").write_text(
        "void execute_q8() {}\n",
        encoding="utf-8",
    )
    record_with_kernel_fallback = QueryMeasurementRecord(
        query_id="8",
        engine="generated_tpch",
        measurement_kind="exact_instantiation",
        runtime_ms=1.0,
        provenance={"runtime_metric_kind": KERNEL_RUNTIME_METRIC_KIND},
    )
    ledger = build_pipeline_evidence_ledger(
        workspace_path=tmp_path,
        workload_objective=_objective(),
        final_runtime_ms_by_query={"8": 1.0},
        baseline_runtime_ms_by_query={"8": 2.0},
        compiler_vectorization_summary={
            "8": {
                "vectorization_applied": True,
                "hot_loop_mapping": {"status": "matched"},
            }
        },
        hardware_counter_summary={
            "8": {
                "hardware_counters_available": True,
                "perf_hotspots_available": True,
                "perf_hotspot_provenance": _query_only_pmu_provenance(),
            }
        },
        measurement_records=(record_with_kernel_fallback,),
    )

    assert "RUNTIME_METRIC_KIND_FALLBACK" not in ledger.failures
    assert "OPTIMIZATION_RUNTIME_MISSING" not in ledger.failures


def test_measurement_provenance_rejects_missing_record_for_critical_query(
    tmp_path: Path,
) -> None:
    (tmp_path / "query_q8.cpp").write_text(
        "void execute_q8() {}\n",
        encoding="utf-8",
    )
    ledger = build_pipeline_evidence_ledger(
        workspace_path=tmp_path,
        workload_objective=_objective(),
        final_runtime_ms_by_query={"8": 1.0},
        baseline_runtime_ms_by_query={"8": 2.0},
        compiler_vectorization_summary={
            "8": {
                "vectorization_applied": True,
                "hot_loop_mapping": {"status": "matched"},
            }
        },
        hardware_counter_summary={
            "8": {
                "hardware_counters_available": True,
                "perf_hotspots_available": True,
                "perf_hotspot_provenance": _query_only_pmu_provenance(),
            }
        },
        measurement_records=(),
    )

    assert "OPTIMIZATION_RUNTIME_MISSING" in ledger.failures


def test_measurement_provenance_rejects_full_csv_query_metric(tmp_path: Path) -> None:
    record_with_full_csv = QueryMeasurementRecord(
        query_id="8",
        engine="generated_tpch",
        measurement_kind="exact_instantiation",
        runtime_ms=1.0,
        provenance={"runtime_metric_kind": QUERY_RUNTIME_METRIC_KIND},
    )
    failures: list[str] = []
    details: dict[str, str] = {}

    status = evaluate_measurement_provenance(
        query_id="8",
        is_critical=True,
        measurement_records=(record_with_full_csv,),
        failures=failures,
        details=details,
    )

    assert status is EvidenceStatus.FAIL
    assert failures == ["OPTIMIZATION_RUNTIME_MISSING"]


def test_measurement_provenance_non_critical_missing_record_is_not_fatal(
    tmp_path: Path,
) -> None:
    failures: list[str] = []
    details: dict[str, str] = {}

    status = evaluate_measurement_provenance(
        query_id="9",
        is_critical=False,
        measurement_records=(),
        failures=failures,
        details=details,
    )

    assert status is EvidenceStatus.MISSING
    assert failures == []
