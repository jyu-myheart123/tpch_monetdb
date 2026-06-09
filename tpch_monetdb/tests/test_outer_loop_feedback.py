import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

import run_outer_loop_tpch_monetdb
from run_outer_loop_tpch_monetdb import _build_prev_storage_plan_feedback
from tpch_monetdb.utils.optimization_summary import (
    OptimizationRunSummary,
    infer_issue_class_by_query,
    render_bottleneck_report,
)
from tpch_monetdb.utils.summary_gates import MEASUREMENT_AGGREGATION_GEOMEAN


def _valid_measurement_repetition(query_ids: list[str]) -> dict[str, object]:
    """Build valid repeated measurement evidence for feedback tests."""
    samples = [10.0, 10.1, 9.9]
    return {
        "scale_factor": 1000,
        "query_ids": query_ids,
        "repetitions": 3,
        "sample_count": 3,
        "aggregate_runtime_ms_samples": samples,
        "aggregate_runtime_ms_median": 10.0,
        "aggregate_runtime_ms_min": 9.9,
        "aggregate_runtime_ms_max": 10.1,
        "per_query_runtime_ms_samples": {qid: samples for qid in query_ids},
        "aggregation_method": MEASUREMENT_AGGREGATION_GEOMEAN,
        "source_command": "test measurement provider",
    }


def _large_data_success_fields(query_ids: list[str]) -> dict[str, object]:
    """Build explicit large-data success fields for feedback tests."""
    return {
        "control_artifact_hashes": {
            "storage_plan.txt": "plan",
            "TODO.md": "todo",
            "workload_objective.json": "objective",
            "data_law_contract.json": "laws",
            "storage_plan_contract.json": "contract",
        },
        "storage_plan_alignment": {"status": "aligned"},
        "todo_reconciliation": {"status": "present"},
        "stage_history": [{"stage": "trace_expert", "query_id": query_ids[0]}],
        "measurement_repetition": _valid_measurement_repetition(query_ids),
        "workload_objective": {
            "critical_query_ids": [],
            "required_artifacts": [
                "workload_objective.json",
                "data_law_contract.json",
                "storage_plan_contract.json",
            ],
            "measurement_policy": {"max_cv": 0.2, "max_spread_ratio": 0.4},
        },
    }


def _make_summary(*, final_correctness: bool, final_runtime: dict[str, float], baseline: dict[str, float]) -> OptimizationRunSummary:
    query_ids = sorted(final_runtime.keys())
    return OptimizationRunSummary(
        benchmark="tpch",
        conv_name="opt_round",
        run_id="opt_round",
        query_list=query_ids,
        is_bespoke_storage=True,
        start_snapshot_hash="abc",
        final_snapshot_hash="def",
        best_runtime_ms_by_query=dict(final_runtime),
        final_runtime_ms_by_query=dict(final_runtime),
        final_correctness=final_correctness,
        completed_at="2026-04-23T00:00:00Z",
        conversation_json="",
        session_db_path="",
        success=True,
        baseline_runtime_ms_by_query=dict(baseline),
        **_large_data_success_fields(query_ids),
    )


def test_infer_issue_class_marks_correctness_when_final_correctness_false() -> None:
    issue_class = infer_issue_class_by_query(
        query_list=["9", "12"],
        final_runtime_ms_by_query={"9": 10.0, "12": 11.0},
        baseline_runtime_ms_by_query={"9": 5.0, "12": 4.0},
        final_correctness=False,
    )
    assert issue_class == {"12": "correctness", "9": "correctness"}


def test_infer_issue_class_uses_query_specific_defaults_for_slow_queries() -> None:
    issue_class = infer_issue_class_by_query(
        query_list=["9", "12", "15"],
        final_runtime_ms_by_query={"9": 10.0, "12": 10.0, "15": 10.0},
        baseline_runtime_ms_by_query={"9": 5.0, "12": 5.0, "15": 5.0},
        final_correctness=True,
    )
    assert issue_class["9"] == "implementation"
    assert issue_class["12"] == "mixed"
    assert issue_class["15"] == "layout"


def test_render_bottleneck_report_includes_issue_class_and_correctness_gate() -> None:
    summary = _make_summary(
        final_correctness=False,
        final_runtime={"9": 10.0},
        baseline={"9": 5.0},
    )
    summary.issue_class_by_query = {"9": "correctness"}
    report = render_bottleneck_report(summary)
    assert "Issue Class" in report
    assert "final_correctness=false" in report
    assert "| 9 | 10.000ms | 5.000ms | 0.50× | correctness | ⚠ slow |" in report


def test_build_prev_storage_plan_feedback_returns_gate_note_for_invalid_summary() -> None:
    summary = _make_summary(
        final_correctness=False,
        final_runtime={"9": 10.0},
        baseline={"9": 5.0},
    )
    feedback = _build_prev_storage_plan_feedback(summary)
    assert "correctness gate" in feedback.lower()
    assert "do not use it as layout evidence" in feedback


def test_build_prev_storage_plan_feedback_returns_report_for_valid_summary() -> None:
    summary = _make_summary(
        final_correctness=True,
        final_runtime={"9": 10.0},
        baseline={"9": 5.0},
    )
    summary.issue_class_by_query = {"9": "implementation"}
    feedback = _build_prev_storage_plan_feedback(summary)
    assert "Previous round (opt_round) results" in feedback
    assert "implementation" in feedback


def test_build_prev_storage_plan_feedback_prefers_validator_metrics() -> None:
    summary = _make_summary(
        final_correctness=True,
        final_runtime={"1": 999.0},
        baseline={"1": 1.0},
    )
    summary.success = False
    summary.final_validation_metrics = {
        "validation/correct": True,
        "validation/scale_factor": 1000,
        "validation/query_001/runtime_metric_kind": "kernel_ms",
        "validation/query_001/no_csv_runtime_ms": 0.207,
        "validation/query_001/baseline_runtime_ms": 3.932,
        "validation/total_no_csv_runtime_ms": 0.207,
        "validation/total_baseline_runtime_ms": 3.932,
        "validation/no_csv_total_speedup": 19.0,
    }

    feedback = _build_prev_storage_plan_feedback(summary)

    assert "Previous round (opt_round) validator results" in feedback
    assert "207.00us" in feedback
    assert "3.932ms" in feedback
    assert "19.00×" in feedback
    assert "storage-layout success proof" in feedback
    assert "999.000ms" not in feedback


def test_build_prev_storage_plan_feedback_rejects_non_measurable_summary() -> None:
    summary = _make_summary(
        final_correctness=True,
        final_runtime={"9": 10.0},
        baseline={"9": 5.0},
    )
    summary.measurement_repetition = {}
    feedback = _build_prev_storage_plan_feedback(summary)
    assert "not measurable" in feedback


def test_measurable_gate_rejects_unknown_exact_shape() -> None:
    from tpch_monetdb.utils.summary_gates import is_measurable_success

    summary = _make_summary(
        final_correctness=True,
        final_runtime={"9": 10.0},
        baseline={"9": 5.0},
    )
    summary.measurement_records = [
        {
            "query_id": "9",
            "engine": "generated_tpch",
            "measurement_kind": "exact_instantiation",
            "measurement_shape_status": "unknown",
        }
    ]

    assert not is_measurable_success(summary)


def test_performance_comparison_report_requires_large_data_success(tmp_path: Path) -> None:
    record = run_outer_loop_tpch_monetdb.RoundRecord(
        outer_loop_name="outer9v1",
        round_index=1,
        query_list=["9"],
    )
    summary = _make_summary(
        final_correctness=True,
        final_runtime={"9": 10.0},
        baseline={"9": 5.0},
    )
    summary.measurement_repetition = {}
    with pytest.raises(ValueError, match="successful large-data summary"):
        run_outer_loop_tpch_monetdb._write_performance_comparison_report(record, summary, tmp_path)
