import subprocess
import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from tpch_monetdb.utils.optimization_summary import (
    OptimizationRunSummary,
    find_latest_optimization_run,
    find_latest_successful_optimization_run,
    persist_optimization_run,
    write_optimization_run_summary,
)
from tpch_monetdb.utils.outer_loop_supervisor import classify_optimization_result
from tpch_monetdb.conversations.optimization_conversation_tpch_monetdb import TpchMonetdbOptimizationConversation
from tpch_monetdb.utils.summary_gates import (
    MEASUREMENT_AGGREGATION_GEOMEAN,
    is_measurable_success,
)


def _valid_measurement_repetition(query_ids: list[str]) -> dict[str, object]:
    """Build valid repeated measurement evidence for optimization summary gates."""
    samples = [8.0, 8.2, 7.9]
    return {
        "scale_factor": 1000,
        "query_ids": query_ids,
        "repetitions": 3,
        "sample_count": 3,
        "aggregate_runtime_ms_samples": samples,
        "aggregate_runtime_ms_median": 8.0,
        "aggregate_runtime_ms_min": 7.9,
        "aggregate_runtime_ms_max": 8.2,
        "per_query_runtime_ms_samples": {qid: samples for qid in query_ids},
        "aggregation_method": MEASUREMENT_AGGREGATION_GEOMEAN,
        "source_command": "test measurement provider",
    }


def _large_data_success_fields(query_ids: list[str]) -> dict[str, object]:
    """Build explicit large-data success fields for optimization summary tests."""
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


def test_failure_optimization_summary_allows_empty_runtime_map(
    tmp_path: Path,
) -> None:
    path = persist_optimization_run(
        benchmark="tpch",
        conv_name="tpch_monetdb_runoptim1v1",
        query_list=["1"],
        is_bespoke_storage=True,
        start_snapshot_hash="start123",
        final_snapshot_hash="final123",
        artifacts_dir=tmp_path,
        success=False,
        final_correctness=False,
        failure_code="FINAL_MEASUREMENT_FAILED",
        failure_detail="Final measurement failed for 1",
        final_runtime_ms_by_query=None,
    )

    assert path.exists()
    latest = find_latest_optimization_run(
        conv_name="tpch_monetdb_runoptim1v1",
        query_list=["1"],
        benchmark="tpch",
        artifacts_dir=tmp_path,
    )
    assert latest is not None
    assert latest.success is False
    assert latest.final_runtime_ms_by_query == {}
    assert latest.failure_code == "FINAL_MEASUREMENT_FAILED"
    assert latest.failure_detail == "Final measurement failed for 1"


def test_failure_optimization_summary_is_not_successful_summary(
    tmp_path: Path,
) -> None:
    persist_optimization_run(
        benchmark="tpch",
        conv_name="tpch_monetdb_runoptim1v1",
        query_list=["1"],
        is_bespoke_storage=True,
        start_snapshot_hash="start123",
        final_snapshot_hash="final123",
        artifacts_dir=tmp_path,
        success=False,
        final_correctness=False,
        failure_code="LOCAL_PHASE_VALIDATION_FAILED",
        failure_detail="local full correctness failed",
    )

    successful = find_latest_successful_optimization_run(
        conv_name="tpch_monetdb_runoptim1v1",
        query_list=["1"],
        benchmark="tpch",
        artifacts_dir=tmp_path,
    )

    assert successful is None


def test_success_optimization_summary_still_requires_runtime_map(
    tmp_path: Path,
) -> None:
    summary = OptimizationRunSummary(
        benchmark="tpch",
        conv_name="tpch_monetdb_runoptim1v1",
        run_id="tpch_monetdb_runoptim1v1",
        query_list=["1"],
        is_bespoke_storage=True,
        start_snapshot_hash="start123",
        final_snapshot_hash="final123",
        best_runtime_ms_by_query={},
        final_runtime_ms_by_query={},
        final_correctness=True,
        completed_at="2026-05-03T00:00:00Z",
        conversation_json="",
        session_db_path="",
        success=True,
    )

    with pytest.raises(ValueError, match="final_runtime_ms_by_query cannot be empty"):
        write_optimization_run_summary(summary, tmp_path)


def test_supervisor_accepts_present_failure_summary_as_present() -> None:
    result = subprocess.CompletedProcess(
        args=["optimization"],
        returncode=0,
        stdout="",
        stderr="",
    )

    decision = classify_optimization_result(
        result,
        summary_found=True,
        retry_count=0,
        retry_budget=1,
    )

    assert decision.failure_code is None
    assert decision.should_retry is False
    assert decision.should_cleanup_runtime is False


def test_supervisor_context_too_large_fails_fast() -> None:
    result = subprocess.CompletedProcess(
        args=["optimization"],
        returncode=1,
        stdout="",
        stderr="litellm.BadRequestError: 413 Request Entity Too Large",
    )

    decision = classify_optimization_result(
        result,
        summary_found=False,
        retry_count=0,
        retry_budget=3,
    )

    assert decision.failure_code == "CONTEXT_TOO_LARGE"
    assert decision.should_retry is False
    assert decision.should_cleanup_runtime is False
    return None


def test_supervisor_context_envelope_fails_fast() -> None:
    result = subprocess.CompletedProcess(
        args=["optimization"],
        returncode=1,
        stdout="[ERROR:CONTEXT_TOO_LARGE] prompt blocked",
        stderr="",
    )

    decision = classify_optimization_result(
        result,
        summary_found=False,
        retry_count=0,
        retry_budget=3,
    )

    assert decision.failure_code == "CONTEXT_TOO_LARGE"
    assert decision.should_retry is False
    return None


def test_supervisor_runtime_segfault_cleans_up_and_retries() -> None:
    result = subprocess.CompletedProcess(
        args=["optimization"],
        returncode=1,
        stdout="resp=exit_code: 0 signal: 11\nExpected output file missing",
        stderr="",
    )

    decision = classify_optimization_result(
        result,
        summary_found=False,
        retry_count=0,
        retry_budget=1,
    )

    assert decision.failure_code == "RUNNER_SEGFAULT"
    assert decision.should_retry is True
    assert decision.should_cleanup_runtime is True
    return None


def test_optimization_conversation_persists_pre_stage_failure_summary(
    tmp_path: Path,
) -> None:
    conversation = TpchMonetdbOptimizationConversation.__new__(TpchMonetdbOptimizationConversation)
    conversation.benchmark = "tpch"
    conversation.conv_name = "tpch_runoptim1v1"
    conversation.artifacts_dir = tmp_path
    conversation.git_snapshotter = type(
        "Snapshotter",
        (),
        {"current_hash": "final123"},
    )()
    conversation.start_snapshot_hash = "start123"
    conversation.query_ids = ["Q1"]
    conversation.bespoke_storage = True

    conversation._persist_failure_summary(
        failure_code="RUNNER_SEGFAULT",
        failure_detail="signal: 11",
    )

    latest = find_latest_optimization_run(
        conv_name="tpch_runoptim1v1",
        query_list=["Q1"],
        benchmark="tpch",
        artifacts_dir=tmp_path,
    )
    assert latest is not None
    assert latest.success is False
    assert latest.failure_code == "RUNNER_SEGFAULT"
    assert latest.failure_detail == "signal: 11"
    return None


def test_optimization_conversation_classifies_context_too_large(
    tmp_path: Path,
) -> None:
    conversation = TpchMonetdbOptimizationConversation.__new__(TpchMonetdbOptimizationConversation)
    conversation.benchmark = "tpch"
    conversation.conv_name = "tpch_runoptim1v1"
    conversation.artifacts_dir = tmp_path
    conversation.git_snapshotter = type(
        "Snapshotter",
        (),
        {"current_hash": "final123"},
    )()
    conversation.start_snapshot_hash = "start123"
    conversation.query_ids = ["Q1"]
    conversation.bespoke_storage = True

    detail = "[ERROR:CONTEXT_TOO_LARGE] 413 Request Entity Too Large"
    conversation._persist_failure_summary(
        failure_code=conversation._classify_failure_text(detail) or "fallback",
        failure_detail=detail,
    )

    latest = find_latest_optimization_run(
        conv_name="tpch_runoptim1v1",
        query_list=["Q1"],
        benchmark="tpch",
        artifacts_dir=tmp_path,
    )
    assert latest is not None
    assert latest.failure_code == "CONTEXT_TOO_LARGE"
    return None


def test_success_summary_is_measurable_only_with_full_evidence() -> None:
    summary = OptimizationRunSummary(
        benchmark="tpch",
        conv_name="opt_round",
        run_id="opt_round",
        query_list=["1"],
        is_bespoke_storage=True,
        start_snapshot_hash="s0",
        final_snapshot_hash="s1",
        best_runtime_ms_by_query={"1": 8.0},
        final_runtime_ms_by_query={"1": 8.0},
        final_correctness=True,
        completed_at="2026-05-08T00:00:00Z",
        conversation_json="",
        session_db_path="",
        success=True,
        baseline_runtime_ms_by_query={"1": 4.0},
        **_large_data_success_fields(["1"]),
    )
    assert is_measurable_success(summary) is True


def test_invalid_storage_alignment_is_advisory_for_measurement_gate() -> None:
    fields = _large_data_success_fields(["1"])
    fields["storage_plan_alignment"] = {
        "status": "invalid",
        "departures": ["STORAGE_PLAN_CONTRACT_CRITICAL_ACCESS_PATHS_MISSING"],
    }
    summary = OptimizationRunSummary(
        benchmark="tpch",
        conv_name="opt_round",
        run_id="opt_round",
        query_list=["1"],
        is_bespoke_storage=True,
        start_snapshot_hash="s0",
        final_snapshot_hash="s1",
        best_runtime_ms_by_query={"1": 8.0},
        final_runtime_ms_by_query={"1": 8.0},
        final_correctness=True,
        completed_at="2026-05-08T00:00:00Z",
        conversation_json="",
        session_db_path="",
        success=True,
        baseline_runtime_ms_by_query={"1": 4.0},
        **fields,
    )

    assert is_measurable_success(summary) is True


def test_success_summary_without_measurement_evidence_is_not_measurable() -> None:
    summary = OptimizationRunSummary(
        benchmark="tpch",
        conv_name="opt_round",
        run_id="opt_round",
        query_list=["1"],
        is_bespoke_storage=True,
        start_snapshot_hash="s0",
        final_snapshot_hash="s1",
        best_runtime_ms_by_query={"1": 8.0},
        final_runtime_ms_by_query={"1": 8.0},
        final_correctness=True,
        completed_at="2026-05-08T00:00:00Z",
        conversation_json="",
        session_db_path="",
        success=True,
        baseline_runtime_ms_by_query={"1": 4.0},
        control_artifact_hashes={"storage_plan.txt": "plan"},
        storage_plan_alignment={"status": "aligned"},
        todo_reconciliation={"status": "present"},
    )
    assert is_measurable_success(summary) is False


def test_persist_optimization_run_round_trips_unit_and_measurement_provenance(
    tmp_path: Path,
) -> None:
    path = persist_optimization_run(
        benchmark="tpch",
        conv_name="tpch_monetdb_runoptim3-7v1",
        query_list=["3", "4", "5", "6", "7"],
        is_bespoke_storage=True,
        start_snapshot_hash="start123",
        final_snapshot_hash="final123",
        artifacts_dir=tmp_path,
        success=True,
        final_correctness=True,
        best_runtime_ms_by_query={"3": 1.0, "4": 1.0, "5": 1.0, "6": 1.0, "7": 1.0},
        final_runtime_ms_by_query={"3": 1.1, "4": 1.1, "5": 1.1, "6": 1.1, "7": 1.1},
        baseline_runtime_ms_by_query={"3": 0.5, "4": 0.5, "5": 0.5, "6": 0.5, "7": 0.5},
        optimization_units=[{"unit_id": "family:single_groupby:3-4-5-6-7"}],
        unit_scores={"family:single_groupby:3-4-5-6-7": 1.23},
        change_scope="family",
        hardware_counter_summary={"backend": "linux_perf_native"},
        compiler_vectorization_summary={"missed_loops": 1},
        build_profile="release",
        target_cpu="icelake",
        hotspot_analysis_degraded=True,
        hotspot_analysis_failure_reason="trace evidence insufficient",
        **_large_data_success_fields(["3", "4", "5", "6", "7"]),
    )

    loaded = find_latest_successful_optimization_run(
        conv_name="tpch_monetdb_runoptim3-7v1",
        query_list=["3", "4", "5", "6", "7"],
        benchmark="tpch",
        artifacts_dir=tmp_path,
    )

    assert path.exists()
    assert loaded is not None
    assert loaded.optimization_units[0]["unit_id"] == "family:single_groupby:3-4-5-6-7"
    assert loaded.change_scope == "family"
    assert loaded.hardware_counter_summary["backend"] == "linux_perf_native"
    assert loaded.compiler_vectorization_summary["missed_loops"] == 1
    assert loaded.hotspot_analysis_degraded is True
    assert loaded.hotspot_analysis_failure_reason == "trace evidence insufficient"


def test_compiler_vectorization_summary_force_refreshes_hot_loop_mapping(
    tmp_path: Path,
) -> None:
    """Final vectorization evidence must come from latest reports, not stale cache."""
    build_dir = tmp_path / "build"
    build_dir.mkdir()
    (build_dir / "vectorization.optimized.txt").write_text(
        "query_q8.cpp:42:3: optimized: loop vectorized using 32 byte vectors\n",
        encoding="utf-8",
    )
    (build_dir / "vectorization.missed.txt").write_text("", encoding="utf-8")

    conversation = TpchMonetdbOptimizationConversation.__new__(TpchMonetdbOptimizationConversation)
    conversation.run_tool = SimpleNamespace(cwd=tmp_path)
    conversation.target_cpu = "icelake"
    conversation.compiler_vectorization_summary = {
        "8": {
            "vectorization_applied": False,
            "optimized_loop_sites": [],
            "hot_loop_mapping": {"status": "unmatched"},
        }
    }

    summary = conversation._collect_compiler_vectorization_summary(
        "8",
        force_refresh=True,
        trace_summary_text="query_q8.cpp is the measured hot loop",
    )

    assert summary["vectorization_applied"] is True
    assert summary["optimized_loop_sites"][0]["file"] == "query_q8.cpp"
    assert summary["hot_loop_mapping"]["status"] == "matched"
    assert conversation.compiler_vectorization_summary["8"] == summary


@pytest.mark.asyncio
async def test_run_stage_rolls_back_family_unit_regression() -> None:
    from tpch_monetdb.conversations.optimization_conversation_tpch_monetdb import StageConfig
    from tpch_monetdb.conversations.optimization_validation import CorrectnessCheckSummary
    from tpch_monetdb.tools.stage_tool_policy import StageRunSummary

    restore_calls: list[str] = []
    conversation = TpchMonetdbOptimizationConversation.__new__(TpchMonetdbOptimizationConversation)
    conversation.benchmark_sf = 100
    conversation.bespoke_storage = True
    conversation.query_ids = ["3", "4"]
    conversation.required_validation_sf_list = [1, 10]
    conversation.revert_on_regression = True
    conversation.query_rt_log = {}
    conversation.best_rt_log = {}
    conversation.wandb_run_hook = None
    conversation.run_tool = SimpleNamespace(run=lambda **_kwargs: ("ok", {}))
    conversation.git_snapshotter = SimpleNamespace(
        current_hash="snap-1",
        restore=lambda snapshot: restore_calls.append(snapshot),
    )

    measurements = iter(
        [
            ({"3": 0.20, "4": 0.18}, {"3": 0.10, "4": 0.10}, 0.1897366596, False),
            ({"3": 0.10, "4": 0.09}, {"3": 0.10, "4": 0.10}, 0.0948683298, False),
        ]
    )
    conversation._measure_scope_runtime = (
        lambda query_ids, scale_factor=None, output_mode="full_csv": next(measurements)
    )

    async def fake_exec(*_args, **_kwargs):
        return StageRunSummary(
            profile_name="optimization_general",
            prompt_index=0,
            prompt_descriptor="trace_expert",
            final_output="done",
            tool_counts={"edit_file": 1},
            written_files=("query_family_single_groupby.cpp",),
            last_compile_summary=None,
            last_run_summary=None,
            todo_before=None,
            todo_after=None,
        )

    conversation._exec = fake_exec

    import tpch_monetdb.conversations.optimization_conversation_tpch_monetdb as optimization_module

    original = optimization_module.run_required_correctness_checks
    optimization_module.run_required_correctness_checks = lambda *args, **kwargs: CorrectnessCheckSummary(
        success=True,
        message="ok",
        metrics={"validation/correct": True},
        failed_scale_factor=None,
    )
    try:
        result = await conversation._run_stage(
            query_id="3",
            stage=StageConfig(
                name="trace_expert",
                get_prompt=lambda _rt: "",
                get_descriptor=lambda: "trace_expert",
                max_turns=10,
            ),
            pretext_optim="",
            rt_before_s=0.10,
            scope_query_ids=("3", "4"),
        )
    finally:
        optimization_module.run_required_correctness_checks = original

    assert result.failed is False
    assert restore_calls == ["snap-1"]
    assert result.runtime_by_query == {"3": 0.10, "4": 0.09}
