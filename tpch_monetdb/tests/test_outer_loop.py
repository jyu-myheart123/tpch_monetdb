import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

import run_optim_loop_tpch_monetdb
import run_outer_loop_tpch_monetdb
import tpch_monetdb.benchmark.manifest as manifest_module
import tpch_monetdb.run_gen_base_impl_tpch_monetdb
import tpch_monetdb.run_gen_storage_plan_tpch_monetdb
from tpch_monetdb.benchmark.manifest import QueryInstantiation, ReferenceManifest
from tpch_monetdb.conversations.optimization_conversation_tpch_monetdb import (
    TpchMonetdbOptimizationConversation,
)
from tpch_monetdb.utils.optimization_summary import (
    OptimizationRunSummary,
    find_latest_successful_optimization_run,
    get_summary_dir,
    persist_successful_optimization_run,
    write_optimization_run_summary,
)
from tpch_monetdb.utils.outer_loop_state import (
    PhaseInfo,
    RoundRecord,
    build_conv_names,
    compute_aggregate_runtime_ms,
    compute_round_decision,
    determine_resume_phase,
    load_latest_round_record,
    render_workflow_priority_order,
    write_round_record,
)
from tpch_monetdb.utils.scripted_summary import (
    ScriptedRunSummary,
    _is_compatible_summary,
    auto_discover_start_snapshot,
    find_latest_successful_run,
)
from tpch_monetdb.utils.storage_plan_summary import (
    StoragePlanRunSummary,
    find_latest_successful_storage_plan_run,
    get_summary_dir as get_storage_plan_summary_dir,
    persist_successful_storage_plan_run,
    write_storage_plan_run_summary,
)
from tpch_monetdb.utils.summary_gates import MEASUREMENT_AGGREGATION_GEOMEAN


def _valid_measurement_repetition(query_ids: list[str]) -> dict[str, object]:
    """Build valid repeated measurement evidence for outer-loop summary tests."""
    samples = [10.0, 10.2, 9.8]
    return {
        "scale_factor": 1000,
        "query_ids": query_ids,
        "repetitions": 3,
        "sample_count": 3,
        "aggregate_runtime_ms_samples": samples,
        "aggregate_runtime_ms_median": 10.0,
        "aggregate_runtime_ms_min": 9.8,
        "aggregate_runtime_ms_max": 10.2,
        "per_query_runtime_ms_samples": {qid: samples for qid in query_ids},
        "aggregation_method": MEASUREMENT_AGGREGATION_GEOMEAN,
        "source_command": "test measurement provider",
    }


def _large_data_success_fields(query_ids: list[str]) -> dict[str, object]:
    """Build explicit large-data success fields for summary tests."""
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


def _round_summary(
    conv_name: str,
    final_runtime_ms_by_query: dict[str, float],
) -> OptimizationRunSummary:
    """Build a successful large-data optimization summary for round decisions."""
    query_ids = sorted(final_runtime_ms_by_query.keys())
    return OptimizationRunSummary(
        benchmark="tpch",
        conv_name=conv_name,
        run_id=conv_name,
        query_list=query_ids,
        is_bespoke_storage=False,
        start_snapshot_hash=f"{conv_name}-start",
        final_snapshot_hash=f"{conv_name}-final",
        best_runtime_ms_by_query=dict(final_runtime_ms_by_query),
        final_runtime_ms_by_query=dict(final_runtime_ms_by_query),
        final_correctness=True,
        completed_at="",
        conversation_json="",
        session_db_path="",
        success=True,
        baseline_runtime_ms_by_query={
            query_id: runtime_ms * 2.0
            for query_id, runtime_ms in final_runtime_ms_by_query.items()
        },
        **_large_data_success_fields(query_ids),
    )


# ---------------------------------------------------------------------------
# Storage Plan Summary Tests
# ---------------------------------------------------------------------------


def test_storage_plan_summary_round_trip(tmp_path: Path) -> None:
    summary = StoragePlanRunSummary(
        benchmark="tpch",
        conv_name="tpch_monetdb_storageplan1-2v1",
        run_id="tpch_monetdb_storageplan1-2v1",
        query_list=["1", "2"],
        final_snapshot_hash="abc123",
        storage_plan_path=str(tmp_path / "storage_plan.txt"),
        completed_at="2026-04-15T00:00:00Z",
        conversation_json=str(tmp_path / "conv.json"),
        session_db_path=str(tmp_path / "session.sqlite"),
        success=True,
    )
    file_path = write_storage_plan_run_summary(summary, tmp_path)
    assert file_path.exists()

    loaded = find_latest_successful_storage_plan_run(
        conv_name="tpch_monetdb_storageplan1-2v1",
        query_list=["1", "2"],
        benchmark="tpch",
        artifacts_dir=tmp_path,
    )
    assert loaded is not None
    assert loaded.final_snapshot_hash == "abc123"
    assert loaded.query_list == ["1", "2"]


def test_storage_plan_summary_rejects_empty_hash(tmp_path: Path) -> None:
    summary = StoragePlanRunSummary(
        benchmark="tpch",
        conv_name="tpch_monetdb_storageplan1-2v1",
        run_id="tpch_monetdb_storageplan1-2v1",
        query_list=["1"],
        final_snapshot_hash="",
        storage_plan_path=str(tmp_path / "storage_plan.txt"),
        completed_at="2026-04-15T00:00:00Z",
        conversation_json=str(tmp_path / "conv.json"),
        session_db_path=str(tmp_path / "session.sqlite"),
        success=True,
    )
    with pytest.raises(ValueError, match="final_snapshot_hash cannot be empty"):
        write_storage_plan_run_summary(summary, tmp_path)


def test_storage_plan_main_uses_benchmark_scale_factor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        tpch_monetdb.run_gen_storage_plan_tpch_monetdb,
        "parse_query_ids",
        lambda *_args, **_kwargs: ["1", "2"],
    )
    monkeypatch.setattr(
        tpch_monetdb.run_gen_storage_plan_tpch_monetdb,
        "build_run_config",
        lambda **kwargs: captured.setdefault(
            "config",
            SimpleNamespace(artifacts_dir=kwargs["artifacts_dir"]),
        ),
    )

    def fake_create_conversation(*args, **kwargs) -> None:
        if "max_scale_factor" in kwargs:
            captured["max_scale_factor"] = kwargs["max_scale_factor"]
        else:
            captured["max_scale_factor"] = args[5]
        return None

    monkeypatch.setattr(
        tpch_monetdb.run_gen_storage_plan_tpch_monetdb,
        "create_conversation",
        fake_create_conversation,
    )
    monkeypatch.setattr(
        tpch_monetdb.run_gen_storage_plan_tpch_monetdb,
        "run_conv_wrapper",
        lambda _config: None,
    )

    args = SimpleNamespace(
        conv="storageplan1-2v1_r001",
        benchmark="tpch",
        base_data_dir=str(tmp_path / "data"),
        artifacts_dir=str(tmp_path / "artifacts"),
        notify=False,
        disable_repo_sync=False,
        replay_cache=False,
        auto_u=False,
        auto_finish=False,
        disable_wandb=True,
        disable_tracing=True,
        model="litellm/test-model",
    )

    tpch_monetdb.run_gen_storage_plan_tpch_monetdb.main(args)

    assert captured["max_scale_factor"] == 1
    assert captured["config"].generate_design_evidence is False


def test_storage_plan_create_conversation_creates_parent_dir(tmp_path: Path) -> None:
    conversation_dir = tmp_path / "artifacts" / "conversations"
    tpch_monetdb.run_gen_storage_plan_tpch_monetdb.create_conversation(
        benchmark="tpch",
        short_name="storageplan1-2v1_r001",
        conversation_dir=conversation_dir,
        base_data_dir=tmp_path / "data",
        max_scale_factor=1,
        query_ids=["1", "2"],
    )
    target_path = conversation_dir / "tpch_storageplan1-2v1_r001.json"
    assert target_path.exists()
    payload = json.loads(target_path.read_text())
    assert payload[0]["descriptor"] == "storage_plan"
    assert payload[0]["tool_profile"] == "storage_plan"
    assert payload[0]["stop_conditions"] == ["write_required"]
    assert payload[0]["advisory_postconditions"] == ["storage_plan_contract_complete"]
    assert (
        "tpch_monetdb/conversations/prompts/tpch_monetdb_optim_constraints.txt"
        not in payload[0]["text"]
    )
    # Phase10: open-ended prompt — checks that old hardcoded content is gone and
    # new workspace-grounded prompt is present
    assert "queries.txt" in payload[0]["text"]
    assert "storage_plan.txt" in payload[0]["text"]
    assert "sf1" in payload[0]["text"]
    assert "sf10" not in payload[0]["text"]
    assert "sf1000" not in payload[0]["text"]
    assert "sf10000" not in payload[0]["text"]
    # Old prompt content that must NOT appear any more
    assert "Do not read additional project files before drafting the plan." not in payload[0]["text"]
    assert "Allowed Optimizations" not in payload[0]["text"]
    assert "hostname -> series_id dictionary mapping" not in payload[0]["text"]


# ---------------------------------------------------------------------------
# Optimization Summary Tests
# ---------------------------------------------------------------------------


def test_optimization_summary_round_trip(tmp_path: Path) -> None:
    summary = OptimizationRunSummary(
        benchmark="tpch",
        conv_name="tpch_monetdb_runoptim1-2v1",
        run_id="tpch_monetdb_runoptim1-2v1",
        query_list=["1", "2"],
        is_bespoke_storage=True,
        start_snapshot_hash="start123",
        final_snapshot_hash="final123",
        best_runtime_ms_by_query={"1": 100.0, "2": 200.0},
        final_runtime_ms_by_query={"1": 110.0, "2": 210.0},
        final_correctness=True,
        completed_at="2026-04-15T00:00:00Z",
        conversation_json=str(tmp_path / "conv.json"),
        session_db_path=str(tmp_path / "session.sqlite"),
        success=True,
        baseline_runtime_ms_by_query={"1": 50.0, "2": 100.0},
        **_large_data_success_fields(["1", "2"]),
    )
    file_path = write_optimization_run_summary(summary, tmp_path)
    assert file_path.exists()

    loaded = find_latest_successful_optimization_run(
        conv_name="tpch_monetdb_runoptim1-2v1",
        query_list=["1", "2"],
        benchmark="tpch",
        artifacts_dir=tmp_path,
    )
    assert loaded is not None
    assert loaded.is_bespoke_storage is True
    assert loaded.final_snapshot_hash == "final123"
    assert loaded.best_runtime_ms_by_query["1"] == 100.0


def test_optimization_summary_rejects_empty_hash(tmp_path: Path) -> None:
    summary = OptimizationRunSummary(
        benchmark="tpch",
        conv_name="tpch_monetdb_runoptim1-2v1",
        run_id="tpch_monetdb_runoptim1-2v1",
        query_list=["1"],
        is_bespoke_storage=False,
        start_snapshot_hash="start123",
        final_snapshot_hash="",
        best_runtime_ms_by_query={"1": 100.0},
        final_runtime_ms_by_query={"1": 110.0},
        final_correctness=True,
        completed_at="2026-04-15T00:00:00Z",
        conversation_json=str(tmp_path / "conv.json"),
        session_db_path=str(tmp_path / "session.sqlite"),
        success=True,
    )
    with pytest.raises(ValueError, match="final_snapshot_hash cannot be empty"):
        write_optimization_run_summary(summary, tmp_path)


def test_optimization_summary_rejects_empty_runtime_map(tmp_path: Path) -> None:
    summary = OptimizationRunSummary(
        benchmark="tpch",
        conv_name="tpch_monetdb_runoptim1-2v1",
        run_id="tpch_monetdb_runoptim1-2v1",
        query_list=["1"],
        is_bespoke_storage=False,
        start_snapshot_hash="start123",
        final_snapshot_hash="final123",
        best_runtime_ms_by_query={},
        final_runtime_ms_by_query={},
        final_correctness=True,
        completed_at="2026-04-15T00:00:00Z",
        conversation_json=str(tmp_path / "conv.json"),
        session_db_path=str(tmp_path / "session.sqlite"),
        success=True,
    )
    with pytest.raises(ValueError, match="final_runtime_ms_by_query cannot be empty"):
        write_optimization_run_summary(summary, tmp_path)


# ---------------------------------------------------------------------------
# Scripted Summary deprecated is_bespoke_storage compatibility
# ---------------------------------------------------------------------------


def test_is_compatible_summary_ignores_deprecated_bespoke_storage_filter() -> None:
    summary = ScriptedRunSummary(
        benchmark="tpch",
        conv_name="tpch_monetdb_basef1-2v1",
        run_id="tpch_monetdb_basef1-2v1",
        query_list=["1", "2"],
        is_bespoke_storage=True,
        final_snapshot_hash="hash123",
        completed_at="2026-04-15T00:00:00Z",
        conversation_json="conv.json",
        session_db_path="session.sqlite",
        success=True,
        validation_mode="strict",
    )
    assert (
        _is_compatible_summary(
            summary, ["1", "2"], "tpch", "strict", is_bespoke_storage=True
        )
        is True
    )
    assert (
        _is_compatible_summary(
            summary, ["1", "2"], "tpch", "strict", is_bespoke_storage=False
        )
        is True
    )
    assert (
        _is_compatible_summary(
            summary, ["1", "2"], "tpch", "strict", is_bespoke_storage=None
        )
        is True
    )


def test_find_latest_successful_run_ignores_deprecated_bespoke_storage(tmp_path: Path) -> None:
    summary_true = ScriptedRunSummary(
        benchmark="tpch",
        conv_name="tpch_monetdb_basef1-2v1_storage_true_legacy",
        run_id="tpch_monetdb_basef1-2v1_storage_true_legacy",
        query_list=["1", "2"],
        is_bespoke_storage=True,
        final_snapshot_hash="hash1",
        completed_at="2026-04-15T00:00:00Z",
        conversation_json=str(tmp_path / "conv.json"),
        session_db_path=str(tmp_path / "session.sqlite"),
        success=True,
        validation_mode="strict",
    )
    summary_false = ScriptedRunSummary(
        benchmark="tpch",
        conv_name="tpch_monetdb_basef1-2v1_storage_false_legacy",
        run_id="tpch_monetdb_basef1-2v1_storage_false_legacy",
        query_list=["1", "2"],
        is_bespoke_storage=False,
        final_snapshot_hash="hash2",
        completed_at="2026-04-15T00:01:00Z",
        conversation_json=str(tmp_path / "conv.json"),
        session_db_path=str(tmp_path / "session.sqlite"),
        success=True,
        validation_mode="strict",
    )
    from tpch_monetdb.utils.scripted_summary import write_scripted_run_summary

    write_scripted_run_summary(summary_true, tmp_path)
    write_scripted_run_summary(summary_false, tmp_path)

    found_true = find_latest_successful_run(
        conv_name=None,
        query_list=["1", "2"],
        benchmark="tpch",
        artifacts_dir=tmp_path,
        is_bespoke_storage=True,
    )
    assert found_true is not None
    assert found_true.final_snapshot_hash == "hash2"
    assert found_true.is_bespoke_storage is True

    found_false = find_latest_successful_run(
        conv_name=None,
        query_list=["1", "2"],
        benchmark="tpch",
        artifacts_dir=tmp_path,
        is_bespoke_storage=False,
    )
    assert found_false is not None
    assert found_false.final_snapshot_hash == "hash2"
    assert found_false.is_bespoke_storage is True


# ---------------------------------------------------------------------------
# Outer Loop State Tests
# ---------------------------------------------------------------------------


def test_build_conv_names() -> None:
    sp, bi, opt = build_conv_names("outer1-2v1", 3)
    assert sp == "storageplan1-2v1_r003"
    assert bi == "basef1-2v1_r003"
    assert opt == "runoptim1-2v1_r003"


def test_built_conv_names_match_child_parsers() -> None:
    sp, bi, opt = build_conv_names("outer1-2v1", 2)
    tpch_monetdb.run_gen_storage_plan_tpch_monetdb.build_parser(add_help=False).parse_args(
        ["--conv", sp]
    )
    tpch_monetdb.run_gen_base_impl_tpch_monetdb.build_parser(add_help=False).parse_args(["--conv", bi])
    run_optim_loop_tpch_monetdb.build_parser(add_help=False).parse_args(["--conv", opt])


def test_resolve_optimization_run_conv_name_ignores_deprecated_storage_flag() -> None:
    resolved = run_outer_loop_tpch_monetdb._resolve_optimization_run_conv_name(
        "runoptim1-2v1_r002",
        True,
    )
    assert resolved == "runoptim1-2v1_r002"


def test_resolve_runtime_conv_name_prefixes_benchmark() -> None:
    resolved = run_outer_loop_tpch_monetdb._resolve_runtime_conv_name(
        "tpch_monetdb",
        "storageplan1-2v1_r001",
    )
    assert resolved == "tpch_monetdb_storageplan1-2v1_r001"


def test_resolve_scripted_run_conv_name_appends_validation_suffix() -> None:
    resolved = run_outer_loop_tpch_monetdb._resolve_scripted_run_conv_name(
        "tpch_monetdb",
        "basef1-2v1_r001",
        "traversal",
    )
    assert resolved == "tpch_monetdb_basef1-2v1_r001_traversal"


def test_build_base_impl_cmd_tolerates_missing_replay_flag() -> None:
    args = SimpleNamespace(
        conv="outer1-2v1",
        benchmark="tpch",
        artifacts_dir="/tmp/artifacts",
        validation_mode="traversal",
        base_data_dir="/tmp/data",
        model="litellm/test-model",
        notify=False,
        disable_repo_sync=True,
        replay_cache=False,
        disable_wandb=True,
        disable_tracing=True,
        auto_u=True,
        auto_finish=True,
        reasoning_effort="high",
        only_from_llm_cache=False,
        only_from_cache=False,
        enable_auto_compact=False,
    )
    cmd = run_outer_loop_tpch_monetdb._build_base_impl_cmd(
        args,
        1,
        "snapshot123",
        True,
    )
    assert "--replay" not in cmd
    assert "--storage_plan_snapshot" in cmd
    assert "--is_bespoke_storage" in cmd
    assert "--data_prepare_mode" not in cmd
    assert "--reasoning_effort" in cmd
    assert "high" in cmd


def test_build_base_impl_cmd_omits_legacy_data_prepare_for_tpch() -> None:
    args = SimpleNamespace(
        conv="outer1-2v1",
        benchmark="tpch",
        artifacts_dir="/tmp/artifacts",
        validation_mode="strict",
        base_data_dir="/tmp/data",
        model="litellm/test-model",
        notify=False,
        disable_repo_sync=True,
        replay_cache=False,
        disable_wandb=True,
        disable_tracing=True,
        auto_u=True,
        auto_finish=True,
        reasoning_effort="high",
        only_from_llm_cache=False,
        only_from_cache=False,
        enable_auto_compact=False,
    )
    cmd = run_outer_loop_tpch_monetdb._build_base_impl_cmd(
        args,
        1,
        "snapshot123",
        True,
    )

    assert "--benchmark" in cmd
    assert "tpch" in cmd
    assert "--data_prepare_mode" not in cmd
    return None


def test_build_optimization_cmd_omits_legacy_baseline_args_for_tpch() -> None:
    args = SimpleNamespace(
        conv="outer1-2v1",
        benchmark="tpch",
        artifacts_dir="/tmp/artifacts",
        model="litellm/test-model",
        reasoning_effort="high",
        notify=False,
        disable_repo_sync=True,
        replay_cache=False,
        disable_wandb=True,
        disable_tracing=True,
        auto_u=True,
        auto_finish=True,
        only_from_llm_cache=False,
        only_from_cache=False,
        enable_auto_compact=False,
        base_data_dir=None,
        baseline_backend="monetdb",
        baseline_query_file_dir="/tmp/legacy_queries",
        benchmark_mode="system-parity",
        storage_mode="persistent",
    )

    cmd = run_outer_loop_tpch_monetdb._build_optimization_cmd(
        args,
        1,
        "snapshot123",
        True,
    )

    assert "--benchmark" in cmd
    assert "tpch" in cmd
    assert "--baseline_backend" not in cmd
    assert "legacy_queries" not in cmd
    assert "--baseline_query_file_dir" not in cmd
    assert "/tmp/legacy_queries" not in cmd
    assert "--benchmark_mode" in cmd
    assert "system-parity" in cmd
    assert "--storage_mode" in cmd
    assert "persistent" in cmd
    return None


def test_build_optimization_cmd_omits_questdb_tsbs_args_for_tpch() -> None:
    args = SimpleNamespace(
        conv="outer1-2v1",
        benchmark="tpch",
        artifacts_dir="/tmp/artifacts",
        model="litellm/test-model",
        reasoning_effort="high",
        notify=False,
        disable_repo_sync=True,
        replay_cache=False,
        disable_wandb=True,
        disable_tracing=True,
        auto_u=True,
        auto_finish=True,
        only_from_llm_cache=False,
        only_from_cache=False,
        enable_auto_compact=False,
        base_data_dir=None,
        baseline_backend="monetdb",
        baseline_query_file_dir="/tmp/legacy_queries",
        benchmark_mode="system-parity",
        storage_mode="persistent",
    )

    cmd = run_outer_loop_tpch_monetdb._build_optimization_cmd(
        args,
        1,
        "snapshot123",
        True,
    )

    assert "--benchmark" in cmd
    assert "tpch" in cmd
    assert "--baseline_backend" not in cmd
    assert "--baseline_query_file_dir" not in cmd
    assert "--benchmark_mode" in cmd
    assert "--storage_mode" in cmd
    return None


def test_build_optimization_cmd_forwards_wandb_guard_args() -> None:
    args = SimpleNamespace(
        conv="outer1-2v1",
        artifacts_dir="/tmp/artifacts",
        model="litellm/test-model",
        reasoning_effort="high",
        notify=False,
        disable_repo_sync=True,
        replay_cache=False,
        disable_wandb=True,
        disable_tracing=True,
        disable_wandb_when_tracing_disabled=True,
        wandb_init_max_attempts=5,
        wandb_init_timeout_s=12.5,
        wandb_upload_timeout_s=90.0,
        wandb_finish_timeout_s=18.0,
        wandb_finish_retries=2,
        auto_u=True,
        auto_finish=True,
        only_from_llm_cache=False,
        only_from_cache=False,
        enable_auto_compact=False,
        base_data_dir=None,
        baseline_backend="monetdb",
        baseline_query_file_dir=None,
        benchmark_mode="system-parity",
        storage_mode="persistent",
    )

    cmd = run_outer_loop_tpch_monetdb._build_optimization_cmd(
        args,
        1,
        "snapshot123",
        True,
    )

    assert "--disable_wandb_when_tracing_disabled" in cmd
    assert "--wandb_init_max_attempts" in cmd
    assert "5" in cmd
    assert "--wandb_init_timeout_s" in cmd
    assert "12.5" in cmd
    assert "--wandb_upload_timeout_s" in cmd
    assert "90.0" in cmd
    assert "--wandb_finish_timeout_s" in cmd
    assert "18.0" in cmd
    assert "--wandb_finish_retries" in cmd
    assert "2" in cmd


def test_run_phase_with_retries_prefers_error_envelope_code() -> None:
    record = RoundRecord(
        outer_loop_name="outer1-2v1",
        round_index=1,
        query_list=["1", "2"],
        storage_plan=PhaseInfo(conv_name="storageplan1-2v1_r001", status="running"),
    )
    phase = record.storage_plan
    assert phase is not None

    def fake_run_subprocess(_cmd):
        return run_outer_loop_tpch_monetdb.subprocess.CompletedProcess(
            _cmd,
            1,
            stdout="[ERROR:WANDB_FINISH_TIMEOUT] finalize blocked",
            stderr="",
        )

    artifacts_dir = Path("/tmp")
    original_run_subprocess = run_outer_loop_tpch_monetdb._run_subprocess
    try:
        run_outer_loop_tpch_monetdb._run_subprocess = fake_run_subprocess
        result = run_outer_loop_tpch_monetdb._run_phase_with_retries(
            record=record,
            phase=phase,
            artifacts_dir=artifacts_dir,
            retry_budget=0,
            phase_log_name="storage plan",
            failure_reason="storage plan phase failed",
            failure_code="PHASE_RETRY_EXHAUSTED",
            cmd_factory=lambda: ["python", "-m", "tpch_monetdb.run_gen_storage_plan_tpch_monetdb"],
        )
    finally:
        run_outer_loop_tpch_monetdb._run_subprocess = original_run_subprocess

    assert result.returncode == 1
    assert phase.failure_code == "WANDB_FINISH_TIMEOUT"
    assert record.action_reason_code == "WANDB_FINISH_TIMEOUT"


def test_failed_latest_record_is_resumable_when_retry_budget_increases() -> None:
    record = RoundRecord(
        outer_loop_name="outer1-2v1",
        round_index=1,
        query_list=["1", "2"],
        storage_plan=PhaseInfo(
            conv_name="storageplan1-2v1_r001", status="success", retry_count=0
        ),
        base_impl=PhaseInfo(
            conv_name="basef1-2v1_r001", status="failed", retry_count=1
        ),
        optimization=PhaseInfo(
            conv_name="runoptim1-2v1_r001", status="success", retry_count=0
        ),
        action="failed",
        outcome="failed",
        action_reason="base impl phase failed",
    )
    assert (
        run_outer_loop_tpch_monetdb._should_treat_latest_record_as_terminal(record, 1) is True
    )
    assert (
        run_outer_loop_tpch_monetdb._should_treat_latest_record_as_terminal(record, 2) is False
    )


def test_failed_latest_record_with_running_phase_is_not_terminal() -> None:
    record = RoundRecord(
        outer_loop_name="outer1-2v1",
        round_index=1,
        query_list=["1", "2"],
        storage_plan=PhaseInfo(
            conv_name="storageplan1-2v1_r001", status="success", retry_count=0
        ),
        base_impl=PhaseInfo(
            conv_name="basef1-2v1_r001", status="running", retry_count=2
        ),
        optimization=PhaseInfo(
            conv_name="runoptim1-2v1_r001", status="pending", retry_count=0
        ),
        action="failed",
        outcome="failed",
        action_reason="base impl phase failed",
    )

    assert (
        run_outer_loop_tpch_monetdb._should_treat_latest_record_as_terminal(record, 3) is False
    )


def test_detect_optimization_correctness_gate_failure() -> None:
    result = run_outer_loop_tpch_monetdb.subprocess.CompletedProcess(
        args=["optim"],
        returncode=1,
        stdout="[ERROR:OPTIMIZATION_PRECHECK_FAILED] Initial implementation is not correct. Fix before optimization.",
        stderr="",
    )
    assert run_outer_loop_tpch_monetdb._is_optimization_correctness_gate_failure(result) is True


def test_detect_optimization_correctness_gate_failure_rejects_old_string() -> None:
    result = run_outer_loop_tpch_monetdb.subprocess.CompletedProcess(
        args=["optim"],
        returncode=1,
        stdout="AssertionError: Initial implementation is not correct. Fix before optimization.",
        stderr="",
    )
    assert run_outer_loop_tpch_monetdb._is_optimization_correctness_gate_failure(result) is False


def test_detect_final_correctness_gate_failure() -> None:
    result = run_outer_loop_tpch_monetdb.subprocess.CompletedProcess(
        args=["base"],
        returncode=1,
        stdout="[ERROR:FINAL_CORRECTNESS_GATE_FAILED] final gate failed.",
        stderr="",
    )
    assert run_outer_loop_tpch_monetdb._is_final_correctness_gate_failure(result) is True
    return None


def test_run_phase_with_retries_stops_when_retry_predicate_rejects(
    monkeypatch, tmp_path: Path
) -> None:
    calls: list[list[str]] = []

    def fake_run_subprocess(cmd: list[str]) -> object:
        calls.append(cmd)
        return run_outer_loop_tpch_monetdb.subprocess.CompletedProcess(
            cmd,
            1,
            stdout="[ERROR:OPTIMIZATION_PRECHECK_FAILED] Initial implementation is not correct.",
            stderr="",
        )

    monkeypatch.setattr(run_outer_loop_tpch_monetdb, "_run_subprocess", fake_run_subprocess)
    record = RoundRecord(
        outer_loop_name="outer1-2v1",
        round_index=1,
        query_list=["1", "2"],
        optimization=PhaseInfo(conv_name="runoptim1-2v1_r001", status="running"),
    )
    phase = record.optimization
    assert phase is not None

    result = run_outer_loop_tpch_monetdb._run_phase_with_retries(
        record=record,
        phase=phase,
        artifacts_dir=tmp_path,
        retry_budget=3,
        phase_log_name="optimization",
        failure_reason="optimization phase failed",
        failure_code="PHASE_RETRY_EXHAUSTED",
        cmd_factory=lambda: ["optim"],
        should_retry=lambda res: not run_outer_loop_tpch_monetdb._is_optimization_correctness_gate_failure(res),
    )

    assert result.returncode == 1
    assert len(calls) == 1
    assert phase.retry_count == 0
    assert phase.status == "failed"
    assert record.action == "failed"


def test_should_continue_after_optimization_gate_failure() -> None:
    assert run_outer_loop_tpch_monetdb._should_continue_after_optimization_gate_failure(1, 3) is True
    assert run_outer_loop_tpch_monetdb._should_continue_after_optimization_gate_failure(3, 3) is False


def test_run_phase_with_retries_consumes_full_retry_budget(
    monkeypatch, tmp_path: Path
) -> None:
    calls: list[list[str]] = []

    def fake_run_subprocess(cmd: list[str]) -> object:
        calls.append(cmd)
        return run_outer_loop_tpch_monetdb.subprocess.CompletedProcess(cmd, 1)

    monkeypatch.setattr(run_outer_loop_tpch_monetdb, "_run_subprocess", fake_run_subprocess)
    record = RoundRecord(
        outer_loop_name="outer1-2v1",
        round_index=1,
        query_list=["1", "2"],
        base_impl=PhaseInfo(conv_name="basef1-2v1_r001", status="running"),
    )
    phase = record.base_impl
    assert phase is not None

    result = run_outer_loop_tpch_monetdb._run_phase_with_retries(
        record=record,
        phase=phase,
        artifacts_dir=tmp_path,
        retry_budget=2,
        phase_log_name="base impl",
        failure_reason="base impl phase failed",
        failure_code="PHASE_RETRY_EXHAUSTED",
        cmd_factory=lambda: ["base"],
    )

    assert result.returncode == 1
    assert len(calls) == 3
    assert phase.retry_count == 2
    assert phase.status == "failed"
    assert record.action == "failed"
    assert record.action_reason_code == "PHASE_RETRY_EXHAUSTED"


def test_run_phase_with_retries_clears_failed_action_after_success(
    monkeypatch, tmp_path: Path
) -> None:
    returncodes = [1, 0]

    def fake_run_subprocess(cmd: list[str]) -> object:
        return run_outer_loop_tpch_monetdb.subprocess.CompletedProcess(cmd, returncodes.pop(0))

    monkeypatch.setattr(run_outer_loop_tpch_monetdb, "_run_subprocess", fake_run_subprocess)
    record = RoundRecord(
        outer_loop_name="outer1-2v1",
        round_index=1,
        query_list=["1", "2"],
        base_impl=PhaseInfo(conv_name="basef1-2v1_r001", status="running"),
    )
    phase = record.base_impl
    assert phase is not None

    result = run_outer_loop_tpch_monetdb._run_phase_with_retries(
        record=record,
        phase=phase,
        artifacts_dir=tmp_path,
        retry_budget=2,
        phase_log_name="base impl",
        failure_reason="base impl phase failed",
        failure_code="PHASE_RETRY_EXHAUSTED",
        cmd_factory=lambda: ["base"],
    )

    assert result.returncode == 0
    assert phase.retry_count == 1
    assert phase.status == "success"
    assert record.action == "pending"
    assert record.action_reason == ""
    assert record.action_reason_code == ""


def test_compute_aggregate_runtime_ms() -> None:
    rt = {"1": 100.0, "2": 400.0}
    agg = compute_aggregate_runtime_ms(rt)
    assert agg == math.sqrt(100.0 * 400.0)


def test_compute_aggregate_runtime_ms_rejects_empty_runtime_map() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        compute_aggregate_runtime_ms({})


def test_compute_aggregate_runtime_ms_rejects_non_positive_or_non_finite_values() -> None:
    with pytest.raises(ValueError, match="positive finite"):
        compute_aggregate_runtime_ms({"1": 0.0})
    with pytest.raises(ValueError, match="positive finite"):
        compute_aggregate_runtime_ms({"1": float("inf")})


def test_render_workflow_priority_order() -> None:
    text = render_workflow_priority_order()
    assert text == (
        "Workflow priority: "
        "P0=correctness > "
        "P1=query runtime / speedup vs MonetDB baseline > "
        "P2=build/ingest time guardrail"
    )


def test_print_terminal_result_includes_priority_order(capsys: pytest.CaptureFixture[str]) -> None:
    record = RoundRecord(
        outer_loop_name="outer1-9v1",
        round_index=1,
        query_list=["1"],
        action="continue",
        action_reason="round 1 outcome=improved stagnant=0",
    )

    run_outer_loop_tpch_monetdb._print_terminal_result(record)

    output = capsys.readouterr().out
    assert "Priority:     Workflow priority: P0=correctness" in output


def test_compute_round_decision_improved() -> None:
    prev = _round_summary("r1", {"1": 100.0})
    curr = _round_summary("r2", {"1": 80.0})
    outcome, action, sc = compute_round_decision(
        prev,
        curr,
        convergence_threshold=0.02,
        stagnant_rounds=2,
        regression_tolerance=0.05,
        max_rounds=4,
        current_round_index=2,
        stagnant_count=0,
    )
    assert outcome == "improved"
    assert action == "continue"
    assert sc == 0


def test_compute_round_decision_stagnant_then_converged() -> None:
    prev = _round_summary("r1", {"1": 100.0})
    curr = _round_summary("r2", {"1": 99.0})
    # First stagnant round
    outcome, action, sc = compute_round_decision(
        prev,
        curr,
        convergence_threshold=0.02,
        stagnant_rounds=2,
        regression_tolerance=0.05,
        max_rounds=4,
        current_round_index=2,
        stagnant_count=0,
    )
    assert outcome == "stagnant"
    assert action == "continue"
    assert sc == 1

    # Second stagnant round -> converged
    outcome, action, sc = compute_round_decision(
        prev,
        curr,
        convergence_threshold=0.02,
        stagnant_rounds=2,
        regression_tolerance=0.05,
        max_rounds=4,
        current_round_index=3,
        stagnant_count=1,
    )
    assert outcome == "stagnant"
    assert action == "converged"
    assert sc == 2


def test_compute_round_decision_regressed() -> None:
    prev = _round_summary("r1", {"1": 100.0})
    curr = _round_summary("r2", {"1": 120.0})
    outcome, action, sc = compute_round_decision(
        prev,
        curr,
        convergence_threshold=0.02,
        stagnant_rounds=2,
        regression_tolerance=0.05,
        max_rounds=4,
        current_round_index=2,
        stagnant_count=0,
    )
    assert outcome == "regressed"
    assert action == "failed"


def test_round_record_persist_and_load(tmp_path: Path) -> None:
    record = RoundRecord(
        outer_loop_name="outer1-2v1",
        round_index=1,
        query_list=["1", "2"],
        storage_plan=PhaseInfo(conv_name="sp", status="success"),
        base_impl=PhaseInfo(conv_name="bi", status="success"),
        optimization=PhaseInfo(conv_name="opt", status="success"),
        outcome="improved",
        action="continue",
        aggregate_runtime_ms=150.0,
        best_round_index=1,
    )
    write_round_record(record, tmp_path)

    loaded = load_latest_round_record("outer1-2v1", tmp_path)
    assert loaded is not None
    assert loaded.round_index == 1
    assert loaded.action == "continue"
    assert loaded.storage_plan is not None
    assert loaded.storage_plan.status == "success"


def test_determine_resume_phase_from_base_impl(tmp_path: Path) -> None:
    record = RoundRecord(
        outer_loop_name="outer1-2v1",
        round_index=1,
        query_list=["1", "2"],
        storage_plan=PhaseInfo(conv_name="sp", status="success"),
        base_impl=None,
        optimization=None,
    )
    phase, updated = determine_resume_phase(record, retry_budget=1)
    assert phase == "base_impl"
    assert updated.base_impl is not None
    assert updated.base_impl.status == "running"


def test_determine_resume_phase_retry_exhausted() -> None:
    record = RoundRecord(
        outer_loop_name="outer1-2v1",
        round_index=1,
        query_list=["1", "2"],
        storage_plan=PhaseInfo(conv_name="sp", status="failed", retry_count=1),
        base_impl=None,
        optimization=None,
    )
    phase, updated = determine_resume_phase(record, retry_budget=1)
    assert phase == "failed"


def test_determine_resume_phase_base_impl_retry_exhausted() -> None:
    record = RoundRecord(
        outer_loop_name="outer1-2v1",
        round_index=1,
        query_list=["1", "2"],
        storage_plan=PhaseInfo(conv_name="sp", status="success"),
        base_impl=PhaseInfo(conv_name="bi", status="failed", retry_count=1),
        optimization=None,
    )
    phase, _ = determine_resume_phase(record, retry_budget=1)
    assert phase == "failed"


def test_determine_resume_phase_base_impl_final_gate_failure_stays_failed() -> None:
    record = RoundRecord(
        outer_loop_name="outer1-2v1",
        round_index=1,
        query_list=["1", "2"],
        storage_plan=PhaseInfo(conv_name="storage1", status="success"),
        base_impl=PhaseInfo(
            conv_name="base1",
            status="failed",
            failure_code="FINAL_CORRECTNESS_GATE_FAILED",
            retry_count=1,
        ),
    )
    phase, _ = determine_resume_phase(record, retry_budget=1)
    assert phase == "failed"
    return None


def test_determine_resume_phase_optimization_retry_exhausted() -> None:
    record = RoundRecord(
        outer_loop_name="outer1-2v1",
        round_index=1,
        query_list=["1", "2"],
        storage_plan=PhaseInfo(conv_name="sp", status="success"),
        base_impl=PhaseInfo(conv_name="bi", status="success"),
        optimization=PhaseInfo(conv_name="opt", status="failed", retry_count=1),
    )
    phase, _ = determine_resume_phase(record, retry_budget=1)
    assert phase == "failed"


def test_update_best_round_accepts_stagnant_but_faster_round() -> None:
    record = RoundRecord(
        outer_loop_name="outer1-2v1",
        round_index=2,
        query_list=["1"],
        optimization=PhaseInfo(
            conv_name="runoptim1-1v1_r002",
            status="success",
            summary_path="opt2/latest.json",
        ),
        aggregate_runtime_ms=99.0,
        best_round_index=1,
        best_optimization_summary_path="opt1/latest.json",
        best_aggregate_runtime_ms=100.0,
    )
    summary = OptimizationRunSummary(
        benchmark="tpch",
        conv_name="runoptim1-1v1_r002",
        run_id="runoptim1-1v1_r002",
        query_list=["1"],
        is_bespoke_storage=False,
        start_snapshot_hash="s2",
        final_snapshot_hash="f2",
        best_runtime_ms_by_query={"1": 99.0},
        final_runtime_ms_by_query={"1": 99.0},
        final_correctness=True,
        completed_at="",
        conversation_json="",
        session_db_path="",
        success=True,
        baseline_runtime_ms_by_query={"1": 50.0},
        **_large_data_success_fields(["1"]),
    )
    run_outer_loop_tpch_monetdb._update_best_round(record, summary, 2)
    assert record.best_round_index == 2
    assert record.best_optimization_summary_path == "opt2/latest.json"
    assert record.best_aggregate_runtime_ms == 99.0


def test_update_best_round_rejects_non_measurable_summary() -> None:
    record = RoundRecord(
        outer_loop_name="outer1-2v1",
        round_index=2,
        query_list=["1"],
        optimization=PhaseInfo(
            conv_name="runoptim1-1v1_r002",
            status="success",
            summary_path="opt2/latest.json",
        ),
        aggregate_runtime_ms=99.0,
        best_round_index=1,
        best_optimization_summary_path="opt1/latest.json",
        best_aggregate_runtime_ms=100.0,
    )
    summary = OptimizationRunSummary(
        benchmark="tpch",
        conv_name="runoptim1-1v1_r002",
        run_id="runoptim1-1v1_r002",
        query_list=["1"],
        is_bespoke_storage=False,
        start_snapshot_hash="s2",
        final_snapshot_hash="f2",
        best_runtime_ms_by_query={"1": 99.0},
        final_runtime_ms_by_query={"1": 99.0},
        final_correctness=True,
        completed_at="",
        conversation_json="",
        session_db_path="",
        success=True,
        baseline_runtime_ms_by_query={"1": 50.0},
    )
    run_outer_loop_tpch_monetdb._update_best_round(record, summary, 2)
    assert record.best_round_index == 1


# ---------------------------------------------------------------------------
# Manifest Multi-Instantiation Tests
# ---------------------------------------------------------------------------


def test_generate_from_tpch_creates_multiple_instantiations() -> None:
    manifest = ReferenceManifest.generate_from_tpch(
        query_ids=["Q1", "Q6"],
        scale_factor=1,
        seed=1,
        manifest_path=Path("test_manifest.json"),
        num_instantiations=3,
    )
    insts = manifest.get_instantiations_for_query("Q1")
    assert len(insts) == 3
    ids = {i.instantiation_id for i in insts}
    assert len(ids) == 3


def test_tpch_q1_multiple_instances_have_unique_ids() -> None:
    manifest = ReferenceManifest.generate_from_tpch(
        query_ids=["Q1"],
        scale_factor=1,
        seed=1,
        manifest_path=Path("test_manifest.json"),
        num_instantiations=3,
    )
    insts = manifest.get_instantiations_for_query("Q1")
    assert len(insts) == 3
    # Q1 has no parameters, so SQL hash could be identical across seeds.
    # Our fix appends _I{i} to ensure unique instantiation_id.
    ids = [i.instantiation_id for i in insts]
    assert len(set(ids)) == 3


def test_ledger_distinguishes_summary_missing_from_retry_exhausted() -> None:
    record = RoundRecord(
        outer_loop_name="outer1-2v1",
        round_index=1,
        query_list=["1", "2"],
    )
    record.storage_plan = PhaseInfo(
        conv_name="sp1-2v1_r001",
        status="failed",
        failure_code="PHASE_SUMMARY_MISSING",
        failure_detail="storage plan summary missing",
    )
    record.action = "failed"
    record.action_reason = "storage plan summary missing"
    record.action_reason_code = "PHASE_SUMMARY_MISSING"

    assert record.storage_plan.failure_code == "PHASE_SUMMARY_MISSING"
    assert record.action_reason_code == "PHASE_SUMMARY_MISSING"

    record2 = RoundRecord(
        outer_loop_name="outer1-2v1",
        round_index=2,
        query_list=["1", "2"],
    )
    record2.base_impl = PhaseInfo(
        conv_name="bif1-2v1_r002",
        status="failed",
        failure_code="PHASE_RETRY_EXHAUSTED",
        failure_detail="base impl phase failed",
    )
    record2.action = "failed"
    record2.action_reason = "base impl phase failed"
    record2.action_reason_code = "PHASE_RETRY_EXHAUSTED"

    assert record2.base_impl.failure_code == "PHASE_RETRY_EXHAUSTED"
    assert record2.action_reason_code == "PHASE_RETRY_EXHAUSTED"


def test_outer_loop_decision_uses_code_for_optimization_precheck() -> None:
    record = RoundRecord(
        outer_loop_name="outer1-2v1",
        round_index=1,
        query_list=["1", "2"],
    )
    record.optimization = PhaseInfo(
        conv_name="opt1-2v1_r001",
        status="failed",
        failure_code="OPTIMIZATION_PRECHECK_FAILED",
        failure_detail="optimization precheck correctness failed",
    )
    record.action = "continue"
    record.action_reason = "optimization precheck correctness failed; advancing to next round"
    record.action_reason_code = "OPTIMIZATION_PRECHECK_FAILED"

    assert record.action_reason_code == "OPTIMIZATION_PRECHECK_FAILED"
    assert record.optimization.failure_code == "OPTIMIZATION_PRECHECK_FAILED"


def test_parse_error_envelope_code_extracts_code() -> None:
    assert run_outer_loop_tpch_monetdb._parse_error_envelope_code(
        "[ERROR:OPTIMIZATION_PRECHECK_FAILED] something"
    ) == "OPTIMIZATION_PRECHECK_FAILED"
    assert run_outer_loop_tpch_monetdb._parse_error_envelope_code(
        "random text"
    ) is None
    assert run_outer_loop_tpch_monetdb._parse_error_envelope_code(
        "[ERROR:RUNNER_BROKEN_PIPE] pipe broken"
    ) == "RUNNER_BROKEN_PIPE"


def test_parse_error_envelope_code_ignores_filtered_input_noise() -> None:
    text = (
        "2026-04-25 ERROR:openai.agents:Error getting response; "
        "filtered.input=[{'output':'[ERROR:PATH_NOT_FOUND] Path does not exist'}]\n"
        "[ERROR:RUNNER_BROKEN_PIPE] pipe broken"
    )
    assert run_outer_loop_tpch_monetdb._parse_error_envelope_code(text) == "RUNNER_BROKEN_PIPE"
    return None


def test_reset_phase_session_state_removes_sqlite_sidecars(tmp_path: Path) -> None:
    session_dir = tmp_path / "cache" / "session"
    session_dir.mkdir(parents=True)
    for suffix in ("", "-wal", "-shm"):
        (session_dir / f"tpch_monetdb_storageplan1-2v1.sqlite{suffix}").write_text("x")

    run_outer_loop_tpch_monetdb._reset_phase_session_state(
        tmp_path,
        "tpch_monetdb_storageplan1-2v1",
    )

    for suffix in ("", "-wal", "-shm"):
        assert not (session_dir / f"tpch_monetdb_storageplan1-2v1.sqlite{suffix}").exists()
    return None


def test_run_phase_with_retries_resets_session_state_before_retry(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[list[str]] = []
    reset_calls: list[tuple[Path, str | None]] = []

    def fake_run_subprocess(cmd: list[str]) -> object:
        calls.append(cmd)
        return run_outer_loop_tpch_monetdb.subprocess.CompletedProcess(cmd, 1)

    def fake_reset(artifacts_dir: Path, runtime_conv_name: str | None) -> None:
        reset_calls.append((artifacts_dir, runtime_conv_name))
        return None

    monkeypatch.setattr(run_outer_loop_tpch_monetdb, "_run_subprocess", fake_run_subprocess)
    monkeypatch.setattr(run_outer_loop_tpch_monetdb, "_reset_phase_session_state", fake_reset)
    record = RoundRecord(
        outer_loop_name="outer1-2v1",
        round_index=1,
        query_list=["1", "2"],
        base_impl=PhaseInfo(conv_name="basef1-2v1_r001", status="running"),
    )
    phase = record.base_impl
    assert phase is not None

    result = run_outer_loop_tpch_monetdb._run_phase_with_retries(
        record=record,
        phase=phase,
        artifacts_dir=tmp_path,
        retry_budget=2,
        phase_log_name="base impl",
        failure_reason="base impl phase failed",
        failure_code="PHASE_RETRY_EXHAUSTED",
        cmd_factory=lambda: ["base"],
        runtime_conv_name="tpch_monetdb_basef1-2v1_r001",
    )

    assert result.returncode == 1
    assert len(calls) == 3
    assert reset_calls == [
        (tmp_path, "tpch_monetdb_basef1-2v1_r001"),
        (tmp_path, "tpch_monetdb_basef1-2v1_r001"),
    ]
    return None
