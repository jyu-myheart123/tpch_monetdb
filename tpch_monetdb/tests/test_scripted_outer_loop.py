"""Tests for phase10 outer-loop feedback and best snapshot recovery."""

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# RoundRecord.best_final_snapshot_hash tests (task 10.2)
# ---------------------------------------------------------------------------

def test_round_record_has_best_final_snapshot_hash() -> None:
    """RoundRecord 应有 best_final_snapshot_hash 字段，默认 None."""
    from tpch_monetdb.utils.outer_loop_state import RoundRecord
    record = RoundRecord(
        outer_loop_name="outertest",
        round_index=1,
        query_list=["q1"],
    )
    assert hasattr(record, "best_final_snapshot_hash")
    assert record.best_final_snapshot_hash is None


def test_round_record_serializes_best_final_snapshot_hash() -> None:
    """RoundRecord.to_dict 应包含 best_final_snapshot_hash."""
    from tpch_monetdb.utils.outer_loop_state import RoundRecord
    record = RoundRecord(
        outer_loop_name="outertest",
        round_index=1,
        query_list=["q1"],
        best_final_snapshot_hash="abc123",
    )
    d = record.to_dict()
    assert d.get("best_final_snapshot_hash") == "abc123"


def test_round_record_deserializes_best_final_snapshot_hash() -> None:
    """RoundRecord.from_dict 应还原 best_final_snapshot_hash."""
    from tpch_monetdb.utils.outer_loop_state import RoundRecord
    d = {
        "outer_loop_name": "outertest",
        "round_index": 1,
        "query_list": ["q1"],
        "best_final_snapshot_hash": "def456",
    }
    record = RoundRecord.from_dict(d)
    assert record.best_final_snapshot_hash == "def456"


# ---------------------------------------------------------------------------
# compute_round_decision continue_with_best tests (task 10.3)
# ---------------------------------------------------------------------------

def _make_opt_summary(final_rt: dict[str, float], success: bool = True, correct: bool = True):
    from tpch_monetdb.utils.optimization_summary import OptimizationRunSummary
    return OptimizationRunSummary(
        benchmark="tpch",
        conv_name="test",
        run_id="test",
        query_list=list(final_rt.keys()),
        is_bespoke_storage=False,
        start_snapshot_hash="a",
        final_snapshot_hash="b",
        best_runtime_ms_by_query=dict(final_rt),
        final_runtime_ms_by_query=dict(final_rt),
        final_correctness=correct,
        completed_at="",
        conversation_json="",
        session_db_path="",
        success=success,
        workload_objective={"critical_query_ids": [], "required_artifacts": []},
        storage_plan_alignment={"status": "aligned"},
        control_artifact_hashes={"workload_objective.json": "hash"},
        measurement_repetition={"aggregate_runtime_ms_samples": [1.0, 1.01]},
    )


def test_compute_round_decision_regression_with_best_snapshot_returns_continue_with_best() -> None:
    """回归且已有 best snapshot 时应返回 continue_with_best."""
    from tpch_monetdb.utils.outer_loop_state import compute_round_decision

    prev = _make_opt_summary({"q1": 10.0})   # fast prev
    curr = _make_opt_summary({"q1": 20.0})   # regressed curr

    outcome, action, _ = compute_round_decision(
        prev_summary=prev,
        curr_summary=curr,
        convergence_threshold=0.02,
        stagnant_rounds=3,
        regression_tolerance=0.05,
        max_rounds=6,
        current_round_index=2,
        stagnant_count=0,
        has_best_snapshot=True,
    )
    assert outcome == "regressed"
    assert action == "continue_with_best"


def test_compute_round_decision_regression_without_best_snapshot_returns_failed() -> None:
    """回归且无 best snapshot 时应返回 failed."""
    from tpch_monetdb.utils.outer_loop_state import compute_round_decision

    prev = _make_opt_summary({"q1": 10.0})
    curr = _make_opt_summary({"q1": 20.0})

    outcome, action, _ = compute_round_decision(
        prev_summary=prev,
        curr_summary=curr,
        convergence_threshold=0.02,
        stagnant_rounds=3,
        regression_tolerance=0.05,
        max_rounds=6,
        current_round_index=2,
        stagnant_count=0,
        has_best_snapshot=False,
    )
    assert outcome == "regressed"
    assert action == "failed"


def test_compute_round_decision_regression_at_max_rounds_returns_failed_even_with_best() -> None:
    """到达 max_rounds 且回归时，即使有 best snapshot 也应返回 failed."""
    from tpch_monetdb.utils.outer_loop_state import compute_round_decision

    prev = _make_opt_summary({"q1": 10.0})
    curr = _make_opt_summary({"q1": 20.0})

    outcome, action, _ = compute_round_decision(
        prev_summary=prev,
        curr_summary=curr,
        convergence_threshold=0.02,
        stagnant_rounds=3,
        regression_tolerance=0.05,
        max_rounds=6,
        current_round_index=6,   # at max_rounds
        stagnant_count=0,
        has_best_snapshot=True,
    )
    assert outcome == "regressed"
    assert action == "failed"


def test_compute_round_decision_objective_failure_never_converges() -> None:
    """Objective failures must route another round instead of stagnant convergence."""
    from tpch_monetdb.utils.outer_loop_state import compute_round_decision

    prev = _make_opt_summary({"8": 10.0})
    curr = _make_opt_summary({"8": 9.9})
    curr.workload_objective = {
        "critical_query_ids": ["8"],
        "critical_query_targets": {
            "8": {
                "min_speedup_vs_baseline": 2.0,
                "requires_vectorization": False,
                "requires_pmu": False,
            }
        },
        "required_artifacts": [],
        "measurement_policy": {"max_cv": 0.2, "max_spread_ratio": 0.4},
    }
    curr.baseline_runtime_ms_by_query = {"8": 10.0}
    curr.final_runtime_ms_by_query = {"8": 9.9}

    outcome, action, stagnant_count = compute_round_decision(
        prev_summary=prev,
        curr_summary=curr,
        convergence_threshold=0.02,
        stagnant_rounds=1,
        regression_tolerance=0.05,
        max_rounds=6,
        current_round_index=2,
        stagnant_count=0,
        has_best_snapshot=True,
    )

    assert outcome == "objective_failed"
    assert action == "continue"
    assert stagnant_count == 0


def test_compute_round_decision_routes_success_false_objective_failure() -> None:
    """success=false must still route machine-readable objective failures."""
    from tpch_monetdb.utils.outer_loop_state import compute_round_decision

    curr = _make_opt_summary({"8": 9.9})
    curr.success = False
    curr.final_correctness = True
    curr.objective_failures = ["PMU_CAPTURE_SCOPE_NOT_QUERY_ONLY"]

    outcome, action, stagnant_count = compute_round_decision(
        prev_summary=None,
        curr_summary=curr,
        convergence_threshold=0.02,
        stagnant_rounds=1,
        regression_tolerance=0.05,
        max_rounds=6,
        current_round_index=2,
        stagnant_count=0,
        has_best_snapshot=False,
    )

    assert outcome == "objective_failed"
    assert action == "continue"
    assert stagnant_count == 0


def test_compute_round_decision_routes_forbidden_final_path_as_objective_failure() -> None:
    """Instrumented final path failures must continue through instrumentation route."""
    from tpch_monetdb.utils.outer_loop_state import compute_round_decision

    curr = _make_opt_summary({"8": 9.9})
    curr.success = False
    curr.final_correctness = True
    curr.failure_code = "FORBIDDEN_INSTRUMENTED_FINAL_PATH"
    curr.objective_failures = ["FORBIDDEN_INSTRUMENTED_FINAL_PATH"]
    curr.objective_failure_route = "instrumentation"

    outcome, action, stagnant_count = compute_round_decision(
        prev_summary=None,
        curr_summary=curr,
        convergence_threshold=0.02,
        stagnant_rounds=1,
        regression_tolerance=0.05,
        max_rounds=6,
        current_round_index=2,
        stagnant_count=0,
        has_best_snapshot=False,
    )

    assert outcome == "objective_failed"
    assert action == "continue"
    assert stagnant_count == 0
    return None


# ---------------------------------------------------------------------------
# _should_treat_latest_record_as_terminal tests
# ---------------------------------------------------------------------------

def test_continue_with_best_not_terminal() -> None:
    """action=continue_with_best 不应被视为终止态."""
    from run_outer_loop_tpch_monetdb import _should_treat_latest_record_as_terminal
    from tpch_monetdb.utils.outer_loop_state import RoundRecord

    record = RoundRecord(
        outer_loop_name="outertest",
        round_index=2,
        query_list=["q1"],
        action="continue_with_best",
    )
    assert not _should_treat_latest_record_as_terminal(record, retry_budget=2)


def test_converged_is_terminal() -> None:
    """action=converged 应被视为终止态."""
    from run_outer_loop_tpch_monetdb import _should_treat_latest_record_as_terminal
    from tpch_monetdb.utils.outer_loop_state import RoundRecord

    record = RoundRecord(
        outer_loop_name="outertest",
        round_index=3,
        query_list=["q1"],
        action="converged",
    )
    assert _should_treat_latest_record_as_terminal(record, retry_budget=2)


def test_storage_plan_cmd_includes_start_snapshot_when_provided() -> None:
    from run_outer_loop_tpch_monetdb import _build_storage_plan_cmd

    args = SimpleNamespace(
        conv="outer1-9v1",
        benchmark="tpch",
        artifacts_dir="/tmp/artifacts",
        base_data_dir=None,
        notify=False,
        disable_repo_sync=False,
        replay_cache=False,
        auto_u=False,
        auto_finish=False,
        disable_wandb=False,
        disable_tracing=False,
        model=None,
        reasoning_effort=None,
        disable_wandb_when_tracing_disabled=False,
        wandb_init_max_attempts=3,
        wandb_init_timeout_s=30.0,
        wandb_upload_timeout_s=120.0,
        wandb_finish_timeout_s=30.0,
        wandb_finish_retries=1,
    )
    cmd = _build_storage_plan_cmd(
        args,
        round_index=2,
        prev_bottleneck_report_path=None,
        start_snapshot="best-snap",
        storage_plan_mode="repair_alignment",
    )
    assert "--start_snapshot" in cmd
    assert cmd[cmd.index("--start_snapshot") + 1] == "best-snap"
    assert "--storage_plan_mode" in cmd
    assert cmd[cmd.index("--storage_plan_mode") + 1] == "repair_alignment"


def test_restore_terminal_best_snapshot_uses_runtime_snapshotter(monkeypatch) -> None:
    from run_outer_loop_tpch_monetdb import _restore_terminal_best_snapshot
    from tpch_monetdb.utils.outer_loop_state import RoundRecord

    fake_snapshotter = MagicMock()
    fake_snapshotter.has_snapshot.return_value = True
    monkeypatch.setattr(
        "run_outer_loop_tpch_monetdb.build_runtime_snapshotter",
        lambda *args, **kwargs: fake_snapshotter,
    )
    monkeypatch.setattr(
        "run_outer_loop_tpch_monetdb._prepare_runtime_workspace",
        lambda *args, **kwargs: None,
    )
    record = RoundRecord(
        outer_loop_name="outertest",
        round_index=3,
        query_list=["q1"],
        action="converged",
        best_final_snapshot_hash="best-snap",
    )
    args = SimpleNamespace(disable_repo_sync=False, keep_csv=False)
    _restore_terminal_best_snapshot(args, record)
    fake_snapshotter.restore.assert_called_once_with("best-snap")


def test_restore_terminal_best_snapshot_requires_best_hash() -> None:
    from run_outer_loop_tpch_monetdb import _restore_terminal_best_snapshot
    from tpch_monetdb.utils.outer_loop_state import RoundRecord

    record = RoundRecord(
        outer_loop_name="outertest",
        round_index=3,
        query_list=["q1"],
        action="converged",
    )
    args = SimpleNamespace(disable_repo_sync=False, keep_csv=False)
    try:
        _restore_terminal_best_snapshot(args, record)
    except RuntimeError as exc:
        assert "best_final_snapshot_hash" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError for missing best_final_snapshot_hash")


def test_scripted_handoff_complete_rejects_missing_lineage_fields() -> None:
    from run_outer_loop_tpch_monetdb import _scripted_handoff_complete

    storage_summary = SimpleNamespace(storage_plan_sha256="planhash")
    base_impl_summary = SimpleNamespace(
        storage_plan_sha256="planhash",
        todo_sha256=None,
        implementation_manifest_sha256="manifesthash",
        control_artifact_hashes={"storage_plan.txt": "planhash"},
        todo_reconciliation={"status": "present"},
    )
    assert _scripted_handoff_complete(storage_summary, base_impl_summary) is False


def test_scripted_handoff_complete_accepts_matching_lineage_fields() -> None:
    from run_outer_loop_tpch_monetdb import _scripted_handoff_complete

    storage_summary = SimpleNamespace(storage_plan_sha256="planhash")
    base_impl_summary = SimpleNamespace(
        storage_plan_sha256="planhash",
        todo_sha256="todohash",
        implementation_manifest_sha256="manifesthash",
        control_artifact_hashes={"storage_plan.txt": "planhash"},
        todo_reconciliation={"status": "present"},
        storage_plan_alignment={"status": "aligned"},
    )
    assert _scripted_handoff_complete(storage_summary, base_impl_summary) is True


def test_scripted_handoff_complete_allows_advisory_storage_alignment() -> None:
    from run_outer_loop_tpch_monetdb import _scripted_handoff_complete

    storage_summary = SimpleNamespace(storage_plan_sha256="planhash")
    base_impl_summary = SimpleNamespace(
        storage_plan_sha256="planhash",
        todo_sha256="todohash",
        implementation_manifest_sha256="manifesthash",
        control_artifact_hashes={"storage_plan.txt": "planhash"},
        todo_reconciliation={"status": "present"},
        storage_plan_alignment={"status": "invalid"},
    )
    assert _scripted_handoff_complete(storage_summary, base_impl_summary) is True


def test_main_next_round_after_continue_with_best_passes_start_snapshot(monkeypatch, tmp_path: Path) -> None:
    import run_outer_loop_tpch_monetdb
    from tpch_monetdb.utils.outer_loop_state import PhaseInfo, RoundRecord

    latest_record = RoundRecord(
        outer_loop_name="outer1-9v1",
        round_index=1,
        query_list=["1"],
        storage_plan=PhaseInfo(conv_name="sp", status="success"),
        base_impl=PhaseInfo(conv_name="bi", status="success"),
        optimization=PhaseInfo(conv_name="opt", status="success"),
        action="continue_with_best",
        best_round_index=1,
        best_optimization_summary_path="/tmp/best.json",
        best_aggregate_runtime_ms=1.0,
        best_final_snapshot_hash="best-snap",
    )
    captured: dict[str, object] = {}

    monkeypatch.setattr(run_outer_loop_tpch_monetdb, "load_latest_round_record", lambda *args, **kwargs: latest_record)
    monkeypatch.setattr(run_outer_loop_tpch_monetdb, "determine_resume_phase", lambda record, retry_budget: ("next_round", record))
    monkeypatch.setattr(run_outer_loop_tpch_monetdb, "find_latest_successful_optimization_run", lambda *args, **kwargs: None)
    monkeypatch.setattr(run_outer_loop_tpch_monetdb, "write_round_record", lambda *args, **kwargs: tmp_path / "round.json")

    def fake_build_storage_plan_cmd(
        args,
        round_index,
        prev_bottleneck_report_path=None,
        start_snapshot=None,
        storage_plan_mode="initial_candidates",
    ):
        captured["round_index"] = round_index
        captured["start_snapshot"] = start_snapshot
        captured["storage_plan_mode"] = storage_plan_mode
        return ["python3", "-m", "tpch_monetdb.run_gen_storage_plan_tpch_monetdb"]

    monkeypatch.setattr(run_outer_loop_tpch_monetdb, "_build_storage_plan_cmd", fake_build_storage_plan_cmd)

    def fake_run_phase_with_retries(**kwargs):
        kwargs["cmd_factory"]()
        raise RuntimeError("stop_after_cmd")

    monkeypatch.setattr(run_outer_loop_tpch_monetdb, "_run_phase_with_retries", fake_run_phase_with_retries)

    args = SimpleNamespace(
        benchmark="tpch",
        conv="outer1-9v1",
        artifacts_dir=str(tmp_path),
        max_rounds=6,
        convergence_threshold=0.02,
        stagnant_rounds=3,
        retry_budget=2,
        bespoke_storage=False,
        validation_mode="strict",
        regression_tolerance=0.05,
        base_data_dir=None,
        notify=False,
        disable_repo_sync=False,
        replay_cache=False,
        auto_u=False,
        auto_finish=False,
        disable_wandb=False,
        disable_tracing=False,
        model=None,
        reasoning_effort=None,
        only_from_llm_cache=False,
        only_from_cache=False,
        enable_auto_compact=False,
        baseline_backend="monetdb",
        benchmark_mode="system-parity",
        storage_mode="persistent",
        baseline_query_file_dir=None,
        disable_wandb_when_tracing_disabled=False,
        wandb_init_max_attempts=3,
        wandb_init_timeout_s=30.0,
        wandb_upload_timeout_s=120.0,
        wandb_finish_timeout_s=30.0,
        wandb_finish_retries=1,
    )

    try:
        run_outer_loop_tpch_monetdb.main(args)
    except RuntimeError as exc:
        assert str(exc) == "stop_after_cmd"
    else:
        raise AssertionError("Expected stop_after_cmd to interrupt main")

    assert captured["round_index"] == 2
    assert captured["start_snapshot"] == "best-snap"


def test_resume_contract_rejects_incomplete_continue_run_snapshot(tmp_path: Path) -> None:
    from tpch_monetdb.main_tpch_monetdb import _validate_resume_contract_fields

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "storage_plan.txt").write_text("layout\n", encoding="utf-8")
    (workspace / "TODO.md").write_text("- [ ] build\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="RESUME_SNAPSHOT_INCOMPLETE"):
        _validate_resume_contract_fields(
            workspace_path=workspace,
            conv_mode="scripted",
            continue_run=True,
            start_snapshot=None,
        )


def test_resume_contract_rejects_incomplete_start_snapshot(tmp_path: Path) -> None:
    from tpch_monetdb.main_tpch_monetdb import _validate_resume_contract_fields

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "storage_plan.txt").write_text("layout\n", encoding="utf-8")
    (workspace / "TODO.md").write_text("- [ ] build\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="RESUME_SNAPSHOT_INCOMPLETE"):
        _validate_resume_contract_fields(
            workspace_path=workspace,
            conv_mode="optimization",
            continue_run=False,
            start_snapshot="snap-1",
        )
