import asyncio
import contextlib
import json
import logging
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from agents.editor import ApplyPatchOperation
from agents.models.reasoning_content_replay import (
    default_should_replay_reasoning_content,
)
from agents.run_context import RunContextWrapper
from agents.tool import FunctionTool

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import run_optim_loop_tpch_monetdb
import tpch_monetdb.main_tpch_monetdb
import tpch_monetdb.run_gen_base_impl_tpch_monetdb
import tpch_monetdb.run_gen_storage_plan_tpch_monetdb
from tpch_monetdb.benchmark.runtime_accounting import (
    KERNEL_RUNTIME_METRIC_KIND,
    QUERY_RUNTIME_METRIC_KIND,
)
from tpch_monetdb.conversations.conversation import COMPACTION_MARKER
from tpch_monetdb.config import resolve_scripted_readiness_scale_factors
from tpch_monetdb.conversations.compact_prompts import format_compact_summary
from tpch_monetdb.conversations.optimization_conversation_tpch_monetdb import (
    GLOBAL_HUMAN_REFERENCE_REQUIRED_CONTROL_ARTIFACTS,
    GlobalOptimizationCandidate,
    GlobalOptimizationHypothesis,
    GlobalHumanReferenceResult,
    StageConfig,
    TpchMonetdbOptimizationConversation,
    parse_global_optimization_hypotheses,
    select_global_winner,
)
from tpch_monetdb.conversations.optimization_instrumentation import TraceEvidenceSummary
from tpch_monetdb.conversations.optimization_validation import (
    required_validation_scale_factors,
    run_required_correctness_checks,
)
from tpch_monetdb.conversations.scripted_conversation import ScriptedConversation
from tpch_monetdb.conversations.tpch_monetdb_prompts_gen import (
    tpch_monetdb_optim_prompt_constraints,
    tpch_monetdb_optim_prompt_trace_expert,
    tpch_monetdb_optim_prompt_global_human_reference,
)
from tpch_monetdb.llm_cache.auto_compact import AutoCompactManager
from tpch_monetdb.llm_cache.cached_litellm import CacheType, CachedLitellmModel
from tpch_monetdb.llm_cache.cached_litellm_compaction import CachedLitellmCompactionSession
from tpch_monetdb.llm_cache.git_snapshotter import GitSnapshotter
from tpch_monetdb.llm_cache.models import get_context_window, get_model_pricing, request_cost_usd
from tpch_monetdb.oracle.validate_cache import (
    CacheMissError,
    TpchValidateCache,
    get_cached_validation_result,
)
from tpch_monetdb.tools.cpu_info import CpuInfoTool, make_cpu_info_tool
from tpch_monetdb.tools.litellm_shell import make_litellm_shell_tool
from tpch_monetdb.tools.stage_tool_policy import (
    BUILD_OPTIMIZATION_FILES,
    CORE_IMPLEMENTATION_FILES,
    FOUNDATION_CORRECTNESS_EDIT_GLOBS,
    OPTIMIZATION_INFRA_LAYOUT_EDIT_GLOBS,
    OPTIMIZATION_QUERY_LOCAL_EDIT_GLOBS,
    QUERY_FOCUSED_EDIT_GLOBS,
    QUERY_EDIT_FILES,
    StageRunSummary,
    TodoState,
    build_tool_profiles,
)
from tpch_monetdb.tools.tool_factory import build_tools
from tpch_monetdb.tools.workspace_editor import WorkspaceEditor
from tpch_monetdb.utils.pipeline_evidence import (
    MeasurementKind,
    MeasurementShapeStatus,
    QueryMeasurementRecord,
)
from tpch_monetdb.utils.cli_config import DEFAULT_MODEL
from tpch_monetdb.utils.model_aliases import is_deepseek_model
from tpch_monetdb.utils.model_setup import setup_model_config
from tpch_monetdb.utils.scripted_summary import (
    ScriptedRunSummary,
    auto_discover_start_snapshot,
    persist_successful_scripted_run,
    write_scripted_run_summary,
)
from tpch_monetdb.utils.pipeline_contracts import PipelineContractError
from tpch_monetdb.utils.pipeline_invariants import require_resume_snapshot_fields
from tpch_monetdb.utils.wandb_stats_logging import WandbRunHook


def test_run_optim_loop_tpch_monetdb_main_preserves_model_and_compaction_config(tmp_path) -> None:
    captured = {}

    def fake_run_conv_wrapper(config) -> None:
        captured["config"] = config
        return None

    run_optim_loop_tpch_monetdb.run_conv_wrapper = fake_run_conv_wrapper

    args = SimpleNamespace(
        bespoke_storage=False,
        conv="runoptim1-9v1",
        notify=False,
        disable_repo_sync=False,
        replay_cache=False,
        start_snapshot="abc123",
        disable_tracing=True,
        disable_wandb=True,
        auto_u=False,
        auto_finish=False,
        only_from_llm_cache=False,
        only_from_cache=False,
        model="litellm/test-model",
        reasoning_effort="xhigh",
        enable_auto_compact=True,
        compaction_model_map={"test-model": "compact-model"},
        artifacts_dir=str(tmp_path / "artifacts"),
    )

    run_optim_loop_tpch_monetdb.main(args)

    assert captured["config"].is_bespoke_storage is True
    assert captured["config"].conv_name == "tpch_runoptim1-9v1"
    assert captured["config"].model == "litellm/test-model"
    assert captured["config"].reasoning_effort == "xhigh"
    assert captured["config"].enable_auto_compact is True
    assert captured["config"].compaction_model_map == {
        "test-model": "compact-model"
    }


def test_run_optim_loop_tpch_monetdb_main_preserves_measurement_runtime_config(tmp_path) -> None:
    captured = {}

    def fake_run_conv_wrapper(config) -> None:
        captured["config"] = config
        return None

    run_optim_loop_tpch_monetdb.run_conv_wrapper = fake_run_conv_wrapper

    args = SimpleNamespace(
        bespoke_storage=False,
        conv="runoptim1-9v1",
        notify=False,
        disable_repo_sync=False,
        replay_cache=False,
        start_snapshot="abc123",
        disable_tracing=True,
        disable_wandb=True,
        auto_u=False,
        auto_finish=False,
        only_from_llm_cache=False,
        only_from_cache=False,
        model="litellm/test-model",
        reasoning_effort="xhigh",
        enable_auto_compact=False,
        compaction_model_map={},
        artifacts_dir=str(tmp_path / "artifacts"),
        baseline_backend=None,
        baseline_query_file_dir=None,
        benchmark_mode="system-parity",
        storage_mode="persistent",
        base_data_dir=None,
        disable_wandb_when_tracing_disabled=False,
        wandb_init_max_attempts=3,
        wandb_init_timeout_s=30.0,
        wandb_upload_timeout_s=120.0,
        wandb_finish_timeout_s=30.0,
        wandb_finish_retries=1,
        target_cpu="icelake",
        hardware_counter_backend="linux_perf_native",
        hardware_counter_runner_cmd=None,
        host_kernel="5.14.0",
        perf_event_paranoid="1",
        large_sf=1000,
    )

    run_optim_loop_tpch_monetdb.main(args)

    assert captured["config"].target_cpu == "icelake"
    assert captured["config"].hardware_counter_backend == "linux_perf_native"
    assert captured["config"].large_sf == 1000


def test_resolve_scripted_readiness_scale_factors() -> None:
    assert resolve_scripted_readiness_scale_factors(
        "strict",
        [1, 10, 100, 1000],
        1000,
    ) == [
        1,
        10,
        100,
        1000,
    ]
    assert resolve_scripted_readiness_scale_factors(
        "traversal",
        [1, 10, 100, 1000],
        1000,
    ) == [1]


def test_tpch_benchmark_scale_factor_defaults_and_supported_benchmark_sfs() -> None:
    from tpch_monetdb.config import (
        TPCH_BENCHMARK_SF_LIST,
        TPCH_VERIFY_SF_LIST,
        get_default_verify_scale_factors,
        get_tpch_benchmark_scale_factors,
        get_tpch_verify_scale_factors,
        get_tpch_benchmark_scale_factor,
        resolve_active_verify_scale_factors,
        resolve_workflow_scale_factors,
    )

    assert TPCH_BENCHMARK_SF_LIST == [1]
    assert TPCH_VERIFY_SF_LIST == [1]
    assert get_tpch_benchmark_scale_factors() == [1]
    assert get_tpch_verify_scale_factors() == [1]
    assert get_tpch_benchmark_scale_factor() == 1
    assert resolve_active_verify_scale_factors(100, [1, 10, 100, 1000]) == [1, 10, 100]
    assert resolve_workflow_scale_factors(100) == [1, 100]
    assert resolve_workflow_scale_factors(1) == [1]
    assert get_default_verify_scale_factors("tpch", "benchmark") == ([1], 1)
    assert get_default_verify_scale_factors("tpch", "verify") == ([1], 1)


def test_main_tpch_monetdb_uses_active_verify_scale_factors_for_validator_and_optimization() -> None:
    source = Path(tpch_monetdb.main_tpch_monetdb.__file__).read_text(encoding="utf-8")
    assert "active_verify_sf_list = resolve_active_verify_scale_factors(" in source
    assert "validator_sf_list = resolve_workflow_scale_factors(" in source
    assert "sf_list=validator_sf_list" in source
    assert "verify_sf_list=active_verify_sf_list" in source


def test_validate_cache_requires_code_identity_for_cache_only(tmp_path) -> None:
    cache = TpchValidateCache(tmp_path)

    with pytest.raises(CacheMissError):
        get_cached_validation_result(
            cache=cache,
            compile_key_hash="",
            query_ids=["1"],
            scale_factor=1,
            params_list=[{"hostnames": ["host_0"]}],
            validator_config={"validation_mode": "strict"},
            only_from_cache=True,
        )


def test_validate_cache_hit_preserves_aggregate_metrics(tmp_path) -> None:
    cache = TpchValidateCache(tmp_path)
    params = {"hostnames": ["host_0"]}
    validator_config = {"validation_mode": "strict"}
    cache.put(
        compile_key_hash="code-a",
        query_id="1",
        scale_factor=1,
        params=params,
        success=True,
        msg="Q1 ok",
        metrics={
            "validation/query_001/no_csv_runtime_ms": 5.0,
            "validation/query_ids_executed": ["1"],
        },
        oracle_result_hash="oracle-a",
        validator_config=validator_config,
    )

    result = get_cached_validation_result(
        cache=cache,
        compile_key_hash="code-a",
        query_ids=["1"],
        scale_factor=1.0,
        params_list=[params],
        validator_config=validator_config,
        only_from_cache=False,
    )

    assert result is not None
    msg, success, metrics, used_cache = result
    assert msg.startswith("All queries passed validation! (from cache)")
    assert success is True
    assert used_cache is True
    assert metrics["validation/query_ids_executed"] == ["1"]
    assert metrics["validation/query_001/no_csv_runtime_ms"] == 5.0


def test_validate_cache_skip_cache_forces_miss(tmp_path) -> None:
    cache = TpchValidateCache(tmp_path)
    params = {"hostnames": ["host_0"]}
    validator_config = {"validation_mode": "strict"}
    cache.put(
        compile_key_hash="code-a",
        query_id="1",
        scale_factor=1,
        params=params,
        success=True,
        msg="Q1 ok",
        metrics={"validation/query_ids_executed": ["1"]},
        oracle_result_hash="oracle-a",
        validator_config=validator_config,
    )

    result = get_cached_validation_result(
        cache=cache,
        compile_key_hash="code-a",
        query_ids=["1"],
        scale_factor=1.0,
        params_list=[params],
        validator_config=validator_config,
        only_from_cache=False,
        skip_cache=True,
    )

    assert result is None
    return None


def test_validate_cache_separates_scale_factor_and_code_identity(tmp_path) -> None:
    cache = TpchValidateCache(tmp_path)
    params = {"hostnames": ["host_0"]}
    validator_config = {"validation_mode": "strict"}
    cache.put(
        compile_key_hash="code-a",
        query_id="1",
        scale_factor=1,
        params=params,
        success=True,
        msg="Q1 ok",
        metrics={},
        oracle_result_hash="oracle-a",
        validator_config=validator_config,
    )

    assert (
        get_cached_validation_result(
            cache=cache,
            compile_key_hash="code-a",
            query_ids=["1"],
            scale_factor=10,
            params_list=[params],
            validator_config=validator_config,
            only_from_cache=False,
        )
        is None
    )
    assert (
        get_cached_validation_result(
            cache=cache,
            compile_key_hash="code-b",
            query_ids=["1"],
            scale_factor=1,
            params_list=[params],
            validator_config=validator_config,
            only_from_cache=False,
        )
        is None
    )


def test_scripted_summary_discovery_uses_latest_compatible_snapshot(tmp_path) -> None:
    older = ScriptedRunSummary(
        benchmark="tpch",
        conv_name="tpch_monetdb_basef1v1",
        run_id="old",
        query_list=["1"],
        is_bespoke_storage=False,
        final_snapshot_hash="old-hash",
        completed_at="2026-04-10T00:00:00Z",
        conversation_json="old.json",
        session_db_path="old.sqlite",
        success=True,
        validation_mode="strict",
    )
    newer = ScriptedRunSummary(
        benchmark="tpch",
        conv_name="tpch_monetdb_basef1v2",
        run_id="new",
        query_list=["1"],
        is_bespoke_storage=False,
        final_snapshot_hash="new-hash",
        completed_at="2026-04-10T00:01:00Z",
        conversation_json="new.json",
        session_db_path="new.sqlite",
        success=True,
        validation_mode="strict",
    )
    incompatible = ScriptedRunSummary(
        benchmark="tpch",
        conv_name="tpch_monetdb_basef2v1",
        run_id="other",
        query_list=["2"],
        is_bespoke_storage=False,
        final_snapshot_hash="other-hash",
        completed_at="2026-04-10T00:02:00Z",
        conversation_json="other.json",
        session_db_path="other.sqlite",
        success=True,
        validation_mode="strict",
    )

    write_scripted_run_summary(older, tmp_path)
    write_scripted_run_summary(newer, tmp_path)
    write_scripted_run_summary(incompatible, tmp_path)

    assert auto_discover_start_snapshot(
        conv_name=None,
        query_list=["1"],
        artifacts_dir=tmp_path,
    ) == "new-hash"
    assert auto_discover_start_snapshot(
        conv_name=None,
        query_list=["1"],
        artifacts_dir=tmp_path,
        explicit_snapshot="explicit-hash",
    ) == "explicit-hash"


def test_scripted_summary_writer_rejects_empty_hash_and_ignores_failed_runs(
    tmp_path,
) -> None:
    with pytest.raises(ValueError, match="final_snapshot_hash cannot be empty"):
        write_scripted_run_summary(
            ScriptedRunSummary(
                benchmark="tpch",
                conv_name="tpch_monetdb_basef1v1",
                run_id="bad",
                query_list=["1"],
                is_bespoke_storage=False,
                final_snapshot_hash="",
                completed_at="2026-04-10T00:00:00Z",
                conversation_json="bad.json",
                session_db_path="bad.sqlite",
                success=True,
                validation_mode="strict",
            ),
            tmp_path,
        )

    success_summary = ScriptedRunSummary(
        benchmark="tpch",
        conv_name="tpch_monetdb_basef1v1",
        run_id="good",
        query_list=["1"],
        is_bespoke_storage=False,
        final_snapshot_hash="good-hash",
        completed_at="2026-04-10T00:00:00Z",
        conversation_json="good.json",
        session_db_path="good.sqlite",
        success=True,
        validation_mode="strict",
    )
    failed_summary = ScriptedRunSummary(
        benchmark="tpch",
        conv_name="tpch_monetdb_basef1v2",
        run_id="failed",
        query_list=["1"],
        is_bespoke_storage=False,
        final_snapshot_hash="failed-hash",
        completed_at="2026-04-10T00:01:00Z",
        conversation_json="failed.json",
        session_db_path="failed.sqlite",
        success=False,
        validation_mode="strict",
    )

    latest_file = write_scripted_run_summary(success_summary, tmp_path)
    write_scripted_run_summary(failed_summary, tmp_path)

    latest_pointer = json.loads(
        (tmp_path / "scripted_runs" / success_summary.conv_name / "latest.json").read_text()
    )

    assert latest_pointer["latest_file"] == latest_file.name
    assert auto_discover_start_snapshot(
        conv_name=None,
        query_list=["1"],
        artifacts_dir=tmp_path,
    ) == "good-hash"


def test_persist_successful_scripted_run_propagates_summary_write_failure(
    tmp_path,
    monkeypatch,
) -> None:
    def fake_write_scripted_run_summary(*_args, **_kwargs) -> Path:
        raise OSError("disk full")

    monkeypatch.setattr(
        "tpch_monetdb.utils.scripted_summary.write_scripted_run_summary",
        fake_write_scripted_run_summary,
    )
    monkeypatch.setattr(
        "tpch_monetdb.utils.scripted_summary.build_storage_plan_alignment",
        lambda _path: {"status": "aligned", "departures": []},
    )

    with pytest.raises(OSError, match="disk full"):
        persist_successful_scripted_run(
            benchmark="legacy",
            conv_name="tpch_monetdb_basef1v1",
            query_list=["1"],
            is_bespoke_storage=False,
            final_snapshot_hash="good-hash",
            conversation_json_path=tmp_path / "conv.json",
            session_db_path=tmp_path / "session.sqlite",
            artifacts_dir=tmp_path,
            validation_mode="strict",
        )


def test_persist_successful_scripted_run_rejects_empty_final_hash(tmp_path) -> None:
    with pytest.raises(ValueError, match="final_snapshot_hash cannot be empty"):
        persist_successful_scripted_run(
            benchmark="legacy",
            conv_name="tpch_monetdb_basef1v1",
            query_list=["1"],
            is_bespoke_storage=False,
            final_snapshot_hash="",
            conversation_json_path=tmp_path / "conv.json",
            session_db_path=tmp_path / "session.sqlite",
            artifacts_dir=tmp_path,
            validation_mode="strict",
            workspace_path=tmp_path,
        )


def test_persist_successful_scripted_run_records_control_artifact_lineage(
    tmp_path: Path,
    monkeypatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "storage_plan.txt").write_text("plan\n", encoding="utf-8")
    (workspace / "TODO.md").write_text("- [x] done\n", encoding="utf-8")
    (workspace / "implementation_manifest.json").write_text(
        '{"trust_mode":"host_sealed_read_only"}\n',
        encoding="utf-8",
    )
    (workspace / "design_evidence.md").write_text("evidence\n", encoding="utf-8")
    monkeypatch.setattr(
        "tpch_monetdb.utils.scripted_summary.build_storage_plan_alignment",
        lambda _path: {"status": "aligned", "departures": []},
    )

    summary_path = persist_successful_scripted_run(
        benchmark="tpch",
        conv_name="tpch_monetdb_basef1v1",
        query_list=["1"],
        is_bespoke_storage=False,
        final_snapshot_hash="hash123",
        conversation_json_path=tmp_path / "conv.json",
        session_db_path=tmp_path / "session.sqlite",
        artifacts_dir=tmp_path,
        validation_mode="strict",
        workspace_path=workspace,
        stage_summaries=[],
    )

    written = json.loads(summary_path.read_text())
    assert written["storage_plan_sha256"]
    assert written["todo_sha256"]
    assert written["implementation_manifest_sha256"]
    assert written["control_artifact_hashes"]["storage_plan.txt"] == written["storage_plan_sha256"]
    assert written["todo_reconciliation"]["completed_count"] == 1
    assert written["storage_plan_alignment"]["status"] == "aligned"


def test_persist_successful_scripted_run_records_invalid_storage_alignment(
    tmp_path: Path,
    monkeypatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "storage_plan.txt").write_text("plan\n", encoding="utf-8")
    (workspace / "TODO.md").write_text("- [x] done\n", encoding="utf-8")
    monkeypatch.setattr(
        "tpch_monetdb.utils.scripted_summary.build_storage_plan_alignment",
        lambda _path: {"status": "invalid", "departures": ["missing Q10 path"]},
    )

    summary_path = persist_successful_scripted_run(
        benchmark="tpch",
        conv_name="tpch_monetdb_basef1v1",
        query_list=["1"],
        is_bespoke_storage=False,
        final_snapshot_hash="hash123",
        conversation_json_path=tmp_path / "conv.json",
        session_db_path=tmp_path / "session.sqlite",
        artifacts_dir=tmp_path,
        validation_mode="strict",
        workspace_path=workspace,
        stage_summaries=[],
    )
    written = json.loads(summary_path.read_text())
    assert written["success"] is True
    assert written["storage_plan_alignment"]["status"] == "invalid"
    assert written["storage_plan_alignment"]["departures"] == ["missing Q10 path"]
    return None


@pytest.mark.parametrize("validation_mode", ["strict", "traversal"])
def test_scripted_entrypoint_no_longer_routes_legacy_readiness(
    tmp_path,
    monkeypatch,
    validation_mode: str,
) -> None:
    captured: dict[str, object] = {}

    def fake_parse_query_ids(*_args, **_kwargs) -> list[str]:
        return ["1", "2"]

    def fake_build_run_config(**kwargs) -> SimpleNamespace:
        config = SimpleNamespace(artifacts_dir=kwargs["artifacts_dir"])
        captured["kwargs"] = kwargs
        captured["config"] = config
        return config

    monkeypatch.setattr(tpch_monetdb.run_gen_base_impl_tpch_monetdb, "parse_query_ids", fake_parse_query_ids)
    monkeypatch.setattr(
        tpch_monetdb.run_gen_base_impl_tpch_monetdb,
        "build_run_config",
        fake_build_run_config,
    )
    monkeypatch.setattr(
        tpch_monetdb.run_gen_base_impl_tpch_monetdb,
        "create_conversation",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        tpch_monetdb.run_gen_base_impl_tpch_monetdb,
        "run_conv_wrapper",
        lambda _config: None,
    )

    args = SimpleNamespace(
        validation_mode=validation_mode,
        conv="basef1-2v1",
        benchmark="tpch",
        base_data_dir=str(tmp_path / "data"),
        artifacts_dir=str(tmp_path / "artifacts"),
        notify=False,
        disable_repo_sync=False,
        replay_cache=False,
        disable_tracing=True,
        disable_wandb=True,
        auto_u=False,
        auto_finish=False,
        replay=False,
        model="litellm/test-model",
    )

    assert not hasattr(tpch_monetdb.run_gen_base_impl_tpch_monetdb, "ensure_tables_ready")
    tpch_monetdb.run_gen_base_impl_tpch_monetdb.main(args)

    assert "data_prepare_mode" not in captured["kwargs"]
    assert captured["kwargs"]["max_scale_factor"] == 1
    assert captured["config"].generate_design_evidence is False
    assert captured["config"].validation_mode == validation_mode
    return None


def test_scripted_entrypoint_omits_questdb_readiness_for_tpch(
    tmp_path,
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_build_run_config(**kwargs) -> SimpleNamespace:
        config = SimpleNamespace(artifacts_dir=kwargs["artifacts_dir"])
        captured["kwargs"] = kwargs
        captured["config"] = config
        return config

    monkeypatch.setattr(
        tpch_monetdb.run_gen_base_impl_tpch_monetdb,
        "parse_query_ids",
        lambda *_args, **_kwargs: ["Q1", "Q2"],
    )
    monkeypatch.setattr(
        tpch_monetdb.run_gen_base_impl_tpch_monetdb,
        "build_run_config",
        fake_build_run_config,
    )
    monkeypatch.setattr(
        tpch_monetdb.run_gen_base_impl_tpch_monetdb,
        "create_conversation",
        lambda *args, **kwargs: captured.setdefault("conversation_kwargs", kwargs),
    )
    monkeypatch.setattr(
        tpch_monetdb.run_gen_base_impl_tpch_monetdb,
        "run_conv_wrapper",
        lambda _config: None,
    )

    args = SimpleNamespace(
        validation_mode="strict",
        conv="basef1-2v1",
        benchmark="tpch",
        base_data_dir=str(tmp_path / "data"),
        artifacts_dir=str(tmp_path / "artifacts"),
        notify=False,
        disable_repo_sync=False,
        replay_cache=False,
        disable_tracing=True,
        disable_wandb=True,
        auto_u=False,
        auto_finish=False,
        replay=False,
        model="litellm/test-model",
    )

    assert not hasattr(tpch_monetdb.run_gen_base_impl_tpch_monetdb, "ensure_tables_ready")
    tpch_monetdb.run_gen_base_impl_tpch_monetdb.main(args)

    assert "data_prepare_mode" not in captured["kwargs"]
    assert captured["kwargs"]["max_scale_factor"] == 1
    assert captured["conversation_kwargs"]["verify_sf_list"] == [1]
    assert captured["conversation_kwargs"]["max_scale_factor"] == 1
    assert captured["config"].generate_design_evidence is False
    return None


def test_scripted_entrypoint_disables_design_evidence_for_tpch(tmp_path, monkeypatch) -> None:
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        tpch_monetdb.run_gen_base_impl_tpch_monetdb,
        "parse_query_ids",
        lambda *_args, **_kwargs: ["1", "2"],
    )

    def fake_build_run_config(**kwargs) -> SimpleNamespace:
        config = SimpleNamespace(artifacts_dir=kwargs["artifacts_dir"])
        captured["config"] = config
        return config

    monkeypatch.setattr(
        tpch_monetdb.run_gen_base_impl_tpch_monetdb,
        "build_run_config",
        fake_build_run_config,
    )
    monkeypatch.setattr(
        tpch_monetdb.run_gen_base_impl_tpch_monetdb,
        "create_conversation",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        tpch_monetdb.run_gen_base_impl_tpch_monetdb,
        "run_conv_wrapper",
        lambda _config: None,
    )

    args = SimpleNamespace(
        validation_mode="strict",
        conv="basef1-2v1",
        benchmark="tpch",
        storage_plan_snapshot="storage-plan-hash",
        base_data_dir=str(tmp_path / "data"),
        artifacts_dir=str(tmp_path / "artifacts"),
        notify=False,
        disable_repo_sync=False,
        replay_cache=False,
        disable_tracing=True,
        disable_wandb=True,
        auto_u=False,
        auto_finish=False,
        replay=False,
        model="litellm/test-model",
    )

    assert not hasattr(tpch_monetdb.run_gen_base_impl_tpch_monetdb, "ensure_tables_ready")
    tpch_monetdb.run_gen_base_impl_tpch_monetdb.main(args)

    assert captured["config"].generate_design_evidence is False
    return None


def test_main_tpch_monetdb_generates_tpch_design_evidence_before_model_setup(tmp_path, monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeSnapshotter:
        def create_empty_snapshot(self, _name: str) -> tuple[str, str]:
            return "", "seed-hash"

        def clean_worktree(self, include_ignored: bool = True) -> None:
            return None

        def is_dirty(self) -> bool:
            return False

        def recreate_repo(self) -> None:
            return None

    monkeypatch.setattr(
        tpch_monetdb.main_tpch_monetdb,
        "resolve_runtime_workspace_path",
        lambda _tpch_monetdb_root: tmp_path / "workspace",
    )
    monkeypatch.setattr(
        tpch_monetdb.main_tpch_monetdb,
        "build_runtime_snapshotter",
        lambda *_args, **_kwargs: FakeSnapshotter(),
    )
    monkeypatch.setattr(tpch_monetdb.main_tpch_monetdb, "get_placeholders_fn", lambda *args, **kwargs: None)
    monkeypatch.setattr(tpch_monetdb.main_tpch_monetdb, "copy_template_to", lambda *_args, **_kwargs: "")
    monkeypatch.setattr(tpch_monetdb.main_tpch_monetdb, "write_query_and_args_file", lambda **_kwargs: "")

    def fake_build_tpch_design_evidence(
        *,
        workspace_path: Path,
        query_ids: list[str],
        benchmark_sf: int,
    ) -> Path:
        captured["workspace_path"] = workspace_path
        captured["query_ids"] = query_ids
        captured["benchmark_sf"] = benchmark_sf
        target = workspace_path / "design_evidence.md"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("ok\n", encoding="utf-8")
        return target

    monkeypatch.setattr(tpch_monetdb.main_tpch_monetdb, "build_tpch_design_evidence", fake_build_tpch_design_evidence)

    def fake_setup_model_config(_model: str) -> SimpleNamespace:
        raise RuntimeError("stop-after-evidence")

    monkeypatch.setattr(tpch_monetdb.main_tpch_monetdb, "setup_model_config", fake_setup_model_config)

    args = SimpleNamespace(
        benchmark="tpch",
        conv_name="tpch_storageplan1-2v1",
        query_list="Q1,Q2",
        storage_plan_snapshot=None,
        start_snapshot=None,
        continue_run=False,
        disable_repo_sync=True,
        artifacts_dir=str(tmp_path / "artifacts"),
        keep_csv=True,
        disable_artifacts_context=False,
        model="gpt-test",
        max_scale_factor=100,
        base_data_dir=str(tmp_path / "data"),
        generate_design_evidence=False,
    )

    with pytest.raises(RuntimeError, match="stop-after-evidence"):
        asyncio.run(tpch_monetdb.main_tpch_monetdb.main(args))

    assert captured["workspace_path"] == tmp_path / "workspace"
    assert captured["query_ids"] == ["Q1", "Q2"]
    assert captured["benchmark_sf"] == 100
    return None


def test_main_tpch_monetdb_storage_plan_snapshot_skips_empty_workspace_bootstrap(
    tmp_path,
    monkeypatch,
) -> None:
    captured: dict[str, object] = {
        "restore_calls": [],
        "create_empty_snapshot_calls": 0,
        "copy_template_to_calls": 0,
        "write_query_and_args_storage_plan": None,
    }

    class FakeSnapshotter:
        current_hash = "seed-hash"

        def create_empty_snapshot(self, _name: str) -> tuple[str, str]:
            captured["create_empty_snapshot_calls"] = int(
                captured["create_empty_snapshot_calls"]
            ) + 1
            return "", "seed-hash"

        def clean_worktree(self, include_ignored: bool = True) -> None:
            return None

        def is_dirty(self) -> bool:
            return False

        def recreate_repo(self) -> None:
            return None

        def has_snapshot(self, snapshot: str) -> bool:
            return snapshot == "storage-plan-hash"

        def restore(self, snapshot: str) -> None:
            captured["restore_calls"].append(snapshot)
            workspace = tmp_path / "workspace"
            workspace.mkdir(parents=True, exist_ok=True)
            (workspace / "storage_plan.txt").write_text("plan\n", encoding="utf-8")
            (workspace / "storage_plan_contract.json").write_text(
                '{"version": 1}\n',
                encoding="utf-8",
            )
            return None

    class FakeConversation:
        def __init__(self, **_kwargs) -> None:
            return None

        async def run(self) -> None:
            raise RuntimeError("stop-after-bootstrap")

    class FakeSession:
        def __init__(self, **_kwargs) -> None:
            return None

        async def add_items(self, _items) -> None:
            return None

    fake_tool_bundle = SimpleNamespace(
        runtime=SimpleNamespace(finish_stage=lambda _output: SimpleNamespace(profile_name="x")),
        tools=[],
        all_tools=[],
    )

    monkeypatch.setattr(
        tpch_monetdb.main_tpch_monetdb,
        "_prepare_runtime_workspace",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        tpch_monetdb.main_tpch_monetdb,
        "resolve_runtime_workspace_path",
        lambda _tpch_monetdb_root: tmp_path / "workspace",
    )
    monkeypatch.setattr(
        tpch_monetdb.main_tpch_monetdb,
        "build_runtime_snapshotter",
        lambda *_args, **_kwargs: FakeSnapshotter(),
    )
    monkeypatch.setattr(
        tpch_monetdb.main_tpch_monetdb,
        "copy_template_to",
        lambda *_args, **_kwargs: captured.__setitem__(
            "copy_template_to_calls",
            int(captured["copy_template_to_calls"]) + 1,
        ) or "",
    )

    def fake_write_query_and_args_file(**kwargs):
        captured["write_query_and_args_storage_plan"] = kwargs.get("storage_plan")
        return ""

    monkeypatch.setattr(
        tpch_monetdb.main_tpch_monetdb,
        "write_query_and_args_file",
        fake_write_query_and_args_file,
    )
    monkeypatch.setattr(tpch_monetdb.main_tpch_monetdb, "get_placeholders_fn", lambda *args, **kwargs: None)
    monkeypatch.setattr(tpch_monetdb.main_tpch_monetdb, "get_query_gen", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        tpch_monetdb.main_tpch_monetdb,
        "setup_model_config",
        lambda _model: SimpleNamespace(
            accounting_model_name="gpt-test",
            use_litellm=False,
            model_name="gpt-test",
            api_key="",
            base_url=None,
            openai_client=None,
        ),
    )
    monkeypatch.setattr(tpch_monetdb.main_tpch_monetdb, "build_tools", lambda **_kwargs: fake_tool_bundle)
    monkeypatch.setattr(
        tpch_monetdb.main_tpch_monetdb,
        "make_run_tool",
        lambda **_kwargs: (
            SimpleNamespace(),
            SimpleNamespace(parse_out_and_validate_output=False),
        ),
    )
    monkeypatch.setattr(tpch_monetdb.main_tpch_monetdb, "_create_compaction_session", lambda **_kwargs: SimpleNamespace())
    monkeypatch.setattr(tpch_monetdb.main_tpch_monetdb, "Agent", lambda **_kwargs: SimpleNamespace(tools=[], name="agent"))
    monkeypatch.setattr(tpch_monetdb.main_tpch_monetdb, "trace", lambda *args, **kwargs: contextlib.nullcontext())
    monkeypatch.setattr(tpch_monetdb.main_tpch_monetdb, "ScriptedConversation", FakeConversation)
    monkeypatch.setattr(tpch_monetdb.main_tpch_monetdb, "AdvancedSQLiteSession", FakeSession)

    args = SimpleNamespace(
        benchmark="tpch",
        conv_name="tpch_basef1-2v1",
        query_list="1,2",
        storage_plan_snapshot="storage-plan-hash",
        start_snapshot=None,
        continue_run=False,
        disable_repo_sync=True,
        artifacts_dir=str(tmp_path / "artifacts"),
        keep_csv=True,
        disable_artifacts_context=False,
        model="gpt-test",
        disable_wandb=True,
        disable_valtool=True,
        run_tool_offer_trace_option=False,
        only_from_cache=False,
        only_from_llm_cache=False,
        replay=False,
        replay_cache=False,
        auto_finish=True,
        auto_u=True,
        notify=False,
        conv_mode="scripted",
        is_bespoke_storage=False,
        enable_auto_compact=False,
        disable_tracing=True,
        max_scale_factor=100,
        base_data_dir=str(tmp_path / "data"),
        generate_design_evidence=False,
    )

    with pytest.raises(RuntimeError, match="stop-after-bootstrap"):
        asyncio.run(tpch_monetdb.main_tpch_monetdb.main(args))

    assert captured["restore_calls"] == ["storage-plan-hash"]
    assert captured["create_empty_snapshot_calls"] == 0
    assert captured["copy_template_to_calls"] == 0
    assert captured["write_query_and_args_storage_plan"] == "plan\n"
    assert (tmp_path / "workspace" / "storage_plan_contract.json").exists()
    return None


def test_main_tpch_monetdb_fails_fast_when_tpch_design_evidence_generation_breaks(
    tmp_path,
    monkeypatch,
) -> None:
    class FakeSnapshotter:
        def create_empty_snapshot(self, _name: str) -> tuple[str, str]:
            return "", "seed-hash"

        def clean_worktree(self, include_ignored: bool = True) -> None:
            return None

        def is_dirty(self) -> bool:
            return False

        def recreate_repo(self) -> None:
            return None

    monkeypatch.setattr(
        tpch_monetdb.main_tpch_monetdb,
        "resolve_runtime_workspace_path",
        lambda _tpch_monetdb_root: tmp_path / "workspace",
    )
    monkeypatch.setattr(
        tpch_monetdb.main_tpch_monetdb,
        "build_runtime_snapshotter",
        lambda *_args, **_kwargs: FakeSnapshotter(),
    )
    monkeypatch.setattr(tpch_monetdb.main_tpch_monetdb, "get_placeholders_fn", lambda *args, **kwargs: None)
    monkeypatch.setattr(tpch_monetdb.main_tpch_monetdb, "copy_template_to", lambda *_args, **_kwargs: "")
    monkeypatch.setattr(tpch_monetdb.main_tpch_monetdb, "write_query_and_args_file", lambda **_kwargs: "")
    monkeypatch.setattr(
        tpch_monetdb.main_tpch_monetdb,
        "build_tpch_design_evidence",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("evidence-failed")),
    )
    monkeypatch.setattr(
        tpch_monetdb.main_tpch_monetdb,
        "setup_model_config",
        lambda _model: (_ for _ in ()).throw(AssertionError("should-not-run")),
    )

    args = SimpleNamespace(
        benchmark="tpch",
        conv_name="tpch_storageplan1-2v1",
        query_list="Q1,Q2",
        storage_plan_snapshot=None,
        start_snapshot=None,
        continue_run=False,
        disable_repo_sync=True,
        artifacts_dir=str(tmp_path / "artifacts"),
        keep_csv=True,
        disable_artifacts_context=False,
        model="gpt-test",
        max_scale_factor=100,
        base_data_dir=str(tmp_path / "data"),
        generate_design_evidence=False,
    )

    with pytest.raises(RuntimeError, match="evidence-failed"):
        asyncio.run(tpch_monetdb.main_tpch_monetdb.main(args))
    return None


def test_main_propagates_scripted_handoff_failure(tmp_path, monkeypatch) -> None:
    artifacts_dir = tmp_path / "artifacts"

    class FakeSnapshotter:
        def __init__(self, *args, **kwargs) -> None:
            self.current_hash = "final-hash"

        def is_dirty(self) -> bool:
            return False

        def create_empty_snapshot(self, _name: str) -> tuple[str, str]:
            return "", "seed-hash"

        def clean_worktree(self, include_ignored: bool = True) -> None:
            return None

        def recreate_repo(self) -> None:
            return None

    class FakeConversation:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

        async def run(self) -> None:
            return None

    class FakeSession:
        def __init__(self, *args, **kwargs) -> None:
            return None

    fake_runtime = SimpleNamespace()
    fake_tool_bundle = SimpleNamespace(
        all_tools=[],
        tools_by_profile={"default_general": []},
        runtime=fake_runtime,
    )
    fake_model_config = SimpleNamespace(
        use_litellm=False,
        accounting_model_name="gpt-test",
        model_name="gpt-test",
        openai_client=None,
        api_key=None,
        base_url=None,
    )

    monkeypatch.setattr(tpch_monetdb.main_tpch_monetdb, "GitSnapshotter", FakeSnapshotter)
    monkeypatch.setattr(tpch_monetdb.main_tpch_monetdb, "get_placeholders_fn", lambda *args, **kwargs: None)
    monkeypatch.setattr(tpch_monetdb.main_tpch_monetdb, "copy_template_to", lambda *_args, **_kwargs: "")
    monkeypatch.setattr(
        tpch_monetdb.main_tpch_monetdb,
        "write_query_and_args_file",
        lambda **_kwargs: "",
    )
    monkeypatch.setattr(tpch_monetdb.main_tpch_monetdb, "setup_model_config", lambda _model: fake_model_config)
    from tpch_monetdb.llm_cache import cached_openai as _cached_openai_mod
    monkeypatch.setattr(
        _cached_openai_mod,
        "CachedOpenAIResponsesModel",
        lambda **_kwargs: SimpleNamespace(total_saved=0),
    )
    monkeypatch.setattr(
        tpch_monetdb.main_tpch_monetdb,
        "_create_compaction_session",
        lambda **_kwargs: SimpleNamespace(),
    )
    monkeypatch.setattr(tpch_monetdb.main_tpch_monetdb, "build_tools", lambda **_kwargs: fake_tool_bundle)
    monkeypatch.setattr(
        tpch_monetdb.main_tpch_monetdb,
        "make_run_tool",
        lambda **_kwargs: (
            SimpleNamespace(),
            SimpleNamespace(parse_out_and_validate_output=False),
        ),
    )
    monkeypatch.setattr(
        tpch_monetdb.main_tpch_monetdb,
        "Agent",
        lambda **_kwargs: SimpleNamespace(tools=[], name="agent"),
    )
    monkeypatch.setattr(
        tpch_monetdb.main_tpch_monetdb,
        "trace",
        lambda *args, **kwargs: contextlib.nullcontext(),
    )
    monkeypatch.setattr(tpch_monetdb.main_tpch_monetdb, "ScriptedConversation", FakeConversation)
    monkeypatch.setattr(tpch_monetdb.main_tpch_monetdb, "AdvancedSQLiteSession", FakeSession)
    monkeypatch.setattr(
        tpch_monetdb.main_tpch_monetdb,
        "_run_final_correctness_gate",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        tpch_monetdb.main_tpch_monetdb,
        "_run_base_impl_promotion_gate",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        tpch_monetdb.main_tpch_monetdb,
        "persist_successful_scripted_run",
        lambda **_kwargs: (_ for _ in ()).throw(OSError("handoff failed")),
    )

    args = SimpleNamespace(
        benchmark="tpch",
        conv_name="tpch_basef1-9v1",
        query_list="Q1",
        storage_plan_snapshot=None,
        start_snapshot=None,
        continue_run=False,
        disable_repo_sync=True,
        artifacts_dir=str(artifacts_dir),
        keep_csv=True,
        disable_artifacts_context=False,
        model="gpt-test",
        disable_wandb=True,
        disable_valtool=False,
        run_tool_offer_trace_option=False,
        only_from_cache=False,
        only_from_llm_cache=False,
        replay=False,
        replay_cache=False,
        auto_finish=True,
        auto_u=True,
        notify=False,
        conv_mode="scripted",
        is_bespoke_storage=False,
        enable_auto_compact=False,
        disable_tracing=True,
        max_scale_factor=100,
        base_data_dir=str(tmp_path / "data"),
        validation_mode="strict",
    )

    with pytest.raises(
        RuntimeError,
        match=r"\[ERROR:HANDOFF_FAILED\].*handoff failed",
    ):
        asyncio.run(tpch_monetdb.main_tpch_monetdb.main(args))


def test_main_tpch_monetdb_scripted_final_gate_requires_validator(
    monkeypatch,
    tmp_path,
) -> None:
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()

    class FakeConversation:
        def __init__(self, **_kwargs) -> None:
            return None

        async def run(self) -> None:
            return None

    class FakeSession:
        def __init__(self, **_kwargs) -> None:
            return None

        async def add_items(self, _items) -> None:
            return None

    class FakeSnapshotter:
        current_hash = "final-hash"

        def create_empty_snapshot(self, conv_name: str) -> None:
            return None

        def has_snapshot(self, snapshot: str) -> bool:
            return False

        def restore(self, snapshot: str) -> None:
            return None

    fake_tool_bundle = SimpleNamespace(
        runtime=SimpleNamespace(finish_stage=lambda _output: SimpleNamespace(profile_name="x")),
        tools=[],
        all_tools=[],
    )

    monkeypatch.setattr(tpch_monetdb.main_tpch_monetdb, "_prepare_runtime_workspace", lambda *args, **kwargs: None)
    monkeypatch.setattr(tpch_monetdb.main_tpch_monetdb, "resolve_runtime_workspace_path", lambda *args, **kwargs: tmp_path / "ws")
    monkeypatch.setattr(tpch_monetdb.main_tpch_monetdb, "build_runtime_snapshotter", lambda *args, **kwargs: FakeSnapshotter())
    monkeypatch.setattr(tpch_monetdb.main_tpch_monetdb, "_snapshot_final_workspace_state", lambda snapshotter, conv_name: None)
    monkeypatch.setattr(tpch_monetdb.main_tpch_monetdb, "copy_template_to", lambda *args, **kwargs: "")
    monkeypatch.setattr(tpch_monetdb.main_tpch_monetdb, "setup_model_config", lambda _model: SimpleNamespace(accounting_model_name="gpt-test", use_litellm=False, model_name="gpt-test", api_key="", base_url=None, openai_client=None))
    monkeypatch.setattr(tpch_monetdb.main_tpch_monetdb, "build_tools", lambda **_kwargs: fake_tool_bundle)
    monkeypatch.setattr(
        tpch_monetdb.main_tpch_monetdb,
        "make_run_tool",
        lambda **_kwargs: (
            SimpleNamespace(),
            SimpleNamespace(parse_out_and_validate_output=False),
        ),
    )
    monkeypatch.setattr(tpch_monetdb.main_tpch_monetdb, "_create_compaction_session", lambda **_kwargs: SimpleNamespace())
    monkeypatch.setattr(tpch_monetdb.main_tpch_monetdb, "Agent", lambda **_kwargs: SimpleNamespace(tools=[], name="agent"))
    monkeypatch.setattr(tpch_monetdb.main_tpch_monetdb, "trace", lambda *args, **kwargs: contextlib.nullcontext())
    monkeypatch.setattr(tpch_monetdb.main_tpch_monetdb, "ScriptedConversation", FakeConversation)
    monkeypatch.setattr(tpch_monetdb.main_tpch_monetdb, "AdvancedSQLiteSession", FakeSession)

    args = SimpleNamespace(
        benchmark="tpch",
        conv_name="tpch_basef1-9v1",
        query_list="Q1",
        storage_plan_snapshot=None,
        start_snapshot=None,
        continue_run=False,
        disable_repo_sync=True,
        artifacts_dir=str(artifacts_dir),
        keep_csv=True,
        disable_artifacts_context=False,
        model="gpt-test",
        disable_wandb=True,
        disable_valtool=True,
        run_tool_offer_trace_option=False,
        only_from_cache=False,
        only_from_llm_cache=False,
        replay=False,
        replay_cache=False,
        auto_finish=True,
        auto_u=True,
        notify=False,
        conv_mode="scripted",
        is_bespoke_storage=False,
        enable_auto_compact=False,
        disable_tracing=True,
        max_scale_factor=100,
        base_data_dir=str(tmp_path / "data"),
        validation_mode="strict",
    )

    with pytest.raises(
        RuntimeError,
        match=r"\[ERROR:FINAL_CORRECTNESS_GATE_FAILED\].*query_validator",
    ):
        asyncio.run(tpch_monetdb.main_tpch_monetdb.main(args))
    return None


def test_main_tpch_monetdb_scripted_final_gate_rejects_only_from_cache(
    monkeypatch,
    tmp_path,
) -> None:
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()

    class FakeConversation:
        def __init__(self, **_kwargs) -> None:
            return None

        async def run(self) -> None:
            return None

    class FakeSession:
        def __init__(self, **_kwargs) -> None:
            return None

        async def add_items(self, _items) -> None:
            return None

    class FakeSnapshotter:
        current_hash = "final-hash"

        def create_empty_snapshot(self, conv_name: str) -> None:
            return None

        def has_snapshot(self, snapshot: str) -> bool:
            return False

        def restore(self, snapshot: str) -> None:
            return None

    fake_tool_bundle = SimpleNamespace(
        runtime=SimpleNamespace(finish_stage=lambda _output: SimpleNamespace(profile_name="x")),
        tools=[],
        all_tools=[],
    )

    monkeypatch.setattr(tpch_monetdb.main_tpch_monetdb, "_prepare_runtime_workspace", lambda *args, **kwargs: None)
    monkeypatch.setattr(tpch_monetdb.main_tpch_monetdb, "resolve_runtime_workspace_path", lambda *args, **kwargs: tmp_path / "ws")
    monkeypatch.setattr(tpch_monetdb.main_tpch_monetdb, "build_runtime_snapshotter", lambda *args, **kwargs: FakeSnapshotter())
    monkeypatch.setattr(tpch_monetdb.main_tpch_monetdb, "_snapshot_final_workspace_state", lambda snapshotter, conv_name: None)
    monkeypatch.setattr(tpch_monetdb.main_tpch_monetdb, "copy_template_to", lambda *args, **kwargs: "")
    monkeypatch.setattr(tpch_monetdb.main_tpch_monetdb, "setup_model_config", lambda _model: SimpleNamespace(accounting_model_name="gpt-test", use_litellm=False, model_name="gpt-test", api_key="", base_url=None, openai_client=None))
    monkeypatch.setattr(tpch_monetdb.main_tpch_monetdb, "build_tools", lambda **_kwargs: fake_tool_bundle)
    monkeypatch.setattr(
        tpch_monetdb.main_tpch_monetdb,
        "make_run_tool",
        lambda **_kwargs: (
            SimpleNamespace(),
            SimpleNamespace(parse_out_and_validate_output=False),
        ),
    )
    monkeypatch.setattr(tpch_monetdb.main_tpch_monetdb, "_create_compaction_session", lambda **_kwargs: SimpleNamespace())
    monkeypatch.setattr(tpch_monetdb.main_tpch_monetdb, "Agent", lambda **_kwargs: SimpleNamespace(tools=[], name="agent"))
    monkeypatch.setattr(tpch_monetdb.main_tpch_monetdb, "trace", lambda *args, **kwargs: contextlib.nullcontext())
    monkeypatch.setattr(tpch_monetdb.main_tpch_monetdb, "ScriptedConversation", FakeConversation)
    monkeypatch.setattr(tpch_monetdb.main_tpch_monetdb, "AdvancedSQLiteSession", FakeSession)

    args = SimpleNamespace(
        benchmark="tpch",
        conv_name="tpch_basef1-9v1",
        query_list="Q1",
        storage_plan_snapshot=None,
        start_snapshot=None,
        continue_run=False,
        disable_repo_sync=True,
        artifacts_dir=str(artifacts_dir),
        keep_csv=True,
        disable_artifacts_context=False,
        model="gpt-test",
        disable_wandb=True,
        disable_valtool=False,
        run_tool_offer_trace_option=False,
        only_from_cache=True,
        only_from_llm_cache=False,
        replay=False,
        replay_cache=False,
        auto_finish=True,
        auto_u=True,
        notify=False,
        conv_mode="scripted",
        is_bespoke_storage=False,
        enable_auto_compact=False,
        disable_tracing=True,
        max_scale_factor=100,
        base_data_dir=str(tmp_path / "data"),
        validation_mode="strict",
    )

    with pytest.raises(
        RuntimeError,
        match=r"\[ERROR:FINAL_CORRECTNESS_GATE_FAILED\].*only_from_cache",
    ):
        asyncio.run(tpch_monetdb.main_tpch_monetdb.main(args))
    return None


def test_optimization_validation_runs_all_required_scale_factors() -> None:
    class FakeRunTool:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def run(self, **kwargs) -> tuple[str, dict[str, object]]:
            self.calls.append(kwargs)
            return "ok", {"validation/correct": True}

    run_tool = FakeRunTool()
    scale_factors = required_validation_scale_factors([1, 10], 100)

    summary = run_required_correctness_checks(
        run_tool,
        scale_factors,
        ["1", "2"],
        trace_mode=False,
        optimize=True,
        external_call=True,
    )

    assert summary.success is True
    assert [call["scale_factor"] for call in run_tool.calls] == [1, 10, 100]
    assert all(call["query_id"] == ["1", "2"] for call in run_tool.calls)
    assert all(call["force_fresh_validation"] is False for call in run_tool.calls)


def test_optimization_validation_stops_on_first_failure() -> None:
    class FakeRunTool:
        def __init__(self) -> None:
            self.calls: list[int] = []

        def run(self, **kwargs) -> tuple[str, dict[str, object]]:
            sf = kwargs["scale_factor"]
            self.calls.append(sf)
            return "failed", {"validation/correct": sf != 10}

    run_tool = FakeRunTool()

    summary = run_required_correctness_checks(
        run_tool,
        [1, 10, 100],
        ["1"],
    )

    assert summary.success is False
    assert summary.failed_scale_factor == 10
    assert run_tool.calls == [1, 10]
    return None


def test_optimization_validation_can_collect_all_failures_with_force_fresh() -> None:
    class FakeRunTool:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def run(self, **kwargs) -> tuple[str, dict[str, object]]:
            self.calls.append(kwargs)
            sf = int(kwargs["scale_factor"])
            return (
                f"failed-sf{sf}",
                {
                    "validation/correct": False,
                    "validation/failure_detail": f"detail-sf{sf}",
                },
            )

    run_tool = FakeRunTool()

    summary = run_required_correctness_checks(
        run_tool,
        [1, 10],
        ["1"],
        fail_fast=False,
        force_fresh_validation=True,
    )

    assert summary.success is False
    assert summary.failed_scale_factor == 1
    assert summary.failure_code == "VALIDATION_INCORRECT"
    assert "failed-sf1" in summary.message
    assert "failed-sf10" in summary.message
    assert "detail-sf1" in summary.failure_detail
    assert "detail-sf10" in summary.failure_detail
    assert all(call["force_fresh_validation"] is True for call in run_tool.calls)
    return None


def test_optimization_validation_sets_failure_code_for_incorrect() -> None:
    class FakeRunTool:
        def run(self, **kwargs) -> tuple[str, dict[str, object]]:
            return "failed", {"validation/correct": False}

    summary = run_required_correctness_checks(
        FakeRunTool(),
        [1],
        ["1"],
    )
    assert summary.success is False
    assert summary.failure_code == "VALIDATION_INCORRECT"
    assert summary.failed_scale_factor == 1


def test_optimization_validation_sets_failure_code_for_no_metrics() -> None:
    class FakeRunTool:
        def run(self, **kwargs) -> tuple[str, None]:
            return "no output", None

    summary = run_required_correctness_checks(
        FakeRunTool(),
        [1],
        ["1"],
    )
    assert summary.success is False
    assert summary.failure_code == "VALIDATION_NO_METRICS"
    assert summary.failed_scale_factor == 1


def test_optimization_validation_propagates_runner_broken_pipe_code() -> None:
    class FakeRunTool:
        def run(self, **kwargs) -> tuple[str, dict[str, object]]:
            return "broken", {
                "validation/correct": False,
                "validation/failure_code": "RUNNER_BROKEN_PIPE",
                "validation/failure_detail": "pipe broke",
            }

    summary = run_required_correctness_checks(
        FakeRunTool(),
        [1],
        ["1"],
    )
    assert summary.success is False
    assert summary.failure_code == "RUNNER_BROKEN_PIPE"
    assert "pipe broke" in summary.failure_detail


def test_active_tpch_monetdb_optimization_prompts_keep_contract_facts_without_hidden_hints() -> None:
    """Active TPC-H MonetDB optimization prompts keep contract facts without legacy helpers."""
    constraints = tpch_monetdb_optim_prompt_constraints()

    # phase10: TPC-H schema 细节迁至 storage plan 阶段，避免在每个 optimization stage 重复注入。
    assert "Refer to the storage plan stage for the full TPC-H schema" in constraints
    assert "Single-threaded C++ only." in constraints
    assert "no query-result caching" in constraints
    assert "no precomputed aggregate sidecars" in constraints
    assert "materialized answer tables" in constraints
    assert "typed columns" in constraints
    assert "hostname -> series_id" not in constraints
    assert "latest-point pointer" not in constraints
    assert "Fixed aggregation kernels" not in constraints
    assert "Bucket-id integer arithmetic" not in constraints

    trace_prompt = tpch_monetdb_optim_prompt_trace_expert(
        query_id="1",
        constraints_str=constraints,
        expert_knowledge="Expert guidance",
        trace_summary="Trace summary",
        current_rt_ms=1000.0,
        target_rt_ms=100.0,
        sf=10,
        storage_is_bespoke=True,
    )
    global_prompt = tpch_monetdb_optim_prompt_global_human_reference(
        constraints_str=constraints,
        hotspot_summary_path="optimization_hotspot_summary.md",
        sf=10,
        storage_is_bespoke=True,
    )
    assert "three phases—loader" not in trace_prompt
    assert "Zero-copy parsing" not in global_prompt
    assert "Cache-line-aligned" not in global_prompt
    assert "Pre-computed indices" not in global_prompt
    assert "trace-driven global diagnosis" in global_prompt
    assert "fixed direction queue" in global_prompt
    return None


def test_tpch_monetdb_optimization_stage_model_is_trace_expert_then_global_reference() -> None:
    """Phase11: 当前路径为 per-query trace_expert 后接全局 human reference."""
    conversation = TpchMonetdbOptimizationConversation.__new__(
        TpchMonetdbOptimizationConversation
    )
    conversation.benchmark_sf = 10
    conversation.bespoke_storage = True

    stage = conversation._build_query_stage(
        query_id="1",
        mandatory_constraints="constraints",
        trace_summary="Trace summary",
    )
    prompt = stage.get_prompt(1000.0)

    assert stage.name == "trace_expert"
    assert stage.max_turns == 420
    assert stage.get_descriptor() == "TPC-H MonetDB Trace+Expert Optim (1)"
    assert "Trace summary" in prompt
    return None


def _patch_tpch_monetdb_optimization_prompt_builders(monkeypatch) -> None:
    monkeypatch.setattr(
        "tpch_monetdb.conversations.optimization_conversation_tpch_monetdb.tpch_monetdb_optim_prompt_pretext",
        lambda **kwargs: "pretext",
    )
    monkeypatch.setattr(
        "tpch_monetdb.conversations.optimization_conversation_tpch_monetdb.tpch_monetdb_optim_prompt_pretext_optim",
        lambda **kwargs: "pretext_optim",
    )
    monkeypatch.setattr(
        "tpch_monetdb.conversations.optimization_conversation_tpch_monetdb.tpch_monetdb_optim_prompt_constraints",
        lambda **kwargs: "constraints",
    )
    monkeypatch.setattr(
        "tpch_monetdb.conversations.optimization_conversation_tpch_monetdb.tpch_monetdb_optim_prompt_pinning",
        lambda **kwargs: "pinning",
    )
    monkeypatch.setattr(
        "tpch_monetdb.conversations.optimization_conversation_tpch_monetdb.tpch_monetdb_optim_prompt_add_timings",
        lambda: "add_timings",
    )
    monkeypatch.setattr(
        "tpch_monetdb.conversations.optimization_conversation_tpch_monetdb.tpch_monetdb_optim_prompt_add_timings_per_query",
        lambda *args, **kwargs: "add_timings_per_query",
    )
    monkeypatch.setattr(
        "tpch_monetdb.conversations.optimization_conversation_tpch_monetdb.tpch_monetdb_optim_prompt_global_diagnosis",
        lambda *args, **kwargs: "global_diagnosis",
    )
    monkeypatch.setattr(
        "tpch_monetdb.conversations.optimization_conversation_tpch_monetdb.tpch_monetdb_optim_prompt_hypothesis_execution",
        lambda *args, **kwargs: "hypothesis_execution",
    )
    return None


def test_wandb_code_include_fn_filters_runtime_workspace_files() -> None:
    root = "/tmp/workspace"
    assert tpch_monetdb.main_tpch_monetdb._wandb_code_include_fn(
        f"{root}/query_impl.cpp", root
    ) is True
    assert tpch_monetdb.main_tpch_monetdb._wandb_code_include_fn(
        f"{root}/CMakeLists.txt", root
    ) is True
    assert tpch_monetdb.main_tpch_monetdb._wandb_code_include_fn(
        f"{root}/build/query_impl.o", root
    ) is False
    assert tpch_monetdb.main_tpch_monetdb._wandb_code_include_fn(
        f"{root}/db/cpu_sf1", root
    ) is False
    assert tpch_monetdb.main_tpch_monetdb._wandb_code_include_fn(
        f"{root}/results.csv", root
    ) is False
    return None


def test_upload_workspace_code_to_wandb_invokes_log_code(tmp_path) -> None:
    workspace_path = tmp_path / "output"
    workspace_path.mkdir()
    (workspace_path / "query_impl.cpp").write_text("int main() { return 0; }\n")
    captured: dict[str, object] = {}

    class FakeRun:
        def log_code(self, *, root: str, name: str, include_fn) -> None:
            captured["root"] = root
            captured["name"] = name
            captured["include_fn"] = include_fn
            return None

    tpch_monetdb.main_tpch_monetdb._upload_workspace_code_to_wandb(FakeRun(), workspace_path)

    assert captured["root"] == str(workspace_path)
    assert captured["name"] == "workspace_code"
    include_fn = captured["include_fn"]
    assert include_fn is not None
    assert include_fn(str(workspace_path / "query_impl.cpp"), str(workspace_path)) is True
    assert include_fn(str(workspace_path / "results.csv"), str(workspace_path)) is False
    return None


def test_run_conv_wrapper_initializes_wandb_even_when_tracing_disabled(
    tmp_path, monkeypatch
) -> None:
    captured: dict[str, object] = {}

    def fake_wandb_init(**kwargs) -> object:
        captured["wandb_init"] = kwargs
        return object()

    def fake_wandb_finish() -> None:
        captured["wandb_finish"] = True
        return None

    def fake_wandb_teardown(*, exit_code: int | None = None) -> None:
        captured["wandb_teardown"] = exit_code
        return None

    monkeypatch.setattr(tpch_monetdb.main_tpch_monetdb, "load_dotenv", lambda: None)
    monkeypatch.setattr(tpch_monetdb.main_tpch_monetdb, "setup_logging", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        tpch_monetdb.main_tpch_monetdb,
        "set_tracing_disabled",
        lambda disabled: captured.setdefault("tracing_disabled", disabled),
    )
    def fake_asyncio_run(coro) -> None:
        coro.close()
        captured["asyncio_run"] = True
        return None

    monkeypatch.setattr(
        tpch_monetdb.main_tpch_monetdb,
        "asyncio",
        SimpleNamespace(run=fake_asyncio_run),
    )
    monkeypatch.setattr(
        tpch_monetdb.main_tpch_monetdb,
        "_upload_workspace_code_to_wandb",
        lambda _run, _workspace_path, timeout_s=0.0: None,
    )
    monkeypatch.setattr(
        tpch_monetdb.main_tpch_monetdb,
        "wandb",
        SimpleNamespace(
            init=fake_wandb_init,
            run=object(),
            finish=fake_wandb_finish,
            teardown=fake_wandb_teardown,
        ),
    )

    args = SimpleNamespace(
        continue_run=False,
        artifacts_dir=str(tmp_path / "artifacts"),
        conv_name="tpch_monetdb_storageplan1-2v1_r001",
        disable_tracing=True,
        disable_wandb=False,
        benchmark="tpch",
        is_bespoke_storage=True,
    )

    tpch_monetdb.main_tpch_monetdb.run_conv_wrapper(args)

    assert captured["tracing_disabled"] is True
    assert captured["asyncio_run"] is True
    assert captured["wandb_finish"] is True
    assert captured["wandb_teardown"] == 0
    assert captured["wandb_init"]["name"] == "tpch_monetdb_storageplan1-2v1_r001"
    assert "tpch" in captured["wandb_init"]["tags"]
    assert "bespoke-storage" in captured["wandb_init"]["tags"]
    assert "generated_tpch" in captured["wandb_init"]["tags"]
    return None


def test_run_conv_wrapper_tears_down_wandb_on_primary_error(
    tmp_path, monkeypatch
) -> None:
    captured: dict[str, object] = {}

    monkeypatch.setattr(tpch_monetdb.main_tpch_monetdb, "load_dotenv", lambda: None)
    monkeypatch.setattr(tpch_monetdb.main_tpch_monetdb, "setup_logging", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        tpch_monetdb.main_tpch_monetdb,
        "set_tracing_disabled",
        lambda disabled: captured.setdefault("tracing_disabled", disabled),
    )

    def fake_asyncio_run(coro) -> None:
        coro.close()
        raise RuntimeError("boom")

    monkeypatch.setattr(
        tpch_monetdb.main_tpch_monetdb,
        "asyncio",
        SimpleNamespace(run=fake_asyncio_run),
    )
    monkeypatch.setattr(
        tpch_monetdb.main_tpch_monetdb,
        "_upload_workspace_code_to_wandb",
        lambda _run, _workspace_path, timeout_s=0.0: None,
    )
    monkeypatch.setattr(
        tpch_monetdb.main_tpch_monetdb,
        "finish_wandb_with_guard",
        lambda **_kwargs: captured.setdefault("wandb_finish", True),
    )
    monkeypatch.setattr(
        tpch_monetdb.main_tpch_monetdb,
        "wandb",
        SimpleNamespace(
            init=lambda **_kwargs: object(),
            run=object(),
            finish=lambda: None,
            teardown=lambda *, exit_code=None: captured.setdefault(
                "wandb_teardown", exit_code
            ),
        ),
    )

    args = SimpleNamespace(
        continue_run=False,
        artifacts_dir=str(tmp_path / "artifacts"),
        conv_name="tpch_monetdb_storageplan1-2v1_r001",
        disable_tracing=True,
        disable_wandb=False,
        benchmark="tpch",
        is_bespoke_storage=False,
    )

    with pytest.raises(RuntimeError, match="boom"):
        tpch_monetdb.main_tpch_monetdb.run_conv_wrapper(args)

    assert captured["wandb_finish"] is True
    assert captured["wandb_teardown"] == 1
    return None


def test_run_conv_wrapper_ignores_teardown_timeout_after_success(
    tmp_path, monkeypatch
) -> None:
    captured: dict[str, object] = {}

    monkeypatch.setattr(tpch_monetdb.main_tpch_monetdb, "load_dotenv", lambda: None)
    monkeypatch.setattr(tpch_monetdb.main_tpch_monetdb, "setup_logging", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        tpch_monetdb.main_tpch_monetdb,
        "set_tracing_disabled",
        lambda disabled: captured.setdefault("tracing_disabled", disabled),
    )

    def fake_asyncio_run(coro) -> None:
        coro.close()
        captured["asyncio_run"] = True
        return None

    monkeypatch.setattr(
        tpch_monetdb.main_tpch_monetdb,
        "asyncio",
        SimpleNamespace(run=fake_asyncio_run),
    )
    monkeypatch.setattr(
        tpch_monetdb.main_tpch_monetdb,
        "_upload_workspace_code_to_wandb",
        lambda _run, _workspace_path, timeout_s=0.0: None,
    )
    monkeypatch.setattr(
        tpch_monetdb.main_tpch_monetdb,
        "run_callable_with_timeout",
        lambda func, timeout_s, operation_name: (
            (_ for _ in ()).throw(TimeoutError("wandb.teardown timed out"))
            if operation_name == "wandb.teardown"
            else func()
        ),
    )
    monkeypatch.setattr(
        tpch_monetdb.main_tpch_monetdb,
        "wandb",
        SimpleNamespace(
            init=lambda **_kwargs: object(),
            run=object(),
            finish=lambda: captured.setdefault("wandb_finish", True),
            teardown=lambda *, exit_code=None: captured.setdefault(
                "wandb_teardown", exit_code
            ),
        ),
    )

    args = SimpleNamespace(
        continue_run=False,
        artifacts_dir=str(tmp_path / "artifacts"),
        conv_name="tpch_monetdb_storageplan1-2v1_r001",
        disable_tracing=True,
        disable_wandb=False,
        benchmark="tpch",
        is_bespoke_storage=False,
    )

    tpch_monetdb.main_tpch_monetdb.run_conv_wrapper(args)

    assert captured["asyncio_run"] is True
    assert captured["wandb_finish"] is True
    return None


def test_log_final_wandb_summary_commits_token_totals(monkeypatch) -> None:
    captured: list[dict[str, object]] = []

    def fake_wandb_log(metrics, step=None, commit=None) -> None:
        captured.append({"metrics": metrics, "step": step, "commit": commit})
        return None

    monkeypatch.setattr(tpch_monetdb.main_tpch_monetdb.wandb, "log", fake_wandb_log)
    hook = SimpleNamespace(
        total_stats={
            "cost_usd": 1.25,
            "input_tokens": 100,
            "cached_tokens": 40,
            "visible_output_tokens": 25,
            "billed_output_tokens": 30,
            "reasoning_tokens": 5,
        },
        pricing_missing_seen=False,
        known_cost_seen=True,
        last_turn=7,
        prompt_idx=2,
    )

    tpch_monetdb.main_tpch_monetdb._log_final_wandb_summary(hook)

    assert captured[0]["step"] == 7
    assert captured[0]["commit"] is True
    assert captured[0]["metrics"]["final/total_tokens"] == 130
    assert captured[0]["metrics"]["final/total_cached_tokens"] == 40
    assert captured[0]["metrics"]["final/total_visible_output_tokens"] == 25
    assert captured[0]["metrics"]["final/total_billed_output_tokens"] == 30
    assert captured[0]["metrics"]["final/pricing_missing"] == 0
    assert captured[0]["metrics"]["final/total_cost_usd"] == 1.25


def test_log_final_wandb_summary_keeps_known_cost_when_pricing_is_incomplete(monkeypatch) -> None:
    captured: list[dict[str, object]] = []

    def fake_wandb_log(metrics, step=None, commit=None) -> None:
        captured.append({"metrics": metrics, "step": step, "commit": commit})
        return None

    monkeypatch.setattr(tpch_monetdb.main_tpch_monetdb.wandb, "log", fake_wandb_log)
    hook = SimpleNamespace(
        total_stats={
            "cost_usd": 1.25,
            "input_tokens": 10,
            "cached_tokens": 0,
            "visible_output_tokens": 3,
            "billed_output_tokens": 4,
            "reasoning_tokens": 1,
        },
        pricing_missing_seen=True,
        known_cost_seen=True,
        last_turn=2,
        prompt_idx=0,
    )

    tpch_monetdb.main_tpch_monetdb._log_final_wandb_summary(hook)

    assert captured[0]["metrics"]["final/pricing_missing"] == 1
    assert captured[0]["metrics"]["final/total_cost_usd"] == 1.25


def test_log_final_wandb_summary_omits_cost_when_no_known_cost_exists(monkeypatch) -> None:
    captured: list[dict[str, object]] = []

    def fake_wandb_log(metrics, step=None, commit=None) -> None:
        captured.append({"metrics": metrics, "step": step, "commit": commit})
        return None

    monkeypatch.setattr(tpch_monetdb.main_tpch_monetdb.wandb, "log", fake_wandb_log)
    hook = SimpleNamespace(
        total_stats={
            "cost_usd": 0.0,
            "input_tokens": 10,
            "cached_tokens": 0,
            "visible_output_tokens": 3,
            "billed_output_tokens": 4,
            "reasoning_tokens": 1,
        },
        pricing_missing_seen=True,
        known_cost_seen=False,
        last_turn=2,
        prompt_idx=0,
    )

    tpch_monetdb.main_tpch_monetdb._log_final_wandb_summary(hook)

    assert captured[0]["metrics"]["final/pricing_missing"] == 1
    assert "final/total_cost_usd" not in captured[0]["metrics"]


def test_wandb_metrics_callback_commits_incrementing_events(
    tmp_path,
    monkeypatch,
) -> None:
    captured: list[dict[str, object]] = []

    def fake_wandb_log(metrics, step=None, commit=None) -> None:
        captured.append({"metrics": metrics, "step": step, "commit": commit})
        return None

    monkeypatch.setattr("tpch_monetdb.utils.wandb_stats_logging.wandb.log", fake_wandb_log)
    monkeypatch.setattr(
        "tpch_monetdb.utils.wandb_stats_logging.calculate_loc_breakdown",
        lambda *_args: {"total": 12, "cpp": 0, "hpp": 0, "py": 0, "other": 0},
    )
    hook = WandbRunHook(
        model="kimi-k2.5",
        git_snapshotter=SimpleNamespace(current_hash="hash-1", working_dir=tmp_path),
    )

    hook.log_metrics_callback({"type": "shell_command"}, log_and_increment=True)

    assert hook.last_turn == 1
    assert captured[0]["step"] == 0
    assert captured[0]["commit"] is True
    assert captured[0]["metrics"]["current_hash"] == "hash-1"
    assert captured[0]["metrics"]["current_loc"] == 12


def test_wandb_optimization_metrics_are_turn_events(
    tmp_path,
    monkeypatch,
) -> None:
    captured: list[dict[str, object]] = []

    def fake_wandb_log(metrics, step=None, commit=None) -> None:
        captured.append({"metrics": metrics, "step": step, "commit": commit})
        return None

    monkeypatch.setattr("tpch_monetdb.utils.wandb_stats_logging.wandb.log", fake_wandb_log)
    monkeypatch.setattr("tpch_monetdb.utils.wandb_stats_logging.calculate_loc", lambda *_args: 9)
    hook = WandbRunHook(
        model="kimi-k2.5",
        git_snapshotter=SimpleNamespace(current_hash="hash-2", working_dir=tmp_path),
    )

    hook.log_optimization_speedup_vs_baseline(
        query_id="1",
        stage_name="trace",
        no_csv_kernel_runtime_ms=2.0,
        baseline_runtime_ms=10.0,
        baseline_engine="baseline",
        baseline_label="baseline",
    )

    assert hook.last_turn == 1
    assert captured[0]["commit"] is True
    assert captured[0]["metrics"]["type"] == "optimization_speedup"
    assert captured[0]["metrics"]["optimization/no_csv_kernel_speedup_vs_baseline"] == 5.0
    assert captured[0]["metrics"]["optimization/no_csv_kernel_runtime_ms"] == 2.0
    assert captured[0]["metrics"]["optimization/baseline_runtime_ms"] == 10.0
    assert captured[0]["metrics"]["optimization/baseline_engine"] == "baseline"
    assert captured[0]["metrics"]["optimization/runtime_metric_kind"] == "kernel_ms"


def test_wandb_optimization_metrics_use_monetdb_baseline_fields(
    tmp_path,
    monkeypatch,
) -> None:
    """TPC-H W&B speedup telemetry 应使用 baseline/MonetDB 字段而不是 QuestDB alias。"""
    captured: list[dict[str, object]] = []

    def fake_wandb_log(metrics, step=None, commit=None) -> None:
        captured.append({"metrics": metrics, "step": step, "commit": commit})
        return None

    monkeypatch.setattr("tpch_monetdb.utils.wandb_stats_logging.wandb.log", fake_wandb_log)
    monkeypatch.setattr("tpch_monetdb.utils.wandb_stats_logging.calculate_loc", lambda *_args: 9)
    hook = WandbRunHook(
        model="kimi-k2.5",
        git_snapshotter=SimpleNamespace(current_hash="hash-2", working_dir=tmp_path),
    )

    hook.log_optimization_speedup_vs_baseline(
        query_id="Q1",
        stage_name="trace",
        no_csv_kernel_runtime_ms=2.0,
        baseline_runtime_ms=10.0,
        baseline_engine="monetdb",
        baseline_label="MonetDB",
    )

    metrics = captured[0]["metrics"]
    assert metrics["optimization/no_csv_kernel_speedup_vs_baseline"] == 5.0
    assert metrics["optimization/baseline_runtime_ms"] == 10.0
    assert metrics["optimization/baseline_engine"] == "monetdb"
    assert metrics["optimization/baseline_label"] == "MonetDB"
    assert "optimization/no_csv_kernel_speedup_vs_questdb" not in metrics
    assert "optimization/questdb_runtime_ms" not in metrics


def test_wandb_hotspot_summary_uses_no_csv_kernel_metric_names(
    tmp_path,
    monkeypatch,
) -> None:
    captured: list[dict[str, object]] = []

    def fake_wandb_log(metrics, step=None, commit=None) -> None:
        captured.append({"metrics": metrics, "step": step, "commit": commit})
        return None

    monkeypatch.setattr("tpch_monetdb.utils.wandb_stats_logging.wandb.log", fake_wandb_log)
    hook = WandbRunHook(
        model="kimi-k2.5",
        git_snapshotter=SimpleNamespace(current_hash="hash-2", working_dir=tmp_path),
    )

    hook.log_query_hotspot_summary(
        stage_name="final_summary",
        query_rt_ms={"1": 2.0},
        baseline_rt_ms={"1": 10.0},
    )

    metrics = captured[0]["metrics"]
    assert metrics["query/1/no_csv_kernel_runtime_ms"] == 2.0
    assert metrics["query/1/no_csv_kernel_speedup_vs_baseline"] == 5.0
    assert metrics["runtime_metric_kind"] == "kernel_ms"
    assert "query/1/rt_ms" not in metrics
    assert "query/1/speedup" not in metrics


def test_wandb_final_summary_uses_no_csv_kernel_metric_names(
    tmp_path,
    monkeypatch,
) -> None:
    captured: list[dict[str, object]] = []

    def fake_wandb_log(metrics, step=None, commit=None) -> None:
        captured.append({"metrics": metrics, "step": step, "commit": commit})
        return None

    monkeypatch.setattr("tpch_monetdb.utils.wandb_stats_logging.wandb.log", fake_wandb_log)
    monkeypatch.setattr("tpch_monetdb.utils.wandb_stats_logging.calculate_loc", lambda *_args: 9)
    hook = WandbRunHook(
        model="kimi-k2.5",
        git_snapshotter=SimpleNamespace(current_hash="hash-2", working_dir=tmp_path),
    )

    hook.log_optimization_final_summary(
        query_id="1",
        baseline_runtime_ms=10.0,
        final_no_csv_kernel_runtime_ms=2.0,
        best_no_csv_kernel_speedup_vs_baseline=6.0,
        final_correctness=True,
        final_snapshot="hash-final",
    )

    metrics = captured[0]["metrics"]
    assert metrics["optimization_final/1/baseline_runtime_ms"] == 10.0
    assert metrics["optimization_final/1/final_no_csv_kernel_runtime_ms"] == 2.0
    assert metrics["optimization_final/1/final_no_csv_kernel_speedup_vs_baseline"] == 5.0
    assert metrics["optimization_final/1/best_no_csv_kernel_speedup_vs_baseline"] == 6.0
    assert metrics["optimization_final/1/baseline_engine"] == "baseline"
    assert metrics["optimization_final/1/runtime_metric_kind"] == "kernel_ms"
    assert "optimization_final/1/final_runtime_ms" not in metrics
    assert "optimization_final/1/final_speedup_vs_baseline" not in metrics


def test_wandb_final_summary_uses_monetdb_baseline_fields(
    tmp_path,
    monkeypatch,
) -> None:
    """TPC-H final summary 应写 generic baseline 字段并避免 QuestDB final alias。"""
    captured: list[dict[str, object]] = []

    def fake_wandb_log(metrics, step=None, commit=None) -> None:
        captured.append({"metrics": metrics, "step": step, "commit": commit})
        return None

    monkeypatch.setattr("tpch_monetdb.utils.wandb_stats_logging.wandb.log", fake_wandb_log)
    monkeypatch.setattr("tpch_monetdb.utils.wandb_stats_logging.calculate_loc", lambda *_args: 9)
    hook = WandbRunHook(
        model="kimi-k2.5",
        git_snapshotter=SimpleNamespace(current_hash="hash-2", working_dir=tmp_path),
    )

    hook.log_optimization_final_summary(
        query_id="Q1",
        baseline_runtime_ms=10.0,
        final_no_csv_kernel_runtime_ms=2.0,
        best_no_csv_kernel_speedup_vs_baseline=6.0,
        final_correctness=True,
        final_snapshot="hash-final",
        baseline_engine="monetdb",
        baseline_label="MonetDB",
    )

    metrics = captured[0]["metrics"]
    assert metrics["optimization_final/Q1/baseline_runtime_ms"] == 10.0
    assert metrics["optimization_final/Q1/final_no_csv_kernel_speedup_vs_baseline"] == 5.0
    assert metrics["optimization_final/Q1/best_no_csv_kernel_speedup_vs_baseline"] == 6.0
    assert metrics["optimization_final/Q1/baseline_engine"] == "monetdb"
    assert metrics["optimization_final/Q1/baseline_label"] == "MonetDB"
    assert "optimization_final/Q1/questdb_runtime_ms" not in metrics
    assert "optimization_final/Q1/final_no_csv_kernel_speedup_vs_questdb" not in metrics
    assert "optimization_final/Q1/best_no_csv_kernel_speedup_vs_questdb" not in metrics


@pytest.mark.asyncio
async def test_wandb_llm_metrics_omit_total_cost_when_pricing_missing(
    tmp_path,
    monkeypatch,
) -> None:
    captured: list[dict[str, object]] = []

    def fake_wandb_log(metrics, step=None, commit=None) -> None:
        captured.append({"metrics": metrics, "step": step, "commit": commit})
        return None

    monkeypatch.setattr("tpch_monetdb.utils.wandb_stats_logging.wandb.log", fake_wandb_log)
    monkeypatch.setattr("tpch_monetdb.utils.wandb_stats_logging.calculate_loc", lambda *_args: 9)
    monkeypatch.setattr(
        "tpch_monetdb.utils.wandb_stats_logging.get_tokens_context_and_dollar_info",
        lambda *_args, **_kwargs: {
            "input_tokens": 11,
            "cached_tokens": 2,
            "visible_output_tokens": 5,
            "billed_output_tokens": 6,
            "reasoning_tokens": 1,
            "context_window_usage": 0.1,
            "cost": None,
            "pricing_missing": True,
            "num_llm_request": 1,
        },
    )
    hook = WandbRunHook(
        model="unknown-model",
        git_snapshotter=SimpleNamespace(current_hash="hash-3", working_dir=tmp_path),
    )

    await hook.on_llm_end(
        SimpleNamespace(usage=object()),
        SimpleNamespace(name="agent"),
        None,
    )

    metrics = captured[0]["metrics"]
    assert metrics["pricing_missing"] is True
    assert metrics["total/pricing_missing"] == 1
    assert "cost_usd" not in metrics
    assert "total/cost_usd" not in metrics


@pytest.mark.asyncio
async def test_wandb_llm_metrics_keep_known_total_cost_after_pricing_miss(
    tmp_path,
    monkeypatch,
) -> None:
    captured: list[dict[str, object]] = []

    def fake_wandb_log(metrics, step=None, commit=None) -> None:
        captured.append({"metrics": metrics, "step": step, "commit": commit})
        return None

    token_stats = iter(
        [
            {
                "input_tokens": 11,
                "cached_tokens": 2,
                "visible_output_tokens": 5,
                "billed_output_tokens": 6,
                "reasoning_tokens": 1,
                "context_window_usage": 0.1,
                "cost": 1.25,
                "pricing_missing": False,
                "num_llm_request": 1,
            },
            {
                "input_tokens": 7,
                "cached_tokens": 0,
                "visible_output_tokens": 3,
                "billed_output_tokens": 3,
                "reasoning_tokens": 0,
                "context_window_usage": 0.2,
                "cost": None,
                "pricing_missing": True,
                "num_llm_request": 1,
            },
            {
                "input_tokens": 13,
                "cached_tokens": 1,
                "visible_output_tokens": 4,
                "billed_output_tokens": 5,
                "reasoning_tokens": 1,
                "context_window_usage": 0.3,
                "cost": 1.75,
                "pricing_missing": False,
                "num_llm_request": 1,
            },
        ]
    )

    monkeypatch.setattr("tpch_monetdb.utils.wandb_stats_logging.wandb.log", fake_wandb_log)
    monkeypatch.setattr("tpch_monetdb.utils.wandb_stats_logging.calculate_loc", lambda *_args: 9)
    monkeypatch.setattr(
        "tpch_monetdb.utils.wandb_stats_logging.get_tokens_context_and_dollar_info",
        lambda *_args, **_kwargs: next(token_stats),
    )
    hook = WandbRunHook(
        model="mixed-model",
        git_snapshotter=SimpleNamespace(current_hash="hash-4", working_dir=tmp_path),
    )

    await hook.on_llm_end(
        SimpleNamespace(usage=object()),
        SimpleNamespace(name="agent"),
        None,
    )
    await hook.on_llm_end(
        SimpleNamespace(usage=object()),
        SimpleNamespace(name="agent"),
        None,
    )
    await hook.on_llm_end(
        SimpleNamespace(usage=object()),
        SimpleNamespace(name="agent"),
        None,
    )

    assert captured[0]["metrics"]["total/cost_usd"] == pytest.approx(1.25)
    assert captured[1]["metrics"]["total/cost_usd"] == pytest.approx(1.25)
    assert captured[1]["metrics"]["total/pricing_missing"] == 1
    assert captured[2]["metrics"]["total/cost_usd"] == pytest.approx(3.0)
    assert captured[2]["metrics"]["total/pricing_missing"] == 1


def test_tpch_monetdb_scripted_parser_defaults_and_storage_plan_wiring() -> None:
    parser = tpch_monetdb.run_gen_base_impl_tpch_monetdb.build_parser(add_help=False)
    args = parser.parse_args(["--conv", "basef1-2v1"])

    assert args.validation_mode == "strict"
    assert not hasattr(args, "data_prepare_mode")
    assert args.model == DEFAULT_MODEL
    assert args.reasoning_effort is None
    assert hasattr(args, "storage_plan_snapshot")
    assert args.storage_plan_snapshot is None
    assert hasattr(args, "is_bespoke_storage")
    assert args.is_bespoke_storage is True


def test_resolve_reasoning_uses_global_default_and_explicit_override() -> None:
    default_args = SimpleNamespace(reasoning_effort=None)
    config = SimpleNamespace(
        use_litellm=True,
        accounting_model_name="kimi-k2.5",
    )

    default_reasoning = tpch_monetdb.main_tpch_monetdb._resolve_reasoning(default_args, config)

    assert default_reasoning is not None
    assert tpch_monetdb.main_tpch_monetdb.DEFAULT_REASONING_EFFORT == "xhigh"
    assert default_reasoning.effort == tpch_monetdb.main_tpch_monetdb.DEFAULT_REASONING_EFFORT

    explicit_args = SimpleNamespace(reasoning_effort="minimal")
    explicit_reasoning = tpch_monetdb.main_tpch_monetdb._resolve_reasoning(explicit_args, config)

    assert explicit_reasoning is not None
    assert explicit_reasoning.effort == "minimal"


def test_resolve_reasoning_uses_deepseek_default_when_unspecified() -> None:
    """deepseek 家族在用户未指定 effort 时，默认走 DEEPSEEK_DEFAULT_REASONING_EFFORT."""
    args = SimpleNamespace(reasoning_effort=None)
    for accounting in ("deepseek-v4-flash", "deepseek-v4-pro"):
        config = SimpleNamespace(
            use_litellm=True,
            accounting_model_name=accounting,
            model_name=f"openai/{accounting}",
        )

        reasoning = tpch_monetdb.main_tpch_monetdb._resolve_reasoning(args, config)

        assert reasoning is not None
        assert tpch_monetdb.main_tpch_monetdb.DEEPSEEK_DEFAULT_REASONING_EFFORT == "xhigh"
        assert reasoning.effort == tpch_monetdb.main_tpch_monetdb.DEEPSEEK_DEFAULT_REASONING_EFFORT


def test_resolve_reasoning_respects_explicit_effort_for_deepseek() -> None:
    """deepseek 家族下，用户显式 CLI 传入仍优先于家族默认."""
    args = SimpleNamespace(reasoning_effort="medium")
    config = SimpleNamespace(
        use_litellm=True,
        accounting_model_name="deepseek-v4-pro",
        model_name="openai/deepseek-v4-pro",
    )

    reasoning = tpch_monetdb.main_tpch_monetdb._resolve_reasoning(args, config)

    assert reasoning is not None
    assert reasoning.effort == "medium"


def test_resolve_reasoning_maps_deepseek_provider_max_to_sdk_xhigh() -> None:
    """DeepSeek provider-only max 不应直接进入 Agents SDK Reasoning schema."""
    args = SimpleNamespace(reasoning_effort="max")
    config = SimpleNamespace(
        use_litellm=True,
        accounting_model_name="deepseek-v4-pro",
        model_name="deepseek/deepseek-v4-pro",
    )

    reasoning = tpch_monetdb.main_tpch_monetdb._resolve_reasoning(args, config)

    assert reasoning is not None
    assert reasoning.effort == "xhigh"
    return None


def test_resolve_reasoning_deepseek_default_is_decoupled_from_global(monkeypatch) -> None:
    """全局默认未来即使变更，deepseek 默认仍由 DEEPSEEK_DEFAULT_REASONING_EFFORT 控制."""
    monkeypatch.setattr(tpch_monetdb.main_tpch_monetdb, "DEFAULT_REASONING_EFFORT", "low")
    args = SimpleNamespace(reasoning_effort=None)
    config = SimpleNamespace(
        use_litellm=True,
        accounting_model_name="deepseek-v4-pro",
        model_name="openai/deepseek-v4-pro",
    )

    reasoning = tpch_monetdb.main_tpch_monetdb._resolve_reasoning(args, config)

    assert reasoning is not None
    assert reasoning.effort == "xhigh"


def test_build_model_settings_allows_litellm_reasoning_effort() -> None:
    args = SimpleNamespace(reasoning_effort=None)
    config = SimpleNamespace(
        use_litellm=True,
        accounting_model_name="kimi-k2.5",
        model_name="anthropic/kimi-k2.5",
    )

    settings = tpch_monetdb.main_tpch_monetdb._build_model_settings(args, config)

    assert settings.reasoning is not None
    assert tpch_monetdb.main_tpch_monetdb.DEFAULT_REASONING_EFFORT == "xhigh"
    assert settings.reasoning.effort == tpch_monetdb.main_tpch_monetdb.DEFAULT_REASONING_EFFORT
    assert settings.extra_args == {"allowed_openai_params": ["reasoning_effort"]}
    assert settings.parallel_tool_calls is True


def test_build_model_settings_allows_gpt55_tool_choice_for_litellm() -> None:
    args = SimpleNamespace(reasoning_effort=None)
    config = SimpleNamespace(
        use_litellm=True,
        accounting_model_name="gpt-5.5",
        model_name="openai/gpt-5.5",
    )

    settings = tpch_monetdb.main_tpch_monetdb._build_model_settings(args, config)

    assert settings.tool_choice == "auto"
    assert settings.extra_args == {
        "allowed_openai_params": ["reasoning_effort", "tool_choice"]
    }
    assert settings.parallel_tool_calls is True
    return None


def test_is_deepseek_model_recognizes_aliases() -> None:
    """is_deepseek_model 必须识别 deepseek-v4 家族及三种 provider 前缀别名."""
    assert is_deepseek_model("deepseek-v4-flash")
    assert is_deepseek_model("deepseek-v4-pro")
    assert is_deepseek_model("anthropic/deepseek-v4-flash")
    assert is_deepseek_model("openai/deepseek-v4-pro")
    assert is_deepseek_model("deepseek/deepseek-v4-flash")

    assert not is_deepseek_model("kimi-k2.5")
    assert not is_deepseek_model("anthropic/claude-opus-4-6")
    assert not is_deepseek_model("gpt-5.1")
    assert not is_deepseek_model("")


def test_build_model_settings_injects_deepseek_thinking_extra_body() -> None:
    """启用 reasoning effort 的 deepseek 模型应按 OpenAI 格式注入 thinking."""
    args = SimpleNamespace(reasoning_effort="high")
    config = SimpleNamespace(
        use_litellm=True,
        accounting_model_name="deepseek-v4-pro",
        model_name="deepseek/deepseek-v4-pro",
    )

    settings = tpch_monetdb.main_tpch_monetdb._build_model_settings(args, config)

    assert settings.reasoning is None
    assert settings.extra_body == {
        "thinking": {"type": "enabled"},
        "reasoning_effort": "high",
    }
    assert settings.extra_args["allowed_openai_params"] == [
        "thinking",
        "reasoning_effort",
    ]
    assert settings.extra_args["additional_drop_params"] == ["extra_body"]


def test_build_model_settings_maps_xhigh_to_max_for_deepseek() -> None:
    """reasoning_effort=xhigh 应映射为 DeepSeek OpenAI 格式 max."""
    args = SimpleNamespace(reasoning_effort="xhigh")
    config = SimpleNamespace(
        use_litellm=True,
        accounting_model_name="deepseek-v4-flash",
        model_name="deepseek/deepseek-v4-flash",
    )

    settings = tpch_monetdb.main_tpch_monetdb._build_model_settings(args, config)

    assert settings.reasoning is None
    assert settings.extra_body == {
        "thinking": {"type": "enabled"},
        "reasoning_effort": "max",
    }
    assert settings.extra_args["allowed_openai_params"] == [
        "thinking",
        "reasoning_effort",
    ]
    assert settings.extra_args["additional_drop_params"] == ["extra_body"]


def test_build_model_settings_maps_deepseek_provider_max_without_sdk_failure() -> None:
    """provider-only max 应先走 SDK-safe xhigh，再写入 DeepSeek 请求体 max."""
    args = SimpleNamespace(reasoning_effort="max")
    config = SimpleNamespace(
        use_litellm=True,
        accounting_model_name="deepseek-v4-pro",
        model_name="deepseek/deepseek-v4-pro",
    )

    settings = tpch_monetdb.main_tpch_monetdb._build_model_settings(args, config)

    assert settings.reasoning is None
    assert settings.extra_body == {
        "thinking": {"type": "enabled"},
        "reasoning_effort": "max",
    }
    return None


def test_reactive_compact_detects_provider_and_context_failures() -> None:
    assert not tpch_monetdb.main_tpch_monetdb.should_reactive_compact(
        RuntimeError("DeepseekException - Internal Server Error")
    )
    assert tpch_monetdb.main_tpch_monetdb.should_reactive_compact(
        RuntimeError("DeepseekException - request body likely exceeds server proxy limit")
    )
    assert tpch_monetdb.main_tpch_monetdb.should_reactive_compact(
        RuntimeError("[ERROR:CONTEXT_TOO_LARGE] prompt too long")
    )
    assert not tpch_monetdb.main_tpch_monetdb.should_reactive_compact(
        RuntimeError("APIConnectionError: Server disconnected")
    )
    assert not tpch_monetdb.main_tpch_monetdb.should_reactive_compact(
        RuntimeError("Connection reset by peer")
    )
    assert not tpch_monetdb.main_tpch_monetdb.should_reactive_compact(RuntimeError("validation failed"))
    return None


def test_build_stage_model_settings_uses_sdk_safe_deepseek_profile() -> None:
    """DeepSeek stage settings should keep the default xhigh -> provider max path."""
    args = SimpleNamespace(reasoning_effort=None)
    config = SimpleNamespace(
        use_litellm=True,
        accounting_model_name="deepseek-v4-pro",
        model_name="deepseek/deepseek-v4-pro",
    )

    settings = tpch_monetdb.main_tpch_monetdb._build_stage_model_settings(args, config, "storage_plan")

    assert settings.reasoning is None
    assert settings.extra_body == {
        "thinking": {"type": "enabled"},
        "reasoning_effort": "max",
    }
    return None


def test_build_stage_model_settings_preserves_explicit_deepseek_effort() -> None:
    """Explicit DeepSeek effort should not be overwritten by stage profiles."""
    args = SimpleNamespace(reasoning_effort="medium")
    config = SimpleNamespace(
        use_litellm=True,
        accounting_model_name="deepseek-v4-pro",
        model_name="deepseek/deepseek-v4-pro",
    )

    settings = tpch_monetdb.main_tpch_monetdb._build_stage_model_settings(args, config, "compile_fix")

    assert settings.reasoning is None
    assert settings.extra_body == {
        "thinking": {"type": "enabled"},
        "reasoning_effort": "high",
    }
    return None


def test_build_model_settings_maps_low_and_medium_to_high_for_deepseek() -> None:
    for effort in ("minimal", "low", "medium"):
        args = SimpleNamespace(reasoning_effort=effort)
        config = SimpleNamespace(
            use_litellm=True,
            accounting_model_name="deepseek-v4-pro",
            model_name="deepseek/deepseek-v4-pro",
        )

        settings = tpch_monetdb.main_tpch_monetdb._build_model_settings(args, config)

        assert settings.reasoning is None
        assert settings.extra_body == {
            "thinking": {"type": "enabled"},
            "reasoning_effort": "high",
        }
        assert settings.extra_args["allowed_openai_params"] == [
            "thinking",
            "reasoning_effort",
        ]
        assert settings.extra_args["additional_drop_params"] == ["extra_body"]


def test_build_model_settings_skips_thinking_when_effort_none_for_deepseek() -> None:
    """reasoning_effort=none 时不应注入 thinking，保持禁用语义."""
    args = SimpleNamespace(reasoning_effort="none")
    config = SimpleNamespace(
        use_litellm=True,
        accounting_model_name="deepseek-v4-pro",
        model_name="deepseek/deepseek-v4-pro",
    )

    settings = tpch_monetdb.main_tpch_monetdb._build_model_settings(args, config)

    assert settings.reasoning is None
    assert settings.extra_body == {"thinking": {"type": "disabled"}}
    assert settings.extra_args == {
        "allowed_openai_params": ["thinking"],
        "additional_drop_params": ["extra_body"],
    }


def test_build_model_settings_does_not_inject_for_non_deepseek() -> None:
    """deepseek 注入路径不得污染 kimi/qwen 等其他家族."""
    args = SimpleNamespace(reasoning_effort="high")
    config = SimpleNamespace(
        use_litellm=True,
        accounting_model_name="kimi-k2.5",
        model_name="anthropic/kimi-k2.5",
    )

    settings = tpch_monetdb.main_tpch_monetdb._build_model_settings(args, config)

    assert settings.extra_body is None
    assert settings.extra_args == {"allowed_openai_params": ["reasoning_effort"]}


def test_build_model_settings_injects_thinking_for_both_deepseek_variants() -> None:
    """flash 和 pro 两款 deepseek 模型都应触发 thinking 注入（含多种 provider 前缀）."""
    args = SimpleNamespace(reasoning_effort="high")
    for alias_model, accounting in (
        ("deepseek/deepseek-v4-pro", "deepseek-v4-pro"),
        ("deepseek/deepseek-v4-flash", "deepseek-v4-flash"),
        ("openai/deepseek-v4-pro", "deepseek-v4-pro"),
    ):
        config = SimpleNamespace(
            use_litellm=True,
            accounting_model_name=accounting,
            model_name=alias_model,
        )

        settings = tpch_monetdb.main_tpch_monetdb._build_model_settings(args, config)

        assert settings.reasoning is None
        assert settings.extra_body == {
            "thinking": {"type": "enabled"},
            "reasoning_effort": "high",
        }
        assert "thinking" in settings.extra_args["allowed_openai_params"]
        assert "reasoning_effort" in settings.extra_args["allowed_openai_params"]
        assert settings.extra_args["additional_drop_params"] == ["extra_body"]


def test_resolve_litellm_reasoning_replay_hook_enables_deepseek_only() -> None:
    """is_deepseek_model 匹配的模型（含多种 provider 前缀）应启用 reasoning replay hook."""
    for model_name in ("deepseek/deepseek-v4-pro", "openai/deepseek-v4-pro", "deepseek-v4-pro"):
        hook = tpch_monetdb.main_tpch_monetdb._resolve_litellm_reasoning_replay_hook(model_name)
        assert hook is default_should_replay_reasoning_content, f"hook missing for {model_name}"

    assert (
        tpch_monetdb.main_tpch_monetdb._resolve_litellm_reasoning_replay_hook(
            "anthropic/kimi-k2.5"
        )
        is None
    )


def test_activate_tool_runtime_passes_prompt_metadata_when_supported() -> None:
    captured: dict[str, object] = {}

    class Runtime:
        def activate(
            self,
            profile_name: str | None,
            prompt_index: int,
            prompt_descriptor: str | None,
            prompt_metadata: dict[str, object] | None = None,
        ) -> None:
            captured["profile_name"] = profile_name
            captured["prompt_index"] = prompt_index
            captured["prompt_descriptor"] = prompt_descriptor
            captured["prompt_metadata"] = prompt_metadata
            return None

    tpch_monetdb.main_tpch_monetdb._activate_tool_runtime(
        Runtime(),
        profile_name="compile_fix",
        prompt_index=3,
        prompt_descriptor="compile_fix",
        prompt_metadata={"rule_area": "runtime"},
    )

    assert captured == {
        "profile_name": "compile_fix",
        "prompt_index": 3,
        "prompt_descriptor": "compile_fix",
        "prompt_metadata": {"rule_area": "runtime"},
    }
    return None


def test_activate_tool_runtime_falls_back_for_legacy_runtime() -> None:
    captured: dict[str, object] = {}

    class Runtime:
        def activate(
            self,
            profile_name: str | None,
            prompt_index: int,
            prompt_descriptor: str | None,
        ) -> None:
            captured["profile_name"] = profile_name
            captured["prompt_index"] = prompt_index
            captured["prompt_descriptor"] = prompt_descriptor
            return None

    tpch_monetdb.main_tpch_monetdb._activate_tool_runtime(
        Runtime(),
        profile_name="compile_fix",
        prompt_index=5,
        prompt_descriptor="compile_fix",
        prompt_metadata={"rule_area": "runtime"},
    )

    assert captured == {
        "profile_name": "compile_fix",
        "prompt_index": 5,
        "prompt_descriptor": "compile_fix",
    }
    return None


def test_main_tpch_monetdb_passes_deepseek_replay_hook_to_litellm_model(
    tmp_path,
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeSnapshotter:
        def create_empty_snapshot(self, _name: str) -> tuple[str, str]:
            return "", "seed-hash"

        def clean_worktree(self, include_ignored: bool = True) -> None:
            return None

        def is_dirty(self) -> bool:
            return False

        def recreate_repo(self) -> None:
            return None

    class StopAfterModel(RuntimeError):
        pass

    class FakeCachedLitellmModel:
        def __init__(self, *args, **kwargs) -> None:
            del args
            captured.update(kwargs)
            raise StopAfterModel("stop-after-model")

    monkeypatch.setattr(
        tpch_monetdb.main_tpch_monetdb,
        "resolve_runtime_workspace_path",
        lambda _tpch_monetdb_root: tmp_path / "workspace",
    )
    monkeypatch.setattr(
        tpch_monetdb.main_tpch_monetdb,
        "build_runtime_snapshotter",
        lambda *_args, **_kwargs: FakeSnapshotter(),
    )
    monkeypatch.setattr(
        tpch_monetdb.main_tpch_monetdb,
        "_prepare_runtime_workspace",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(tpch_monetdb.main_tpch_monetdb, "get_query_gen", lambda _benchmark: None)
    monkeypatch.setattr(
        tpch_monetdb.main_tpch_monetdb,
        "get_placeholders_fn",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        tpch_monetdb.main_tpch_monetdb,
        "copy_template_to",
        lambda *_args, **_kwargs: "",
    )
    monkeypatch.setattr(
        tpch_monetdb.main_tpch_monetdb,
        "write_query_and_args_file",
        lambda **_kwargs: "",
    )
    monkeypatch.setattr(
        tpch_monetdb.main_tpch_monetdb,
        "setup_model_config",
        lambda _model: SimpleNamespace(
            use_litellm=True,
            model_name="openai/deepseek-v4-pro",
            accounting_model_name="deepseek-v4-pro",
            api_key="test-key",
            base_url="https://api.deepseek.com",
            openai_client=None,
        ),
    )
    monkeypatch.setattr(
        tpch_monetdb.main_tpch_monetdb,
        "make_run_tool",
        lambda **_kwargs: (SimpleNamespace(name="run_tool"), SimpleNamespace()),
    )
    monkeypatch.setattr(
        tpch_monetdb.main_tpch_monetdb,
        "_create_compaction_session",
        lambda **_kwargs: object(),
    )
    monkeypatch.setattr(
        tpch_monetdb.main_tpch_monetdb,
        "build_tools",
        lambda **_kwargs: SimpleNamespace(all_tools=[]),
    )
    monkeypatch.setattr(
        tpch_monetdb.main_tpch_monetdb,
        "get_dataset_name",
        lambda _benchmark: "tpch",
    )
    monkeypatch.setattr(
        "tpch_monetdb.llm_cache.cached_litellm.CachedLitellmModel",
        FakeCachedLitellmModel,
    )

    args = SimpleNamespace(
        benchmark="tpch",
        conv_name="tpch_deepseek_hook",
        query_list="1",
        storage_plan_snapshot=None,
        start_snapshot=None,
        continue_run=False,
        disable_repo_sync=True,
        artifacts_dir=str(tmp_path / "artifacts"),
        keep_csv=True,
        disable_artifacts_context=False,
        model="litellm/openai/deepseek-v4-pro",
        max_scale_factor=100,
        base_data_dir=str(tmp_path / "data"),
        generate_design_evidence=False,
        disable_wandb=True,
        disable_valtool=True,
        run_tool_offer_trace_option=False,
        only_from_cache=False,
        replay=False,
        compaction_model_map=None,
    )

    with pytest.raises(StopAfterModel, match="stop-after-model"):
        asyncio.run(tpch_monetdb.main_tpch_monetdb.main(args))

    assert (
        captured["should_replay_reasoning_content"]
        is default_should_replay_reasoning_content
    )


def test_tpch_monetdb_optimization_parser_marks_bespoke_storage_as_deprecated_noop() -> None:
    help_text = run_optim_loop_tpch_monetdb.build_parser().format_help()
    normalized_help = " ".join(help_text.split())

    assert "--bespoke_storage" in help_text
    assert "deprecated compatibility flag" in normalized_help.lower()
    assert "always storage-enabled" in normalized_help.lower()


def test_tpch_monetdb_tool_factory_builds_structured_tool_bundle(tmp_path) -> None:
    async def on_invoke(_ctx: RunContextWrapper[object], _args_json: str) -> str:
        return "ok"

    run_tool = FunctionTool(
        name="run",
        description="run",
        params_json_schema={"type": "object", "properties": {}},
        on_invoke_tool=on_invoke,
    )

    bundle = build_tools(
        use_litellm=True,
        workspace_path=tmp_path,
        cache_path=tmp_path / "cache",
        shell_executor=None,
        workspace_editor=None,
        compile_cache_dir=tmp_path / "compile",
        run_tool_wrapper=run_tool,
        git_snapshotter=None,
        wandb_metrics_hook=None,
    )

    tool_names = [tool.name for tool in bundle.all_tools]
    assert tool_names == [
        "read_file",
        "read_artifact",
        "list_files",
        "grep_repo",
        "edit_file",
        "write_file",
        "apply_patch",
        "shell",
        "cpu_info",
        "compile",
        "run",
    ]
    assert [tool.name for tool in bundle.tools_by_profile["finish_skeleton"]] == [
        "read_file",
        "read_artifact",
        "list_files",
        "grep_repo",
        "edit_file",
        "apply_patch",
    ]
    assert [tool.name for tool in bundle.tools_by_profile["todo_sync"]] == [
        "read_file",
        "read_artifact",
        "write_file",
    ]
    assert [tool.name for tool in bundle.tools_by_profile["implement_queries_writeonly"]] == [
        "read_file",
        "read_artifact",
        "list_files",
        "grep_repo",
        "edit_file",
        "write_file",
        "apply_patch",
    ]
    assert [tool.name for tool in bundle.tools_by_profile["correctness_queries_writeonly"]] == [
        "read_file",
        "read_artifact",
        "list_files",
        "grep_repo",
        "edit_file",
        "compile",
        "run",
        "write_file",
        "apply_patch",
    ]
    assert [tool.name for tool in bundle.tools_by_profile["correctness_foundation"]] == [
        "read_file",
        "read_artifact",
        "list_files",
        "grep_repo",
        "edit_file",
        "compile",
        "run",
        "write_file",
        "apply_patch",
    ]
    assert [tool.name for tool in bundle.tools_by_profile["optimization_instrumentation"]] == [
        "read_file",
        "read_artifact",
        "list_files",
        "grep_repo",
        "cpu_info",
        "edit_file",
        "apply_patch",
        "compile",
        "run",
    ]
    assert [tool.name for tool in bundle.tools_by_profile["benchmark"]] == [
        "read_file",
        "read_artifact",
        "list_files",
        "grep_repo",
        "edit_file",
        "compile",
        "run",
        "apply_patch",
        "cpu_info",
    ]


def test_tool_profiles_define_finish_skeleton_constraints() -> None:
    profiles = build_tool_profiles()
    legacy_general = profiles["legacy_general"]
    storage_plan = profiles["storage_plan"]
    finish_skeleton = profiles["finish_skeleton"]
    compile_fix = profiles["compile_fix"]
    implement_queries = profiles["implement_queries"]
    implement_write_only = profiles["implement_queries_writeonly"]
    correctness_write_only = profiles["correctness_queries_writeonly"]
    correctness_foundation = profiles["correctness_foundation"]
    optimize_build = profiles["optimize_build"]
    benchmark = profiles["benchmark"]
    optimization_general = profiles["optimization_general"]
    optimization_infra_layout = profiles["optimization_infra_layout"]
    optimization_instrumentation = profiles["optimization_instrumentation"]
    optimization_control = profiles["optimization_control"]

    assert legacy_general.tool_names == (
        "read_file",
        "read_artifact",
        "list_files",
        "grep_repo",
        "edit_file",
        "write_file",
        "shell",
        "compile",
        "run",
    )
    assert "cpu_info" in benchmark.tool_names
    assert "cpu_info" in optimization_general.tool_names
    assert "cpu_info" in optimization_infra_layout.tool_names
    assert "cpu_info" in optimization_instrumentation.tool_names
    assert "shell" not in optimization_instrumentation.tool_names
    assert "workload_objective.json" in optimization_control.read_globs
    assert "data_law_contract.json" in optimization_control.read_globs
    assert "storage_plan_contract.json" in optimization_control.read_globs
    assert finish_skeleton.tool_names == (
        "read_file",
        "read_artifact",
        "list_files",
        "grep_repo",
        "edit_file",
        "apply_patch",
    )
    assert storage_plan.tool_names == (
        "read_file",
        "read_artifact",
        "list_files",
        "grep_repo",
        "write_file",
    )
    assert storage_plan.write_globs == ("storage_plan.txt", "storage_plan_contract.json")
    assert storage_plan.max_consecutive_observations == 36
    assert storage_plan.hard_consecutive_observations == 96
    # phase10: finish_skeleton 的 edit scope 扩展到包含 companion query globs。
    assert finish_skeleton.edit_globs == QUERY_EDIT_FILES
    for dispatcher in CORE_IMPLEMENTATION_FILES:
        assert dispatcher in finish_skeleton.edit_globs
    assert optimize_build.edit_globs == BUILD_OPTIMIZATION_FILES
    assert optimization_general.edit_globs == OPTIMIZATION_QUERY_LOCAL_EDIT_GLOBS
    assert optimization_infra_layout.edit_globs == OPTIMIZATION_INFRA_LAYOUT_EDIT_GLOBS
    for dispatcher in CORE_IMPLEMENTATION_FILES:
        assert dispatcher not in optimization_general.edit_globs
        assert dispatcher in optimization_infra_layout.edit_globs
    assert finish_skeleton.max_consecutive_observations == 48
    assert finish_skeleton.hard_consecutive_observations == 144
    assert "compile" in compile_fix.tool_names
    assert "run" in compile_fix.tool_names
    assert implement_queries.max_consecutive_observations == 48
    assert implement_queries.hard_consecutive_observations == 192
    assert "compile" not in implement_write_only.tool_names
    assert "run" not in implement_write_only.tool_names
    assert "write_file" in implement_write_only.tool_names
    assert implement_write_only.edit_globs == QUERY_FOCUSED_EDIT_GLOBS
    assert implement_write_only.write_globs == QUERY_FOCUSED_EDIT_GLOBS
    assert implement_write_only.allow_write_overwrite is True
    assert implement_write_only.allow_write_create is True
    assert correctness_write_only.edit_globs == QUERY_FOCUSED_EDIT_GLOBS
    assert correctness_foundation.edit_globs == FOUNDATION_CORRECTNESS_EDIT_GLOBS
    assert "query_impl.hpp" not in implement_write_only.edit_globs
    assert "query_impl.cpp" not in implement_write_only.edit_globs
    assert "query_impl.hpp" not in correctness_write_only.edit_globs
    assert "query_impl.cpp" not in correctness_write_only.edit_globs
    assert "query_impl.hpp" in correctness_foundation.edit_globs
    assert "query_impl.cpp" in correctness_foundation.edit_globs
    assert "builder_impl.hpp" in correctness_foundation.edit_globs
    assert "builder_impl.cpp" in correctness_foundation.edit_globs
    assert "loader_impl.hpp" in correctness_foundation.edit_globs
    assert "loader_impl.cpp" in correctness_foundation.edit_globs
    assert "query_api.hpp" not in implement_write_only.edit_globs
    assert "query_api.hpp" not in correctness_write_only.edit_globs
    assert "query_api.hpp" not in correctness_foundation.edit_globs
    assert "query_family_*.cpp" in implement_write_only.edit_globs
    assert "query_family_*.hpp" in implement_write_only.edit_globs
    assert "query_family_*.cpp" in correctness_write_only.edit_globs
    assert "query_family_*.hpp" in correctness_write_only.edit_globs
    assert "compile" in correctness_write_only.tool_names
    assert "run" in correctness_write_only.tool_names
    assert "write_file" in correctness_write_only.tool_names
    assert correctness_write_only.write_globs == QUERY_FOCUSED_EDIT_GLOBS
    assert "compile" in correctness_foundation.tool_names
    assert "run" in correctness_foundation.tool_names
    assert "write_file" in correctness_foundation.tool_names
    assert correctness_foundation.write_globs == FOUNDATION_CORRECTNESS_EDIT_GLOBS
    assert correctness_write_only.allow_write_overwrite is True
    assert correctness_write_only.allow_write_create is False
    assert correctness_foundation.allow_write_overwrite is True
    assert correctness_foundation.allow_write_create is False
    assert implement_write_only.max_consecutive_observations == 48
    assert implement_write_only.hard_consecutive_observations == 192
    assert correctness_write_only.max_consecutive_observations == 24
    assert correctness_write_only.hard_consecutive_observations == 96
    assert correctness_foundation.max_consecutive_observations == 24
    assert correctness_foundation.hard_consecutive_observations == 96


def test_todo_state_tracks_in_progress_and_valid_transition() -> None:
    before = TodoState.from_text(
        "\n".join(
            [
                "## Build",
                "- [ ] Parse ILP",
                "- [>] Build engine",
                "- [x] Wire dispatcher",
            ]
        )
    )
    after = TodoState.from_text(
        "\n".join(
            [
                "## Build",
                "- [>] Parse ILP",
                "- [x] Build engine",
                "- [x] Wire dispatcher",
            ]
        )
    )

    assert before.pending_count == 1
    assert before.in_progress_count == 1
    assert after.completed_count == 2
    assert after.is_valid_successor(before) is True
    assert after.progressed_count_from(before) == 2


def test_todo_state_rejects_backward_transition() -> None:
    before = TodoState.from_text("- [x] Build engine\n")
    after = TodoState.from_text("- [ ] Build engine\n")

    assert after.is_valid_successor(before) is False
    assert after.progressed_count_from(before) == 0


def test_query_output_mode_normalization_accepts_supported_modes() -> None:
    """RunTool output mode 参数应只接受定义好的 CSV/诊断模式。"""
    from tpch_monetdb.tools.tpch.run import _normalize_query_output_mode

    assert _normalize_query_output_mode("full_csv") == "full_csv"
    assert _normalize_query_output_mode("no-output") == "no_output"
    assert _normalize_query_output_mode("hash_only") == "hash_only"
    with pytest.raises(ValueError, match="Invalid query output mode"):
        _normalize_query_output_mode("json")
    return None


def test_query_output_mode_reaches_persistent_runner_env(tmp_path, monkeypatch) -> None:
    """RunTool 应把 output mode 通过环境变量传给常驻 runner。"""
    from tpch_monetdb.tools.tpch import run as run_module

    captured: dict[str, object] = {}

    class FakeProc:
        def __init__(self, cmd, *, echo_output, cwd, extra_env) -> None:
            captured["cmd"] = cmd
            captured["echo_output"] = echo_output
            captured["cwd"] = cwd
            captured["extra_env"] = extra_env
            return None

    monkeypatch.setattr(run_module, "FasttestProc", FakeProc)

    tool = run_module.RunTool.__new__(run_module.RunTool)
    tool.cwd = tmp_path

    runner = tool._runner_factory(
        "./db /data/sf1",
        output_mode="hash_only",
        trace_mode=False,
    )

    assert isinstance(runner, FakeProc)
    assert captured["extra_env"] == {"TPCH_MONETDB_QUERY_OUTPUT_MODE": "hash_only"}
    return None


def test_trace_mode_runner_env_sets_bounded_trace_path(tmp_path, monkeypatch) -> None:
    """RunTool trace 模式应向子进程传入可清理的 trace 文件路径。"""
    from tpch_monetdb.tools.tpch import run as run_module

    captured: dict[str, object] = {}

    class FakeProc:
        def __init__(self, cmd, *, echo_output, cwd, extra_env) -> None:
            captured["extra_env"] = extra_env
            return None

    monkeypatch.setattr(run_module, "FasttestProc", FakeProc)

    tool = run_module.RunTool.__new__(run_module.RunTool)
    tool.cwd = tmp_path

    tool._runner_factory(
        "./db /data/sf1",
        output_mode="no_output",
        trace_mode=True,
    )

    assert captured["extra_env"] == {
        "TPCH_MONETDB_QUERY_OUTPUT_MODE": "no_output",
        "TPCH_MONETDB_TRACE_OUTPUT_PATH": str(tmp_path / "tracing_output.log"),
        "TPCH_MONETDB_TRACE_APPEND": "0",
    }
    return None


def test_trace_mode_preparation_removes_stale_trace_file(tmp_path) -> None:
    """trace 执行前应清理旧 tracing_output.log，避免 append 证据串线。"""
    from tpch_monetdb.tools.tpch import run as run_module

    stale_trace = tmp_path / "tracing_output.log"
    stale_trace.write_text("PROFILE old 1\n", encoding="utf-8")
    tool = run_module.RunTool.__new__(run_module.RunTool)
    tool.cwd = tmp_path

    tool._prepare_trace_output_file(trace_mode=True)

    assert not stale_trace.exists()
    return None


def test_run_worker_rejects_non_csv_output_for_validator(
    tmp_path,
    monkeypatch,
) -> None:
    """validator 路径必须保持 full CSV，避免 correctness 读不到 result*.csv。"""
    from tpch_monetdb.tools.tpch import run as run_module

    class FakeCompiler:
        def set_extra_cxxflags(self, _flags) -> None:
            return None

        def build_cached(self, **_kwargs):
            return None, False, "hash"

    class FakeValidator:
        sf_list = [1]

    monkeypatch.setattr(run_module, "make_compiler", lambda *_args, **_kwargs: FakeCompiler())

    tool = run_module.RunTool(
        cwd=tmp_path,
        dataset_name="tpch",
        base_data_dir=str(tmp_path),
        query_validator=FakeValidator(),
        wandb_metrics_hook=None,
        compile_cache_dir=None,
        git_snapshotter=None,
    )

    with pytest.raises(RuntimeError, match="Correctness validation requires"):
        tool.run_worker(scale_factor=1, optimize=False, output_mode="no_output")
    return None


def test_optimization_exec_callback_passes_no_output_mode(tmp_path) -> None:
    """optimization measurement 回调应能把 no_output 传给 RunTool。"""
    from tpch_monetdb.conversations.optimization_conversation_tpch_monetdb import (
        QUERY_OUTPUT_MODE_NO_OUTPUT,
        TpchMonetdbOptimizationConversation,
    )
    from tpch_monetdb.tools.tpch.run import RunWorkerResult

    calls: list[dict[str, object]] = []

    class FakeRunTool:
        def run_worker(self, **kwargs) -> RunWorkerResult:
            calls.append(kwargs)
            return RunWorkerResult(
                msg="ok",
                resp="resp",
                out="1 | Execution ms: 1.000\n1 | Query ms: 2.000\n",
                err="",
            )

    conversation = TpchMonetdbOptimizationConversation.__new__(TpchMonetdbOptimizationConversation)
    conversation.benchmark_sf = 100
    conversation.run_tool = FakeRunTool()

    callback = conversation._make_exec_callback(
        "1",
        output_mode=QUERY_OUTPUT_MODE_NO_OUTPUT,
    )
    callback(["1 args"], 30)

    assert calls[0]["output_mode"] == "no_output"
    assert calls[0]["stdin_args_data"] == ["1 args"]
    return None


def test_workspace_editor_allows_nested_file_paths(tmp_path) -> None:
    editor = WorkspaceEditor(root=tmp_path, wandb_metrics_hook=None)
    operation = ApplyPatchOperation(
        type="create_file",
        path="sub/dir/file.txt",
        diff="+hello\n",
    )

    result = editor.create_file(operation)

    assert result.output is not None
    assert "Created" in result.output
    assert (tmp_path / "sub" / "dir" / "file.txt").read_text(encoding="utf-8") == "hello"


def test_workspace_editor_rejects_noop_patch(tmp_path) -> None:
    target = tmp_path / "query_impl.cpp"
    target.write_text("void query() {\n    return;\n}\n", encoding="utf-8")
    editor = WorkspaceEditor(root=tmp_path, wandb_metrics_hook=None)
    operation = ApplyPatchOperation(
        type="update_file",
        path="query_impl.cpp",
        diff="@@\n-void query() {\n+void query() {\n",
    )

    result = editor.update_file(operation)

    assert result.output is not None
    assert "patch produced no changes" in result.output
    assert target.read_text(encoding="utf-8") == "void query() {\n    return;\n}\n"


def test_git_snapshotter_allows_repeated_snapshot_names(tmp_path) -> None:
    snapshotter = GitSnapshotter(working_dir=tmp_path, cache_repo=None)
    target = tmp_path / "state.txt"
    target.write_text("v1\n", encoding="utf-8")
    _, first_commit = snapshotter.snapshot("same-request-hash")

    target.write_text("v2\n", encoding="utf-8")
    _, second_commit = snapshotter.snapshot("same-request-hash")

    assert first_commit is not None
    assert second_commit is not None
    assert first_commit != second_commit


def test_git_snapshotter_allows_repeated_empty_snapshot_names(tmp_path) -> None:
    snapshotter = GitSnapshotter(working_dir=tmp_path, cache_repo=None)
    first_commit = snapshotter.create_empty_snapshot("same-conv")
    second_commit = snapshotter.create_empty_snapshot("same-conv")

    assert first_commit
    assert second_commit
    assert first_commit != second_commit


def test_git_snapshotter_checkout_paths_from_snapshot_restores_selected_files(
    tmp_path,
) -> None:
    snapshotter = GitSnapshotter(working_dir=tmp_path, cache_repo=None)
    first = tmp_path / "query_q1.cpp"
    second = tmp_path / "query_q9.cpp"
    first.write_text("q1-v1\n", encoding="utf-8")
    second.write_text("q9-v1\n", encoding="utf-8")
    _, candidate_snapshot = snapshotter.snapshot("candidate")
    assert candidate_snapshot is not None

    first.write_text("q1-v2\n", encoding="utf-8")
    second.write_text("q9-v2\n", encoding="utf-8")
    snapshotter.checkout_paths_from_snapshot(candidate_snapshot, ("query_q1.cpp",))

    assert first.read_text(encoding="utf-8") == "q1-v1\n"
    assert second.read_text(encoding="utf-8") == "q9-v2\n"


def test_git_snapshotter_clean_worktree_removes_untracked_files(tmp_path) -> None:
    snapshotter = GitSnapshotter(working_dir=tmp_path, cache_repo=None)
    snapshotter.create_empty_snapshot("seed")
    target = tmp_path / "storage_plan.txt"
    target.write_text("plan\n", encoding="utf-8")

    assert snapshotter.is_dirty() is True

    snapshotter.clean_worktree()

    assert snapshotter.is_dirty() is False
    assert target.exists() is False


def test_git_snapshotter_clean_worktree_restores_tracked_files(tmp_path) -> None:
    snapshotter = GitSnapshotter(working_dir=tmp_path, cache_repo=None)
    target = tmp_path / "query_impl.cpp"
    target.write_text("v1\n", encoding="utf-8")
    snapshotter.snapshot("seed")
    target.write_text("v2\n", encoding="utf-8")
    extra = tmp_path / "storage_plan.txt"
    extra.write_text("plan\n", encoding="utf-8")

    assert snapshotter.is_dirty() is True

    snapshotter.clean_worktree()

    assert snapshotter.is_dirty() is False
    assert target.read_text(encoding="utf-8") == "v1\n"
    assert extra.exists() is False


def test_prepare_runtime_workspace_cleans_dirty_runtime_repo(tmp_path) -> None:
    calls: list[bool] = []

    class FakeSnapshotter:
        def __init__(self) -> None:
            self._dirty_checks = [True, False]

        def is_dirty(self) -> bool:
            return self._dirty_checks.pop(0)

        def clean_worktree(self, include_ignored: bool = True) -> None:
            calls.append(include_ignored)
            return None

    tpch_monetdb.main_tpch_monetdb._prepare_runtime_workspace(FakeSnapshotter(), tmp_path)

    assert calls == [True]


def test_prepare_runtime_workspace_also_cleans_when_already_clean(tmp_path) -> None:
    calls: list[bool] = []

    class FakeSnapshotter:
        def __init__(self) -> None:
            self._dirty_checks = [False, False]

        def is_dirty(self) -> bool:
            return self._dirty_checks.pop(0)

        def clean_worktree(self, include_ignored: bool = True) -> None:
            calls.append(include_ignored)
            return None

    tpch_monetdb.main_tpch_monetdb._prepare_runtime_workspace(FakeSnapshotter(), tmp_path)

    assert calls == [True]


def test_snapshot_final_workspace_state_commits_dirty_workspace() -> None:
    captured: list[str] = []

    class FakeSnapshotter:
        def __init__(self) -> None:
            self.working_dir = Path("/tmp/runtime")

        def is_dirty(self) -> bool:
            return True

        def snapshot(self, name: str) -> tuple[str | None, str | None]:
            captured.append(name)
            return "parent", "commit"

    tpch_monetdb.main_tpch_monetdb._snapshot_final_workspace_state(FakeSnapshotter(), "tpch_monetdb_storageplan1-15v1_r001")

    assert captured == ["tpch_monetdb_storageplan1-15v1_r001-finalize"]


def test_provider_prefixed_kimi_aliases_resolve_to_builtin_pricing() -> None:
    direct = get_model_pricing("kimi-k2.5")
    anthropic_prefixed = get_model_pricing("anthropic/kimi-k2.5")
    openai_prefixed = get_model_pricing("openai/kimi-k2.5")

    assert anthropic_prefixed == direct
    assert openai_prefixed == direct
    assert direct.cached_input == pytest.approx(0.10 / 1_000_000)
    assert direct.input == pytest.approx(0.60 / 1_000_000)
    assert direct.output == pytest.approx(3.00 / 1_000_000)
    assert get_context_window("anthropic/kimi-k2.5") == 262144
    assert get_context_window("openai/kimi-k2.5") == 262144


def test_gpt55_pricing_uses_official_rates() -> None:
    direct = get_model_pricing("gpt-5.5")
    openai_prefixed = get_model_pricing("openai/gpt-5.5")

    assert openai_prefixed == direct
    assert direct.input == pytest.approx(5.00 / 1_000_000)
    assert direct.cached_input == pytest.approx(0.50 / 1_000_000)
    assert direct.output == pytest.approx(30.00 / 1_000_000)
    assert direct.context_window == 272_000
    assert direct.max_output_tokens == 128_000
    assert direct.tier_threshold is None
    assert direct.long_input is None
    assert direct.long_cached_input is None
    assert direct.long_output is None
    assert get_context_window("openai/gpt-5.5") == 272_000


def test_gpt55_cost_uses_flat_official_rates() -> None:
    cost = request_cost_usd(
        "openai/gpt-5.5",
        input_tokens=272_000,
        cached_tokens=72_000,
        output_tokens=100_000,
    )
    expected = (200_000 * 5.00 + 72_000 * 0.50 + 100_000 * 30.00) / 1_000_000
    assert cost == pytest.approx(expected)


def test_kimi_cost_uses_official_cached_and_uncached_rates() -> None:
    cost = request_cost_usd(
        "anthropic/kimi-k2.5",
        input_tokens=1_000_000,
        cached_tokens=100_000,
        output_tokens=2_000_000,
    )

    assert cost == pytest.approx(6.55)


def test_glm5_short_input_pricing() -> None:
    cost = request_cost_usd(
        "glm-5",
        input_tokens=31_999,
        cached_tokens=0,
        output_tokens=1_000_000,
    )
    assert cost == pytest.approx(3.231999)


def test_glm5_long_input_pricing() -> None:
    cost = request_cost_usd(
        "glm-5",
        input_tokens=32_000,
        cached_tokens=0,
        output_tokens=1_000_000,
    )
    assert cost == pytest.approx(3.232)


def test_glm5_cached_input_pricing() -> None:
    cost = request_cost_usd(
        "glm-5",
        input_tokens=50_000,
        cached_tokens=10_000,
        output_tokens=500_000,
    )
    expected = (40_000 * 1.00 + 10_000 * 0.20 + 500_000 * 3.20) / 1_000_000
    assert cost == pytest.approx(expected)


def test_glm5_context_window_is_200k() -> None:
    assert get_context_window("glm-5") == 200_000


def test_provider_prefixed_glm5_aliases_resolve_to_builtin_pricing() -> None:
    direct = get_model_pricing("glm-5")
    zhipu_prefixed = get_model_pricing("zhipu/glm-5")
    openai_prefixed = get_model_pricing("openai/glm-5")
    anthropic_prefixed = get_model_pricing("anthropic/glm-5")

    assert zhipu_prefixed == direct
    assert openai_prefixed == direct
    assert anthropic_prefixed == direct
    assert direct.input == pytest.approx(1.00 / 1_000_000)
    assert direct.cached_input == pytest.approx(0.20 / 1_000_000)
    assert direct.output == pytest.approx(3.20 / 1_000_000)
    assert direct.context_window == 200_000
    assert direct.tier_threshold is None
    assert direct.long_input is None
    assert direct.long_cached_input is None
    assert direct.long_output is None


def test_qwen36_plus_short_input_uses_tier1_pricing() -> None:
    cost = request_cost_usd(
        "qwen3.6-plus",
        input_tokens=255_999,
        cached_tokens=0,
        output_tokens=1_000_000,
    )
    assert cost == pytest.approx((255_999 * 0.276 + 1_000_000 * 1.651) / 1_000_000)


def test_qwen36_plus_long_input_uses_tier2_pricing() -> None:
    cost = request_cost_usd(
        "qwen3.6-plus",
        input_tokens=256_000,
        cached_tokens=0,
        output_tokens=1_000_000,
    )
    assert cost == pytest.approx((256_000 * 1.101 + 1_000_000 * 6.602) / 1_000_000)


def test_qwen36_plus_context_window_is_1m() -> None:
    assert get_context_window("qwen3.6-plus") == 1_000_000


def test_provider_prefixed_qwen36_plus_aliases_resolve_to_builtin_pricing() -> None:
    direct = get_model_pricing("qwen3.6-plus")
    anthropic_prefixed = get_model_pricing("anthropic/qwen3.6-plus")
    openai_prefixed = get_model_pricing("openai/qwen3.6-plus")

    assert anthropic_prefixed == direct
    assert openai_prefixed == direct
    assert direct.input == pytest.approx(0.276 / 1_000_000)
    assert direct.cached_input == pytest.approx(0.276 / 1_000_000)
    assert direct.output == pytest.approx(1.651 / 1_000_000)
    assert direct.context_window == 1_000_000
    assert direct.tier_threshold == 256_000
    assert direct.long_input == pytest.approx(1.101 / 1_000_000)
    assert direct.long_cached_input == pytest.approx(1.101 / 1_000_000)
    assert direct.long_output == pytest.approx(6.602 / 1_000_000)


def test_deepseek_v4_flash_pricing_uses_official_rates() -> None:
    """Verify deepseek-v4-flash pricing matches the official rate card.

    cache miss input = $0.14/M, cache hit input = $0.0028/M, output = $0.28/M.
    """
    direct = get_model_pricing("deepseek-v4-flash")

    assert direct.input == pytest.approx(0.14 / 1_000_000)
    assert direct.cached_input == pytest.approx(0.0028 / 1_000_000)
    assert direct.output == pytest.approx(0.28 / 1_000_000)
    assert direct.context_window == 1_000_000
    assert direct.tier_threshold is None


def test_deepseek_v4_pro_pricing_uses_official_rates() -> None:
    """Verify deepseek-v4-pro pricing uses the 2.5-discounted rate card.

    cache miss input = $0.435/M, cache hit input = $0.003625/M, output = $0.87/M.
    """
    direct = get_model_pricing("deepseek-v4-pro")

    assert direct.input == pytest.approx(0.435 / 1_000_000)
    assert direct.cached_input == pytest.approx(0.003625 / 1_000_000)
    assert direct.output == pytest.approx(0.87 / 1_000_000)
    assert direct.context_window == 1_000_000
    assert direct.tier_threshold is None


def test_deepseek_v4_context_window_is_1m() -> None:
    """Both deepseek-v4 variants advertise a 1M-token context window."""
    assert get_context_window("deepseek-v4-flash") == 1_000_000
    assert get_context_window("deepseek-v4-pro") == 1_000_000


def test_provider_prefixed_deepseek_aliases_resolve_to_builtin_pricing() -> None:
    """Ensure deepseek pricing is reachable via deepseek/ openai/ anthropic/ prefixes."""
    flash_direct = get_model_pricing("deepseek-v4-flash")
    pro_direct = get_model_pricing("deepseek-v4-pro")

    for prefix in ("deepseek/", "openai/", "anthropic/"):
        assert get_model_pricing(f"{prefix}deepseek-v4-flash") == flash_direct
        assert get_model_pricing(f"{prefix}deepseek-v4-pro") == pro_direct


def test_deepseek_v4_flash_cost_uses_official_cached_and_uncached_rates() -> None:
    """Verify hybrid cache-hit/miss billing math for deepseek-v4-flash."""
    cost = request_cost_usd(
        "anthropic/deepseek-v4-flash",
        input_tokens=1_000_000,
        cached_tokens=100_000,
        output_tokens=2_000_000,
    )

    expected = (900_000 * 0.14 + 100_000 * 0.0028 + 2_000_000 * 0.28) / 1_000_000
    assert cost == pytest.approx(expected)


def test_deepseek_v4_pro_cost_uses_official_cached_and_uncached_rates() -> None:
    """Verify hybrid cache-hit/miss billing math for discounted deepseek-v4-pro."""
    cost = request_cost_usd(
        "deepseek/deepseek-v4-pro",
        input_tokens=1_000_000,
        cached_tokens=100_000,
        output_tokens=2_000_000,
    )

    expected = (900_000 * 0.435 + 100_000 * 0.003625 + 2_000_000 * 0.87) / 1_000_000
    assert cost == pytest.approx(expected)


def test_setup_model_config_tracks_deepseek_accounting_model_name(monkeypatch) -> None:
    """setup_model_config should normalize provider-prefixed deepseek names for accounting."""
    monkeypatch.setenv("LITELLM_API_KEY", "test-key")
    monkeypatch.setenv("LITELLM_BASE_URL", "https://api.deepseek.com")

    config = setup_model_config("litellm/openai/deepseek-v4-pro")

    assert config.model_name == "openai/deepseek-v4-pro"
    assert config.accounting_model_name == "deepseek-v4-pro"
    assert config.base_url == "https://api.deepseek.com"


def test_setup_model_config_rejects_anthropic_deepseek_models(monkeypatch) -> None:
    monkeypatch.setenv("LITELLM_API_KEY", "test-key")
    monkeypatch.setenv("LITELLM_BASE_URL", "https://api.deepseek.com")

    with pytest.raises(RuntimeError, match="only supported via the native LiteLLM DeepSeek provider"):
        setup_model_config("litellm/anthropic/deepseek-v4-pro")


def test_setup_model_config_allows_legacy_openai_deepseek_with_warning(monkeypatch, caplog) -> None:
    """openai/deepseek-v4 路径仍然可用但输出废弃警告."""
    monkeypatch.setenv("LITELLM_API_KEY", "test-key")
    monkeypatch.setenv("LITELLM_BASE_URL", "https://api.deepseek.com")

    import logging
    caplog.set_level(logging.WARNING, logger="tpch_monetdb.utils.model_setup")

    config = setup_model_config("litellm/openai/deepseek-v4-pro")
    assert config.use_litellm is True
    assert "Deprecated" in caplog.text


def test_setup_model_config_allows_native_deepseek_path(monkeypatch) -> None:
    """deepseek/deepseek-v4-pro 原生路径无需 LITELLM_BASE_URL."""
    monkeypatch.setenv("LITELLM_API_KEY", "test-key")
    monkeypatch.delenv("LITELLM_BASE_URL", raising=False)

    config = setup_model_config("litellm/deepseek/deepseek-v4-pro")
    assert config.use_litellm is True
    assert config.model_name == "deepseek/deepseek-v4-pro"
    assert config.base_url is None  # native provider has default endpoint


def test_setup_model_config_tracks_accounting_model_name(monkeypatch) -> None:
    monkeypatch.setenv("LITELLM_API_KEY", "test-key")
    monkeypatch.setenv(
        "LITELLM_BASE_URL",
        "https://coding.dashscope.aliyuncs.com/apps/anthropic",
    )

    config = setup_model_config("litellm/anthropic/kimi-k2.5")

    assert config.model_name == "anthropic/kimi-k2.5"
    assert config.accounting_model_name == "kimi-k2.5"
    assert config.base_url == "https://coding.dashscope.aliyuncs.com/apps/anthropic"


def test_setup_model_config_tracks_qwen_accounting_model_name(monkeypatch) -> None:
    monkeypatch.setenv("LITELLM_API_KEY", "test-key")
    monkeypatch.setenv(
        "LITELLM_BASE_URL",
        "https://coding.dashscope.aliyuncs.com/apps/anthropic",
    )

    config = setup_model_config("litellm/anthropic/qwen3.6-plus")

    assert config.model_name == "anthropic/qwen3.6-plus"
    assert config.accounting_model_name == "qwen3.6-plus"
    assert config.base_url == "https://coding.dashscope.aliyuncs.com/apps/anthropic"


def test_log_pending_exception_logs_primary_and_wandb_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, BaseException | None]] = []

    def fake_error(message: str, *, exc_info=None) -> None:
        calls.append((message, exc_info))
        return None

    monkeypatch.setattr(tpch_monetdb.main_tpch_monetdb.logger, "error", fake_error)
    pending = RuntimeError("primary")
    finalize = RuntimeError("finalize")
    teardown = RuntimeError("teardown")

    tpch_monetdb.main_tpch_monetdb._log_pending_exception(pending, finalize, teardown)

    assert calls == [
        ("Primary exception while finalizing TPC-H MonetDB run", pending),
        ("W&B finalize failed while propagating primary exception", finalize),
        ("W&B teardown failed while propagating primary exception", teardown),
    ]
    return None


def test_create_compaction_session_passes_litellm_base_url(tmp_path) -> None:
    underlying_session = SimpleNamespace()

    session = tpch_monetdb.main_tpch_monetdb._create_compaction_session(
        use_litellm=True,
        session_id="conv-test",
        model_name="anthropic/kimi-k2.5",
        api_key="test-key",
        base_url="https://coding.dashscope.aliyuncs.com/apps/anthropic",
        client=None,
        underlying_session=underlying_session,
        cache_path=tmp_path,
        wandb_metrics_hook=None,
        compaction_model_map=None,
    )

    assert isinstance(session, CachedLitellmCompactionSession)
    assert session.base_url == "https://coding.dashscope.aliyuncs.com/apps/anthropic"
    assert session._underlying_session is underlying_session


@pytest.mark.asyncio
async def test_litellm_compaction_summary_uses_base_url(tmp_path, monkeypatch) -> None:
    captured: dict[str, object] = {}

    async def fake_acompletion(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="compacted summary")
                )
            ]
        )

    monkeypatch.setattr(
        "tpch_monetdb.llm_cache.cached_litellm_compaction.litellm.acompletion",
        fake_acompletion,
    )

    session = CachedLitellmCompactionSession(
        session_id="conv-test",
        model="anthropic/kimi-k2.5",
        api_key="test-key",
        base_url="https://coding.dashscope.aliyuncs.com/apps/anthropic",
        cache_dir=tmp_path,
        wandb_metrics_hook=None,
        compaction_model_map=None,
    )

    summary = await session._generate_summary(
        [{"role": "user", "content": "hello"}],
        "anthropic/kimi-k2.5",
    )

    assert "compacted summary" in summary
    assert captured["api_key"] == "test-key"
    assert captured["base_url"] == "https://coding.dashscope.aliyuncs.com/apps/anthropic"
    assert captured["model"] == "anthropic/kimi-k2.5"
    assert captured["temperature"] == 0.0


@pytest.mark.asyncio
async def test_litellm_compaction_summary_disables_deepseek_thinking(tmp_path, monkeypatch) -> None:
    captured: dict[str, object] = {}

    async def fake_acompletion(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="compacted summary")
                )
            ]
        )

    monkeypatch.setattr(
        "tpch_monetdb.llm_cache.cached_litellm_compaction.litellm.acompletion",
        fake_acompletion,
    )

    session = CachedLitellmCompactionSession(
        session_id="conv-test",
        model="deepseek/deepseek-v4-flash",
        api_key="test-key",
        base_url=None,
        cache_dir=tmp_path,
        wandb_metrics_hook=None,
        compaction_model_map=None,
    )

    summary = await session._generate_summary(
        [{"role": "user", "content": "hello"}],
        "deepseek/deepseek-v4-flash",
    )

    assert "compacted summary" in summary
    assert captured["model"] == "deepseek/deepseek-v4-flash"
    assert captured["thinking"] == {"type": "disabled"}
    assert "reasoning_effort" not in captured
    assert captured["allowed_openai_params"] == ["thinking", "reasoning_effort"]
    assert captured["additional_drop_params"] == ["extra_body"]
    assert "temperature" not in captured


@pytest.mark.asyncio
async def test_litellm_compaction_summary_normalizes_list_content(
    tmp_path, monkeypatch
) -> None:
    async def fake_acompletion(**_kwargs):
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=[
                            {"type": "text", "text": "<summary>first line</summary>"},
                            {"type": "text", "text": "ignored tail"},
                        ]
                    )
                )
            ]
        )

    monkeypatch.setattr(
        "tpch_monetdb.llm_cache.cached_litellm_compaction.litellm.acompletion",
        fake_acompletion,
    )

    session = CachedLitellmCompactionSession(
        session_id="conv-test",
        model="anthropic/kimi-k2.5",
        api_key="test-key",
        base_url="https://coding.dashscope.aliyuncs.com/apps/anthropic",
        cache_dir=tmp_path,
        wandb_metrics_hook=None,
        compaction_model_map=None,
    )

    summary = await session._generate_summary(
        [{"role": "user", "content": "hello"}],
        "anthropic/kimi-k2.5",
    )

    assert summary.startswith("[Compaction Summary v3]")
    assert "first line" in summary
    assert "source_refs:" in summary


@pytest.mark.asyncio
async def test_litellm_compaction_summary_rejects_empty_content(
    tmp_path, monkeypatch
) -> None:
    async def fake_acompletion(**_kwargs):
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=None)
                )
            ]
        )

    monkeypatch.setattr(
        "tpch_monetdb.llm_cache.cached_litellm_compaction.litellm.acompletion",
        fake_acompletion,
    )

    session = CachedLitellmCompactionSession(
        session_id="conv-test",
        model="anthropic/kimi-k2.5",
        api_key="test-key",
        base_url="https://coding.dashscope.aliyuncs.com/apps/anthropic",
        cache_dir=tmp_path,
        wandb_metrics_hook=None,
        compaction_model_map=None,
    )

    with pytest.raises(RuntimeError, match="Compaction model returned empty content"):
        await session._generate_summary(
            [{"role": "user", "content": "hello"}],
            "anthropic/kimi-k2.5",
        )


@pytest.mark.asyncio
async def test_litellm_compaction_summary_retries_transient_network_failure(
    tmp_path, monkeypatch
) -> None:
    class FakeAPIConnectionError(Exception):
        pass

    attempts = {"count": 0}
    sleeps: list[float] = []

    async def fake_acompletion(**_kwargs):
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise FakeAPIConnectionError("Server disconnected")
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="compacted summary")
                )
            ]
        )

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)
        return None

    monkeypatch.setattr(
        "tpch_monetdb.llm_cache.cached_litellm_compaction.litellm.acompletion",
        fake_acompletion,
    )
    monkeypatch.setattr(
        "tpch_monetdb.llm_cache.litellm_retry.asyncio.sleep",
        fake_sleep,
    )

    session = CachedLitellmCompactionSession(
        session_id="conv-test",
        model="anthropic/kimi-k2.5",
        api_key="test-key",
        base_url="https://coding.dashscope.aliyuncs.com/apps/anthropic",
        cache_dir=tmp_path,
        wandb_metrics_hook=None,
        compaction_model_map=None,
    )

    summary = await session._generate_summary(
        [{"role": "user", "content": "hello"}],
        "anthropic/kimi-k2.5",
    )

    assert "compacted summary" in summary
    assert attempts["count"] == 3
    assert sleeps == [1.0, 2.0]


@pytest.mark.asyncio
async def test_litellm_compaction_keeps_session_when_summary_ineffective(
    tmp_path, monkeypatch
) -> None:
    class FakeUnderlying:
        def __init__(self) -> None:
            self.items = [
                {"role": "user", "content": "short-1"},
                {"role": "assistant", "content": "short-2"},
                {"role": "user", "content": "short-3"},
            ]
            self.clear_calls = 0
            self.added_items: list[dict[str, object]] = []

        async def get_items(self) -> list[dict[str, object]]:
            return list(self.items)

        async def clear_session(self) -> None:
            self.clear_calls += 1
            self.items = []

        async def add_items(self, items) -> None:
            self.added_items = list(items)
            self.items.extend(items)

    async def fake_generate_summary(_items, _model) -> str:
        return "[Compaction Summary v3]\n" + ("expanded summary " * 5000)

    underlying = FakeUnderlying()
    original_items = list(underlying.items)
    session = CachedLitellmCompactionSession(
        session_id="conv-test",
        model="anthropic/kimi-k2.5",
        api_key="test-key",
        base_url="https://example.com",
        cache_dir=tmp_path,
        wandb_metrics_hook=None,
        compaction_model_map=None,
    )
    session.set_underlying_session(underlying)
    monkeypatch.setattr(session, "_generate_summary", fake_generate_summary)

    attempt = await session.run_compaction({"force_trigger": True})

    assert attempt.status == "success"
    assert attempt.effective is False
    assert underlying.clear_calls == 0
    assert underlying.added_items == []
    assert underlying.items == original_items
    assert list(tmp_path.glob("*.pkl")) == []


def test_format_compact_summary_rejects_invalid_inputs() -> None:
    assert format_compact_summary("<summary>ok</summary>") == "Summary:\nok"

    with pytest.raises(TypeError, match="requires a string response"):
        format_compact_summary(None)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="non-empty string response"):
        format_compact_summary("   ")

    with pytest.raises(ValueError, match="empty <summary> block"):
        format_compact_summary("<summary>   </summary>")


def test_auto_compact_manager_uses_kimi_alias_context_window() -> None:
    manager = AutoCompactManager("anthropic/kimi-k2.5")
    usage = manager.describe_usage(230000)

    assert manager.get_effective_context_window() == 242144
    assert manager.get_threshold() == 229144
    assert manager.get_warning_threshold() == 209144
    assert manager.get_blocking_threshold() == 239144
    assert usage["current_tokens"] == 230000
    assert usage["is_above_auto_compact_threshold"] is True
    assert usage["is_above_warning_threshold"] is True
    assert usage["is_at_blocking_limit"] is False


def test_auto_compact_manager_uses_deepseek_1m_context_window() -> None:
    """deepseek-v4 系列应基于 1M 上下文计算所有阈值，证实 1M 已贯穿 auto_compact 链路."""
    manager = AutoCompactManager("anthropic/deepseek-v4-pro")

    # context_window = 1_000_000
    # effective = 1_000_000 - MAX_OUTPUT_RESERVE(20_000) = 980_000
    # threshold = 980_000 - AUTO_COMPACT_BUFFER_TOKENS(13_000) = 967_000
    # warning  = 967_000 - WARNING_THRESHOLD_BUFFER_TOKENS(20_000) = 947_000
    # blocking = 980_000 - BLOCKING_THRESHOLD_BUFFER_TOKENS(3_000) = 977_000
    assert manager.context_window == 1_000_000
    assert manager.get_effective_context_window() == 980_000
    assert manager.get_threshold() == 967_000
    assert manager.get_warning_threshold() == 947_000
    assert manager.get_blocking_threshold() == 977_000

    # 在 800K 时（kimi 早就触发了），deepseek 仍未到压缩阈值
    usage = manager.describe_usage(800_000)
    assert usage["is_above_auto_compact_threshold"] is False
    assert usage["is_above_warning_threshold"] is False
    assert usage["is_at_blocking_limit"] is False

    # flash 同款 1M 上下文
    flash_manager = AutoCompactManager("anthropic/deepseek-v4-flash")
    assert flash_manager.context_window == 1_000_000
    assert flash_manager.get_threshold() == 967_000


def test_auto_compact_manager_uses_earlier_threshold_for_correctness_queries() -> None:
    manager = AutoCompactManager("anthropic/kimi-k2.5")

    default_warning = manager.get_warning_threshold()
    correctness_warning = manager.get_warning_threshold("correctness_queries_writeonly")
    foundation_warning = manager.get_warning_threshold("correctness_foundation")

    assert correctness_warning < default_warning
    assert foundation_warning == correctness_warning
    assert manager.should_compact(correctness_warning + 1, "correctness_queries_writeonly") is False
    assert manager.should_compact(manager.get_threshold("correctness_queries_writeonly"), "correctness_queries_writeonly") is True
    assert manager.should_compact(manager.get_threshold("correctness_foundation"), "correctness_foundation") is True


def test_contextual_input_keeps_dynamic_refs_after_current_task() -> None:
    rendered = tpch_monetdb.main_tpch_monetdb._build_contextual_input(
        stage_hint="stage rules",
        scoped_stage_rules="[Rule File: scripted.md]\nkeep stage scope",
        stage_contract="- OBLIG_Q9_JOIN_PROFIT_AGGREGATION",
        current_task="implement q9",
        stage_memory="[Stage Memory v3]\nopen_failures: []",
        artifact_context="[Artifact Refs]\n- artifact_ref=run_q9",
    )

    runtime_pos = rendered.index("[Runtime Stage Hint]")
    rules_pos = rendered.index("[Scoped Stage Rules]")
    contract_pos = rendered.index("[Stage Contract]")
    task_pos = rendered.index("[Current Task]")
    memory_pos = rendered.index("[Stage Memory v3]")
    refs_pos = rendered.index("[Artifact Refs]")

    assert runtime_pos < rules_pos < contract_pos < task_pos < memory_pos < refs_pos
    assert "path=" not in rendered


def test_stage_contract_renders_q1_q9_obligations() -> None:
    contract = tpch_monetdb.main_tpch_monetdb._build_stage_contract(("2", "9", "1"))

    assert "OBLIG_Q1_LINEITEM_SCAN_AGGREGATION" in contract
    assert "OBLIG_Q9_JOIN_PROFIT_AGGREGATION" in contract
    assert "reusable lineitem columns" in contract
    assert "aggregate profit at query time" in contract
    assert tpch_monetdb.main_tpch_monetdb._build_stage_contract(("2", "3")) == ""


def test_litellm_compaction_split_items_preserves_all_when_keep_recent_covers_tail(
    tmp_path,
) -> None:
    session = CachedLitellmCompactionSession(
        session_id="conv-test",
        model="anthropic/kimi-k2.5",
        api_key="test-key",
        base_url="https://example.com",
        cache_dir=tmp_path,
        wandb_metrics_hook=None,
        compaction_model_map=None,
    )

    items = [
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
    ]

    candidate_items, preserved_items = session._split_items(items, keep_recent=10)

    assert candidate_items == []
    assert preserved_items == items


def test_litellm_compaction_stage_memory_selection_preserves_summary_and_candidates(
    tmp_path,
) -> None:
    session = CachedLitellmCompactionSession(
        session_id="conv-test",
        model="anthropic/kimi-k2.5",
        api_key="test-key",
        base_url="https://example.com",
        cache_dir=tmp_path,
        wandb_metrics_hook=None,
        compaction_model_map=None,
    )
    items: list[dict[str, object]] = [
        {"role": "system", "content": "rules"},
        {"role": "assistant", "content": "[Stage Summary]\nStage: compile_fix"},
        {"type": "function_call", "call_id": "c1", "name": "compile", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "c1", "output": "compile ok"},
        {"type": "function_call", "call_id": "r1", "name": "run", "arguments": "{}"},
        {
            "type": "function_call_output",
            "call_id": "r1",
            "output": "Validation failed:\nQuestDB rows mismatch",
        },
    ]
    for idx in range(20):
        items.append({"role": "assistant", "content": f"tail-{idx}"})

    selection = session._select_items(
        items=items,
        keep_recent=0,
        selection_policy="stage_memory_v3",
        preserve_limit_items=12,
        min_candidate_items=8,
    )

    preserved_text = "\n".join(
        str(item.get("content") or item.get("output") or "")
        for item in selection.preserved_items
        if isinstance(item, dict)
    )
    assert "[Stage Summary]" in preserved_text
    assert "Validation failed" in preserved_text
    assert selection.candidate_count >= 8
    assert selection.candidate_count > 0
    assert selection.preserved_count <= len(items)


def test_litellm_compaction_stage_memory_v3_preserves_semantic_evidence(
    tmp_path,
) -> None:
    session = CachedLitellmCompactionSession(
        session_id="conv-test",
        model="anthropic/kimi-k2.5",
        api_key="test-key",
        base_url="https://example.com",
        cache_dir=tmp_path,
        wandb_metrics_hook=None,
        compaction_model_map=None,
    )
    items: list[dict[str, object]] = [
        {"role": "system", "content": "rules"},
        {"role": "assistant", "content": "[Stage Memory v3]\nopen_failures: [q9 regression]"},
        {"role": "assistant", "content": "[Optimization Control Summary]\nworkload_objective.json read"},
        {"role": "assistant", "content": "artifact_ref: run_q9\nsha256: abc\nQ9 failed"},
    ]
    for idx in range(30):
        items.append({"role": "assistant", "content": f"stale-tail-{idx}"})

    selection = session._select_items(
        items=items,
        keep_recent=0,
        selection_policy="stage_memory_v3",
        preserve_limit_items=6,
        min_candidate_items=10,
    )

    preserved_text = "\n".join(
        str(item.get("content") or item.get("output") or "")
        for item in selection.preserved_items
        if isinstance(item, dict)
    )
    assert selection.selection_policy == "stage_memory_v3"
    assert "[Stage Memory v3]" in preserved_text
    assert "[Optimization Control Summary]" in preserved_text
    assert "artifact_ref: run_q9" in preserved_text
    assert selection.candidate_count >= 10


def test_litellm_compaction_select_items_keep_recent_path_wraps_split_items(
    tmp_path,
) -> None:
    session = CachedLitellmCompactionSession(
        session_id="conv-test",
        model="anthropic/kimi-k2.5",
        api_key="test-key",
        base_url="https://example.com",
        cache_dir=tmp_path,
        wandb_metrics_hook=None,
        compaction_model_map=None,
    )
    items = [
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "u2"},
    ]

    selection = session._select_items(
        items=items,
        keep_recent=2,
        selection_policy=None,
        preserve_limit_items=12,
        min_candidate_items=0,
    )

    assert selection.candidate_count == 1
    assert selection.preserved_count == 3
    assert selection.skip_reason is None


def test_litellm_compaction_rejects_removed_selection_policy(tmp_path) -> None:
    session = CachedLitellmCompactionSession(
        session_id="conv-test",
        model="anthropic/kimi-k2.5",
        api_key="test-key",
        base_url="https://example.com",
        cache_dir=tmp_path,
        wandb_metrics_hook=None,
        compaction_model_map=None,
    )

    with pytest.raises(ValueError, match="Unsupported compaction selection_policy"):
        session._select_items(
            items=[{"role": "user", "content": "u1"}],
            keep_recent=0,
            selection_policy="stage_memory_" + "v2",
            preserve_limit_items=12,
            min_candidate_items=0,
        )


def test_litellm_compaction_split_items_does_not_orphan_tool_output(
    tmp_path,
) -> None:
    session = CachedLitellmCompactionSession(
        session_id="conv-test",
        model="anthropic/kimi-k2.5",
        api_key="test-key",
        base_url="https://example.com",
        cache_dir=tmp_path,
        wandb_metrics_hook=None,
        compaction_model_map=None,
    )
    items = [
        {"role": "assistant", "content": "older"},
        {"type": "function_call", "call_id": "c1", "name": "edit_file", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "c1", "output": "updated"},
        {"role": "assistant", "content": "tail"},
    ]

    candidate_items, preserved_items = session._split_items(items, keep_recent=2)

    assert candidate_items == [{"role": "assistant", "content": "older"}]
    assert preserved_items == items[1:]


def test_litellm_compaction_stage_memory_releases_tool_pair_atomically(
    tmp_path,
) -> None:
    session = CachedLitellmCompactionSession(
        session_id="conv-test",
        model="anthropic/kimi-k2.5",
        api_key="test-key",
        base_url="https://example.com",
        cache_dir=tmp_path,
        wandb_metrics_hook=None,
        compaction_model_map=None,
    )
    items: list[dict[str, object]] = [
        {"role": "assistant", "content": "[Stage Summary]\nStage: compile_fix"},
        {"type": "function_call", "call_id": "c1", "name": "compile", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "c1", "output": "compile output"},
        {"type": "function_call", "call_id": "r1", "name": "run", "arguments": "{}"},
        {
            "type": "function_call_output",
            "call_id": "r1",
            "output": "Validation failed:\nrow mismatch",
        },
    ]
    for idx in range(5):
        items.append({"role": "assistant", "content": f"tail-{idx}"})

    selection = session._select_items(
        items=items,
        keep_recent=0,
        selection_policy="stage_memory_v3",
        preserve_limit_items=5,
        min_candidate_items=7,
    )

    compile_names = [
        item.get("name")
        for item in selection.preserved_items
        if isinstance(item, dict) and item.get("type") == "function_call"
    ]
    compile_outputs = [
        item.get("output")
        for item in selection.preserved_items
        if isinstance(item, dict) and item.get("type") == "function_call_output"
    ]
    assert "compile" not in compile_names
    assert "compile output" not in compile_outputs
    candidate_names = [
        item.get("name")
        for item in selection.candidate_items
        if isinstance(item, dict) and item.get("type") == "function_call"
    ]
    candidate_outputs = [
        item.get("output")
        for item in selection.candidate_items
        if isinstance(item, dict) and item.get("type") == "function_call_output"
    ]
    assert "compile" in candidate_names
    assert "compile output" in candidate_outputs


def test_litellm_compaction_stage_memory_preserves_leading_system_prompts(
    tmp_path,
) -> None:
    session = CachedLitellmCompactionSession(
        session_id="conv-test",
        model="anthropic/kimi-k2.5",
        api_key="test-key",
        base_url="https://example.com",
        cache_dir=tmp_path,
        wandb_metrics_hook=None,
        compaction_model_map=None,
    )
    items: list[dict[str, object]] = [
        {"role": "system", "content": "system rules"},
        {"role": "assistant", "content": "[Stage Summary]\nStage: compile_fix"},
        {"type": "function_call", "call_id": "r1", "name": "run", "arguments": "{}"},
        {
            "type": "function_call_output",
            "call_id": "r1",
            "output": "Validation failed:\nrow mismatch",
        },
    ]
    for idx in range(10):
        items.append({"role": "assistant", "content": f"tail-{idx}"})

    selection = session._select_items(
        items=items,
        keep_recent=0,
        selection_policy="stage_memory_v3",
        preserve_limit_items=5,
        min_candidate_items=4,
    )

    assert selection.preserved_items[0]["role"] == "system"
    assert selection.preserved_items[0]["content"] == "system rules"
    assert selection.preserved_count == len(selection.preserved_items)
    assert selection.candidate_count > 0


def test_litellm_compaction_context_diagnostics_detects_tool_and_read_bloat(
    tmp_path,
) -> None:
    session = CachedLitellmCompactionSession(
        session_id="conv-test",
        model="anthropic/kimi-k2.5",
        api_key="test-key",
        base_url="https://example.com",
        cache_dir=tmp_path,
        wandb_metrics_hook=None,
        compaction_model_map=None,
    )
    huge_text = "x" * 40000
    items: list[dict[str, object]] = [
        {"type": "function_call", "call_id": "read1", "name": "read_file", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "read1", "output": huge_text},
        {"type": "function_call", "call_id": "run1", "name": "run", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "run1", "output": huge_text},
        {"role": "assistant", "content": "small"},
    ]

    diagnostics = session._build_context_diagnostics(
        items,
        {"is_above_warning_threshold": True},
    )

    assert diagnostics["context/near_capacity"] == 1
    assert diagnostics["context/tool_bloat_tokens"] > 0
    assert diagnostics["context/read_bloat_tokens"] > 0


def test_litellm_compaction_chunks_large_candidate_window(tmp_path) -> None:
    session = CachedLitellmCompactionSession(
        session_id="conv-test",
        model="anthropic/kimi-k2.5",
        api_key="test-key",
        base_url="https://example.com",
        cache_dir=tmp_path,
        wandb_metrics_hook=None,
        compaction_model_map=None,
    )

    large_items = [
        {"role": "user", "content": f"{index}:" + ("x" * 5000)}
        for index in range(1200)
    ]

    chunks = session._chunk_items_for_summary(large_items, "anthropic/kimi-k2.5")

    assert len(chunks) >= 2
    assert sum(len(chunk) for chunk in chunks) == len(large_items)
    return None


@pytest.mark.asyncio
async def test_auto_compact_manager_handles_empty_compaction_content() -> None:
    manager = AutoCompactManager("anthropic/kimi-k2.5")

    class FakeSession:
        async def get_items(self) -> list[dict[str, str]]:
            return [{"role": "user", "content": f"m{i}"} for i in range(12)]

    class FakeCompactionSession:
        async def run_compaction(self, _args) -> None:
            raise RuntimeError("Compaction model returned empty content")

    succeeded = await manager.compact(
        session=FakeSession(),
        compaction_session=FakeCompactionSession(),
    )

    assert succeeded is False
    assert manager.consecutive_failures == 1


@pytest.mark.asyncio
async def test_auto_compact_manager_does_not_retry_when_v3_has_zero_candidates() -> None:
    manager = AutoCompactManager("anthropic/kimi-k2.5")
    captured_args: list[dict[str, object]] = []

    class FakeSession:
        async def get_items(self) -> list[dict[str, str]]:
            return [{"role": "user", "content": f"m{i}"} for i in range(3)]

    class FakeCompactionSession:
        async def run_compaction(self, args):
            captured_args.append(dict(args))
            return SimpleNamespace(
                status="skipped",
                candidate_count=0,
                preserved_count=3,
                skip_reason="too_few_items_total",
            )

    succeeded = await manager.compact(
        session=FakeSession(),
        compaction_session=FakeCompactionSession(),
        current_tokens=250000,
        profile_name="compile_fix",
    )

    assert succeeded is False
    assert len(captured_args) == 1
    assert captured_args[0]["selection_policy"] == "stage_memory_v3"
    assert manager.last_failure_info["reason"] == "too_few_items_total"


@pytest.mark.asyncio
async def test_auto_compact_manager_rejects_ineffective_success() -> None:
    manager = AutoCompactManager("anthropic/kimi-k2.5")
    captured_args: list[dict[str, object]] = []

    class FakeSession:
        async def get_items(self) -> list[dict[str, str]]:
            return [
                {"role": "system", "content": "system rules"},
                {"role": "user", "content": "m0"},
                {"role": "assistant", "content": "m1"},
                {"role": "user", "content": "m2"},
            ]

    class FakeCompactionSession:
        async def run_compaction(self, args):
            captured_args.append(dict(args))
            return SimpleNamespace(
                status="success",
                candidate_count=86,
                preserved_count=232,
                skip_reason=None,
                effective=False,
                chunk_count=1,
                estimated_candidate_tokens=0,
                estimated_candidate_chars=0,
                pre_tokens=570300,
                post_tokens=568500,
                pre_body_bytes=1800000,
                post_body_bytes=1795000,
            )

    succeeded = await manager.compact(
        session=FakeSession(),
        compaction_session=FakeCompactionSession(),
        current_tokens=250000,
        profile_name="compile_fix",
    )

    assert succeeded is False
    assert len(captured_args) == 1
    assert captured_args[0]["selection_policy"] == "stage_memory_v3"
    assert manager.last_failure_info["reason"] == "ineffective_compaction"


@pytest.mark.asyncio
async def test_auto_compact_manager_treats_skipped_attempt_as_failure() -> None:
    manager = AutoCompactManager("anthropic/kimi-k2.5")

    class FakeSession:
        async def get_items(self) -> list[dict[str, str]]:
            return [{"role": "user", "content": f"m{i}"} for i in range(40)]

    class FakeCompactionSession:
        async def run_compaction(self, _args):
            return SimpleNamespace(
                status="skipped",
                candidate_count=0,
                preserved_count=40,
                skip_reason="insufficient_items",
            )

    succeeded = await manager.compact(
        session=FakeSession(),
        compaction_session=FakeCompactionSession(),
        current_tokens=250000,
        profile_name="compile_fix",
    )

    assert succeeded is False
    assert manager.consecutive_failures == 1


@pytest.mark.asyncio
async def test_auto_compact_manager_logs_critical_when_threshold_exceeded_but_compaction_cannot_progress(
    caplog,
) -> None:
    from tpch_monetdb.llm_cache.auto_compact import MAX_CONSECUTIVE_FAILURES

    manager = AutoCompactManager("anthropic/kimi-k2.5")
    manager.consecutive_failures = MAX_CONSECUTIVE_FAILURES - 1

    class FakeSession:
        async def get_items(self) -> list[dict[str, str]]:
            return [{"role": "user", "content": "x" * 120000}]

    class FakeCompactionSession:
        async def run_compaction(self, _args):
            return SimpleNamespace(
                status="skipped",
                candidate_count=0,
                preserved_count=1,
                skip_reason="too_few_items_total",
            )

    with caplog.at_level("CRITICAL"):
        succeeded = await manager.compact(
            session=FakeSession(),
            compaction_session=FakeCompactionSession(),
            current_tokens=250000,
            profile_name="compile_fix",
        )

    assert succeeded is False
    assert "Circuit breaker triggered with context still above threshold" in caplog.text


def test_workspace_editor_rejects_empty_create_diff(tmp_path) -> None:
    editor = WorkspaceEditor(root=tmp_path, wandb_metrics_hook=None)
    operation = ApplyPatchOperation(type="create_file", path="TODO.md", diff="")

    result = editor.create_file(operation)

    assert result.output is not None
    assert "requires non-empty '+' diff lines" in result.output
    assert not (tmp_path / "TODO.md").exists()


def test_workspace_editor_rejects_create_for_existing_file(tmp_path) -> None:
    target = tmp_path / "TODO.md"
    target.write_text("existing\n", encoding="utf-8")
    editor = WorkspaceEditor(root=tmp_path, wandb_metrics_hook=None)
    operation = ApplyPatchOperation(type="create_file", path="TODO.md", diff="+new\n")

    result = editor.create_file(operation)

    assert result.output is not None
    assert "cannot overwrite existing file" in result.output
    assert target.read_text(encoding="utf-8") == "existing\n"


def test_read_only_litellm_shell_rejects_executable_command(tmp_path) -> None:
    shell_tool = make_litellm_shell_tool(
        cwd=tmp_path,
        cache_dir=tmp_path / "cache",
        git_snapshotter=None,
        wandb_metrics_hook=None,
        read_only=True,
    )

    result = asyncio.run(
        shell_tool.on_invoke_tool(
            RunContextWrapper(context=None),
            json.dumps({"command": "echo hi | ./db", "timeout_ms": 1000}),
        )
    )

    assert "Only read-only inspection commands are allowed." in result


def test_cpu_info_tool_returns_structured_summary(tmp_path) -> None:
    cpu_tool = make_cpu_info_tool(
        cwd=tmp_path,
        cache_dir=tmp_path / "cpu-cache",
        git_snapshotter=None,
        wandb_metrics_hook=None,
    )

    result = asyncio.run(
        cpu_tool.on_invoke_tool(
            RunContextWrapper(context=None),
            json.dumps({"timeout_ms": 1000}),
        )
    )
    payload = json.loads(result)

    assert "arch" in payload
    assert "flags" in payload
    assert "cache_summary" in payload
    assert "numa_summary" in payload
    assert "target_cpu_hint" in payload
    assert payload["target_cpu_hint"] in ("native", None)


def test_cpu_info_tool_omits_native_hint_without_hardware_probe_evidence(
    tmp_path, monkeypatch
) -> None:
    tool = CpuInfoTool(
        cwd=tmp_path,
        cache_dir=tmp_path / "cpu-cache",
        git_snapshotter=None,
        wandb_metrics_hook=None,
    )

    async def fake_run_probe(_command: str, _timeout_ms: int | None) -> dict[str, object]:
        if _command == "uname -m":
            return {
                "command": _command,
                "stdout": "x86_64\n",
                "stderr": "",
                "exit_code": 0,
                "timed_out": False,
            }
        return {
            "command": _command,
            "stdout": "",
            "stderr": "missing\n",
            "exit_code": 127,
            "timed_out": False,
        }

    monkeypatch.setattr(tool, "_run_probe", fake_run_probe)
    payload = json.loads(asyncio.run(tool(timeout_ms=1000)))

    assert payload["arch"] == "x86_64"
    assert payload["target_cpu_hint"] is None
    assert payload["vectorization_recommendation"] == "vectorization_support_unclear"


def test_cpu_info_tool_does_not_cache_environment_dependent_results(
    tmp_path, monkeypatch
) -> None:
    tool = CpuInfoTool(
        cwd=tmp_path,
        cache_dir=tmp_path / "cpu-cache",
        git_snapshotter=None,
        wandb_metrics_hook=None,
    )
    datasets = [
        {
            "uname -m": "x86_64\n",
            "lscpu": "Architecture: x86_64\nModel name: CPU-A\nFlags: avx2\n",
            "cat /proc/cpuinfo": "flags\t: avx2\n",
        },
        {
            "uname -m": "x86_64\n",
            "lscpu": "Architecture: x86_64\nModel name: CPU-B\nFlags: avx512f avx2\n",
            "cat /proc/cpuinfo": "flags\t: avx512f avx2\n",
        },
    ]
    state = {"invocation": -1}

    async def fake_run_probe(_command: str, _timeout_ms: int | None) -> dict[str, object]:
        if _command == "uname -m":
            state["invocation"] += 1
        current = datasets[state["invocation"]]
        return {
            "command": _command,
            "stdout": current[_command],
            "stderr": "",
            "exit_code": 0,
            "timed_out": False,
        }

    monkeypatch.setattr(tool, "_run_probe", fake_run_probe)
    first = json.loads(asyncio.run(tool(timeout_ms=1000)))
    second = json.loads(asyncio.run(tool(timeout_ms=1000)))

    assert first["model_name"] == "CPU-A"
    assert second["model_name"] == "CPU-B"
    assert first["vectorization_flags"] == ["avx2"]
    assert second["vectorization_flags"] == ["avx512f", "avx2"]


@pytest.mark.asyncio
async def test_observation_limit_warns_then_hits_hard_cap(tmp_path) -> None:
    async def on_invoke(_ctx: RunContextWrapper[object], _args_json: str) -> str:
        return "ok"

    run_tool = FunctionTool(
        name="run",
        description="run",
        params_json_schema={"type": "object", "properties": {}},
        on_invoke_tool=on_invoke,
    )

    bundle = build_tools(
        use_litellm=True,
        workspace_path=tmp_path,
        cache_path=tmp_path / "cache",
        shell_executor=None,
        workspace_editor=None,
        compile_cache_dir=tmp_path / "compile",
        run_tool_wrapper=run_tool,
        git_snapshotter=None,
        wandb_metrics_hook=None,
    )

    bundle.runtime.activate("finish_skeleton", 0, "finish_skeleton")
    list_tool = next(
        tool
        for tool in bundle.tools_by_profile["finish_skeleton"]
        if tool.name == "list_files"
    )
    args_json = json.dumps({"path": ".", "limit": 20})
    soft_limit = bundle.runtime.profiles["finish_skeleton"].max_consecutive_observations
    hard_limit = bundle.runtime.profiles["finish_skeleton"].hard_consecutive_observations
    assert hard_limit is not None

    for _ in range(soft_limit):
        result = await list_tool.on_invoke_tool(None, args_json)
        assert isinstance(result, str)

    result = await list_tool.on_invoke_tool(None, args_json)
    assert "consecutive observation calls" in result

    for _ in range(hard_limit - soft_limit - 1):
        result = await list_tool.on_invoke_tool(None, args_json)
        assert isinstance(result, str)

    with pytest.raises(RuntimeError, match="hard upper bound"):
        await list_tool.on_invoke_tool(None, args_json)


@pytest.mark.asyncio
async def test_apply_patch_success_counts_as_stage_write(tmp_path) -> None:
    target = tmp_path / "loader_impl.cpp"
    target.write_text("int value = 1;\n", encoding="utf-8")

    async def on_invoke(_ctx: RunContextWrapper[object], _args_json: str) -> str:
        return "ok"

    run_tool = FunctionTool(
        name="run",
        description="run",
        params_json_schema={"type": "object", "properties": {}},
        on_invoke_tool=on_invoke,
    )

    bundle = build_tools(
        use_litellm=True,
        workspace_path=tmp_path,
        cache_path=tmp_path / "cache",
        shell_executor=None,
        workspace_editor=None,
        compile_cache_dir=tmp_path / "compile",
        run_tool_wrapper=run_tool,
        git_snapshotter=None,
        wandb_metrics_hook=None,
    )

    bundle.runtime.activate("finish_skeleton", 0, "finish_skeleton")
    patch_tool = next(
        tool
        for tool in bundle.tools_by_profile["finish_skeleton"]
        if tool.name == "apply_patch"
    )

    result = await patch_tool.on_invoke_tool(
        RunContextWrapper(context=None),
        json.dumps(
            {
                "type": "update_file",
                "path": "loader_impl.cpp",
                "diff": "@@\n-int value = 1;\n+int value = 2;\n",
            }
        ),
    )

    summary = bundle.runtime.finish_stage(result)

    assert result == "Updated loader_impl.cpp"
    assert summary.has_writes is True
    assert summary.written_files == ("loader_impl.cpp",)
    assert target.read_text(encoding="utf-8") == "int value = 2;\n"


@pytest.mark.asyncio
async def test_failed_apply_patch_does_not_count_as_stage_write(tmp_path) -> None:
    target = tmp_path / "loader_impl.cpp"
    target.write_text("int value = 1;\n", encoding="utf-8")

    async def on_invoke(_ctx: RunContextWrapper[object], _args_json: str) -> str:
        return "ok"

    run_tool = FunctionTool(
        name="run",
        description="run",
        params_json_schema={"type": "object", "properties": {}},
        on_invoke_tool=on_invoke,
    )

    bundle = build_tools(
        use_litellm=True,
        workspace_path=tmp_path,
        cache_path=tmp_path / "cache",
        shell_executor=None,
        workspace_editor=None,
        compile_cache_dir=tmp_path / "compile",
        run_tool_wrapper=run_tool,
        git_snapshotter=None,
        wandb_metrics_hook=None,
    )

    bundle.runtime.activate("finish_skeleton", 0, "finish_skeleton")
    patch_tool = next(
        tool
        for tool in bundle.tools_by_profile["finish_skeleton"]
        if tool.name == "apply_patch"
    )

    result = await patch_tool.on_invoke_tool(
        RunContextWrapper(context=None),
        json.dumps(
            {
                "type": "update_file",
                "path": "loader_impl.cpp",
                "diff": "@@\n-int value = 1;\n+int value = 1;\n",
            }
        ),
    )

    summary = bundle.runtime.finish_stage(result)

    assert result.startswith("Error:")
    assert summary.has_writes is False
    assert summary.written_files == ()
    assert target.read_text(encoding="utf-8") == "int value = 1;\n"


@pytest.mark.asyncio
async def test_compile_failure_requires_real_write_tool_before_retry(tmp_path) -> None:
    compile_calls = {"count": 0}

    async def compile_invoke(_ctx: RunContextWrapper[object], _args_json: str) -> str:
        compile_calls["count"] += 1
        return "compile failed"

    async def run_invoke(_ctx: RunContextWrapper[object], _args_json: str) -> str:
        return "run failed"

    compile_tool = FunctionTool(
        name="compile",
        description="compile",
        params_json_schema={"type": "object", "properties": {}},
        on_invoke_tool=compile_invoke,
    )
    run_tool = FunctionTool(
        name="run",
        description="run",
        params_json_schema={"type": "object", "properties": {}},
        on_invoke_tool=run_invoke,
    )

    from tpch_monetdb.tools.tpch_monetdb_agent_tools import build_tpch_monetdb_agent_tools

    bundle = build_tpch_monetdb_agent_tools(
        workspace_path=tmp_path,
        cache_path=tmp_path / "cache",
        compile_tool=compile_tool,
        run_tool=run_tool,
        git_snapshotter=None,
        wandb_metrics_hook=None,
        apply_patch_tool=None,
    )
    bundle.runtime.activate("compile_fix", 0, "compile_fix")
    compile_wrapper = next(
        tool
        for tool in bundle.tools_by_profile["compile_fix"]
        if tool.name == "compile"
    )

    first = await compile_wrapper.on_invoke_tool(
        RunContextWrapper(context=None),
        json.dumps({}),
    )
    second = await compile_wrapper.on_invoke_tool(
        RunContextWrapper(context=None),
        json.dumps({}),
    )

    assert "[ERROR:COMPILE_FAILED]" in first
    assert "[Evidence]\ncompile failed" in first
    assert "[ERROR:MUST_WRITE_FIRST]" in second
    assert "Available write tools in this stage: edit_file, apply_patch" in second
    assert compile_calls["count"] == 1


def test_scripted_conversation_accepts_legacy_string_array(tmp_path) -> None:
    conv_path = tmp_path / "conv.json"
    conv_path.write_text(json.dumps(["prompt-1", "prompt-2"]), encoding="utf-8")

    conv = ScriptedConversation(
        conversation_json_path=conv_path,
        callback=lambda *_args: None,
        replay=False,
        notify=False,
        auto_finish=True,
        auto_u=True,
        replay_cache=False,
        model=None,
        workspace_root=tmp_path,
    )

    assert [step.text for step in conv.prompts] == ["prompt-1", "prompt-2"]
    assert conv.prompts[0].max_turns is None
    assert conv.prompts[0].tool_profile == "legacy_general"


@pytest.mark.asyncio
async def test_scripted_conversation_fail_fast_on_empty_required_file(tmp_path) -> None:
    conv_path = tmp_path / "conv.json"
    conv_path.write_text(
        json.dumps(
            [
                {
                    "text": "write todo",
                    "descriptor": "Stage 1",
                    "tool_profile": "todo_plan",
                    "required_nonempty_files": ["TODO.md"],
                }
            ]
        ),
        encoding="utf-8",
    )

    async def callback(_text, _descriptor, _idx, _max_turns, _metadata):
        (tmp_path / "TODO.md").write_text("", encoding="utf-8")
        return None

    conv = ScriptedConversation(
        conversation_json_path=conv_path,
        callback=callback,
        replay=False,
        notify=False,
        auto_finish=True,
        auto_u=True,
        replay_cache=False,
        model=None,
        workspace_root=tmp_path,
    )

    with pytest.raises(RuntimeError, match="required file TODO.md is empty"):
        await conv.run()


@pytest.mark.asyncio
async def test_scripted_conversation_retries_for_missing_required_primary_file(
    tmp_path,
) -> None:
    conv_path = tmp_path / "conv_missing_primary.json"
    conv_path.write_text(
        json.dumps(
            [
                {
                    "text": "implement query 12",
                    "descriptor": "implement_query_12",
                    "tool_profile": "implement_queries_writeonly",
                    "required_nonempty_files": ["query_q12.cpp"],
                }
            ]
        ),
        encoding="utf-8",
    )

    from tpch_monetdb.tools.stage_tool_policy import StageRunSummary

    calls: list[str] = []

    async def callback(_text, _descriptor, _idx, _max_turns, _metadata):
        calls.append(_text)
        if len(calls) == 1:
            (tmp_path / "query_q11.cpp").write_text("void execute_q12() {}\n", encoding="utf-8")
            return StageRunSummary(
                profile_name="implement_queries_writeonly",
                prompt_index=0,
                prompt_descriptor="implement_query_12",
                final_output="wrote wrong file",
                tool_counts={"write_file": 1},
                written_files=("query_q11.cpp",),
                last_compile_summary=None,
                last_run_summary=None,
                todo_before=None,
                todo_after=None,
            )
        (tmp_path / "query_q12.cpp").write_text("void execute_q12() {}\n", encoding="utf-8")
        return StageRunSummary(
            profile_name="implement_queries_writeonly",
            prompt_index=0,
            prompt_descriptor="implement_query_12",
            final_output="fixed file placement",
            tool_counts={"write_file": 1},
            written_files=("query_q12.cpp",),
            last_compile_summary=None,
            last_run_summary=None,
            todo_before=None,
            todo_after=None,
        )

    conv = ScriptedConversation(
        conversation_json_path=conv_path,
        callback=callback,
        replay=False,
        notify=False,
        auto_finish=True,
        auto_u=True,
        replay_cache=False,
        model=None,
        workspace_root=tmp_path,
    )

    used = await conv.run()

    assert used[0] == "implement query 12"
    assert "[FILE CONTRACT REMEDIATION]" in used[1]
    assert "required file query_q12.cpp does not exist" in used[1]
    assert (tmp_path / "query_q12.cpp").exists()
    assert len(calls) == 2


@pytest.mark.asyncio
async def test_scripted_conversation_retries_for_empty_required_primary_file(
    tmp_path,
) -> None:
    conv_path = tmp_path / "conv_empty_primary.json"
    conv_path.write_text(
        json.dumps(
            [
                {
                    "text": "implement query 12",
                    "descriptor": "implement_query_12",
                    "tool_profile": "implement_queries_writeonly",
                    "required_nonempty_files": ["query_q12.cpp"],
                }
            ]
        ),
        encoding="utf-8",
    )

    from tpch_monetdb.tools.stage_tool_policy import StageRunSummary

    calls = 0

    async def callback(_text, _descriptor, _idx, _max_turns, _metadata):
        nonlocal calls
        calls += 1
        if calls == 1:
            (tmp_path / "query_q12.cpp").write_text("", encoding="utf-8")
            return StageRunSummary(
                profile_name="implement_queries_writeonly",
                prompt_index=0,
                prompt_descriptor="implement_query_12",
                final_output="created empty file",
                tool_counts={"write_file": 1},
                written_files=("query_q12.cpp",),
                last_compile_summary=None,
                last_run_summary=None,
                todo_before=None,
                todo_after=None,
            )
        (tmp_path / "query_q12.cpp").write_text("void execute_q12() {}\n", encoding="utf-8")
        return StageRunSummary(
            profile_name="implement_queries_writeonly",
            prompt_index=0,
            prompt_descriptor="implement_query_12",
            final_output="filled file",
            tool_counts={"write_file": 1},
            written_files=("query_q12.cpp",),
            last_compile_summary=None,
            last_run_summary=None,
            todo_before=None,
            todo_after=None,
        )

    conv = ScriptedConversation(
        conversation_json_path=conv_path,
        callback=callback,
        replay=False,
        notify=False,
        auto_finish=True,
        auto_u=True,
        replay_cache=False,
        model=None,
        workspace_root=tmp_path,
    )

    used = await conv.run()

    assert "[FILE CONTRACT REMEDIATION]" in used[1]
    assert "required file query_q12.cpp is empty" in used[1]
    assert calls == 2


@pytest.mark.asyncio
async def test_scripted_conversation_retries_validation_after_latest_write(
    tmp_path,
) -> None:
    conv_path = tmp_path / "conv_validation_rerun.json"
    conv_path.write_text(
        json.dumps(
            [
                {
                    "text": "fix query 5",
                    "descriptor": "correctness_q5",
                    "tool_profile": "correctness_queries_writeonly",
                    "stop_conditions": ["validation_passed"],
                    "expected_query_id": "5",
                }
            ]
        ),
        encoding="utf-8",
    )

    calls: list[tuple[str, str | None]] = []

    async def callback(
        text: str,
        descriptor: str | None,
        _idx: int,
        _max_turns: int | None,
        _metadata: dict[str, object],
    ) -> StageRunSummary:
        calls.append((text, descriptor))
        if len(calls) == 1:
            return StageRunSummary(
                profile_name="correctness_queries_writeonly",
                prompt_index=0,
                prompt_descriptor="correctness_q5",
                final_output="edited after validation",
                tool_counts={"run": 1, "edit_file": 1},
                written_files=("query_q5.cpp",),
                last_compile_summary="compile ok",
                last_run_summary="run ok",
                todo_before=None,
                todo_after=None,
                last_validation_summary="Validation passed before edit",
                validation_passed=None,
                run_write_revision=0,
                write_revision=1,
            )
        return StageRunSummary(
            profile_name="correctness_queries_writeonly",
            prompt_index=0,
            prompt_descriptor="correctness_q5",
            final_output="validated",
            tool_counts={"run": 1},
            written_files=(),
            last_compile_summary="compile ok",
            last_run_summary="run ok",
            todo_before=None,
            todo_after=None,
            last_validation_summary="Validation passed",
            validation_passed=True,
            run_write_revision=0,
            write_revision=0,
        )

    conv = ScriptedConversation(
        conversation_json_path=conv_path,
        callback=callback,
        replay=False,
        notify=False,
        auto_finish=True,
        auto_u=True,
        replay_cache=False,
        model=None,
        workspace_root=tmp_path,
    )

    used = await conv.run()

    assert used[0] == "fix query 5"
    assert "[VALIDATION RERUN REQUIRED]" in used[1]
    assert "Expected query id: 5" in used[1]
    assert calls[1][1] == "correctness_q5__validation_rerun_remediation"
    assert len(calls) == 2
    return None


@pytest.mark.asyncio
async def test_scripted_conversation_fails_after_three_file_contract_remediations(
    tmp_path,
) -> None:
    conv_path = tmp_path / "conv_missing_primary_hard_fail.json"
    conv_path.write_text(
        json.dumps(
            [
                {
                    "text": "implement query 12",
                    "descriptor": "implement_query_12",
                    "tool_profile": "implement_queries_writeonly",
                    "required_nonempty_files": ["query_q12.cpp"],
                }
            ]
        ),
        encoding="utf-8",
    )

    from tpch_monetdb.tools.stage_tool_policy import StageRunSummary

    calls = 0

    async def callback(_text, _descriptor, _idx, _max_turns, _metadata):
        nonlocal calls
        calls += 1
        (tmp_path / "query_q11.cpp").write_text("void execute_q12() {}\n", encoding="utf-8")
        return StageRunSummary(
            profile_name="implement_queries_writeonly",
            prompt_index=0,
            prompt_descriptor="implement_query_12",
            final_output="still wrong file",
            tool_counts={"write_file": 1},
            written_files=("query_q11.cpp",),
            last_compile_summary=None,
            last_run_summary=None,
            todo_before=None,
            todo_after=None,
        )

    conv = ScriptedConversation(
        conversation_json_path=conv_path,
        callback=callback,
        replay=False,
        notify=False,
        auto_finish=True,
        auto_u=True,
        replay_cache=False,
        model=None,
        workspace_root=tmp_path,
    )

    with pytest.raises(RuntimeError, match="required file query_q12.cpp does not exist"):
        await conv.run()
    assert calls == 4


@pytest.mark.asyncio
async def test_scripted_conversation_allows_nonempty_required_file(tmp_path) -> None:
    conv_path = tmp_path / "conv.json"
    conv_path.write_text(
        json.dumps(
            [
                {
                    "text": "write todo",
                    "descriptor": "Stage 1",
                    "max_turns": 16,
                    "tool_profile": "todo_plan",
                    "required_nonempty_files": ["TODO.md"],
                    "required_updated_files": ["TODO.md"],
                }
            ]
        ),
        encoding="utf-8",
    )

    class Summary:
        written_files = ("TODO.md",)
        has_writes = True
        todo_progressed = False

    async def callback(_text, _descriptor, _idx, _max_turns, _metadata):
        (tmp_path / "TODO.md").write_text("- [ ] item\n", encoding="utf-8")
        from tpch_monetdb.tools.stage_tool_policy import StageRunSummary

        return StageRunSummary(
            profile_name="todo_plan",
            prompt_index=0,
            prompt_descriptor="Stage 1",
            final_output=None,
            tool_counts={"write_file": 1},
            written_files=("TODO.md",),
            last_compile_summary=None,
            last_run_summary=None,
            todo_before=None,
            todo_after=None,
        )

    conv = ScriptedConversation(
        conversation_json_path=conv_path,
        callback=callback,
        replay=False,
        notify=False,
        auto_finish=True,
        auto_u=True,
        replay_cache=False,
        model=None,
        workspace_root=tmp_path,
    )

    used = await conv.run()

    assert used == ["write todo"]


@pytest.mark.asyncio
async def test_scripted_conversation_enforces_write_required_stop_condition(
    tmp_path,
) -> None:
    conv_path = tmp_path / "conv.json"
    conv_path.write_text(
        json.dumps(
            [
                {
                    "text": "finish skeleton",
                    "descriptor": "finish_skeleton",
                    "tool_profile": "finish_skeleton",
                    "stop_conditions": ["write_required"],
                }
            ]
        ),
        encoding="utf-8",
    )

    from tpch_monetdb.tools.stage_tool_policy import StageRunSummary

    async def callback(_text, _descriptor, _idx, _max_turns, _metadata):
        return StageRunSummary(
            profile_name="finish_skeleton",
            prompt_index=0,
            prompt_descriptor="finish_skeleton",
            final_output=None,
            tool_counts={"read_file": 3},
            written_files=(),
            last_compile_summary=None,
            last_run_summary=None,
            todo_before=None,
            todo_after=None,
        )

    conv = ScriptedConversation(
        conversation_json_path=conv_path,
        callback=callback,
        replay=False,
        notify=False,
        auto_finish=True,
        auto_u=True,
        replay_cache=False,
        model=None,
        workspace_root=tmp_path,
    )

    with pytest.raises(RuntimeError, match="finished without any file write"):
        await conv.run()


@pytest.mark.asyncio
async def test_scripted_conversation_enforces_validation_passed_stop_condition(
    tmp_path,
) -> None:
    conv_path = tmp_path / "conv_validation.json"
    conv_path.write_text(
        json.dumps(
            [
                {
                    "text": "correctness q1",
                    "descriptor": "correctness_q1",
                    "tool_profile": "correctness_queries_writeonly",
                    "required_nonempty_files": ["query_q1.cpp"],
                    "stop_conditions": ["validation_passed"],
                }
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / "query_q1.cpp").write_text("namespace query_q1 {}\n", encoding="utf-8")

    from tpch_monetdb.tools.stage_tool_policy import StageRunSummary

    async def callback(_text, _descriptor, _idx, _max_turns, _metadata):
        return StageRunSummary(
            profile_name="correctness_queries_writeonly",
            prompt_index=0,
            prompt_descriptor="correctness_q1",
            final_output="still failing",
            tool_counts={"run": 1},
            written_files=(),
            last_compile_summary=None,
            last_run_summary="Validation failed:\nrow mismatch",
            todo_before=None,
            todo_after=None,
            last_validation_summary="Validation failed:\nrow mismatch",
            run_succeeded=False,
            validation_passed=False,
            last_failure_kind="validation",
        )

    conv = ScriptedConversation(
        conversation_json_path=conv_path,
        callback=callback,
        replay=False,
        notify=False,
        auto_finish=True,
        auto_u=True,
        replay_cache=False,
        model=None,
        workspace_root=tmp_path,
    )

    with pytest.raises(RuntimeError, match="validation did not pass"):
        await conv.run()


def test_scripted_conversation_reports_stale_validation_revision(tmp_path) -> None:
    from tpch_monetdb.conversations.scripted_conversation import PromptStep

    conv = ScriptedConversation(
        conversation_json_path=tmp_path / "conv_stale_validation.json",
        callback=lambda *_args: None,
        replay=False,
        notify=False,
        auto_finish=True,
        auto_u=True,
        replay_cache=False,
        model=None,
        workspace_root=tmp_path,
    )
    step = PromptStep(
        text="validate q5",
        descriptor="correctness_query_5",
        stop_conditions=("validation_passed",),
    )
    summary = StageRunSummary(
        profile_name="correctness_queries_writeonly",
        prompt_index=24,
        prompt_descriptor="correctness_query_5",
        final_output="validated then edited",
        tool_counts={"run": 4, "edit_file": 1},
        written_files=("query_q5.cpp",),
        last_compile_summary="**Compilation successfull**",
        last_run_summary="All queries passed validation!",
        todo_before=None,
        todo_after=None,
        last_validation_summary="All queries passed validation!",
        validation_passed=None,
        run_write_revision=3,
        write_revision=4,
    )

    with pytest.raises(RuntimeError) as exc_info:
        conv._validate_step_postconditions(
            step=step,
            idx=24,
            callback_result=summary,
        )

    message = str(exc_info.value)
    assert "validation was not rerun after the latest file write" in message
    assert "Recent writes: query_q5.cpp" in message
    assert "last_run=3, current_write=4" in message
    return None


def test_stage_tool_runtime_validation_state_recovers_after_success(tmp_path) -> None:
    from tpch_monetdb.tools.tpch_monetdb_agent_tools import StageToolRuntime

    runtime = StageToolRuntime(tmp_path)
    runtime.activate(
        "correctness_queries_writeonly",
        0,
        "correctness_q8",
    )

    failure_output = (
        "Validation failed for Q8\n"
        "Expected row count mismatch\n"
    )
    runtime.record_execution("run", failure_output, success=False)

    success_output = (
        "exit_code: 0 signal: 0\n"
        "All queries passed validation!\n"
    )
    runtime.record_execution("run", success_output, success=True)

    summary = runtime.finish_stage("Q8 fixed")

    assert summary.validation_passed is True
    assert summary.last_failure_kind is None
    assert summary.last_validation_summary == "All queries passed validation!"
    assert summary.run_succeeded is True


def test_stage_runtime_tracks_todo_state_in_memory(tmp_path) -> None:
    from tpch_monetdb.tools.tpch_monetdb_agent_tools import StageToolRuntime

    runtime = StageToolRuntime(tmp_path)
    (tmp_path / "TODO.md").write_text("- [ ] Build engine\n", encoding="utf-8")
    runtime.activate("todo_sync", 0, "todo_sync")
    result = runtime.write_file("TODO.md", "- [>] Build engine\n- [ ] Run query\n")
    summary = runtime.finish_stage(result)

    assert summary.todo_before is not None
    assert summary.todo_after is not None
    assert summary.todo_after.in_progress_count == 1
    assert summary.todo_after.pending_count == 1
    assert summary.todo_after.source_text == "- [>] Build engine\n- [ ] Run query\n"


def test_stage_runtime_requires_required_control_artifacts_before_write(tmp_path) -> None:
    from tpch_monetdb.tools.tpch_monetdb_agent_tools import StageToolRuntime

    (tmp_path / "TODO.md").write_text("- [ ] Build\n", encoding="utf-8")
    (tmp_path / "query_q8.cpp").write_text("seed\n", encoding="utf-8")
    runtime = StageToolRuntime(tmp_path)
    runtime.activate(
        "implement_queries_writeonly",
        0,
        "implement_q8",
        prompt_metadata={"required_control_artifacts": ["TODO.md"]},
    )

    with pytest.raises(PipelineContractError, match="CONTROL_ARTIFACT_NOT_ACKNOWLEDGED"):
        runtime.write_file("query_q8.cpp", "rewritten\n")

    runtime.read_file("TODO.md", None, None)
    assert runtime.write_file("query_q8.cpp", "rewritten\n") == "Updated query_q8.cpp"


def test_stage_runtime_normalizes_control_artifact_read_paths(tmp_path: Path) -> None:
    """read_file 等价相对路径应计入 required control artifact 读取状态."""
    from tpch_monetdb.tools.tpch_monetdb_agent_tools import StageToolRuntime

    (tmp_path / "TODO.md").write_text("- [ ] Build\n", encoding="utf-8")
    (tmp_path / "query_q8.cpp").write_text("seed\n", encoding="utf-8")
    runtime = StageToolRuntime(tmp_path)
    runtime.activate(
        "implement_queries_writeonly",
        0,
        "implement_q8",
        prompt_metadata={"required_control_artifacts": ["TODO.md"]},
    )

    runtime.read_file("./TODO.md", None, None)

    assert runtime.write_file("query_q8.cpp", "rewritten\n") == "Updated query_q8.cpp"


def test_storage_plan_profile_can_write_required_contract(tmp_path) -> None:
    """storage_plan 阶段必须能写出编排要求的两个产物。"""
    from tpch_monetdb.tools.tpch_monetdb_agent_tools import StageToolRuntime

    for artifact_name in (
        "workload_objective.json",
        "data_law_contract.json",
        "design_evidence.md",
    ):
        (tmp_path / artifact_name).write_text("{}\n", encoding="utf-8")

    runtime = StageToolRuntime(tmp_path)
    runtime.activate(
        "storage_plan",
        0,
        "storage_plan",
        prompt_metadata={
            "required_control_artifacts": [
                "workload_objective.json",
                "data_law_contract.json",
                "design_evidence.md",
            ],
            "control_artifacts_injected": [
                "workload_objective.json",
                "data_law_contract.json",
                "design_evidence.md",
            ],
        },
    )

    assert runtime.write_file("storage_plan.txt", "plan\n") == "Created storage_plan.txt"
    assert (
        runtime.write_file("storage_plan_contract.json", '{"version": 1}\n')
        == "Created storage_plan_contract.json"
    )
    return None


def test_stage_runtime_records_all_tracked_control_artifact_reads(tmp_path) -> None:
    """所有统一清单里的控制产物读操作都应满足后续写入确认 gate。"""
    from tpch_monetdb.tools.tpch_monetdb_agent_tools import StageToolRuntime

    for artifact_name in (
        "workload_objective.json",
        "data_law_contract.json",
        "query_q8.cpp",
    ):
        (tmp_path / artifact_name).write_text("{}\n", encoding="utf-8")

    runtime = StageToolRuntime(tmp_path)
    runtime.activate(
        "implement_queries_writeonly",
        0,
        "implement_q8",
        prompt_metadata={
            "required_control_artifacts": [
                "workload_objective.json",
                "data_law_contract.json",
            ],
        },
    )

    runtime.read_file("workload_objective.json", None, None)
    runtime.read_file("data_law_contract.json", None, None)
    assert runtime.write_file("query_q8.cpp", "rewritten\n") == "Updated query_q8.cpp"
    return None


def test_stage_runtime_records_optimization_hotspot_summary_reads(tmp_path) -> None:
    """优化热点摘要作为控制性产物读取后应满足后续 TODO 写入 gate。"""
    from tpch_monetdb.tools.tpch_monetdb_agent_tools import StageToolRuntime

    for artifact_name in ("TODO.md", "optimization_hotspot_summary.md"):
        (tmp_path / artifact_name).write_text("seed\n", encoding="utf-8")

    runtime = StageToolRuntime(tmp_path)
    runtime.activate(
        "optimization_todo_sync",
        0,
        "optimization_todo_sync",
        prompt_metadata={
            "required_control_artifacts": ["optimization_hotspot_summary.md"],
        },
    )

    runtime.read_file("optimization_hotspot_summary.md", None, None)
    assert runtime.write_file("TODO.md", "- [x] synced\n") == "Updated TODO.md"
    return None


def test_stage_runtime_rejected_large_control_read_does_not_acknowledge(
    tmp_path,
) -> None:
    """大控制产物 full-read 被拒绝时，不应被当作已消费证据。"""
    from tpch_monetdb.tools.tpch_monetdb_agent_tools import StageToolRuntime

    (tmp_path / "TODO.md").write_text("- [ ] Build\n", encoding="utf-8")
    (tmp_path / "optimization_hotspot_summary.md").write_text(
        "line\n" * 250_001,
        encoding="utf-8",
    )
    runtime = StageToolRuntime(tmp_path)
    runtime.activate(
        "optimization_todo_sync",
        0,
        "optimization_todo_sync",
        prompt_metadata={
            "required_control_artifacts": ["optimization_hotspot_summary.md"],
        },
    )

    full_result = runtime.read_file("optimization_hotspot_summary.md", None, None)
    assert "exceeds read_file full-read limit" in full_result
    with pytest.raises(PipelineContractError, match="CONTROL_ARTIFACT_NOT_ACKNOWLEDGED"):
        runtime.write_file("TODO.md", "- [x] synced\n")

    runtime.read_file("optimization_hotspot_summary.md", 1, 1)
    assert runtime.write_file("TODO.md", "- [x] synced\n") == "Updated TODO.md"
    return None


def test_stage_runtime_accepts_injected_control_artifacts_for_run_scope(tmp_path) -> None:
    from tpch_monetdb.tools.tpch_monetdb_agent_tools import StageToolRuntime

    (tmp_path / "TODO.md").write_text("- [ ] Build\n", encoding="utf-8")
    runtime = StageToolRuntime(tmp_path)
    runtime.activate(
        "optimization_instrumentation",
        0,
        "trace_expert",
        prompt_metadata={
            "required_control_artifacts": ["TODO.md"],
            "control_artifacts_injected": ["TODO.md"],
            "active_query_ids": ["3", "4"],
            "active_unit_query_ids": ["3", "4"],
        },
    )

    runtime.validate_run_request(
        json.dumps({"scale_factor": 1, "optimize": False, "query_id": ["3", "4"]})
    )


def test_stage_runtime_carries_active_unit_metadata_into_stage_summary(tmp_path) -> None:
    from tpch_monetdb.tools.tpch_monetdb_agent_tools import StageToolRuntime

    (tmp_path / "TODO.md").write_text("- [ ] Build\n", encoding="utf-8")
    runtime = StageToolRuntime(tmp_path)
    runtime.activate(
        "implement_queries_writeonly",
        1,
        "implement_query_3",
        prompt_metadata={
            "required_control_artifacts": ["TODO.md"],
            "control_artifacts_injected": ["TODO.md"],
            "active_unit_id": "query:3",
            "active_unit_kind": "query",
            "active_unit_files": ["query_q3.cpp", "query_q3.hpp"],
            "active_unit_query_ids": ["3"],
        },
    )
    summary = runtime.finish_stage("done")
    assert summary.active_unit_id == "query:3"
    assert summary.active_unit_kind == "query"
    assert "query_q3.cpp" in summary.active_unit_files
    assert summary.active_unit_query_ids == ("3",)


def test_run_stage_correctness_gate_uses_full_scale_for_family_units(monkeypatch) -> None:
    from tpch_monetdb.conversations.optimization_conversation_tpch_monetdb import (
        OptimizationValidationPolicy,
        TpchMonetdbOptimizationConversation,
    )
    from tpch_monetdb.conversations.optimization_validation import CorrectnessCheckSummary

    conversation = TpchMonetdbOptimizationConversation.__new__(TpchMonetdbOptimizationConversation)
    conversation.query_ids = ["3", "4", "5", "6", "7"]
    conversation.run_tool = SimpleNamespace()

    captured: dict[str, object] = {}

    def fake_run_required_correctness_checks(
        run_tool,
        scale_factors,
        query_ids,
        **kwargs,
    ):
        captured["scale_factors"] = tuple(scale_factors)
        captured["query_ids"] = list(query_ids)
        return CorrectnessCheckSummary(
            success=True,
            message="ok",
            metrics={"validation/correct": True},
            failed_scale_factor=None,
        )

    monkeypatch.setattr(
        "tpch_monetdb.conversations.optimization_conversation_tpch_monetdb.run_required_correctness_checks",
        fake_run_required_correctness_checks,
    )

    policy = OptimizationValidationPolicy(
        light_scale_factors=(1,),
        full_scale_factors=(1, 10, 100),
        heavyweight_scale_factors=(),
    )

    summary = conversation._run_stage_correctness_gate(
        query_id="3",
        written_files=("query_q3.cpp",),
        policy=policy,
        scope_query_ids=("3", "4", "5", "6", "7"),
    )

    assert summary.success is True
    assert captured["scale_factors"] == (1, 10, 100)
    assert captured["query_ids"] == ["3", "4", "5", "6", "7"]


def test_build_unit_validation_plan_and_rollback_policy() -> None:
    from tpch_monetdb.conversations.optimization_validation import (
        build_unit_validation_plan,
        should_rollback_unit_regression,
    )

    plan = build_unit_validation_plan(
        query_id="3",
        scope_query_ids=("3", "4", "5", "6", "7"),
        written_files=("query_q3.cpp",),
        all_query_ids=("3", "4", "5", "6", "7"),
        light_scale_factors=(1,),
        full_scale_factors=(1, 10, 100),
    )
    assert plan.scope_query_ids == ("3", "4", "5", "6", "7")
    assert plan.scale_factors == (1, 10, 100)
    assert should_rollback_unit_regression(
        rt_before_s=0.10,
        rt_after_s=0.12,
        revert_on_regression=True,
    ) is True


def _write_global_human_reference_control_artifacts(tmp_path: Path) -> None:
    """Write the control artifacts required by the global human-reference stage."""
    (tmp_path / "optimization_hotspot_summary.md").write_text(
        "# Optimization Hotspot Summary\n",
        encoding="utf-8",
    )
    (tmp_path / "TODO.md").write_text("- [x] done\n", encoding="utf-8")
    (tmp_path / "storage_plan.txt").write_text("storage plan\n", encoding="utf-8")
    (tmp_path / "workload_objective.json").write_text(
        json.dumps({
            "version": 1,
            "objective_id": "tpch-large-data-objective-v1",
            "query_ids": ["1"],
            "critical_query_ids": [],
            "required_artifacts": [
                "workload_objective.json",
                "data_law_contract.json",
                "storage_plan_contract.json",
            ],
        }),
        encoding="utf-8",
    )
    (tmp_path / "data_law_contract.json").write_text(
        json.dumps({"laws": [{"law_id": "LAW_ROW_SCALING"}]}),
        encoding="utf-8",
    )
    (tmp_path / "storage_plan_contract.json").write_text(
        json.dumps({
            "version": 1,
            "candidate_layouts": [
                {
                    "id": "conservative-row-major",
                    "layout_kind": "conservative",
                    "data_law_ids": ["LAW_ROW_SCALING"],
                    "evidence_refs": ["design_evidence.md#ILP Data Profile"],
                    "query_family_fit": "baseline",
                    "correctness_risk": "low",
                    "build_ingest_complexity": "low",
                    "vectorization_readiness": "medium",
                    "memory_locality": "medium",
                },
                {
                    "id": "hybrid-host-blocks",
                    "layout_kind": "hybrid",
                    "data_law_ids": ["LAW_ROW_SCALING"],
                    "evidence_refs": ["design_evidence.md#Layout Decision Signals"],
                    "query_family_fit": "host queries",
                    "correctness_risk": "medium",
                    "build_ingest_complexity": "medium",
                    "vectorization_readiness": "high",
                    "memory_locality": "high",
                },
                {
                    "id": "aggressive-columnar-sidecars",
                    "layout_kind": "aggressive",
                    "data_law_ids": ["LAW_ROW_SCALING"],
                    "evidence_refs": ["design_evidence.md#Layout Structure Evidence"],
                    "query_family_fit": "global scans",
                    "correctness_risk": "high",
                    "build_ingest_complexity": "high",
                    "vectorization_readiness": "high",
                    "memory_locality": "high",
                },
            ],
            "selected_layout": {
                "candidate_id": "hybrid-host-blocks",
                "data_law_ids": ["LAW_ROW_SCALING"],
                "selection_rationale": "Best test fixture balance.",
            },
            "selected_layout_obligations": [
                {
                    "id": "obl1",
                    "file_scope": ["builder_impl.hpp", "query_q1.cpp"],
                    "query_ids": ["1"],
                    "evidence_refs": ["design_evidence.md#ILP Data Profile"],
                }
            ],
            "query_family_costs": {"1": {}},
        }),
        encoding="utf-8",
    )
    return None


def _make_global_human_reference_conversation(tmp_path: Path) -> TpchMonetdbOptimizationConversation:
    """Build a minimal conversation object for global human-reference tests."""
    conversation = TpchMonetdbOptimizationConversation.__new__(TpchMonetdbOptimizationConversation)
    conversation.benchmark_sf = 10
    conversation.bespoke_storage = True
    conversation.query_ids = ["1"]
    conversation.required_validation_sf_list = [1]
    conversation.regression_tolerance = 0.05
    conversation.measurement_repetition = {}
    conversation.hardware_counter_summary_by_query = {}
    conversation.compiler_vectorization_summary = {}
    conversation.run_tool = SimpleNamespace(cwd=tmp_path, run=lambda **_kwargs: ("ok", {}))
    conversation._get_baseline_runtime_ms_by_query = lambda: {"1": 100.0}
    conversation.measurement_records = [
        QueryMeasurementRecord(
            query_id="1",
            engine="generated_tpch",
            measurement_kind=MeasurementKind.EXACT_INSTANTIATION,
            runtime_ms=80.0,
            measurement_shape_status=MeasurementShapeStatus.UNKNOWN,
            provenance={"runtime_metric_kind": KERNEL_RUNTIME_METRIC_KIND},
        ).to_dict()
    ]
    return conversation


def _global_diagnosis_lines(hypothesis_ids: list[str]) -> str:
    """Build compact JSON-lines hypotheses for autonomous global tests."""
    lines = []
    for hypothesis_id in hypothesis_ids:
        lines.append(json.dumps({
            "id": hypothesis_id,
            "summary": f"candidate {hypothesis_id}",
            "evidence": ["optimization_hotspot_summary.md#q1"],
            "affected_queries": ["1"],
            "suspected_runtime_path": ["query"],
            "expected_mechanism": "reduce Q1 work",
            "expected_impact": {"1": 0.2},
            "correctness_risk": "low",
            "implementation_scope": ["query_q1.cpp"],
            "verification_plan": ["validate Q1"],
            "evidence_gap": False,
        }))
    return "\n".join(lines)


def test_parse_global_optimization_hypotheses_json_lines() -> None:
    """diagnosis stage 输出的 JSON lines 应解析为结构化 hypotheses。"""
    hypotheses = parse_global_optimization_hypotheses(
        _global_diagnosis_lines(["h_001", "h_002"])
    )

    assert [hypothesis.id for hypothesis in hypotheses] == ["h_001", "h_002"]
    assert hypotheses[0].affected_queries == ("1",)
    assert hypotheses[0].implementation_scope == ("query_q1.cpp",)
    return None


def test_select_global_winner_uses_candidate_pool_not_first_accept() -> None:
    """winner selection 应比较候选池，而不是 first accepted stop。"""
    h1 = GlobalOptimizationHypothesis(
        id="h_001",
        summary="small win",
        evidence=("trace",),
        affected_queries=("1",),
    )
    h2 = GlobalOptimizationHypothesis(
        id="h_002",
        summary="large win",
        evidence=("trace",),
        affected_queries=("1",),
    )
    winner = select_global_winner(
        (
            GlobalOptimizationCandidate(
                hypothesis=h1,
                snapshot_hash="s1",
                accepted=True,
                runtime_by_query={"1": 0.09},
            ),
            GlobalOptimizationCandidate(
                hypothesis=h2,
                snapshot_hash="s2",
                accepted=True,
                runtime_by_query={"1": 0.05},
            ),
        ),
        {"1": 100.0},
    )

    assert winner is not None
    assert winner.hypothesis.id == "h_002"
    return None


def test_select_global_winner_uses_affected_query_scope_over_aggregate() -> None:
    """winner selection 应先比较 hypothesis affected queries，而不是全量 aggregate。"""
    h1 = GlobalOptimizationHypothesis(
        id="h_critical",
        summary="critical win",
        evidence=("trace",),
        affected_queries=("1",),
    )
    h2 = GlobalOptimizationHypothesis(
        id="h_aggregate",
        summary="aggregate win",
        evidence=("trace",),
        affected_queries=("2",),
    )
    winner = select_global_winner(
        (
            GlobalOptimizationCandidate(
                hypothesis=h1,
                snapshot_hash="s1",
                accepted=True,
                runtime_by_query={"1": 0.05, "2": 1.00},
            ),
            GlobalOptimizationCandidate(
                hypothesis=h2,
                snapshot_hash="s2",
                accepted=True,
                runtime_by_query={"1": 0.01, "2": 0.09},
            ),
        ),
        {"1": 100.0, "2": 100.0},
    )

    assert winner is not None
    assert winner.hypothesis.id == "h_critical"
    return None


@pytest.mark.asyncio
async def test_global_human_reference_retries_after_regression(
    monkeypatch,
    tmp_path,
) -> None:
    """global human-reference 应在回归候选被拒绝后换方向继续尝试。"""
    from tpch_monetdb.conversations.optimization_validation import CorrectnessCheckSummary
    from tpch_monetdb.tools.stage_tool_policy import StageRunSummary

    _write_global_human_reference_control_artifacts(tmp_path)
    conversation = _make_global_human_reference_conversation(tmp_path)
    restore_calls: list[str] = []
    snapshot_calls: list[str] = []
    conversation.git_snapshotter = SimpleNamespace(
        current_hash="base",
        restore=lambda snapshot: restore_calls.append(snapshot),
        snapshot=lambda name: snapshot_calls.append(name) or ("base", f"{name}-hash"),
    )
    measure_results = iter([
        {"1": 0.20},
        {"1": 0.08},
    ])
    conversation._measure_all_queries = lambda: next(measure_results)

    exec_metadata: list[dict[str, object]] = []

    async def fake_exec(*_args, **kwargs):
        exec_metadata.append(kwargs["prompt_metadata"])
        if len(exec_metadata) == 1:
            return StageRunSummary(
                profile_name="optimization_control",
                prompt_index=1,
                prompt_descriptor="diagnosis",
                final_output=_global_diagnosis_lines(["h_001", "h_002"]),
                tool_counts={"read_file": 1},
                written_files=(),
                last_compile_summary=None,
                last_run_summary=None,
                todo_before=None,
                todo_after=None,
                control_artifacts_read=GLOBAL_HUMAN_REFERENCE_REQUIRED_CONTROL_ARTIFACTS,
            )
        return StageRunSummary(
            profile_name="optimization_general",
            prompt_index=len(exec_metadata),
            prompt_descriptor="global",
            final_output="done",
            tool_counts={"edit_file": 1},
            written_files=(f"query_q1_attempt_{len(exec_metadata)}.cpp",),
            last_compile_summary=None,
            last_run_summary=None,
            todo_before=None,
            todo_after=None,
            control_artifacts_read=GLOBAL_HUMAN_REFERENCE_REQUIRED_CONTROL_ARTIFACTS,
        )

    conversation._exec = fake_exec
    monkeypatch.setattr(
        "tpch_monetdb.conversations.optimization_conversation_tpch_monetdb.run_required_correctness_checks",
        lambda *args, **kwargs: CorrectnessCheckSummary(
            success=True,
            message="ok",
            metrics={"validation/correct": True},
            failed_scale_factor=None,
        ),
    )

    result = await conversation._run_global_human_reference(
        mandatory_constraints="constraints",
        hotspot_summary_path=tmp_path / "optimization_hotspot_summary.md",
        before_rt_log={"1": 0.10},
    )

    assert result.accepted is True
    assert result.runtime_by_query == {"1": 0.08}
    assert [attempt.accepted for attempt in result.attempts] == [False, True]
    assert result.attempts[0].rejection_code == "GLOBAL_REGRESSION"
    assert result.attempts[0].regressed_queries == ("1",)
    assert result.winner is not None
    assert result.winner.hypothesis.id == "h_002"
    assert restore_calls.count("base") >= 2
    assert restore_calls[-1] == "global_hypothesis_h_002_2-hash"
    assert snapshot_calls == [
        "global_rejected_h_001_1",
        "global_hypothesis_h_002_2",
    ]
    assert exec_metadata[0]["required_control_artifacts"] == list(
        GLOBAL_HUMAN_REFERENCE_REQUIRED_CONTROL_ARTIFACTS
    )
    assert exec_metadata[0]["patch_scope_verdict"] == "global_diagnosis"
    assert exec_metadata[1]["patch_scope_verdict"] == "global_hypothesis"
    return None


@pytest.mark.asyncio
async def test_global_human_reference_records_all_rejected_attempts(
    monkeypatch,
    tmp_path,
) -> None:
    """全部 global 候选被拒绝时应保留 local runtime 并记录尝试账本。"""
    from tpch_monetdb.conversations.optimization_validation import CorrectnessCheckSummary
    from tpch_monetdb.tools.stage_tool_policy import StageRunSummary

    _write_global_human_reference_control_artifacts(tmp_path)
    conversation = _make_global_human_reference_conversation(tmp_path)
    restore_calls: list[str] = []
    conversation.git_snapshotter = SimpleNamespace(
        current_hash="base",
        restore=lambda snapshot: restore_calls.append(snapshot),
        snapshot=lambda name: ("base", f"{name}-hash"),
    )
    conversation._measure_all_queries = lambda: {"1": 0.20}

    async def fake_exec(*_args, **_kwargs):
        prompt_index = len(getattr(fake_exec, "calls", [])) + 1
        fake_exec.calls = [*getattr(fake_exec, "calls", []), prompt_index]
        if prompt_index == 1:
            return StageRunSummary(
                profile_name="optimization_control",
                prompt_index=1,
                prompt_descriptor="diagnosis",
                final_output=_global_diagnosis_lines(["h_001", "h_002", "h_003"]),
                tool_counts={"read_file": 1},
                written_files=(),
                last_compile_summary=None,
                last_run_summary=None,
                todo_before=None,
                todo_after=None,
                control_artifacts_read=GLOBAL_HUMAN_REFERENCE_REQUIRED_CONTROL_ARTIFACTS,
            )
        return StageRunSummary(
            profile_name="optimization_general",
            prompt_index=prompt_index,
            prompt_descriptor="global",
            final_output="done",
            tool_counts={"edit_file": 1},
            written_files=("query_q1.cpp",),
            last_compile_summary=None,
            last_run_summary=None,
            todo_before=None,
            todo_after=None,
            control_artifacts_read=GLOBAL_HUMAN_REFERENCE_REQUIRED_CONTROL_ARTIFACTS,
        )

    conversation._exec = fake_exec
    monkeypatch.setattr(
        "tpch_monetdb.conversations.optimization_conversation_tpch_monetdb.run_required_correctness_checks",
        lambda *args, **kwargs: CorrectnessCheckSummary(
            success=True,
            message="ok",
            metrics={"validation/correct": True},
            failed_scale_factor=None,
        ),
    )

    result = await conversation._run_global_human_reference(
        mandatory_constraints="constraints",
        hotspot_summary_path=tmp_path / "optimization_hotspot_summary.md",
        before_rt_log={"1": 0.10},
    )

    assert result.accepted is False
    assert result.runtime_by_query == {"1": 0.10}
    assert len(result.attempts) == 3
    assert all(attempt.rejection_code == "GLOBAL_REGRESSION" for attempt in result.attempts)
    assert len(result.candidates) == 3
    assert restore_calls.count("base") >= 3
    return None


@pytest.mark.asyncio
async def test_global_human_reference_rejects_missing_structured_causality(
    monkeypatch,
    tmp_path,
) -> None:
    """global candidate 缺少 official measurement/causal runtime improvement 时必须拒绝。"""
    from tpch_monetdb.conversations.optimization_validation import CorrectnessCheckSummary
    from tpch_monetdb.tools.stage_tool_policy import StageRunSummary

    _write_global_human_reference_control_artifacts(tmp_path)
    conversation = _make_global_human_reference_conversation(tmp_path)
    conversation.measurement_records = []
    restore_calls: list[str] = []
    snapshot_calls: list[str] = []
    conversation.git_snapshotter = SimpleNamespace(
        current_hash="base",
        restore=lambda snapshot: restore_calls.append(snapshot),
        snapshot=lambda name: snapshot_calls.append(name) or ("base", f"{name}-hash"),
    )
    conversation._measure_all_queries = lambda: {"1": 0.10}

    async def fake_exec(*_args, **_kwargs):
        prompt_index = len(getattr(fake_exec, "calls", [])) + 1
        fake_exec.calls = [*getattr(fake_exec, "calls", []), prompt_index]
        if prompt_index == 1:
            return StageRunSummary(
                profile_name="optimization_control",
                prompt_index=1,
                prompt_descriptor="diagnosis",
                final_output=_global_diagnosis_lines(["h_001"]),
                tool_counts={"read_file": 1},
                written_files=(),
                last_compile_summary=None,
                last_run_summary=None,
                todo_before=None,
                todo_after=None,
                control_artifacts_read=GLOBAL_HUMAN_REFERENCE_REQUIRED_CONTROL_ARTIFACTS,
            )
        return StageRunSummary(
            profile_name="optimization_general",
            prompt_index=prompt_index,
            prompt_descriptor="global",
            final_output="done",
            tool_counts={"edit_file": 1},
            written_files=("query_q1.cpp",),
            last_compile_summary=None,
            last_run_summary=None,
            todo_before=None,
            todo_after=None,
            control_artifacts_read=GLOBAL_HUMAN_REFERENCE_REQUIRED_CONTROL_ARTIFACTS,
        )

    conversation._exec = fake_exec
    monkeypatch.setattr(
        "tpch_monetdb.conversations.optimization_conversation_tpch_monetdb.run_required_correctness_checks",
        lambda *args, **kwargs: CorrectnessCheckSummary(
            success=True,
            message="ok",
            metrics={"validation/correct": True},
            failed_scale_factor=None,
        ),
    )

    result = await conversation._run_global_human_reference(
        mandatory_constraints="constraints",
        hotspot_summary_path=tmp_path / "optimization_hotspot_summary.md",
        before_rt_log={"1": 0.10},
    )

    assert result.accepted is False
    assert result.candidates[0].rejection_codes == (
        "OPTIMIZATION_RUNTIME_MISSING",
        "CAUSALITY_EVIDENCE_MISSING",
    )
    assert result.candidates[0].measurement_gaps == ("OPTIMIZATION_RUNTIME_MISSING",)
    assert restore_calls[-1] == "base"
    assert snapshot_calls == ["global_rejected_h_001_1"]
    return None


def test_record_global_regression_keeps_rejected_attempts_from_accepted_result() -> None:
    """accepted global result 也应保留之前被拒绝的候选尝试。"""
    from tpch_monetdb.conversations.optimization_conversation_tpch_monetdb import (
        GlobalHumanReferenceAttempt,
        TpchMonetdbOptimizationConversation,
    )

    conversation = TpchMonetdbOptimizationConversation.__new__(TpchMonetdbOptimizationConversation)
    conversation.global_regression_records = []

    conversation._record_global_regression(
        GlobalHumanReferenceResult(
            runtime_by_query={"1": 0.08},
            written_files=("query_q1.cpp",),
            accepted=True,
            attempts=(
                GlobalHumanReferenceAttempt(
                    attempt_index=1,
                    written_files=("query_q1.cpp",),
                    accepted=False,
                    rejection_code="GLOBAL_REGRESSION",
                    rejection_detail="regressed",
                    regressed_queries=("1",),
                ),
                GlobalHumanReferenceAttempt(
                    attempt_index=2,
                    written_files=("query_q1.cpp",),
                    accepted=True,
                ),
            ),
        )
    )

    assert conversation.global_regression_records == [
        {
            "stage_name": "global_human_reference",
            "attempt_index": 1,
            "accepted": False,
            "rejection_code": "GLOBAL_REGRESSION",
            "regressed_queries": ["1"],
            "objective_failures": [],
            "failure_detail": "regressed",
        }
    ]
    return None


def test_correctness_runtime_requires_single_file_rebuild_after_corrupt_compile(
    tmp_path,
) -> None:
    from tpch_monetdb.tools.tpch_monetdb_agent_tools import (
        RecoverableStagePolicyError,
        StageToolRuntime,
    )

    runtime = StageToolRuntime(tmp_path)
    (tmp_path / "query_q2.cpp").write_text("clean seed\n", encoding="utf-8")
    (tmp_path / "query_shared_utils.cpp").write_text("helper seed\n", encoding="utf-8")
    runtime.activate("correctness_queries_writeonly", 0, "correctness_q2")

    runtime.record_execution(
        "compile",
        "query_q2.cpp:18:5: error: expected unqualified-id",
        success=False,
    )
    runtime.record_execution(
        "compile",
        "query_q2.cpp:36:5: error: extraneous closing brace ('}')",
        success=False,
    )
    runtime.record_execution(
        "compile",
        "query_q2.cpp:70:13: error: redefinition of 'execute_q2'",
        success=False,
    )

    assert "Single-file rebuild mode" in runtime.generate_stage_hint()

    with pytest.raises(RecoverableStagePolicyError, match="query_q2.cpp"):
        runtime.edit_file("query_q2.cpp", "clean", "fresh", False)

    with pytest.raises(RecoverableStagePolicyError, match="query_q2.cpp"):
        runtime.validate_apply_patch(
            json.dumps(
                {
                    "type": "update_file",
                    "path": "query_q2.cpp",
                    "diff": "@@\n-clean seed\n+fresh seed",
                }
            )
        )

    with pytest.raises(RecoverableStagePolicyError, match="query_q2.cpp"):
        runtime.write_file("query_shared_utils.cpp", "rewritten helper\n")

    result = runtime.write_file("query_q2.cpp", "rebuilt query body\n",)
    assert result == "Updated query_q2.cpp"
    assert "Single-file rebuild mode" in runtime.generate_stage_hint()

    with pytest.raises(RecoverableStagePolicyError, match="query_q2.cpp"):
        runtime.edit_file("query_q2.cpp", "rebuilt", "fixed", False)

    runtime.record_execution("compile", "**Compilation successfull**", success=True)
    assert "Single-file rebuild mode" not in runtime.generate_stage_hint()
    assert runtime.edit_file("query_q2.cpp", "rebuilt", "fixed", False).startswith("Updated")


def test_implement_queries_writeonly_allows_focused_write_file(tmp_path) -> None:
    from tpch_monetdb.tools.tpch_monetdb_agent_tools import RecoverableStagePolicyError, StageToolRuntime

    runtime = StageToolRuntime(tmp_path)
    runtime.activate("implement_queries_writeonly", 0, "implement_q8")

    result = runtime.write_file("query_q8.cpp", "void execute_q8() {}\n")

    assert result == "Created query_q8.cpp"
    assert (tmp_path / "query_q8.cpp").read_text(encoding="utf-8") == "void execute_q8() {}\n"

    with pytest.raises(RecoverableStagePolicyError, match="builder_impl.cpp"):
        runtime.write_file("builder_impl.cpp", "bad\n")

    return None


@pytest.mark.parametrize(
    "profile_name",
    ["correctness_queries_writeonly", "correctness_foundation"],
)
def test_correctness_runtime_restricts_run_to_primary_query_only(
    tmp_path: Path,
    profile_name: str,
) -> None:
    from tpch_monetdb.tools.tpch_monetdb_agent_tools import (
        RecoverableStagePolicyError,
        StageToolRuntime,
    )

    runtime = StageToolRuntime(tmp_path)
    runtime.activate(profile_name, 0, "correctness_q1")

    runtime.validate_run_request(json.dumps({"scale_factor": 1, "optimize": False, "query_id": ["1"]}))

    with pytest.raises(RecoverableStagePolicyError, match='query_id=\\["1"\\]'):
        runtime.validate_run_request(
            json.dumps({"scale_factor": 1, "optimize": False, "query_id": None})
        )

    with pytest.raises(RecoverableStagePolicyError, match='query_id=\\["1"\\]'):
        runtime.validate_run_request(
            json.dumps({"scale_factor": 1, "optimize": False, "query_id": ["1", "2"]})
        )


def test_cached_compaction_session_success_does_not_set_error_on_span(tmp_path, monkeypatch) -> None:
    from tpch_monetdb.llm_cache.cached_compaction_session import CachedOpenAIResponsesCompactionSession

    class FakeSpan:
        def __init__(self):
            self.error_calls: list[dict[str, object]] = []

        def set_error(self, data: dict[str, object]) -> None:
            self.error_calls.append(data)

    fake_span = FakeSpan()

    def fake_custom_span(_name: str, _data: dict | None = None):
        class Ctx:
            def __enter__(self):
                return fake_span
            def __exit__(self, *args):
                return None
        return Ctx()

    monkeypatch.setattr("tpch_monetdb.llm_cache.cached_compaction_session.custom_span", fake_custom_span)

    class FakeUnderlying:
        async def get_items(self):
            return [{"role": "user", "content": "hello"}]
        async def clear_session(self):
            return None
        async def add_items(self, _items):
            return None

    class FakeCompacted:
        output = [{"role": "user", "content": "summary"}]

    class FakeClient:
        class responses:
            @staticmethod
            async def compact(**_kwargs):
                return FakeCompacted()

    session = CachedOpenAIResponsesCompactionSession(
        cache_dir=tmp_path,
        wandb_metrics_hook=None,
        model="gpt-test",
        session_id="test-session",
        underlying_session=FakeUnderlying(),
        client=FakeClient(),
    )
    session._response_id = "resp-1"

    async def run() -> None:
        await session.run_compaction()

    import asyncio
    asyncio.run(run())

    assert fake_span.error_calls == []


def test_cached_litellm_missing_snapshot_is_treated_as_cache_miss(tmp_path) -> None:
    class FakeSnapshotter:
        def __init__(self) -> None:
            self.fetch_calls = 0

        def has_snapshot(self, _commit_hash: str) -> bool:
            return False

        def fetch_snapshots(self) -> None:
            self.fetch_calls += 1
            return None

        def is_dirty(self) -> bool:
            return False

    cache_file = tmp_path / "cached.pkl"
    cache_file.write_bytes(b"placeholder")
    model = object.__new__(CachedLitellmModel)
    model.snapshotter = FakeSnapshotter()
    cached = CacheType(response=None, parent_hash="deadbeef")

    allowed = model._prepare_cached_response(cache_file, cached)

    assert allowed is False
    assert model.snapshotter.fetch_calls == 1
    assert cache_file.exists() is False


def test_tpch_base_conversation_emits_profiled_steps(tmp_path) -> None:
    conversation_dir = tmp_path / "conversations"
    artifacts_dir = tmp_path / "artifacts"
    tpch_monetdb.run_gen_base_impl_tpch_monetdb.create_conversation(
        short_name="basef1-2v1",
        query_ids=["1", "2"],
        verify_sf_list=[1],
        max_scale_factor=1,
        artifacts_dir=artifacts_dir,
        benchmark="tpch",
        conversation_dir=conversation_dir,
        sample_query_args_dict={},
        base_data_dir=tmp_path / "real_data_root",
    )

    target_path = conversation_dir / "tpch_basef1-2v1.json"
    data = json.loads(target_path.read_text(encoding="utf-8"))
    assert COMPACTION_MARKER not in data

    def stage(descriptor: str) -> dict:
        """Return a generated prompt step by descriptor."""
        return next(
            item for item in data
            if isinstance(item, dict) and item.get("descriptor") == descriptor
        )

    stage1 = stage("todo_plan")
    stage2 = stage("finish_skeleton")
    stage3 = stage("compile_fix")
    stage4 = stage("todo_sync")
    stage5 = stage("add_timings")
    stage6 = stage("implement_query_1")
    stage7 = stage("correctness_query_1")
    stage8 = stage("todo_sync_q1")
    stage9 = stage("implement_query_2")
    stage10_query = stage("correctness_query_2")
    stage11 = stage("todo_sync_q2")
    stage10 = stage("all_queries_correctness")
    stage12 = stage("benchmark")
    stage13 = stage("todo_sync_benchmark")
    stage14 = stage("optimize_build")
    stage_descriptors = [
        item.get("descriptor")
        for item in data
        if isinstance(item, dict) and "descriptor" in item
    ]

    assert stage1["descriptor"] == "todo_plan"
    assert stage1["tool_profile"] == "todo_plan"
    assert stage1["max_turns"] == 120
    assert stage1["required_updated_files"] == ["TODO.md"]
    assert "Every checklist item in this initial TODO plan must start unchecked" in stage1["text"]
    assert "The supported TPC-H workload spans Q1-Q22" in stage1["text"]
    assert "hostname->series_id" not in stage1["text"]
    assert "latest_row_id" not in stage1["text"]
    assert stage2["descriptor"] == "finish_skeleton"
    assert stage2["tool_profile"] == "finish_skeleton"
    assert stage2["max_turns"] == 384
    assert "Do not edit `TODO.md` in this step." in stage2["text"]
    assert "Prefer `apply_patch` for non-trivial skeleton edits." not in stage2["text"]
    assert "do not audit parser or interface mismatches in this stage" not in stage2["text"]
    assert "Do not call the compile tool or the run tool in this step." not in stage2["text"]
    assert stage2["stop_conditions"] == ["write_required"]
    assert stage3["descriptor"] == "compile_fix"
    assert stage3["tool_profile"] == "compile_fix"
    assert stage3["max_turns"] == 512
    assert stage4["descriptor"] == "todo_sync"
    assert stage4["tool_profile"] == "todo_sync"
    assert stage4["max_turns"] == 64
    assert stage4["stop_conditions"] == ["write_required"]
    assert stage5["descriptor"] == "add_timings"
    assert stage5["tool_profile"] == "add_timings"
    assert stage5["max_turns"] == 160
    assert stage6["descriptor"] == "implement_query_1"
    assert stage6["tool_profile"] == "implement_queries_writeonly"
    assert stage6["max_turns"] == 160
    assert stage6["required_nonempty_files"] == ["query_q1.cpp"]
    assert stage6["active_unit_id"] == "query:1"
    assert stage6["active_unit_kind"] == "query"
    assert "required_updated_files" not in stage6
    assert "stop_conditions" not in stage6
    assert "Primary implementation must live in `query_q1.cpp`" in stage6["text"]
    assert "TPC-H table loading" in stage6["text"]
    assert "latest_row_id" not in stage6["text"]
    assert "ISO-8601 UTC with `Z` suffix" not in stage6["text"]
    assert stage7["descriptor"] == "correctness_query_1"
    assert stage7["tool_profile"] == "correctness_queries_writeonly"
    assert stage7["max_turns"] == 320
    assert stage7["required_nonempty_files"] == ["query_q1.cpp"]
    assert stage7["stop_conditions"] == ["validation_passed"]
    assert stage8["descriptor"] == "todo_sync_q1"
    assert stage8["tool_profile"] == "todo_sync"
    assert stage_descriptors[stage_descriptors.index("todo_sync_q1") + 1] == "implement_query_2"
    assert stage9["descriptor"] == "implement_query_2"
    assert stage9["tool_profile"] == "implement_queries_writeonly"
    assert stage9["required_nonempty_files"] == ["query_q2.cpp"]
    assert "query_q2.cpp" in stage9["text"]
    assert "Engine consumer" in stage9["text"]
    assert stage10_query["descriptor"] == "correctness_query_2"
    assert stage10_query["tool_profile"] == "correctness_queries_writeonly"
    assert stage10_query["stop_conditions"] == ["validation_passed"]
    assert stage10["descriptor"] == "all_queries_correctness"
    assert stage10["max_turns"] == 320
    assert stage10["stop_conditions"] == ["validation_passed"]
    assert stage11["descriptor"] == "todo_sync_q2"
    assert stage12["descriptor"] == "benchmark"
    assert stage12["max_turns"] == 120
    assert stage12["tool_profile"] == "benchmark"
    assert 'query_id=["1", "2"]' in stage12["text"]
    assert "not as a pass/fail gate" in stage12["text"]
    assert stage13["descriptor"] == "todo_sync_benchmark"
    assert stage13["tool_profile"] == "todo_sync"
    assert stage14["descriptor"] == "optimize_build"
    assert stage14["max_turns"] == 192
    assert "Stay builder-centric by default" in stage14["text"]
    assert "builder_impl.hpp" not in stage14["text"]
    assert "include Q1, Q8, Q11, and Q15 when they are present" in stage14["text"]
    assert "If a change causes correctness regression or `Broken pipe`" in stage14["text"]
    assert "Workflow priority: P0=correctness > P1=query runtime / speedup vs QuestDB > P2=build/ingest time guardrail" not in stage14["text"]
    assert "do not use that number as a pass/fail gate" not in stage14["text"]


def test_tpch_base_conversation_inserts_observation_benchmark_probes(tmp_path) -> None:
    """Base generation should measure performance every five completed queries."""
    conversation_dir = tmp_path / "conversations"
    artifacts_dir = tmp_path / "artifacts"
    query_ids = [str(query_id) for query_id in range(1, 13)]
    tpch_monetdb.run_gen_base_impl_tpch_monetdb.create_conversation(
        short_name="basef1-12v1",
        query_ids=query_ids,
        verify_sf_list=[1],
        max_scale_factor=1,
        artifacts_dir=artifacts_dir,
        benchmark="tpch",
        conversation_dir=conversation_dir,
        sample_query_args_dict={},
        base_data_dir=tmp_path / "real_data_root",
    )

    target_path = conversation_dir / "tpch_basef1-12v1.json"
    data = json.loads(target_path.read_text(encoding="utf-8"))
    stages = {
        item["descriptor"]: item
        for item in data
        if isinstance(item, dict) and "descriptor" in item
    }
    stage_descriptors = [
        item.get("descriptor")
        for item in data
        if isinstance(item, dict) and "descriptor" in item
    ]

    assert "base_perf_probe_q1_q5" in stages
    assert "base_perf_probe_q6_q10" in stages
    assert "base_perf_probe_q11_q12" not in stages
    assert stages["base_perf_probe_q1_q5"]["tool_profile"] == "benchmark"
    assert stages["base_perf_probe_q1_q5"]["active_unit_query_ids"] == [
        "1",
        "2",
        "3",
        "4",
        "5",
    ]
    assert (
        'query_id=["1", "2", "3", "4", "5"]'
        in stages["base_perf_probe_q1_q5"]["text"]
    )
    assert "stop_conditions" not in stages["base_perf_probe_q1_q5"]
    assert (
        stage_descriptors[stage_descriptors.index("base_perf_probe_q1_q5") + 1]
        == "todo_sync_perf_q1_q5"
    )
    assert (
        stage_descriptors[stage_descriptors.index("base_perf_probe_q6_q10") + 1]
        == "todo_sync_perf_q6_q10"
    )
    assert stages["benchmark"]["active_unit_query_ids"] == query_ids
    assert (
        'query_id=["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"]'
        in stages["benchmark"]["text"]
    )
    return None


def test_scripted_required_outputs_match_tool_profile_scope(tmp_path) -> None:
    """所有 scripted 阶段声明的 required 输出必须落在对应工具 profile 的写入范围内。"""
    conversation_dir = tmp_path / "conversations"
    storage_conversation_dir = tmp_path / "storage_conversations"
    artifacts_dir = tmp_path / "artifacts"
    query_ids = [str(query_id) for query_id in range(1, 23)]

    tpch_monetdb.run_gen_storage_plan_tpch_monetdb.create_conversation(
        benchmark="tpch",
        short_name="storage_scope_check",
        conversation_dir=storage_conversation_dir,
        base_data_dir=tmp_path / "real_data_root",
        max_scale_factor=1,
        query_ids=query_ids,
    )
    tpch_monetdb.run_gen_base_impl_tpch_monetdb.create_conversation(
        short_name="base_scope_check",
        query_ids=query_ids,
        verify_sf_list=[1],
        max_scale_factor=1,
        artifacts_dir=artifacts_dir,
        benchmark="tpch",
        conversation_dir=conversation_dir,
        sample_query_args_dict={},
        base_data_dir=tmp_path / "real_data_root",
        storage_plan_snapshot="storage-plan-hash",
    )

    profiles = build_tool_profiles()
    conversation_paths = (
        storage_conversation_dir / "tpch_storage_scope_check.json",
        conversation_dir / "tpch_base_scope_check.json",
    )
    for conversation_path in conversation_paths:
        data = json.loads(conversation_path.read_text(encoding="utf-8"))
        if conversation_path.name == "tpch_base_scope_check.json":
            assert COMPACTION_MARKER not in data
        for index, step in enumerate(data):
            if not isinstance(step, dict):
                continue
            profile_name = step.get("tool_profile", "legacy_general")
            profile = profiles[profile_name]
            descriptor = step.get("descriptor", f"index_{index}")
            for relative_path in step.get("required_nonempty_files", []):
                assert (
                    profile.allows_write(relative_path)
                    or profile.allows_edit(relative_path)
                    or profile.allows_create(relative_path)
                ), (
                    f"{conversation_path.name}:{descriptor} requires {relative_path} "
                    f"but profile {profile_name} cannot write/edit/create it"
                )
            for relative_path in step.get("required_updated_files", []):
                assert (
                    profile.allows_write(relative_path)
                    or profile.allows_edit(relative_path)
                    or profile.allows_create(relative_path)
                ), (
                    f"{conversation_path.name}:{descriptor} requires updated "
                    f"{relative_path} but profile {profile_name} cannot update it"
                )
    return None


def test_tpch_base_extra_query_implementation_stage_allows_noop(tmp_path) -> None:
    conversation_dir = tmp_path / "conversations"
    artifacts_dir = tmp_path / "artifacts"
    tpch_monetdb.run_gen_base_impl_tpch_monetdb.create_conversation(
        short_name="basef1-2-3v1",
        query_ids=["1", "2", "3"],
        verify_sf_list=[1],
        max_scale_factor=1,
        artifacts_dir=artifacts_dir,
        benchmark="tpch",
        conversation_dir=conversation_dir,
        sample_query_args_dict={},
        base_data_dir=tmp_path / "real_data_root",
    )

    target_path = conversation_dir / "tpch_basef1-2-3v1.json"
    data = json.loads(target_path.read_text(encoding="utf-8"))
    implement_q3 = next(
        item for item in data
        if isinstance(item, dict)
        and item["descriptor"] == "implement_query_3"
    )

    assert implement_q3["tool_profile"] == "implement_queries_writeonly"
    assert implement_q3["required_nonempty_files"] == ["query_q3.cpp"]
    assert implement_q3["active_unit_id"] == "query:3"
    assert "required_updated_files" not in implement_q3
    assert "stop_conditions" not in implement_q3


def test_tpch_base_conversation_uses_real_base_data_dir_in_prompt(tmp_path) -> None:
    conversation_dir = tmp_path / "conversations"
    artifacts_dir = tmp_path / "artifacts"
    data_root = tmp_path / "custom_data_root"
    tpch_monetdb.run_gen_base_impl_tpch_monetdb.create_conversation(
        short_name="basef1-2v1",
        query_ids=["1", "2"],
        verify_sf_list=[1],
        max_scale_factor=1,
        artifacts_dir=artifacts_dir,
        benchmark="tpch",
        conversation_dir=conversation_dir,
        sample_query_args_dict={},
        base_data_dir=data_root,
    )

    target_path = conversation_dir / "tpch_basef1-2v1.json"
    data = json.loads(target_path.read_text(encoding="utf-8"))
    stage1 = data[1]

    assert f"{data_root}/sf<SCALE_FACTOR>" in stage1["text"]
    assert "TPC-H table directory" in stage1["text"]
    assert "customer.tbl" in stage1["text"]


def test_main_tpch_monetdb_preserves_cli_max_scale_factor(monkeypatch, tmp_path) -> None:
    """验证 TPC-H verify defaults 不会覆盖 CLI 传入的 max_scale_factor."""
    source = Path(tpch_monetdb.main_tpch_monetdb.__file__).read_text(encoding="utf-8")
    assert "default_verify_sf_list, _default_max_sf = get_default_verify_scale_factors(" in source


def test_bespoke_runtime_provider_rejects_missing_batch_timing_index() -> None:
    from tpch_monetdb.benchmark.providers import GeneratedTpchRuntimeProvider

    provider = GeneratedTpchRuntimeProvider()

    with pytest.raises(ValueError, match="Timing index 1 out of range"):
        provider._parse_timing("", "1 | Execution ms: 10.0", query_id="2", index=1)


def test_cached_litellm_config_kwargs_are_not_shared_across_instances(tmp_path, monkeypatch) -> None:
    from tpch_monetdb.llm_cache.cached_litellm import CachedLitellmModel

    monkeypatch.setattr(
        "tpch_monetdb.llm_cache.cached_litellm.LitellmModel.__init__",
        lambda self, *args, **kwargs: None,
    )

    m1 = CachedLitellmModel(llm_cache_dir=tmp_path / "c1", model="test")
    m2 = CachedLitellmModel(llm_cache_dir=tmp_path / "c2", model="test")
    m1.config_kwargs["extra"] = "value"

    assert "extra" not in m2.config_kwargs


def test_prompt_cache_diagnostics_distinguishes_input_prefix_from_tail() -> None:
    from tpch_monetdb.llm_cache.prompt_cache_diagnostics import (
        changed_prompt_components,
        summarize_prompt_cache_payload,
    )

    base_payload = {
        "model": "openai/gpt-5.5",
        "system_instructions": "stable",
        "input": [{"idx": idx, "text": "stable"} for idx in range(5)],
        "model_settings": {"temperature": 0},
        "tools": [],
        "output_schema": None,
        "conversation_id": None,
        "previous_response_id": None,
        "prompt": None,
        "query_gen_list": ["1"],
        "artifacts_in_context": "artifact-hash",
        "config_kwargs": "stream_llm=True",
    }
    changed_payload = {
        **base_payload,
        "input": [
            *base_payload["input"][:4],
            {"idx": 4, "text": "tail changed"},
        ],
    }

    before = summarize_prompt_cache_payload(base_payload, stream=True)
    after = summarize_prompt_cache_payload(changed_payload, stream=True)

    changed = changed_prompt_components(before, after)
    assert "input" in changed
    assert "input_prefix" not in changed


def test_prompt_cache_diagnostics_reports_provider_cache_break(caplog) -> None:
    from tpch_monetdb.llm_cache.prompt_cache_diagnostics import PromptCacheDiagnostics

    diagnostics = PromptCacheDiagnostics()
    usage_hit = SimpleNamespace(
        input_tokens=6_000,
        input_tokens_details=SimpleNamespace(cached_tokens=5_000),
        request_usage_entries=[],
    )
    usage_miss = SimpleNamespace(
        input_tokens=6_100,
        input_tokens_details=SimpleNamespace(cached_tokens=0),
        request_usage_entries=[],
    )
    base_payload = {
        "model": "openai/gpt-5.5",
        "system_instructions": "stable",
        "input": "hello",
        "model_settings": {"temperature": 0},
        "tools": [],
        "output_schema": None,
        "conversation_id": None,
        "previous_response_id": None,
        "prompt": None,
        "query_gen_list": ["1"],
        "artifacts_in_context": "artifact-hash",
        "config_kwargs": "stream_llm=True",
    }

    first = diagnostics.begin_request(
        request_hash="h1",
        payload=base_payload,
        stream=True,
    )
    diagnostics.complete_request(first, usage_hit, model="openai/gpt-5.5")
    second = diagnostics.begin_request(
        request_hash="h2",
        payload={**base_payload, "system_instructions": "changed"},
        stream=True,
    )

    with caplog.at_level(
        logging.WARNING,
        logger="tpch_monetdb.llm_cache.prompt_cache_diagnostics",
    ):
        diagnostics.complete_request(second, usage_miss, model="openai/gpt-5.5")

    assert "LLM provider prompt cache break" in caplog.text
    assert "system_instructions" in caplog.text


def test_prompt_cache_diagnostics_rate_limits_near_zero_warning(caplog) -> None:
    from tpch_monetdb.llm_cache.prompt_cache_diagnostics import PromptCacheDiagnostics

    diagnostics = PromptCacheDiagnostics()
    usage_zero = SimpleNamespace(
        input_tokens=6_000,
        input_tokens_details=SimpleNamespace(cached_tokens=0),
        request_usage_entries=[],
    )
    base_payload = {
        "model": "openai/gpt-5.5",
        "system_instructions": "v1",
        "input": "hello",
        "model_settings": {"temperature": 0},
        "tools": [],
        "output_schema": None,
        "conversation_id": None,
        "previous_response_id": None,
        "prompt": None,
        "query_gen_list": ["1"],
        "artifacts_in_context": "artifact-hash",
        "config_kwargs": "stream_llm=True",
    }

    first = diagnostics.begin_request(
        request_hash="h1",
        payload=base_payload,
        stream=True,
    )
    diagnostics.complete_request(first, usage_zero, model="openai/gpt-5.5")

    with caplog.at_level(
        logging.WARNING,
        logger="tpch_monetdb.llm_cache.prompt_cache_diagnostics",
    ):
        second = diagnostics.begin_request(
            request_hash="h2",
            payload={**base_payload, "system_instructions": "v2"},
            stream=True,
        )
        diagnostics.complete_request(second, usage_zero, model="openai/gpt-5.5")
        third = diagnostics.begin_request(
            request_hash="h3",
            payload={**base_payload, "system_instructions": "v3"},
            stream=True,
        )
        diagnostics.complete_request(third, usage_zero, model="openai/gpt-5.5")

    assert caplog.text.count("LLM provider prompt cache read is near zero") == 1


def test_prompt_cache_diagnostics_warns_for_native_deepseek_near_zero_cache(caplog) -> None:
    from tpch_monetdb.llm_cache.prompt_cache_diagnostics import PromptCacheDiagnostics

    diagnostics = PromptCacheDiagnostics()
    usage_zero = SimpleNamespace(
        input_tokens=6_000,
        input_tokens_details=SimpleNamespace(cached_tokens=0),
        request_usage_entries=[],
    )
    base_payload = {
        "model": "deepseek/deepseek-v4-pro",
        "system_instructions": "v1",
        "input": "hello",
        "model_settings": {"thinking": {"type": "enabled"}},
        "tools": [],
        "output_schema": None,
        "conversation_id": None,
        "previous_response_id": None,
        "prompt": None,
        "query_gen_list": ["1"],
        "artifacts_in_context": "artifact-hash",
        "config_kwargs": "stream_llm=True",
    }

    first = diagnostics.begin_request(
        request_hash="h1",
        payload=base_payload,
        stream=False,
    )
    diagnostics.complete_request(first, usage_zero, model="deepseek/deepseek-v4-pro")

    with caplog.at_level(
        logging.WARNING,
        logger="tpch_monetdb.llm_cache.prompt_cache_diagnostics",
    ):
        second = diagnostics.begin_request(
            request_hash="h2",
            payload={**base_payload, "system_instructions": "v2"},
            stream=False,
        )
        diagnostics.complete_request(second, usage_zero, model="deepseek/deepseek-v4-pro")

    assert "LLM provider prompt cache read is near zero" in caplog.text
    assert "deepseek/deepseek-v4-pro" in caplog.text


def test_cached_litellm_hash_payload_keeps_stream_key_optional(tmp_path, monkeypatch) -> None:
    from tpch_monetdb.llm_cache.cached_litellm import CachedLitellmModel

    monkeypatch.setattr(
        "tpch_monetdb.llm_cache.cached_litellm.LitellmModel.__init__",
        lambda self, *args, **kwargs: None,
    )
    model = CachedLitellmModel(llm_cache_dir=tmp_path / "cache", model="openai/test")
    model.model = "openai/test"
    model_settings = SimpleNamespace(to_json_dict=lambda: {"temperature": 0})

    non_stream_payload = model._build_hash_payload(
        system_instructions="system",
        input="input",
        model_settings=model_settings,
        tools=[],
        output_schema=None,
        handoffs=[],
        previous_response_id=None,
        conversation_id=None,
        prompt=None,
        stream=False,
    )
    stream_payload = model._build_hash_payload(
        system_instructions="system",
        input="input",
        model_settings=model_settings,
        tools=[],
        output_schema=None,
        handoffs=[],
        previous_response_id=None,
        conversation_id=None,
        prompt=None,
        stream=True,
    )

    assert "stream" not in non_stream_payload
    assert stream_payload["stream"] is True
    assert model._hash_payload(
        system_instructions="system",
        input="input",
        model_settings=model_settings,
        tools=[],
        output_schema=None,
        handoffs=[],
        previous_response_id=None,
        conversation_id=None,
        prompt=None,
        stream=False,
    ) == model._hash_cache_payload(non_stream_payload)


@pytest.mark.asyncio
async def test_cached_litellm_retries_transient_network_failure(
    tmp_path, monkeypatch
) -> None:
    from tpch_monetdb.llm_cache.cached_litellm import CachedLitellmModel

    class FakeAPIConnectionError(Exception):
        pass

    attempts = {"count": 0}
    sleeps: list[float] = []

    async def fake_super_get_response(self, *args, **kwargs):
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise FakeAPIConnectionError("Server disconnected")
        return SimpleNamespace(usage=SimpleNamespace())

    async def fake_sleep(delay: float) -> None:
        sleeps.append(delay)
        return None

    class FakeSnapshotter:
        def __init__(self, working_dir: Path) -> None:
            self.working_dir = working_dir

        def snapshot(self, _req_hash: str) -> tuple[str, str]:
            return "", "commit"

        def push_snapshots(self) -> None:
            return None

    monkeypatch.setattr(
        "tpch_monetdb.llm_cache.cached_litellm.LitellmModel.get_response",
        fake_super_get_response,
    )
    monkeypatch.setattr("tpch_monetdb.llm_cache.litellm_retry.asyncio.sleep", fake_sleep)
    monkeypatch.setattr(
        "tpch_monetdb.llm_cache.cached_litellm.CachedLitellmModel._ensure_usage_entries",
        staticmethod(lambda _usage: None),
    )
    monkeypatch.setattr(
        "tpch_monetdb.llm_cache.cached_litellm.get_tokens_context_and_dollar_info",
        lambda *_args, **_kwargs: {"cost": None},
    )
    monkeypatch.setattr(
        "tpch_monetdb.llm_cache.cached_litellm.utils.dump_pickle",
        lambda *_args, **_kwargs: None,
    )

    model = CachedLitellmModel(
        model="anthropic/test-model",
        llm_cache_dir=tmp_path / "cache",
        snapshotter=FakeSnapshotter(tmp_path),
    )

    response = await model.get_response(
        system_instructions=None,
        input="hello",
        model_settings=SimpleNamespace(to_json_dict=lambda: {}),
        tools=[],
        output_schema=None,
        handoffs=[],
        previous_response_id=None,
        conversation_id=None,
        prompt=None,
    )

    assert response.usage is not None
    assert attempts["count"] == 3
    assert sleeps == [1.0, 2.0]


def test_cached_openai_config_kwargs_are_not_shared_across_instances(tmp_path, monkeypatch) -> None:
    from tpch_monetdb.llm_cache.cached_openai import CachedOpenAIResponsesModel

    monkeypatch.setattr(
        "tpch_monetdb.llm_cache.cached_openai.OpenAIResponsesModel.__init__",
        lambda self, *args, **kwargs: None,
    )

    m1 = CachedOpenAIResponsesModel(llm_cache_dir=tmp_path / "c1", session_id="s1")
    m2 = CachedOpenAIResponsesModel(llm_cache_dir=tmp_path / "c2", session_id="s2")
    m1.config_kwargs["extra"] = "value"

    assert "extra" not in m2.config_kwargs


def test_run_tool_reraises_file_not_found_when_db_exists(monkeypatch, tmp_path) -> None:
    from tpch_monetdb.tools.tpch.run import RunTool

    class FakeCompiler:
        def set_extra_cxxflags(self, _flags) -> None:
            return None
        def build_cached(self, **_kwargs):
            return None, False, "key"

    class FakeRunner:
        def run_batch(
            self,
            args_list: list[str],
            timeout: int,
        ) -> tuple[str, str, str]:
            raise FileNotFoundError("missing data file")

    monkeypatch.setattr("tpch_monetdb.tools.tpch.run.make_compiler", lambda *_a, **_k: FakeCompiler())
    monkeypatch.setattr("tpch_monetdb.tools.tpch.run.FastTestPool.get", lambda _c, _f: FakeRunner())

    (tmp_path / "db").write_text("fake executable", encoding="utf-8")
    run_tool = RunTool(
        cwd=tmp_path,
        dataset_name="tpch",
        base_data_dir="/tmp/tpch_monetdb_data",
        query_validator=None,
    )

    with pytest.raises(FileNotFoundError, match="missing data file"):
        run_tool.run(scale_factor=1, optimize=False, query_id=["1"])


def test_setup_model_config_raises_for_missing_openai_key(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("LITELLM_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="OPENAI_API_KEY must be set"):
        setup_model_config("gpt-4")


def test_run_gen_storage_plan_assert_has_descriptive_message() -> None:
    import tpch_monetdb.run_gen_storage_plan_tpch_monetdb
    source = Path(tpch_monetdb.run_gen_storage_plan_tpch_monetdb.__file__).read_text(encoding="utf-8")
    assert "Expected conv name starting with" in source


def test_sandbox_unavailable_reason_has_single_return() -> None:
    source = Path(tpch_monetdb.tools.sandbox.__file__).read_text(encoding="utf-8")
    # After fix, the function should contain only one return for unsupported platform
    assert source.count('return f"Sandbox unsupported on platform') == 1


def test_get_affinity_prompt_allows_numa_true() -> None:
    from tpch_monetdb.utils.general_utils import get_affinity_prompt

    prompt = get_affinity_prompt(include_numa=True, filename="test.hpp")
    assert "NUMA placement:" in prompt
    assert "pin_process_to_numa_node" in prompt


def test_calculate_loc_uses_stripped_stdout_for_json(monkeypatch, tmp_path) -> None:
    from tpch_monetdb.utils import cloc_utils

    class FakeResult:
        returncode = 0
        stdout = "  \n  \n"
        stderr = ""

    monkeypatch.setattr(cloc_utils.subprocess, "run", lambda *_a, **_k: FakeResult())

    assert cloc_utils.calculate_loc(tmp_path, current_hash="abc", working_dir=tmp_path) == 0


def test_wandb_run_hook_instances_do_not_share_counters(tmp_path) -> None:
    from tpch_monetdb.utils.wandb_stats_logging import WandbRunHook

    hook1 = WandbRunHook(
        model="m1",
        git_snapshotter=SimpleNamespace(current_hash="h1", working_dir=tmp_path),
    )
    hook2 = WandbRunHook(
        model="m2",
        git_snapshotter=SimpleNamespace(current_hash="h2", working_dir=tmp_path),
    )
    hook1.logged_turn = 5
    hook1.apply_patch_added_ctr = 3

    assert hook2.logged_turn == -1
    assert hook2.apply_patch_added_ctr == 0


def test_main_tpch_monetdb_uses_deterministic_wandb_run_id(monkeypatch, tmp_path) -> None:
    captured: dict[str, object] = {}

    def fake_wandb_init(**kwargs) -> object:
        captured["wandb_init"] = kwargs
        return object()

    monkeypatch.setattr(tpch_monetdb.main_tpch_monetdb, "load_dotenv", lambda: None)
    monkeypatch.setattr(tpch_monetdb.main_tpch_monetdb, "setup_logging", lambda *_a, **_k: None)
    monkeypatch.setattr(tpch_monetdb.main_tpch_monetdb, "set_tracing_disabled", lambda _d: None)
    monkeypatch.setattr(
        tpch_monetdb.main_tpch_monetdb,
        "_upload_workspace_code_to_wandb",
        lambda _r, _w, timeout_s=0.0: None,
    )
    monkeypatch.setattr(tpch_monetdb.main_tpch_monetdb, "wandb", SimpleNamespace(init=fake_wandb_init, run=object(), finish=lambda: None))
    monkeypatch.setattr(tpch_monetdb.main_tpch_monetdb, "asyncio", SimpleNamespace(run=lambda c: c.close()))

    args = SimpleNamespace(
        continue_run=False,
        artifacts_dir=str(tmp_path / "artifacts"),
        conv_name="tpch_monetdb_testv1",
        disable_tracing=True,
        disable_wandb=False,
        benchmark="tpch",
        is_bespoke_storage=False,
        disable_repo_sync=True,
    )

    tpch_monetdb.main_tpch_monetdb.run_conv_wrapper(args)

    import hashlib

    expected_id = hashlib.md5("tpch_monetdb_testv1".encode("utf-8")).hexdigest()
    assert captured["wandb_init"]["id"] == expected_id
    assert captured["wandb_init"]["resume"] == "allow"


def test_run_conv_wrapper_can_link_disable_tracing_to_disable_wandb(
    monkeypatch,
    tmp_path,
) -> None:
    captured: dict[str, object] = {}

    def fail_wandb_init(**_kwargs):
        raise AssertionError("should not init wandb")

    monkeypatch.setattr(tpch_monetdb.main_tpch_monetdb, "load_dotenv", lambda: None)
    monkeypatch.setattr(tpch_monetdb.main_tpch_monetdb, "setup_logging", lambda *_a, **_k: None)
    monkeypatch.setattr(
        tpch_monetdb.main_tpch_monetdb,
        "set_tracing_disabled",
        lambda disabled: captured.setdefault("tracing_disabled", disabled),
    )
    monkeypatch.setattr(
        tpch_monetdb.main_tpch_monetdb,
        "init_wandb_run_with_retry",
        fail_wandb_init,
    )
    monkeypatch.setattr(
        tpch_monetdb.main_tpch_monetdb,
        "asyncio",
        SimpleNamespace(run=lambda coro: coro.close()),
    )

    args = SimpleNamespace(
        continue_run=False,
        artifacts_dir=str(tmp_path / "artifacts"),
        conv_name="tpch_monetdb_testv1",
        disable_tracing=True,
        disable_wandb=False,
        disable_wandb_when_tracing_disabled=True,
        benchmark="tpch",
        is_bespoke_storage=False,
        disable_repo_sync=True,
    )

    tpch_monetdb.main_tpch_monetdb.run_conv_wrapper(args)

    assert captured["tracing_disabled"] is True
    assert args.disable_wandb is True


def test_upload_workspace_code_to_wandb_raises_timeout(tmp_path) -> None:
    import time

    workspace_path = tmp_path / "output"
    workspace_path.mkdir(parents=True, exist_ok=True)
    (workspace_path / "query_impl.cpp").write_text("int main(){return 0;}", encoding="utf-8")

    class SlowRun:
        def log_code(self, *, root: str, name: str, include_fn) -> None:
            _ = (root, name, include_fn)
            time.sleep(0.05)
            return None

    with pytest.raises(RuntimeError) as exc_info:
        tpch_monetdb.main_tpch_monetdb._upload_workspace_code_to_wandb(
            SlowRun(),
            workspace_path,
            timeout_s=0.01,
        )

    assert "[ERROR:WANDB_LOG_CODE_TIMEOUT]" in str(exc_info.value)


def test_request_cost_usd_clamps_negative_billable_input() -> None:
    from tpch_monetdb.llm_cache.models import request_cost_usd

    cost = request_cost_usd(
        "kimi-k2.5",
        input_tokens=100,
        cached_tokens=200,
        output_tokens=50,
    )
    assert cost >= 0


def test_get_tokens_context_and_dollar_info_returns_none_cost_for_unknown_model(
    monkeypatch,
) -> None:
    from tpch_monetdb.utils.token_usage import get_tokens_context_and_dollar_info

    class FakeUsage:
        input_tokens = 10
        output_tokens = 5
        input_tokens_details = SimpleNamespace(cached_tokens=0)
        output_tokens_details = SimpleNamespace(reasoning_tokens=0)
        request_usage_entries = [
            SimpleNamespace(
                input_tokens=10,
                output_tokens=5,
                input_tokens_details=SimpleNamespace(cached_tokens=0),
                output_tokens_details=SimpleNamespace(reasoning_tokens=0),
            )
        ]

    info = get_tokens_context_and_dollar_info(
        FakeUsage(), model="unknown-model-xyz", last_entry_only=True
    )

    assert info["cost"] is None
    assert info["pricing_missing"] is True
    assert info["visible_output_tokens"] == 5
    assert info["billed_output_tokens"] == 5


def test_get_tokens_context_and_dollar_info_splits_visible_and_billed_output() -> None:
    from tpch_monetdb.utils.token_usage import get_tokens_context_and_dollar_info

    class FakeUsage:
        input_tokens = 10
        output_tokens = 15
        input_tokens_details = SimpleNamespace(cached_tokens=2)
        output_tokens_details = SimpleNamespace(reasoning_tokens=5)
        request_usage_entries = [
            SimpleNamespace(
                input_tokens=10,
                output_tokens=15,
                input_tokens_details=SimpleNamespace(cached_tokens=2),
                output_tokens_details=SimpleNamespace(reasoning_tokens=5),
            )
        ]

    info = get_tokens_context_and_dollar_info(
        FakeUsage(), model="kimi-k2.5", last_entry_only=True
    )

    assert info["visible_output_tokens"] == 10
    assert info["billed_output_tokens"] == 15
    assert info["cost"] is not None
    assert info["pricing_missing"] is False


def test_tpch_monetdb_main_source_uses_default_max_turns_75() -> None:
    source = Path(tpch_monetdb.main_tpch_monetdb.__file__).read_text(encoding="utf-8")
    assert "max_turns = 75" in source


def test_tpch_monetdb_main_reactive_compact_checks_effectiveness_before_retry() -> None:
    source = Path(tpch_monetdb.main_tpch_monetdb.__file__).read_text(encoding="utf-8")

    assert "compact_ok = await auto_compact_manager.compact" in source
    assert "if not compact_ok:" in source
    assert "aborting unchanged retry" in source


def test_fasttest_terminate_kills_nonresponsive_process(monkeypatch) -> None:
    from tpch_monetdb.misc.tpch.fasttest_proc import FasttestProc
    import subprocess

    class FakeProc:
        def __init__(self) -> None:
            self.kill_called = False
            self.wait_calls: list[int | None] = []
            self.stdin = None
            self.stdout = None
            self.stderr = None
            self.returncode = 0

        def wait(self, timeout: int | None = None) -> int:
            self.wait_calls.append(timeout)
            if len(self.wait_calls) <= 2:
                raise subprocess.TimeoutExpired(cmd="cmd", timeout=timeout)
            return 0

        def kill(self) -> None:
            self.kill_called = True

    fake_proc = FakeProc()
    popen_kwargs: dict[str, object] = {}

    def fake_popen(*_args, **kwargs):
        popen_kwargs.update(kwargs)
        return fake_proc

    monkeypatch.setattr("tpch_monetdb.misc.tpch.fasttest_proc.subprocess.Popen", fake_popen)

    runner = FasttestProc(command="./db", cwd=Path("/tmp"))
    runner._start()
    runner.terminate()

    assert fake_proc.kill_called is True
    assert fake_proc.wait_calls == [5, 2, 5]
    assert popen_kwargs["start_new_session"] is True


def test_fasttest_run_writes_line_control_and_reads_response(monkeypatch) -> None:
    from tpch_monetdb.misc.tpch.fasttest_proc import FasttestProc
    import os

    control_r, control_w = os.pipe()
    response_r, response_w = os.pipe()
    os.write(response_w, b"exit_code: 0 signal: 0\n")

    def fake_start(self: FasttestProc) -> None:
        self._p2c_w = control_w
        self._c2p_file = object()
        self._c2p_r = response_r
        self._stdout_fd = None
        self._stderr_fd = None
        return None

    monkeypatch.setattr(FasttestProc, "_start", fake_start)

    try:
        runner = FasttestProc(command="./db", cwd=Path("/tmp"))
        resp, out, err = runner.run(timeout=0)
        assert os.read(control_r, 4) == b"run\n"
        assert resp == "exit_code: 0 signal: 0"
        assert out == ""
        assert err == ""
    finally:
        for fd in (control_r, control_w, response_r, response_w):
            try:
                os.close(fd)
            except OSError:
                pass
    return None


def test_fasttest_run_batch_writes_stdin_batch_then_line_control(monkeypatch) -> None:
    from tpch_monetdb.misc.tpch.fasttest_proc import FasttestProc

    writes: list[bytes] = []
    stdin_writes: list[bytes] = []

    class FakeStdin:
        def write(self, data: bytes) -> int:
            stdin_writes.append(data)
            return len(data)

        def flush(self) -> None:
            return None

    def fake_write(fd: int, data: bytes) -> int:
        assert fd == 123
        if not writes:
            split_at = 2
            writes.append(data[:split_at])
            return split_at
        writes.append(data)
        return len(data)

    def fake_start(self: FasttestProc) -> None:
        self._p2c_w = 123
        self._c2p_file = object()
        self._c2p_r = 456
        self._stdin = FakeStdin()
        return None

    def fake_read_response(
        self: FasttestProc,
        timeout: int,
    ) -> tuple[str, str, str]:
        assert timeout == 7
        return "exit_code: 0 signal: 0", "", ""

    monkeypatch.setattr(FasttestProc, "_start", fake_start)
    monkeypatch.setattr(FasttestProc, "_read_run_response", fake_read_response)
    monkeypatch.setattr("tpch_monetdb.misc.tpch.fasttest_proc.os.write", fake_write)

    runner = FasttestProc(command="./db", cwd=Path("/tmp"))
    resp, out, err = runner.run_batch(["1", "8 test"], timeout=7)

    assert resp == "exit_code: 0 signal: 0"
    assert out == ""
    assert err == ""
    assert len(writes) == 2
    assert b"".join(writes) == b"run\n"
    assert stdin_writes == [b"1\n", b"8 test\n", b"\n"]
    return None


def test_fasttest_read_run_response_timeout_terminates_runner(monkeypatch) -> None:
    from tpch_monetdb.misc.tpch.fasttest_proc import FasttestProc

    class FakeProc:
        def __init__(self) -> None:
            self.wait_calls: list[int | None] = []
            self.stdin = None
            self.stdout = None
            self.stderr = None
            self.returncode = 0

        def wait(self, timeout: int | None = None) -> int:
            self.wait_calls.append(timeout)
            return 0

    fake_proc = FakeProc()
    monotonic_values = iter([0.0, 2.0])
    monkeypatch.setattr(
        "tpch_monetdb.misc.tpch.fasttest_proc.time.monotonic",
        lambda: next(monotonic_values),
    )

    runner = FasttestProc(command="./db", cwd=Path("/tmp"))
    runner._proc = fake_proc
    runner._c2p_r = 123
    runner._c2p_file = object()

    resp, out, err = runner._read_run_response(timeout=1)

    assert "Terminated after 1 seconds due to timeout." in resp
    assert out == ""
    assert err == ""
    assert fake_proc.wait_calls == [5]
    assert runner._proc is None
    return None


@pytest.mark.asyncio
async def test_ask_choice_cancels_task_and_avoids_notification_spam(monkeypatch, tmp_path) -> None:
    from tpch_monetdb.conversations.conversation import AbstractConversation
    import asyncio

    notifications: list[tuple[str, bool]] = []

    def track_notification(msg: str, check_tmux: bool = True) -> None:
        notifications.append((msg, check_tmux))

    monkeypatch.setattr("tpch_monetdb.conversations.conversation.send_notification", track_notification)

    call_count = [0]

    async def fake_wait_for(coro, timeout):
        call_count[0] += 1
        if call_count[0] == 1:
            return await coro
        if call_count[0] == 2:
            raise asyncio.TimeoutError
        return await coro

    monkeypatch.setattr(asyncio, "wait_for", fake_wait_for)

    cancel_calls: list[bool] = []
    original_create_task = asyncio.create_task

    def tracked_create_task(coro):
        task = original_create_task(coro)
        orig_cancel = task.cancel

        def patched_cancel(*a, **k):
            cancel_calls.append(True)
            return orig_cancel(*a, **k)

        task.cancel = patched_cancel
        return task

    monkeypatch.setattr(asyncio, "create_task", tracked_create_task)

    class FakeSession:
        async def prompt_async(self, _text) -> str:
            if call_count[0] == 1:
                return "invalid"
            return "u"

    class DummyConv(AbstractConversation):
        async def run(self) -> list[str] | None:
            return None

    conv = DummyConv(
        conversation_json_path=tmp_path / "conv.json",
        callback=lambda *a, **k: None,
        notify=True,
    )
    conv._session = FakeSession()

    result = await conv._ask_choice("test prompt")

    assert result == "u"
    assert len(cancel_calls) == 1
    # at least one tmux-checked notification and one fallback notification
    assert len([n for n in notifications if n[1] is True]) >= 1
    assert len([n for n in notifications if n[1] is False]) == 1


@pytest.mark.asyncio
async def test_optimization_speedup_logging_skipped_on_stage_failure(monkeypatch, tmp_path) -> None:
    from tpch_monetdb.conversations.optimization_conversation_tpch_monetdb import TpchMonetdbOptimizationConversation
    from tpch_monetdb.conversations.optimization_validation import CorrectnessCheckSummary
    from types import SimpleNamespace

    conversation = TpchMonetdbOptimizationConversation.__new__(TpchMonetdbOptimizationConversation)
    conversation.benchmark_sf = 10
    conversation.bespoke_storage = True
    restore_calls: list[str] = []
    conversation.git_snapshotter = SimpleNamespace(
        current_hash="h1",
        restore=lambda h: restore_calls.append(h),
    )
    conversation.query_ids = ["1"]
    conversation.required_validation_sf_list = [1]
    conversation.revert_on_regression = True
    conversation.conversation_json_path = tmp_path / "conv.json"
    conversation.callback = lambda *a, **k: None
    conversation.replay = False
    conversation.notify = False
    conversation.auto_finish = False
    conversation.allowed_choices = ("u", "r", "i", "c")
    conversation.model = None
    conversation.auto_u = False
    conversation.replay_cache = False
    conversation.workspace_root = None
    conversation.used = []
    conversation.get_choice = None

    log_calls: dict[str, bool] = {"speedup": False}

    class FakeHook:
        def log_optimization_stage(self, **kwargs) -> None:
            return None

        def log_optimization_speedup_vs_baseline(self, **kwargs) -> None:
            log_calls["speedup"] = True

    conversation.wandb_run_hook = FakeHook()
    conversation.run_tool = SimpleNamespace(run=lambda **k: ("ok", {}))
    conversation._measure_with_manifest = lambda **k: (0.1, 1.0, 10.0, False)

    def fake_run_required_correctness_checks(*args, **kwargs):
        return CorrectnessCheckSummary(
            success=False,
            message="validation failed",
            metrics={"validation/correct": False},
            failed_scale_factor=1,
        )

    monkeypatch.setattr(
        "tpch_monetdb.conversations.optimization_conversation_tpch_monetdb.run_required_correctness_checks",
        fake_run_required_correctness_checks,
    )

    monkeypatch.setattr(conversation, "_exec", lambda *a, **k: asyncio.sleep(0))
    monkeypatch.setattr(conversation, "process_prompt", lambda *a, **k: asyncio.sleep(0))
    stage = StageConfig(
        name="trace_expert",
        get_prompt=lambda _rt: "",
        get_descriptor=lambda: "trace_expert",
        max_turns=10,
    )

    result = await conversation._run_stage(
        query_id="1",
        stage=stage,
        pretext_optim="",
        rt_before_s=0.5,
    )
    assert result.failed is True
    assert result.failure_message is not None
    assert restore_calls == ["h1"]
    assert log_calls["speedup"] is False


@pytest.mark.asyncio
async def test_optimization_run_survives_stage_end_gate_failure(monkeypatch, tmp_path) -> None:
    from tpch_monetdb.conversations.optimization_conversation_tpch_monetdb import TpchMonetdbOptimizationConversation
    from tpch_monetdb.conversations.optimization_validation import CorrectnessCheckSummary

    (tmp_path / "queries.txt").write_text("1\n", encoding="utf-8")

    conversation = TpchMonetdbOptimizationConversation.__new__(TpchMonetdbOptimizationConversation)
    conversation.benchmark_sf = 10
    conversation.bespoke_storage = True
    conversation.query_ids = ["1"]
    conversation.required_validation_sf_list = [1]
    conversation.revert_on_regression = False
    conversation.regression_tolerance = 0.1
    conversation.git_snapshotter = SimpleNamespace(
        current_hash="h1",
        restore=lambda h: None,
    )
    conversation.conversation_json_path = tmp_path / "conv.json"
    conversation.callback = lambda *a, **k: None
    conversation.replay = False
    conversation.notify = False
    conversation.auto_finish = True
    conversation.allowed_choices = ("u",)
    conversation.model = None
    conversation.auto_u = True
    conversation.replay_cache = False
    conversation.workspace_root = None
    conversation.used = []
    conversation.get_choice = lambda: "u"

    class FakeSession:
        def __init__(self, user_turns=None):
            self._current_branch_id = "main"
            self._user_turns = user_turns or {"main": {1}}
            self.switch_log: list[tuple] = []

        async def switch_to_branch(self, b) -> None:
            self.switch_log.append(("switch_to", b))
            self._current_branch_id = b

        async def create_branch_from_turn(self, t, branch_name) -> str:
            self.switch_log.append(("create_branch", t, branch_name, self._current_branch_id))
            turns = self._user_turns.get(self._current_branch_id, set())
            if t not in turns:
                raise ValueError(
                    f"Turn {t} does not contain a user message "
                    f"in branch '{self._current_branch_id}'"
                )
            self._user_turns[branch_name] = set()
            self._current_branch_id = branch_name
            return branch_name

        async def get_conversation_turns(self) -> list[dict]:
            turns = self._user_turns.get(self._current_branch_id, set())
            return [{"turn": t} for t in sorted(turns)]

    conversation.session = FakeSession()
    conversation.run_tool = SimpleNamespace(
        cwd=tmp_path,
        run=lambda **k: ("ok", {}),
        reset_runtime_state=lambda **kwargs: None,
    )
    conversation._make_exec_callback = lambda qid, **_kwargs: lambda args, t: ("resp", "1 | Execution ms: 10.0", "")
    conversation._measure_with_manifest = lambda **k: (0.1, 1.0, 10.0, False)
    conversation._measure_all_queries = lambda: {"1": 0.1}
    conversation._delete_result_csvs = lambda cwd: None
    conversation._check_correctness_with_scale_factors = (
        lambda qids, trace_mode, scale_factors: asyncio.sleep(0, result=True)
    )
    conversation._summarize_trace_evidence_for_queries = lambda qids: TraceEvidenceSummary(
        qids=tuple(qids),
        sufficient=True,
        message="ok",
    )
    conversation.wandb_run_hook = None
    conversation.conv_name = "test_conv"
    conversation.artifacts_dir = tmp_path / "artifacts"
    conversation.start_snapshot_hash = "h0"
    conversation.query_rt_log = {}
    conversation.best_rt_log = {}
    conversation.global_regression_records = []

    async def dummy_exec(*args, **kwargs) -> None:
        return None
    conversation._exec = dummy_exec

    async def dummy_check_correctness(*args, **kwargs) -> bool:
        return True
    conversation._check_correctness = dummy_check_correctness

    async def dummy_run_stage(**kwargs) -> None:
        return None
    conversation._run_stage = dummy_run_stage

    conversation._sample_trace_for_query = lambda query_id: SimpleNamespace(
        issue_class="scan_bound",
        summary_text="trace summary",
        sampled_instantiations=("i1",),
    )
    conversation._build_query_stage = lambda **kwargs: StageConfig(
        name="trace_expert",
        get_prompt=lambda _rt: "",
        get_descriptor=lambda: "trace_expert",
        max_turns=10,
    )
    conversation._run_stage_correctness_gate = lambda **kwargs: CorrectnessCheckSummary(
        success=True,
        message="ok",
        metrics={"validation/correct": True},
        failed_scale_factor=None,
    )
    conversation._persist_hotspot_summary = lambda records: tmp_path / "hotspots.md"
    conversation._collect_query_output_split_measurements = lambda qids: {}
    async def dummy_global_human_reference(**kwargs) -> GlobalHumanReferenceResult:
        return GlobalHumanReferenceResult(
            runtime_by_query={"1": 0.1},
            written_files=(),
            accepted=True,
        )

    conversation._run_global_human_reference = dummy_global_human_reference
    _patch_tpch_monetdb_optimization_prompt_builders(monkeypatch)

    gate_calls = [0]

    def fake_gate(*args, **kwargs):
        gate_calls[0] += 1
        if gate_calls[0] <= 1:
            return CorrectnessCheckSummary(
                success=True,
                message="ok",
                metrics={"validation/correct": True},
                failed_scale_factor=None,
            )
        raise RuntimeError("gate exploded")

    monkeypatch.setattr(
        "tpch_monetdb.conversations.optimization_conversation_tpch_monetdb.run_required_correctness_checks",
        fake_gate,
    )

    result = await conversation.run()
    assert result == []
    assert gate_calls[0] == 2


@pytest.mark.asyncio
async def test_optimization_run_skips_success_summary_on_final_measurement_failure(
    monkeypatch,
    tmp_path,
) -> None:
    from tpch_monetdb.conversations.optimization_conversation_tpch_monetdb import TpchMonetdbOptimizationConversation
    from tpch_monetdb.conversations.optimization_validation import CorrectnessCheckSummary

    (tmp_path / "queries.txt").write_text("1\n", encoding="utf-8")

    conversation = TpchMonetdbOptimizationConversation.__new__(TpchMonetdbOptimizationConversation)
    conversation.benchmark_sf = 10
    conversation.bespoke_storage = True
    conversation.query_ids = ["1"]
    conversation.required_validation_sf_list = [1]
    conversation.revert_on_regression = False
    conversation.regression_tolerance = 0.1
    conversation.git_snapshotter = SimpleNamespace(
        current_hash="h1",
        restore=lambda h: None,
    )
    conversation.conversation_json_path = tmp_path / "conv.json"
    conversation.callback = lambda *a, **k: None
    conversation.replay = False
    conversation.notify = False
    conversation.auto_finish = True
    conversation.allowed_choices = ("u",)
    conversation.model = None
    conversation.auto_u = True
    conversation.replay_cache = False
    conversation.workspace_root = None
    conversation.used = []
    conversation.get_choice = lambda: "u"

    class FakeSession:
        def __init__(self, user_turns=None):
            self._current_branch_id = "main"
            self._user_turns = user_turns or {"main": {1}}
            self.switch_log: list[tuple] = []

        async def switch_to_branch(self, b) -> None:
            self.switch_log.append(("switch_to", b))
            self._current_branch_id = b

        async def create_branch_from_turn(self, t, branch_name) -> str:
            self.switch_log.append(("create_branch", t, branch_name, self._current_branch_id))
            turns = self._user_turns.get(self._current_branch_id, set())
            if t not in turns:
                raise ValueError(
                    f"Turn {t} does not contain a user message "
                    f"in branch '{self._current_branch_id}'"
                )
            self._user_turns[branch_name] = set()
            self._current_branch_id = branch_name
            return branch_name

        async def get_conversation_turns(self) -> list[dict]:
            turns = self._user_turns.get(self._current_branch_id, set())
            return [{"turn": t} for t in sorted(turns)]

    conversation.session = FakeSession()
    conversation.run_tool = SimpleNamespace(
        cwd=tmp_path,
        run=lambda **k: ("ok", {}),
        reset_runtime_state=lambda **kwargs: None,
    )
    measure_calls = {"count": 0}

    def fake_measure(**kwargs):
        measure_calls["count"] += 1
        if measure_calls["count"] == 1:
            return (0.1, 1.0, 10.0, False)
        raise RuntimeError("final measurement exploded")

    conversation._make_exec_callback = lambda qid, **_kwargs: lambda args, t: (
        "resp",
        "1 | Execution ms: 10.0",
        "",
    )
    conversation._measure_with_manifest = fake_measure
    conversation._measure_all_queries = lambda: {"1": 0.1}
    conversation._delete_result_csvs = lambda cwd: None
    conversation._check_correctness_with_scale_factors = (
        lambda qids, trace_mode, scale_factors: asyncio.sleep(0, result=True)
    )
    conversation._summarize_trace_evidence_for_queries = lambda qids: TraceEvidenceSummary(
        qids=tuple(qids),
        sufficient=True,
        message="ok",
    )
    final_summary_calls: list[dict[str, object]] = []
    conversation.wandb_run_hook = SimpleNamespace(
        log_optimization_final_summary=lambda **kwargs: final_summary_calls.append(kwargs),
        log_query_hotspot_summary=lambda **kwargs: None,
        log_ingest_summary=lambda **kwargs: None,
    )
    conversation.conv_name = "test_conv"
    conversation.artifacts_dir = tmp_path / "artifacts"
    conversation.start_snapshot_hash = "h0"
    conversation.query_rt_log = {}
    conversation.best_rt_log = {}
    conversation.global_regression_records = []

    async def dummy_exec(*args, **kwargs) -> None:
        return None

    async def dummy_check_correctness(*args, **kwargs) -> bool:
        return True

    async def dummy_run_stage(**kwargs) -> None:
        return None

    async def dummy_finish() -> list[str]:
        return []

    conversation._exec = dummy_exec
    conversation._check_correctness = dummy_check_correctness
    conversation._run_stage = dummy_run_stage
    conversation.ask_to_finish_and_save = dummy_finish

    conversation._sample_trace_for_query = lambda query_id: SimpleNamespace(
        issue_class="scan_bound",
        summary_text="trace summary",
        sampled_instantiations=("i1",),
    )
    conversation._build_query_stage = lambda **kwargs: StageConfig(
        name="trace_expert",
        get_prompt=lambda _rt: "",
        get_descriptor=lambda: "trace_expert",
        max_turns=10,
    )
    conversation._run_stage_correctness_gate = lambda **kwargs: CorrectnessCheckSummary(
        success=True,
        message="ok",
        metrics={"validation/correct": True},
        failed_scale_factor=None,
    )
    conversation._persist_hotspot_summary = lambda records: tmp_path / "hotspots.md"
    conversation._collect_query_output_split_measurements = lambda qids: {}

    async def dummy_global_human_reference(**kwargs) -> GlobalHumanReferenceResult:
        return GlobalHumanReferenceResult(
            runtime_by_query={"1": 0.1},
            written_files=(),
            accepted=True,
        )

    conversation._run_global_human_reference = dummy_global_human_reference
    _patch_tpch_monetdb_optimization_prompt_builders(monkeypatch)
    monkeypatch.setattr(
        "tpch_monetdb.conversations.optimization_conversation_tpch_monetdb.run_required_correctness_checks",
        lambda *args, **kwargs: CorrectnessCheckSummary(
            success=True,
            message="ok",
            metrics={"validation/correct": True},
            failed_scale_factor=None,
        ),
    )
    monkeypatch.setattr(
        "tpch_monetdb.utils.optimization_summary.persist_optimization_run",
        lambda **kwargs: None,
    )

    result = await conversation.run()

    assert result == []
    assert measure_calls["count"] == 2
    assert final_summary_calls == []


@pytest.mark.asyncio
async def test_optimization_run_no_longer_uses_legacy_stage_accounting(
    monkeypatch,
    tmp_path,
) -> None:
    from tpch_monetdb.conversations.optimization_conversation_tpch_monetdb import TpchMonetdbOptimizationConversation
    from tpch_monetdb.conversations.optimization_validation import CorrectnessCheckSummary

    (tmp_path / "queries.txt").write_text("1\n", encoding="utf-8")

    conversation = TpchMonetdbOptimizationConversation.__new__(TpchMonetdbOptimizationConversation)
    conversation.benchmark_sf = 10
    conversation.bespoke_storage = True
    conversation.query_ids = ["1"]
    conversation.required_validation_sf_list = [1]
    conversation.revert_on_regression = False
    conversation.regression_tolerance = 0.1
    conversation.git_snapshotter = SimpleNamespace(
        current_hash="h1",
        restore=lambda h: None,
    )
    conversation.conversation_json_path = tmp_path / "conv.json"
    conversation.callback = lambda *args, **kwargs: None
    conversation.replay = False
    conversation.notify = False
    conversation.auto_finish = True
    conversation.allowed_choices = ("u",)
    conversation.model = None
    conversation.auto_u = True
    conversation.replay_cache = False
    conversation.workspace_root = None
    conversation.used = []
    conversation.get_choice = lambda: "u"

    class FakeSession:
        def __init__(self, user_turns=None):
            self._current_branch_id = "main"
            self._user_turns = user_turns or {"main": {1}}
            self.switch_log: list[tuple] = []

        async def switch_to_branch(self, branch_name) -> None:
            self.switch_log.append(("switch_to", branch_name))
            self._current_branch_id = branch_name

        async def create_branch_from_turn(self, turn_nr, branch_name) -> str:
            self.switch_log.append(("create_branch", turn_nr, branch_name, self._current_branch_id))
            turns = self._user_turns.get(self._current_branch_id, set())
            if turn_nr not in turns:
                raise ValueError(
                    f"Turn {turn_nr} does not contain a user message "
                    f"in branch '{self._current_branch_id}'"
                )
            self._user_turns[branch_name] = set()
            self._current_branch_id = branch_name
            return branch_name

        async def get_conversation_turns(self) -> list[dict]:
            turns = self._user_turns.get(self._current_branch_id, set())
            return [{"turn": t} for t in sorted(turns)]

    conversation.session = FakeSession()
    conversation.run_tool = SimpleNamespace(
        cwd=tmp_path,
        run=lambda **kwargs: ("ok", {}),
        reset_runtime_state=lambda **kwargs: None,
    )
    conversation._make_exec_callback = lambda qid, **_kwargs: lambda args, turn_nr: (
        "resp",
        "1 | Execution ms: 10.0",
        "",
    )
    conversation._measure_with_manifest = lambda **kwargs: (0.1, 1.0, 10.0, False)
    measure_all_calls = {"count": 0}

    def fake_measure_all_queries() -> dict[str, float]:
        measure_all_calls["count"] += 1
        return {"1": 0.08}

    conversation._measure_all_queries = fake_measure_all_queries
    conversation._delete_result_csvs = lambda cwd: None
    conversation._check_correctness_with_scale_factors = (
        lambda qids, trace_mode, scale_factors: asyncio.sleep(0, result=True)
    )
    conversation._summarize_trace_evidence_for_queries = lambda qids: TraceEvidenceSummary(
        qids=tuple(qids),
        sufficient=True,
        message="ok",
    )
    conversation.wandb_run_hook = None
    conversation.conv_name = "test_conv"
    conversation.artifacts_dir = tmp_path / "artifacts"
    conversation.start_snapshot_hash = "h0"
    conversation.query_rt_log = {}
    conversation.best_rt_log = {}
    conversation.global_regression_records = []
    conversation._get_baseline_runtime_ms_by_query = lambda: {"1": 1000.0}
    conversation._refresh_query_baselines_for_stage = lambda written_files: None
    conversation._refresh_ingest_baseline_for_stage = lambda written_files: None
    conversation._log_ingest_comparison_if_complete = (
        lambda stage_name, validation_metrics: None
    )
    conversation._collect_baselines_at_checkpoint = lambda: None

    async def dummy_exec(*args, **kwargs) -> None:
        return None

    async def dummy_check_correctness(*args, **kwargs) -> bool:
        return True

    async def dummy_finish() -> list[str]:
        return []

    async def dummy_run_stage(**kwargs) -> None:
        stage_name = kwargs["stage"].name
        if stage_name == "trace_expert":
            conversation.query_rt_log[kwargs["query_id"]] = 0.1
        else:
            conversation.query_rt_log[kwargs["query_id"]] = 0.09
        return None

    conversation._exec = dummy_exec
    conversation._check_correctness = dummy_check_correctness
    conversation.ask_to_finish_and_save = dummy_finish
    conversation._run_stage = dummy_run_stage

    conversation._sample_trace_for_query = lambda query_id: SimpleNamespace(
        issue_class="scan_bound",
        summary_text="trace summary",
        sampled_instantiations=("i1",),
    )
    conversation._build_query_stage = lambda **kwargs: StageConfig(
        name="trace_expert",
        get_prompt=lambda _rt: "",
        get_descriptor=lambda: "trace_expert",
        max_turns=10,
    )
    conversation._run_stage_correctness_gate = lambda **kwargs: CorrectnessCheckSummary(
        success=True,
        message="ok",
        metrics={"validation/correct": True},
        failed_scale_factor=None,
    )
    conversation._persist_hotspot_summary = lambda records: tmp_path / "hotspots.md"
    conversation._collect_query_output_split_measurements = lambda qids: {}

    async def dummy_global_human_reference(**kwargs) -> GlobalHumanReferenceResult:
        return GlobalHumanReferenceResult(
            runtime_by_query={"1": 0.08},
            written_files=("query_impl.cpp",),
            accepted=True,
        )

    conversation._run_global_human_reference = dummy_global_human_reference
    _patch_tpch_monetdb_optimization_prompt_builders(monkeypatch)
    monkeypatch.setattr(
        "tpch_monetdb.conversations.optimization_conversation_tpch_monetdb.run_required_correctness_checks",
        lambda *args, **kwargs: CorrectnessCheckSummary(
            success=True,
            message="ok",
            metrics={"validation/correct": True},
            failed_scale_factor=None,
        ),
    )
    monkeypatch.setattr(
        "tpch_monetdb.utils.optimization_summary.persist_optimization_run",
        lambda **kwargs: None,
    )

    result = await conversation.run()

    assert result == []
    assert measure_all_calls["count"] == 1
    assert not hasattr(conversation, "_render_stage_accounting")


@pytest.mark.asyncio
async def test_global_patch_unit_salvage_keeps_non_regressed_query_unit(
    monkeypatch,
) -> None:
    """global 候选退化时应只保留未退化的 query-scoped patch unit."""
    from tpch_monetdb.conversations.optimization_conversation_tpch_monetdb import (
        GlobalOptimizationHypothesis,
        TpchMonetdbOptimizationConversation,
    )
    from tpch_monetdb.conversations.optimization_validation import CorrectnessCheckSummary

    conversation = TpchMonetdbOptimizationConversation.__new__(TpchMonetdbOptimizationConversation)
    conversation.query_ids = ["1", "9"]
    conversation.required_validation_sf_list = [1]
    conversation.regression_tolerance = 0.0
    conversation.run_tool = SimpleNamespace()
    conversation._measure_all_queries = lambda: {"1": 0.08, "9": 0.1}
    conversation._collect_candidate_structured_gate_failures = (
        lambda **_kwargs: ((), (), (), {"1": 1.25})
    )

    class FakeSnapshotter:
        def __init__(self) -> None:
            self.current_hash = "base"
            self.restores: list[str] = []
            self.checkout_calls: list[tuple[str, tuple[str, ...]]] = []

        def restore(self, snapshot: str) -> None:
            self.restores.append(snapshot)
            self.current_hash = snapshot
            return None

        def checkout_paths_from_snapshot(
            self,
            commit_hash: str,
            paths: list[str],
        ) -> None:
            self.checkout_calls.append((commit_hash, tuple(paths)))
            return None

        def snapshot(self, _name: str) -> tuple[str, str]:
            self.current_hash = "salvaged"
            return "parent", "salvaged"

    snapshotter = FakeSnapshotter()
    conversation.git_snapshotter = snapshotter
    monkeypatch.setattr(
        "tpch_monetdb.conversations.optimization_conversation_tpch_monetdb.run_required_correctness_checks",
        lambda *args, **kwargs: CorrectnessCheckSummary(
            success=True,
            message="ok",
            metrics={"validation/correct": True},
            failed_scale_factor=None,
        ),
    )

    candidate = await conversation._try_salvage_global_patch_units(
        hypothesis=GlobalOptimizationHypothesis(
            id="h1",
            summary="split q1/q9",
            evidence=("trace:q1", "trace:q9"),
            affected_queries=("1", "9"),
        ),
        base_snapshot="base",
        candidate_snapshot="candidate",
        written_files=("query_q1.cpp", "query_q9.cpp"),
        before_rt_log={"1": 0.1, "9": 0.1},
        after_rt_log={"1": 0.08, "9": 0.2},
        regressed_queries=("9",),
    )

    assert candidate is not None
    assert candidate.partial is True
    assert candidate.written_files == ("query_q1.cpp",)
    assert [result.unit.unit_id for result in candidate.accepted_units] == ["h1:q1"]
    assert [result.unit.unit_id for result in candidate.rejected_units] == ["h1:q9"]
    assert snapshotter.checkout_calls == [("candidate", ("query_q1.cpp",))]


@pytest.mark.asyncio
async def test_optimization_run_keeps_local_result_when_global_candidates_rejected(
    monkeypatch,
    tmp_path,
) -> None:
    """global 候选全拒绝时不应把已通过的 local 优化整轮标失败。"""
    from tpch_monetdb.conversations.optimization_conversation_tpch_monetdb import (
        GlobalHumanReferenceAttempt,
        TpchMonetdbOptimizationConversation,
    )
    from tpch_monetdb.conversations.optimization_validation import CorrectnessCheckSummary

    _write_global_human_reference_control_artifacts(tmp_path)
    (tmp_path / "queries.txt").write_text("1\n", encoding="utf-8")
    conversation = TpchMonetdbOptimizationConversation.__new__(TpchMonetdbOptimizationConversation)
    conversation.benchmark_sf = 10
    conversation.large_sf = 10
    conversation.bespoke_storage = True
    conversation.query_ids = ["1"]
    conversation.required_validation_sf_list = [1]
    conversation.revert_on_regression = False
    conversation.regression_tolerance = 0.1
    conversation.git_snapshotter = SimpleNamespace(
        current_hash="h1",
        restore=lambda h: None,
    )
    conversation.conversation_json_path = tmp_path / "conv.json"
    conversation.callback = lambda *args, **kwargs: None
    conversation.replay = False
    conversation.notify = False
    conversation.auto_finish = True
    conversation.allowed_choices = ("u",)
    conversation.model = None
    conversation.auto_u = True
    conversation.replay_cache = False
    conversation.workspace_root = None
    conversation.used = []
    conversation.get_choice = lambda: "u"

    class FakeSession:
        def __init__(self) -> None:
            self._current_branch_id = "main"

        async def switch_to_branch(self, branch_name) -> None:
            self._current_branch_id = branch_name
            return None

        async def create_branch_from_turn(self, turn_nr, branch_name) -> str:
            self._current_branch_id = branch_name
            return branch_name

        async def get_conversation_turns(self) -> list[dict]:
            return [{"turn": 1}]

    conversation.session = FakeSession()
    conversation.run_tool = SimpleNamespace(
        cwd=tmp_path,
        run=lambda **kwargs: ("ok", {}),
        reset_runtime_state=lambda **kwargs: None,
    )
    conversation._make_exec_callback = lambda qid, **_kwargs: lambda args, turn_nr: (
        "resp",
        "1 | Execution ms: 10.0",
        "",
    )
    conversation._measure_with_manifest = lambda **kwargs: (0.1, 1.0, 10.0, False)
    conversation._measure_all_queries = lambda: {"1": 0.1}
    conversation._delete_result_csvs = lambda cwd: None
    conversation._check_correctness_with_scale_factors = (
        lambda qids, trace_mode, scale_factors: asyncio.sleep(0, result=True)
    )
    conversation._summarize_trace_evidence_for_queries = lambda qids: TraceEvidenceSummary(
        qids=tuple(qids),
        sufficient=True,
        message="ok",
    )
    conversation.wandb_run_hook = None
    conversation.conv_name = "test_conv"
    conversation.artifacts_dir = tmp_path / "artifacts"
    conversation.start_snapshot_hash = "h0"
    conversation.query_rt_log = {}
    conversation.best_rt_log = {}
    conversation.global_regression_records = []
    conversation._get_baseline_runtime_ms_by_query = lambda: {"1": 1000.0}
    conversation._refresh_query_baselines_for_stage = lambda written_files: None
    conversation._refresh_ingest_baseline_for_stage = lambda written_files: None
    conversation._log_ingest_comparison_if_complete = (
        lambda stage_name, validation_metrics: None
    )
    conversation._collect_baselines_at_checkpoint = lambda: None

    async def dummy_exec(*args, **kwargs) -> None:
        return None

    async def dummy_check_correctness(*args, **kwargs) -> bool:
        return True

    async def dummy_finish() -> list[str]:
        return []

    async def dummy_run_stage(**kwargs) -> None:
        conversation.query_rt_log[kwargs["query_id"]] = 0.1
        return None

    conversation._exec = dummy_exec
    conversation._check_correctness = dummy_check_correctness
    conversation.ask_to_finish_and_save = dummy_finish
    conversation._run_stage = dummy_run_stage

    conversation._sample_trace_for_query = lambda query_id: SimpleNamespace(
        issue_class="scan_bound",
        summary_text="trace summary",
        sampled_instantiations=("i1",),
    )
    conversation._build_query_stage = lambda **kwargs: StageConfig(
        name="trace_expert",
        get_prompt=lambda _rt: "",
        get_descriptor=lambda: "trace_expert",
        max_turns=10,
    )
    conversation._run_stage_correctness_gate = lambda **kwargs: CorrectnessCheckSummary(
        success=True,
        message="ok",
        metrics={"validation/correct": True},
        failed_scale_factor=None,
    )
    conversation._persist_hotspot_summary = lambda records: tmp_path / "hotspots.md"
    conversation._collect_query_output_split_measurements = lambda qids: {}

    async def dummy_global_human_reference(**kwargs) -> GlobalHumanReferenceResult:
        return GlobalHumanReferenceResult(
            runtime_by_query={"1": 0.1},
            written_files=(),
            accepted=False,
            attempts=(
                GlobalHumanReferenceAttempt(
                    attempt_index=1,
                    written_files=("query_q1.cpp",),
                    accepted=False,
                    rejection_code="GLOBAL_REGRESSION",
                    rejection_detail="regressed",
                    regressed_queries=("1",),
                ),
            ),
        )

    conversation._run_global_human_reference = dummy_global_human_reference
    _patch_tpch_monetdb_optimization_prompt_builders(monkeypatch)
    monkeypatch.setattr(
        "tpch_monetdb.conversations.optimization_conversation_tpch_monetdb.run_required_correctness_checks",
        lambda *args, **kwargs: CorrectnessCheckSummary(
            success=True,
            message="ok",
            metrics={"validation/correct": True},
            failed_scale_factor=None,
        ),
    )
    monkeypatch.setattr(
        "tpch_monetdb.conversations.optimization_conversation_tpch_monetdb.build_objective_failure_report",
        lambda _summary: SimpleNamespace(failures=(), details={}),
    )

    persisted: list[dict[str, object]] = []
    monkeypatch.setattr(
        "tpch_monetdb.utils.optimization_summary.persist_optimization_run",
        lambda **kwargs: persisted.append(kwargs) or (tmp_path / "summary.json"),
    )

    result = await conversation.run()

    assert result == []
    assert persisted
    assert persisted[0]["success"] is True
    assert persisted[0]["failure_code"] is None
    assert persisted[0]["global_regression_records"] == [
        {
            "stage_name": "global_human_reference",
            "attempt_index": 1,
            "accepted": False,
            "rejection_code": "GLOBAL_REGRESSION",
            "regressed_queries": ["1"],
            "objective_failures": [],
            "failure_detail": "regressed",
        }
    ]
    return None


def test_bespoketpch_monetdb_runtime_provider_is_canonical_import() -> None:
    """验证 GeneratedTpchRuntimeProvider 是主导导入路径."""
    from tpch_monetdb.benchmark.providers import GeneratedTpchRuntimeProvider
    from tpch_monetdb.benchmark import GeneratedTpchRuntimeProvider as ExportedProvider

    assert GeneratedTpchRuntimeProvider is ExportedProvider
    provider = GeneratedTpchRuntimeProvider()
    assert provider is not None


def test_bespoke_runtime_provider_alias_emits_deprecation_warning() -> None:
    """验证 BespokeRuntimeProvider 兼容别名仍可导入但产生 DeprecationWarning."""
    import warnings

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        from tpch_monetdb.benchmark.providers import BespokeRuntimeProvider

        provider = BespokeRuntimeProvider()
        assert provider is not None

        deprecation_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert len(deprecation_warnings) >= 1
        assert "GeneratedTpchRuntimeProvider" in str(deprecation_warnings[0].message)


def test_tpch_monetdb_naming_gate_rejects_bare_bespoke() -> None:
    """验证 tpch_monetdb/ 内不会新引入裸 Bespoke 品牌标签.

    允许例外仅为 Generated TPC-H 与历史来源 BespokeOLAP。
    """
    import re

    tpch_monetdb_root = Path(__file__).resolve().parents[2] / "tpch_monetdb"
    assert tpch_monetdb_root.exists()

    forbidden_pattern = re.compile(r"\bBespoke\b(?!TPC-H MonetDB|OLAP)")
    violations: list[str] = []

    current_file = Path(__file__).resolve()
    for py_file in tpch_monetdb_root.rglob("*.py"):
        if py_file.resolve() == current_file:
            continue
        text = py_file.read_text(encoding="utf-8")
        for match in forbidden_pattern.finditer(text):
            line_num = text[: match.start()].count("\n") + 1
            violations.append(f"{py_file}:{line_num}")

    assert violations == [], f"Found bare Bespoke labels: {violations}"


def test_parse_bespoke_timing_prefers_query_ms() -> None:
    """Query ms 优先于 Execution ms 作为主 runtime."""
    from tpch_monetdb.benchmark.runtime_accounting import parse_query_timing

    stdout = "1 | Query ms: 5.123\n1 | Execution ms: 3.456"
    result = parse_query_timing(stdout, "", query_id="1")
    assert result.primary_runtime_ms == 5.123
    assert result.kernel_runtime_ms == 3.456
    assert result.fallback_reason is None


def test_parse_bespoke_timing_rejects_execution_ms_as_official() -> None:
    """缺失 Query ms 时 Execution ms 只能作为诊断 kernel_ms。"""
    from tpch_monetdb.benchmark.runtime_accounting import parse_query_timing

    stdout = "1 | Execution ms: 3.456"
    with pytest.raises(ValueError, match="Official Query ms missing"):
        parse_query_timing(stdout, "", query_id="1")


def test_parse_bespoke_timing_batch_index_skips_kernel_lines() -> None:
    """batch 模式下 index 指向第 N 个 query，而非第 N 行 timing."""
    from tpch_monetdb.benchmark.runtime_accounting import parse_query_timing

    stdout = (
        "1 | Query ms: 1.111\n1 | Execution ms: 0.555\n"
        "2 | Query ms: 2.222\n2 | Execution ms: 1.111\n"
        "3 | Query ms: 3.333\n3 | Execution ms: 1.666"
    )
    assert parse_query_timing(stdout, "", query_id="1", index=0).primary_runtime_ms == 1.111
    assert parse_query_timing(stdout, "", query_id="2", index=1).primary_runtime_ms == 2.222
    assert parse_query_timing(stdout, "", query_id="3", index=2).primary_runtime_ms == 3.333


def test_parse_bespoke_ingest_timing_splits_load_build() -> None:
    """Ingest timing 解析支持 Load ms + Build ms + Ingest ms."""
    from tpch_monetdb.benchmark.runtime_accounting import parse_ingest_timing

    stdout = "Load ms: 10.500\nBuild ms: 20.250\nIngest ms: 30.750"
    result = parse_ingest_timing(stdout, "")
    assert result.ingest_ms == 30.750
    assert result.load_ms == 10.500
    assert result.build_ms == 20.250


def test_parse_bespoke_ingest_timing_derives_from_load_build() -> None:
    """Ingest ms 缺失时自动从 Load ms + Build ms 汇总."""
    from tpch_monetdb.benchmark.runtime_accounting import parse_ingest_timing

    stdout = "Load ms: 10.500\nBuild ms: 20.250"
    result = parse_ingest_timing(stdout, "")
    assert result.ingest_ms == pytest.approx(30.750)
    assert result.load_ms == 10.500
    assert result.build_ms == 20.250


def test_parse_bespoke_ingest_timing_rejects_missing_all() -> None:
    """没有任何 ingest timing 时显式失败."""
    from tpch_monetdb.benchmark.runtime_accounting import parse_ingest_timing

    with pytest.raises(ValueError, match="No ingest timing"):
        parse_ingest_timing("", "")


def test_parse_bespoke_timing_preserves_sub_millisecond_precision() -> None:
    """亚毫秒精度必须保留（%.3f）."""
    from tpch_monetdb.benchmark.runtime_accounting import parse_query_timing

    stdout = "1 | Query ms: 0.047"
    result = parse_query_timing(stdout, "", query_id="1")
    assert result.primary_runtime_ms == 0.047


def test_questdb_baseline_provider_removed_from_providers_module() -> None:
    """providers 模块不再暴露 QuestDB HTTP baseline provider。"""
    import tpch_monetdb.benchmark.providers as providers

    assert not hasattr(providers, "QuestDBBaselineProvider")


def test_questdb_baseline_provider_removed_from_public_benchmark_api() -> None:
    """公共 benchmark 包不再导出 QuestDB HTTP baseline provider。"""
    import tpch_monetdb.benchmark as benchmark

    assert "QuestDBBaselineProvider" not in benchmark.__all__
    assert not hasattr(benchmark, "QuestDBBaselineProvider")


def test_batch3_contracts_are_registered() -> None:
    """TPC-H Q10, Q13, Q14 contracts 已注册且字段完整."""
    from tpch_monetdb.dataset.gen_tpch.tpch_queries import QUERY_CONTRACTS, get_contract

    q10 = get_contract("Q10")
    assert q10.tables == ("customer", "orders", "lineitem", "nation")
    assert "join" in q10.features
    assert "aggregation" in q10.features

    q13 = get_contract("Q13")
    assert q13.tables == ("customer", "orders")
    assert "left_outer_join" in q13.features
    assert q13.result_ordered is True

    q14 = get_contract("Q14")
    assert q14.tables == ("lineitem", "part")
    assert "ratio" in q14.features
    assert q14.parameter_names == ("DATE",)

    assert set(QUERY_CONTRACTS.keys()) >= {"Q10", "Q13", "Q14"}


def test_batch3_sampler_returns_non_empty_args() -> None:
    """TPC-H Q10, Q13, Q14 formal instantiation 返回非空 args."""
    from tpch_monetdb.dataset.gen_tpch.gen_tpch_query import instantiate_tpch_query

    args = {
        query_id: instantiate_tpch_query(query_id=query_id, scale_factor=1, seed=42)["args_string"]
        for query_id in ("10", "13", "14")
    }
    assert args["10"].startswith("Q10 ")
    assert args["13"].startswith("Q13 ")
    assert args["14"].startswith("Q14 ")


def test_batch4_contracts_are_registered() -> None:
    """TPC-H Q11, Q12, Q15 contracts 已注册且字段完整."""
    from tpch_monetdb.dataset.gen_tpch.tpch_queries import QUERY_CONTRACTS, get_contract

    q11 = get_contract("Q11")
    assert q11.tables == ("partsupp", "supplier", "nation")
    assert "having" in q11.features
    assert "aggregation" in q11.features

    q12 = get_contract("Q12")
    assert q12.tables == ("orders", "lineitem")
    assert "case" in q12.features
    assert "SHIPMODE1" in q12.parameter_names

    q15 = get_contract("Q15")
    assert q15.tables == ("lineitem", "supplier")
    assert "cte" in q15.features
    assert "max_subquery" in q15.features

    assert set(QUERY_CONTRACTS.keys()) >= {"Q11", "Q12", "Q15"}


def test_batch4_sampler_returns_non_empty_args() -> None:
    """TPC-H Q11, Q12, Q15 formal instantiation 返回非空 args."""
    from tpch_monetdb.dataset.gen_tpch.gen_tpch_query import instantiate_tpch_query

    args = {
        query_id: instantiate_tpch_query(query_id=query_id, scale_factor=1, seed=42)["args_string"]
        for query_id in ("11", "12", "15")
    }
    assert args["11"].startswith("Q11 ")
    assert args["12"].startswith("Q12 ")
    assert args["15"].startswith("Q15 ")


def test_formal_instantiation_populates_batch3_and_batch4_params() -> None:
    """TPC-H Q10-Q15 formal instantiation 应补齐参数并写入 key=value args_string。"""
    from tpch_monetdb.dataset.gen_tpch.gen_tpch_query import instantiate_tpch_query

    for query_id in ("10", "11", "12", "13", "14", "15"):
        inst = instantiate_tpch_query(query_id=query_id, scale_factor=1, seed=42)
        params = inst["params_json"]
        assert inst["query_id"] == f"Q{query_id}"
        assert inst["args_string"].startswith(f"Q{query_id}")
        for key, value in params.items():
            assert f"{key}=" in inst["args_string"], query_id
            assert value is not None


def test_formal_instantiation_replaces_all_sql_placeholders_for_batch3_and_batch4() -> None:
    from tpch_monetdb.dataset.gen_tpch.gen_tpch_query import instantiate_tpch_query

    for query_id in ("10", "11", "12", "13", "14", "15"):
        inst = instantiate_tpch_query(query_id=query_id, scale_factor=1, seed=42)
        assert "[" not in inst["sql"], query_id
        assert "]" not in inst["sql"], query_id


def test_gen_tpch_args_str_emits_all_query_parsers_for_dispatch_template() -> None:
    from tpch_monetdb.dataset.query_gen_factory import get_placeholders_fn
    from tpch_monetdb.utils.general_utils import gen_tpch_args_str

    args_str, _ = gen_tpch_args_str(["Q1"], gen_placeholders_fn=get_placeholders_fn("tpch"))

    assert "parse_q1" in args_str
    assert "parse_q10" in args_str
    assert "parse_q22" in args_str


def test_generated_query_registry_uses_generated_parsers(tmp_path) -> None:
    from tpch_monetdb.dataset.query_gen_factory import get_placeholders_fn
    from tpch_monetdb.tools.tpch.utils import copy_template_to, make_compiler
    from tpch_monetdb.utils.general_utils import gen_tpch_args_str

    copy_template_to(tmp_path, "tpch")
    args_str, _ = gen_tpch_args_str(["Q1"], gen_placeholders_fn=get_placeholders_fn("tpch"))
    (tmp_path / "args_parser.hpp").write_text(args_str, encoding="utf-8")
    for query_id in ("10", "15"):
        (tmp_path / f"query_q{query_id}.hpp").write_text(
            "\n".join(
                [
                    "#pragma once",
                    '#include "builder_impl.hpp"',
                    '#include "args_parser.hpp"',
                    f"void execute_q{query_id}(Engine& engine, const Q{query_id}Args& args);",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        (tmp_path / f"query_q{query_id}.cpp").write_text(
            "\n".join(
                [
                    f'#include "query_q{query_id}.hpp"',
                    f"void execute_q{query_id}(Engine& engine, const Q{query_id}Args& args) {{",
                    "    (void)engine; (void)args;",
                    "}",
                    "",
                ]
            ),
            encoding="utf-8",
        )

    make_compiler(tmp_path)

    query_impl = (
        ROOT / "misc" / "tpch" / "templates" / "query_impl.cpp"
    ).read_text(encoding="utf-8")
    registry = (tmp_path / "build" / "generated" / "query_registry_generated.cpp").read_text(
        encoding="utf-8"
    )

    assert '#include "query_registry_generated.hpp"' in query_impl
    assert "dispatch_query(*engine, request);" in query_impl
    assert "parse_q10(request)" in registry
    assert "parse_q15(request)" in registry
    assert "execute_q10(engine, args)" in registry
    assert "execute_q15(engine, args)" in registry
    assert "query_q10::run" not in registry
    assert "parse_host_list" not in registry
    assert "read_quoted" not in registry


def test_compile_tool_rebuilds_compiler_before_each_call(tmp_path, monkeypatch) -> None:
    from tpch_monetdb.tools.tpch import compile as compile_module

    created: list[object] = []

    class FakeCompiler:
        def set_extra_cxxflags(self, _flags) -> None:
            return None

        def build(self):
            return None

    def fake_make_compiler(*_args, **_kwargs):
        compiler = FakeCompiler()
        created.append(compiler)
        return compiler

    monkeypatch.setattr(compile_module, "make_compiler", fake_make_compiler)

    tool = compile_module.CompileTool(tmp_path)
    assert len(created) == 1

    result = tool(optimize=False)

    assert result == "**Compilation successfull**"
    assert len(created) == 2


def test_run_tool_rebuilds_compiler_before_build_cached(tmp_path, monkeypatch) -> None:
    from tpch_monetdb.tools.tpch import run as run_module

    created: list[object] = []

    class FakeCompiler:
        def set_extra_cxxflags(self, _flags) -> None:
            return None

        def build_cached(self, **_kwargs):
            return "compile failed", False, "hash"

    def fake_make_compiler(*_args, **_kwargs):
        compiler = FakeCompiler()
        created.append(compiler)
        return compiler

    monkeypatch.setattr(run_module, "make_compiler", fake_make_compiler)

    tool = run_module.RunTool(
        cwd=tmp_path,
        dataset_name="tpch",
        base_data_dir=str(tmp_path),
        query_validator=None,
        wandb_metrics_hook=None,
        compile_cache_dir=None,
        git_snapshotter=None,
    )
    assert len(created) == 1

    result = tool.run_worker(
        scale_factor=1,
        optimize=False,
        stdin_args_data=["1"],
    )

    assert result.msg == "compile failed"
    assert len(created) == 2


def test_run_tool_uses_release_safe_flags_when_optimize_false(tmp_path, monkeypatch) -> None:
    from tpch_monetdb.tools.tpch import run as run_module

    captured_flags: list[str] = []

    class FakeCompiler:
        def set_extra_cxxflags(self, flags) -> None:
            captured_flags.extend(flags)
            return None

        def build_cached(self, **_kwargs):
            return "compile failed", False, "hash"

    monkeypatch.setattr(run_module, "make_compiler", lambda *_args, **_kwargs: FakeCompiler())

    tool = run_module.RunTool(
        cwd=tmp_path,
        dataset_name="tpch",
        base_data_dir=str(tmp_path),
        query_validator=None,
        wandb_metrics_hook=None,
        compile_cache_dir=None,
        git_snapshotter=None,
    )

    result = tool.run_worker(
        scale_factor=1,
        optimize=False,
        stdin_args_data=["10"],
    )

    assert result.msg == "compile failed"
    assert "-O2" in captured_flags
    assert "-O3" not in captured_flags
    assert "-flto" not in captured_flags
    return None


def test_run_tool_validator_infra_failure_returns_structured_metrics(
    tmp_path,
    monkeypatch,
) -> None:
    from tpch_monetdb.misc.tpch.fasttest_proc import RunnerInfraFailureError
    from tpch_monetdb.tools.tpch import run as run_module

    class FakeCompiler:
        def set_extra_cxxflags(self, _flags) -> None:
            return None

        def build_cached(self, **_kwargs):
            return None, False, "hash"

    class FakeValidator:
        sf_list = [1]

        def exec_and_validate(self, **_kwargs):
            raise RunnerInfraFailureError("RUNNER_TIMEOUT", "timed out")

    monkeypatch.setattr(run_module, "make_compiler", lambda *_args, **_kwargs: FakeCompiler())

    tool = run_module.RunTool(
        cwd=tmp_path,
        dataset_name="tpch",
        base_data_dir=str(tmp_path),
        query_validator=FakeValidator(),
        wandb_metrics_hook=None,
        compile_cache_dir=None,
        git_snapshotter=None,
    )

    result = tool.run_worker(
        scale_factor=1,
        optimize=False,
        query_id=["10"],
    )

    assert "[ERROR:RUNNER_TIMEOUT]" in result.msg
    assert result.metrics["validation/failure_code"] == "RUNNER_TIMEOUT"
    assert result.metrics["validation/failure_detail"] == "timed out"
    assert result.metrics["validation/fasttest_optimize"] is False
    return None


def test_run_tool_adds_target_cpu_and_vectorization_flags(tmp_path, monkeypatch) -> None:
    from tpch_monetdb.tools.tpch import run as run_module

    captured_flags: list[str] = []

    class FakeCompiler:
        def set_extra_cxxflags(self, flags) -> None:
            captured_flags.extend(flags)
            return None

        def build_cached(self, **_kwargs):
            return "compile failed", False, "hash"

    monkeypatch.setattr(run_module, "make_compiler", lambda *_args, **_kwargs: FakeCompiler())

    tool = run_module.RunTool(
        cwd=tmp_path,
        dataset_name="tpch",
        base_data_dir=str(tmp_path),
        query_validator=None,
        wandb_metrics_hook=None,
        compile_cache_dir=None,
        git_snapshotter=None,
        target_cpu="icelake",
        emit_vectorization_reports=True,
    )

    result = tool.run_worker(
        scale_factor=1,
        optimize=True,
        stdin_args_data=["1"],
    )

    assert result.msg == "compile failed"
    assert "-O3" in captured_flags
    assert "-flto" in captured_flags
    assert "-O2" not in captured_flags
    assert "-march=icelake" in captured_flags
    assert any(
        flag.startswith("-fopt-info-vec-optimized=") for flag in captured_flags
    )
    assert any(
        flag.startswith("-fopt-info-vec-missed=") for flag in captured_flags
    )
    return None


def test_run_raw_worker_requires_explicit_stdin_args(tmp_path, monkeypatch) -> None:
    from tpch_monetdb.tools.tpch import run as run_module

    class FakeCompiler:
        def set_extra_cxxflags(self, _flags) -> None:
            return None

        def build_cached(self, **_kwargs):
            return None, False, "hash"

    monkeypatch.setattr(run_module, "make_compiler", lambda *_args, **_kwargs: FakeCompiler())
    tool = run_module.RunTool(
        cwd=tmp_path,
        dataset_name="tpch",
        base_data_dir=str(tmp_path),
        query_validator=None,
        wandb_metrics_hook=None,
        compile_cache_dir=None,
        git_snapshotter=None,
    )

    with pytest.raises(RuntimeError, match="requires stdin_args_data"):
        tool.run_raw_worker(scale_factor=1, optimize=False)
    return None


def test_run_raw_worker_bypasses_validator_cache(tmp_path, monkeypatch) -> None:
    from tpch_monetdb.tools.tpch import run as run_module

    captured: dict[str, object] = {}

    class FakeCompiler:
        def set_extra_cxxflags(self, _flags) -> None:
            return None

        def build_cached(self, **_kwargs):
            return None, False, "hash"

    class FakeValidator:
        sf_list = [1]

        def exec_and_validate(self, **_kwargs):
            raise AssertionError("validator should not be called")

    class FakeRunner:
        def __init__(self) -> None:
            self.batches: list[list[str]] = []

        def run_batch(
            self,
            args_list: list[str],
            timeout: int,
        ) -> tuple[str, str, str]:
            self.batches.append(list(args_list))
            captured["timeout"] = timeout
            return "resp", "out", "err"

    fake_runner = FakeRunner()
    monkeypatch.setattr(run_module, "make_compiler", lambda *_args, **_kwargs: FakeCompiler())
    monkeypatch.setattr(run_module.FastTestPool, "get", lambda _cmd, _factory: fake_runner)

    tool = run_module.RunTool(
        cwd=tmp_path,
        dataset_name="tpch",
        base_data_dir=str(tmp_path),
        query_validator=FakeValidator(),
        wandb_metrics_hook=None,
        compile_cache_dir=None,
        git_snapshotter=None,
    )

    result = tool.run_raw_worker(
        scale_factor=1,
        optimize=False,
        query_id=["1"],
        trace_mode=True,
        stdin_args_data=["explicit args"],
    )

    assert fake_runner.batches == [["explicit args"]]
    assert result.resp == "resp"
    assert result.out == "out"
    assert result.err == "err"
    assert result.metrics is None
    return None


def test_main_tpch_monetdb_context_too_large_envelope_is_structured() -> None:
    envelope = tpch_monetdb.main_tpch_monetdb._context_too_large_envelope(
        prompt_index=7,
        descriptor="Trace->File",
        detail="413 Request Entity Too Large",
    )
    text = str(envelope)
    assert "[ERROR:CONTEXT_TOO_LARGE]" in text
    assert "Trace->File" in text
    assert "Recoverable: no" in text
    return None


@pytest.mark.asyncio
async def test_run_stage_preserves_written_files_for_routing(
    monkeypatch,
    tmp_path,
) -> None:
    """_run_stage 应保留 stage summary 的 written_files 供 routing 使用。"""
    from tpch_monetdb.conversations.optimization_conversation_tpch_monetdb import TpchMonetdbOptimizationConversation
    from tpch_monetdb.conversations.optimization_validation import CorrectnessCheckSummary
    from tpch_monetdb.tools.stage_tool_policy import StageRunSummary

    conversation = TpchMonetdbOptimizationConversation.__new__(TpchMonetdbOptimizationConversation)
    conversation.benchmark_sf = 10
    conversation.bespoke_storage = True
    conversation.git_snapshotter = SimpleNamespace(
        current_hash="h1",
        restore=lambda _hash: None,
    )
    conversation.query_ids = ["1"]
    conversation.required_validation_sf_list = [1]
    conversation.revert_on_regression = False
    conversation.query_rt_log = {}
    conversation.best_rt_log = {}
    conversation.wandb_run_hook = None
    conversation.run_tool = SimpleNamespace(run=lambda **_kwargs: ("ok", {}))
    conversation._measure_with_manifest = lambda **_kwargs: (0.1, 1.0, 10.0, False)

    async def fake_exec(*_args, **_kwargs) -> StageRunSummary:
        return StageRunSummary(
            profile_name="optimization_general",
            prompt_index=0,
            prompt_descriptor="trace",
            final_output="done",
            tool_counts={"edit_file": 1},
            written_files=("query_impl.cpp",),
            last_compile_summary=None,
            last_run_summary=None,
            todo_before=None,
            todo_after=None,
        )

    conversation._exec = fake_exec

    monkeypatch.setattr(
        "tpch_monetdb.conversations.optimization_conversation_tpch_monetdb.run_required_correctness_checks",
        lambda *args, **kwargs: CorrectnessCheckSummary(
            success=True,
            message="ok",
            metrics={"validation/correct": True},
            failed_scale_factor=None,
        ),
    )
    stage = StageConfig(
        name="trace_expert",
        get_prompt=lambda _rt: "",
        get_descriptor=lambda: "trace_expert",
        max_turns=10,
    )

    result = await conversation._run_stage(
        query_id="1",
        stage=stage,
        pretext_optim="",
        rt_before_s=0.5,
    )

    assert result.failed is False
    assert result.written_files == ("query_impl.cpp",)


@pytest.mark.asyncio
async def test_run_stage_lazy_suspected_skips_speedup_telemetry(
    monkeypatch,
) -> None:
    """lazy-build 可疑时仍记 stage telemetry，但不得发正式 no-CSV speedup telemetry。"""
    from tpch_monetdb.conversations.optimization_conversation_tpch_monetdb import TpchMonetdbOptimizationConversation
    from tpch_monetdb.conversations.optimization_validation import CorrectnessCheckSummary
    from tpch_monetdb.tools.stage_tool_policy import StageRunSummary

    conversation = TpchMonetdbOptimizationConversation.__new__(TpchMonetdbOptimizationConversation)
    conversation.benchmark_sf = 10
    conversation.bespoke_storage = True
    conversation.git_snapshotter = SimpleNamespace(
        current_hash="h1",
        restore=lambda _hash: None,
    )
    conversation.query_ids = ["1"]
    conversation.required_validation_sf_list = [1]
    conversation.revert_on_regression = False
    conversation.query_rt_log = {}
    conversation.best_rt_log = {}
    conversation.run_tool = SimpleNamespace(run=lambda **_kwargs: ("ok", {}))
    conversation._measure_with_manifest = lambda **_kwargs: (0.1, 1.0, 10.0, True)

    hook_calls: dict[str, object] = {"stage": 0, "speedup": 0, "speedup_kwargs": None}

    class FakeHook:
        def log_optimization_stage(self, **kwargs) -> None:
            hook_calls["stage"] = int(hook_calls["stage"]) + 1
            return None

        def log_optimization_speedup_vs_baseline(self, **kwargs) -> None:
            hook_calls["speedup"] = int(hook_calls["speedup"]) + 1
            hook_calls["speedup_kwargs"] = kwargs
            return None

    conversation.wandb_run_hook = FakeHook()

    async def fake_exec(*_args, **_kwargs) -> StageRunSummary:
        return StageRunSummary(
            profile_name="optimization_general",
            prompt_index=0,
            prompt_descriptor="trace",
            final_output="done",
            tool_counts={"edit_file": 1},
            written_files=("query_impl.cpp",),
            last_compile_summary=None,
            last_run_summary=None,
            todo_before=None,
            todo_after=None,
        )

    conversation._exec = fake_exec

    monkeypatch.setattr(
        "tpch_monetdb.conversations.optimization_conversation_tpch_monetdb.run_required_correctness_checks",
        lambda *args, **kwargs: CorrectnessCheckSummary(
            success=True,
            message="ok",
            metrics={"validation/correct": True},
            failed_scale_factor=None,
        ),
    )
    stage = StageConfig(
        name="trace_expert",
        get_prompt=lambda _rt: "",
        get_descriptor=lambda: "trace_expert",
        max_turns=10,
    )

    result = await conversation._run_stage(
        query_id="1",
        stage=stage,
        pretext_optim="",
        rt_before_s=0.5,
    )

    assert result.failed is False
    assert hook_calls["stage"] == 1
    assert hook_calls["speedup"] == 0
    assert conversation.query_rt_log == {}
    assert conversation.best_rt_log == {}


@pytest.mark.asyncio
async def test_run_stage_non_lazy_emits_speedup_telemetry(
    monkeypatch,
) -> None:
    """正常测量时应继续发正式 no-CSV speedup telemetry。"""
    from tpch_monetdb.conversations.optimization_conversation_tpch_monetdb import TpchMonetdbOptimizationConversation
    from tpch_monetdb.conversations.optimization_validation import CorrectnessCheckSummary
    from tpch_monetdb.tools.stage_tool_policy import StageRunSummary

    conversation = TpchMonetdbOptimizationConversation.__new__(TpchMonetdbOptimizationConversation)
    conversation.benchmark_sf = 10
    conversation.bespoke_storage = True
    conversation.git_snapshotter = SimpleNamespace(
        current_hash="h1",
        restore=lambda _hash: None,
    )
    conversation.query_ids = ["1"]
    conversation.required_validation_sf_list = [1]
    conversation.revert_on_regression = False
    conversation.query_rt_log = {}
    conversation.best_rt_log = {}
    conversation.run_tool = SimpleNamespace(run=lambda **_kwargs: ("ok", {}))
    conversation._measure_with_manifest = lambda **_kwargs: (0.1, 1.0, 10.0, False)

    conversation.benchmark = "tpch"
    conversation.baseline_provider = SimpleNamespace(engine="monetdb")

    hook_calls: dict[str, object] = {"stage": 0, "speedup": 0, "speedup_kwargs": None}

    class FakeHook:
        def log_optimization_stage(self, **kwargs) -> None:
            hook_calls["stage"] = int(hook_calls["stage"]) + 1
            return None

        def log_optimization_speedup_vs_baseline(self, **kwargs) -> None:
            hook_calls["speedup"] = int(hook_calls["speedup"]) + 1
            hook_calls["speedup_kwargs"] = kwargs
            return None

    conversation.wandb_run_hook = FakeHook()

    async def fake_exec(*_args, **_kwargs) -> StageRunSummary:
        return StageRunSummary(
            profile_name="optimization_general",
            prompt_index=0,
            prompt_descriptor="trace",
            final_output="done",
            tool_counts={"edit_file": 1},
            written_files=("query_impl.cpp",),
            last_compile_summary=None,
            last_run_summary=None,
            todo_before=None,
            todo_after=None,
        )

    conversation._exec = fake_exec

    monkeypatch.setattr(
        "tpch_monetdb.conversations.optimization_conversation_tpch_monetdb.run_required_correctness_checks",
        lambda *args, **kwargs: CorrectnessCheckSummary(
            success=True,
            message="ok",
            metrics={"validation/correct": True},
            failed_scale_factor=None,
        ),
    )
    stage = StageConfig(
        name="trace_expert",
        get_prompt=lambda _rt: "",
        get_descriptor=lambda: "trace_expert",
        max_turns=10,
    )

    result = await conversation._run_stage(
        query_id="1",
        stage=stage,
        pretext_optim="",
        rt_before_s=0.5,
    )

    assert result.failed is False
    assert hook_calls["stage"] == 1
    assert hook_calls["speedup"] == 1
    speedup_kwargs = hook_calls["speedup_kwargs"]
    assert isinstance(speedup_kwargs, dict)
    assert speedup_kwargs["baseline_engine"] == "monetdb"
    assert speedup_kwargs["baseline_label"] == "MonetDB"
    assert conversation.query_rt_log == {"1": 0.1}
    assert conversation.best_rt_log == {"1": 0.1}


def test_runtime_measurement_carries_benchmark_dimensions() -> None:
    """RuntimeMeasurement 携带 benchmark_mode / storage_mode / workers / engine 维度."""
    from tpch_monetdb.benchmark.manifest import RuntimeMeasurement

    m = RuntimeMeasurement(
        instantiation_id="i1",
        runtime_ms=10.0,
        num_runs=3,
        all_runtimes_ms=[9.0, 10.0, 11.0],
        timestamp="2024-01-01T00:00:00Z",
        benchmark_mode="system-parity",
        storage_mode="persistent",
        workers=1,
        engine="generated_tpch",
    )
    assert m.benchmark_mode == "system-parity"
    assert m.storage_mode == "persistent"
    assert m.workers == 1
    assert m.engine == "generated_tpch"


def test_bespoke_runtime_provider_emits_runtime_metadata() -> None:
    """GeneratedTpchRuntimeProvider 应回填 phase9 metadata."""
    from tpch_monetdb.benchmark.manifest import QueryInstantiation
    from tpch_monetdb.benchmark.providers import GeneratedTpchRuntimeProvider
    from tpch_monetdb.benchmark.runtime_accounting import (
        KERNEL_RUNTIME_METRIC_KIND,
        QUERY_RUNTIME_METRIC_KIND,
    )

    provider = GeneratedTpchRuntimeProvider(
        benchmark_mode="query-latency",
        storage_mode="tmpfs",
    )
    inst = QueryInstantiation(
        query_id="1",
        scale_factor=1,
        instantiation_id="inst-q1",
        params_json={"hostnames": ("host_0",)},
        args_string="1 ('host_0')",
        sql="SELECT 1",
        sql_hash="abc",
    )

    observed_timeouts: list[int] = []

    def fake_exec_callback(_args_list, timeout_s):
        observed_timeouts.append(timeout_s)
        return "", "1 | Query ms: 1.250\n1 | Execution ms: 0.750\n", ""

    measurement = provider.measure(inst, fake_exec_callback)
    assert measurement.runtime_ms == 1.25
    assert measurement.benchmark_mode == "query-latency"
    assert measurement.storage_mode == "tmpfs"
    assert measurement.workers == 1
    assert measurement.engine == "generated_tpch"
    assert measurement.provenance["runtime_metric_kind"] == QUERY_RUNTIME_METRIC_KIND
    assert measurement.provenance["kernel_runtime_metric_kind"] == KERNEL_RUNTIME_METRIC_KIND
    assert measurement.provenance["query_runs_ms"] == [1.25, 1.25, 1.25]
    assert measurement.provenance["kernel_runs_ms"] == [0.75, 0.75, 0.75]
    assert "fallback_reasons" not in measurement.provenance
    assert observed_timeouts[0] >= 180
    assert observed_timeouts[1:] == [60, 60, 60]
    assert measurement.provenance["timeout_policy"]["cold_start_timeout_s"] >= 180
    assert measurement.provenance["timeout_policy"]["warm_query_timeout_s"] == 60
    assert measurement._query_samples.measured_runs_ms == [1.25, 1.25, 1.25]
    assert measurement._query_samples.kernel_runs_ms == [0.75, 0.75, 0.75]
    return None


def test_bespoke_runtime_provider_rejects_missing_official_query_metric() -> None:
    """GeneratedTpchRuntimeProvider 缺 Query ms 时 measurement 无效。"""
    from tpch_monetdb.benchmark.manifest import QueryInstantiation
    from tpch_monetdb.benchmark.providers import GeneratedTpchRuntimeProvider

    provider = GeneratedTpchRuntimeProvider()
    inst = QueryInstantiation(
        query_id="1",
        scale_factor=1,
        instantiation_id="inst-q1",
        params_json={"hostnames": ("host_0",)},
        args_string="1 ('host_0')",
        sql="SELECT 1",
        sql_hash="abc",
    )

    def fake_exec_callback(_args_list, _timeout_s):
        return "", "1 | Execution ms: 0.750\n", ""

    with pytest.raises(ValueError, match="Official Query ms missing"):
        provider.measure(inst, fake_exec_callback)
    return None


def test_bespoke_runtime_provider_accepts_no_csv_kernel_primary() -> None:
    """Optimization 路径应允许 no_output 的 Execution ms 作为 primary runtime。"""
    from tpch_monetdb.benchmark.manifest import QueryInstantiation
    from tpch_monetdb.benchmark.providers import GeneratedTpchRuntimeProvider
    from tpch_monetdb.benchmark.runtime_accounting import KERNEL_RUNTIME_METRIC_KIND

    provider = GeneratedTpchRuntimeProvider()
    inst = QueryInstantiation(
        query_id="1",
        scale_factor=1,
        instantiation_id="inst-q1",
        params_json={"hostnames": ("host_0",)},
        args_string="1 ('host_0')",
        sql="SELECT 1",
        sql_hash="abc",
    )

    def fake_exec_callback(_args_list, _timeout_s):
        return "", "1 | Execution ms: 0.750\n", ""

    measurement = provider.measure(
        inst,
        fake_exec_callback,
        primary_metric_kind=KERNEL_RUNTIME_METRIC_KIND,
    )

    assert measurement.runtime_ms == 0.75
    assert measurement.provenance["runtime_metric_kind"] == KERNEL_RUNTIME_METRIC_KIND
    assert measurement.provenance["primary_runs_ms"] == [0.75, 0.75, 0.75]
    return None


def test_bespoke_runtime_provider_rejects_mixed_metric_kinds() -> None:
    """同一 measured set 内只要缺 Query ms 就应拒绝。"""
    from tpch_monetdb.benchmark.manifest import QueryInstantiation
    from tpch_monetdb.benchmark.providers import GeneratedTpchRuntimeProvider

    provider = GeneratedTpchRuntimeProvider()
    inst = QueryInstantiation(
        query_id="1",
        scale_factor=1,
        instantiation_id="inst-q1",
        params_json={"hostnames": ("host_0",)},
        args_string="1 ('host_0')",
        sql="SELECT 1",
        sql_hash="abc",
    )
    outputs = iter(
        [
            "1 | Query ms: 1.000\n1 | Execution ms: 0.500\n",
            "1 | Query ms: 1.100\n1 | Execution ms: 0.550\n",
            "1 | Execution ms: 0.600\n",
            "1 | Query ms: 1.200\n1 | Execution ms: 0.650\n",
        ]
    )

    def fake_exec_callback(_args_list, _timeout_s):
        return "", next(outputs), ""

    with pytest.raises(ValueError, match="Official Query ms missing"):
        provider.measure(inst, fake_exec_callback)
    return None


def test_generated_runtime_provider_rejects_timeout_before_timing_parse() -> None:
    """Provider must not parse stale timing from a timed-out runner response."""
    from tpch_monetdb.benchmark.manifest import QueryInstantiation
    from tpch_monetdb.benchmark.providers import GeneratedTpchRuntimeProvider
    from tpch_monetdb.benchmark.runtime_accounting import RuntimeExecutionFailureError

    provider = GeneratedTpchRuntimeProvider()
    inst = QueryInstantiation(
        query_id="10",
        scale_factor=1,
        instantiation_id="inst-q10",
        params_json={},
        args_string="10",
        sql="SELECT 1",
        sql_hash="abc",
    )

    def fake_exec_callback(_args_list, _timeout_s):
        return (
            "Terminated after 30 seconds due to timeout",
            "10 | Execution ms: 12.000\n",
            "",
        )

    with pytest.raises(RuntimeExecutionFailureError, match="RUNNER_TIMEOUT"):
        provider.measure(inst, fake_exec_callback)
    return None


def test_optimization_exec_callback_passes_provider_timeout_to_run_worker() -> None:
    """Manifest provider timeout must reach the manual-stdin run_worker path."""
    from tpch_monetdb.conversations.optimization_conversation_tpch_monetdb import (
        TpchMonetdbOptimizationConversation,
    )

    conversation = TpchMonetdbOptimizationConversation.__new__(
        TpchMonetdbOptimizationConversation
    )
    conversation.benchmark_sf = 1
    captured: dict[str, object] = {}

    def fake_run_worker(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            resp="exit_code: 0 signal: 0\n",
            out="1 | Execution ms: 1.0\n",
            err="",
            msg="ok",
            metrics=None,
        )

    conversation.run_tool = SimpleNamespace(run_worker=fake_run_worker)
    callback = conversation._make_exec_callback(
        "1",
        output_mode="no_output",
    )

    callback(["1"], 123)

    assert captured["execution_timeout_s"] == 123
    assert captured["stdin_args_data"] == ["1"]
    return None


def test_optimization_exec_callback_rejects_structured_runner_failure() -> None:
    """Structured run_worker failure metrics must stop provider measurement."""
    from tpch_monetdb.conversations.optimization_conversation_tpch_monetdb import (
        TpchMonetdbOptimizationConversation,
    )

    conversation = TpchMonetdbOptimizationConversation.__new__(
        TpchMonetdbOptimizationConversation
    )
    conversation.benchmark_sf = 1

    def fake_run_worker(**_kwargs):
        return SimpleNamespace(
            resp="",
            out="1 | Execution ms: 1.0\n",
            err="[ERROR:RUNNER_TIMEOUT] timed out",
            msg="[ERROR:RUNNER_TIMEOUT] timed out",
            metrics={
                "validation/failure_code": "RUNNER_TIMEOUT",
                "validation/failure_detail": "timed out",
            },
        )

    conversation.run_tool = SimpleNamespace(run_worker=fake_run_worker)
    callback = conversation._make_exec_callback(
        "1",
        output_mode="no_output",
    )

    with pytest.raises(RuntimeError, match="RUNNER_TIMEOUT"):
        callback(["1"], 123)
    return None


def test_questdb_baseline_provider_symbol_cannot_be_imported() -> None:
    """QuestDBBaselineProvider 符号删除后不能再被直接导入。"""
    with pytest.raises(ImportError):
        from tpch_monetdb.benchmark.providers import QuestDBBaselineProvider  # noqa: F401


def test_manifest_rejects_incompatible_runtime_dimensions(tmp_path: Path) -> None:
    """manifest 禁止同一 instantiation_id 的跨模式混写."""
    from tpch_monetdb.benchmark.manifest import ReferenceManifest, RuntimeMeasurement

    manifest = ReferenceManifest(tmp_path / "reference_manifest.json")
    manifest.record_runtime(
        RuntimeMeasurement(
            instantiation_id="inst-1",
            runtime_ms=10.0,
            num_runs=3,
            all_runtimes_ms=[10.0],
            timestamp="2024-01-01T00:00:00Z",
            benchmark_mode="system-parity",
            storage_mode="persistent",
            workers=1,
            engine="questdb",
        )
    )

    with pytest.raises(ValueError, match="benchmark_mode"):
        manifest.record_runtime(
            RuntimeMeasurement(
                instantiation_id="inst-1",
                runtime_ms=9.0,
                num_runs=3,
                all_runtimes_ms=[9.0],
                timestamp="2024-01-01T00:00:01Z",
                benchmark_mode="query-latency",
                storage_mode="persistent",
                workers=1,
                engine="questdb",
            )
        )


def test_manifest_upgrades_legacy_runtime_metadata(tmp_path: Path) -> None:
    """manifest 允许从 legacy None metadata 升级到显式 phase9 维度."""
    from tpch_monetdb.benchmark.manifest import ReferenceManifest, RuntimeMeasurement

    manifest = ReferenceManifest(tmp_path / "reference_manifest.json")
    manifest.record_runtime(
        RuntimeMeasurement(
            instantiation_id="inst-1",
            runtime_ms=10.0,
            num_runs=1,
            all_runtimes_ms=[10.0],
            timestamp="2024-01-01T00:00:00Z",
        )
    )
    manifest.record_runtime(
        RuntimeMeasurement(
            instantiation_id="inst-1",
            runtime_ms=9.0,
            num_runs=1,
            all_runtimes_ms=[9.0],
            timestamp="2024-01-01T00:00:01Z",
            benchmark_mode="system-parity",
            storage_mode="persistent",
            workers=1,
            engine="questdb",
        )
    )

    stored = manifest.get_runtime("inst-1")
    assert stored is not None
    assert stored.runtime_ms == 9.0
    assert stored.benchmark_mode == "system-parity"
    assert stored.storage_mode == "persistent"
    assert stored.workers == 1
    assert stored.engine == "questdb"


def test_manifest_lookup_runtime_detects_compatible_and_stale_entries(
    tmp_path: Path,
) -> None:
    """lookup_runtime 应区分 compatible 与 stale cached baseline。"""
    from tpch_monetdb.benchmark.manifest import ReferenceManifest, RuntimeMeasurement

    manifest = ReferenceManifest(tmp_path / "reference_manifest.json")
    manifest.record_runtime(
        RuntimeMeasurement(
            instantiation_id="inst-1",
            runtime_ms=10.0,
            num_runs=1,
            all_runtimes_ms=[10.0],
            timestamp="2024-01-01T00:00:00Z",
            benchmark_mode="system-parity",
            storage_mode="persistent",
            workers=1,
            engine="questdb",
        )
    )

    compatible = manifest.lookup_runtime(
        "inst-1",
        benchmark_mode="system-parity",
        storage_mode="persistent",
        workers=1,
        engine="questdb",
    )
    stale = manifest.lookup_runtime(
        "inst-1",
        benchmark_mode="query-latency",
        storage_mode="persistent",
        workers=1,
        engine="questdb",
    )
    stale_by_start_time = manifest.lookup_runtime(
        "inst-1",
        benchmark_mode="system-parity",
        storage_mode="persistent",
        workers=1,
        engine="questdb",
        baseline_run_started_at="2024-01-02T00:00:00Z",
    )

    assert compatible.status == "compatible"
    assert compatible.measurement is not None
    assert stale.status == "stale"
    assert stale.measurement is not None
    assert stale_by_start_time.status == "stale"
    assert stale_by_start_time.reason == "timestamp_before_run_start"


def test_build_run_config_rejects_legacy_tpch_monetdb_benchmark() -> None:
    """Legacy benchmark should be rejected after TPC-H replacement."""
    from tpch_monetdb.utils.cli_config import build_run_config

    with pytest.raises(ValueError, match="Unknown benchmark"):
        build_run_config(
            benchmark="legacy",
            conv_name="runoptim1-9v1",
            query_list="1,9",
            notify=False,
            conv_mode="optimization",
            baseline_backend="legacy-http",
            baseline_query_file_dir="/tmp/legacy-query-files",
            benchmark_mode="query-latency",
            storage_mode="tmpfs",
        )


def test_build_run_config_defaults_tpch_to_monetdb_without_legacy_aliases() -> None:
    """TPC-H config 应默认使用 MonetDB baseline，且不携带 TPC-H MonetDB legacy alias。"""
    from tpch_monetdb.utils.cli_config import build_run_config

    config = build_run_config(
        benchmark="tpch",
        conv_name="runoptim1-9v1",
        query_list="Q1,Q9",
        notify=False,
        conv_mode="optimization",
    )

    assert config.baseline_backend == "monetdb"
    assert config.baseline_query_file_dir is None


def test_build_run_config_carries_wandb_guard_dimensions() -> None:
    from tpch_monetdb.utils.cli_config import build_run_config

    config = build_run_config(
        benchmark="tpch",
        conv_name="storageplan1-9v1",
        query_list="1,9",
        notify=False,
        conv_mode="scripted",
        disable_wandb_when_tracing_disabled=True,
        wandb_init_max_attempts=5,
        wandb_init_timeout_s=12.5,
        wandb_upload_timeout_s=90.0,
        wandb_finish_timeout_s=18.0,
        wandb_finish_retries=2,
    )

    assert config.disable_wandb_when_tracing_disabled is True
    assert config.wandb_init_max_attempts == 5
    assert config.wandb_init_timeout_s == 12.5
    assert config.wandb_upload_timeout_s == 90.0
    assert config.wandb_finish_timeout_s == 18.0
    assert config.wandb_finish_retries == 2


def test_run_optim_loop_parser_drops_legacy_baseline_dimensions() -> None:
    """run_optim_loop_tpch_monetdb parser 不再暴露 TSBS/QuestDB baseline 参数."""
    parser = run_optim_loop_tpch_monetdb.build_parser(add_help=False)
    args = parser.parse_args(
        [
            "--conv",
            "runoptim1-9v1",
            "--benchmark_mode",
            "query-latency",
            "--storage_mode",
            "tmpfs",
        ]
    )

    assert not hasattr(args, "baseline_backend")
    assert not hasattr(args, "baseline_query_file_dir")
    assert args.benchmark_mode == "query-latency"
    assert args.storage_mode == "tmpfs"


def test_optimization_conversation_rejects_legacy_tpch_monetdb_benchmark(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """QuestDB/TSBS baseline 删除后，optimization 不再接受 TPC-H MonetDB benchmark."""
    monkeypatch.setattr(
        TpchMonetdbOptimizationConversation,
        "_initialize_manifest",
        lambda self: None,
    )

    with pytest.raises(ValueError, match="only supports benchmark='tpch'"):
        TpchMonetdbOptimizationConversation(
            benchmark="legacy",
            query_ids=["1"],
            run_tool=SimpleNamespace(),
            verify_sf_list=[1],
            benchmark_sf=1,
            git_snapshotter=SimpleNamespace(),
            session=SimpleNamespace(),
            wandb_run_hook=None,
            manifest_path=tmp_path / "reference_manifest.json",
            conversation_json_path=tmp_path / "conv.json",
            callback=lambda *_args, **_kwargs: None,
            workspace_root=tmp_path,
        )


def test_optimization_conversation_accepts_measurement_runtime_args(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """初始化时应消费 measurement 参数，而不是把它们透传到父类构造器."""
    from tpch_monetdb.tools.tpch.hardware_counters import build_hardware_counter_preflight

    monkeypatch.setattr(
        TpchMonetdbOptimizationConversation,
        "_initialize_manifest",
        lambda self: None,
    )

    preflight = build_hardware_counter_preflight(
        backend="linux_perf_native",
        target_cpu="icelake",
        runner_cmd=None,
        host_kernel="5.14.0",
        perf_event_paranoid="1",
        large_sf=1000,
    )

    conversation = TpchMonetdbOptimizationConversation(
        query_ids=["1"],
        run_tool=SimpleNamespace(),
        verify_sf_list=[1],
        benchmark_sf=1,
        git_snapshotter=SimpleNamespace(),
        session=SimpleNamespace(),
        wandb_run_hook=None,
        manifest_path=tmp_path / "reference_manifest.json",
        conversation_json_path=tmp_path / "conv.json",
        callback=lambda *_args, **_kwargs: None,
        workspace_root=tmp_path,
        baseline_backend="monetdb",
        target_cpu="icelake",
        hardware_counter_preflight=preflight,
        large_sf=1000,
    )

    assert conversation.target_cpu == "icelake"
    assert conversation.hardware_counter_preflight is preflight
    assert conversation.large_sf == 1000
    return None


def test_bootstrap_sets_litellm_local_cost_map_env() -> None:
    import os
    os.environ.pop("LITELLM_LOCAL_MODEL_COST_MAP", None)
    from tpch_monetdb.bootstrap_env import bootstrap_runtime_env
    bootstrap_runtime_env()
    assert os.environ.get("LITELLM_LOCAL_MODEL_COST_MAP") == "true"
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "false"
    bootstrap_runtime_env()
    assert os.environ.get("LITELLM_LOCAL_MODEL_COST_MAP") == "true"
    return None


def test_scripted_summary_wandb_fields_default_zero() -> None:
    from tpch_monetdb.utils.scripted_summary import ScriptedRunSummary
    s = ScriptedRunSummary(
        benchmark="tpch",
        conv_name="test",
        run_id="test",
        query_list=["1"],
        is_bespoke_storage=False,
        final_snapshot_hash="abc",
        completed_at="2026-01-01T00:00:00Z",
        conversation_json="conv.json",
        session_db_path="session.sqlite",
        success=True,
    )
    assert s.wandb_primary_run_id is None
    assert s.wandb_final_run_id is None
    assert s.wandb_init_attempt_count == 0
    assert s.wandb_attempted_run_ids == []
    assert s.wandb_retry_used is False
    assert s.wandb_first_failure_excerpt is None


def test_scripted_summary_from_dict_ignores_unknown_keys() -> None:
    from tpch_monetdb.utils.scripted_summary import ScriptedRunSummary
    data = {
        "benchmark": "tpch",
        "conv_name": "test",
        "run_id": "test",
        "query_list": ["1"],
        "is_bespoke_storage": False,
        "final_snapshot_hash": "abc",
        "completed_at": "2026-01-01T00:00:00Z",
        "conversation_json": "conv.json",
        "session_db_path": "session.sqlite",
        "success": True,
    }
    s = ScriptedRunSummary.from_dict(data)
    assert s.conv_name == "test"
    assert s.wandb_init_attempt_count == 0


def test_scripted_summary_from_dict_reads_new_wandb_fields() -> None:
    from tpch_monetdb.utils.scripted_summary import ScriptedRunSummary
    data = {
        "benchmark": "tpch",
        "conv_name": "test",
        "run_id": "test",
        "query_list": ["1"],
        "is_bespoke_storage": False,
        "final_snapshot_hash": "abc",
        "completed_at": "2026-01-01T00:00:00Z",
        "conversation_json": "conv.json",
        "session_db_path": "session.sqlite",
        "success": True,
        "wandb_primary_run_id": "pid",
        "wandb_final_run_id": "fid",
        "wandb_init_attempt_count": 2,
        "wandb_attempted_run_ids": ["pid", "fid"],
        "wandb_retry_used": True,
        "wandb_first_failure_excerpt": "err",
    }
    s = ScriptedRunSummary.from_dict(data)
    assert s.wandb_primary_run_id == "pid"
    assert s.wandb_final_run_id == "fid"
    assert s.wandb_init_attempt_count == 2
    assert s.wandb_retry_used is True


def test_persist_scripted_run_with_wandb_result(tmp_path, monkeypatch) -> None:
    from tpch_monetdb.utils.scripted_summary import persist_successful_scripted_run, ScriptedRunSummary
    from types import SimpleNamespace

    fake_wandb_result = SimpleNamespace(
        primary_run_id="prun",
        final_run_id="frun",
        attempt_count=2,
        attempted_run_ids=["prun", "frun"],
        used_fallback=True,
        first_failure_excerpt="initial failure",
    )
    (tmp_path / "snap").mkdir()
    monkeypatch.setattr(
        "tpch_monetdb.utils.scripted_summary.build_storage_plan_alignment",
        lambda _path: {"status": "aligned", "departures": []},
    )
    path = persist_successful_scripted_run(
        benchmark="tpch",
        conv_name="test_conv",
        query_list=["1"],
        is_bespoke_storage=False,
        final_snapshot_hash="abc123",
        conversation_json_path=tmp_path / "conv.json",
        session_db_path=tmp_path / "session.sqlite",
        artifacts_dir=tmp_path,
        validation_mode="strict",
        wandb_result=fake_wandb_result,
    )
    assert path.exists()
    import json
    data = json.loads(path.read_text())
    assert data["wandb_primary_run_id"] == "prun"
    assert data["wandb_final_run_id"] == "frun"
    assert data["wandb_init_attempt_count"] == 2
    assert data["wandb_retry_used"] is True
    assert data["wandb_first_failure_excerpt"] == "initial failure"


# ── per-query session branch 创建回归测试 ──────────────────────────

def _make_stateful_fake_session(user_turns=None, start_branch="main"):
    """构建一个遵循 SDK create_branch_from_turn 契约的 FakeSession。

    SDK 契约要点：
    - create_branch_from_turn 在当前 branch 中校验 turn 是否含 user message
    - 成功后 self._current_branch_id 切换到新 branch
    - _copy_messages_to_new_branch 使用 branch_turn_number < from_turn_number（不含 branch 点本身），
      因此新 branch 初始无 user turns

    参数 start_branch 允许测试从非 main 分支启动，防止断言把 source branch 硬编码为 'main'。
    """
    class _FakeSession:
        def __init__(self, user_turns=None, start_branch="main"):
            self._current_branch_id = start_branch
            self._user_turns = user_turns or {start_branch: {3}}
            self.switch_log: list[tuple] = []

        async def switch_to_branch(self, b) -> None:
            self.switch_log.append(("switch_to", b))
            self._current_branch_id = b

        async def create_branch_from_turn(self, t, branch_name) -> str:
            self.switch_log.append(("create_branch", t, branch_name, self._current_branch_id))
            turns = self._user_turns.get(self._current_branch_id, set())
            if t not in turns:
                raise ValueError(
                    f"Turn {t} does not contain a user message "
                    f"in branch '{self._current_branch_id}'"
                )
            self._user_turns[branch_name] = set()
            self._current_branch_id = branch_name
            return branch_name

        async def get_conversation_turns(self) -> list[dict]:
            turns = self._user_turns.get(self._current_branch_id, set())
            return [{"turn": t} for t in sorted(turns)]

    return _FakeSession(user_turns, start_branch)


def _make_optimization_conversation_stub(tmp_path, query_ids, session):
    """构建一个最小可用的 TpchMonetdbOptimizationConversation stub，仅覆盖到 branch 创建链路。"""
    from tpch_monetdb.conversations.optimization_conversation_tpch_monetdb import TpchMonetdbOptimizationConversation
    from tpch_monetdb.conversations.optimization_validation import CorrectnessCheckSummary

    (tmp_path / "queries.txt").write_text("\n".join(query_ids) + "\n", encoding="utf-8")

    c = TpchMonetdbOptimizationConversation.__new__(TpchMonetdbOptimizationConversation)
    c.benchmark_sf = 10
    c.bespoke_storage = True
    c.query_ids = list(query_ids)
    c.required_validation_sf_list = [1]
    c.revert_on_regression = False
    c.regression_tolerance = 0.1
    c.git_snapshotter = SimpleNamespace(current_hash="h1", restore=lambda h: None)
    c.conversation_json_path = tmp_path / "conv.json"
    c.callback = lambda *a, **k: None
    c.replay = False
    c.notify = False
    c.auto_finish = True
    c.allowed_choices = ("u",)
    c.model = None
    c.auto_u = True
    c.replay_cache = False
    c.workspace_root = None
    c.used = []
    c.get_choice = lambda: "u"
    c.session = session
    c.run_tool = SimpleNamespace(
        cwd=tmp_path,
        run=lambda **k: ("ok", {}),
        reset_runtime_state=lambda **kwargs: None,
    )
    c._make_exec_callback = lambda qid, **_kwargs: lambda args, t: ("resp", "1 | Execution ms: 10.0", "")
    c._measure_with_manifest = lambda **k: (0.1, 1.0, 10.0, False)
    c._measure_all_queries = lambda: {"1": 0.1, "2": 0.1}
    c._delete_result_csvs = lambda cwd: None
    c._check_correctness_with_scale_factors = lambda qids, trace_mode, scale_factors: asyncio.sleep(0, result=True)
    from tpch_monetdb.conversations.optimization_instrumentation import TraceEvidenceSummary
    c._summarize_trace_evidence_for_queries = lambda qids: TraceEvidenceSummary(
        qids=tuple(qids), sufficient=True, message="ok",
    )
    c.wandb_run_hook = None
    c.conv_name = "test_conv"
    c.artifacts_dir = tmp_path / "artifacts"
    c.start_snapshot_hash = "h0"
    c.query_rt_log = {}
    c.best_rt_log = {}
    c.global_regression_records = []

    async def dummy_exec(*args, **kwargs) -> None:
        return None
    c._exec = dummy_exec

    async def dummy_check_correctness(*args, **kwargs) -> bool:
        return True
    c._check_correctness = dummy_check_correctness

    async def dummy_run_stage(**kwargs) -> None:
        return None
    c._run_stage = dummy_run_stage

    c._sample_trace_for_query = lambda query_id: SimpleNamespace(
        issue_class="scan_bound",
        summary_text="trace summary",
        sampled_instantiations=("i1",),
    )
    from tpch_monetdb.conversations.optimization_conversation_tpch_monetdb import StageConfig
    c._build_query_stage = lambda **kwargs: StageConfig(
        name="trace_expert",
        get_prompt=lambda _rt: "",
        get_descriptor=lambda: "trace_expert",
        max_turns=10,
    )
    c._run_stage_correctness_gate = lambda **kwargs: CorrectnessCheckSummary(
        success=True, message="ok", metrics={"validation/correct": True}, failed_scale_factor=None,
    )
    c._persist_hotspot_summary = lambda records: tmp_path / "hotspots.md"
    c._collect_query_output_split_measurements = lambda qids: {}

    from tpch_monetdb.conversations.optimization_conversation_tpch_monetdb import GlobalHumanReferenceResult
    async def dummy_global_human_reference(**kwargs) -> GlobalHumanReferenceResult:
        return GlobalHumanReferenceResult(
            runtime_by_query={q: 0.1 for q in query_ids}, written_files=(), accepted=True,
        )
    c._run_global_human_reference = dummy_global_human_reference

    from tpch_monetdb.conversations.optimization_conversation_tpch_monetdb import OptimizationFailureState
    c._persist_failure_summary = lambda **kwargs: OptimizationFailureState(**kwargs)

    return c


@pytest.mark.asyncio
async def test_per_query_branch_creation_repairs_sdk_branch_side_effect(
    monkeypatch, tmp_path,
) -> None:
    """复现测试：create_branch_from_turn 成功后切换当前 branch，若未在每次迭代前
    切回 source branch，第二次 create_branch_from_turn 将在新 branch 中校验
    branch point（新 branch 不含 user turns），抛出 ValueError；修复后通过。

    起始 branch 设为非 main，防止断言把 source branch 硬编码为 'main'。"""
    SOURCE_BRANCH = "optimization_session"
    _patch_tpch_monetdb_optimization_prompt_builders(monkeypatch)
    monkeypatch.setattr(
        "tpch_monetdb.conversations.optimization_conversation_tpch_monetdb.run_required_correctness_checks",
        lambda *a, **k: SimpleNamespace(
            success=True, message="ok", metrics={"validation/correct": True},
            failure_code=None, failure_detail=None,
        ),
    )

    session = _make_stateful_fake_session({SOURCE_BRANCH: {5}}, start_branch=SOURCE_BRANCH)
    conversation = _make_optimization_conversation_stub(tmp_path, ["1", "2"], session)

    result = await conversation.run()
    assert result == []

    assert len(session.switch_log) >= 4
    create_sources = [entry[3] for entry in session.switch_log if entry[0] == "create_branch"]
    assert all(src == SOURCE_BRANCH for src in create_sources), (
        f"All create_branch_from_turn calls must validate in source branch "
        f"'{SOURCE_BRANCH}', got sources: {create_sources}"
    )


@pytest.mark.asyncio
async def test_per_query_branch_creation_empty_turns_raises_runtime_error(
    monkeypatch, tmp_path,
) -> None:
    """get_conversation_turns() 返回 [] 时必须抛出明确 RuntimeError（且先落 failure summary），
    而非 IndexError。起始 branch 设为非 main，防止把 source branch 硬编码为 'main'。"""
    SOURCE_BRANCH = "optimization_session"
    _patch_tpch_monetdb_optimization_prompt_builders(monkeypatch)
    monkeypatch.setattr(
        "tpch_monetdb.conversations.optimization_conversation_tpch_monetdb.run_required_correctness_checks",
        lambda *a, **k: SimpleNamespace(
            success=True, message="ok", metrics={"validation/correct": True},
            failure_code=None, failure_detail=None,
        ),
    )

    session = _make_stateful_fake_session({SOURCE_BRANCH: set()}, start_branch=SOURCE_BRANCH)
    conversation = _make_optimization_conversation_stub(tmp_path, ["1"], session)

    with pytest.raises(RuntimeError, match="No user turns available"):
        await conversation.run()


@pytest.mark.asyncio
async def test_per_query_branch_creation_switch_before_each_create(
    monkeypatch, tmp_path,
) -> None:
    """顺序测试：每次 create_branch_from_turn 调用前必须先 switch_to_branch(source_branch)，
    确保始终在同一 source context 下校验 turn。

    起始 branch 设为非 main，断言切回的是启动时的 source branch 变量，而非字面量 'main'。"""
    SOURCE_BRANCH = "optimization_session"
    _patch_tpch_monetdb_optimization_prompt_builders(monkeypatch)
    monkeypatch.setattr(
        "tpch_monetdb.conversations.optimization_conversation_tpch_monetdb.run_required_correctness_checks",
        lambda *a, **k: SimpleNamespace(
            success=True, message="ok", metrics={"validation/correct": True},
            failure_code=None, failure_detail=None,
        ),
    )

    session = _make_stateful_fake_session({SOURCE_BRANCH: {7}}, start_branch=SOURCE_BRANCH)
    conversation = _make_optimization_conversation_stub(tmp_path, ["1", "2", "3"], session)

    result = await conversation.run()
    assert result == []

    ordered_ops = [(entry[0], entry[1]) for entry in session.switch_log]
    create_ops = [i for i, op in enumerate(ordered_ops) if op[0] == "create_branch"]
    assert len(create_ops) == 3, f"Expected 3 query branch calls, got {len(create_ops)}"

    for idx in create_ops:
        assert idx > 0, f"create_branch at position {idx} must be preceded by switch_to_branch"
        prev_op = ordered_ops[idx - 1]
        assert prev_op == ("switch_to", SOURCE_BRANCH), (
            f"create_branch at position {idx} must be preceded by "
            f"switch_to('{SOURCE_BRANCH}'), got {prev_op}"
        )


@pytest.mark.asyncio
async def test_optimization_run_processes_query_unit_independently(monkeypatch, tmp_path) -> None:
    SOURCE_BRANCH = "optimization_session"
    _patch_tpch_monetdb_optimization_prompt_builders(monkeypatch)
    monkeypatch.setattr(
        "tpch_monetdb.conversations.optimization_conversation_tpch_monetdb.run_required_correctness_checks",
        lambda *a, **k: SimpleNamespace(
            success=True, message="ok", metrics={"validation/correct": True},
            failure_code=None, failure_detail=None,
        ),
    )

    session = _make_stateful_fake_session({SOURCE_BRANCH: {5}}, start_branch=SOURCE_BRANCH)
    conversation = _make_optimization_conversation_stub(tmp_path, ["3", "4"], session)
    stage_calls: list[tuple[str, tuple[str, ...] | None]] = []

    async def stop_after_first_unit(**kwargs):
        stage_calls.append((kwargs["query_id"], kwargs.get("scope_query_ids")))
        raise RuntimeError("stop-after-first-unit")

    conversation._run_stage = stop_after_first_unit

    with pytest.raises(RuntimeError, match="stop-after-first-unit"):
        await conversation.run()

    assert stage_calls == [("3", ("3",))]


def test_require_resume_snapshot_fields_reports_missing_contract_keys() -> None:
    """resume gate 必须用结构化错误拒绝缺字段的旧 snapshot。"""
    with pytest.raises(PipelineContractError, match="RESUME_SNAPSHOT_INCOMPLETE"):
        require_resume_snapshot_fields(
            {
                "storage_plan_sha256": "plan",
                "todo_sha256": "todo",
            },
            stage="runtime_wiring_test",
        )
    return None


def test_require_resume_snapshot_fields_accepts_complete_contract_snapshot() -> None:
    """完整的新契约 snapshot 不应被 runtime gate 拒绝。"""
    require_resume_snapshot_fields(
        {
            "implementation_manifest_sha256": "manifest",
            "storage_plan_sha256": "plan",
            "todo_sha256": "todo",
            "todo_reconciliation": {"status": "completed"},
            "control_artifact_hashes": {"plan": "plan"},
        },
        stage="runtime_wiring_test",
    )
    return None
