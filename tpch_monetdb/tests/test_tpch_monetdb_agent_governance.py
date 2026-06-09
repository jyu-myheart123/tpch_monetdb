import asyncio
import contextlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from agents.run_context import RunContextWrapper

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tpch_monetdb.main_tpch_monetdb
import tpch_monetdb.run_gen_base_impl_tpch_monetdb
from tpch_monetdb.conversations.scripted_conversation import PromptStep
from tpch_monetdb.conversations.compact_prompts import COMPACT_SYSTEM_PROMPT
from tpch_monetdb.conversations.conversation import AbstractConversation
from tpch_monetdb.tools.stage_tool_policy import (
    StageRunSummary,
    TodoState,
    extract_validation_summary,
)
from tpch_monetdb.tools.tpch_monetdb_agent_tools import build_tpch_monetdb_agent_tools
from tpch_monetdb.tools.tool_factory import build_tools
from tpch_monetdb.utils.agent_rules import RuleAssembly, RuleScope, load_agent_rules, log_rule_assembly
from tpch_monetdb.utils.stage_summary import render_stage_summary
from tpch_monetdb.utils.wandb_init import init_wandb_run_with_retry


def _make_run_tool():
    async def _invoke(_ctx, _args_json: str) -> str:
        return "run ok"

    return SimpleNamespace(
        name="run",
        description="run",
        params_json_schema={"type": "object", "properties": {}},
        on_invoke_tool=_invoke,
    )


def test_prompt_step_exposes_rule_area_metadata() -> None:
    step = PromptStep.from_json_value(
        {
            "text": "prompt",
            "descriptor": "todo_plan",
            "tool_profile": "todo_plan",
            "rule_area": "runtime",
        }
    )

    assert step.rule_area == "runtime"
    assert step.to_callback_metadata() == {
        "tool_profile": "todo_plan",
        "rule_area": "runtime",
    }


def test_production_auto_conversation_rejects_replace_insert_choices(tmp_path: Path) -> None:
    """Production auto_u conversations must not bypass registered prompt assets."""
    class DummyConversation(AbstractConversation):
        async def run(self) -> list[str] | None:
            return []

    async def callback(*_args, **_kwargs) -> None:
        return None

    with pytest.raises(ValueError, match="replace/insert"):
        DummyConversation(
            conversation_json_path=tmp_path / "conv.json",
            callback=callback,
            auto_u=True,
            allowed_choices=("u", "r", "i", "c"),
        )
    return None


def test_agent_rule_loader_splits_global_and_scoped_rules(tmp_path: Path) -> None:
    rules_dir = tmp_path / "agent_rules"
    rules_dir.mkdir()
    (rules_dir / "kernel.md").write_text(
        "---\npriority: 0\nstages: []\n---\nGLOBAL"
    )
    (rules_dir / "stage.md").write_text(
        "---\npriority: 20\nstages: [compile_fix]\nareas: [runtime]\n---\nSCOPED"
    )
    (rules_dir / "path.md").write_text(
        "---\npriority: 30\nstages: [compile_fix]\npaths: [tpch_monetdb/oracle/*]\n---\nPATH"
    )

    global_rules = load_agent_rules(
        rules_dir,
        scope=RuleScope(stage_name=None, area_name=None),
        include_global=True,
        token_budget=100,
    )
    scoped_rules = load_agent_rules(
        rules_dir,
        scope=RuleScope(
            stage_name="compile_fix",
            area_name="runtime",
            candidate_paths=("tpch_monetdb/oracle/x.py",),
        ),
        include_global=False,
        token_budget=100,
    )

    assert "GLOBAL" in global_rules.text
    assert scoped_rules.included_files == ("stage.md", "path.md")
    assert "SCOPED" in scoped_rules.text
    assert "PATH" in scoped_rules.text


def test_agent_rule_loader_logs_truncation(caplog, tmp_path: Path) -> None:
    rules_dir = tmp_path / "agent_rules"
    rules_dir.mkdir()
    (rules_dir / "kernel.md").write_text(
        "---\npriority: 0\nstages: []\n---\n" + ("A" * 120)
    )
    (rules_dir / "runtime.md").write_text(
        "---\npriority: 10\nstages: []\n---\n" + ("B" * 120)
    )

    with caplog.at_level("INFO"):
        assembly = load_agent_rules(
            rules_dir,
            scope=RuleScope(stage_name=None, area_name=None),
            include_global=True,
            token_budget=20,
        )
        log_rule_assembly("global", assembly)

    assert assembly.was_truncated is True
    assert "Rule assembly (global): loaded=" in caplog.text
    assert "truncated=" in caplog.text


@pytest.mark.asyncio
async def test_stage_hint_and_read_output_are_marked(tmp_path: Path) -> None:
    (tmp_path / "loader_impl.cpp").write_text("int value = 1;\n", encoding="utf-8")

    bundle = build_tools(
        use_litellm=True,
        workspace_path=tmp_path,
        cache_path=tmp_path / "cache",
        shell_executor=None,
        workspace_editor=None,
        compile_cache_dir=tmp_path / "compile",
        run_tool_wrapper=_make_run_tool(),
        git_snapshotter=None,
        wandb_metrics_hook=None,
    )
    bundle.runtime.activate("finish_skeleton", 0, "finish_skeleton")

    assert bundle.runtime.generate_stage_hint().startswith("[Stage Context]\n")

    read_tool = next(
        tool for tool in bundle.tools_by_profile["finish_skeleton"] if tool.name == "read_file"
    )
    result = await read_tool.on_invoke_tool(
        RunContextWrapper(context=None),
        json.dumps({"file_path": "loader_impl.cpp"}),
    )

    assert result.startswith("[Evidence]\n")
    assert "loader_impl.cpp lines" in result


@pytest.mark.asyncio
async def test_compile_failure_returns_error_envelope_and_evidence(tmp_path: Path) -> None:
    async def compile_invoke(_ctx, _args_json: str) -> str:
        return "compile failed"

    compile_tool = SimpleNamespace(
        name="compile",
        description="compile",
        params_json_schema={"type": "object", "properties": {}},
        on_invoke_tool=compile_invoke,
    )

    bundle = build_tpch_monetdb_agent_tools(
        workspace_path=tmp_path,
        cache_path=tmp_path / "cache",
        compile_tool=compile_tool,
        run_tool=_make_run_tool(),
        git_snapshotter=None,
        wandb_metrics_hook=None,
        apply_patch_tool=None,
    )
    compile_wrapper = next(
        tool for tool in bundle.tools_by_profile["compile_fix"] if tool.name == "compile"
    )
    bundle.runtime.activate("compile_fix", 0, "compile_fix")

    result = await compile_wrapper.on_invoke_tool(
        RunContextWrapper(context=None),
        json.dumps({}),
    )

    assert "[ERROR:COMPILE_FAILED]" in result
    assert "Allowed next actions: edit_file, apply_patch" in result
    assert "[Evidence]\ncompile failed" in result


def test_render_stage_summary_includes_current_state() -> None:
    summary = StageRunSummary(
        profile_name="compile_fix",
        prompt_index=1,
        prompt_descriptor="compile_fix",
        final_output="Need to fix compile error",
        tool_counts={"read_file": 2, "compile": 1},
        written_files=("query_impl.cpp",),
        last_compile_summary="error: missing ;",
        last_run_summary=None,
        todo_before=TodoState.from_text("- [ ] one\n"),
        todo_after=TodoState.from_text("- [x] one\n"),
        last_validation_summary="Validation failed: row mismatch",
    )

    rendered = render_stage_summary(summary)

    assert "[Stage Summary]" in rendered
    assert "Files changed: query_impl.cpp" in rendered
    assert "Last validate result: Validation failed: row mismatch" in rendered
    assert "Validation passed: (unknown)" in rendered
    assert "Current blocker: Validation failed: row mismatch" in rendered
    assert "TODO progress: completed=1" in rendered


def test_render_stage_summary_does_not_repeat_rule_text() -> None:
    summary = StageRunSummary(
        profile_name="optimization_general",
        prompt_index=1,
        prompt_descriptor="trace",
        final_output="Focused on one bottleneck and updated a helper.",
        tool_counts={"read_file": 1, "edit_file": 1},
        written_files=("query_impl.cpp",),
        last_compile_summary="compile ok",
        last_run_summary="1 | Execution ms: 10.0",
        todo_before=None,
        todo_after=None,
    )

    rendered = render_stage_summary(summary)

    assert "Google C++ style" not in rendered
    assert "Workflow priority" not in rendered
    assert "do not cheat by shifting work" not in rendered


@pytest.mark.asyncio
async def test_optimization_exec_rejects_baseline_owned_written_files(
    tmp_path: Path,
) -> None:
    """生产回调层应拒绝写入 baseline-owned 路径。"""
    from tpch_monetdb.conversations.optimization_conversation_tpch_monetdb import (
        TpchMonetdbOptimizationConversation,
    )
    from tpch_monetdb.tools.stage_tool_policy import StageRunSummary

    conversation = TpchMonetdbOptimizationConversation.__new__(TpchMonetdbOptimizationConversation)

    async def fake_process_prompt(
        *_args,
        **_kwargs,
    ) -> tuple[str, str, StageRunSummary]:
        return (
            "u",
            "prompt",
            StageRunSummary(
                profile_name="optimization_general",
                prompt_index=0,
                prompt_descriptor="trace",
                final_output="done",
                tool_counts={"edit_file": 1},
                written_files=("tpch_monetdb/oracle/monetdb_oracle.py",),
                last_compile_summary=None,
                last_run_summary=None,
                todo_before=None,
                todo_after=None,
            ),
    )

    conversation.process_prompt = fake_process_prompt

    with pytest.raises(RuntimeError, match="baseline-owned"):
        await conversation._exec(
            "prompt",
            "trace",
            max_turns=450,
            tool_profile="optimization_general",
        )


@pytest.mark.asyncio
async def test_optimization_exec_rejects_host_owned_contract_writes(
    tmp_path: Path,
) -> None:
    """生产回调层应拒绝 agent 修改 host-owned objective contract."""
    from tpch_monetdb.conversations.optimization_conversation_tpch_monetdb import (
        TpchMonetdbOptimizationConversation,
    )
    from tpch_monetdb.tools.stage_tool_policy import StageRunSummary

    conversation = TpchMonetdbOptimizationConversation.__new__(TpchMonetdbOptimizationConversation)

    async def fake_process_prompt(
        *_args,
        **_kwargs,
    ) -> tuple[str, str, StageRunSummary]:
        return (
            "u",
            "prompt",
            StageRunSummary(
                profile_name="optimization_general",
                prompt_index=0,
                prompt_descriptor="trace",
                final_output="done",
                tool_counts={"edit_file": 1},
                written_files=("workload_objective.json",),
                last_compile_summary=None,
                last_run_summary=None,
                todo_before=None,
                todo_after=None,
            ),
        )

    conversation.process_prompt = fake_process_prompt

    with pytest.raises(RuntimeError, match="host-owned"):
        await conversation._exec(
            "prompt",
            "trace",
            max_turns=450,
            tool_profile="optimization_general",
        )


def test_stage_runtime_records_validation_summary_from_run_output(tmp_path) -> None:
    from tpch_monetdb.tools.tpch_monetdb_agent_tools import StageToolRuntime

    runtime = StageToolRuntime(tmp_path)
    runtime.activate("compile_fix", 0, "compile_fix")
    runtime.record_execution(
        "run",
        "Validation failed:\nQuestDB rows mismatch\nvalidator diff follows",
        success=False,
    )

    summary = runtime.finish_stage("Need to fix validator mismatch")

    assert summary.last_run_summary is not None
    assert summary.last_validation_summary is not None
    assert "Validation failed" in summary.last_validation_summary
    assert summary.run_succeeded is False
    assert summary.validation_passed is False
    assert summary.last_failure_kind == "validation"


def test_stage_runtime_recognizes_tpch_validation_pass_marker(tmp_path) -> None:
    from tpch_monetdb.tools.tpch_monetdb_agent_tools import StageToolRuntime

    runtime = StageToolRuntime(tmp_path)
    runtime.activate("correctness_queries_writeonly", 0, "correctness_q1")
    runtime.record_execution(
        "run",
        (
            "exit_code: 0 signal: 0\n"
            "TPC-H generated runtime validation PASS: "
            "TPC-H validation PASS for Q1; columns=ok; rows=1/1; ordered=True"
        ),
        success=True,
    )

    summary = runtime.finish_stage("Q1 validation passed")

    assert summary.last_validation_summary is not None
    assert "TPC-H validation PASS for Q1" in summary.last_validation_summary
    assert summary.validation_passed is True
    assert summary.last_failure_kind is None
    return None


def test_stage_runtime_marks_benchmark_run_for_performance_comparison(tmp_path) -> None:
    from tpch_monetdb.tools.tpch_monetdb_agent_tools import StageToolRuntime

    runtime = StageToolRuntime(tmp_path)
    args_json = json.dumps(
        {
            "scale_factor": 1,
            "optimize": False,
            "query_id": ["Q6"],
        }
    )

    runtime.activate("benchmark", 0, "benchmark")
    benchmark_args = json.loads(runtime.run_args_with_stage_metadata(args_json))

    runtime.activate("correctness_foundation", 1, "correctness_q6")
    correctness_args = json.loads(runtime.run_args_with_stage_metadata(args_json))

    assert benchmark_args["__base_performance_comparison"] is True
    assert benchmark_args["optimize"] is True
    assert correctness_args["optimize"] is False
    assert "__base_performance_comparison" not in correctness_args
    return None


@pytest.mark.asyncio
async def test_run_tool_accepts_hidden_base_performance_marker(
    tmp_path,
    monkeypatch,
) -> None:
    from tpch_monetdb.tools.tpch import run as run_module

    captured: dict[str, object] = {}

    def fake_run_tool_call(
        self,
        scale_factor: float,
        optimize: bool,
        query_id: list[str] | None = None,
        trace_mode: bool = False,
        enable_performance_comparison: bool = False,
    ) -> str:
        """Capture internal run invocation arguments without compiling C++."""
        del self, trace_mode
        captured["scale_factor"] = scale_factor
        captured["optimize"] = optimize
        captured["query_id"] = query_id
        captured["enable_performance_comparison"] = enable_performance_comparison
        return "run ok"

    monkeypatch.setattr(run_module.RunTool, "__call__", fake_run_tool_call)
    run_tool, _ = run_module.make_run_tool(
        cwd=tmp_path,
        dataset_name="tpch",
        base_data_dir=str(tmp_path),
    )
    result = await run_tool.on_invoke_tool(
        RunContextWrapper(context=None),
        json.dumps(
            {
                "scale_factor": 1,
                "optimize": False,
                "query_id": ["Q6"],
                "__base_performance_comparison": True,
            }
        ),
    )

    assert result == "run ok"
    assert "__base_performance_comparison" not in json.dumps(
        run_tool.params_json_schema
    )
    assert captured["enable_performance_comparison"] is True
    assert captured["query_id"] == ["Q6"]
    return None


def test_stage_runtime_recognizes_tpch_validation_fail_marker(tmp_path) -> None:
    from tpch_monetdb.tools.tpch_monetdb_agent_tools import StageToolRuntime

    runtime = StageToolRuntime(tmp_path)
    runtime.activate("correctness_queries_writeonly", 0, "correctness_q6")
    runtime.record_execution(
        "run",
        (
            "TPC-H generated runtime validation FAIL: "
            "TPC-H validation FAIL for Q6; columns=ok; rows=1/1; "
            "ordered=True; first_mismatch=null:revenue:Cell value differs"
        ),
        success=False,
    )

    summary = runtime.finish_stage("Q6 validation failed")

    assert summary.last_validation_summary is not None
    assert "TPC-H validation FAIL for Q6" in summary.last_validation_summary
    assert summary.validation_passed is False
    assert summary.last_failure_kind == "validation"
    return None


def test_stage_runtime_clears_stale_validation_summary_on_non_validation_run(
    tmp_path,
) -> None:
    from tpch_monetdb.tools.tpch_monetdb_agent_tools import StageToolRuntime

    runtime = StageToolRuntime(tmp_path)
    runtime.activate("compile_fix", 0, "compile_fix")
    runtime.record_execution(
        "run",
        "Validation failed:\nQuestDB rows mismatch\nvalidator diff follows",
        success=False,
    )
    runtime.record_execution(
        "run",
        "1 | Execution ms: 10.0",
        success=True,
    )

    summary = runtime.finish_stage("Recovered after rerun")
    rendered = render_stage_summary(summary)

    assert summary.last_validation_summary is None
    assert summary.validation_passed is None
    assert summary.last_failure_kind is None
    assert "Last validate result: (none)" in rendered
    assert "Validation failed" not in rendered


def test_stage_runtime_read_file_refuses_large_full_read(tmp_path: Path) -> None:
    """read_file 对大文件必须拒绝无界全量读取，但允许有界切片。"""
    from tpch_monetdb.tools.tpch_monetdb_agent_tools import StageToolRuntime

    big_file = tmp_path / "tracing_output.log"
    big_file.write_text("line\n" * 450_001, encoding="utf-8")
    runtime = StageToolRuntime(tmp_path)
    runtime.activate("finish_skeleton", 0, "finish_skeleton")

    full_result = runtime.read_file("tracing_output.log", None, None)
    sliced_result = runtime.read_file("tracing_output.log", 2, 2)
    grep_result = runtime.grep_repo("line", "tracing_output.log", None, 10)

    assert "exceeds read_file full-read limit" in full_result
    assert "streamed large file" in sliced_result
    assert "2: line" in sliced_result
    assert "3: line" in sliced_result
    assert "grep_repo skipped 1 large file" in grep_result
    return None


@pytest.mark.parametrize(
    "profile_name",
    ["correctness_queries_writeonly", "correctness_foundation"],
)
def test_correctness_runtime_preserves_validation_passed_after_cache_hit(
    tmp_path,
    profile_name: str,
) -> None:
    from tpch_monetdb.tools.tpch_monetdb_agent_tools import StageToolRuntime

    runtime = StageToolRuntime(tmp_path)
    runtime.activate(profile_name, 0, "correctness_q1")
    runtime.record_execution(
        "run",
        "exit_code: 0 signal: 0\nValidation passed for Q1\n",
        success=True,
    )
    runtime.record_execution(
        "run",
        "Validation results loaded from cache for 1 queries",
        success=True,
    )

    summary = runtime.finish_stage("Q1 validated from cached rerun")

    assert summary.last_validation_summary == "Validation passed for Q1"
    assert summary.validation_passed is True
    assert summary.last_failure_kind is None
    assert summary.run_succeeded is True


def test_stage_runtime_invalidates_validation_after_later_write(tmp_path) -> None:
    from tpch_monetdb.tools.tpch_monetdb_agent_tools import StageToolRuntime

    runtime = StageToolRuntime(tmp_path)
    runtime.activate("correctness_queries_writeonly", 0, "correctness_q1")
    runtime.record_execution(
        "run",
        "exit_code: 0 signal: 0\nValidation passed for Q1\n",
        success=True,
    )
    target = tmp_path / "query_q1.cpp"
    target.write_text("void execute_q1() {}\n", encoding="utf-8")
    runtime._record_write("edit_file", target)

    summary = runtime.finish_stage("edited after validation")

    assert summary.validation_passed is None
    assert summary.run_write_revision < summary.write_revision
    assert summary.written_files == ("query_q1.cpp",)
    return None


@pytest.mark.parametrize(
    "profile_name",
    ["correctness_queries_writeonly", "correctness_foundation"],
)
def test_correctness_runtime_blocks_edit_after_current_validation(
    tmp_path,
    profile_name: str,
) -> None:
    from tpch_monetdb.tools.tpch_monetdb_agent_tools import (
        RecoverableStagePolicyError,
        StageToolRuntime,
    )

    target = tmp_path / "query_q1.cpp"
    target.write_text("int value() { return 1; }\n", encoding="utf-8")
    runtime = StageToolRuntime(tmp_path)
    runtime.activate(profile_name, 0, "correctness_q1")
    runtime.record_execution(
        "run",
        "exit_code: 0 signal: 0\nValidation passed for Q1\n",
        success=True,
    )

    with pytest.raises(
        RecoverableStagePolicyError,
        match="VALIDATION_ALREADY_PASSED_NO_MORE_WRITES",
    ):
        runtime.edit_file("query_q1.cpp", "return 1", "return 2", False)

    summary = runtime.finish_stage("Q1 validated")
    assert target.read_text(encoding="utf-8") == "int value() { return 1; }\n"
    assert summary.validation_passed is True
    assert summary.run_write_revision == summary.write_revision
    assert summary.written_files == ()
    return None


@pytest.mark.parametrize(
    "profile_name",
    ["correctness_queries_writeonly", "correctness_foundation"],
)
def test_correctness_runtime_blocks_write_file_after_current_validation(
    tmp_path,
    profile_name: str,
) -> None:
    from tpch_monetdb.tools.tpch_monetdb_agent_tools import (
        RecoverableStagePolicyError,
        StageToolRuntime,
    )

    target = tmp_path / "query_q1.cpp"
    target.write_text("old\n", encoding="utf-8")
    runtime = StageToolRuntime(tmp_path)
    runtime.activate(profile_name, 0, "correctness_q1")
    runtime.record_execution(
        "run",
        "exit_code: 0 signal: 0\nAll queries passed validation!\n",
        success=True,
    )

    with pytest.raises(
        RecoverableStagePolicyError,
        match="VALIDATION_ALREADY_PASSED_NO_MORE_WRITES",
    ):
        runtime.write_file("query_q1.cpp", "new\n")

    summary = runtime.finish_stage("Q1 validated")
    assert target.read_text(encoding="utf-8") == "old\n"
    assert summary.validation_passed is True
    assert summary.written_files == ()
    return None


@pytest.mark.parametrize(
    "profile_name",
    ["correctness_queries_writeonly", "correctness_foundation"],
)
def test_correctness_runtime_blocks_apply_patch_after_current_validation(
    tmp_path,
    profile_name: str,
) -> None:
    from tpch_monetdb.tools.tpch_monetdb_agent_tools import (
        RecoverableStagePolicyError,
        StageToolRuntime,
    )

    target = tmp_path / "query_q1.cpp"
    target.write_text("old\n", encoding="utf-8")
    runtime = StageToolRuntime(tmp_path)
    runtime.activate(profile_name, 0, "correctness_q1")
    runtime.record_execution(
        "run",
        "exit_code: 0 signal: 0\nAll queries passed validation!\n",
        success=True,
    )

    with pytest.raises(
        RecoverableStagePolicyError,
        match="VALIDATION_ALREADY_PASSED_NO_MORE_WRITES",
    ):
        runtime.validate_apply_patch(
            json.dumps(
                {
                    "type": "update_file",
                    "path": "query_q1.cpp",
                    "diff": "@@\n-old\n+new\n",
                }
            )
        )

    summary = runtime.finish_stage("Q1 validated")
    assert target.read_text(encoding="utf-8") == "old\n"
    assert summary.validation_passed is True
    assert summary.written_files == ()
    return None


def test_correctness_runtime_allows_edit_after_validation_failure(tmp_path) -> None:
    from tpch_monetdb.tools.tpch_monetdb_agent_tools import StageToolRuntime

    target = tmp_path / "query_q1.cpp"
    target.write_text("int value() { return 1; }\n", encoding="utf-8")
    runtime = StageToolRuntime(tmp_path)
    runtime.activate("correctness_queries_writeonly", 0, "correctness_q1")
    runtime.record_execution(
        "run",
        "Validation failed for Q1\nQuestDB rows mismatch",
        success=False,
    )

    result = runtime.edit_file("query_q1.cpp", "return 1", "return 2", False)

    summary = runtime.finish_stage("Q1 edited after failure")
    assert result == "Updated query_q1.cpp with 1 replacement(s)"
    assert target.read_text(encoding="utf-8") == "int value() { return 2; }\n"
    assert summary.validation_passed is None
    assert summary.written_files == ("query_q1.cpp",)
    return None


def test_correctness_runtime_preserves_validation_failure_after_cache_hit(
    tmp_path,
) -> None:
    from tpch_monetdb.tools.tpch_monetdb_agent_tools import StageToolRuntime

    runtime = StageToolRuntime(tmp_path)
    runtime.activate("correctness_queries_writeonly", 0, "correctness_q1")
    runtime.record_execution(
        "run",
        "Validation failed for Q1\nQuestDB rows mismatch",
        success=False,
    )
    runtime.record_execution(
        "run",
        "Validation results loaded from cache for 1 queries",
        success=True,
    )

    summary = runtime.finish_stage("Q1 still needs a real validation pass")

    assert summary.last_validation_summary == "Validation failed for Q1"
    assert summary.validation_passed is False
    assert summary.last_failure_kind == "validation"
    assert summary.run_succeeded is True


def test_extract_validation_summary_ignores_generic_questdb_lines() -> None:
    assert extract_validation_summary("QuestDB connection ok") is None
    assert extract_validation_summary("noise only") is None


def test_extract_validation_summary_keeps_full_failure_payload_compact() -> None:
    output = "\n".join(
        [
            "Validation failed:",
            "Q9 summary:",
            "Comparison Report for Q9:",
            "Q9 comparison_report_json:",
            '{"cell_mismatches": [{"row": 2, "column": "hostname"}]}',
            "Q9 questdb_result_json:",
            '{"rows": [["2016-01-01T00:00:00.000000Z", "host_2"]]}',
            "Q9 generated_tpch_result_json:",
            '{"rows": [["2016-01-01T00:00:00.000000Z", "host_10"]]}',
        ]
    )

    summary = extract_validation_summary(output)

    assert summary == "Validation failed:"


def test_scripted_entrypoint_drops_questdb_readiness_hook() -> None:
    parser = tpch_monetdb.run_gen_base_impl_tpch_monetdb.build_parser(add_help=False)
    args = parser.parse_args(["--conv", "basef1-9v1"])

    assert not hasattr(tpch_monetdb.run_gen_base_impl_tpch_monetdb, "ensure_tables_ready")
    assert not hasattr(args, "data_prepare_mode")
    return None


def test_compaction_prompt_keeps_l2_state_only() -> None:
    assert "### 1. Current Goal" in COMPACT_SYSTEM_PROMPT
    assert "### 4. TODO And Blockers" in COMPACT_SYSTEM_PROMPT
    assert "Do not add rules reminders" in COMPACT_SYSTEM_PROMPT
    assert "### 1. Primary Task" not in COMPACT_SYSTEM_PROMPT


def test_main_writes_stage_summary_before_auto_compact(tmp_path, monkeypatch) -> None:
    order: list[str] = []
    summary_items: list[dict[str, str]] = []

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
            self.callback = kwargs["callback"]

        async def run(self) -> None:
            await self.callback(
                "prompt",
                "compile_fix",
                0,
                4,
                {"tool_profile": "compile_fix", "rule_area": "runtime"},
            )
            return None

    class FakeSession:
        def __init__(self, *args, **kwargs) -> None:
            return None

        async def add_items(self, items) -> None:
            order.append("add_items")
            summary_items.extend(items)
            return None

    class FakeAutoCompactManager:
        def __init__(self, *_args, **_kwargs) -> None:
            self.consecutive_failures = 0
            self.last_failure_info = None
            self.warning_threshold = 1
            self.threshold = 1
            self.blocking_threshold = 999_999
            return None

        def get_threshold(self) -> int:
            return 1

        def get_warning_threshold(self, _profile_name=None) -> int:
            return 1

        def get_blocking_threshold(self) -> int:
            return self.blocking_threshold

        def should_compact(self, _token_usage: int, _profile_name=None) -> bool:
            return True

        async def estimate_request_tokens(self, session=None, new_input: str = "") -> int:
            return 0

        async def compact(self, **_kwargs) -> None:
            order.append("compact")
            return None

    class FakeRuntime:
        def activate(self, profile_name, prompt_index, prompt_descriptor) -> None:
            self.profile_name = profile_name
            self.prompt_index = prompt_index
            self.prompt_descriptor = prompt_descriptor
            return None

        def generate_stage_hint(self) -> str:
            return "[Stage Context]\nStage: compile_fix"

        def finish_stage(self, final_output: str | None) -> StageRunSummary:
            return StageRunSummary(
                profile_name="compile_fix",
                prompt_index=0,
                prompt_descriptor="compile_fix",
                final_output=final_output,
                tool_counts={"compile": 1},
                written_files=("query_impl.cpp",),
                last_compile_summary="compile ok",
                last_run_summary=None,
                todo_before=None,
                todo_after=None,
            )

    async def fake_runner_run(*_args, **_kwargs) -> SimpleNamespace:
        return SimpleNamespace(
            final_output="done",
            context_wrapper=SimpleNamespace(
                usage=SimpleNamespace(
                    total_tokens=123,
                    input_tokens=1,
                    output_tokens=1,
                    reasoning_tokens=0,
                )
            ),
        )

    fake_tool_bundle = SimpleNamespace(
        all_tools=[],
        tools_by_profile={"compile_fix": []},
        runtime=FakeRuntime(),
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
    monkeypatch.setattr(tpch_monetdb.main_tpch_monetdb, "write_query_and_args_file", lambda **_kwargs: "")
    monkeypatch.setattr(tpch_monetdb.main_tpch_monetdb, "setup_model_config", lambda _model: fake_model_config)
    from tpch_monetdb.llm_cache import cached_openai as _cached_openai_mod
    monkeypatch.setattr(_cached_openai_mod, "CachedOpenAIResponsesModel", lambda **_kwargs: SimpleNamespace(total_saved=0))
    monkeypatch.setattr(tpch_monetdb.main_tpch_monetdb, "_create_compaction_session", lambda **_kwargs: SimpleNamespace())
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
        lambda **_kwargs: SimpleNamespace(tools=[], name="agent", instructions=""),
    )
    monkeypatch.setattr(
        tpch_monetdb.main_tpch_monetdb,
        "Runner",
        SimpleNamespace(run=fake_runner_run),
    )
    monkeypatch.setattr(tpch_monetdb.main_tpch_monetdb, "trace", lambda *args, **kwargs: contextlib.nullcontext())
    monkeypatch.setattr(tpch_monetdb.main_tpch_monetdb, "ScriptedConversation", FakeConversation)
    monkeypatch.setattr(tpch_monetdb.main_tpch_monetdb, "AdvancedSQLiteSession", FakeSession)
    monkeypatch.setattr(tpch_monetdb.main_tpch_monetdb, "AutoCompactManager", FakeAutoCompactManager)
    monkeypatch.setattr(tpch_monetdb.main_tpch_monetdb, "get_tokens_context_and_dollar_info", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(tpch_monetdb.main_tpch_monetdb, "truncate_model_final_output", lambda _value: "")
    monkeypatch.setattr(tpch_monetdb.main_tpch_monetdb, "persist_successful_scripted_run", lambda **_kwargs: None)
    monkeypatch.setattr(tpch_monetdb.main_tpch_monetdb, "_run_final_correctness_gate", lambda **_kwargs: None)
    monkeypatch.setattr(tpch_monetdb.main_tpch_monetdb, "_run_base_impl_promotion_gate", lambda **_kwargs: None)

    args = SimpleNamespace(
        benchmark="tpch",
        conv_name="tpch_monetdb_basef1-9v1",
        query_list="1",
        storage_plan_snapshot=None,
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
        enable_auto_compact=True,
        disable_tracing=True,
        max_scale_factor=100,
        base_data_dir=str(tmp_path / "data"),
        validation_mode="strict",
    )

    asyncio.run(tpch_monetdb.main_tpch_monetdb.main(args))

    assert order == ["add_items", "compact"]
    assert summary_items
    assert summary_items[0]["role"] == "assistant"
    assert "[Stage Summary]" in summary_items[0]["content"]


def test_main_compaction_marker_preserves_recent_items(tmp_path, monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeSnapshotter:
        def __init__(self, *args, **kwargs) -> None:
            self.current_hash = "seed-hash"

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
            self.callback = kwargs["callback"]

        async def run(self) -> None:
            await self.callback(
                tpch_monetdb.main_tpch_monetdb.COMPACTION_MARKER,
                "compact",
                12,
                4,
                {"tool_profile": "compile_fix", "rule_area": "runtime"},
            )
            return None

    class FakeCompactionSession:
        def set_underlying_session(self, _session) -> None:
            return None

        async def run_compaction(self, args) -> None:
            captured["args"] = args
            return None

    class FakeUnderlyingSession:
        def __init__(self, *args, **kwargs) -> None:
            return None

        async def add_items(self, _items) -> None:
            return None

    fake_tool_bundle = SimpleNamespace(
        all_tools=[],
        tools_by_profile={"compile_fix": []},
        runtime=SimpleNamespace(),
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
    monkeypatch.setattr(tpch_monetdb.main_tpch_monetdb, "write_query_and_args_file", lambda **_kwargs: "")
    monkeypatch.setattr(tpch_monetdb.main_tpch_monetdb, "setup_model_config", lambda _model: fake_model_config)
    from tpch_monetdb.llm_cache import cached_openai as _cached_openai_mod
    monkeypatch.setattr(_cached_openai_mod, "CachedOpenAIResponsesModel", lambda **_kwargs: SimpleNamespace(total_saved=0))
    monkeypatch.setattr(
        tpch_monetdb.main_tpch_monetdb,
        "_create_compaction_session",
        lambda **_kwargs: FakeCompactionSession(),
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
        lambda **_kwargs: SimpleNamespace(tools=[], name="agent", instructions=""),
    )
    monkeypatch.setattr(tpch_monetdb.main_tpch_monetdb, "trace", lambda *args, **kwargs: contextlib.nullcontext())
    monkeypatch.setattr(tpch_monetdb.main_tpch_monetdb, "ScriptedConversation", FakeConversation)
    monkeypatch.setattr(tpch_monetdb.main_tpch_monetdb, "AdvancedSQLiteSession", FakeUnderlyingSession)
    monkeypatch.setattr(tpch_monetdb.main_tpch_monetdb, "get_tokens_context_and_dollar_info", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(tpch_monetdb.main_tpch_monetdb, "truncate_model_final_output", lambda _value: "")
    monkeypatch.setattr(tpch_monetdb.main_tpch_monetdb, "persist_successful_scripted_run", lambda **_kwargs: None)
    monkeypatch.setattr(tpch_monetdb.main_tpch_monetdb, "_run_final_correctness_gate", lambda **_kwargs: None)
    monkeypatch.setattr(tpch_monetdb.main_tpch_monetdb, "_run_base_impl_promotion_gate", lambda **_kwargs: None)

    args = SimpleNamespace(
        benchmark="tpch",
        conv_name="tpch_monetdb_basef1-9v1",
        query_list="1",
        storage_plan_snapshot=None,
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
        validation_mode="strict",
    )

    asyncio.run(tpch_monetdb.main_tpch_monetdb.main(args))

    assert captured["args"] == {
        "force_trigger": True,
        "compaction_mode": "input",
        "selection_policy": "stage_memory_v3",
        "preserve_limit_items": 12,
        "min_candidate_items": 0,
    }


def test_main_reinjects_rules_and_stage_hint_after_compaction(tmp_path, monkeypatch) -> None:
    load_calls: list[tuple[str | None, str | None, bool]] = []
    compose_calls: list[tuple[bool, bool]] = []
    hint_calls: list[str] = []
    runner_inputs: list[str] = []
    runner_instructions: list[str] = []

    class FakeSnapshotter:
        def __init__(self, *args, **kwargs) -> None:
            self.current_hash = "seed-hash"

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
            self.callback = kwargs["callback"]

        async def run(self) -> None:
            await self.callback(
                "prompt-1",
                "compile_fix",
                0,
                4,
                {"tool_profile": "compile_fix", "rule_area": "runtime"},
            )
            await self.callback(
                tpch_monetdb.main_tpch_monetdb.COMPACTION_MARKER,
                "compaction",
                1,
                4,
                {"tool_profile": "compile_fix", "rule_area": "runtime"},
            )
            await self.callback(
                "prompt-2",
                "compile_fix",
                2,
                4,
                {"tool_profile": "compile_fix", "rule_area": "runtime"},
            )
            return None

    class FakeCompactionSession:
        def set_underlying_session(self, _session) -> None:
            return None

        async def run_compaction(self, _args) -> None:
            return None

    class FakeUnderlyingSession:
        def __init__(self, *args, **kwargs) -> None:
            return None

        async def add_items(self, _items) -> None:
            return None

    class FakeRuntime:
        def activate(self, profile_name, prompt_index, prompt_descriptor) -> None:
            self.profile_name = profile_name
            return None

        def generate_stage_hint(self) -> str:
            hint_calls.append(self.profile_name)
            return f"[Stage Context]\\nStage: {self.profile_name}"

        def finish_stage(self, final_output: str | None) -> StageRunSummary:
            return StageRunSummary(
                profile_name="compile_fix",
                prompt_index=0,
                prompt_descriptor="compile_fix",
                final_output=final_output,
                tool_counts={"compile": 1},
                written_files=("query_impl.cpp",),
                last_compile_summary="compile ok",
                last_run_summary=None,
                todo_before=None,
                todo_after=None,
            )

    async def fake_runner_run(*_args, **_kwargs) -> SimpleNamespace:
        runner_inputs.append(_kwargs["input"])
        runner_instructions.append(_args[0].instructions)
        return SimpleNamespace(
            final_output="done",
            context_wrapper=SimpleNamespace(
                usage=SimpleNamespace(
                    total_tokens=10,
                    input_tokens=1,
                    output_tokens=1,
                    reasoning_tokens=0,
                )
            ),
        )

    fake_tool_bundle = SimpleNamespace(
        all_tools=[],
        tools_by_profile={"compile_fix": []},
        runtime=FakeRuntime(),
    )
    fake_model_config = SimpleNamespace(
        use_litellm=False,
        accounting_model_name="gpt-test",
        model_name="gpt-test",
        openai_client=None,
        api_key=None,
        base_url=None,
    )

    def fake_load_agent_rules(
        tpch_monetdb_root,
        *,
        stage_name,
        area_name,
        candidate_paths=(),
        include_global,
        token_budget=0,
    ) -> RuleAssembly:
        load_calls.append((stage_name, area_name, include_global))
        text = (
            "global-rule"
            if include_global
            else f"scoped-rule stage={stage_name} area={area_name}"
        )
        return RuleAssembly(
            text=text,
            included_files=(),
            truncated_files=(),
            excluded_files=(),
            char_budget=0,
        )

    def fake_compose_agent_instructions(
        base_instructions,
        global_rules,
        scoped_rules=None,
    ) -> str:
        compose_calls.append((bool(global_rules.text), bool(scoped_rules and scoped_rules.text)))
        return "instructions"

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
    monkeypatch.setattr(tpch_monetdb.main_tpch_monetdb, "setup_model_config", lambda _model: fake_model_config)
    monkeypatch.setattr(
        tpch_monetdb.main_tpch_monetdb,
        "_create_compaction_session",
        lambda **_kwargs: FakeCompactionSession(),
    )
    monkeypatch.setattr(tpch_monetdb.main_tpch_monetdb, "_load_agent_rules", fake_load_agent_rules)
    monkeypatch.setattr(tpch_monetdb.main_tpch_monetdb, "_compose_agent_instructions", fake_compose_agent_instructions)
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
        lambda **_kwargs: SimpleNamespace(
            tools=_kwargs["tools"],
            name=_kwargs["name"],
            instructions=_kwargs["instructions"],
        ),
    )
    monkeypatch.setattr(
        tpch_monetdb.main_tpch_monetdb,
        "Runner",
        SimpleNamespace(run=fake_runner_run),
    )
    monkeypatch.setattr(tpch_monetdb.main_tpch_monetdb, "trace", lambda *args, **kwargs: contextlib.nullcontext())
    monkeypatch.setattr(tpch_monetdb.main_tpch_monetdb, "ScriptedConversation", FakeConversation)
    monkeypatch.setattr(tpch_monetdb.main_tpch_monetdb, "AdvancedSQLiteSession", FakeUnderlyingSession)
    monkeypatch.setattr(tpch_monetdb.main_tpch_monetdb, "get_tokens_context_and_dollar_info", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(tpch_monetdb.main_tpch_monetdb, "truncate_model_final_output", lambda _value: "")
    monkeypatch.setattr(tpch_monetdb.main_tpch_monetdb, "persist_successful_scripted_run", lambda **_kwargs: None)
    monkeypatch.setattr(tpch_monetdb.main_tpch_monetdb, "_run_final_correctness_gate", lambda **_kwargs: None)
    monkeypatch.setattr(tpch_monetdb.main_tpch_monetdb, "_run_base_impl_promotion_gate", lambda **_kwargs: None)

    args = SimpleNamespace(
        benchmark="tpch",
        conv_name="tpch_monetdb_basef1-9v1",
        query_list="1",
        storage_plan_snapshot=None,
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
        validation_mode="strict",
    )

    asyncio.run(tpch_monetdb.main_tpch_monetdb.main(args))

    assert load_calls[0] == (None, None, True)
    assert load_calls.count(("compile_fix", "runtime", False)) == 2
    assert compose_calls == [(True, False)]
    assert hint_calls == ["compile_fix", "compile_fix"]
    assert runner_instructions == ["instructions", "instructions"]
    assert all("scoped-rule" not in instructions for instructions in runner_instructions)
    assert len(runner_inputs) == 2
    assert all("[Scoped Stage Rules]" in item for item in runner_inputs)
    assert all(
        "scoped-rule stage=compile_fix area=runtime" in item
        for item in runner_inputs
    )


def test_main_start_snapshot_skips_query_impl_rewrite(tmp_path, monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeSnapshotter:
        def __init__(self, working_dir: Path) -> None:
            captured["working_dir"] = working_dir
            self.current_hash = "restored-hash"

        def is_dirty(self) -> bool:
            return False

        def has_snapshot(self, _snapshot: str) -> bool:
            return True

        def restore(self, _snapshot: str) -> None:
            captured["restored"] = True
            return None

        def clean_worktree(self, include_ignored: bool = True) -> None:
            return None

        def recreate_repo(self) -> None:
            return None

    class FakeConversation:
        def __init__(self, **_kwargs) -> None:
            return None

        async def run(self) -> None:
            return None

    fake_tool_bundle = SimpleNamespace(
        all_tools=[],
        tools_by_profile={},
        runtime=SimpleNamespace(),
    )
    fake_model_config = SimpleNamespace(
        use_litellm=False,
        accounting_model_name="gpt-test",
        model_name="gpt-test",
        openai_client=None,
        api_key=None,
        base_url=None,
    )

    monkeypatch.setattr(
        tpch_monetdb.main_tpch_monetdb,
        "resolve_runtime_workspace_path",
        lambda _tpch_monetdb_root: ROOT / "output",
    )
    monkeypatch.setattr(
        tpch_monetdb.main_tpch_monetdb,
        "build_runtime_snapshotter",
        lambda _tpch_monetdb_root, disable_repo_sync, keep_csv: FakeSnapshotter(ROOT / "output"),
    )
    monkeypatch.setattr(tpch_monetdb.main_tpch_monetdb, "get_placeholders_fn", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        tpch_monetdb.main_tpch_monetdb,
        "copy_template_to",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("copy_template_to should not run")),
    )
    def _record_query_args(**kwargs):
        """phase10 起 bootstrap 不再向 dispatcher 注入示例；记录 out_dir 以便断言."""
        captured.setdefault("out_dir", kwargs["out_dir"])
        captured.setdefault("query_artifact_call", True)
        return ""

    monkeypatch.setattr(
        tpch_monetdb.main_tpch_monetdb,
        "write_query_and_args_file",
        _record_query_args,
    )
    monkeypatch.setattr(tpch_monetdb.main_tpch_monetdb, "setup_model_config", lambda _model: fake_model_config)
    from tpch_monetdb.llm_cache import cached_openai as _cached_openai_mod
    monkeypatch.setattr(_cached_openai_mod, "CachedOpenAIResponsesModel", lambda **_kwargs: SimpleNamespace(total_saved=0))
    monkeypatch.setattr(tpch_monetdb.main_tpch_monetdb, "_create_compaction_session", lambda **_kwargs: SimpleNamespace())
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
        lambda **_kwargs: SimpleNamespace(tools=[], name="agent", instructions=""),
    )
    monkeypatch.setattr(tpch_monetdb.main_tpch_monetdb, "trace", lambda *args, **kwargs: contextlib.nullcontext())
    monkeypatch.setattr(tpch_monetdb.main_tpch_monetdb, "ScriptedConversation", FakeConversation)
    monkeypatch.setattr(
        tpch_monetdb.main_tpch_monetdb,
        "AdvancedSQLiteSession",
        lambda *args, **kwargs: SimpleNamespace(add_items=lambda *_a, **_k: None),
    )
    monkeypatch.setattr(tpch_monetdb.main_tpch_monetdb, "get_tokens_context_and_dollar_info", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(tpch_monetdb.main_tpch_monetdb, "truncate_model_final_output", lambda _value: "")
    monkeypatch.setattr(tpch_monetdb.main_tpch_monetdb, "persist_successful_scripted_run", lambda **_kwargs: None)
    monkeypatch.setattr(tpch_monetdb.main_tpch_monetdb, "_run_final_correctness_gate", lambda **_kwargs: None)
    monkeypatch.setattr(tpch_monetdb.main_tpch_monetdb, "_run_base_impl_promotion_gate", lambda **_kwargs: None)
    monkeypatch.setattr(tpch_monetdb.main_tpch_monetdb, "_validate_resume_contract_fields", lambda **_kwargs: None)

    args = SimpleNamespace(
        benchmark="tpch",
        conv_name="tpch_monetdb_basef1-9v1",
        query_list="1",
        storage_plan_snapshot=None,
        start_snapshot="seed-hash",
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
        validation_mode="strict",
    )

    asyncio.run(tpch_monetdb.main_tpch_monetdb.main(args))

    assert captured["restored"] is True
    assert captured["query_artifact_call"] is True
    assert Path(captured["out_dir"]) == ROOT / "output"
    assert Path(captured["working_dir"]) == ROOT / "output"


def test_wandb_init_retry_switches_to_new_run_id_on_failure() -> None:
    from tpch_monetdb.utils.wandb_init import WandbInitResult
    calls: list[dict[str, object]] = []

    def fake_init(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise RuntimeError(
                "run deadbeef was previously created and deleted; try a new run id"
            )
        return SimpleNamespace(id=kwargs["id"], name=kwargs["name"])

    args = SimpleNamespace(conv_name="tpch_monetdb_storageplan1-15v4_r001")
    result = init_wandb_run_with_retry(
        wandb_module=SimpleNamespace(init=fake_init),
        args=args,
        entity="test-entity",
        project="test-project",
        tags=["tpch_monetdb"],
        max_attempts=3,
    )

    assert isinstance(result, WandbInitResult)
    assert len(calls) == 2
    assert calls[0]["id"] != calls[1]["id"]
    assert calls[0]["resume"] == "allow"
    assert calls[1]["resume"] == "never"
    assert result.run.id == calls[1]["id"]
    assert result.used_fallback is True
    assert result.attempt_count == 2
    assert result.first_failure_excerpt is not None
    assert "previously created and deleted" in result.first_failure_excerpt


def test_wandb_init_retry_returns_full_error_after_exhaustion() -> None:
    calls: list[dict[str, object]] = []

    def always_fail(**kwargs):
        calls.append(kwargs)
        raise RuntimeError(
            "run deadbeef was previously created and deleted; try a new run id"
        )

    args = SimpleNamespace(conv_name="tpch_monetdb_storageplan1-15v4_r001")
    with pytest.raises(RuntimeError) as exc_info:
        init_wandb_run_with_retry(
            wandb_module=SimpleNamespace(init=always_fail),
            args=args,
            entity="test-entity",
            project="test-project",
            tags=["tpch_monetdb"],
            max_attempts=3,
        )

    message = str(exc_info.value)
    assert len(calls) == 3
    assert "[ERROR:WANDB_INIT_FAILED]" in message
    assert "attempted_run_ids" in message
    assert "attempt_count=3" in message
    assert "first_failure_excerpt" in message
    assert "previously created and deleted" in message
    # Full tracebacks for attempt 2 and 3 must NOT be in the envelope
    assert "Attempt 2/3" not in message
    assert "Attempt 3/3" not in message


def test_wandb_init_first_attempt_success_records_no_fallback() -> None:
    from tpch_monetdb.utils.wandb_init import WandbInitResult
    calls: list[dict[str, object]] = []

    def fake_init(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(id=kwargs["id"], name=kwargs["name"])

    args = SimpleNamespace(conv_name="tpch_monetdb_storageplan1-2v1_r001")
    result = init_wandb_run_with_retry(
        wandb_module=SimpleNamespace(init=fake_init),
        args=args,
        entity="e",
        project="p",
        tags=[],
        max_attempts=3,
    )

    assert isinstance(result, WandbInitResult)
    assert len(calls) == 1
    assert result.attempt_count == 1
    assert result.used_fallback is False
    assert result.first_failure_excerpt is None
    assert result.primary_run_id == result.final_run_id
    assert result.attempted_run_ids == [result.primary_run_id]


def test_cached_compiler_requires_matching_artifact_fingerprints(tmp_path) -> None:
    from tpch_monetdb.misc.tpch.compiler_cached import CachedCompiler, CompileCacheType

    compiler = object.__new__(CachedCompiler)
    compiler.workdir = tmp_path
    compiler.app_name = "db"
    compiler.build_dir_path = tmp_path / "build"
    compiler.libs = {"query": object()}
    compiler.build_dir_path.mkdir()
    (tmp_path / "db").write_text("app-v1\n", encoding="utf-8")
    (compiler.build_dir_path / "libquery.so").write_text("lib-v1\n", encoding="utf-8")

    fingerprints = compiler._artifact_fingerprints()
    cached = CompileCacheType(outputs=None, artifact_fingerprints=fingerprints)

    assert compiler._artifacts_available() is True
    assert compiler._cached_artifacts_match(cached) is True

    (compiler.build_dir_path / "libquery.so").write_text(
        "lib-v2-changed\n",
        encoding="utf-8",
    )

    assert compiler._cached_artifacts_match(cached) is False
    assert compiler._cached_artifacts_match(CompileCacheType(outputs=None)) is False
    return None


def test_run_tool_recovers_on_first_broken_pipe(monkeypatch, tmp_path) -> None:
    """First broken pipe triggers fresh-runner replay and succeeds."""
    from tpch_monetdb.misc.tpch.fasttest_proc import RunnerTransportError
    from tpch_monetdb.tools.tpch import run as run_module
    from tpch_monetdb.tools.tpch.pool import FastTestPool

    call_count = {"send": 0}

    class FakeRunner:
        def __init__(self, fail_first: bool) -> None:
            self.fail_first = fail_first
            self.send_calls: list[str] = []

        def send(self, line: str) -> None:
            self.send_calls.append(line)
            call_count["send"] += 1
            if self.fail_first and call_count["send"] <= 2:
                raise RunnerTransportError("broken pipe")

        def run(self, timeout: int = 0) -> tuple[str, str, str]:
            return "ok", "out", "err"

        def run_batch(
            self,
            args_list: list[str],
            timeout: int = 0,
        ) -> tuple[str, str, str]:
            for line in args_list:
                self.send(line)
            return self.run(timeout=timeout)

    original_get = FastTestPool.get
    original_terminate = FastTestPool.terminate

    runners: list[FakeRunner] = []

    def fake_get(key, factory):
        if not runners:
            runners.append(FakeRunner(fail_first=True))
        return runners[-1]

    def fake_terminate(key):
        runners.append(FakeRunner(fail_first=False))
        return True

    class FakeCompiler:
        def set_extra_cxxflags(self, _flags) -> None:
            return None

        def build_cached(self, **_kwargs):
            return None, False, "hash"

    monkeypatch.setattr(FastTestPool, "get", fake_get)
    monkeypatch.setattr(FastTestPool, "terminate", fake_terminate)
    monkeypatch.setattr(run_module, "make_compiler", lambda *_a, **_k: FakeCompiler())

    run_tool = run_module.RunTool(
        cwd=tmp_path,
            dataset_name="tpch",
        base_data_dir=str(tmp_path),
        query_validator=None,
    )

    msg, metrics = run_tool.run(
        scale_factor=1,
        optimize=False,
        query_id=["1"],
    )

    assert len(runners) == 2
    assert "ok" in msg

    monkeypatch.setattr(FastTestPool, "get", original_get)
    monkeypatch.setattr(FastTestPool, "terminate", original_terminate)
    return None


def test_run_tool_returns_runner_broken_pipe_on_persistent_failure(monkeypatch, tmp_path) -> None:
    """Persistent broken pipe after replay returns structured failure metrics."""
    from tpch_monetdb.misc.tpch.fasttest_proc import (
        RunnerTransportError,
    )
    from tpch_monetdb.tools.tpch import run as run_module
    from tpch_monetdb.tools.tpch.pool import FastTestPool

    class AlwaysBrokenRunner:
        def send(self, line: str) -> None:
            raise RunnerTransportError("always broken")

        def run(self, timeout: int = 0) -> tuple[str, str, str]:
            return "ok", "out", "err"

        def run_batch(
            self,
            args_list: list[str],
            timeout: int = 0,
        ) -> tuple[str, str, str]:
            for line in args_list:
                self.send(line)
            return self.run(timeout=timeout)

    monkeypatch.setattr(FastTestPool, "get", lambda _k, _f: AlwaysBrokenRunner())
    monkeypatch.setattr(FastTestPool, "terminate", lambda _k: True)
    monkeypatch.setattr(
        run_module,
        "make_compiler",
        lambda *_a, **_k: SimpleNamespace(
            set_extra_cxxflags=lambda _flags: None,
            build_cached=lambda **_kwargs: (None, False, "hash"),
        ),
    )

    run_tool = run_module.RunTool(
        cwd=tmp_path,
            dataset_name="tpch",
        base_data_dir=str(tmp_path),
        query_validator=None,
    )

    msg, metrics = run_tool.run(
        scale_factor=1,
        optimize=False,
        query_id=["1"],
    )

    assert "[ERROR:RUNNER_BROKEN_PIPE]" in msg
    assert metrics is not None
    assert metrics["validation/failure_code"] == "RUNNER_BROKEN_PIPE"
    assert "[ERROR:RUNNER_BROKEN_PIPE]" in metrics["validation/failure_detail"]
    return None


def test_run_tool_does_not_replay_on_non_transport_exceptions(monkeypatch, tmp_path) -> None:
    """Non-transport exceptions must not trigger runner replay."""
    from tpch_monetdb.tools.tpch import run as run_module
    from tpch_monetdb.tools.tpch.pool import FastTestPool

    class FileNotFoundRunner:
        def send(self, line: str) -> None:
            raise FileNotFoundError("./db not found")

        def run(self, timeout: int = 0) -> tuple[str, str, str]:
            return "ok", "out", "err"

        def run_batch(
            self,
            args_list: list[str],
            timeout: int = 0,
        ) -> tuple[str, str, str]:
            for line in args_list:
                self.send(line)
            return self.run(timeout=timeout)

    monkeypatch.setattr(FastTestPool, "get", lambda _k, _f: FileNotFoundRunner())
    monkeypatch.setattr(FastTestPool, "terminate", lambda _k: True)
    monkeypatch.setattr(
        run_module,
        "make_compiler",
        lambda *_a, **_k: SimpleNamespace(
            set_extra_cxxflags=lambda _flags: None,
            build_cached=lambda **_kwargs: (None, False, "hash"),
        ),
    )

    run_tool = run_module.RunTool(
        cwd=tmp_path,
            dataset_name="tpch",
        base_data_dir=str(tmp_path),
        query_validator=None,
    )

    with pytest.raises(FileNotFoundError, match="./db not found"):
        run_tool.run(
            scale_factor=1,
            optimize=False,
            query_id=["1"],
        )


def test_deduplicate_message_filter_passes_first_suppresses_repeat() -> None:
    import logging
    from tpch_monetdb.llm_cache.logger import DeduplicateMessageFilter

    records: list[logging.LogRecord] = []

    f = DeduplicateMessageFilter("test", [r"cost map.*timeout"])
    for i in range(3):
        r = logging.LogRecord("test", logging.WARNING, "", 0, "cost map fetch timeout", (), None)
        if f.filter(r):
            records.append(r)

    assert len(records) == 1


def test_deduplicate_message_filter_passes_distinct_messages() -> None:
    import logging
    from tpch_monetdb.llm_cache.logger import DeduplicateMessageFilter

    f = DeduplicateMessageFilter("test", [r"cost map.*timeout"])
    msgs_passed = 0
    for suffix in ["timeout: attempt 1", "timeout: attempt 2", "timeout: attempt 3"]:
        r = logging.LogRecord("test", logging.WARNING, "", 0, f"cost map fetch {suffix}", (), None)
        if f.filter(r):
            msgs_passed += 1

    assert msgs_passed == 3


def test_wandb_exhaustion_envelope_does_not_contain_full_tracebacks() -> None:
    from types import SimpleNamespace
    calls: list = []

    def always_fail(**kwargs):
        calls.append(kwargs)
        raise ValueError("boom error for attempt")

    args = SimpleNamespace(conv_name="test_conv")
    with pytest.raises(RuntimeError) as exc_info:
        init_wandb_run_with_retry(
            wandb_module=SimpleNamespace(init=always_fail),
            args=args,
            entity="e",
            project="p",
            tags=[],
            max_attempts=3,
        )

    msg = str(exc_info.value)
    assert "first_failure_excerpt" in msg
    assert "attempt_count=3" in msg
    assert "attempted_run_ids" in msg
    assert "boom error for attempt" in msg
    # Old attempt-line format (full dump of all attempts) must not appear
    assert "Attempt 2/3" not in msg
    assert "Attempt 3/3" not in msg


def test_wandb_init_retry_returns_timeout_code_after_exhaustion() -> None:
    import time
    from types import SimpleNamespace

    def slow_init(**_kwargs):
        time.sleep(0.05)
        return None

    args = SimpleNamespace(conv_name="test_conv")
    with pytest.raises(RuntimeError) as exc_info:
        init_wandb_run_with_retry(
            wandb_module=SimpleNamespace(init=slow_init),
            args=args,
            entity="e",
            project="p",
            tags=[],
            max_attempts=2,
            init_timeout_s=0.01,
        )

    msg = str(exc_info.value)
    assert "[ERROR:WANDB_INIT_TIMEOUT]" in msg
    assert "attempt_count=2" in msg
