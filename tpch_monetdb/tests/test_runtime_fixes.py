import json
import sys
from pathlib import Path

import pytest
from agents.run_context import RunContextWrapper
from agents.tool import FunctionTool

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tpch_monetdb.tools.tool_factory import build_tools
from tpch_monetdb.tools.tpch_monetdb_agent_tools import build_tpch_monetdb_agent_tools
from tpch_monetdb.tools.tpch.utils import copy_template_to
from tpch_monetdb.utils.general_utils import gen_tpch_args_str


def _make_run_tool() -> FunctionTool:
    async def on_invoke(_ctx: RunContextWrapper[object], _args_json: str) -> str:
        return "ok"

    return FunctionTool(
        name="run",
        description="run",
        params_json_schema={"type": "object", "properties": {}},
        on_invoke_tool=on_invoke,
    )


@pytest.mark.asyncio
async def test_apply_patch_success_counts_as_stage_write(tmp_path) -> None:
    target = tmp_path / "loader_impl.cpp"
    target.write_text("int value = 1;\n", encoding="utf-8")

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


@pytest.mark.asyncio
async def test_run_tool_output_is_truncated_before_returning_to_agent(tmp_path) -> None:
    huge_trace = "PROFILE query_q1_scan 1\n" * 4000

    async def compile_invoke(_ctx: RunContextWrapper[object], _args_json: str) -> str:
        return "**Compilation successfull**"

    async def run_invoke(_ctx: RunContextWrapper[object], _args_json: str) -> str:
        return (
            "All queries passed validation!\n"
            f"stdout: {huge_trace}\n"
            "stderr: \n"
            "exit_code: 0 signal: 0"
        )

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

    bundle = build_tpch_monetdb_agent_tools(
        workspace_path=tmp_path,
        cache_path=tmp_path / "cache",
        compile_tool=compile_tool,
        run_tool=run_tool,
        git_snapshotter=None,
        wandb_metrics_hook=None,
        apply_patch_tool=None,
    )
    bundle.runtime.activate(
        "optimization_instrumentation",
        0,
        "Add Timings for Queries 1",
        prompt_metadata={"active_query_ids": ["1"]},
    )
    run_wrapper = next(
        tool
        for tool in bundle.tools_by_profile["optimization_instrumentation"]
        if tool.name == "run"
    )

    result = await run_wrapper.on_invoke_tool(
        RunContextWrapper(context=None),
        json.dumps(
            {
                "scale_factor": 1000,
                "optimize": True,
                "query_id": ["1"],
                "trace_mode": True,
            }
        ),
    )

    assert result.startswith("[Evidence]\nAll queries passed validation!")
    assert "chars truncated" in result
    assert len(result) < 25000
    assert "exit_code: 0 signal: 0" in result


@pytest.mark.asyncio
async def test_repeated_compile_without_write_returns_stalled_stage(tmp_path) -> None:
    async def compile_invoke(_ctx, _args_json: str) -> str:
        return "**Compilation successfull**"

    compile_tool = FunctionTool(
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
    third = await compile_wrapper.on_invoke_tool(
        RunContextWrapper(context=None),
        json.dumps({}),
    )

    assert "[Evidence]\n**Compilation successfull**" == first
    assert "[Evidence]\n**Compilation successfull**" == second
    assert "[ERROR:STALLED_STAGE]" in third
    assert "without any intervening write" in third


@pytest.mark.asyncio
async def test_list_files_accepts_workspace_root_slash(tmp_path) -> None:
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
    list_tool = next(
        tool
        for tool in bundle.tools_by_profile["finish_skeleton"]
        if tool.name == "list_files"
    )

    result = await list_tool.on_invoke_tool(
        RunContextWrapper(context=None),
        json.dumps({"path": "/", "limit": 20}),
    )

    assert "loader_impl.cpp" in result


@pytest.mark.asyncio
async def test_read_only_tools_path_outside_workspace_is_recoverable(tmp_path) -> None:
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

    list_tool = next(
        tool
        for tool in bundle.tools_by_profile["finish_skeleton"]
        if tool.name == "list_files"
    )
    read_tool = next(
        tool
        for tool in bundle.tools_by_profile["finish_skeleton"]
        if tool.name == "read_file"
    )
    grep_tool = next(
        tool
        for tool in bundle.tools_by_profile["finish_skeleton"]
        if tool.name == "grep_repo"
    )

    list_result = await list_tool.on_invoke_tool(
        RunContextWrapper(context=None),
        json.dumps({"path": "..", "limit": 20}),
    )
    assert "[ERROR:PATH_OUTSIDE_WORKSPACE]" in list_result
    assert "Recoverable: yes" in list_result

    read_result = await read_tool.on_invoke_tool(
        RunContextWrapper(context=None),
        json.dumps({"file_path": "../loader_impl.cpp"}),
    )
    assert "[ERROR:PATH_OUTSIDE_WORKSPACE]" in read_result
    assert "Recoverable: yes" in read_result

    grep_result = await grep_tool.on_invoke_tool(
        RunContextWrapper(context=None),
        json.dumps({"pattern": "value", "path": "..", "limit": 20}),
    )
    assert "[ERROR:PATH_OUTSIDE_WORKSPACE]" in grep_result
    assert "Recoverable: yes" in grep_result


@pytest.mark.asyncio
async def test_read_only_tools_accept_configured_external_data_root(tmp_path) -> None:
    data_root = tmp_path / "data"
    (data_root / "sf1").mkdir(parents=True, exist_ok=True)
    (data_root / "sf1" / "customer.tbl").write_text("1|customer|\n", encoding="utf-8")

    bundle = build_tools(
        use_litellm=True,
        workspace_path=tmp_path,
        cache_path=tmp_path / "cache",
        extra_read_roots=(data_root,),
        shell_executor=None,
        workspace_editor=None,
        compile_cache_dir=tmp_path / "compile",
        run_tool_wrapper=_make_run_tool(),
        git_snapshotter=None,
        wandb_metrics_hook=None,
    )

    bundle.runtime.activate("storage_plan", 0, "storage_plan")

    list_tool = next(
        tool
        for tool in bundle.tools_by_profile["storage_plan"]
        if tool.name == "list_files"
    )
    read_tool = next(
        tool
        for tool in bundle.tools_by_profile["storage_plan"]
        if tool.name == "read_file"
    )

    list_result = await list_tool.on_invoke_tool(
        RunContextWrapper(context=None),
        json.dumps({"path": str(data_root), "limit": 20}),
    )
    assert "sf1/" in list_result

    read_result = await read_tool.on_invoke_tool(
        RunContextWrapper(context=None),
        json.dumps({"file_path": str(data_root / "sf1" / "customer.tbl")}),
    )
    assert "1|customer|" in read_result


@pytest.mark.asyncio
async def test_edit_path_outside_workspace_remains_fatal(tmp_path) -> None:
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
    edit_tool = next(
        tool
        for tool in bundle.tools_by_profile["finish_skeleton"]
        if tool.name == "edit_file"
    )

    with pytest.raises(Exception, match="PATH_OUTSIDE_WORKSPACE"):
        await edit_tool.on_invoke_tool(
            RunContextWrapper(context=None),
            json.dumps(
                {
                    "file_path": "../loader_impl.cpp",
                    "old_string": "int value = 1;",
                    "new_string": "int value = 2;",
                }
            ),
        )


@pytest.mark.asyncio
async def test_observation_limit_warns_after_stage_budget(tmp_path) -> None:
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
    list_tool = next(
        tool
        for tool in bundle.tools_by_profile["finish_skeleton"]
        if tool.name == "list_files"
    )

    soft_limit = bundle.runtime.profiles["finish_skeleton"].max_consecutive_observations

    for _ in range(soft_limit):
        result = await list_tool.on_invoke_tool(
            RunContextWrapper(context=None),
            json.dumps({"path": ".", "limit": 20}),
        )
        assert isinstance(result, str)

    warned = await list_tool.on_invoke_tool(
        RunContextWrapper(context=None),
        json.dumps({"path": ".", "limit": 20}),
    )
    assert "consecutive observation calls" in warned
    assert "Write tools available: edit_file, apply_patch." in warned
    assert "make one write inside the active implementation scope" in warned
    assert "compile" not in warned
    assert "run" not in warned


@pytest.mark.asyncio
async def test_compile_fix_observation_warning_only_lists_true_write_tools(tmp_path) -> None:
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

    bundle.runtime.activate("compile_fix", 0, "compile_fix")
    list_tool = next(
        tool
        for tool in bundle.tools_by_profile["compile_fix"]
        if tool.name == "list_files"
    )

    soft_limit = bundle.runtime.profiles["compile_fix"].max_consecutive_observations

    for _ in range(soft_limit):
        result = await list_tool.on_invoke_tool(
            RunContextWrapper(context=None),
            json.dumps({"path": ".", "limit": 20}),
        )
        assert isinstance(result, str)

    warned = await list_tool.on_invoke_tool(
        RunContextWrapper(context=None),
        json.dumps({"path": ".", "limit": 20}),
    )

    assert "Write tools available: edit_file, apply_patch." in warned


@pytest.mark.asyncio
async def test_todo_plan_can_explore_past_soft_limit_without_fatal(tmp_path) -> None:
    (tmp_path / "queries.txt").write_text("Q1\n", encoding="utf-8")

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

    bundle.runtime.activate("todo_plan", 0, "todo_plan")
    list_tool = next(
        tool
        for tool in bundle.tools_by_profile["todo_plan"]
        if tool.name == "list_files"
    )

    soft_limit = bundle.runtime.profiles["todo_plan"].max_consecutive_observations

    for _ in range(soft_limit):
        result = await list_tool.on_invoke_tool(
            RunContextWrapper(context=None),
            json.dumps({"path": ".", "limit": 20}),
        )
        assert "Fatal:" not in result

    warned = await list_tool.on_invoke_tool(
        RunContextWrapper(context=None),
        json.dumps({"path": ".", "limit": 20}),
    )
    assert "Hard limit: none in this stage." in warned

    for _ in range(4):
        result = await list_tool.on_invoke_tool(
            RunContextWrapper(context=None),
            json.dumps({"path": ".", "limit": 20}),
        )
        assert "Fatal:" not in result


@pytest.mark.asyncio
async def test_todo_sync_unknown_stage_tool_becomes_recoverable_error(tmp_path) -> None:
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

    (tmp_path / "TODO.md").write_text("- [ ] Build engine\n", encoding="utf-8")
    bundle.runtime.activate("todo_sync", 0, "todo_sync")
    list_tool = next(tool for tool in bundle.all_tools if tool.name == "list_files")

    result = await list_tool.on_invoke_tool(
        RunContextWrapper(context=None),
        json.dumps({"path": ".", "limit": 20}),
    )

    assert "[ERROR:TOOL_NOT_ALLOWED]" in result
    assert "profile todo_sync" in result
    return None


def test_copy_template_to_includes_reference_api_headers(tmp_path) -> None:
    copy_template_to(tmp_path, "tpch")

    for filename in ("loader_api.hpp", "builder_api.hpp", "query_api.hpp"):
        assert (tmp_path / filename).exists()


def test_copy_template_to_excludes_runtime_harness_files(tmp_path) -> None:
    copy_template_to(tmp_path, "tpch")

    for filename in ("db.cpp", "loader_api.cpp", "builder_api.cpp", "query_api.cpp"):
        assert (tmp_path / filename).exists() is False


def test_gen_tpch_args_str_handles_quoted_key_value_literals() -> None:
    def _placeholders(*, query_name: str) -> dict[str, object]:
        if query_name == "Q1":
            return {"REGION": "MIDDLE EAST"}
        return {}

    args_str, _ = gen_tpch_args_str(["Q1"], gen_placeholders_fn=_placeholders)

    assert "if (pos < text.size() && text[pos] == '\"')" in args_str
    assert "const char ch = text[pos++];" in args_str
    assert 'args.REGION = require_arg(kv, "REGION", "Q1");' in args_str


# ---------------------------------------------------------------------------
# Stage Runtime Policy Tests (W3)
# ---------------------------------------------------------------------------


def test_stage_policy_has_fixed_fields_for_high_risk_stages() -> None:
    from tpch_monetdb.runtime_stage_policy import STAGE_RUNTIME_POLICIES

    for name in (
        "todo_plan",
        "compile_fix",
        "correctness_primary_query",
        "correctness_single_query",
        "all_queries_correctness",
    ):
        policy = STAGE_RUNTIME_POLICIES[name]
        assert policy.base_turns > 0
        assert policy.extra_turns > 0
        assert policy.max_extensions >= 0
        assert policy.proactive_compact_on_warning is True
        assert policy.block_on_context_saturation is True


def test_budget_tracker_extends_on_progress() -> None:
    from tpch_monetdb.runtime_stage_policy import StageBudgetTracker
    from tpch_monetdb.tools.stage_tool_policy import StageRunSummary

    tracker = StageBudgetTracker()
    assert tracker.compute_effective_max_turns("todo_plan") == 120

    summary1 = StageRunSummary(
        profile_name="todo_plan",
        prompt_index=0,
        prompt_descriptor="todo_plan",
        final_output="done",
        tool_counts={"write_file": 1},
        written_files=("TODO.md",),
        last_compile_summary=None,
        last_run_summary=None,
        todo_before=None,
        todo_after=None,
    )
    tracker.record_stage_result("todo_plan", summary1)
    assert tracker.compute_effective_max_turns("todo_plan") == 180

    summary2 = StageRunSummary(
        profile_name="todo_plan",
        prompt_index=1,
        prompt_descriptor="todo_plan",
        final_output="done",
        tool_counts={"write_file": 1},
        written_files=("TODO.md",),
        last_compile_summary=None,
        last_run_summary=None,
        todo_before=None,
        todo_after=None,
    )
    tracker.record_stage_result("todo_plan", summary2)
    assert tracker.compute_effective_max_turns("todo_plan") == 240

    # Max extensions = 2, so third progress should not extend
    summary3 = StageRunSummary(
        profile_name="todo_plan",
        prompt_index=2,
        prompt_descriptor="todo_plan",
        final_output="done",
        tool_counts={"write_file": 1},
        written_files=("TODO.md",),
        last_compile_summary=None,
        last_run_summary=None,
        todo_before=None,
        todo_after=None,
    )
    tracker.record_stage_result("todo_plan", summary3)
    assert tracker.compute_effective_max_turns("todo_plan") == 240


def test_budget_tracker_does_not_extend_without_progress() -> None:
    from tpch_monetdb.runtime_stage_policy import StageBudgetTracker
    from tpch_monetdb.tools.stage_tool_policy import StageRunSummary

    tracker = StageBudgetTracker()
    summary = StageRunSummary(
        profile_name="todo_plan",
        prompt_index=0,
        prompt_descriptor="todo_plan",
        final_output="done",
        tool_counts={},
        written_files=(),
        last_compile_summary=None,
        last_run_summary=None,
        todo_before=None,
        todo_after=None,
    )
    tracker.record_stage_result("todo_plan", summary)
    assert tracker.compute_effective_max_turns("todo_plan") == 120


def test_budget_tracker_detects_compile_summary_change() -> None:
    from tpch_monetdb.runtime_stage_policy import StageBudgetTracker
    from tpch_monetdb.tools.stage_tool_policy import StageRunSummary

    tracker = StageBudgetTracker()
    summary1 = StageRunSummary(
        profile_name="compile_fix",
        prompt_index=0,
        prompt_descriptor="compile_fix",
        final_output="error",
        tool_counts={},
        written_files=(),
        last_compile_summary="error: missing ;",
        last_run_summary=None,
        todo_before=None,
        todo_after=None,
    )
    tracker.record_stage_result("compile_fix", summary1)
    assert tracker.compute_effective_max_turns("compile_fix") == 512

    summary2 = StageRunSummary(
        profile_name="compile_fix",
        prompt_index=1,
        prompt_descriptor="compile_fix",
        final_output="ok",
        tool_counts={},
        written_files=(),
        last_compile_summary="compile ok",
        last_run_summary=None,
        todo_before=None,
        todo_after=None,
    )
    tracker.record_stage_result("compile_fix", summary2)
    assert tracker.compute_effective_max_turns("compile_fix") == 640


def test_budget_tracker_extends_correctness_q1_via_prefix_policy() -> None:
    from tpch_monetdb.runtime_stage_policy import StageBudgetTracker
    from tpch_monetdb.tools.stage_tool_policy import StageRunSummary

    tracker = StageBudgetTracker()
    assert tracker.compute_effective_max_turns("correctness_q1") == 256

    summary = StageRunSummary(
        profile_name="correctness_primary_query",
        prompt_index=0,
        prompt_descriptor="correctness_q1",
        final_output="done",
        tool_counts={"write_file": 1},
        written_files=("loader_impl.cpp",),
        last_compile_summary=None,
        last_run_summary=None,
        todo_before=None,
        todo_after=None,
    )
    tracker.record_stage_result("correctness_q1", summary)
    assert tracker.compute_effective_max_turns("correctness_q1") == 320


def test_budget_tracker_extends_correctness_query_n_via_prefix_policy() -> None:
    from tpch_monetdb.runtime_stage_policy import StageBudgetTracker
    from tpch_monetdb.tools.stage_tool_policy import StageRunSummary

    tracker = StageBudgetTracker()
    assert tracker.compute_effective_max_turns("correctness_query_3") == 320

    summary = StageRunSummary(
        profile_name="correctness_single_query",
        prompt_index=0,
        prompt_descriptor="correctness_query_3",
        final_output="done",
        tool_counts={"write_file": 1},
        written_files=("loader_impl.cpp",),
        last_compile_summary=None,
        last_run_summary=None,
        todo_before=None,
        todo_after=None,
    )
    tracker.record_stage_result("correctness_query_3", summary)
    assert tracker.compute_effective_max_turns("correctness_query_3") == 384


def test_postcondition_diagnostic_includes_stage_context() -> None:
    from tpch_monetdb.conversations.scripted_conversation import (
        PromptStep,
        ScriptedConversation,
    )
    from tpch_monetdb.tools.stage_tool_policy import StageRunSummary

    conv = ScriptedConversation.__new__(ScriptedConversation)
    step = PromptStep(
        text="prompt",
        descriptor="compile_fix",
        required_nonempty_files=["loader_impl.cpp"],
    )
    summary = StageRunSummary(
        profile_name="compile_fix",
        prompt_index=0,
        prompt_descriptor="compile_fix",
        final_output="done",
        tool_counts={},
        written_files=("builder_impl.cpp",),
        last_compile_summary="compile ok",
        last_run_summary=None,
        todo_before=None,
        todo_after=None,
    )
    diag = conv._format_postcondition_diagnostic(
        step=step,
        idx=0,
        failed_postcondition="required file loader_impl.cpp does not exist",
        summary=summary,
    )
    assert "[ERROR:STAGE_POSTCONDITION_FAILED]" in diag
    assert "compile_fix" in diag
    assert "Recent writes: builder_impl.cpp" in diag
    assert "Last compile: compile ok" in diag


def test_postcondition_diagnostic_truncates_large_summaries() -> None:
    from tpch_monetdb.conversations.scripted_conversation import (
        PromptStep,
        ScriptedConversation,
    )
    from tpch_monetdb.tools.stage_tool_policy import StageRunSummary

    conv = ScriptedConversation.__new__(ScriptedConversation)
    step = PromptStep(text="prompt", descriptor="test")
    summary = StageRunSummary(
        profile_name="test",
        prompt_index=0,
        prompt_descriptor="test",
        final_output="done",
        tool_counts={},
        written_files=(),
        last_compile_summary="x" * 500,
        last_run_summary=None,
        todo_before=None,
        todo_after=None,
    )
    diag = conv._format_postcondition_diagnostic(
        step=step,
        idx=0,
        failed_postcondition="write_required",
        summary=summary,
    )
    assert "...[TRUNCATED]" in diag
    assert len(diag) <= 1300


def test_postcondition_diagnostic_avoids_full_file_dumps() -> None:
    from tpch_monetdb.conversations.scripted_conversation import (
        PromptStep,
        ScriptedConversation,
    )
    from tpch_monetdb.tools.stage_tool_policy import StageRunSummary

    conv = ScriptedConversation.__new__(ScriptedConversation)
    step = PromptStep(text="prompt", descriptor="test")
    summary = StageRunSummary(
        profile_name="test",
        prompt_index=0,
        prompt_descriptor="test",
        final_output="done",
        tool_counts={},
        written_files=(),
        last_compile_summary=None,
        last_run_summary=None,
        todo_before=None,
        todo_after=None,
    )
    diag = conv._format_postcondition_diagnostic(
        step=step,
        idx=0,
        failed_postcondition="missing file",
        summary=summary,
    )
    assert "[ERROR:STAGE_POSTCONDITION_FAILED]" in diag
    assert "Recent writes:" not in diag


def test_storage_plan_contract_advisory_builds_repair_prompt(tmp_path: Path) -> None:
    from tpch_monetdb.conversations.scripted_conversation import (
        PromptStep,
        ScriptedConversation,
    )
    from tpch_monetdb.tools.stage_tool_policy import StageRunSummary

    (tmp_path / "storage_plan.txt").write_text("layout\n", encoding="utf-8")
    (tmp_path / "workload_objective.json").write_text(
        '{"query_ids":["1","6","10"]}\n',
        encoding="utf-8",
    )
    (tmp_path / "storage_plan_contract.json").write_text(
        '{"version":2,"selected_base_candidate_id":"layout_a"}\n',
        encoding="utf-8",
    )

    conv = ScriptedConversation.__new__(ScriptedConversation)
    conv.workspace_root = tmp_path
    step = PromptStep(
        text="storage prompt",
        descriptor="storage_plan",
        required_nonempty_files=(
            "storage_plan.txt",
            "storage_plan_contract.json",
        ),
        advisory_postconditions=("storage_plan_contract_complete",),
    )
    summary = StageRunSummary(
        profile_name="storage_plan",
        prompt_index=0,
        prompt_descriptor="storage_plan",
        final_output="done",
        tool_counts={},
        written_files=("storage_plan.txt", "storage_plan_contract.json"),
        last_compile_summary=None,
        last_run_summary=None,
        todo_before=None,
        todo_after=None,
    )

    recoverable = conv._get_recoverable_postcondition(
        step=step,
        idx=0,
        callback_result=summary,
    )

    assert recoverable is not None
    assert recoverable.descriptor_suffix == "storage_plan_contract_remediation"
    assert "STORAGE_PLAN_CONTRACT_COMMITTED_LAYOUT_FIELDS_MISSING" in (
        recoverable.failed_postcondition
    )
    assert "Do not start base implementation work" in recoverable.remediation_prompt
    assert conv._is_soft_advisory_failure(recoverable.failed_postcondition) is True


def test_storage_plan_contract_advisory_does_not_make_validation_fatal(
    tmp_path: Path,
) -> None:
    from tpch_monetdb.conversations.scripted_conversation import (
        PromptStep,
        ScriptedConversation,
    )
    from tpch_monetdb.tools.stage_tool_policy import StageRunSummary

    conv = ScriptedConversation.__new__(ScriptedConversation)
    conv.workspace_root = tmp_path
    step = PromptStep(
        text="storage prompt",
        descriptor="storage_plan",
        advisory_postconditions=("storage_plan_contract_complete",),
    )
    summary = StageRunSummary(
        profile_name="storage_plan",
        prompt_index=0,
        prompt_descriptor="storage_plan",
        final_output="done",
        tool_counts={},
        written_files=(),
        last_compile_summary=None,
        last_run_summary=None,
        todo_before=None,
        todo_after=None,
    )

    conv._validate_step_postconditions(
        step=step,
        idx=0,
        callback_result=summary,
    )


def test_get_policy_for_stage_matches_correctness_q1_q2() -> None:
    from tpch_monetdb.runtime_stage_policy import get_policy_for_stage, STAGE_RUNTIME_POLICIES
    p1 = get_policy_for_stage("correctness_q1")
    p2 = get_policy_for_stage("correctness_q2")
    primary = STAGE_RUNTIME_POLICIES["correctness_primary_query"]
    assert p1 is primary
    assert p2 is primary


def test_get_policy_for_stage_matches_correctness_query_n() -> None:
    from tpch_monetdb.runtime_stage_policy import get_policy_for_stage, STAGE_RUNTIME_POLICIES
    for qid in ["3", "4", "9", "15"]:
        p = get_policy_for_stage(f"correctness_query_{qid}")
        assert p is STAGE_RUNTIME_POLICIES["correctness_single_query"], qid


def test_get_policy_for_stage_exact_match_still_wins() -> None:
    from tpch_monetdb.runtime_stage_policy import get_policy_for_stage, STAGE_RUNTIME_POLICIES
    assert get_policy_for_stage("all_queries_correctness") is STAGE_RUNTIME_POLICIES["all_queries_correctness"]
    assert get_policy_for_stage("todo_plan") is STAGE_RUNTIME_POLICIES["todo_plan"]
    assert get_policy_for_stage("unknown_stage") is None


def test_parse_error_envelope_codes_returns_all() -> None:
    from run_outer_loop_tpch_monetdb import _parse_error_envelope_codes
    text = "[ERROR:RUNNER_BROKEN_PIPE] something\n[ERROR:OPTIMIZATION_PRECHECK_FAILED] gate"
    codes = _parse_error_envelope_codes(text)
    assert codes == {"RUNNER_BROKEN_PIPE", "OPTIMIZATION_PRECHECK_FAILED"}


def test_is_optimization_correctness_gate_failure_ignores_prefix_noise() -> None:
    import subprocess
    from run_outer_loop_tpch_monetdb import _is_optimization_correctness_gate_failure
    result = subprocess.CompletedProcess(
        args=[],
        returncode=1,
        stdout="[ERROR:RUNNER_BROKEN_PIPE] runner died\n[ERROR:OPTIMIZATION_PRECHECK_FAILED] gate\n",
        stderr="",
    )
    assert _is_optimization_correctness_gate_failure(result) is True


def test_is_optimization_correctness_gate_failure_false_when_only_other_error() -> None:
    import subprocess
    from run_outer_loop_tpch_monetdb import _is_optimization_correctness_gate_failure
    result = subprocess.CompletedProcess(
        args=[],
        returncode=1,
        stdout="[ERROR:RUNNER_BROKEN_PIPE] runner died\n",
        stderr="",
    )
    assert _is_optimization_correctness_gate_failure(result) is False
