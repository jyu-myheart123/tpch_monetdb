"""Phase10 on-demand query module build graph regression tests.

锁定 Section 3：copy_template_to / make_compiler / stage_tool_policy / apply_patch
都支持按需 query module。
"""

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
from tpch_monetdb.dataset.query_gen_factory import get_placeholders_fn
from tpch_monetdb.tools.stage_tool_policy import (
    COMPANION_QUERY_GLOBS,
    CORE_IMPLEMENTATION_FILES,
    FOUNDATION_CORRECTNESS_EDIT_GLOBS,
    HOST_OWNED_WRITE_GLOBS,
    OPTIMIZATION_INFRA_LAYOUT_EDIT_GLOBS,
    OPTIMIZATION_QUERY_LOCAL_EDIT_GLOBS,
    QUERY_FOCUSED_EDIT_GLOBS,
    QUERY_CREATE_GLOBS,
    QUERY_EDIT_FILES,
    build_tool_profiles,
)
from tpch_monetdb.tools.tpch.utils import (
    _collect_companion_query_sources,
    _discover_companion_query_files,
    copy_template_to,
    make_compiler,
)
from tpch_monetdb.utils.general_utils import gen_tpch_args_str


def test_query_edit_scope_includes_companion_globs() -> None:
    """phase10 查询 edit scope 涵盖 dispatcher 与 companion glob."""
    assert "query_*.cpp" in COMPANION_QUERY_GLOBS
    assert "query_*.hpp" in COMPANION_QUERY_GLOBS
    for dispatcher in CORE_IMPLEMENTATION_FILES:
        assert dispatcher in QUERY_EDIT_FILES
    for glob in COMPANION_QUERY_GLOBS:
        assert glob in QUERY_EDIT_FILES
    return None


def _make_run_tool() -> FunctionTool:
    async def on_invoke(_ctx: RunContextWrapper[object], _args_json: str) -> str:
        return "ok"

    return FunctionTool(
        name="run",
        description="run",
        params_json_schema={"type": "object", "properties": {}},
        on_invoke_tool=on_invoke,
    )


def test_scripted_optimization_profiles_accept_companion_query_files() -> None:
    """实现阶段 edit/create scope 与 query module 口径一致."""
    profiles = build_tool_profiles()
    for profile_name in (
        "finish_skeleton",
        "compile_fix",
        "add_timings",
        "implement_queries",
        "correctness",
        "benchmark",
        "optimization_infra_layout",
    ):
        profile = profiles[profile_name]
        assert "query_*.cpp" in profile.edit_globs, profile_name
        assert "query_*.hpp" in profile.edit_globs, profile_name
    for profile_name in (
        "implement_queries_writeonly",
        "correctness_queries_writeonly",
        "correctness_foundation",
        "optimization_general",
    ):
        profile = profiles[profile_name]
        assert "query_*.cpp" not in profile.edit_globs, profile_name
        assert "query_*.hpp" not in profile.edit_globs, profile_name
    for profile_name in (
        "implement_queries_writeonly",
        "correctness_queries_writeonly",
        "correctness_foundation",
        "optimization_general",
    ):
        profile = profiles[profile_name]
        assert "query_q*.cpp" in profile.edit_globs, profile_name
        assert "query_q*.hpp" in profile.edit_globs, profile_name
        assert "query_family_*.cpp" in profile.edit_globs, profile_name
        assert "query_family_*.hpp" in profile.edit_globs, profile_name
        assert "query_shared_*.cpp" in profile.edit_globs, profile_name
        assert "query_shared_*.hpp" in profile.edit_globs, profile_name
    implement_write_only = profiles["implement_queries_writeonly"]
    assert "write_file" in implement_write_only.tool_names
    assert implement_write_only.write_globs == QUERY_FOCUSED_EDIT_GLOBS
    assert implement_write_only.allow_write_create is True
    assert implement_write_only.allow_write_overwrite is True
    for profile_name in (
        "finish_skeleton",
        "compile_fix",
        "implement_queries_writeonly",
        "correctness_queries_writeonly",
        "correctness_foundation",
        "correctness",
        "benchmark",
        "optimize_build",
        "optimization_general",
    ):
        profile = profiles[profile_name]
        assert profile.create_globs == QUERY_CREATE_GLOBS, profile_name
    for profile_name in ("todo_plan", "storage_plan", "add_timings", "implement_queries"):
        profile = profiles[profile_name]
        assert profile.create_globs == (), profile_name
    assert "apply_patch" in profiles["optimization_general"].tool_names
    assert profiles["optimization_general"].edit_globs == OPTIMIZATION_QUERY_LOCAL_EDIT_GLOBS
    assert profiles["optimization_infra_layout"].edit_globs == OPTIMIZATION_INFRA_LAYOUT_EDIT_GLOBS
    return None


def test_per_query_profiles_exclude_builder_loader_edit_scope() -> None:
    """per-query implement/correctness 仅允许 query 家族文件编辑。"""
    profiles = build_tool_profiles()
    for profile_name in (
        "implement_queries_writeonly",
        "correctness_queries_writeonly",
        "optimization_general",
    ):
        profile = profiles[profile_name]
        assert profile.edit_globs == QUERY_FOCUSED_EDIT_GLOBS
        assert "query_impl.hpp" not in profile.edit_globs
        assert "query_impl.cpp" not in profile.edit_globs
        assert "builder_impl.hpp" not in profile.edit_globs
        assert "builder_impl.cpp" not in profile.edit_globs
        assert "loader_impl.hpp" not in profile.edit_globs
        assert "loader_impl.cpp" not in profile.edit_globs
        assert "query_q*.cpp" in profile.edit_globs
        assert "query_q*.hpp" in profile.edit_globs
        assert "query_family_*.cpp" in profile.edit_globs
        assert "query_family_*.hpp" in profile.edit_globs
        assert "query_shared_*.cpp" in profile.edit_globs
        assert "query_shared_*.hpp" in profile.edit_globs
        assert "query_api.hpp" not in profile.edit_globs
    return None


def test_correctness_foundation_allows_first_gate_core_dataflow_scope() -> None:
    """Q1 correctness may fix loader/builder/query_impl because it is the first dataflow gate."""
    profiles = build_tool_profiles()
    profile = profiles["correctness_foundation"]
    assert profile.edit_globs == FOUNDATION_CORRECTNESS_EDIT_GLOBS
    assert profile.write_globs == FOUNDATION_CORRECTNESS_EDIT_GLOBS
    for core_file in CORE_IMPLEMENTATION_FILES:
        assert core_file in profile.edit_globs
        assert core_file in profile.write_globs
    for query_glob in QUERY_FOCUSED_EDIT_GLOBS:
        assert query_glob in profile.edit_globs
        assert query_glob in profile.write_globs
    assert "query_*.cpp" not in profile.edit_globs
    assert "query_*.hpp" not in profile.edit_globs
    assert profile.create_globs == QUERY_CREATE_GLOBS
    assert profile.allow_write_overwrite is True
    assert profile.allow_write_create is False
    return None


def test_host_owned_write_globs_cover_sealed_and_generated_artifacts() -> None:
    assert "workload_objective.json" in HOST_OWNED_WRITE_GLOBS
    assert "data_law_contract.json" in HOST_OWNED_WRITE_GLOBS
    assert "implementation_manifest.json" in HOST_OWNED_WRITE_GLOBS
    assert "host_sealed_manifest.json" in HOST_OWNED_WRITE_GLOBS
    assert "generated/query_q*.cpp" in HOST_OWNED_WRITE_GLOBS
    assert "generated/query_family_*.hpp" in HOST_OWNED_WRITE_GLOBS
    assert "build/generated/query_registry_generated.cpp" in HOST_OWNED_WRITE_GLOBS
    return None


def test_legacy_general_profile_still_rejects_host_owned_writes() -> None:
    profile = build_tool_profiles()["legacy_general"]
    assert profile.allows_write("query_q1.cpp") is True
    assert profile.allows_write("workload_objective.json") is False
    assert profile.allows_write("host_sealed_manifest.json") is False
    assert profile.allows_write("generated/query_q9.cpp") is False
    assert profile.allows_write("build/generated/query_registry_generated.cpp") is False
    return None


def test_apply_patch_wrapper_description_mentions_stage_dependent_creation(
    tmp_path: Path,
) -> None:
    """暴露给模型的 apply_patch 文案应包含 stage-dependent create_file 能力."""
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

    apply_patch_tool = next(tool for tool in bundle.all_tools if tool.name == "apply_patch")

    assert "update_file only" not in apply_patch_tool.description
    assert "create a focused query module" in apply_patch_tool.description
    assert "current stage" in apply_patch_tool.description
    return None


def test_optimize_build_profile_allows_loader_and_query_edit_scope() -> None:
    """optimize_build edit scope 与 prompt/rule 一致，允许最小必要触碰 loader/query."""
    profiles = build_tool_profiles()
    build = profiles["optimize_build"]
    assert "builder_impl.cpp" in build.edit_globs
    assert "builder_impl.hpp" in build.edit_globs
    assert "loader_impl.cpp" in build.edit_globs
    assert "loader_impl.hpp" in build.edit_globs
    assert "query_*.cpp" in build.edit_globs
    assert "query_*.hpp" in build.edit_globs
    assert build.create_globs == QUERY_CREATE_GLOBS
    return None


def test_copy_template_to_only_copies_core_skeleton_files(tmp_path: Path) -> None:
    """骨架模板只复制核心实现文件，不复制 speculative query family 空壳."""
    copy_template_to(tmp_path, "tpch")

    for filename in CORE_IMPLEMENTATION_FILES:
        assert (tmp_path / filename).is_file(), filename
    for filename in (
        "query_lastpoint.cpp",
        "query_lastpoint.hpp",
        "query_double_groupby.cpp",
        "query_double_groupby.hpp",
    ):
        assert (tmp_path / filename).exists() is False, filename
    return None


def test_discover_companion_query_files_excludes_dispatcher(tmp_path: Path) -> None:
    """模板扫描器跳过 dispatcher 入口并按文件名排序返回 companion."""
    (tmp_path / "query_impl.cpp").write_text("")
    (tmp_path / "query_impl.hpp").write_text("")
    (tmp_path / "query_q2.cpp").write_text("")
    (tmp_path / "query_q2.hpp").write_text("")
    (tmp_path / "query_shared_groupby.cpp").write_text("")

    found = _discover_companion_query_files(tmp_path)
    assert "query_impl.cpp" not in found
    assert "query_impl.hpp" not in found
    assert set(found) == {
        "query_q2.cpp",
        "query_q2.hpp",
        "query_shared_groupby.cpp",
    }
    return None


def test_collect_companion_query_sources_used_by_compiler(tmp_path: Path) -> None:
    """make_compiler 收集 workspace 中的 companion .cpp 并排除 dispatcher."""
    (tmp_path / "query_impl.cpp").write_text("")
    (tmp_path / "query_q2.cpp").write_text("")
    (tmp_path / "query_shared_groupby.cpp").write_text("")
    (tmp_path / "query_q2.hpp").write_text("")  # 只有 .cpp 入编译

    companions = _collect_companion_query_sources(tmp_path)
    assert "query_impl.cpp" not in companions
    assert set(companions) == {"query_q2.cpp", "query_shared_groupby.cpp"}
    return None


def test_query_q_module_compiles_without_odr_conflict(tmp_path: Path) -> None:
    """创建 query_q1 模块后应可与 dispatcher seed 一起编译，不发生重复定义."""
    copy_template_to(tmp_path, "tpch")
    args_str, _ = gen_tpch_args_str(
        ["Q1"],
        gen_placeholders_fn=get_placeholders_fn("tpch"),
    )
    (tmp_path / "args_parser.hpp").write_text(args_str, encoding="utf-8")
    (tmp_path / "query_q1.hpp").write_text(
        "\n".join(
            [
                "#pragma once",
                '#include "builder_impl.hpp"',
                '#include "args_parser.hpp"',
                "void execute_q1(Engine& engine, const Q1Args& args);",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / "query_q1.cpp").write_text(
        "\n".join(
            [
                '#include "query_q1.hpp"',
                "void execute_q1(Engine& engine, const Q1Args& args) {",
                "    (void)engine; (void)args;",
                "}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    compiler = make_compiler(tmp_path)
    result = compiler.build()

    assert result is None
    return None


def test_tpch_seed_q1_to_q22_modules_compile_with_generated_registry(
    tmp_path: Path,
) -> None:
    """TPC-H template copy should include seed query modules that compile."""
    copy_template_to(tmp_path, "tpch")
    args_str, _ = gen_tpch_args_str(
        [
            "Q1",
            "Q2",
            "Q3",
            "Q4",
            "Q5",
            "Q6",
            "Q7",
            "Q8",
            "Q9",
            "Q10",
            "Q11",
            "Q12",
            "Q13",
            "Q14",
            "Q15",
            "Q16",
            "Q17",
            "Q18",
            "Q19",
            "Q20",
            "Q21",
            "Q22",
        ],
        gen_placeholders_fn=get_placeholders_fn("tpch"),
    )
    (tmp_path / "args_parser.hpp").write_text(args_str, encoding="utf-8")

    assert (tmp_path / "query_q1.cpp").exists()
    assert (tmp_path / "query_q2.cpp").exists()
    assert (tmp_path / "query_q3.cpp").exists()
    assert (tmp_path / "query_q4.cpp").exists()
    assert (tmp_path / "query_q5.cpp").exists()
    assert (tmp_path / "query_q6.cpp").exists()
    assert (tmp_path / "query_q7.cpp").exists()
    assert (tmp_path / "query_q8.cpp").exists()
    assert (tmp_path / "query_q9.cpp").exists()
    assert (tmp_path / "query_q10.cpp").exists()
    assert (tmp_path / "query_q11.cpp").exists()
    assert (tmp_path / "query_q12.cpp").exists()
    assert (tmp_path / "query_q13.cpp").exists()
    assert (tmp_path / "query_q14.cpp").exists()
    assert (tmp_path / "query_q15.cpp").exists()
    assert (tmp_path / "query_q16.cpp").exists()
    assert (tmp_path / "query_q17.cpp").exists()
    assert (tmp_path / "query_q18.cpp").exists()
    assert (tmp_path / "query_q19.cpp").exists()
    assert (tmp_path / "query_q20.cpp").exists()
    assert (tmp_path / "query_q21.cpp").exists()
    assert (tmp_path / "query_q22.cpp").exists()
    assert (tmp_path / "query_shared_tpch.hpp").exists()

    compiler = make_compiler(
        tmp_path,
        validate_requested_query_modules=True,
        required_query_ids=[
            "Q1",
            "Q2",
            "Q3",
            "Q4",
            "Q5",
            "Q6",
            "Q7",
            "Q8",
            "Q9",
            "Q10",
            "Q11",
            "Q12",
            "Q13",
            "Q14",
            "Q15",
            "Q16",
            "Q17",
            "Q18",
            "Q19",
            "Q20",
            "Q21",
            "Q22",
        ],
    )
    result = compiler.build()

    assert result is None
    return None


def test_query_registry_fails_fast_when_requested_entrypoint_missing(tmp_path: Path) -> None:
    copy_template_to(tmp_path, "tpch")
    (tmp_path / "queries.txt").write_text("Query 1:\nQuery 2:\n", encoding="utf-8")
    for suffix in ("hpp", "cpp"):
        query_file = tmp_path / f"query_q2.{suffix}"
        if query_file.exists():
            query_file.unlink()
    (tmp_path / "query_q1.hpp").write_text(
        "void execute_q1(Engine& engine, const Q1Args& args);\n",
        encoding="utf-8",
    )
    (tmp_path / "query_q1.cpp").write_text(
        "void execute_q1(Engine& engine, const Q1Args& args) { (void)engine; (void)args; }\n",
        encoding="utf-8",
    )

    with pytest.raises(
        RuntimeError,
        match="Generated query registry missing entrypoints for requested queries: Q2",
    ):
        make_compiler(tmp_path, validate_requested_query_modules=True)
    assert (tmp_path / "build" / "generated" / "query_registry_generated.cpp").exists() is False
    return None


def test_plain_make_compiler_allows_foundation_before_all_query_modules(
    tmp_path: Path,
) -> None:
    """foundation compile 不应因后续 family entrypoint 尚未生成而提前失败。"""
    copy_template_to(tmp_path, "tpch")
    args_str, _ = gen_tpch_args_str(
        ["Q1", "Q2"],
        gen_placeholders_fn=get_placeholders_fn("tpch"),
    )
    (tmp_path / "queries.txt").write_text("Query 1:\nQuery 2:\n", encoding="utf-8")
    (tmp_path / "args_parser.hpp").write_text(args_str, encoding="utf-8")
    (tmp_path / "query_q1.hpp").write_text(
        "void execute_q1(Engine& engine, const Q1Args& args);\n",
        encoding="utf-8",
    )
    (tmp_path / "query_q1.cpp").write_text(
        "void execute_q1(Engine& engine, const Q1Args& args) { (void)engine; (void)args; }\n",
        encoding="utf-8",
    )

    compiler = make_compiler(tmp_path)

    assert compiler is not None
    assert (tmp_path / "build" / "generated" / "query_registry_generated.cpp").exists()
    return None


def test_query_shared_module_must_not_define_dispatch_query(tmp_path: Path) -> None:
    copy_template_to(tmp_path, "tpch")
    (tmp_path / "query_shared_impl.cpp").write_text(
        "\n".join(
            [
                '#include "query_impl.hpp"',
                "void dispatch_query(Engine& engine, const QueryRequest& request) {",
                "    (void)engine; (void)request;",
                "}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="must not define dispatcher symbols"):
        make_compiler(tmp_path)
    return None


def test_query_shared_module_must_not_define_dispatch_helpers_or_entrypoints(tmp_path: Path) -> None:
    copy_template_to(tmp_path, "tpch")
    (tmp_path / "query_shared_impl.cpp").write_text(
        "\n".join(
            [
                '#include "query_impl.hpp"',
                '#include "args_parser.hpp"',
                "void dispatch_unimplemented_query(const QueryRequest& request) {",
                "    (void)request;",
                "}",
                "void execute_q1(Engine& engine, const Q1Args& args) {",
                "    (void)engine; (void)args;",
                "}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="must not define dispatcher symbols"):
        make_compiler(tmp_path)
    return None


@pytest.mark.asyncio
async def test_apply_patch_create_file_allowed_for_query_modules(tmp_path: Path) -> None:
    """实现阶段允许在受控 query namespace 里 create_file."""
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
                "type": "create_file",
                "path": "query_q2.cpp",
                "diff": "+int q2() { return 2; }\n",
            }
        ),
    )

    summary = bundle.runtime.finish_stage(result)

    assert result == "Created query_q2.cpp"
    assert (tmp_path / "query_q2.cpp").read_text(encoding="utf-8") == "int q2() { return 2; }"
    assert summary.written_files == ("query_q2.cpp",)
    return None


@pytest.mark.asyncio
async def test_apply_patch_create_file_rejects_non_query_namespace(tmp_path: Path) -> None:
    """实现阶段 create_file 仍受 create_globs 约束."""
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
                "type": "create_file",
                "path": "notes.txt",
                "diff": "+hello\n",
            }
        ),
    )

    assert "[ERROR:CREATE_DENIED]" in result
    assert "Allowed next actions: query_q*.cpp, query_q*.hpp, query_family_*.cpp, query_family_*.hpp, query_shared_*.cpp, query_shared_*.hpp" in result
    assert (tmp_path / "notes.txt").exists() is False
    return None


@pytest.mark.asyncio
async def test_write_file_rejects_host_owned_artifacts_even_in_legacy_profile(tmp_path: Path) -> None:
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
    bundle.runtime.activate("legacy_general", 0, "legacy_general")
    write_tool = next(
        tool
        for tool in bundle.tools_by_profile["legacy_general"]
        if tool.name == "write_file"
    )

    result = await write_tool.on_invoke_tool(
        RunContextWrapper(context=None),
        json.dumps(
            {
                "file_path": "workload_objective.json",
                "content": "{}\n",
            }
        ),
    )

    assert "[ERROR:WRITE_DENIED]" in result
    assert (tmp_path / "workload_objective.json").exists() is False
    return None


@pytest.mark.asyncio
async def test_apply_patch_create_file_denied_when_stage_has_no_create_scope(tmp_path: Path) -> None:
    """有 apply_patch 但无 create_globs 的阶段仍拒绝 create_file."""
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
    bundle.runtime.activate("add_timings", 0, "add_timings")
    patch_tool = next(
        tool
        for tool in bundle.tools_by_profile["add_timings"]
        if tool.name == "apply_patch"
    )

    result = await patch_tool.on_invoke_tool(
        RunContextWrapper(context=None),
        json.dumps(
            {
                "type": "create_file",
                "path": "query_q3.cpp",
                "diff": "+int q3() { return 3; }\n",
            }
        ),
    )

    assert "[ERROR:PATCH_OP_DENIED]" in result
    assert "Allowed next actions: update_file" in result
    assert (tmp_path / "query_q3.cpp").exists() is False
    return None
