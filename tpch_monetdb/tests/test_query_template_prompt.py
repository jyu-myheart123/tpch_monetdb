"""Phase10 on-demand query module / prompt cutover regression tests.

锁定 Section 4：运行时骨架模板精简、dispatcher 不再带 speculative companion
include、scripted prompt 指导模型按需创建 focused query 文件。
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TEMPLATES_DIR = ROOT / "misc" / "tpch" / "templates"

CORE_SEED_FILES = (
    "loader_impl.hpp",
    "loader_impl.cpp",
    "builder_impl.hpp",
    "builder_impl.cpp",
    "query_impl.hpp",
    "query_impl.cpp",
)

REMOVED_FAMILY_TEMPLATES = (
    "query_lastpoint",
    "query_single_groupby",
    "query_cpu_max_all",
    "query_double_groupby",
    "query_high_cpu",
    "query_high_cpu_all",
    "query_groupby_orderby_limit",
)

QUERY_TEMPLATE_FORBIDDEN_TOKENS = (
    ".csv_output",
    "should_materialize_query_output",
    "format_tpch_",
    "append_tpch_",
    "engine.lineitems",
    "engine.orders",
    "engine.parts",
    "engine.suppliers",
    "engine.customers",
    "engine.nations",
    "engine.regions",
    "engine.partsupps",
    "std::map",
    "std::unordered",
    "for (",
)


def test_seed_templates_only_keep_core_implementation_files() -> None:
    """phase10 骨架模板只保留核心实现文件，去掉 speculative family 空壳."""
    for filename in CORE_SEED_FILES:
        assert (TEMPLATES_DIR / filename).is_file(), filename
    for family in REMOVED_FAMILY_TEMPLATES:
        assert not (TEMPLATES_DIR / f"{family}.cpp").exists()
        assert not (TEMPLATES_DIR / f"{family}.hpp").exists()
    return None


def test_query_templates_do_not_embed_concrete_query_algorithms() -> None:
    """query_q*.cpp 模板只保留入口骨架，不预置 TPC-H 查询算法."""
    for query_id in range(1, 23):
        source_path = TEMPLATES_DIR / f"query_q{query_id}.cpp"
        text = source_path.read_text()
        assert f"void execute_q{query_id}(Engine&, const Q{query_id}Args&)" in text
        assert f'raise_missing_template_query_body("Q{query_id}")' in text
        for token in QUERY_TEMPLATE_FORBIDDEN_TOKENS:
            assert token not in text, f"{source_path.name} contains {token}"
    return None


def test_dispatcher_query_impl_drops_example_marker_and_speculative_companion_includes() -> None:
    """query_impl.cpp 不再包含 example marker，也不静态 include 占位 companion header."""
    text = (TEMPLATES_DIR / "query_impl.cpp").read_text()
    assert "<<example parser call code>>" not in text
    for family in REMOVED_FAMILY_TEMPLATES:
        assert f'#include "{family}.hpp"' not in text, family
    return None


def test_scripted_base_prompt_references_on_demand_query_modules() -> None:
    """run_gen_base_impl_tpch_monetdb 的 base prompt 引导模型按需创建 focused query 模块."""
    text = (
        ROOT / "conversations" / "prompts" / "scripted" / "base_impl"
        / "todo_plan_stage.txt"
    ).read_text()
    assert "`query_q*.cpp`" in text
    assert "`query_shared_*.cpp`" in text
    assert "`query_q2.cpp`" in text
    assert "`query_q9.cpp`" in text
    assert "creating a new `query_q*.cpp` / `query_q*.hpp`" in text
    return None


def test_storage_plan_prompt_is_loaded_from_asset_instead_of_large_inline_prose() -> None:
    text = (ROOT / "run_gen_storage_plan_tpch_monetdb.py").read_text()
    assert 'render_scripted_prompt_asset(' in text
    assert (
        "Your task is to analyze the TPC-H MonetDB workload and produce a creative in-memory storage-layout summary."
        not in text
    )
    return None


def test_base_impl_stage_assets_are_loaded_from_prompt_files() -> None:
    text = (ROOT / "run_gen_base_impl_tpch_monetdb.py").read_text()
    assert 'render_scripted_prompt_asset(' in text
    assert "Use the host-owned sealed manifest to decide the active query unit for query" not in text
    return None


def test_todo_plan_prompt_reads_design_evidence_and_adds_fidelity_sections() -> None:
    """todo_plan 应先读 design_evidence，并显式加入 fidelity TODO 章节。"""
    text = (
        ROOT / "conversations" / "prompts" / "scripted" / "base_impl"
        / "todo_plan_stage.txt"
    ).read_text()
    assert (
        "Read `workload_objective.json`, `data_law_contract.json`, "
        "`storage_plan_contract.json`, `storage_plan.txt`, and "
        "`design_evidence.md` before writing the plan."
        in text
    )
    assert "- `## Loader Fidelity`" in text
    assert "- `## Output Fidelity`" in text
    assert "- `## Join and Ordering Fidelity`" in text
    assert "- `## Per-Query Access Paths`" in text
    assert "- `## Vectorization-Ready Layout Tasks`" in text
    assert "- `## Base Performance Observations`" in text
    assert "requires_vectorization=true" in text
    assert "Do not create any collapsed summary checkbox" in text
    assert "base performance observation evidence" in text
    return None


def test_todo_sync_prompts_preserve_query_specific_todo_shape() -> None:
    """TODO sync prompts must not collapse detailed query tasks into summaries."""
    quick_text = (
        ROOT / "conversations" / "prompts" / "scripted" / "base_impl"
        / "todo_sync_quick.txt"
    ).read_text()
    full_text = (
        ROOT / "conversations" / "prompts" / "scripted" / "base_impl"
        / "todo_sync.txt"
    ).read_text()
    for text in (quick_text, full_text):
        lower_text = text.lower()
        assert "do not delete" in lower_text
        assert "do not merge query-specific" in lower_text
        assert "Q1-Q22 all implemented" in text
        assert "speedup below 1.0x" in text
    return None


def test_todo_plan_and_add_timings_prompts_require_query_id_timing_labels() -> None:
    todo_text = (
        ROOT / "conversations" / "prompts" / "scripted" / "base_impl"
        / "todo_plan_stage.txt"
    ).read_text()
    add_timings_text = (
        ROOT / "conversations" / "prompts" / "scripted" / "base_impl"
        / "add_timings.txt"
    ).read_text()
    assert '"<QUERY_ID> | Execution ms: <time>"' in todo_text
    assert '"<QUERY_ID> | Query ms: <time>"' in todo_text
    assert "<RUN_NR> | Execution ms" not in todo_text
    assert '"<QUERY_ID> | Execution ms: YYY"' in add_timings_text
    assert '"<QUERY_ID> | Query ms: ZZZ"' in add_timings_text
    assert "must exclude in-memory CSV string formatting/materialization" in add_timings_text
    assert "has_kernel_ms_override" in add_timings_text
    assert "kernel_ms_override" in add_timings_text
    assert "result-materialization boundary" in add_timings_text
    assert "<RUN_NR> | Execution ms" not in add_timings_text
    return None


def test_finish_skeleton_prompt_prioritizes_fidelity_risks() -> None:
    """finish_skeleton 应要求先看 design_evidence，并优先处理 fidelity 风险。"""
    text = (
        ROOT / "conversations" / "prompts" / "scripted" / "base_impl"
        / "finish_skeleton_stage.txt"
    ).read_text()
    assert "Read `design_evidence.md` before deciding the first vertical slice." in text
    assert "Prioritize loader fidelity, output fidelity, join fidelity, ordering fidelity, and numeric formatting immediately" in text
    assert "Do not defer missing table rows, date parsing, join-key typing, ORDER BY stability, or decimal/float formatting" in text
    return None


def test_query_impl_path_hint_includes_on_demand_query_modules() -> None:
    """query_impl_path 变量描述提及 dispatcher 加按需 query module."""
    text = (ROOT / "run_gen_base_impl_tpch_monetdb.py").read_text()
    assert "dispatcher ABI" in text
    assert "`query_q*.cpp`" in text
    assert "`query_shared_*.cpp`" in text
    return None


def test_builder_template_keeps_engine_as_data_container() -> None:
    """builder_impl.hpp 不再声明 query 方法，避免把 query 执行入口绑回 Engine."""
    text = (TEMPLATES_DIR / "builder_impl.hpp").read_text()
    assert "inline void query_lastpoint() {}" not in text
    assert "void query_lastpoint();" not in text
    assert "void query_single_groupby(" not in text
    return None


def test_query_impl_template_routes_to_query_modules_without_engine_stub_defs() -> None:
    """query_impl.cpp 用 dispatcher + per-query helper 路由，避免与 query_q*.cpp 重定义冲突。"""
    text = (TEMPLATES_DIR / "query_impl.cpp").read_text()
    assert '#include "query_registry_generated.hpp"' in text
    assert "dispatch_query(*engine, request);" in text
    assert "__has_include" not in text
    assert "void Engine::query_lastpoint() {}" not in text
    assert "void Engine::query_single_groupby(" not in text
    assert "void execute_q1(" not in text
    assert "void execute_q15(" not in text
    return None


def test_scripted_stages_enforce_primary_query_files_with_shared_edits_allowed() -> None:
    """per-query stage 文案要求主落点 query_q{qid}.cpp，同时允许共享文件联动修复。"""
    texts = [
        (
            ROOT / "conversations" / "prompts" / "scripted" / "base_impl"
            / "implement_q1.txt"
        ).read_text(),
        (
            ROOT / "conversations" / "prompts" / "scripted" / "base_impl"
            / "implement_q2.txt"
        ).read_text(),
        (
            ROOT / "conversations" / "prompts" / "scripted" / "base_impl"
            / "independent_query_implementation.txt"
        ).read_text(),
    ]
    text = "\n".join(texts)
    assert "Primary implementation must live in `query_q1.cpp`" in text
    assert "Primary implementation must live in `query_q2.cpp`" in text
    assert "Primary implementation must live in `query_q$qid.cpp`" in text
    assert "You may also edit existing `query_q*.cpp` / `query_q*.hpp`" in text
    assert "dispatch_query(...)" in text
    assert "dispatch_unimplemented_query(...)" in text
    assert "query_impl.cpp` thin and do not add another `Engine::query_*` definition or any `execute_q*` stub" in text
    assert "Do not place non-declarative query logic in `builder_impl.hpp`" in text
    assert "Keep `query_impl.cpp` thin" in text
    assert "do not add another `Engine::query_*` definition" in text
    assert "build-generated registry under `build/generated/`" in text
    assert "do not hand-edit generated registry files" in text
    assert "Engine consumer" in text
    assert "TPC-H table loading, source-file discovery, and raw-row reconstruction belong to loader/builder" in text
    assert "critical_query_access_paths" in text
    assert "selected access path" in text
    assert "Ignore rejected candidate ideas" in text
    assert "repair RawData, `builder_impl`, or Engine layout first" in text
    return None


def test_scripted_correctness_prompts_preserve_engine_boundary() -> None:
    prompt_dir = ROOT / "conversations" / "prompts" / "scripted" / "base_impl"
    text = "\n".join(
        (prompt_dir / name).read_text()
        for name in (
            "correctness_q1.txt",
            "correctness_q2.txt",
            "single_query_correctness.txt",
            "family_kernel_correctness.txt",
        )
    )
    assert "Correctness fixes must preserve the Engine boundary" in text
    assert "instead of reading or reconstructing source `.tbl` rows inside query code" in text
    assert "table columns, join keys, date filters" in text
    return None


def test_todo_and_finish_skeleton_prompts_require_query_ready_engine_boundary() -> None:
    prompt_dir = ROOT / "conversations" / "prompts" / "scripted" / "base_impl"
    text = "\n".join(
        (prompt_dir / name).read_text()
        for name in ("todo_plan_stage.txt", "finish_skeleton_stage.txt")
    )
    assert "Queries must be Engine consumers" in text
    assert "query-ready indexes/layout" in text
    assert "Per-query Engine dependencies and expected runtime shape" in text
    assert "Build Engine as the query-ready boundary" in text
    return None


def test_code_style_mentions_query_q_primary_logic_and_thin_dispatcher() -> None:
    """code-style 只补充结构原则，不取代 runtime 约束。"""
    text = (ROOT / "agent_rules" / "code-style.md").read_text()
    assert "per-query ABI entrypoints in `query_q{qid}.cpp`" in text
    assert "keep `query_impl.cpp` as dispatcher/routing glue" in text
    assert "query_shared_*" in text
    return None
