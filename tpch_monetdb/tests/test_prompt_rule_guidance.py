"""Regression tests for phase10 prompt/rule/guidance corrections.

锁定 prompt/rule 矫正点：SIMD 策略、Q1-Q22 文案、txt prompt 资产加载、
运行时阶段提示与条件式 optimize_build 行为。
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PROMPTS_DIR = ROOT / "conversations" / "prompts"
SHARED_PROMPTS_DIR = PROMPTS_DIR / "shared"
OPTIMIZATION_PROMPTS_DIR = PROMPTS_DIR / "optimization"
OPT_BASE_PROMPTS_DIR = OPTIMIZATION_PROMPTS_DIR / "base"
OPT_INSTRUMENTATION_PROMPTS_DIR = OPTIMIZATION_PROMPTS_DIR / "instrumentation"
OPT_STAGE_PROMPTS_DIR = OPTIMIZATION_PROMPTS_DIR / "stages"
AGENT_RULES_DIR = ROOT / "agent_rules"


def _read(path: Path) -> str:
    """读取文本并返回原始字符串."""
    return path.read_text()


def test_agent_rule_filenames_are_tpch_first() -> None:
    """agent rule 文件名不应保留 QuestDB/TSBS 主路径暗示."""
    rule_names = {path.name for path in AGENT_RULES_DIR.glob("*.md")}
    assert "tpch-validator.md" in rule_names
    assert not any(
        "questdb" in name.lower() or "tsbs" in name.lower()
        for name in rule_names
    )
    return None



def test_optim_constraints_drops_simd_ban_and_restores_anti_cheating() -> None:
    """optim_constraints 不禁 SIMD，且保留 TPC-H 防作弊与通用布局约束."""
    text = _read(SHARED_PROMPTS_DIR / "tpch_monetdb_optim_constraints.txt")
    lowered = text.lower()
    assert "out of scope for this optimization loop" not in lowered
    assert "simd intrinsics (let compiler auto-vectorize" not in lowered
    assert "no materialized answer tables" in lowered
    assert "no precomputed aggregate sidecars" in lowered
    assert "no query-specific" in lowered
    assert "remain general" in lowered
    assert "8 tpc-h tables" in lowered
    assert "typed columns" in lowered
    assert "foreign-key adjacency" in lowered
    # indexes/views 没有被重新放开
    assert "allow materialized answer" not in lowered
    assert "code-style" in lowered
    assert "optimization.md" in lowered
    assert "google c++ style" not in lowered
    assert "procedural / imperative" not in lowered


def test_optimization_stage_prompts_do_not_repeat_style_or_generic_workflow() -> None:
    """stage prompt 不应重复 style / generic workflow 文案."""
    for name in (
        "tpch_monetdb_optim_w_trace.txt",
        "tpch_monetdb_optim_w_expert_knowledge.txt",
        "tpch_monetdb_optim_w_human_reference.txt",
    ):
        text = _read(OPT_STAGE_PROMPTS_DIR / name).lower()
        assert "google c++ style" not in text
        assert "procedural / imperative" not in text
        assert "declarative dispatch" not in text
        assert "make sure the performance improved" not in text
        assert "avoid regressions on the other queries" not in text


def test_add_timings_prompt_covers_full_q1_q22_and_fixes_profile_scope_macro() -> None:
    """add-timings prompt 覆盖 Q1-Q22 并修复 PROFILE_SCOPE 的 token-paste 错误."""
    text = _read(
        OPT_INSTRUMENTATION_PROMPTS_DIR / "tpch_monetdb_optim_add_timings_collect_stats.txt"
    )
    assert "Q1-Q22" in text
    assert "TPC-H table parsing" in text
    assert "9 query execution paths" not in text
    assert "ILP lines parsed" not in text
    assert "TPCH_MONETDB_CONCAT_INNER(a, b) a##b" in text
    assert "TPCH_MONETDB_CONCAT(a, b) TPCH_MONETDB_CONCAT_INNER(a, b)" in text
    assert "PROFILE_SCOPE(name) ScopedTimer TPCH_MONETDB_CONCAT(_timer_, __LINE__)(name)" in text


def test_pretext_and_pinning_assets_exist_as_txt() -> None:
    """pretext / pretext_optim / pinning 必须作为独立 txt 资产存在."""
    for asset in (
        OPT_BASE_PROMPTS_DIR / "tpch_monetdb_optim_pretext.txt",
        OPT_BASE_PROMPTS_DIR / "tpch_monetdb_optim_pretext_optim.txt",
        OPT_BASE_PROMPTS_DIR / "tpch_monetdb_optim_pinning.txt",
    ):
        assert asset.exists(), f"missing prompt asset: {asset.name}"
        content = asset.read_text().strip()
        assert content, f"prompt asset {asset.name} is empty"


def test_finish_skeleton_prompt_is_fail_fast_not_stub_gate() -> None:
    """foundation skeleton 阶段只能建立 ABI/Engine 边界，不能靠假输出过 gate。"""
    text = _read(
        PROMPTS_DIR / "scripted" / "base_impl" / "finish_skeleton_stage.txt"
    )
    lowered = text.lower()
    assert "for now, use stubs" not in lowered
    assert "minimal stubs" not in lowered
    assert "fake csv" in lowered
    assert "synthetic results" in lowered
    assert "catch-all runtime path" in lowered
    assert "fail fast" in lowered


def test_merged_expert_prompt_asset_exists_at_canonical_stage_path() -> None:
    """merged expert stage prompt 资产应存在于 canonical stages 目录."""
    asset = OPT_STAGE_PROMPTS_DIR / "tpch_monetdb_optim_w_expert_knowledge.txt"
    assert asset.exists()
    assert asset.read_text().strip()


def test_prompts_gen_loads_active_text_assets() -> None:
    """tpch_monetdb_prompts_gen 只暴露当前 active prompt builder."""
    from tpch_monetdb.conversations.tpch_monetdb_prompts_gen import (
        load_expert_knowledge,
        tpch_monetdb_optim_prompt_pinning,
        tpch_monetdb_optim_prompt_pretext,
        tpch_monetdb_optim_prompt_pretext_optim,
        tpch_monetdb_optim_prompt_trace_expert,
    )

    pretext = tpch_monetdb_optim_prompt_pretext(queries_path="queries.txt", num_queries=22)
    assert "queries.txt" in pretext
    assert "Q1-Q22" in pretext
    assert "TPC-H relational data" in pretext
    assert "22 queries" in pretext

    pretext_optim = tpch_monetdb_optim_prompt_pretext_optim(bespoke_storage=True)
    assert "primary optimization gate" in pretext_optim
    # 不再把 Key constraints 块塞回这里（避免与 constraints.txt 重复）
    assert "Key constraints" not in pretext_optim

    pinning = tpch_monetdb_optim_prompt_pinning(core_id=3)
    assert "core 3" in pinning
    assert "taskset -c 3" in pinning

    trace_prompt = tpch_monetdb_optim_prompt_trace_expert(
        query_id="1",
        constraints_str="constraints",
        expert_knowledge=load_expert_knowledge(),
        trace_summary="Trace summary",
        query_guidance="Additional implementation guidance:\n- prefer a dense accumulator",
        current_rt_ms=1000.0,
        target_rt_ms=500.0,
        sf=10,
        storage_is_bespoke=True,
    )
    assert "layout/access-path bound" in trace_prompt
    assert "kernel/compute bound" in trace_prompt
    assert "Additional implementation guidance:" in trace_prompt
    assert "dense accumulator" in trace_prompt


def test_prompts_gen_does_not_export_legacy_prompt_helpers() -> None:
    """tpch_monetdb_prompts_gen 不再导出 legacy prompt helper."""
    src = (ROOT / "conversations" / "tpch_monetdb_prompts_gen.py").read_text()
    assert "def tpch_monetdb_optim_prompt_with_expert_knowledge" not in src
    assert "def tpch_monetdb_optim_prompt_with_human_reference" not in src
    assert "def tpch_monetdb_optim_prompt_w_trace" not in src


def test_split_stage_helpers_and_assets_are_removed_from_active_catalog() -> None:
    """旧的 layout/kernel helper 与 split-stage 资产应已清理."""
    src = (ROOT / "conversations" / "tpch_monetdb_prompts_gen.py").read_text()
    assert "def tpch_monetdb_optim_prompt_with_expert_layout" not in src
    assert "def tpch_monetdb_optim_prompt_with_expert_kernel" not in src
    assert not (OPT_STAGE_PROMPTS_DIR / "tpch_monetdb_optim_w_expert_layout.txt").exists()
    assert not (OPT_STAGE_PROMPTS_DIR / "tpch_monetdb_optim_w_expert_kernel.txt").exists()


def test_kernel_rule_is_conditional_about_write_first() -> None:
    """kernel.md 的 failure recovery 从绝对"write first"改成条件描述."""
    text = _read(AGENT_RULES_DIR / "kernel.md")
    assert "After failed `compile` or `run`, write first. Then retry." not in text
    assert "prefer one targeted read of the error" in text
    assert "only mandatory when" in text.lower() or "only mandatory when" in text


def test_scripted_rule_scope_matches_split_query_layout() -> None:
    """scripted.md 已经将 agent scope 放宽到按需 query module."""
    text = _read(AGENT_RULES_DIR / "scripted.md")
    assert "query_q*.cpp" in text
    assert "query_family_*.cpp" in text
    assert "query_shared_*.cpp" in text
    assert "focused query modules" in text
    assert "query_api.hpp" in text
    assert "host-facing API files" in text
    assert "`query_*.hpp`" not in text
    normalized = " ".join(text.split())
    assert (
        "`query_impl.cpp` is no longer the only query implementation file"
        in normalized
    )
    assert "manifest-owned `query_family_*.cpp`" in text
    return None


def test_scripted_rule_optimize_build_is_no_longer_absolute_builder_only() -> None:
    """optimize_build 规则改为条件式，允许 ingest 证据下的最小跨界."""
    text = _read(AGENT_RULES_DIR / "scripted.md")
    assert "builder-only" not in text
    assert "prioritises builder" in text
    # 明确允许在 ingest 证据下最小必要地触碰 loader
    assert "may touch loader or query" in text or "may touch loader" in text


def test_optimization_rule_requires_multi_run_and_focused_edits() -> None:
    """optimization.md 覆盖 measure/revert/focus/style 准则."""
    text = _read(AGENT_RULES_DIR / "optimization.md")
    assert "multiple runs" in text.lower() or "≥" in text or "median" in text.lower()
    assert "regression_tolerance" in text
    assert "scatter-shot" in text.lower()
    assert "query_q12.cpp" in text
    assert "query_shared_groupby.hpp" in text
    assert "trace instrumentation off" in text.lower()
    assert "code-style.md" in text
    assert "query_family_*" in text
    return None


def test_runtime_stage_hints_only_expose_scope_and_stop_conditions() -> None:
    """stage hint 只应暴露 scope 和 stop condition，不重复 workflow 说明."""
    tools_path = ROOT / "tools" / "tpch_monetdb_agent_tools.py"
    asset_path = ROOT / "conversations" / "prompts" / "shared" / "runtime" / "stage_hint_stop_condition.txt"
    storage_plan_condition_path = (
        ROOT / "conversations" / "prompts" / "shared" / "runtime"
        / "stage_hint_condition_storage_plan.txt"
    )
    text = tools_path.read_text()
    asset_text = asset_path.read_text()
    storage_plan_condition = storage_plan_condition_path.read_text()
    assert "Stop condition:" in asset_text
    assert "storage_plan_contract.json" in storage_plan_condition
    assert "Stop condition:" not in text
    assert "write TODO.md and stop this stage." not in text
    assert "write storage_plan.txt and stop this stage." not in text
    assert "current query validation must pass before this stage can advance." not in text
    assert "(none available in this stage)" not in text
    assert "(no editable files)" not in text
    assert "Creatable scope:" not in text
    assert "Goal: write a complete" not in text
    assert "Do not reconcile args_parser.hpp or QueryRequest here." not in text


def test_optimize_build_prompt_is_conditional_not_absolute_builder_only() -> None:
    """run_gen_base_impl_tpch_monetdb 的 optimize_build prompt 不再是绝对 builder-only 且不以硬件计数器作 gate."""
    text = (
        ROOT / "conversations" / "prompts" / "scripted" / "base_impl"
        / "optimize_build.txt"
    ).read_text()
    assert "This stage is build-only" not in text
    assert "Stay builder-centric by default" in text
    assert "hardware counters such as cache-miss are diagnostic only" not in text
    assert "Workflow priority:" not in text
    src = (ROOT / "run_gen_base_impl_tpch_monetdb.py").read_text()
    assert "Stay builder-centric by default" not in src


def test_optimize_build_prompt_allows_conditional_loader_touching() -> None:
    """optimize_build prompt 允许 loader 最小必要介入."""
    text = (
        ROOT / "conversations" / "prompts" / "scripted" / "base_impl"
        / "optimize_build.txt"
    ).read_text()
    assert "Touch loader or query code only when ingest" in text


def test_base_impl_stage_prompts_are_kept_in_english() -> None:
    """base_impl 运行时 stage prompt 不应再混入中文指令文本."""
    prompt_dir = ROOT / "conversations" / "prompts" / "scripted" / "base_impl"
    text = "\n".join(path.read_text() for path in prompt_dir.glob("*.txt"))
    assert (
        "Before making any changes, use the read_file tool to read TODO.md, "
        "storage_plan.txt, and storage_plan_contract.json."
        in text
    )
    assert "workload_objective.json" in text
    assert "data_law_contract.json" in text
    assert "Before making any changes, use the read_file tool to read:" in text
    assert "**Query family**:" in text
    assert "**Create only a thin entrypoint**" in text
    assert "**Refactor `query_q2.cpp`**:" in text
    assert "Do not run compile or run." in text
    assert "在开始任何工作之前" not in text
    assert "查询族" not in text
    assert "只创建薄入口点" not in text
    assert "不要运行 compile 或 run" not in text
    assert "重构 `query_q2.cpp`" not in text
    src = (ROOT / "run_gen_base_impl_tpch_monetdb.py").read_text()
    assert "Before making any changes, use the read_file tool" not in src


def test_code_style_mentions_family_kernel_ownership() -> None:
    """code-style 需要承认 family kernel ownership，而不是只允许 per-query 主逻辑."""
    text = (ROOT / "agent_rules" / "code-style.md").read_text()
    assert "ABI entrypoints in `query_q{qid}.cpp`" in text
    assert "manifest-owned `query_family_*` files" in text
    assert "query_shared_*" in text
    return None


def test_prompt_rule_lint_blocks_legacy_primary_logic_constraint_in_rules_and_assets() -> None:
    """family kernel 方案下，agent rules 与 scripted prompt 资产都不能回流旧硬约束."""
    code_style_text = (ROOT / "agent_rules" / "code-style.md").read_text()
    scripted_text = (ROOT / "agent_rules" / "scripted.md").read_text()
    optimization_text = (ROOT / "agent_rules" / "optimization.md").read_text()
    asset_root = ROOT / "conversations" / "prompts" / "scripted"
    asset_text = "\n".join(path.read_text() for path in sorted(asset_root.rglob("*.txt")))
    forbidden = "Keep per-query primary logic in `query_q{qid}.cpp`"
    assert forbidden not in code_style_text
    assert forbidden not in scripted_text
    assert forbidden not in optimization_text
    assert forbidden not in asset_text
    assert "query_family_*" in scripted_text
    assert "query_family_*" in optimization_text
    return None
