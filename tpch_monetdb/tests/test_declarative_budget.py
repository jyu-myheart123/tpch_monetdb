"""Tests for phase10 declarative budget config and optimization stage model.

验证:
- phase10 预算配置值符合规格
- runtime_stage_policy.py 从 phase10 配置读取预算
- run_outer_loop_tpch_monetdb.py 的 argparse 默认值来自 phase10 配置
- _check_convergence 早停已被移除
- 当前 optimization 路径: trace_expert -> global_human_reference
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TPCH_MONETDB_ROOT = ROOT / "tpch_monetdb"
if str(TPCH_MONETDB_ROOT) not in sys.path:
    sys.path.insert(0, str(TPCH_MONETDB_ROOT))


# ---------------------------------------------------------------------------
# phase10 budget config tests
# ---------------------------------------------------------------------------

def test_compile_fix_budget_is_512() -> None:
    """compile_fix 预算应为 512（增加 SIMD/向量化编译路径的裕量）."""
    from tpch_monetdb.config import PHASE10_STAGE_TURN_BUDGETS
    assert PHASE10_STAGE_TURN_BUDGETS["compile_fix"] == 512


def test_add_timings_budget_is_160() -> None:
    """add_timings 预算应为 160（15 个查询，3 批次跑满）."""
    from tpch_monetdb.config import PHASE10_STAGE_TURN_BUDGETS
    assert PHASE10_STAGE_TURN_BUDGETS["add_timings"] == 160


def test_trace_expert_budget_is_420() -> None:
    """trace_expert stage 默认预算应为 420."""
    from tpch_monetdb.config import PHASE10_OPTIM_STAGE_MAX_TURNS

    assert PHASE10_OPTIM_STAGE_MAX_TURNS["trace_expert"] == 420
    return None


def test_correctness_budgets_increased() -> None:
    """correctness 相关阶段预算应高于旧值."""
    from tpch_monetdb.config import PHASE10_STAGE_TURN_BUDGETS
    assert PHASE10_STAGE_TURN_BUDGETS["correctness_single_query"] >= 256
    assert PHASE10_STAGE_TURN_BUDGETS["all_queries_correctness"] >= 256


def test_optim_stage_turns_defined_for_current_stages() -> None:
    """当前活动优化阶段应有独立预算配置."""
    from tpch_monetdb.config import PHASE10_OPTIM_STAGE_MAX_TURNS
    assert PHASE10_OPTIM_STAGE_MAX_TURNS == {
        "trace_expert": 420,
        "global_human_reference": 360,
    }
    return None


def test_profile_observation_limits_defined_for_real_profiles() -> None:
    """真实 tool_profile 的 observation limit 应从 phase10 配置读取."""
    from tpch_monetdb.config import get_profile_observation_limits

    assert get_profile_observation_limits("finish_skeleton") == (48, 144)
    assert get_profile_observation_limits("correctness_foundation") == (24, 96)
    assert get_profile_observation_limits("optimize_build") == (24, 96)


def test_stalled_and_failure_limits_defined() -> None:
    """stalled execution / auto-compact circuit breaker 应有真实 phase10 配置."""
    from tpch_monetdb.config import (
        get_max_consecutive_failures,
        get_max_stalled_executions,
    )

    assert get_max_stalled_executions() == 3
    assert get_max_consecutive_failures() == 5


def test_outer_defaults_max_rounds_is_6() -> None:
    """outer loop 默认 max_rounds 应为 6."""
    from tpch_monetdb.config import get_outer_loop_defaults
    d = get_outer_loop_defaults()
    assert d["max_rounds"] == 6


def test_outer_defaults_stagnant_rounds_is_3() -> None:
    """outer loop 默认 stagnant_rounds 应为 3."""
    from tpch_monetdb.config import get_outer_loop_defaults
    d = get_outer_loop_defaults()
    assert d["stagnant_rounds"] == 3


def test_outer_defaults_retry_budget_is_2() -> None:
    """outer loop 默认 retry_budget 应为 2."""
    from tpch_monetdb.config import get_outer_loop_defaults
    d = get_outer_loop_defaults()
    assert d["retry_budget"] == 2


def test_get_stage_turn_budget_falls_back_to_75() -> None:
    """未知 stage 应回退到 75."""
    from tpch_monetdb.config import get_stage_turn_budget
    assert get_stage_turn_budget("nonexistent_stage") == 75


# ---------------------------------------------------------------------------
# runtime_stage_policy tests
# ---------------------------------------------------------------------------

def test_runtime_stage_policy_compile_fix_matches_declared_budget() -> None:
    """STAGE_RUNTIME_POLICIES compile_fix.base_turns 应与 phase10 配置一致."""
    from tpch_monetdb.config import PHASE10_STAGE_TURN_BUDGETS
    from tpch_monetdb.runtime_stage_policy import STAGE_RUNTIME_POLICIES
    assert STAGE_RUNTIME_POLICIES["compile_fix"].base_turns == PHASE10_STAGE_TURN_BUDGETS["compile_fix"]


def test_runtime_stage_policy_correctness_single_query_matches_declared_budget() -> None:
    """STAGE_RUNTIME_POLICIES correctness_single_query.base_turns 应与 phase10 配置一致."""
    from tpch_monetdb.config import PHASE10_STAGE_TURN_BUDGETS
    from tpch_monetdb.runtime_stage_policy import STAGE_RUNTIME_POLICIES
    assert (
        STAGE_RUNTIME_POLICIES["correctness_single_query"].base_turns
        == PHASE10_STAGE_TURN_BUDGETS["correctness_single_query"]
    )


def test_default_stage_turn_budget_is_alias_of_declared_budget() -> None:
    """DEFAULT_STAGE_TURN_BUDGET 应为 phase10 配置的别名，不是独立副本."""
    from tpch_monetdb.config import PHASE10_STAGE_TURN_BUDGETS
    from tpch_monetdb.runtime_stage_policy import DEFAULT_STAGE_TURN_BUDGET
    for key in PHASE10_STAGE_TURN_BUDGETS:
        assert DEFAULT_STAGE_TURN_BUDGET.get(key) == PHASE10_STAGE_TURN_BUDGETS[key]


# ---------------------------------------------------------------------------
# Three-stage path tests
# ---------------------------------------------------------------------------

def test_three_stage_prompt_files_exist() -> None:
    """三个活动优化阶段的 prompt 文件应存在于 canonical stages 目录."""
    prompts_dir = TPCH_MONETDB_ROOT / "conversations" / "prompts" / "optimization" / "stages"
    required = [
        "tpch_monetdb_optim_w_trace.txt",
        "tpch_monetdb_optim_w_expert_knowledge.txt",
        "tpch_monetdb_optim_w_human_reference.txt",
    ]
    for fname in required:
        assert (prompts_dir / fname).exists(), f"Missing prompt file: {fname}"


def test_expert_knowledge_prompt_covers_layout_and_kernel_modes() -> None:
    """merged expert prompt 应同时覆盖 layout/access-path 与 kernel/compute 两类瓶颈."""
    prompts_dir = TPCH_MONETDB_ROOT / "conversations" / "prompts" / "optimization" / "stages"
    text = (prompts_dir / "tpch_monetdb_optim_w_expert_knowledge.txt").read_text().lower()
    assert "layout/access-path bound" in text
    assert "kernel/compute bound" in text
    assert "rows scanned" in text
    assert "join fanout" in text
    assert "group cardinality" in text
    assert "sort/top-k cardinality" in text
    assert "${expert_knowledge}" in text
    assert "${query_guidance}" in text


def test_expert_knowledge_asset_covers_scale_and_container_strategy() -> None:
    """shared expert knowledge 应覆盖规模敏感策略与容器选择层级."""
    prompts_dir = TPCH_MONETDB_ROOT / "conversations" / "prompts" / "shared"
    text = (prompts_dir / "tpch_monetdb_expert_knowledge.txt").read_text().lower()
    assert "table cardinality" in text
    assert "join fanout" in text
    assert "group cardinality" in text
    assert "candidate/output cardinality" in text
    assert "direct indexing" in text
    assert "std::unordered_map" in text
    assert "open-addressing" in text


def test_prompts_gen_removes_legacy_expert_knowledge_function() -> None:
    """tpch_monetdb_prompts_gen 不应保留 legacy expert prompt 兼容函数."""
    src = (TPCH_MONETDB_ROOT / "conversations" / "tpch_monetdb_prompts_gen.py").read_text()
    assert "tpch_monetdb_optim_prompt_with_expert_knowledge" not in src
    return None


def test_optimization_conversation_imports_expert_knowledge() -> None:
    """optimization_conversation_tpch_monetdb 应导入当前 trace_expert prompt 构建函数."""
    src = (TPCH_MONETDB_ROOT / "conversations" / "optimization_conversation_tpch_monetdb.py").read_text()
    assert "tpch_monetdb_optim_prompt_trace_expert" in src
    assert "tpch_monetdb_optim_prompt_with_expert_knowledge" not in src
    assert "get_optim_stage_max_turns" in src
    return None


def test_optimization_conversation_uses_current_two_level_stage_model() -> None:
    """optimization_conversation_tpch_monetdb 应使用 per-query trace_expert 与全局 hypothesis competition."""
    src = (TPCH_MONETDB_ROOT / "conversations" / "optimization_conversation_tpch_monetdb.py").read_text()
    assert "def _build_query_stage" in src
    assert "name=\"trace_expert\"" in src
    assert "def _run_global_human_reference" in src
    assert "tpch_monetdb_optim_prompt_global_diagnosis" in src
    assert "tpch_monetdb_optim_prompt_hypothesis_execution" in src
    assert "select_global_winner" in src
    assert "def _build_stages" not in src
    assert "name=\"trace\"" not in src
    assert "name=\"expert_knowledge\"" not in src
    assert "name=\"human_reference\"" not in src
    assert '"expert_layout"' not in src
    assert '"expert_kernel"' not in src
    return None


def test_inner_loop_convergence_check_removed() -> None:
    """_check_convergence 驱动的早停逻辑应被移除."""
    src = (TPCH_MONETDB_ROOT / "conversations" / "optimization_conversation_tpch_monetdb.py").read_text()
    # 会话内 stagnant break 不应再存在
    assert "Convergence detected. Stopping optimization loop early" not in src
    assert "stagnant_count >= self.min_stagnant_rounds" not in src
    assert "def _check_convergence" not in src
