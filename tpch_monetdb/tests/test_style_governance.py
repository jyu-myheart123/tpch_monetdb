"""Tests for phase10 generated code style governance.

验证:
- code-style.md 存在且元数据正确
- code-style.md 包含所有必要的 Google C++ style 约束
- code-style.md 不引入 repo-wide clang-format 要求
- agent_rules 装配链可以加载 code-style.md
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

AGENT_RULES_DIR = ROOT / "tpch_monetdb" / "agent_rules"
CODE_STYLE_PATH = AGENT_RULES_DIR / "code-style.md"


def _read_code_style() -> str:
    return CODE_STYLE_PATH.read_text()


# ---------------------------------------------------------------------------
# File existence and metadata tests (task 11.1)
# ---------------------------------------------------------------------------

def test_code_style_md_exists() -> None:
    """code-style.md 应存在于 agent_rules 目录."""
    assert CODE_STYLE_PATH.exists(), f"Missing {CODE_STYLE_PATH}"


def test_code_style_has_stages_metadata() -> None:
    """code-style.md 应包含 stages 元数据，覆盖 scripted 和 optimization."""
    text = _read_code_style()
    assert "finish_skeleton" in text
    assert "compile_fix" in text
    assert "optimization_general" in text


def test_code_style_has_priority_metadata() -> None:
    """code-style.md 应有 priority 元数据."""
    text = _read_code_style()
    assert "priority:" in text


# ---------------------------------------------------------------------------
# Content tests (task 11.1)
# ---------------------------------------------------------------------------

def test_code_style_covers_naming_conventions() -> None:
    """code-style.md 应覆盖命名约定（snake_case, PascalCase）."""
    text = _read_code_style().lower()
    assert "snake_case" in text
    assert "pascalcase" in text


def test_code_style_covers_google_cpp_layout_rules() -> None:
    """code-style.md 应覆盖 Google C++ 的括号/控制流布局要求."""
    text = _read_code_style().lower()
    assert "opening brace" in text
    assert "always use braces" in text
    assert "control-flow" in text or "control flow" in text


def test_code_style_covers_four_programming_styles() -> None:
    """code-style.md 应明确过程式/命令式/声明式/函数式规则."""
    text = _read_code_style().lower()
    assert "procedural" in text
    assert "imperative" in text
    assert "declarative" in text
    assert "functional" in text


def test_code_style_covers_simd_guard() -> None:
    """code-style.md 应要求 SIMD 使用 __AVX2__ 守卫."""
    text = _read_code_style()
    assert "__AVX2__" in text or "avx2" in text.lower()


def test_code_style_covers_scalar_fallback() -> None:
    """code-style.md 应要求 SIMD 路径保留 scalar fallback."""
    text = _read_code_style().lower()
    assert "scalar fallback" in text or "scalar" in text


def test_code_style_no_repo_wide_clang_format() -> None:
    """code-style.md 不应要求 repo-wide .clang-format 文件."""
    text = _read_code_style().lower()
    assert "repo-wide .clang-format" not in text
    assert "clang-format" not in text or "repo-wide" not in text


def test_code_style_limits_scope_to_tpch_monetdb_generated_code() -> None:
    """code-style.md 应明确约束范围仅限 TPC-H MonetDB 生成代码."""
    text = _read_code_style()
    assert "tpch_monetdb/misc/tpch/templates" in text


# ---------------------------------------------------------------------------
# Agent rules loading integration test (task 11.2)
# ---------------------------------------------------------------------------

def test_agent_rules_assembly_can_load_code_style() -> None:
    """load_agent_rules 应能加载并包含 code-style.md (scripted/optimization scope)."""
    from tpch_monetdb.utils.agent_rules import load_agent_rules, RuleScope

    scope = RuleScope(stage_name="finish_skeleton", area_name="runtime")
    assembly = load_agent_rules(AGENT_RULES_DIR, scope=scope, include_global=True)
    assert "code-style.md" in assembly.included_files or any(
        "code-style" in f for f in assembly.included_files
    ), f"code-style.md not included. included_files={assembly.included_files}"


def test_optimization_scope_also_loads_code_style() -> None:
    """optimization_general 也应加载同一份 code-style.md."""
    from tpch_monetdb.utils.agent_rules import load_agent_rules, RuleScope

    scope = RuleScope(stage_name="optimization_general", area_name="provider")
    assembly = load_agent_rules(AGENT_RULES_DIR, scope=scope, include_global=True)
    assert any("code-style" in f for f in assembly.included_files)


def test_code_style_not_loaded_for_unrelated_stage() -> None:
    """code-style.md 不应被加载到与 TPC-H MonetDB 生成代码无关的 stage（例如 todo_plan）."""
    from tpch_monetdb.utils.agent_rules import load_agent_rules, RuleScope

    scope = RuleScope(stage_name="todo_plan", area_name="runtime")
    assembly = load_agent_rules(AGENT_RULES_DIR, scope=scope, include_global=False)
    # It should not be in the included files for a stage outside its scope
    code_style_included = any("code-style" in f for f in assembly.included_files)
    assert not code_style_included, (
        f"code-style.md should not be loaded for todo_plan stage. "
        f"included_files={assembly.included_files}"
    )
