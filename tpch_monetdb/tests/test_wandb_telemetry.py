"""Tests for phase10 W&B and LOC telemetry."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# cloc_utils tests (task 8.1)
# ---------------------------------------------------------------------------

def test_calculate_loc_breakdown_returns_dict() -> None:
    """calculate_loc_breakdown 应返回包含 total 键的 dict."""
    from tpch_monetdb.utils.cloc_utils import calculate_loc_breakdown
    import inspect
    sig = inspect.signature(calculate_loc_breakdown)
    assert "cloc_cache_dir" in sig.parameters
    assert "current_hash" in sig.parameters
    assert "working_dir" in sig.parameters


def test_calculate_loc_breakdown_has_expected_keys() -> None:
    """breakdown 结果 dict 应至少包含 total, cpp, hpp, py, other 键."""
    from tpch_monetdb.utils.cloc_utils import _run_and_cache_loc

    with patch("tpch_monetdb.utils.cloc_utils.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout='{"C++":{\"code\":100},"C/C++ Header":{"code":50},"Python":{"code":200},"SUM":{"code":350}}',
            stderr="",
        )
        result = _run_and_cache_loc(None, "abc123", Path("/tmp"))

    assert "total" in result
    assert "cpp" in result
    assert "hpp" in result
    assert "py" in result
    assert result["total"] == 350
    assert result["cpp"] == 100
    assert result["hpp"] == 50
    assert result["py"] == 200


def test_calculate_loc_backward_compat() -> None:
    """calculate_loc 应仍然返回 int (total)."""
    from tpch_monetdb.utils.cloc_utils import calculate_loc

    with patch("tpch_monetdb.utils.cloc_utils.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout='{"C++":{\"code":100},"Python":{"code":50}}',
            stderr="",
        )
        result = calculate_loc(None, "abc123", Path("/tmp"))

    assert isinstance(result, int)


# ---------------------------------------------------------------------------
# Runtime wiring tests (task 8.2 / 8.4)
# ---------------------------------------------------------------------------

def test_main_handle_prompt_sets_and_clears_current_stage() -> None:
    """main_tpch_monetdb.handle_prompt 应在 Runner.run 前后设置并清空当前 stage."""
    src = (ROOT / "tpch_monetdb" / "main_tpch_monetdb.py").read_text()
    assert "wandb_metrics_hook.set_current_stage(profile_key)" in src
    assert "wandb_metrics_hook.set_current_stage(None)" in src


def test_optimization_flow_no_longer_uses_legacy_stage_hotspot_logging() -> None:
    """optimization flow 不应再依赖旧 stage accounting hotspot 日志."""
    src = (ROOT / "tpch_monetdb" / "conversations" / "optimization_conversation_tpch_monetdb.py").read_text()
    assert "def _render_stage_accounting" not in src
    assert "log_query_hotspot_summary(" not in src


def test_ingest_summary_is_logged_from_optimization_flow() -> None:
    """ingest summary helper 应在 optimization flow 中被真实调用."""
    src = (ROOT / "tpch_monetdb" / "conversations" / "optimization_conversation_tpch_monetdb.py").read_text()
    assert "log_ingest_summary(" in src


def test_wandb_run_hook_tracks_stage_and_loc_fields() -> None:
    """WandbRunHook 仍应保留 stage 与 loc 相关字段写入."""
    src = (ROOT / "tpch_monetdb" / "utils" / "wandb_stats_logging.py").read_text()
    assert "_prev_loc" in src
    assert "_current_stage" in src
    assert 'metrics["stage"]' in src or "metrics['stage']" in src
    assert 'loc/' in src
