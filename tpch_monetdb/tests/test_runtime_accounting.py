"""Phase9 runtime accounting focused tests.

Covers tasks 2.4, 2.6, 3.7, 4.5, 5.7, 6.4, 7.4 from the phase9 runtime-gap closure plan.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tpch_monetdb.benchmark.runtime_accounting import (
    KERNEL_RUNTIME_METRIC_KIND,
    QUERY_RUNTIME_METRIC_KIND,
    RUNTIME_SCHEMA_VERSION,
    QuerySamples,
    build_runtime_timeout_policy,
    check_ingest_completeness,
    detect_runtime_execution_failure,
    derive_bespoke_ingest_metrics,
    is_lazy_build_suspected,
    parse_ingest_timing,
    parse_ingest_timing_from_text,
    parse_query_timing,
    parse_query_timing_by_id,
    parse_query_timing_by_id_with_metric_kind,
)
from tpch_monetdb.oracle.validate_cache import build_validate_cache_context


# ---------------------------------------------------------------------------
# Task 2.4 — Provider and validator use same parser implementation
# ---------------------------------------------------------------------------


STDOUT_QUERY_MS = "1 | Query ms: 12.5\n2 | Query ms: 8.3\n"
STDOUT_KERNEL_ONLY = "1 | Execution ms: 9.0\n2 | Execution ms: 7.2\n"
STDOUT_MIXED = "1 | Query ms: 12.5\n1 | Execution ms: 9.0\n2 | Query ms: 8.3\n2 | Execution ms: 7.2\n"


def test_shared_parser_provider_and_validator_give_same_primary():
    """同一份 stdout 在 provider path (index) 和 validator path (by_id) 得到相同主 runtime。"""
    result_idx0 = parse_query_timing(STDOUT_MIXED, "", "Q1", index=0)
    result_idx1 = parse_query_timing(STDOUT_MIXED, "", "Q2", index=1)

    impl_map, kernel_map, fallback_map = parse_query_timing_by_id(STDOUT_MIXED, ["1", "2"])

    assert result_idx0.primary_runtime_ms == impl_map["1"] == 12.5
    assert result_idx1.primary_runtime_ms == impl_map["2"] == 8.3
    assert result_idx0.primary_metric_kind == QUERY_RUNTIME_METRIC_KIND
    assert result_idx1.primary_metric_kind == QUERY_RUNTIME_METRIC_KIND
    assert result_idx0.kernel_runtime_ms == kernel_map["1"] == 9.0
    assert result_idx1.kernel_runtime_ms == kernel_map["2"] == 7.2
    assert not fallback_map


def test_shared_parser_rejects_kernel_only_official_runtime():
    """Execution ms 不再作为 official runtime fallback。"""
    with pytest.raises(ValueError, match="Official Query ms missing"):
        parse_query_timing(STDOUT_KERNEL_ONLY, "", "Q1", index=0)
    kernel_result = parse_query_timing(
        STDOUT_KERNEL_ONLY,
        "",
        "Q1",
        index=0,
        primary_metric_kind=KERNEL_RUNTIME_METRIC_KIND,
    )
    assert kernel_result.primary_runtime_ms == 9.0
    assert kernel_result.primary_metric_kind == KERNEL_RUNTIME_METRIC_KIND

    impl_map, kernel_map, fallback_map = parse_query_timing_by_id(STDOUT_KERNEL_ONLY, ["1"])
    assert impl_map == {}
    assert kernel_map["1"] == 9.0
    assert fallback_map["1"] == "official_query_runtime_missing"


def test_shared_parser_by_id_reports_missing_official_metric_kind() -> None:
    """validator path 应能区分 official Query ms 缺失与 kernel 诊断口径。"""
    impl_map, kernel_map, fallback_map, metric_kind_map = (
        parse_query_timing_by_id_with_metric_kind(STDOUT_KERNEL_ONLY, ["1"])
    )

    assert impl_map == {}
    assert kernel_map["1"] == 9.0
    assert fallback_map["1"] == "official_query_runtime_missing"
    assert metric_kind_map["1"] == KERNEL_RUNTIME_METRIC_KIND
    return None


def test_shared_parser_index_out_of_range_raises():
    """index 越界时显式抛出 ValueError（不静默回退）。"""
    with pytest.raises(ValueError, match="out of range"):
        parse_query_timing(STDOUT_QUERY_MS, "", "Q3", index=5)


def test_shared_parser_no_timing_raises():
    """完全没有 timing 输出时显式抛出 ValueError。"""
    with pytest.raises(ValueError, match="No official timing"):
        parse_query_timing("some random output", "", "Q1")


def test_runtime_timeout_policy_separates_cold_and_warm_runs() -> None:
    """SF1 cold-start timeout must be much larger than warm query timeout."""
    policy = build_runtime_timeout_policy(1, num_queries=1)

    assert policy.cold_start_timeout_s >= 180
    assert policy.warm_query_timeout_s == 60
    assert policy.trace_timeout_s >= 240

    baseline_policy = build_runtime_timeout_policy(
        1,
        num_queries=1,
        baseline_runtime_ms=40.0,
    )
    assert baseline_policy.warm_query_timeout_s == 10
    return None


def test_runtime_execution_failure_detects_timeout_before_timing_parse() -> None:
    """Timeout response wins even when stdout still contains old timing lines."""
    failure = detect_runtime_execution_failure(
        "Terminated after 30 seconds due to timeout",
        "1 | Execution ms: 12.0\n",
        "",
    )

    assert failure is not None
    assert failure.failure_code == "RUNNER_TIMEOUT"
    return None


# ---------------------------------------------------------------------------
# Task 2.6 — Cache schema version isolates old and new runs
# ---------------------------------------------------------------------------


def test_cache_context_includes_runtime_schema_version():
    """build_validate_cache_context 必须包含 runtime_schema_version。"""
    ctx = build_validate_cache_context(
        validation_mode="strict",
        trace_mode=False,
        other_config=None,
        data_config={},
        allowed_query_ids=["1"],
        oracle_http_url="http://localhost:9000",
        oracle_timeout_s=30,
        output_stdout_stderr=False,
    )
    assert "runtime_schema_version" in ctx
    assert ctx["runtime_schema_version"] == RUNTIME_SCHEMA_VERSION


def test_different_schema_versions_yield_different_cache_keys():
    """runtime_schema_version 变化后，cache key 应不同（旧口径 cache 不复用）。"""
    from tpch_monetdb.oracle.validate_cache import TpchValidateCache
    import json, hashlib

    params = {"hostnames": ["host_0"], "time_start": "2016-01-01", "time_end": "2016-01-02",
              "bucket_width": None, "threshold": None, "limit": None}

    ctx_v1 = {"runtime_schema_version": "phase9_query_e2e_v1"}
    ctx_v2 = {"runtime_schema_version": RUNTIME_SCHEMA_VERSION}

    cache = TpchValidateCache.__new__(TpchValidateCache)

    key1 = cache._compute_cache_key("abc123", "1", 1.0, params, ctx_v1)
    key2 = cache._compute_cache_key("abc123", "1", 1.0, params, ctx_v2)
    assert key1 != key2, "Different runtime_schema_version must produce different cache keys"


# ---------------------------------------------------------------------------
# Task 3.7 — Query runtime takes median, not first run
# ---------------------------------------------------------------------------


def test_query_samples_median_vs_first():
    """正式 runtime 取 median，而非 first run。"""
    samples = QuerySamples(measured_runs_ms=[100.0, 10.0, 11.0])
    assert samples.first_query_ms == 100.0
    assert samples.median_query_ms == 11.0


def test_query_samples_single_element():
    """单样本时 first == median。"""
    samples = QuerySamples(measured_runs_ms=[50.0])
    assert samples.first_query_ms == 50.0
    assert samples.median_query_ms == 50.0


def test_query_samples_empty():
    """空样本时返回 None。"""
    samples = QuerySamples(measured_runs_ms=[])
    assert samples.first_query_ms is None
    assert samples.median_query_ms is None


# ---------------------------------------------------------------------------
# Task 4.5 — Lazy-build suspicion detection
# ---------------------------------------------------------------------------


def test_lazy_build_ratio_threshold_triggers():
    """first > 3 * median 应触发 lazy-build 检测。"""
    samples = QuerySamples(measured_runs_ms=[300.0, 99.0, 101.0])
    assert samples.median_query_ms == pytest.approx(100.0, abs=1)
    assert is_lazy_build_suspected(samples) is True


def test_lazy_build_absolute_threshold_triggers():
    """first - median > 1.0 即使倍数不超出也应触发。"""
    samples = QuerySamples(measured_runs_ms=[3.5, 2.0, 2.0])
    assert is_lazy_build_suspected(samples) is True


def test_lazy_build_not_triggered_normal_runs():
    """正常 warmup 曲线不应触发 lazy-build。"""
    samples = QuerySamples(measured_runs_ms=[10.5, 10.0, 10.2])
    assert is_lazy_build_suspected(samples) is False


def test_lazy_build_not_triggered_empty():
    """空样本不应触发（无数据无结论）。"""
    samples = QuerySamples(measured_runs_ms=[])
    assert is_lazy_build_suspected(samples) is False


def test_lazy_build_not_triggered_single():
    """单样本 first == median，不应触发。"""
    samples = QuerySamples(measured_runs_ms=[50.0])
    assert is_lazy_build_suspected(samples) is False


# ---------------------------------------------------------------------------
# Task 5.7 — TSBS ingest loader fields preserved through provider
# ---------------------------------------------------------------------------


def test_ingest_timing_parser_full_output():
    """Ingest ms 优先于 Load+Build 汇总。"""
    text = "Ingest ms: 500.0\nLoad ms: 200.0\nBuild ms: 300.0"
    result = parse_ingest_timing(text, "")
    assert result.ingest_ms == 500.0
    assert result.load_ms == 200.0
    assert result.build_ms == 300.0


def test_ingest_timing_parser_fallback_sum():
    """Ingest ms 缺失时 Load+Build 汇总。"""
    text = "Load ms: 200.0\nBuild ms: 300.0"
    result = parse_ingest_timing(text, "")
    assert result.ingest_ms == pytest.approx(500.0)
    assert result.load_ms == 200.0
    assert result.build_ms == 300.0


def test_ingest_timing_parser_raises_on_missing():
    """没有任何 timing 时显式抛出。"""
    with pytest.raises(ValueError, match="No ingest timing"):
        parse_ingest_timing("no timing here", "")


def test_ingest_timing_from_text_returns_none_on_missing():
    """parse_ingest_timing_from_text（validator path）缺失时返回 None 而非抛出。"""
    ingest_ms, load_ms, build_ms = parse_ingest_timing_from_text("random output")
    assert ingest_ms is None
    assert load_ms is None
    assert build_ms is None


# ---------------------------------------------------------------------------
# Task 6.4 — Ingest completeness gate rejects partial data
# ---------------------------------------------------------------------------


def _full_ingest_args():
    return dict(
        baseline_ingest_ms=5000.0,
        baseline_ingest_rows_per_sec=1000.0,
        baseline_ingest_metrics_per_sec=8000.0,
        baseline_workers=1,
        bespoke_ingest_ms=4000.0,
        bespoke_ingest_rows_per_sec=1200.0,
        bespoke_ingest_metrics_per_sec=9600.0,
    )


def test_ingest_completeness_full_data_passes():
    """完整数据通过 completeness gate。"""
    ok, missing = check_ingest_completeness(**_full_ingest_args())
    assert ok is True
    assert missing == []


def test_ingest_completeness_missing_baseline_rows_per_sec():
    """缺少 baseline_ingest_rows_per_sec 时 gate 拒绝。"""
    args = _full_ingest_args()
    args["baseline_ingest_rows_per_sec"] = None
    ok, missing = check_ingest_completeness(**args)
    assert ok is False
    assert "baseline_ingest_rows_per_sec" in missing


def test_ingest_completeness_missing_bespoke_ingest_rows_per_sec():
    """缺少 generated_tpch_ingest_rows_per_sec 时 gate 拒绝。"""
    args = _full_ingest_args()
    args["bespoke_ingest_rows_per_sec"] = None
    ok, missing = check_ingest_completeness(**args)
    assert ok is False
    assert "generated_tpch_ingest_rows_per_sec" in missing


def test_ingest_completeness_workers_not_1():
    """workers != 1 时 gate 拒绝。"""
    args = _full_ingest_args()
    args["baseline_workers"] = 4
    ok, missing = check_ingest_completeness(**args)
    assert ok is False
    assert any("workers" in m for m in missing)


def test_ingest_completeness_missing_metrics_per_sec():
    """缺少 baseline_ingest_metrics_per_sec 时 gate 拒绝。"""
    args = _full_ingest_args()
    args["baseline_ingest_metrics_per_sec"] = None
    ok, missing = check_ingest_completeness(**args)
    assert ok is False
    assert "baseline_ingest_metrics_per_sec" in missing


def test_derive_bespoke_ingest_metrics_no_longer_uses_legacy_row_count():
    """旧 TSBS ingest 删除后不再从 row-count helper 派生吞吐。"""
    derived, missing = derive_bespoke_ingest_metrics(scale_factor=1, bespoke_ingest_ms=4000.0)
    assert derived is None
    assert missing == ["legacy ingest throughput derivation removed"]


def test_derive_bespoke_ingest_metrics_validates_runtime_before_legacy_message():
    """非法 runtime 仍应在 legacy removal message 之前被拒绝。"""
    derived, missing = derive_bespoke_ingest_metrics(scale_factor=999, bespoke_ingest_ms=0.0)
    assert derived is None
    assert missing == ["generated_tpch_ingest_ms=0.0 (must be > 0)"]
