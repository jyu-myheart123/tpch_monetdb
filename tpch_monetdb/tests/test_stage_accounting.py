"""Tests for phase10 query and ingest optimization summaries.

验证:
- render_bottleneck_report 正确标记慢查询
- baseline_runtime_ms_by_query 存入 OptimizationRunSummary
- ingest comparison 作为 secondary metric 不影响主 gate
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# render_bottleneck_report tests (task 7.5 / 10.1)
# ---------------------------------------------------------------------------

def _make_summary(**overrides):
    from tpch_monetdb.utils.optimization_summary import OptimizationRunSummary

    defaults = dict(
        benchmark="tpch",
        conv_name="test_conv",
        run_id="test_run",
        query_list=["q1", "q13", "q14"],
        is_bespoke_storage=True,
        start_snapshot_hash="abc",
        final_snapshot_hash="def",
        best_runtime_ms_by_query={"q1": 0.19, "q13": 23.1, "q14": 22.2},
        final_runtime_ms_by_query={"q1": 0.19, "q13": 23.1, "q14": 22.2},
        final_correctness=True,
        completed_at="2026-04-20T00:00:00Z",
        conversation_json="",
        session_db_path="",
        success=True,
        baseline_runtime_ms_by_query={"q1": 4.3, "q13": 10.5, "q14": 10.9},
    )
    defaults.update(overrides)
    return OptimizationRunSummary(**defaults)


def test_render_bottleneck_report_marks_slow_queries() -> None:
    """Q13 和 Q14 的 speedup < 1.0× 应被标注为 ⚠ slow."""
    from tpch_monetdb.utils.optimization_summary import render_bottleneck_report

    summary = _make_summary()
    report = render_bottleneck_report(summary)

    assert "⚠ slow" in report
    assert "q13" in report
    assert "q14" in report


def test_render_bottleneck_report_marks_good_queries() -> None:
    """Q1 的 speedup > 1.0× 应被标注为 ✅ good."""
    from tpch_monetdb.utils.optimization_summary import render_bottleneck_report

    summary = _make_summary()
    report = render_bottleneck_report(summary)

    assert "✅ good" in report


def test_render_bottleneck_report_includes_aggregate_speedup() -> None:
    """报告应包含聚合 speedup."""
    from tpch_monetdb.utils.optimization_summary import render_bottleneck_report

    summary = _make_summary()
    report = render_bottleneck_report(summary)

    assert "Aggregate no-CSV kernel speedup" in report


def test_render_bottleneck_report_uses_monetdb_label_for_tpch() -> None:
    """TPC-H bottleneck report 应显示 MonetDB baseline，而不是 QuestDB."""
    from tpch_monetdb.utils.optimization_summary import render_bottleneck_report

    summary = _make_summary(benchmark="tpch")
    report = render_bottleneck_report(summary)

    assert "MonetDB ms" in report
    assert "QuestDB ms" not in report


def test_render_bottleneck_report_no_slow_when_all_fast() -> None:
    """所有查询都快时应输出 'No slow queries detected'."""
    from tpch_monetdb.utils.optimization_summary import render_bottleneck_report

    summary = _make_summary(
        final_runtime_ms_by_query={"q1": 0.1, "q13": 2.0, "q14": 2.0},
        baseline_runtime_ms_by_query={"q1": 4.3, "q13": 10.5, "q14": 10.9},
    )
    report = render_bottleneck_report(summary)

    assert "No slow queries detected" in report


def test_render_validation_kernel_report_uses_validator_metrics() -> None:
    """最终报告应只使用 validator 输出的 no-CSV kernel metrics."""
    from tpch_monetdb.utils.optimization_summary import (
        ValidationKernelReportStatus,
        build_validation_kernel_report,
        render_validation_kernel_report,
    )

    metrics = {
        "validation/correct": True,
        "validation/scale_factor": 1000,
        "validation/query_001/runtime_metric_kind": "kernel_ms",
        "validation/query_001/no_csv_runtime_ms": 0.207,
        "validation/query_001/baseline_runtime_ms": 3.932,
        "validation/query_002/runtime_metric_kind": "kernel_ms",
        "validation/query_002/no_csv_runtime_ms": 0.016,
        "validation/query_002/baseline_runtime_ms": 1.462,
        "validation/total_no_csv_runtime_ms": 0.223,
        "validation/total_baseline_runtime_ms": 5.394,
        "validation/no_csv_total_speedup": 24.188,
    }

    report = build_validation_kernel_report(metrics, ["1", "2"], "round1")
    rendered = render_validation_kernel_report(report)

    assert report.status == ValidationKernelReportStatus.AVAILABLE
    assert report.is_available()
    assert report.scale_factor == 1000
    assert report.rows[0].query_id == "1"
    assert report.rows[0].implementation_ms == 0.207
    assert report.rows[0].baseline_ms == 3.932
    assert round(report.rows[0].speedup, 2) == 19.00
    assert report.total_speedup == 24.188
    assert "round1" in rendered
    assert "Validator baseline ms" in rendered
    assert "Validator QuestDB ms" not in rendered


def test_render_bottleneck_report_accepts_monetdb_measurement_records_for_tpch() -> None:
    """TPC-H exact-instantiation provenance 应要求 monetdb baseline engine."""
    from tpch_monetdb.utils.optimization_summary import render_bottleneck_report

    summary = _make_summary(
        benchmark="tpch",
        query_list=["q1"],
        final_runtime_ms_by_query={"q1": 0.2},
        baseline_runtime_ms_by_query={"q1": 1.0},
        measurement_records=[
            {
                "query_id": "q1",
                "engine": "generated_tpch",
                "measurement_kind": "exact_instantiation",
                "provenance": {"runtime_metric_kind": "kernel_ms"},
                "runtime_ms": 0.2,
            },
            {
                "query_id": "q1",
                "engine": "monetdb",
                "measurement_kind": "exact_instantiation",
                "measurement_shape_status": "known",
                "runtime_ms": 1.0,
            },
        ],
    )

    report = render_bottleneck_report(summary)

    assert "MonetDB ms" in report


def test_render_validation_kernel_report_refuses_missing_validator_metrics() -> None:
    """缺少 validator 字段时不应回退到 diagnostic summary 表格."""
    from tpch_monetdb.utils.optimization_summary import (
        ValidationKernelReportStatus,
        build_validation_kernel_report,
        render_validation_kernel_report,
    )

    report = build_validation_kernel_report(
        {
            "validation/correct": True,
            "validation/query_001/runtime_metric_kind": "query_e2e_ms",
        },
        ["1"],
        "round1",
    )

    assert report.status == ValidationKernelReportStatus.INCOMPLETE_METRICS
    assert not report.is_available()
    assert report.missing_metrics == ("Q1:runtime_metric_kind=query_e2e_ms",)
    assert render_validation_kernel_report(report)


def test_render_bottleneck_report_rejects_mixed_measurement_kind() -> None:
    """Speedup report must not mix fixed-validation and exact-instantiation data."""
    from tpch_monetdb.utils.optimization_summary import render_bottleneck_report

    summary = _make_summary(
        query_list=["q1"],
        final_runtime_ms_by_query={"q1": 0.2},
        baseline_runtime_ms_by_query={"q1": 1.0},
        measurement_records=[
            {
                "query_id": "q1",
                "engine": "generated_tpch",
                "measurement_kind": "exact_instantiation",
                "provenance": {"runtime_metric_kind": "kernel_ms"},
                "runtime_ms": 0.2,
            },
            {
                "query_id": "q1",
                "engine": "questdb",
                "measurement_kind": "fixed_validation",
                "runtime_ms": 1.0,
            },
        ],
    )

    try:
        render_bottleneck_report(summary)
    except ValueError as exc:
        assert "measurement provenance is inconsistent" in str(exc)
        return None
    raise AssertionError("render_bottleneck_report should reject mixed provenance")


def test_render_bottleneck_report_marks_unknown_measurement_shape() -> None:
    """Exact speedup report must label unknown row/output shape metadata."""
    from tpch_monetdb.utils.optimization_summary import render_bottleneck_report

    summary = _make_summary(
        query_list=["q1"],
        final_runtime_ms_by_query={"q1": 0.2},
        baseline_runtime_ms_by_query={"q1": 1.0},
        measurement_records=[
            {
                "query_id": "q1",
                "engine": "generated_tpch",
                "measurement_kind": "exact_instantiation",
                "measurement_shape_status": "unknown",
                "provenance": {"runtime_metric_kind": "kernel_ms"},
                "runtime_ms": 0.2,
            },
            {
                "query_id": "q1",
                "engine": "monetdb",
                "measurement_kind": "exact_instantiation",
                "measurement_shape_status": "unknown",
                "runtime_ms": 1.0,
            },
        ],
    )

    report = render_bottleneck_report(summary)

    assert "Measurement shape status" in report
    assert "Measurement shape warning" in report
    assert "heavy-load conclusions" in report
    return None


def test_optimization_run_summary_has_baseline_field() -> None:
    """OptimizationRunSummary 应有 baseline_runtime_ms_by_query 字段."""
    summary = _make_summary()
    assert hasattr(summary, "baseline_runtime_ms_by_query")
    assert summary.baseline_runtime_ms_by_query == {"q1": 4.3, "q13": 10.5, "q14": 10.9}


def test_optimization_run_summary_baseline_defaults_to_empty() -> None:
    """未提供 baseline 时应默认为空 dict."""
    from tpch_monetdb.utils.optimization_summary import OptimizationRunSummary
    summary = OptimizationRunSummary(
        benchmark="tpch",
        conv_name="x",
        run_id="x",
        query_list=["q1"],
        is_bespoke_storage=False,
        start_snapshot_hash="a",
        final_snapshot_hash="b",
        best_runtime_ms_by_query={"q1": 1.0},
        final_runtime_ms_by_query={"q1": 1.0},
        final_correctness=True,
        completed_at="2026-04-20",
        conversation_json="",
        session_db_path="",
        success=True,
    )
    assert summary.baseline_runtime_ms_by_query == {}


# ---------------------------------------------------------------------------
# persist_successful_optimization_run signature test (task 7.3)
# ---------------------------------------------------------------------------

def test_persist_successful_optimization_run_accepts_baseline_kwarg() -> None:
    """persist_successful_optimization_run 应接受 baseline_runtime_ms_by_query 参数."""
    import inspect
    from tpch_monetdb.utils.optimization_summary import persist_successful_optimization_run

    sig = inspect.signature(persist_successful_optimization_run)
    assert "baseline_runtime_ms_by_query" in sig.parameters
    assert "final_validation_metrics" in sig.parameters
