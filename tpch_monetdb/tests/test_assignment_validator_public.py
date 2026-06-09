from __future__ import annotations

import json
from pathlib import Path

import pytest

from tpch_monetdb.oracle.result import TpchQueryResult
from tpch_monetdb.oracle.tpch_validator import (
    TpchCellMismatch,
    TpchValidationReport,
    TpchValidator,
    compare_tpch_results,
    parse_runtime_csv,
)


def _result(query_id: str, columns: list[str], rows: list[list[object]], *, source: str = "test") -> TpchQueryResult:
    """Build a compact TpchQueryResult for validator public tests."""
    return TpchQueryResult(
        query_id=query_id,
        query_type="tpch",
        columns=columns,
        column_types=["STRING" for _ in columns],
        rows=rows,
        row_count=len(rows),
        source=source,
    )


def test_tpch_query_result_initializes_created_at_and_row_count() -> None:
    """TpchQueryResult should fill created_at and row_count when omitted."""
    result = TpchQueryResult(query_id="Q6", query_type="tpch", columns=["revenue"], rows=[[100.0]])

    assert result.created_at.endswith("Z")
    assert result.row_count == 1
    return None


def test_tpch_query_result_to_dict_converts_sorted_by_to_list() -> None:
    """TpchQueryResult.to_dict should be JSON-friendly."""
    result = TpchQueryResult(query_id="Q1", sorted_by=("l_returnflag", "l_linestatus"))

    payload = result.to_dict()

    assert payload["sorted_by"] == ["l_returnflag", "l_linestatus"]
    return None


def test_tpch_query_result_from_dict_restores_sorted_by_tuple() -> None:
    """TpchQueryResult.from_dict should restore tuple metadata."""
    result = TpchQueryResult.from_dict({"query_id": "Q1", "sorted_by": ["a", "b"]})

    assert result.sorted_by == ("a", "b")
    return None


def test_tpch_query_result_json_round_trip() -> None:
    """TpchQueryResult should round-trip through JSON text."""
    original = TpchQueryResult(query_id="Q6", columns=["revenue"], rows=[[100.0]], sorted_by=("revenue",))

    restored = TpchQueryResult.from_json(original.to_json())

    assert restored.query_id == "Q6"
    assert restored.columns == ["revenue"]
    assert restored.rows == [[100.0]]
    assert restored.sorted_by == ("revenue",)
    return None


def test_tpch_query_result_summary_is_compact() -> None:
    """TpchQueryResult.get_summary should expose stable logging fields."""
    result = TpchQueryResult(query_id="Q6", query_type="tpch", columns=["revenue"], rows=[[1]], source="monetdb")

    summary = result.get_summary()

    assert summary["query_id"] == "Q6"
    assert summary["columns"] == ["revenue"]
    assert summary["row_count"] == 1
    assert summary["source"] == "monetdb"
    return None


def test_validation_report_to_dict_serializes_mismatches() -> None:
    """TpchValidationReport.to_dict should serialize tuple and dataclass fields."""
    report = TpchValidationReport(
        query_id="Q6",
        overall_pass=False,
        column_check_pass=True,
        row_count_check_pass=True,
        value_check_pass=False,
        result_ordered=False,
        sorted_by=("revenue",),
        expected_row_count=1,
        actual_row_count=1,
        mismatches=[
            TpchCellMismatch(
                row=0,
                column="revenue",
                expected=100.0,
                actual=101.0,
                diff_type="float",
                message="Cell value differs",
            )
        ],
    )

    payload = report.to_dict()

    assert payload["sorted_by"] == ["revenue"]
    assert payload["mismatches"][0]["column"] == "revenue"
    return None


def test_validation_report_summary_mentions_first_mismatch() -> None:
    """TpchValidationReport.get_summary should expose first mismatch details."""
    report = TpchValidationReport(
        query_id="Q6",
        overall_pass=False,
        column_check_pass=True,
        row_count_check_pass=True,
        value_check_pass=False,
        result_ordered=False,
        sorted_by=(),
        expected_row_count=1,
        actual_row_count=1,
        mismatches=[
            TpchCellMismatch(0, "revenue", None, 0.0, "null", "Cell value differs")
        ],
    )

    summary = report.get_summary()

    assert "TPC-H validation FAIL for Q6" in summary
    assert "first_mismatch=null:revenue:Cell value differs" in summary
    assert "expected=NULL" in summary
    return None


def test_parse_runtime_csv_reads_columns_rows_and_types(tmp_path: Path) -> None:
    """parse_runtime_csv should parse headers, values, and inferred column types."""
    csv_path = tmp_path / "q6.csv"
    csv_path.write_text("revenue,label,empty\n100.25,ok,\n", encoding="utf-8")

    result = parse_runtime_csv(csv_path, "Q6")

    assert result.query_id == "Q6"
    assert result.columns == ["revenue", "label", "empty"]
    assert result.rows == [[100.25, "ok", None]]
    assert result.column_types == ["DOUBLE", "STRING", "UNKNOWN"]
    assert result.source_protocol == "csv"
    return None


def test_parse_runtime_csv_empty_file_raises(tmp_path: Path) -> None:
    """parse_runtime_csv should reject empty CSV files."""
    csv_path = tmp_path / "empty.csv"
    csv_path.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="empty"):
        parse_runtime_csv(csv_path, "Q6")
    return None


def test_validator_facade_parses_and_compares(tmp_path: Path) -> None:
    """TpchValidator should expose parse and compare facade methods."""
    csv_path = tmp_path / "q6.csv"
    csv_path.write_text("revenue\n100\n", encoding="utf-8")
    validator = TpchValidator()

    actual = validator.parse_runtime_csv(csv_path, "Q6")
    expected = _result("Q6", ["revenue"], [[100.0]], source="monetdb")
    report = validator.compare_results(expected, actual)

    assert report.overall_pass is True
    assert "TPC-H validation PASS" in report.get_summary()
    return None


def test_compare_results_passes_identical_rows() -> None:
    """Identical result objects should validate successfully."""
    expected = _result("Q6", ["revenue"], [[100.0]], source="monetdb")
    actual = _result("Q6", ["revenue"], [[100.0]], source="generated_runtime")

    report = compare_tpch_results(expected, actual)

    assert report.overall_pass is True
    assert report.column_check_pass is True
    assert report.row_count_check_pass is True
    assert report.value_check_pass is True
    return None


def test_compare_results_fails_when_column_order_differs() -> None:
    """Column order should be part of the result contract."""
    expected = _result("Q6", ["a", "b"], [[1, 2]])
    actual = _result("Q6", ["b", "a"], [[2, 1]])

    report = compare_tpch_results(expected, actual)

    assert report.overall_pass is False
    assert report.column_check_pass is False
    assert report.mismatches[0].diff_type == "columns"
    return None


def test_compare_results_fails_when_row_count_differs() -> None:
    """Row count mismatches should produce structured diagnostics."""
    expected = _result("Q6", ["revenue"], [[1.0], [2.0]])
    actual = _result("Q6", ["revenue"], [[1.0]])

    report = compare_tpch_results(expected, actual)

    assert report.overall_pass is False
    assert report.row_count_check_pass is False
    assert report.mismatches[0].diff_type == "row_count"
    return None


def test_ordered_query_fails_for_permuted_rows() -> None:
    """Ordered TPC-H contracts should reject row permutations."""
    columns = ["l_returnflag", "l_linestatus", "sum_qty"]
    expected = _result("Q1", columns, [["A", "F", 1.0], ["R", "F", 2.0]])
    actual = _result("Q1", columns, [["R", "F", 2.0], ["A", "F", 1.0]])

    report = compare_tpch_results(expected, actual)

    assert report.overall_pass is False
    assert report.result_ordered is True
    assert report.mismatches[0].diff_type == "ordering"
    return None


def test_unordered_query_allows_permuted_rows() -> None:
    """Unordered TPC-H contracts should compare rows as a multiset."""
    expected = _result("Q6", ["revenue"], [[1.0], [2.0]])
    actual = _result("Q6", ["revenue"], [[2.0], [1.0]])

    report = compare_tpch_results(expected, actual)

    assert report.overall_pass is True
    return None


def test_unordered_query_preserves_duplicate_row_counts() -> None:
    """Multiset comparison should detect duplicate count differences."""
    expected = _result("Q6", ["revenue"], [[1.0], [1.0], [2.0]])
    actual = _result("Q6", ["revenue"], [[1.0], [2.0], [2.0]])

    report = compare_tpch_results(expected, actual)

    assert report.overall_pass is False
    assert report.mismatches[0].diff_type in {"missing_row", "extra_row"}
    return None


def test_float_tolerance_accepts_small_drift() -> None:
    """Float values within tolerance should pass validation."""
    expected = _result("Q6", ["revenue"], [[100.0]])
    actual = _result("Q6", ["revenue"], [[100.005]])

    report = compare_tpch_results(expected, actual)

    assert report.overall_pass is True
    return None


def test_float_tolerance_rejects_large_drift() -> None:
    """Float values outside tolerance should produce a float mismatch."""
    expected = _result("Q6", ["revenue"], [[100.0]])
    actual = _result("Q6", ["revenue"], [[102.0]])

    report = compare_tpch_results(expected, actual)

    assert report.overall_pass is False
    assert report.mismatches[0].diff_type == "float"
    assert report.mismatches[0].column == "revenue"
    return None


def test_null_and_zero_are_not_equal() -> None:
    """SQL NULL and numeric zero should not compare equal."""
    expected = _result("Q6", ["revenue"], [[None]])
    actual = _result("Q6", ["revenue"], [[0.0]])

    report = compare_tpch_results(expected, actual)

    assert report.overall_pass is False
    assert report.mismatches[0].diff_type == "null"
    return None


def test_compare_results_records_contract_diagnostics() -> None:
    """Validation diagnostics should include contract strategy and tolerance metadata."""
    expected = _result("Q6", ["revenue"], [[100.0]])
    actual = _result("Q6", ["revenue"], [[100.0]])

    report = compare_tpch_results(expected, actual)

    assert report.diagnostics["expected_source"] == "test"
    assert report.diagnostics["actual_source"] == "test"
    assert "comparison_strategy" in report.diagnostics
    assert report.diagnostics["float_atol"] == pytest.approx(1e-2)
    assert report.diagnostics["float_rtol"] == pytest.approx(1e-2)
    return None
