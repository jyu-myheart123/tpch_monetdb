"""TPC-H result validation against a MonetDB baseline."""

from __future__ import annotations

import csv
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from tpch_monetdb.dataset.gen_tpch.tpch_queries import TpchQueryContract, get_contract
from tpch_monetdb.oracle.result import TpchQueryResult


@dataclass(frozen=True)
class TpchCellMismatch:
    """One value, row, column, or ordering mismatch."""

    row: int | None
    column: str
    expected: Any
    actual: Any
    diff_type: str
    message: str = ""


@dataclass(frozen=True)
class TpchValidationReport:
    """Structured TPC-H validation report."""

    query_id: str
    overall_pass: bool
    column_check_pass: bool
    row_count_check_pass: bool
    value_check_pass: bool
    result_ordered: bool
    sorted_by: tuple[str, ...]
    expected_row_count: int
    actual_row_count: int
    mismatches: list[TpchCellMismatch] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly report dictionary."""
        result = asdict(self)
        result["sorted_by"] = list(self.sorted_by)
        result["mismatches"] = [asdict(m) for m in self.mismatches]
        return result

    def get_summary(self) -> str:
        """Return a compact human-readable validation summary."""
        status = "PASS" if self.overall_pass else "FAIL"
        parts = [f"TPC-H validation {status} for {self.query_id}"]
        
        if self.mismatches:
            first = self.mismatches[0]
            row_label = _format_summary_row(first.row)
            expected_val = _format_summary_value(first.expected)
            actual_val = _format_summary_value(first.actual)
            parts.append(f"first_mismatch={first.diff_type}:{first.column}:{first.message}")
            parts.append(f"expected={expected_val}")
            parts.append(f"actual={actual_val}")
            parts.append(f"at_row={row_label}")
        
        return " | ".join(parts)


class TpchValidator:
    """Compare generated runtime results against MonetDB TPC-H results."""

    def compare_results(
        self,
        expected: TpchQueryResult,
        actual: TpchQueryResult,
        query_id: str | None = None,
    ) -> TpchValidationReport:
        """Compare baseline and runtime result objects using the query contract."""
        return compare_tpch_results(expected, actual)

    def parse_runtime_csv(
        self,
        csv_path: Path,
        query_id: str,
        *,
        source: str = "generated_runtime",
    ) -> TpchQueryResult:
        """Parse a generated runtime CSV file into TpchQueryResult."""
        return parse_runtime_csv(csv_path, query_id, source=source)


def compare_tpch_results(
    expected: TpchQueryResult,
    actual: TpchQueryResult,
    contract: TpchQueryContract | None = None,
) -> TpchValidationReport:
    """Compare two TPC-H result objects with contract-aware policies."""
    if contract is None:
        contract = get_contract(expected.query_id)
    
    mismatches: list[TpchCellMismatch] = []
    diagnostics: dict[str, Any] = {
        "expected_source": expected.source,
        "actual_source": actual.source,
        "comparison_strategy": contract.comparison.strategy,
        "float_atol": contract.float_atol,
        "float_rtol": contract.float_rtol,
    }
    
    column_check_pass = expected.columns == actual.columns
    if not column_check_pass:
        mismatches.append(TpchCellMismatch(
            row=None,
            column="columns",
            expected=expected.columns,
            actual=actual.columns,
            diff_type="columns",
            message="Column mismatch"
        ))
    
    row_count_check_pass = expected.row_count == actual.row_count
    if not row_count_check_pass:
        mismatches.append(TpchCellMismatch(
            row=None,
            column="row_count",
            expected=expected.row_count,
            actual=actual.row_count,
            diff_type="row_count",
            message="Row count mismatch"
        ))
    
    value_check_pass = True
    if column_check_pass and row_count_check_pass:
        if contract.result_ordered:
            value_mismatches = _compare_ordered_rows(expected, actual, contract)
        else:
            value_mismatches = _compare_unordered_rows(expected, actual, contract)
        mismatches.extend(value_mismatches)
        value_check_pass = len(value_mismatches) == 0
    
    overall_pass = column_check_pass and row_count_check_pass and value_check_pass
    
    return _build_report(
        contract=contract,
        expected=expected,
        actual=actual,
        column_check_pass=column_check_pass,
        row_count_check_pass=row_count_check_pass,
        mismatches=mismatches,
        diagnostics=diagnostics,
    )


def parse_runtime_csv(
    csv_path: Path,
    query_id: str,
    *,
    source: str = "generated_runtime",
) -> TpchQueryResult:
    """Parse a runtime CSV file into a TPC-H TpchQueryResult."""
    content = csv_path.read_text(encoding="utf-8")
    if not content.strip():
        raise ValueError("empty CSV file")
    
    reader = csv.reader(content.splitlines())
    rows = list(reader)
    
    if not rows:
        raise ValueError("empty CSV file")
    
    columns = rows[0]
    data_rows = rows[1:]
    
    parsed_rows: list[list[Any]] = []
    for row in data_rows:
        parsed_row = [_parse_csv_cell(cell) for cell in row]
        parsed_rows.append(parsed_row)
    
    column_types = _infer_column_types(parsed_rows, len(columns))
    
    return TpchQueryResult(
        query_id=query_id,
        query_type="tpch",
        columns=columns,
        column_types=column_types,
        rows=parsed_rows,
        row_count=len(parsed_rows),
        source=source,
        source_protocol="csv",
    )


def _build_report(
    *,
    contract: TpchQueryContract,
    expected: TpchQueryResult,
    actual: TpchQueryResult,
    column_check_pass: bool,
    row_count_check_pass: bool,
    mismatches: list[TpchCellMismatch],
    diagnostics: dict[str, Any],
) -> TpchValidationReport:
    """Build a validation report from accumulated comparison evidence."""
    overall_pass = column_check_pass and row_count_check_pass and (len(mismatches) == 0)
    value_check_pass = len(mismatches) == 0
    
    return TpchValidationReport(
        query_id=contract.query_id,
        overall_pass=overall_pass,
        column_check_pass=column_check_pass,
        row_count_check_pass=row_count_check_pass,
        value_check_pass=value_check_pass,
        result_ordered=contract.result_ordered,
        sorted_by=contract.sorted_by,
        expected_row_count=expected.row_count,
        actual_row_count=actual.row_count,
        mismatches=mismatches,
        diagnostics=diagnostics,
    )


def _compare_ordered_rows(
    expected: TpchQueryResult,
    actual: TpchQueryResult,
    contract: TpchQueryContract,
) -> list[TpchCellMismatch]:
    """Compare rows position-by-position for ordered TPC-H results."""
    mismatches: list[TpchCellMismatch] = []
    
    for row_idx, (expected_row, actual_row) in enumerate(zip(expected.rows, actual.rows)):
        row_mismatches = _compare_row_values(
            row_idx=row_idx,
            expected_row=expected_row,
            actual_row=actual_row,
            columns=expected.columns,
            contract=contract,
        )
        
        if row_mismatches and not mismatches:
            mismatches.append(TpchCellMismatch(
                row=row_idx,
                column="ordering",
                expected=expected_row,
                actual=actual_row,
                diff_type="ordering",
                message="Row ordering mismatch"
            ))
        mismatches.extend(row_mismatches)
    
    if len(expected.rows) > len(actual.rows):
        for row_idx in range(len(actual.rows), len(expected.rows)):
            mismatches.append(TpchCellMismatch(
                row=row_idx,
                column="row",
                expected=expected.rows[row_idx],
                actual=None,
                diff_type="missing_row",
                message="Missing row in actual result"
            ))
    elif len(actual.rows) > len(expected.rows):
        for row_idx in range(len(expected.rows), len(actual.rows)):
            mismatches.append(TpchCellMismatch(
                row=row_idx,
                column="row",
                expected=None,
                actual=actual.rows[row_idx],
                diff_type="extra_row",
                message="Extra row in actual result"
            ))
    
    return mismatches


def _compare_unordered_rows(
    expected: TpchQueryResult,
    actual: TpchQueryResult,
    contract: TpchQueryContract,
) -> list[TpchCellMismatch]:
    """Compare rows as a multiset with tolerance-aware greedy matching."""
    mismatches: list[TpchCellMismatch] = []
    
    if len(expected.rows) != len(actual.rows):
        return mismatches
    
    actual_rows_with_idx = [(i, row) for i, row in enumerate(actual.rows)]
    matched_actual_indices = set()
    
    for expected_row_idx, expected_row in enumerate(expected.rows):
        matched_idx = _find_matching_row(
            expected_row,
            [(i, r) for i, r in actual_rows_with_idx if i not in matched_actual_indices],
            expected.columns,
            contract,
        )
        
        if matched_idx is None:
            if len(expected.rows) == 1 and len(actual.rows) == 1:
                mismatches = _compare_row_values(
                    row_idx=0,
                    expected_row=expected.rows[0],
                    actual_row=actual.rows[0],
                    columns=expected.columns,
                    contract=contract,
                )
                return mismatches
            mismatches.append(TpchCellMismatch(
                row=expected_row_idx,
                column="row",
                expected=expected_row,
                actual=None,
                diff_type="missing_row",
                message="Row not found in actual result"
            ))
        else:
            matched_actual_indices.add(matched_idx)
    
    for actual_row_idx, actual_row in enumerate(actual.rows):
        if actual_row_idx not in matched_actual_indices:
            mismatches.append(TpchCellMismatch(
                row=actual_row_idx,
                column="row",
                expected=None,
                actual=actual_row,
                diff_type="extra_row",
                message="Extra row in actual result"
            ))
    
    return mismatches


def _rows_match_unordered(
    expected: TpchQueryResult,
    actual: TpchQueryResult,
    contract: TpchQueryContract,
) -> bool:
    """Return whether two result objects contain the same rows ignoring order."""
    if expected.row_count != actual.row_count:
        return False
    
    mismatches = _compare_unordered_rows(expected, actual, contract)
    return len(mismatches) == 0


def _find_matching_row(
    expected_row: list[Any],
    actual_rows: list[tuple[int, list[Any]]],
    columns: list[str],
    contract: TpchQueryContract,
) -> int | None:
    """Find an actual row that matches one expected row with tolerance."""
    for actual_idx, actual_row in actual_rows:
        if len(expected_row) != len(actual_row):
            continue
        
        all_match = True
        for col_idx, (expected_val, actual_val) in enumerate(zip(expected_row, actual_row)):
            if not _values_equal(expected_val, actual_val, contract):
                all_match = False
                break
        
        if all_match:
            return actual_idx
    
    return None


def _compare_row_values(
    *,
    row_idx: int,
    expected_row: list[Any],
    actual_row: list[Any],
    columns: list[str],
    contract: TpchQueryContract,
) -> list[TpchCellMismatch]:
    """Compare all cells for one row."""
    mismatches: list[TpchCellMismatch] = []
    
    max_len = max(len(expected_row), len(actual_row))
    for col_idx in range(max_len):
        if col_idx >= len(expected_row):
            expected_val = None
        else:
            expected_val = expected_row[col_idx]
        
        if col_idx >= len(actual_row):
            actual_val = None
        else:
            actual_val = actual_row[col_idx]
        
        if col_idx >= len(columns):
            column_name = f"col_{col_idx}"
        else:
            column_name = columns[col_idx]
        
        if not _values_equal(expected_val, actual_val, contract):
            diff_type = _diff_type(expected_val, actual_val)
            mismatches.append(TpchCellMismatch(
                row=row_idx,
                column=column_name,
                expected=expected_val,
                actual=actual_val,
                diff_type=diff_type,
                message="Cell value differs"
            ))
    
    return mismatches


def _values_equal(expected: Any, actual: Any, contract: TpchQueryContract) -> bool:
    """Return whether two scalar values match under TPC-H comparison rules."""
    if expected is None and actual is None:
        return True
    
    if expected is None or actual is None:
        return False
    
    expected_decimal = _to_decimal(expected)
    actual_decimal = _to_decimal(actual)
    
    if expected_decimal is not None and actual_decimal is not None:
        diff = abs(expected_decimal - actual_decimal)
        atol = Decimal(str(contract.float_atol))
        rtol = Decimal(str(contract.float_rtol))
        tolerance = atol + rtol * abs(expected_decimal)
        return diff <= tolerance
    
    return expected == actual


def _format_summary_row(row: int | None) -> str:
    """Return a compact row label for validation summaries."""
    if row is None:
        return "N/A"
    return str(row)


def _format_summary_value(value: Any, max_len: int = 160) -> str:
    """Return a stable compact scalar value for validation summaries."""
    if value is None:
        return "NULL"
    
    if isinstance(value, float):
        return f"{value:.10g}"
    
    s = str(value)
    if len(s) > max_len:
        s = s[:max_len - 3] + "..."
    
    return s


def _to_decimal(value: Any) -> Decimal | None:
    """Convert a scalar to Decimal when it is numeric, otherwise return None."""
    if isinstance(value, Decimal):
        return value
    
    if isinstance(value, float):
        try:
            return Decimal(str(value))
        except InvalidOperation:
            return None
    
    if isinstance(value, (int, str)):
        try:
            return Decimal(value)
        except InvalidOperation:
            return None
    
    return None


def _diff_type(expected: Any, actual: Any) -> str:
    """Classify a mismatch as float, null, or value."""
    if expected is None or actual is None:
        return "null"
    
    if isinstance(expected, float) or isinstance(actual, float):
        return "float"
    
    return "value"


def _normalized_sort_columns(sorted_by: tuple[str, ...]) -> tuple[str, ...]:
    """Normalize ORDER BY expressions to column names for diagnostics."""
    result = []
    for col in sorted_by:
        normalized = col.split()[0]
        result.append(normalized)
    return tuple(result)


def _parse_csv_cell(value: str) -> Any:
    """Parse one CSV cell into a stable scalar representation."""
    value = value.strip()
    
    if not value:
        return None
    
    if value.lower() == 'null':
        return None
    
    try:
        return float(value)
    except ValueError:
        pass
    
    return value


def _infer_column_types(rows: list[list[Any]], column_count: int) -> list[str]:
    """Infer coarse column types from parsed CSV rows."""
    types = []
    for col_idx in range(column_count):
        first_val = _first_non_null(rows, col_idx)
        types.append(_infer_value_type(first_val))
    return types


def _first_non_null(rows: list[list[Any]], column_idx: int) -> Any:
    """Return the first non-null value in one parsed column."""
    for row in rows:
        if column_idx < len(row) and row[column_idx] is not None:
            return row[column_idx]
    return None


def _infer_value_type(value: Any) -> str:
    """Return a coarse comparator type label for one parsed value."""
    if value is None:
        return "UNKNOWN"
    
    if isinstance(value, int):
        return "INTEGER"
    
    if isinstance(value, float):
        return "DOUBLE"
    
    return "STRING"