from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pytest

from tpch_monetdb.benchmark.runtime_accounting import MEASURED_RUNS, WARMUP_RUNS
from tpch_monetdb.dataset.query_gen_factory import get_placeholders_fn
from tpch_monetdb.oracle.result import TpchQueryResult
from tpch_monetdb.oracle.tpch_runtime_validator import TpchRuntimeValidator
from tpch_monetdb.oracle.tpch_validator import (
    TpchValidator,
    compare_tpch_results,
    parse_runtime_csv,
)
from tpch_monetdb.oracle.validate_cache import CacheMissError
from tpch_monetdb.tools.tpch.run import RunTool
from tpch_monetdb.tools.tpch.utils import copy_template_to
from tpch_monetdb.utils.general_utils import gen_tpch_args_str


REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_DIR = REPO_ROOT / "docker" / "tpch-monetdb" / "fixtures" / "tiny-tpch"


def _result(
    *,
    query_id: str,
    columns: list[str],
    rows: list[list[object]],
    source: str,
) -> TpchQueryResult:
    """Build a compact TPC-H result object for validator tests."""
    return TpchQueryResult(
        query_id=query_id,
        query_type="tpch",
        columns=columns,
        column_types=["STRING" for _ in columns],
        rows=rows,
        row_count=len(rows),
        source=source,
    )


def test_ordered_result_with_permuted_rows_fails_and_reports_sort_columns() -> None:
    """Verify ordered TPC-H results fail when runtime rows are merely permuted."""
    columns = ["l_returnflag", "l_linestatus", "sum_qty"]
    expected = _result(
        query_id="Q1",
        columns=columns,
        rows=[["A", "F", 1.0], ["R", "F", 2.0]],
        source="monetdb",
    )
    actual = _result(
        query_id="Q1",
        columns=columns,
        rows=[["R", "F", 2.0], ["A", "F", 1.0]],
        source="generated_runtime",
    )

    report = compare_tpch_results(expected, actual)

    assert report.overall_pass is False
    assert report.result_ordered is True
    assert report.diagnostics["ordering_violation"] is True
    assert report.mismatches[0].diff_type == "ordering"
    assert "l_returnflag" in report.mismatches[0].column
    return None


def test_float_tolerance_passes_small_error_and_fails_large_error() -> None:
    """Verify decimal/float tolerance accepts small drift and reports large drift."""
    expected = _result(
        query_id="Q6",
        columns=["revenue"],
        rows=[[100.0]],
        source="monetdb",
    )
    close_actual = _result(
        query_id="Q6",
        columns=["revenue"],
        rows=[[100.005]],
        source="generated_runtime",
    )
    far_actual = _result(
        query_id="Q6",
        columns=["revenue"],
        rows=[[102.0]],
        source="generated_runtime",
    )

    close_report = compare_tpch_results(expected, close_actual)
    far_report = compare_tpch_results(expected, far_actual)

    assert close_report.overall_pass is True
    assert far_report.overall_pass is False
    assert far_report.mismatches[0].diff_type == "float"
    assert far_report.mismatches[0].column == "revenue"
    return None


def test_null_mismatch_summary_includes_expected_and_actual_values() -> None:
    """Verify NULL mismatches expose values in the model-visible summary."""
    expected_null = _result(
        query_id="Q6",
        columns=["revenue"],
        rows=[[None]],
        source="monetdb",
    )
    actual_zero = _result(
        query_id="Q6",
        columns=["revenue"],
        rows=[[0.0]],
        source="generated_runtime",
    )
    expected_value = _result(
        query_id="Q6",
        columns=["revenue"],
        rows=[[5.0]],
        source="monetdb",
    )
    actual_null = _result(
        query_id="Q6",
        columns=["revenue"],
        rows=[[None]],
        source="generated_runtime",
    )

    null_expected_report = compare_tpch_results(expected_null, actual_zero)
    null_actual_report = compare_tpch_results(expected_value, actual_null)

    assert null_expected_report.overall_pass is False
    assert "first_mismatch=null:revenue:Cell value differs" in null_expected_report.get_summary()
    assert "row=0" in null_expected_report.get_summary()
    assert "expected=NULL" in null_expected_report.get_summary()
    assert "actual=0.000000" in null_expected_report.get_summary()
    assert null_actual_report.overall_pass is False
    assert "expected=5.000000" in null_actual_report.get_summary()
    assert "actual=NULL" in null_actual_report.get_summary()
    return None


def test_column_and_row_count_failures_are_structured() -> None:
    """Verify column and row-count failures produce explicit diagnostics."""
    expected = _result(
        query_id="Q6",
        columns=["revenue"],
        rows=[[100.0], [101.0]],
        source="monetdb",
    )
    wrong_columns = _result(
        query_id="Q6",
        columns=["wrong"],
        rows=[[100.0], [101.0]],
        source="generated_runtime",
    )
    wrong_rows = _result(
        query_id="Q6",
        columns=["revenue"],
        rows=[[100.0]],
        source="generated_runtime",
    )

    column_report = compare_tpch_results(expected, wrong_columns)
    row_report = compare_tpch_results(expected, wrong_rows)

    assert column_report.column_check_pass is False
    assert column_report.mismatches[0].diff_type == "columns"
    assert row_report.row_count_check_pass is False
    assert row_report.mismatches[0].diff_type == "row_count"
    return None


def test_parse_runtime_csv_creates_tpch_result(tmp_path: Path) -> None:
    """Verify generated runtime CSV is parsed into TpchQueryResult."""
    csv_path = tmp_path / "result_q6.csv"
    csv_path.write_text("revenue\n100.25\n", encoding="utf-8")

    result = parse_runtime_csv(csv_path, "Q6")

    assert result.query_id == "Q6"
    assert result.query_type == "tpch"
    assert result.columns == ["revenue"]
    assert result.column_types == ["DOUBLE"]
    assert result.rows == [[100.25]]
    assert result.source == "generated_runtime"
    assert result.source_protocol == "csv"
    return None


def test_parse_runtime_csv_empty_cell_represents_sql_null(tmp_path: Path) -> None:
    """Verify an empty runtime CSV cell is parsed as SQL NULL."""
    csv_path = tmp_path / "result_q6_null.csv"
    csv_path.write_text("revenue\n\"\"\n", encoding="utf-8")

    result = parse_runtime_csv(csv_path, "Q6")

    assert result.rows == [[None]]
    return None


def test_tpch_validator_facade_compares_and_parses_csv(tmp_path: Path) -> None:
    """Verify TpchValidator facade exposes compare and CSV parse operations."""
    validator = TpchValidator()
    csv_path = tmp_path / "result.csv"
    csv_path.write_text("revenue\n100\n", encoding="utf-8")

    parsed = validator.parse_runtime_csv(csv_path, "Q6")
    expected = _result(
        query_id="Q6",
        columns=["revenue"],
        rows=[[100.0]],
        source="monetdb",
    )
    report = validator.compare_results(expected, parsed)

    assert parsed.rows == [[100]]
    assert report.overall_pass is True
    assert "TPC-H validation PASS" in report.get_summary()
    return None


class _FakeMonetDBOracle:
    """Small fake oracle that returns deterministic TPC-H baseline results."""

    def __init__(self) -> None:
        """Initialize call capture state."""
        self.calls: list[dict[str, object]] = []
        return None

    def execute_sql(
        self,
        sql: str,
        *,
        query_id: str,
        query_type: str = "tpch",
        params: dict[str, object] | None = None,
        sorted_by: tuple[str, ...] = (),
    ) -> TpchQueryResult:
        """Record the exact SQL call and return a matching tiny TPC-H baseline."""
        self.calls.append(
            {
                "sql": sql,
                "query_id": query_id,
                "query_type": query_type,
                "params": {} if params is None else dict(params),
                "sorted_by": sorted_by,
            }
        )
        if query_id == "Q1":
            q1_rows = _compute_tiny_q1_rows(params)
            return TpchQueryResult(
                query_id=query_id,
                query_type=query_type,
                params={} if params is None else dict(params),
                sql=sql,
                columns=[
                    "l_returnflag",
                    "l_linestatus",
                    "sum_qty",
                    "sum_base_price",
                    "sum_disc_price",
                    "sum_charge",
                    "avg_qty",
                    "avg_price",
                    "avg_disc",
                    "count_order",
                ],
                column_types=[
                    "STRING",
                    "STRING",
                    "DOUBLE",
                    "DOUBLE",
                    "DOUBLE",
                    "DOUBLE",
                    "DOUBLE",
                    "DOUBLE",
                    "DOUBLE",
                    "INTEGER",
                ],
                rows=q1_rows,
                row_count=len(q1_rows),
                sorted_by=sorted_by,
                source="monetdb",
                source_protocol="native-mapi",
            )
        if query_id == "Q3":
            q3_rows = _compute_tiny_q3_rows(params)
            return TpchQueryResult(
                query_id=query_id,
                query_type=query_type,
                params={} if params is None else dict(params),
                sql=sql,
                columns=["l_orderkey", "revenue", "o_orderdate", "o_shippriority"],
                column_types=["INTEGER", "DOUBLE", "STRING", "INTEGER"],
                rows=q3_rows,
                row_count=len(q3_rows),
                sorted_by=sorted_by,
                source="monetdb",
                source_protocol="native-mapi",
            )
        if query_id == "Q2":
            q2_rows = _compute_q2_min_cost_rows(params)
            return TpchQueryResult(
                query_id=query_id,
                query_type=query_type,
                params={} if params is None else dict(params),
                sql=sql,
                columns=[
                    "s_acctbal",
                    "s_name",
                    "n_name",
                    "p_partkey",
                    "p_mfgr",
                    "s_address",
                    "s_phone",
                    "s_comment",
                ],
                column_types=[
                    "DOUBLE",
                    "STRING",
                    "STRING",
                    "INTEGER",
                    "STRING",
                    "STRING",
                    "STRING",
                    "STRING",
                ],
                rows=q2_rows,
                row_count=len(q2_rows),
                sorted_by=sorted_by,
                source="monetdb",
                source_protocol="native-mapi",
            )
        if query_id == "Q4":
            q4_rows = _compute_tiny_q4_rows(params)
            return TpchQueryResult(
                query_id=query_id,
                query_type=query_type,
                params={} if params is None else dict(params),
                sql=sql,
                columns=["o_orderpriority", "order_count"],
                column_types=["STRING", "INTEGER"],
                rows=q4_rows,
                row_count=len(q4_rows),
                sorted_by=sorted_by,
                source="monetdb",
                source_protocol="native-mapi",
            )
        if query_id == "Q5":
            q5_rows = _compute_tiny_q5_rows(params)
            return TpchQueryResult(
                query_id=query_id,
                query_type=query_type,
                params={} if params is None else dict(params),
                sql=sql,
                columns=["n_name", "revenue"],
                column_types=["STRING", "DOUBLE"],
                rows=q5_rows,
                row_count=len(q5_rows),
                sorted_by=sorted_by,
                source="monetdb",
                source_protocol="native-mapi",
            )
        if query_id == "Q7":
            q7_rows = _compute_q7_shipping_rows(params)
            return TpchQueryResult(
                query_id=query_id,
                query_type=query_type,
                params={} if params is None else dict(params),
                sql=sql,
                columns=["supp_nation", "cust_nation", "l_year", "revenue"],
                column_types=["STRING", "STRING", "INTEGER", "DOUBLE"],
                rows=q7_rows,
                row_count=len(q7_rows),
                sorted_by=sorted_by,
                source="monetdb",
                source_protocol="native-mapi",
            )
        if query_id == "Q8":
            q8_rows = _compute_q8_market_share_rows(params)
            return TpchQueryResult(
                query_id=query_id,
                query_type=query_type,
                params={} if params is None else dict(params),
                sql=sql,
                columns=["o_year", "mkt_share"],
                column_types=["INTEGER", "DOUBLE"],
                rows=q8_rows,
                row_count=len(q8_rows),
                sorted_by=sorted_by,
                source="monetdb",
                source_protocol="native-mapi",
            )
        if query_id == "Q9":
            q9_rows = _compute_q9_profit_rows(params)
            return TpchQueryResult(
                query_id=query_id,
                query_type=query_type,
                params={} if params is None else dict(params),
                sql=sql,
                columns=["nation", "o_year", "sum_profit"],
                column_types=["STRING", "INTEGER", "DOUBLE"],
                rows=q9_rows,
                row_count=len(q9_rows),
                sorted_by=sorted_by,
                source="monetdb",
                source_protocol="native-mapi",
            )
        if query_id == "Q10":
            q10_rows = _compute_tiny_q10_rows(params)
            return TpchQueryResult(
                query_id=query_id,
                query_type=query_type,
                params={} if params is None else dict(params),
                sql=sql,
                columns=[
                    "c_custkey",
                    "c_name",
                    "revenue",
                    "c_acctbal",
                    "n_name",
                    "c_address",
                    "c_phone",
                    "c_comment",
                ],
                column_types=[
                    "INTEGER",
                    "STRING",
                    "DOUBLE",
                    "DOUBLE",
                    "STRING",
                    "STRING",
                    "STRING",
                    "STRING",
                ],
                rows=q10_rows,
                row_count=len(q10_rows),
                sorted_by=sorted_by,
                source="monetdb",
                source_protocol="native-mapi",
            )
        if query_id == "Q11":
            q11_rows = _compute_tiny_q11_rows(params)
            return TpchQueryResult(
                query_id=query_id,
                query_type=query_type,
                params={} if params is None else dict(params),
                sql=sql,
                columns=["ps_partkey", "value"],
                column_types=["INTEGER", "DOUBLE"],
                rows=q11_rows,
                row_count=len(q11_rows),
                sorted_by=sorted_by,
                source="monetdb",
                source_protocol="native-mapi",
            )
        if query_id == "Q12":
            q12_rows = _compute_tiny_q12_rows(params)
            return TpchQueryResult(
                query_id=query_id,
                query_type=query_type,
                params={} if params is None else dict(params),
                sql=sql,
                columns=["l_shipmode", "high_line_count", "low_line_count"],
                column_types=["STRING", "INTEGER", "INTEGER"],
                rows=q12_rows,
                row_count=len(q12_rows),
                sorted_by=sorted_by,
                source="monetdb",
                source_protocol="native-mapi",
            )
        if query_id == "Q13":
            q13_rows = _compute_tiny_q13_rows(params)
            return TpchQueryResult(
                query_id=query_id,
                query_type=query_type,
                params={} if params is None else dict(params),
                sql=sql,
                columns=["c_count", "custdist"],
                column_types=["INTEGER", "INTEGER"],
                rows=q13_rows,
                row_count=len(q13_rows),
                sorted_by=sorted_by,
                source="monetdb",
                source_protocol="native-mapi",
            )
        if query_id == "Q14":
            return TpchQueryResult(
                query_id=query_id,
                query_type=query_type,
                params={} if params is None else dict(params),
                sql=sql,
                columns=["promo_revenue"],
                column_types=["DOUBLE"],
                rows=[[_compute_tiny_q14_promo_revenue(params)]],
                row_count=1,
                sorted_by=sorted_by,
                source="monetdb",
                source_protocol="native-mapi",
            )
        if query_id == "Q15":
            q15_rows = _compute_tiny_q15_rows(params)
            return TpchQueryResult(
                query_id=query_id,
                query_type=query_type,
                params={} if params is None else dict(params),
                sql=sql,
                columns=[
                    "s_suppkey",
                    "s_name",
                    "s_address",
                    "s_phone",
                    "total_revenue",
                ],
                column_types=["INTEGER", "STRING", "STRING", "STRING", "DOUBLE"],
                rows=q15_rows,
                row_count=len(q15_rows),
                sorted_by=sorted_by,
                source="monetdb",
                source_protocol="native-mapi",
            )
        if query_id == "Q16":
            q16_rows = _compute_tiny_q16_rows(params)
            return TpchQueryResult(
                query_id=query_id,
                query_type=query_type,
                params={} if params is None else dict(params),
                sql=sql,
                columns=["p_brand", "p_type", "p_size", "supplier_cnt"],
                column_types=["STRING", "STRING", "INTEGER", "INTEGER"],
                rows=q16_rows,
                row_count=len(q16_rows),
                sorted_by=sorted_by,
                source="monetdb",
                source_protocol="native-mapi",
            )
        if query_id == "Q17":
            return TpchQueryResult(
                query_id=query_id,
                query_type=query_type,
                params={} if params is None else dict(params),
                sql=sql,
                columns=["avg_yearly"],
                column_types=["DOUBLE"],
                rows=[[_compute_q17_two_line_avg_yearly(params)]],
                row_count=1,
                sorted_by=sorted_by,
                source="monetdb",
                source_protocol="native-mapi",
            )
        if query_id == "Q18":
            q18_rows = _compute_q18_large_order_rows(params)
            return TpchQueryResult(
                query_id=query_id,
                query_type=query_type,
                params={} if params is None else dict(params),
                sql=sql,
                columns=[
                    "c_name",
                    "c_custkey",
                    "o_orderkey",
                    "o_orderdate",
                    "o_totalprice",
                    "sum_l_quantity",
                ],
                column_types=[
                    "STRING",
                    "INTEGER",
                    "INTEGER",
                    "STRING",
                    "DOUBLE",
                    "DOUBLE",
                ],
                rows=q18_rows,
                row_count=len(q18_rows),
                sorted_by=sorted_by,
                source="monetdb",
                source_protocol="native-mapi",
            )
        if query_id == "Q19":
            return TpchQueryResult(
                query_id=query_id,
                query_type=query_type,
                params={} if params is None else dict(params),
                sql=sql,
                columns=["revenue"],
                column_types=["DOUBLE"],
                rows=[[_compute_q19_branch_revenue(params)]],
                row_count=1,
                sorted_by=sorted_by,
                source="monetdb",
                source_protocol="native-mapi",
            )
        if query_id == "Q20":
            q20_rows = _compute_q20_promotion_rows(params)
            return TpchQueryResult(
                query_id=query_id,
                query_type=query_type,
                params={} if params is None else dict(params),
                sql=sql,
                columns=["s_name", "s_address"],
                column_types=["STRING", "STRING"],
                rows=q20_rows,
                row_count=len(q20_rows),
                sorted_by=sorted_by,
                source="monetdb",
                source_protocol="native-mapi",
            )
        if query_id == "Q21":
            q21_rows = _compute_q21_wait_rows(params)
            return TpchQueryResult(
                query_id=query_id,
                query_type=query_type,
                params={} if params is None else dict(params),
                sql=sql,
                columns=["s_name", "numwait"],
                column_types=["STRING", "INTEGER"],
                rows=q21_rows,
                row_count=len(q21_rows),
                sorted_by=sorted_by,
                source="monetdb",
                source_protocol="native-mapi",
            )
        if query_id == "Q22":
            q22_rows = _compute_q22_phone_prefix_rows(params)
            return TpchQueryResult(
                query_id=query_id,
                query_type=query_type,
                params={} if params is None else dict(params),
                sql=sql,
                columns=["cntrycode", "numcust", "totacctbal"],
                column_types=["STRING", "INTEGER", "DOUBLE"],
                rows=q22_rows,
                row_count=len(q22_rows),
                sorted_by=sorted_by,
                source="monetdb",
                source_protocol="native-mapi",
            )
        rows = (
            [[_compute_tiny_q6_revenue(params)]]
            if query_id == "Q6"
            else [[5.0]]
        )
        return TpchQueryResult(
            query_id=query_id,
            query_type=query_type,
            params={} if params is None else dict(params),
            sql=sql,
            columns=["revenue"],
            column_types=["DOUBLE"],
            rows=rows,
            row_count=1,
            sorted_by=sorted_by,
            source="monetdb",
            source_protocol="native-mapi",
        )

    def execute_sql_benchmark(
        self,
        sql: str,
        *,
        query_id: str,
        query_type: str = "tpch",
        params: dict[str, object] | None = None,
        sorted_by: tuple[str, ...] = (),
        num_runs: int = 3,
    ) -> tuple[TpchQueryResult, float]:
        """Record a benchmark baseline call and return deterministic timing."""
        del num_runs
        result = self.execute_sql(
            sql,
            query_id=query_id,
            query_type=query_type,
            params=params,
            sorted_by=sorted_by,
        )
        return result, 40.0 if query_id == "Q6" else 20.0


def _compute_tiny_q1_rows(params: dict[str, object] | None) -> list[list[object]]:
    """Compute the MonetDB-equivalent Q1 rows for the one-row tiny fixture."""
    if params is None:
        return []
    shipdate = date.fromisoformat("1995-03-15")
    cutoff_date = date.fromisoformat("1998-12-01") - timedelta(days=int(params["DELTA"]))
    if shipdate > cutoff_date:
        return []
    quantity = 10.0
    extended_price = 100.0
    discount = 0.05
    tax = 0.07
    discounted_price = extended_price * (1.0 - discount)
    return [
        [
            "R",
            "F",
            quantity,
            extended_price,
            discounted_price,
            discounted_price * (1.0 + tax),
            quantity,
            extended_price,
            discount,
            1,
        ]
    ]


def _compute_tiny_q3_rows(params: dict[str, object] | None) -> list[list[object]]:
    """Compute the MonetDB-equivalent Q3 rows for the one-row tiny fixture."""
    if params is None:
        return []
    customer_segment = "BUILDING"
    orderdate = date.fromisoformat("1995-01-01")
    shipdate = date.fromisoformat("1995-03-15")
    cutoff_date = date.fromisoformat(str(params["DATE"]))
    if str(params["SEGMENT"]) != customer_segment:
        return []
    if orderdate >= cutoff_date:
        return []
    if shipdate <= cutoff_date:
        return []
    revenue = 100.0 * (1.0 - 0.05)
    return [[1, revenue, orderdate.isoformat(), 0]]


def _compute_q2_min_cost_rows(params: dict[str, object] | None) -> list[list[object]]:
    """Compute the MonetDB-equivalent Q2 rows for the min-cost fixture."""
    if params is None:
        return []
    if int(params["SIZE"]) != 9:
        return []
    if str(params["TYPE"]) != "COPPER":
        return []
    if str(params["REGION"]) != "AFRICA":
        return []
    return [
        [
            400.0,
            "Supplier#000000002",
            "ALGERIA",
            1,
            "Manufacturer#1",
            "2 Supplier Street",
            "10-000-000-0002",
            "q2 cheapest supplier fixture",
        ]
    ]


def _write_q2_min_cost_fixture(root: Path) -> None:
    """Write a temporary TPC-H fixture that gives Q2 a non-empty result."""
    root.mkdir(parents=True, exist_ok=True)
    for table_path in FIXTURE_DIR.glob("*.tbl"):
        (root / table_path.name).write_text(
            table_path.read_text(encoding="utf-8"),
            encoding="utf-8",
        )
    (root / "region.tbl").write_text(
        "0|AFRICA|q2 region fixture|\n",
        encoding="utf-8",
    )
    (root / "nation.tbl").write_text(
        "0|ALGERIA|0|q2 nation fixture|\n",
        encoding="utf-8",
    )
    (root / "part.tbl").write_text(
        "1|Part#000000001|Manufacturer#1|Brand#11|SMALL BRUSHED COPPER|"
        "9|SM BOX|100.00|q2 matching part fixture|\n",
        encoding="utf-8",
    )
    (root / "supplier.tbl").write_text(
        "1|Supplier#000000001|1 Supplier Street|0|10-000-000-0001|"
        "900.00|q2 higher cost supplier fixture|\n"
        "2|Supplier#000000002|2 Supplier Street|0|10-000-000-0002|"
        "400.00|q2 cheapest supplier fixture|\n",
        encoding="utf-8",
    )
    (root / "partsupp.tbl").write_text(
        "1|1|10|50.00|q2 higher cost partsupp fixture|\n"
        "1|2|10|25.00|q2 minimum cost partsupp fixture|\n",
        encoding="utf-8",
    )
    return None


def _compute_tiny_q4_rows(params: dict[str, object] | None) -> list[list[object]]:
    """Compute the MonetDB-equivalent Q4 rows for the one-row tiny fixture."""
    if params is None:
        return []
    orderdate = date.fromisoformat("1995-01-01")
    commitdate = date.fromisoformat("1995-03-20")
    receiptdate = date.fromisoformat("1995-03-25")
    start_date = date.fromisoformat(str(params["DATE"]))
    end_date = _add_months(start_date, 3)
    if orderdate < start_date or orderdate >= end_date:
        return []
    if not commitdate < receiptdate:
        return []
    return [["5-LOW", 1]]


def _compute_tiny_q5_rows(params: dict[str, object] | None) -> list[list[object]]:
    """Compute the MonetDB-equivalent Q5 rows for the one-row tiny fixture."""
    if params is None:
        return []
    orderdate = date.fromisoformat("1995-01-01")
    start_date = date.fromisoformat(str(params["DATE"]))
    end_date = start_date.replace(year=start_date.year + 1)
    if str(params["REGION"]) != "AFRICA":
        return []
    if orderdate < start_date or orderdate >= end_date:
        return []
    return [["ALGERIA", 95.0]]


def _compute_q7_shipping_rows(params: dict[str, object] | None) -> list[list[object]]:
    """Compute the MonetDB-equivalent Q7 rows for the nation-pair fixture."""
    if params is None:
        return []
    selected = {str(params["NATION1"]), str(params["NATION2"])}
    if selected != {"EGYPT", "CHINA"}:
        return []
    return [
        ["CHINA", "EGYPT", 1996, 180.0],
        ["EGYPT", "CHINA", 1995, 95.0],
    ]


def _write_q7_nation_pair_fixture(root: Path) -> None:
    """Write a temporary TPC-H fixture that gives Q7 bidirectional rows."""
    root.mkdir(parents=True, exist_ok=True)
    for table_path in FIXTURE_DIR.glob("*.tbl"):
        (root / table_path.name).write_text(
            table_path.read_text(encoding="utf-8"),
            encoding="utf-8",
        )
    (root / "nation.tbl").write_text(
        "0|EGYPT|0|q7 egypt nation fixture|\n"
        "1|CHINA|0|q7 china nation fixture|\n",
        encoding="utf-8",
    )
    (root / "supplier.tbl").write_text(
        "1|Supplier#000000001|1 Supplier Street|0|10-000-000-0001|"
        "100.00|q7 egypt supplier fixture|\n"
        "2|Supplier#000000002|2 Supplier Street|1|10-000-000-0002|"
        "200.00|q7 china supplier fixture|\n",
        encoding="utf-8",
    )
    (root / "customer.tbl").write_text(
        "1|Customer#000000001|1 Main Street|1|30-000-000-0001|"
        "1000.00|BUILDING|q7 china customer fixture|\n"
        "2|Customer#000000002|2 Main Street|0|20-000-000-0002|"
        "2000.00|BUILDING|q7 egypt customer fixture|\n",
        encoding="utf-8",
    )
    (root / "orders.tbl").write_text(
        "1|1|O|1000.00|1995-01-01|5-LOW|Clerk#000000001|0|"
        "q7 china customer order fixture|\n"
        "2|2|O|2000.00|1996-01-01|5-LOW|Clerk#000000002|0|"
        "q7 egypt customer order fixture|\n",
        encoding="utf-8",
    )
    (root / "lineitem.tbl").write_text(
        "1|1|1|1|10.00|100.00|0.05|0.07|N|F|1995-03-15|"
        "1995-03-20|1995-03-25|DELIVER IN PERSON|TRUCK|q7 egypt to china fixture|\n"
        "2|1|2|1|10.00|200.00|0.10|0.07|N|F|1996-07-01|"
        "1996-07-05|1996-07-10|DELIVER IN PERSON|TRUCK|q7 china to egypt fixture|\n",
        encoding="utf-8",
    )
    return None


def _compute_q8_market_share_rows(params: dict[str, object] | None) -> list[list[object]]:
    """Compute the MonetDB-equivalent Q8 rows for the market-share fixture."""
    if params is None:
        return []
    if str(params["NATION"]) != "EGYPT":
        return []
    if str(params["REGION"]) != "MIDDLE EAST":
        return []
    if str(params["TYPE"]) != "STANDARD PLATED TIN":
        return []
    return [[1995, 0.25], [1996, 1.0]]


def _write_q8_market_share_fixture(root: Path) -> None:
    """Write a temporary TPC-H fixture that gives Q8 non-empty ratio rows."""
    root.mkdir(parents=True, exist_ok=True)
    for table_path in FIXTURE_DIR.glob("*.tbl"):
        (root / table_path.name).write_text(
            table_path.read_text(encoding="utf-8"),
            encoding="utf-8",
        )
    (root / "region.tbl").write_text(
        "0|MIDDLE EAST|q8 region fixture|\n",
        encoding="utf-8",
    )
    (root / "nation.tbl").write_text(
        "0|EGYPT|0|q8 selected supplier nation fixture|\n"
        "1|IRAN|0|q8 customer and other supplier nation fixture|\n",
        encoding="utf-8",
    )
    (root / "part.tbl").write_text(
        "1|Part#000000001|Manufacturer#1|Brand#11|STANDARD PLATED TIN|"
        "1|SM BOX|100.00|q8 matching part fixture|\n",
        encoding="utf-8",
    )
    (root / "supplier.tbl").write_text(
        "1|Supplier#000000001|1 Supplier Street|0|10-000-000-0001|"
        "100.00|q8 selected supplier fixture|\n"
        "2|Supplier#000000002|2 Supplier Street|1|10-000-000-0002|"
        "200.00|q8 other supplier fixture|\n",
        encoding="utf-8",
    )
    (root / "customer.tbl").write_text(
        "1|Customer#000000001|1 Main Street|1|30-000-000-0001|"
        "1000.00|BUILDING|q8 regional customer fixture|\n",
        encoding="utf-8",
    )
    (root / "orders.tbl").write_text(
        "1|1|O|1000.00|1995-01-01|5-LOW|Clerk#000000001|0|"
        "q8 1995 order fixture|\n"
        "2|1|O|2000.00|1995-06-01|5-LOW|Clerk#000000002|0|"
        "q8 1995 denominator order fixture|\n"
        "3|1|O|3000.00|1996-01-01|5-LOW|Clerk#000000003|0|"
        "q8 1996 order fixture|\n",
        encoding="utf-8",
    )
    (root / "lineitem.tbl").write_text(
        "1|1|1|1|10.00|100.00|0.00|0.07|N|F|1995-03-15|"
        "1995-03-20|1995-03-25|DELIVER IN PERSON|TRUCK|q8 selected 1995 fixture|\n"
        "2|1|2|1|10.00|300.00|0.00|0.07|N|F|1995-06-15|"
        "1995-06-20|1995-06-25|DELIVER IN PERSON|TRUCK|q8 other 1995 fixture|\n"
        "3|1|1|1|10.00|200.00|0.00|0.07|N|F|1996-03-15|"
        "1996-03-20|1996-03-25|DELIVER IN PERSON|TRUCK|q8 selected 1996 fixture|\n",
        encoding="utf-8",
    )
    return None


def _compute_q9_profit_rows(params: dict[str, object] | None) -> list[list[object]]:
    """Compute the MonetDB-equivalent Q9 rows for the profit fixture."""
    if params is None:
        return []
    if str(params["COLOR"]) != "gainsboro":
        return []
    return [
        ["ALGERIA", 1996, 30.0],
        ["ALGERIA", 1995, 40.0],
        ["BRAZIL", 1995, 70.0],
    ]


def _write_q9_profit_fixture(root: Path) -> None:
    """Write a temporary TPC-H fixture that gives Q9 non-empty profit rows."""
    root.mkdir(parents=True, exist_ok=True)
    for table_path in FIXTURE_DIR.glob("*.tbl"):
        (root / table_path.name).write_text(
            table_path.read_text(encoding="utf-8"),
            encoding="utf-8",
        )
    (root / "nation.tbl").write_text(
        "0|ALGERIA|0|q9 algeria nation fixture|\n"
        "1|BRAZIL|0|q9 brazil nation fixture|\n",
        encoding="utf-8",
    )
    (root / "part.tbl").write_text(
        "1|gainsboro profit part fixture|Manufacturer#1|Brand#11|"
        "STANDARD PLATED TIN|1|SM BOX|100.00|q9 matching part fixture|\n",
        encoding="utf-8",
    )
    (root / "supplier.tbl").write_text(
        "1|Supplier#000000001|1 Supplier Street|0|10-000-000-0001|"
        "100.00|q9 algeria supplier fixture|\n"
        "2|Supplier#000000002|2 Supplier Street|1|10-000-000-0002|"
        "200.00|q9 brazil supplier fixture|\n",
        encoding="utf-8",
    )
    (root / "partsupp.tbl").write_text(
        "1|1|10|10.00|q9 algeria supply cost fixture|\n"
        "1|2|10|20.00|q9 brazil supply cost fixture|\n",
        encoding="utf-8",
    )
    (root / "orders.tbl").write_text(
        "1|1|O|1000.00|1995-01-01|5-LOW|Clerk#000000001|0|"
        "q9 algeria 1995 order fixture|\n"
        "2|1|O|2000.00|1996-01-01|5-LOW|Clerk#000000002|0|"
        "q9 algeria 1996 order fixture|\n"
        "3|1|O|3000.00|1995-06-01|5-LOW|Clerk#000000003|0|"
        "q9 brazil 1995 order fixture|\n",
        encoding="utf-8",
    )
    (root / "lineitem.tbl").write_text(
        "1|1|1|1|5.00|100.00|0.10|0.07|N|F|1995-03-15|"
        "1995-03-20|1995-03-25|DELIVER IN PERSON|TRUCK|q9 algeria 1995 fixture|\n"
        "2|1|1|1|2.00|50.00|0.00|0.07|N|F|1996-03-15|"
        "1996-03-20|1996-03-25|DELIVER IN PERSON|TRUCK|q9 algeria 1996 fixture|\n"
        "3|1|2|1|4.00|200.00|0.25|0.07|N|F|1995-07-15|"
        "1995-07-20|1995-07-25|DELIVER IN PERSON|TRUCK|q9 brazil 1995 fixture|\n",
        encoding="utf-8",
    )
    return None


def _compute_tiny_q10_rows(params: dict[str, object] | None) -> list[list[object]]:
    """Compute the MonetDB-equivalent Q10 rows for the one-row tiny fixture."""
    if params is None:
        return []
    orderdate = date.fromisoformat("1995-01-01")
    start_date = date.fromisoformat(str(params["DATE"]))
    end_date = _add_months(start_date, 3)
    if orderdate < start_date or orderdate >= end_date:
        return []
    return [
        [
            1,
            "Customer#000000001",
            95.0,
            1000.0,
            "ALGERIA",
            "1 Main Street",
            "10-000-000-0000",
            "tiny customer fixture",
        ]
    ]


def _compute_tiny_q11_rows(params: dict[str, object] | None) -> list[list[object]]:
    """Compute the MonetDB-equivalent Q11 rows for the one-row tiny fixture."""
    if params is None:
        return []
    if str(params["NATION"]) != "ALGERIA":
        return []
    value = 50.0 * 10.0
    threshold = value * float(params["FRACTION"])
    if value <= threshold:
        return []
    return [[1, value]]


def _compute_tiny_q13_rows(params: dict[str, object] | None) -> list[list[object]]:
    """Compute the MonetDB-equivalent Q13 rows for the one-row tiny fixture."""
    if params is None:
        return []
    comment = "tiny orders fixture"
    word1 = str(params["WORD1"])
    word2 = str(params["WORD2"])
    first_pos = comment.find(word1)
    excluded = first_pos >= 0 and comment.find(word2, first_pos + len(word1)) >= 0
    order_count = 0 if excluded else 1
    return [[order_count, 1]]


def _compute_tiny_q6_revenue(params: dict[str, object] | None) -> float | None:
    """Compute the MonetDB-equivalent Q6 revenue for the one-row tiny fixture."""
    if params is None:
        return 5.0
    shipdate = date.fromisoformat("1995-03-15")
    discount = 0.05
    quantity = 10.0
    extended_price = 100.0
    start_date = date.fromisoformat(str(params["DATE"]))
    end_date = start_date.replace(year=start_date.year + 1)
    expected_discount = float(params["DISCOUNT"])
    max_quantity = float(params["QUANTITY"])
    if shipdate < start_date or shipdate >= end_date:
        return None
    if (
        discount < expected_discount - 0.01
        or discount > expected_discount + 0.01
    ):
        return None
    if quantity >= max_quantity:
        return None
    return extended_price * discount


def _compute_tiny_q12_rows(params: dict[str, object] | None) -> list[list[object]]:
    """Compute the MonetDB-equivalent Q12 rows for the one-row tiny fixture."""
    if params is None:
        return []
    receiptdate = date.fromisoformat("1995-03-25")
    commitdate = date.fromisoformat("1995-03-20")
    shipdate = date.fromisoformat("1995-03-15")
    shipmode = "TRUCK"
    orderpriority = "5-LOW"
    start_date = date.fromisoformat(str(params["DATE"]))
    end_date = start_date.replace(year=start_date.year + 1)
    selected_shipmodes = {str(params["SHIPMODE1"]), str(params["SHIPMODE2"])}
    if shipmode not in selected_shipmodes:
        return []
    if not (commitdate < receiptdate and shipdate < commitdate):
        return []
    if receiptdate < start_date or receiptdate >= end_date:
        return []
    high_count = 1 if orderpriority in {"1-URGENT", "2-HIGH"} else 0
    low_count = 0 if high_count else 1
    return [[shipmode, high_count, low_count]]


def _compute_tiny_q14_promo_revenue(params: dict[str, object] | None) -> float | None:
    """Compute the MonetDB-equivalent Q14 ratio for the one-row tiny fixture."""
    if params is None:
        return None
    shipdate = date.fromisoformat("1995-03-15")
    start_date = date.fromisoformat(str(params["DATE"]))
    end_date = _add_one_month(start_date)
    if shipdate < start_date or shipdate >= end_date:
        return None
    extended_price = 100.0
    discount = 0.05
    discounted_revenue = extended_price * (1.0 - discount)
    promo_revenue = discounted_revenue
    if discounted_revenue == 0.0:
        return None
    return 100.0 * promo_revenue / discounted_revenue


def _compute_tiny_q15_rows(params: dict[str, object] | None) -> list[list[object]]:
    """Compute the MonetDB-equivalent Q15 rows for the one-row tiny fixture."""
    if params is None:
        return []
    shipdate = date.fromisoformat("1995-03-15")
    start_date = date.fromisoformat(str(params["DATE"]))
    end_date = _add_months(start_date, 3)
    if shipdate < start_date or shipdate >= end_date:
        return []
    return [
        [
            1,
            "Supplier#000000001",
            "1 Supplier Street",
            "10-000-000-0001",
            95.0,
        ]
    ]


def _compute_tiny_q16_rows(params: dict[str, object] | None) -> list[list[object]]:
    """Compute the MonetDB-equivalent Q16 rows for the one-row tiny fixture."""
    if params is None:
        return []
    part_brand = "Brand#11"
    part_type = "PROMO BURNISHED COPPER"
    part_size = 1
    supplier_comment = "tiny supplier fixture"
    first_pos = supplier_comment.find("Customer")
    supplier_excluded = (
        first_pos >= 0
        and supplier_comment.find("Complaints", first_pos + len("Customer")) >= 0
    )
    selected_sizes = {int(params[f"SIZE{idx}"]) for idx in range(1, 9)}
    if supplier_excluded:
        return []
    if part_brand == str(params["BRAND"]):
        return []
    if part_type.startswith(str(params["TYPE"])):
        return []
    if part_size not in selected_sizes:
        return []
    return [[part_brand, part_type, part_size, 1]]


def _compute_q17_two_line_avg_yearly(params: dict[str, object] | None) -> float | None:
    """Compute the MonetDB-equivalent Q17 result for the two-line fixture."""
    if params is None:
        return None
    if str(params["BRAND"]) != "Brand#11":
        return None
    if str(params["CONTAINER"]) != "SM BOX":
        return None
    average_quantity = (10.0 + 1.0) / 2.0
    low_quantity = 1.0
    if low_quantity >= 0.2 * average_quantity:
        return None
    return 100.0 / 7.0


def _write_q17_two_line_fixture(root: Path) -> None:
    """Write a temporary TPC-H fixture that gives Q17 a non-empty result."""
    root.mkdir(parents=True, exist_ok=True)
    for table_path in FIXTURE_DIR.glob("*.tbl"):
        (root / table_path.name).write_text(
            table_path.read_text(encoding="utf-8"),
            encoding="utf-8",
        )
    (root / "part.tbl").write_text(
        "1|Part#000000001|Manufacturer#1|Brand#11|PROMO BURNISHED COPPER|"
        "1|SM BOX|100.00|tiny part fixture|\n",
        encoding="utf-8",
    )
    (root / "lineitem.tbl").write_text(
        "1|1|1|1|10.00|100.00|0.05|0.07|R|F|1995-03-15|"
        "1995-03-20|1995-03-25|DELIVER IN PERSON|TRUCK|tiny lineitem fixture|\n"
        "1|1|1|2|1.00|100.00|0.05|0.07|R|F|1995-03-15|"
        "1995-03-20|1995-03-25|DELIVER IN PERSON|TRUCK|tiny lineitem fixture 2|\n",
        encoding="utf-8",
    )
    return None


def _compute_q18_large_order_rows(params: dict[str, object] | None) -> list[list[object]]:
    """Compute the MonetDB-equivalent Q18 rows for the large-order fixture."""
    if params is None:
        return []
    quantity_sum = 313.0
    if quantity_sum <= float(params["QUANTITY"]):
        return []
    return [
        [
            "Customer#000000001",
            1,
            1,
            "1995-01-01",
            1000.0,
            quantity_sum,
        ]
    ]


def _write_q18_large_order_fixture(root: Path) -> None:
    """Write a temporary TPC-H fixture that gives Q18 a non-empty result."""
    root.mkdir(parents=True, exist_ok=True)
    for table_path in FIXTURE_DIR.glob("*.tbl"):
        (root / table_path.name).write_text(
            table_path.read_text(encoding="utf-8"),
            encoding="utf-8",
        )
    (root / "lineitem.tbl").write_text(
        "1|1|1|1|313.00|100.00|0.05|0.07|R|F|1995-03-15|"
        "1995-03-20|1995-03-25|DELIVER IN PERSON|TRUCK|large lineitem fixture|\n",
        encoding="utf-8",
    )
    return None


def _compute_q19_branch_revenue(params: dict[str, object] | None) -> float | None:
    """Compute the MonetDB-equivalent Q19 result for the branch fixture."""
    if params is None:
        return None
    part_brand = "Brand#32"
    part_container = "SM BOX"
    part_size = 1
    lineitem_quantity = 1.0
    shipmode = "AIR"
    shipinstruct = "DELIVER IN PERSON"
    if shipinstruct != "DELIVER IN PERSON" or shipmode not in {"AIR", "AIR REG"}:
        return None
    branch_matches = (
        part_brand == str(params["BRAND1"])
        and part_container in {"SM CASE", "SM BOX", "SM PACK", "SM PKG"}
        and lineitem_quantity >= float(params["QUANTITY1"])
        and lineitem_quantity <= float(params["QUANTITY1"]) + 10.0
        and 1 <= part_size <= 5
    )
    if not branch_matches:
        return None
    return 100.0 * (1.0 - 0.05)


def _write_q19_branch_fixture(root: Path) -> None:
    """Write a temporary TPC-H fixture that gives Q19 a non-empty result."""
    root.mkdir(parents=True, exist_ok=True)
    for table_path in FIXTURE_DIR.glob("*.tbl"):
        (root / table_path.name).write_text(
            table_path.read_text(encoding="utf-8"),
            encoding="utf-8",
        )
    (root / "part.tbl").write_text(
        "1|Part#000000001|Manufacturer#1|Brand#32|PROMO BURNISHED COPPER|"
        "1|SM BOX|100.00|q19 branch part fixture|\n",
        encoding="utf-8",
    )
    (root / "lineitem.tbl").write_text(
        "1|1|1|1|1.00|100.00|0.05|0.07|R|F|1995-03-15|"
        "1995-03-20|1995-03-25|DELIVER IN PERSON|AIR|q19 branch lineitem fixture|\n",
        encoding="utf-8",
    )
    return None


def _compute_q20_promotion_rows(params: dict[str, object] | None) -> list[list[object]]:
    """Compute the MonetDB-equivalent Q20 rows for the promotion fixture."""
    if params is None:
        return []
    part_name = "gainsboro promotion part fixture"
    nation_name = "CANADA"
    shipdate = date.fromisoformat("1995-03-15")
    start_date = date.fromisoformat(str(params["DATE"]))
    end_date = start_date.replace(year=start_date.year + 1)
    quantity_sum = 10.0
    available_quantity = 10
    if not part_name.startswith(str(params["COLOR"])):
        return []
    if nation_name != str(params["NATION"]):
        return []
    if shipdate < start_date or shipdate >= end_date:
        return []
    if available_quantity <= 0.5 * quantity_sum:
        return []
    return [["Supplier#000000001", "1 Supplier Street"]]


def _write_q20_promotion_fixture(root: Path) -> None:
    """Write a temporary TPC-H fixture that gives Q20 a non-empty result."""
    root.mkdir(parents=True, exist_ok=True)
    for table_path in FIXTURE_DIR.glob("*.tbl"):
        (root / table_path.name).write_text(
            table_path.read_text(encoding="utf-8"),
            encoding="utf-8",
        )
    (root / "nation.tbl").write_text(
        "0|CANADA|0|q20 nation fixture|\n",
        encoding="utf-8",
    )
    (root / "part.tbl").write_text(
        "1|gainsboro promotion part fixture|Manufacturer#1|Brand#11|"
        "PROMO BURNISHED COPPER|1|SM CAN|100.00|q20 part fixture|\n",
        encoding="utf-8",
    )
    (root / "partsupp.tbl").write_text(
        "1|1|10|50.00|q20 partsupp fixture|\n",
        encoding="utf-8",
    )
    (root / "supplier.tbl").write_text(
        "1|Supplier#000000001|1 Supplier Street|0|10-000-000-0001|"
        "100.00|q20 supplier fixture|\n",
        encoding="utf-8",
    )
    (root / "lineitem.tbl").write_text(
        "1|1|1|1|10.00|100.00|0.05|0.07|R|F|1995-03-15|"
        "1995-03-20|1995-03-25|DELIVER IN PERSON|TRUCK|q20 lineitem fixture|\n",
        encoding="utf-8",
    )
    return None


def _compute_q21_wait_rows(params: dict[str, object] | None) -> list[list[object]]:
    """Compute the MonetDB-equivalent Q21 rows for the supplier-wait fixture."""
    if params is None:
        return []
    if str(params["NATION"]) != "SAUDI ARABIA":
        return []
    order_status = "F"
    supplier_one_late = True
    supplier_two_late = False
    has_other_supplier = True
    if order_status != "F":
        return []
    if not supplier_one_late:
        return []
    if not has_other_supplier:
        return []
    if supplier_two_late:
        return []
    return [["Supplier#000000001", 1]]


def _write_q21_supplier_wait_fixture(root: Path) -> None:
    """Write a temporary TPC-H fixture that gives Q21 a non-empty result."""
    root.mkdir(parents=True, exist_ok=True)
    for table_path in FIXTURE_DIR.glob("*.tbl"):
        (root / table_path.name).write_text(
            table_path.read_text(encoding="utf-8"),
            encoding="utf-8",
        )
    (root / "nation.tbl").write_text(
        "0|SAUDI ARABIA|0|q21 nation fixture|\n",
        encoding="utf-8",
    )
    (root / "orders.tbl").write_text(
        "1|1|F|1000.00|1995-01-01|5-LOW|Clerk#000000001|0|"
        "q21 final order fixture|\n",
        encoding="utf-8",
    )
    (root / "supplier.tbl").write_text(
        "1|Supplier#000000001|1 Supplier Street|0|10-000-000-0001|"
        "100.00|q21 late supplier fixture|\n"
        "2|Supplier#000000002|2 Supplier Street|0|10-000-000-0002|"
        "200.00|q21 other supplier fixture|\n",
        encoding="utf-8",
    )
    (root / "lineitem.tbl").write_text(
        "1|1|1|1|10.00|100.00|0.05|0.07|N|F|1995-03-15|"
        "1995-03-20|1995-03-25|DELIVER IN PERSON|TRUCK|q21 late lineitem fixture|\n"
        "1|1|2|2|10.00|100.00|0.05|0.07|N|F|1995-03-15|"
        "1995-03-25|1995-03-20|DELIVER IN PERSON|TRUCK|q21 on-time other lineitem fixture|\n",
        encoding="utf-8",
    )
    return None


def _compute_q22_phone_prefix_rows(params: dict[str, object] | None) -> list[list[object]]:
    """Compute the MonetDB-equivalent Q22 rows for the phone-prefix fixture."""
    if params is None:
        return []
    selected_prefixes = {str(params[f"I{idx}"]) for idx in range(1, 8)}
    prefix = "30"
    if prefix not in selected_prefixes:
        return []
    balances = [100.0, 10.0]
    average_balance = sum(balances) / len(balances)
    high_customer_has_orders = False
    if high_customer_has_orders:
        return []
    if balances[0] <= average_balance:
        return []
    return [[prefix, 1, balances[0]]]


def _write_q22_phone_prefix_fixture(root: Path) -> None:
    """Write a temporary TPC-H fixture that gives Q22 a non-empty result."""
    root.mkdir(parents=True, exist_ok=True)
    for table_path in FIXTURE_DIR.glob("*.tbl"):
        (root / table_path.name).write_text(
            table_path.read_text(encoding="utf-8"),
            encoding="utf-8",
        )
    (root / "customer.tbl").write_text(
        "1|Customer#000000001|1 Main Street|0|30-000-000-0001|"
        "100.00|BUILDING|q22 high balance customer fixture|\n"
        "2|Customer#000000002|2 Main Street|0|30-000-000-0002|"
        "10.00|BUILDING|q22 low balance customer fixture|\n",
        encoding="utf-8",
    )
    (root / "orders.tbl").write_text(
        "2|2|O|100.00|1995-01-01|5-LOW|Clerk#000000002|0|"
        "q22 order for low balance customer fixture|\n",
        encoding="utf-8",
    )
    return None


def _add_one_month(value: date) -> date:
    """Return the first-day-safe one-month offset used by Q14 test params."""
    return _add_months(value, 1)


def _add_months(value: date, months: int) -> date:
    """Return the month offset used by tiny-fixture date-window checks."""
    month_index = value.month - 1 + months
    year = value.year + month_index // 12
    month = month_index % 12 + 1
    return date(year, month, value.day)


def test_tpch_runtime_validator_runs_generated_runtime_and_compares_csv(
    tmp_path: Path,
) -> None:
    """Verify TPC-H runtime validator bridges exec callback, CSV parse, and oracle."""
    oracle = _FakeMonetDBOracle()
    validator = TpchRuntimeValidator(
        workspace_path=tmp_path,
        oracle=oracle,
        sf_list=[1],
        seed=7,
    )
    captured: dict[str, object] = {}

    def fake_exec_callback(
        args_list: list[str],
        timeout_s: int,
    ) -> tuple[str, str, str]:
        """Write the generated runtime CSV expected by the validator."""
        captured["args_list"] = list(args_list)
        captured["timeout_s"] = timeout_s
        (tmp_path / "result1.csv").write_text(
            "revenue\n5.000000\n",
            encoding="utf-8",
        )
        return "exit_code: 0 signal: 0\n", "Q6 | Query ms: 1.000\n", ""

    msg, success, metrics, used_cache = validator.exec_and_validate(
        exec_callback_fn=fake_exec_callback,
        scale_factor=1,
        query_id=["Q6"],
    )

    assert success is True
    assert used_cache is False
    assert "TPC-H generated runtime validation PASS" in msg
    assert metrics["validation/correct"] is True
    assert metrics["validation/query_ids_executed"] == ["Q6"]
    assert metrics["validation/baseline_engine"] == "monetdb"
    assert metrics["validation/runtime_engine"] == "generated_runtime"
    assert captured["args_list"][0].startswith("Q6 ")
    assert oracle.calls[0]["query_id"] == "Q6"
    return None


def test_tpch_runtime_validator_exposes_q6_null_mismatch_values(
    tmp_path: Path,
) -> None:
    """Verify RunTool-visible validation text includes Q6 NULL mismatch values."""
    oracle = _FakeMonetDBOracle()
    validator = TpchRuntimeValidator(
        workspace_path=tmp_path,
        oracle=oracle,
        sf_list=[1],
        seed=1,
    )

    def fake_exec_callback(
        args_list: list[str],
        timeout_s: int,
    ) -> tuple[str, str, str]:
        """Write a generated Q6 zero result for a no-match SQL SUM case."""
        del args_list, timeout_s
        (tmp_path / "result1.csv").write_text(
            "revenue\n0.000000\n",
            encoding="utf-8",
        )
        return "exit_code: 0 signal: 0\n", "Q6 | Query ms: 1.000\n", ""

    msg, success, metrics, used_cache = validator.exec_and_validate(
        exec_callback_fn=fake_exec_callback,
        scale_factor=1,
        query_id=["Q6"],
    )

    assert success is False
    assert used_cache is False
    assert metrics["validation/correct"] is False
    assert "TPC-H generated runtime validation FAIL" in msg
    assert "first_mismatch=null:revenue:Cell value differs" in msg
    assert "expected=NULL" in msg
    assert "actual=0" in msg
    return None


def test_tpch_runtime_validator_emits_base_benchmark_speedup(
    tmp_path: Path,
) -> None:
    """Verify benchmark-stage validation includes generated vs MonetDB speedup."""
    oracle = _FakeMonetDBOracle()
    validator = TpchRuntimeValidator(
        workspace_path=tmp_path,
        oracle=oracle,
        sf_list=[1],
        seed=7,
    )
    performance_calls: list[list[str]] = []
    performance_timeouts: list[int] = []

    def fake_exec_callback(
        args_list: list[str],
        timeout_s: int,
    ) -> tuple[str, str, str]:
        """Write the correctness CSV consumed by the validator."""
        del args_list, timeout_s
        (tmp_path / "result1.csv").write_text(
            "revenue\n5.000000\n",
            encoding="utf-8",
        )
        return "exit_code: 0 signal: 0\n", "Q6 | Query ms: 1.000\n", ""

    def fake_performance_callback(
        args_list: list[str],
        timeout_s: int,
    ) -> tuple[str, str, str]:
        """Return no-output kernel timing for base benchmark comparison."""
        performance_timeouts.append(timeout_s)
        performance_calls.append(list(args_list))
        return "exit_code: 0 signal: 0\n", "1 | Execution ms: 10.000\n", ""

    msg, success, metrics, used_cache = validator.exec_and_validate(
        exec_callback_fn=fake_exec_callback,
        scale_factor=1,
        query_id=["Q6"],
        other_config={"enable_performance_comparison": True},
        performance_exec_callback_fn=fake_performance_callback,
    )

    assert success is True
    assert used_cache is False
    assert "TPC-H generated runtime validation PASS" in msg
    assert "Base benchmark performance comparison" in msg
    assert "Q6: generated=10.000 ms; MonetDB=40.000 ms; speedup=4.000x" in msg
    assert len(performance_calls) == WARMUP_RUNS + MEASURED_RUNS
    assert performance_timeouts[0] >= 180
    assert performance_timeouts[1:] == [60, 60, 60]
    assert metrics["validation/performance_comparison_enabled"] is True
    assert metrics["validation/performance_timeout_policy"]["cold_start_timeout_s"] >= 180
    assert metrics["validation/performance_timeout_policy"]["warm_query_timeout_s"] == 60
    assert metrics["validation/generated_kernel_runtime_ms_by_query"] == {
        "Q6": 10.0
    }
    assert metrics["validation/monetdb_baseline_runtime_ms_by_query"] == {
        "Q6": 40.0
    }
    assert metrics["validation/speedup_vs_monetdb_by_query"] == {"Q6": 4.0}
    assert metrics["validation/speedup_vs_monetdb_total"] == 4.0
    return None


def test_run_tool_template_query_body_fails_before_validation(tmp_path: Path) -> None:
    """Verify copied query templates do not pretend to implement TPC-H algorithms."""
    copy_template_to(tmp_path, "tpch")
    args_str, _ = gen_tpch_args_str(
        ["Q1"],
        gen_placeholders_fn=get_placeholders_fn("tpch"),
    )
    (tmp_path / "args_parser.hpp").write_text(args_str, encoding="utf-8")

    oracle = _FakeMonetDBOracle()
    validator = TpchRuntimeValidator(
        workspace_path=tmp_path,
        oracle=oracle,
        sf_list=[1],
        allowed_query_ids=["Q1"],
        seed=3,
        cache_dir=tmp_path / "validate_cache",
    )
    run_tool = RunTool(
        cwd=tmp_path,
        dataset_name="tpch",
        base_data_dir=str(FIXTURE_DIR),
        query_validator=validator,
        compile_cache_dir=tmp_path / "compile_cache",
    )

    try:
        message, metrics = run_tool.run(
            scale_factor=1,
            optimize=False,
            query_id=["Q1"],
        )
    finally:
        run_tool.reset_runtime_state()

    assert message == "TPC-H generated runtime failed before validation"
    assert metrics is not None
    assert metrics["validation/correct"] is False
    assert metrics["validation/failure"] == "generated runtime returned non-zero exit"
    assert "Template query body is absent for Q1" in metrics["validation/stderr"]
    assert oracle.calls == []
    return None


@pytest.mark.skip(reason="template query bodies are intentionally absent; base generation owns concrete TPC-H algorithms")
def test_run_tool_executes_tpch_q1_runtime_against_validator(tmp_path: Path) -> None:
    """Verify RunTool validates the TPC-H Q1 scan/group/order runtime path."""
    copy_template_to(tmp_path, "tpch")
    args_str, _ = gen_tpch_args_str(
        ["Q1"],
        gen_placeholders_fn=get_placeholders_fn("tpch"),
    )
    (tmp_path / "args_parser.hpp").write_text(args_str, encoding="utf-8")

    oracle = _FakeMonetDBOracle()
    validator = TpchRuntimeValidator(
        workspace_path=tmp_path,
        oracle=oracle,
        sf_list=[1],
        allowed_query_ids=["Q1"],
        seed=3,
        cache_dir=tmp_path / "validate_cache",
    )
    run_tool = RunTool(
        cwd=tmp_path,
        dataset_name="tpch",
        base_data_dir=str(FIXTURE_DIR),
        query_validator=validator,
        compile_cache_dir=tmp_path / "compile_cache",
    )

    try:
        message, metrics = run_tool.run(
            scale_factor=1,
            optimize=False,
            query_id=["Q1"],
        )
    finally:
        run_tool.reset_runtime_state()

    assert "TPC-H generated runtime validation PASS" in message
    assert metrics is not None
    assert metrics["validation/correct"] is True
    assert metrics["validation/query_ids_executed"] == ["Q1"]
    assert oracle.calls[0]["query_id"] == "Q1"
    assert (tmp_path / "result1.csv").read_text(encoding="utf-8") == (
        "l_returnflag,l_linestatus,sum_qty,sum_base_price,sum_disc_price,"
        "sum_charge,avg_qty,avg_price,avg_disc,count_order\n"
        "R,F,10.000000,100.000000,95.000000,101.650000,"
        "10.000000,100.000000,0.050000,1\n"
    )
    return None



@pytest.mark.skip(reason="template query bodies are intentionally absent; base generation owns concrete TPC-H algorithms")
def test_run_tool_executes_tpch_q2_runtime_against_validator(tmp_path: Path) -> None:
    """Verify RunTool validates the TPC-H Q2 min-cost supplier runtime path."""
    copy_template_to(tmp_path, "tpch")
    args_str, _ = gen_tpch_args_str(
        ["Q2"],
        gen_placeholders_fn=get_placeholders_fn("tpch"),
    )
    (tmp_path / "args_parser.hpp").write_text(args_str, encoding="utf-8")
    fixture_root = tmp_path / "q2_fixture"
    _write_q2_min_cost_fixture(fixture_root)

    oracle = _FakeMonetDBOracle()
    validator = TpchRuntimeValidator(
        workspace_path=tmp_path,
        oracle=oracle,
        sf_list=[1],
        allowed_query_ids=["Q2"],
        seed=1,
        cache_dir=tmp_path / "validate_cache",
    )
    run_tool = RunTool(
        cwd=tmp_path,
        dataset_name="tpch",
        base_data_dir=str(fixture_root),
        query_validator=validator,
        compile_cache_dir=tmp_path / "compile_cache",
    )

    try:
        message, metrics = run_tool.run(
            scale_factor=1,
            optimize=False,
            query_id=["Q2"],
        )
    finally:
        run_tool.reset_runtime_state()

    assert "TPC-H generated runtime validation PASS" in message
    assert metrics is not None
    assert metrics["validation/correct"] is True
    assert metrics["validation/query_ids_executed"] == ["Q2"]
    assert oracle.calls[0]["query_id"] == "Q2"
    assert (tmp_path / "result1.csv").read_text(encoding="utf-8") == (
        "s_acctbal,s_name,n_name,p_partkey,p_mfgr,s_address,s_phone,s_comment\n"
        "400.000000,Supplier#000000002,ALGERIA,1,Manufacturer#1,"
        "2 Supplier Street,10-000-000-0002,q2 cheapest supplier fixture\n"
    )
    return None



@pytest.mark.skip(reason="template query bodies are intentionally absent; base generation owns concrete TPC-H algorithms")
def test_run_tool_executes_tpch_q3_runtime_against_validator(tmp_path: Path) -> None:
    """Verify RunTool validates the TPC-H Q3 multi-join/order runtime path."""
    copy_template_to(tmp_path, "tpch")
    args_str, _ = gen_tpch_args_str(
        ["Q3"],
        gen_placeholders_fn=get_placeholders_fn("tpch"),
    )
    (tmp_path / "args_parser.hpp").write_text(args_str, encoding="utf-8")

    oracle = _FakeMonetDBOracle()
    validator = TpchRuntimeValidator(
        workspace_path=tmp_path,
        oracle=oracle,
        sf_list=[1],
        allowed_query_ids=["Q3"],
        seed=15,
        cache_dir=tmp_path / "validate_cache",
    )
    run_tool = RunTool(
        cwd=tmp_path,
        dataset_name="tpch",
        base_data_dir=str(FIXTURE_DIR),
        query_validator=validator,
        compile_cache_dir=tmp_path / "compile_cache",
    )

    try:
        message, metrics = run_tool.run(
            scale_factor=1,
            optimize=False,
            query_id=["Q3"],
        )
    finally:
        run_tool.reset_runtime_state()

    assert "TPC-H generated runtime validation PASS" in message
    assert metrics is not None
    assert metrics["validation/correct"] is True
    assert metrics["validation/query_ids_executed"] == ["Q3"]
    assert oracle.calls[0]["query_id"] == "Q3"
    assert (tmp_path / "result1.csv").read_text(encoding="utf-8") == (
        "l_orderkey,revenue,o_orderdate,o_shippriority\n"
        "1,95.000000,1995-01-01,0\n"
    )
    return None



@pytest.mark.skip(reason="template query bodies are intentionally absent; base generation owns concrete TPC-H algorithms")
def test_run_tool_executes_tpch_q4_runtime_against_validator(tmp_path: Path) -> None:
    """Verify RunTool validates the TPC-H Q4 exists/group/order runtime path."""
    copy_template_to(tmp_path, "tpch")
    args_str, _ = gen_tpch_args_str(
        ["Q4"],
        gen_placeholders_fn=get_placeholders_fn("tpch"),
    )
    (tmp_path / "args_parser.hpp").write_text(args_str, encoding="utf-8")

    oracle = _FakeMonetDBOracle()
    validator = TpchRuntimeValidator(
        workspace_path=tmp_path,
        oracle=oracle,
        sf_list=[1],
        allowed_query_ids=["Q4"],
        seed=25,
        cache_dir=tmp_path / "validate_cache",
    )
    run_tool = RunTool(
        cwd=tmp_path,
        dataset_name="tpch",
        base_data_dir=str(FIXTURE_DIR),
        query_validator=validator,
        compile_cache_dir=tmp_path / "compile_cache",
    )

    try:
        message, metrics = run_tool.run(
            scale_factor=1,
            optimize=False,
            query_id=["Q4"],
        )
    finally:
        run_tool.reset_runtime_state()

    assert "TPC-H generated runtime validation PASS" in message
    assert metrics is not None
    assert metrics["validation/correct"] is True
    assert metrics["validation/query_ids_executed"] == ["Q4"]
    assert oracle.calls[0]["query_id"] == "Q4"
    assert (tmp_path / "result1.csv").read_text(encoding="utf-8") == (
        "o_orderpriority,order_count\n5-LOW,1\n"
    )
    return None



@pytest.mark.skip(reason="template query bodies are intentionally absent; base generation owns concrete TPC-H algorithms")
def test_run_tool_executes_tpch_q5_runtime_against_validator(tmp_path: Path) -> None:
    """Verify RunTool validates the TPC-H Q5 six-table join runtime path."""
    copy_template_to(tmp_path, "tpch")
    args_str, _ = gen_tpch_args_str(
        ["Q5"],
        gen_placeholders_fn=get_placeholders_fn("tpch"),
    )
    (tmp_path / "args_parser.hpp").write_text(args_str, encoding="utf-8")

    oracle = _FakeMonetDBOracle()
    validator = TpchRuntimeValidator(
        workspace_path=tmp_path,
        oracle=oracle,
        sf_list=[1],
        allowed_query_ids=["Q5"],
        seed=43,
        cache_dir=tmp_path / "validate_cache",
    )
    run_tool = RunTool(
        cwd=tmp_path,
        dataset_name="tpch",
        base_data_dir=str(FIXTURE_DIR),
        query_validator=validator,
        compile_cache_dir=tmp_path / "compile_cache",
    )

    try:
        message, metrics = run_tool.run(
            scale_factor=1,
            optimize=False,
            query_id=["Q5"],
        )
    finally:
        run_tool.reset_runtime_state()

    assert "TPC-H generated runtime validation PASS" in message
    assert metrics is not None
    assert metrics["validation/correct"] is True
    assert metrics["validation/query_ids_executed"] == ["Q5"]
    assert oracle.calls[0]["query_id"] == "Q5"
    assert (tmp_path / "result1.csv").read_text(encoding="utf-8") == (
        "n_name,revenue\nALGERIA,95.000000\n"
    )
    return None



@pytest.mark.skip(reason="template query bodies are intentionally absent; base generation owns concrete TPC-H algorithms")
def test_run_tool_executes_tpch_q6_runtime_against_validator(tmp_path: Path) -> None:
    """Verify RunTool compiles and validates the TPC-H Q6 generated runtime path."""
    copy_template_to(tmp_path, "tpch")
    args_str, _ = gen_tpch_args_str(
        ["Q6"],
        gen_placeholders_fn=get_placeholders_fn("tpch"),
    )
    (tmp_path / "args_parser.hpp").write_text(args_str, encoding="utf-8")

    oracle = _FakeMonetDBOracle()
    validator = TpchRuntimeValidator(
        workspace_path=tmp_path,
        oracle=oracle,
        sf_list=[1],
        allowed_query_ids=["Q6"],
        seed=7,
        cache_dir=tmp_path / "validate_cache",
    )
    run_tool = RunTool(
        cwd=tmp_path,
        dataset_name="tpch",
        base_data_dir=str(FIXTURE_DIR),
        query_validator=validator,
        compile_cache_dir=tmp_path / "compile_cache",
    )

    try:
        message, metrics = run_tool.run(
            scale_factor=1,
            optimize=False,
            query_id=["Q6"],
        )
    finally:
        run_tool.reset_runtime_state()

    assert "TPC-H generated runtime validation PASS" in message
    assert metrics is not None
    assert metrics["validation/correct"] is True
    assert metrics["validation/query_ids_executed"] == ["Q6"]
    assert metrics["validation/baseline_engine"] == "monetdb"
    assert metrics["validation/runtime_engine"] == "generated_runtime"
    assert oracle.calls[0]["query_id"] == "Q6"
    assert (tmp_path / "result1.csv").read_text(encoding="utf-8") == (
        "revenue\n5.000000\n"
    )
    return None



@pytest.mark.skip(reason="template query bodies are intentionally absent; base generation owns concrete TPC-H algorithms")
def test_run_tool_executes_tpch_q7_runtime_against_validator(tmp_path: Path) -> None:
    """Verify RunTool validates the TPC-H Q7 bidirectional nation-pair path."""
    copy_template_to(tmp_path, "tpch")
    args_str, _ = gen_tpch_args_str(
        ["Q7"],
        gen_placeholders_fn=get_placeholders_fn("tpch"),
    )
    (tmp_path / "args_parser.hpp").write_text(args_str, encoding="utf-8")
    fixture_root = tmp_path / "q7_fixture"
    _write_q7_nation_pair_fixture(fixture_root)

    oracle = _FakeMonetDBOracle()
    validator = TpchRuntimeValidator(
        workspace_path=tmp_path,
        oracle=oracle,
        sf_list=[1],
        allowed_query_ids=["Q7"],
        seed=1,
        cache_dir=tmp_path / "validate_cache",
    )
    run_tool = RunTool(
        cwd=tmp_path,
        dataset_name="tpch",
        base_data_dir=str(fixture_root),
        query_validator=validator,
        compile_cache_dir=tmp_path / "compile_cache",
    )

    try:
        message, metrics = run_tool.run(
            scale_factor=1,
            optimize=False,
            query_id=["Q7"],
        )
    finally:
        run_tool.reset_runtime_state()

    assert "TPC-H generated runtime validation PASS" in message
    assert metrics is not None
    assert metrics["validation/correct"] is True
    assert metrics["validation/query_ids_executed"] == ["Q7"]
    assert oracle.calls[0]["query_id"] == "Q7"
    assert (tmp_path / "result1.csv").read_text(encoding="utf-8") == (
        "supp_nation,cust_nation,l_year,revenue\n"
        "CHINA,EGYPT,1996,180.000000\n"
        "EGYPT,CHINA,1995,95.000000\n"
    )
    return None



@pytest.mark.skip(reason="template query bodies are intentionally absent; base generation owns concrete TPC-H algorithms")
def test_run_tool_executes_tpch_q8_runtime_against_validator(tmp_path: Path) -> None:
    """Verify RunTool validates the TPC-H Q8 national market-share path."""
    copy_template_to(tmp_path, "tpch")
    args_str, _ = gen_tpch_args_str(
        ["Q8"],
        gen_placeholders_fn=get_placeholders_fn("tpch"),
    )
    (tmp_path / "args_parser.hpp").write_text(args_str, encoding="utf-8")
    fixture_root = tmp_path / "q8_fixture"
    _write_q8_market_share_fixture(fixture_root)

    oracle = _FakeMonetDBOracle()
    validator = TpchRuntimeValidator(
        workspace_path=tmp_path,
        oracle=oracle,
        sf_list=[1],
        allowed_query_ids=["Q8"],
        seed=1,
        cache_dir=tmp_path / "validate_cache",
    )
    run_tool = RunTool(
        cwd=tmp_path,
        dataset_name="tpch",
        base_data_dir=str(fixture_root),
        query_validator=validator,
        compile_cache_dir=tmp_path / "compile_cache",
    )

    try:
        message, metrics = run_tool.run(
            scale_factor=1,
            optimize=False,
            query_id=["Q8"],
        )
    finally:
        run_tool.reset_runtime_state()

    assert "TPC-H generated runtime validation PASS" in message
    assert metrics is not None
    assert metrics["validation/correct"] is True
    assert metrics["validation/query_ids_executed"] == ["Q8"]
    assert oracle.calls[0]["query_id"] == "Q8"
    assert (tmp_path / "result1.csv").read_text(encoding="utf-8") == (
        "o_year,mkt_share\n"
        "1995,0.250000\n"
        "1996,1.000000\n"
    )
    return None



@pytest.mark.skip(reason="template query bodies are intentionally absent; base generation owns concrete TPC-H algorithms")
def test_run_tool_executes_tpch_q9_runtime_against_validator(tmp_path: Path) -> None:
    """Verify RunTool validates the TPC-H Q9 profit-by-nation-year path."""
    copy_template_to(tmp_path, "tpch")
    args_str, _ = gen_tpch_args_str(
        ["Q9"],
        gen_placeholders_fn=get_placeholders_fn("tpch"),
    )
    (tmp_path / "args_parser.hpp").write_text(args_str, encoding="utf-8")
    fixture_root = tmp_path / "q9_fixture"
    _write_q9_profit_fixture(fixture_root)

    oracle = _FakeMonetDBOracle()
    validator = TpchRuntimeValidator(
        workspace_path=tmp_path,
        oracle=oracle,
        sf_list=[1],
        allowed_query_ids=["Q9"],
        seed=3,
        cache_dir=tmp_path / "validate_cache",
    )
    run_tool = RunTool(
        cwd=tmp_path,
        dataset_name="tpch",
        base_data_dir=str(fixture_root),
        query_validator=validator,
        compile_cache_dir=tmp_path / "compile_cache",
    )

    try:
        message, metrics = run_tool.run(
            scale_factor=1,
            optimize=False,
            query_id=["Q9"],
        )
    finally:
        run_tool.reset_runtime_state()

    assert "TPC-H generated runtime validation PASS" in message
    assert metrics is not None
    assert metrics["validation/correct"] is True
    assert metrics["validation/query_ids_executed"] == ["Q9"]
    assert oracle.calls[0]["query_id"] == "Q9"
    assert (tmp_path / "result1.csv").read_text(encoding="utf-8") == (
        "nation,o_year,sum_profit\n"
        "ALGERIA,1996,30.000000\n"
        "ALGERIA,1995,40.000000\n"
        "BRAZIL,1995,70.000000\n"
    )
    return None



@pytest.mark.skip(reason="template query bodies are intentionally absent; base generation owns concrete TPC-H algorithms")
def test_run_tool_executes_tpch_q10_runtime_against_validator(tmp_path: Path) -> None:
    """Verify RunTool validates the TPC-H Q10 customer top-k runtime path."""
    copy_template_to(tmp_path, "tpch")
    args_str, _ = gen_tpch_args_str(
        ["Q10"],
        gen_placeholders_fn=get_placeholders_fn("tpch"),
    )
    (tmp_path / "args_parser.hpp").write_text(args_str, encoding="utf-8")

    oracle = _FakeMonetDBOracle()
    validator = TpchRuntimeValidator(
        workspace_path=tmp_path,
        oracle=oracle,
        sf_list=[1],
        allowed_query_ids=["Q10"],
        seed=20,
        cache_dir=tmp_path / "validate_cache",
    )
    run_tool = RunTool(
        cwd=tmp_path,
        dataset_name="tpch",
        base_data_dir=str(FIXTURE_DIR),
        query_validator=validator,
        compile_cache_dir=tmp_path / "compile_cache",
    )

    try:
        message, metrics = run_tool.run(
            scale_factor=1,
            optimize=False,
            query_id=["Q10"],
        )
    finally:
        run_tool.reset_runtime_state()

    assert "TPC-H generated runtime validation PASS" in message
    assert metrics is not None
    assert metrics["validation/correct"] is True
    assert metrics["validation/query_ids_executed"] == ["Q10"]
    assert oracle.calls[0]["query_id"] == "Q10"
    assert (tmp_path / "result1.csv").read_text(encoding="utf-8") == (
        "c_custkey,c_name,revenue,c_acctbal,n_name,c_address,c_phone,c_comment\n"
        "1,Customer#000000001,95.000000,1000.000000,ALGERIA,"
        "1 Main Street,10-000-000-0000,tiny customer fixture\n"
    )
    return None



@pytest.mark.skip(reason="template query bodies are intentionally absent; base generation owns concrete TPC-H algorithms")
def test_run_tool_executes_tpch_q11_runtime_against_validator(tmp_path: Path) -> None:
    """Verify RunTool validates the TPC-H Q11 scalar-threshold runtime path."""
    copy_template_to(tmp_path, "tpch")
    args_str, _ = gen_tpch_args_str(
        ["Q11"],
        gen_placeholders_fn=get_placeholders_fn("tpch"),
    )
    (tmp_path / "args_parser.hpp").write_text(args_str, encoding="utf-8")

    oracle = _FakeMonetDBOracle()
    validator = TpchRuntimeValidator(
        workspace_path=tmp_path,
        oracle=oracle,
        sf_list=[1],
        allowed_query_ids=["Q11"],
        seed=31,
        cache_dir=tmp_path / "validate_cache",
    )
    run_tool = RunTool(
        cwd=tmp_path,
        dataset_name="tpch",
        base_data_dir=str(FIXTURE_DIR),
        query_validator=validator,
        compile_cache_dir=tmp_path / "compile_cache",
    )

    try:
        message, metrics = run_tool.run(
            scale_factor=1,
            optimize=False,
            query_id=["Q11"],
        )
    finally:
        run_tool.reset_runtime_state()

    assert "TPC-H generated runtime validation PASS" in message
    assert metrics is not None
    assert metrics["validation/correct"] is True
    assert metrics["validation/query_ids_executed"] == ["Q11"]
    assert oracle.calls[0]["query_id"] == "Q11"
    assert (tmp_path / "result1.csv").read_text(encoding="utf-8") == (
        "ps_partkey,value\n1,500.000000\n"
    )
    return None



@pytest.mark.skip(reason="template query bodies are intentionally absent; base generation owns concrete TPC-H algorithms")
def test_run_tool_executes_tpch_q12_runtime_against_validator(tmp_path: Path) -> None:
    """Verify RunTool validates the TPC-H Q12 join/case/group runtime path."""
    copy_template_to(tmp_path, "tpch")
    args_str, _ = gen_tpch_args_str(
        ["Q12"],
        gen_placeholders_fn=get_placeholders_fn("tpch"),
    )
    (tmp_path / "args_parser.hpp").write_text(args_str, encoding="utf-8")

    oracle = _FakeMonetDBOracle()
    validator = TpchRuntimeValidator(
        workspace_path=tmp_path,
        oracle=oracle,
        sf_list=[1],
        allowed_query_ids=["Q12"],
        seed=5,
        cache_dir=tmp_path / "validate_cache",
    )
    run_tool = RunTool(
        cwd=tmp_path,
        dataset_name="tpch",
        base_data_dir=str(FIXTURE_DIR),
        query_validator=validator,
        compile_cache_dir=tmp_path / "compile_cache",
    )

    try:
        message, metrics = run_tool.run(
            scale_factor=1,
            optimize=False,
            query_id=["Q12"],
        )
    finally:
        run_tool.reset_runtime_state()

    assert "TPC-H generated runtime validation PASS" in message
    assert metrics is not None
    assert metrics["validation/correct"] is True
    assert metrics["validation/query_ids_executed"] == ["Q12"]
    assert oracle.calls[0]["query_id"] == "Q12"
    assert (tmp_path / "result1.csv").read_text(encoding="utf-8") == (
        "l_shipmode,high_line_count,low_line_count\nTRUCK,0,1\n"
    )
    return None



@pytest.mark.skip(reason="template query bodies are intentionally absent; base generation owns concrete TPC-H algorithms")
def test_run_tool_executes_tpch_q13_runtime_against_validator(tmp_path: Path) -> None:
    """Verify RunTool validates the TPC-H Q13 outer-join distribution path."""
    copy_template_to(tmp_path, "tpch")
    args_str, _ = gen_tpch_args_str(
        ["Q13"],
        gen_placeholders_fn=get_placeholders_fn("tpch"),
    )
    (tmp_path / "args_parser.hpp").write_text(args_str, encoding="utf-8")

    oracle = _FakeMonetDBOracle()
    validator = TpchRuntimeValidator(
        workspace_path=tmp_path,
        oracle=oracle,
        sf_list=[1],
        allowed_query_ids=["Q13"],
        seed=2,
        cache_dir=tmp_path / "validate_cache",
    )
    run_tool = RunTool(
        cwd=tmp_path,
        dataset_name="tpch",
        base_data_dir=str(FIXTURE_DIR),
        query_validator=validator,
        compile_cache_dir=tmp_path / "compile_cache",
    )

    try:
        message, metrics = run_tool.run(
            scale_factor=1,
            optimize=False,
            query_id=["Q13"],
        )
    finally:
        run_tool.reset_runtime_state()

    assert "TPC-H generated runtime validation PASS" in message
    assert metrics is not None
    assert metrics["validation/correct"] is True
    assert metrics["validation/query_ids_executed"] == ["Q13"]
    assert oracle.calls[0]["query_id"] == "Q13"
    assert (tmp_path / "result1.csv").read_text(encoding="utf-8") == (
        "c_count,custdist\n1,1\n"
    )
    return None



@pytest.mark.skip(reason="template query bodies are intentionally absent; base generation owns concrete TPC-H algorithms")
def test_run_tool_executes_tpch_q14_runtime_against_validator(tmp_path: Path) -> None:
    """Verify RunTool validates the TPC-H Q14 join/case/ratio runtime path."""
    copy_template_to(tmp_path, "tpch")
    args_str, _ = gen_tpch_args_str(
        ["Q14"],
        gen_placeholders_fn=get_placeholders_fn("tpch"),
    )
    (tmp_path / "args_parser.hpp").write_text(args_str, encoding="utf-8")

    oracle = _FakeMonetDBOracle()
    validator = TpchRuntimeValidator(
        workspace_path=tmp_path,
        oracle=oracle,
        sf_list=[1],
        allowed_query_ids=["Q14"],
        seed=44,
        cache_dir=tmp_path / "validate_cache",
    )
    run_tool = RunTool(
        cwd=tmp_path,
        dataset_name="tpch",
        base_data_dir=str(FIXTURE_DIR),
        query_validator=validator,
        compile_cache_dir=tmp_path / "compile_cache",
    )

    try:
        message, metrics = run_tool.run(
            scale_factor=1,
            optimize=False,
            query_id=["Q14"],
        )
    finally:
        run_tool.reset_runtime_state()

    assert "TPC-H generated runtime validation PASS" in message
    assert metrics is not None
    assert metrics["validation/correct"] is True
    assert metrics["validation/query_ids_executed"] == ["Q14"]
    assert oracle.calls[0]["query_id"] == "Q14"
    assert (tmp_path / "result1.csv").read_text(encoding="utf-8") == (
        "promo_revenue\n100.000000\n"
    )
    return None



@pytest.mark.skip(reason="template query bodies are intentionally absent; base generation owns concrete TPC-H algorithms")
def test_run_tool_executes_tpch_q15_runtime_against_validator(tmp_path: Path) -> None:
    """Verify RunTool validates the TPC-H Q15 CTE/max-supplier runtime path."""
    copy_template_to(tmp_path, "tpch")
    args_str, _ = gen_tpch_args_str(
        ["Q15"],
        gen_placeholders_fn=get_placeholders_fn("tpch"),
    )
    (tmp_path / "args_parser.hpp").write_text(args_str, encoding="utf-8")

    oracle = _FakeMonetDBOracle()
    validator = TpchRuntimeValidator(
        workspace_path=tmp_path,
        oracle=oracle,
        sf_list=[1],
        allowed_query_ids=["Q15"],
        seed=25,
        cache_dir=tmp_path / "validate_cache",
    )
    run_tool = RunTool(
        cwd=tmp_path,
        dataset_name="tpch",
        base_data_dir=str(FIXTURE_DIR),
        query_validator=validator,
        compile_cache_dir=tmp_path / "compile_cache",
    )

    try:
        message, metrics = run_tool.run(
            scale_factor=1,
            optimize=False,
            query_id=["Q15"],
        )
    finally:
        run_tool.reset_runtime_state()

    assert "TPC-H generated runtime validation PASS" in message
    assert metrics is not None
    assert metrics["validation/correct"] is True
    assert metrics["validation/query_ids_executed"] == ["Q15"]
    assert oracle.calls[0]["query_id"] == "Q15"
    assert (tmp_path / "result1.csv").read_text(encoding="utf-8") == (
        "s_suppkey,s_name,s_address,s_phone,total_revenue\n"
        "1,Supplier#000000001,1 Supplier Street,10-000-000-0001,95.000000\n"
    )
    return None



@pytest.mark.skip(reason="template query bodies are intentionally absent; base generation owns concrete TPC-H algorithms")
def test_run_tool_executes_tpch_q16_runtime_against_validator(tmp_path: Path) -> None:
    """Verify RunTool validates the TPC-H Q16 anti-join/distinct runtime path."""
    copy_template_to(tmp_path, "tpch")
    args_str, _ = gen_tpch_args_str(
        ["Q16"],
        gen_placeholders_fn=get_placeholders_fn("tpch"),
    )
    (tmp_path / "args_parser.hpp").write_text(args_str, encoding="utf-8")

    oracle = _FakeMonetDBOracle()
    validator = TpchRuntimeValidator(
        workspace_path=tmp_path,
        oracle=oracle,
        sf_list=[1],
        allowed_query_ids=["Q16"],
        seed=3,
        cache_dir=tmp_path / "validate_cache",
    )
    run_tool = RunTool(
        cwd=tmp_path,
        dataset_name="tpch",
        base_data_dir=str(FIXTURE_DIR),
        query_validator=validator,
        compile_cache_dir=tmp_path / "compile_cache",
    )

    try:
        message, metrics = run_tool.run(
            scale_factor=1,
            optimize=False,
            query_id=["Q16"],
        )
    finally:
        run_tool.reset_runtime_state()

    assert "TPC-H generated runtime validation PASS" in message
    assert metrics is not None
    assert metrics["validation/correct"] is True
    assert metrics["validation/query_ids_executed"] == ["Q16"]
    assert oracle.calls[0]["query_id"] == "Q16"
    assert (tmp_path / "result1.csv").read_text(encoding="utf-8") == (
        "p_brand,p_type,p_size,supplier_cnt\n"
        "Brand#11,PROMO BURNISHED COPPER,1,1\n"
    )
    return None



@pytest.mark.skip(reason="template query bodies are intentionally absent; base generation owns concrete TPC-H algorithms")
def test_run_tool_executes_tpch_q17_runtime_against_validator(tmp_path: Path) -> None:
    """Verify RunTool validates the TPC-H Q17 correlated-threshold runtime path."""
    copy_template_to(tmp_path, "tpch")
    args_str, _ = gen_tpch_args_str(
        ["Q17"],
        gen_placeholders_fn=get_placeholders_fn("tpch"),
    )
    (tmp_path / "args_parser.hpp").write_text(args_str, encoding="utf-8")
    fixture_root = tmp_path / "q17_fixture"
    _write_q17_two_line_fixture(fixture_root)

    oracle = _FakeMonetDBOracle()
    validator = TpchRuntimeValidator(
        workspace_path=tmp_path,
        oracle=oracle,
        sf_list=[1],
        allowed_query_ids=["Q17"],
        seed=2,
        cache_dir=tmp_path / "validate_cache",
    )
    run_tool = RunTool(
        cwd=tmp_path,
        dataset_name="tpch",
        base_data_dir=str(fixture_root),
        query_validator=validator,
        compile_cache_dir=tmp_path / "compile_cache",
    )

    try:
        message, metrics = run_tool.run(
            scale_factor=1,
            optimize=False,
            query_id=["Q17"],
        )
    finally:
        run_tool.reset_runtime_state()

    assert "TPC-H generated runtime validation PASS" in message
    assert metrics is not None
    assert metrics["validation/correct"] is True
    assert metrics["validation/query_ids_executed"] == ["Q17"]
    assert oracle.calls[0]["query_id"] == "Q17"
    assert (tmp_path / "result1.csv").read_text(encoding="utf-8") == (
        "avg_yearly\n14.285714\n"
    )
    return None



@pytest.mark.skip(reason="template query bodies are intentionally absent; base generation owns concrete TPC-H algorithms")
def test_run_tool_executes_tpch_q18_runtime_against_validator(tmp_path: Path) -> None:
    """Verify RunTool validates the TPC-H Q18 having/top-k runtime path."""
    copy_template_to(tmp_path, "tpch")
    args_str, _ = gen_tpch_args_str(
        ["Q18"],
        gen_placeholders_fn=get_placeholders_fn("tpch"),
    )
    (tmp_path / "args_parser.hpp").write_text(args_str, encoding="utf-8")
    fixture_root = tmp_path / "q18_fixture"
    _write_q18_large_order_fixture(fixture_root)

    oracle = _FakeMonetDBOracle()
    validator = TpchRuntimeValidator(
        workspace_path=tmp_path,
        oracle=oracle,
        sf_list=[1],
        allowed_query_ids=["Q18"],
        seed=2,
        cache_dir=tmp_path / "validate_cache",
    )
    run_tool = RunTool(
        cwd=tmp_path,
        dataset_name="tpch",
        base_data_dir=str(fixture_root),
        query_validator=validator,
        compile_cache_dir=tmp_path / "compile_cache",
    )

    try:
        message, metrics = run_tool.run(
            scale_factor=1,
            optimize=False,
            query_id=["Q18"],
        )
    finally:
        run_tool.reset_runtime_state()

    assert "TPC-H generated runtime validation PASS" in message
    assert metrics is not None
    assert metrics["validation/correct"] is True
    assert metrics["validation/query_ids_executed"] == ["Q18"]
    assert oracle.calls[0]["query_id"] == "Q18"
    assert (tmp_path / "result1.csv").read_text(encoding="utf-8") == (
        "c_name,c_custkey,o_orderkey,o_orderdate,o_totalprice,sum_l_quantity\n"
        "Customer#000000001,1,1,1995-01-01,1000.000000,313.000000\n"
    )
    return None



@pytest.mark.skip(reason="template query bodies are intentionally absent; base generation owns concrete TPC-H algorithms")
def test_run_tool_executes_tpch_q19_runtime_against_validator(tmp_path: Path) -> None:
    """Verify RunTool validates the TPC-H Q19 OR-predicate runtime path."""
    copy_template_to(tmp_path, "tpch")
    args_str, _ = gen_tpch_args_str(
        ["Q19"],
        gen_placeholders_fn=get_placeholders_fn("tpch"),
    )
    (tmp_path / "args_parser.hpp").write_text(args_str, encoding="utf-8")
    fixture_root = tmp_path / "q19_fixture"
    _write_q19_branch_fixture(fixture_root)

    oracle = _FakeMonetDBOracle()
    validator = TpchRuntimeValidator(
        workspace_path=tmp_path,
        oracle=oracle,
        sf_list=[1],
        allowed_query_ids=["Q19"],
        seed=2,
        cache_dir=tmp_path / "validate_cache",
    )
    run_tool = RunTool(
        cwd=tmp_path,
        dataset_name="tpch",
        base_data_dir=str(fixture_root),
        query_validator=validator,
        compile_cache_dir=tmp_path / "compile_cache",
    )

    try:
        message, metrics = run_tool.run(
            scale_factor=1,
            optimize=False,
            query_id=["Q19"],
        )
    finally:
        run_tool.reset_runtime_state()

    assert "TPC-H generated runtime validation PASS" in message
    assert metrics is not None
    assert metrics["validation/correct"] is True
    assert metrics["validation/query_ids_executed"] == ["Q19"]
    assert oracle.calls[0]["query_id"] == "Q19"
    assert (tmp_path / "result1.csv").read_text(encoding="utf-8") == (
        "revenue\n95.000000\n"
    )
    return None



@pytest.mark.skip(reason="template query bodies are intentionally absent; base generation owns concrete TPC-H algorithms")
def test_run_tool_executes_tpch_q20_runtime_against_validator(tmp_path: Path) -> None:
    """Verify RunTool validates the TPC-H Q20 nested-subquery runtime path."""
    copy_template_to(tmp_path, "tpch")
    args_str, _ = gen_tpch_args_str(
        ["Q20"],
        gen_placeholders_fn=get_placeholders_fn("tpch"),
    )
    (tmp_path / "args_parser.hpp").write_text(args_str, encoding="utf-8")
    fixture_root = tmp_path / "q20_fixture"
    _write_q20_promotion_fixture(fixture_root)

    oracle = _FakeMonetDBOracle()
    validator = TpchRuntimeValidator(
        workspace_path=tmp_path,
        oracle=oracle,
        sf_list=[1],
        allowed_query_ids=["Q20"],
        seed=4,
        cache_dir=tmp_path / "validate_cache",
    )
    run_tool = RunTool(
        cwd=tmp_path,
        dataset_name="tpch",
        base_data_dir=str(fixture_root),
        query_validator=validator,
        compile_cache_dir=tmp_path / "compile_cache",
    )

    try:
        message, metrics = run_tool.run(
            scale_factor=1,
            optimize=False,
            query_id=["Q20"],
        )
    finally:
        run_tool.reset_runtime_state()

    assert "TPC-H generated runtime validation PASS" in message
    assert metrics is not None
    assert metrics["validation/correct"] is True
    assert metrics["validation/query_ids_executed"] == ["Q20"]
    assert oracle.calls[0]["query_id"] == "Q20"
    assert (tmp_path / "result1.csv").read_text(encoding="utf-8") == (
        "s_name,s_address\nSupplier#000000001,1 Supplier Street\n"
    )
    return None



@pytest.mark.skip(reason="template query bodies are intentionally absent; base generation owns concrete TPC-H algorithms")
def test_run_tool_executes_tpch_q21_runtime_against_validator(tmp_path: Path) -> None:
    """Verify RunTool validates the TPC-H Q21 exists/not-exists runtime path."""
    copy_template_to(tmp_path, "tpch")
    args_str, _ = gen_tpch_args_str(
        ["Q21"],
        gen_placeholders_fn=get_placeholders_fn("tpch"),
    )
    (tmp_path / "args_parser.hpp").write_text(args_str, encoding="utf-8")
    fixture_root = tmp_path / "q21_fixture"
    _write_q21_supplier_wait_fixture(fixture_root)

    oracle = _FakeMonetDBOracle()
    validator = TpchRuntimeValidator(
        workspace_path=tmp_path,
        oracle=oracle,
        sf_list=[1],
        allowed_query_ids=["Q21"],
        seed=27,
        cache_dir=tmp_path / "validate_cache",
    )
    run_tool = RunTool(
        cwd=tmp_path,
        dataset_name="tpch",
        base_data_dir=str(fixture_root),
        query_validator=validator,
        compile_cache_dir=tmp_path / "compile_cache",
    )

    try:
        message, metrics = run_tool.run(
            scale_factor=1,
            optimize=False,
            query_id=["Q21"],
        )
    finally:
        run_tool.reset_runtime_state()

    assert "TPC-H generated runtime validation PASS" in message
    assert metrics is not None
    assert metrics["validation/correct"] is True
    assert metrics["validation/query_ids_executed"] == ["Q21"]
    assert oracle.calls[0]["query_id"] == "Q21"
    assert (tmp_path / "result1.csv").read_text(encoding="utf-8") == (
        "s_name,numwait\nSupplier#000000001,1\n"
    )
    return None



@pytest.mark.skip(reason="template query bodies are intentionally absent; base generation owns concrete TPC-H algorithms")
def test_run_tool_executes_tpch_q22_runtime_against_validator(tmp_path: Path) -> None:
    """Verify RunTool validates the TPC-H Q22 phone-prefix runtime path."""
    copy_template_to(tmp_path, "tpch")
    args_str, _ = gen_tpch_args_str(
        ["Q22"],
        gen_placeholders_fn=get_placeholders_fn("tpch"),
    )
    (tmp_path / "args_parser.hpp").write_text(args_str, encoding="utf-8")
    fixture_root = tmp_path / "q22_fixture"
    _write_q22_phone_prefix_fixture(fixture_root)

    oracle = _FakeMonetDBOracle()
    validator = TpchRuntimeValidator(
        workspace_path=tmp_path,
        oracle=oracle,
        sf_list=[1],
        allowed_query_ids=["Q22"],
        seed=1,
        cache_dir=tmp_path / "validate_cache",
    )
    run_tool = RunTool(
        cwd=tmp_path,
        dataset_name="tpch",
        base_data_dir=str(fixture_root),
        query_validator=validator,
        compile_cache_dir=tmp_path / "compile_cache",
    )

    try:
        message, metrics = run_tool.run(
            scale_factor=1,
            optimize=False,
            query_id=["Q22"],
        )
    finally:
        run_tool.reset_runtime_state()

    assert "TPC-H generated runtime validation PASS" in message
    assert metrics is not None
    assert metrics["validation/correct"] is True
    assert metrics["validation/query_ids_executed"] == ["Q22"]
    assert oracle.calls[0]["query_id"] == "Q22"
    assert (tmp_path / "result1.csv").read_text(encoding="utf-8") == (
        "cntrycode,numcust,totacctbal\n30,1,100.000000\n"
    )
    return None


def test_tpch_runtime_validator_reports_generated_runtime_failure(
    tmp_path: Path,
) -> None:
    """Verify non-zero generated runtime responses fail before CSV comparison."""
    validator = TpchRuntimeValidator(
        workspace_path=tmp_path,
        oracle=_FakeMonetDBOracle(),
        sf_list=[1],
        seed=7,
    )

    def fake_exec_callback(_args_list: list[str], _timeout_s: int) -> tuple[str, str, str]:
        """Return a non-zero runtime response."""
        return "exit_code: 1 signal: 0\n", "", "runtime failed"

    msg, success, metrics, used_cache = validator.exec_and_validate(
        exec_callback_fn=fake_exec_callback,
        scale_factor=1,
        query_id=["Q6"],
    )

    assert success is False
    assert used_cache is False
    assert msg == "TPC-H generated runtime failed before validation"
    assert metrics["validation/correct"] is False
    assert metrics["validation/failure"] == "generated runtime returned non-zero exit"
    assert metrics["validation/stderr"] == "runtime failed"
    return None


def test_tpch_runtime_validator_replays_success_from_cache(tmp_path: Path) -> None:
    """Verify TPC-H runtime validator cache avoids runtime and oracle work."""
    oracle = _FakeMonetDBOracle()
    validator = TpchRuntimeValidator(
        workspace_path=tmp_path,
        oracle=oracle,
        sf_list=[1],
        seed=7,
        cache_dir=tmp_path / "validate_cache",
    )

    def fresh_exec_callback(args_list: list[str], _timeout_s: int) -> tuple[str, str, str]:
        """Write the generated runtime CSV for the fresh cache population."""
        assert args_list[0].startswith("Q6 ")
        (tmp_path / "result1.csv").write_text("revenue\n5.000000\n", encoding="utf-8")
        return "exit_code: 0 signal: 0\n", "", ""

    first = validator.exec_and_validate(
        exec_callback_fn=fresh_exec_callback,
        scale_factor=1,
        query_id=["Q6"],
        compile_key_hash="compile-key",
    )

    def forbidden_exec_callback(
        _args_list: list[str],
        _timeout_s: int,
    ) -> tuple[str, str, str]:
        """Fail if cache replay incorrectly executes generated runtime."""
        raise AssertionError("exec callback should not run on TPC-H cache hit")

    second = validator.exec_and_validate(
        exec_callback_fn=forbidden_exec_callback,
        scale_factor=1,
        query_id=["Q6"],
        compile_key_hash="compile-key",
        only_from_cache=True,
    )

    assert first[1] is True
    assert second[1] is True
    assert second[3] is True
    assert second[2]["validation/used_cache"] is True
    assert len(oracle.calls) == 1
    return None


def test_tpch_runtime_validator_cache_only_miss_raises(tmp_path: Path) -> None:
    """Verify TPC-H cache-only mode fails explicitly on a missing cache entry."""
    validator = TpchRuntimeValidator(
        workspace_path=tmp_path,
        oracle=_FakeMonetDBOracle(),
        sf_list=[1],
        seed=7,
        cache_dir=tmp_path / "validate_cache",
    )

    def forbidden_exec_callback(
        _args_list: list[str],
        _timeout_s: int,
    ) -> tuple[str, str, str]:
        """Fail if cache-only mode tries to execute generated runtime."""
        raise AssertionError("exec callback should not run on cache-only miss")

    with pytest.raises(CacheMissError):
        validator.exec_and_validate(
            exec_callback_fn=forbidden_exec_callback,
            scale_factor=1,
            query_id=["Q6"],
            compile_key_hash="missing-key",
            only_from_cache=True,
        )
    return None
