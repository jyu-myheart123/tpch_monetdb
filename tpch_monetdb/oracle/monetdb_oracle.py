"""MonetDB native/MAPI oracle for TPC-H query execution."""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Callable

from tpch_monetdb.dataset.gen_tpch.gen_tpch_query import gen_query
from tpch_monetdb.dataset.gen_tpch.tpch_queries import get_contract
from tpch_monetdb.oracle.result import TpchQueryResult


ConnectionFactory = Callable[[], Any]


@dataclass(frozen=True)
class MonetDBConnectionConfig:
    """Connection settings for an in-container MonetDB native/MAPI endpoint."""

    hostname: str = "127.0.0.1"
    port: int = 50000
    database: str = "tpch_smoke"
    username: str = "monetdb"
    password: str = "monetdb"
    autocommit: bool = True


class MonetDBOracle:
    """Execute instantiated TPC-H SQL through MonetDB native/MAPI."""

    def __init__(
        self,
        config: MonetDBConnectionConfig | None = None,
        connection_factory: ConnectionFactory | None = None,
    ) -> None:
        """Initialize the oracle with either connection settings or an injected factory."""
        self.config = config or MonetDBConnectionConfig()
        self.connection_factory = connection_factory
        return None

    def connect(self) -> Any:
        """Open a MonetDB DB-API connection, importing pymonetdb only when needed."""
        if self.connection_factory is not None:
            return self.connection_factory()

        import pymonetdb

        return pymonetdb.connect(
            hostname=self.config.hostname,
            port=self.config.port,
            database=self.config.database,
            username=self.config.username,
            password=self.config.password,
            autocommit=self.config.autocommit,
        )

    def execute_sql(
        self,
        sql: str,
        *,
        query_id: str,
        query_type: str = "tpch",
        params: dict[str, Any] | None = None,
        sorted_by: tuple[str, ...] = (),
    ) -> TpchQueryResult:
        """Execute raw SQL and return the shared TpchQueryResult structure."""
        connection = self.connect()
        try:
            return self._execute_sql_on_connection(
                connection,
                sql,
                query_id=query_id,
                query_type=query_type,
                params={} if params is None else params,
                sorted_by=sorted_by,
            )
        finally:
            connection.close()

    def execute(
        self,
        query_id: str,
        *,
        seed: int = 7,
        scale_factor: float = 1.0,
    ) -> TpchQueryResult:
        """Instantiate one TPC-H query and execute it through MonetDB."""
        contract = get_contract(query_id)
        _template, sql, placeholders = gen_query(
            query_name=contract.query_id,
            seed=seed,
            scale_factor=scale_factor,
        )
        return self.execute_sql(
            sql,
            query_id=contract.query_id,
            query_type="tpch",
            params=placeholders,
            sorted_by=contract.sorted_by,
        )

    def execute_benchmark(
        self,
        query_id: str,
        *,
        seed: int = 7,
        scale_factor: float = 1.0,
        num_runs: int = 3,
    ) -> tuple[TpchQueryResult, float]:
        """Run one instantiated query repeatedly and return the median runtime."""
        if num_runs <= 0:
            raise ValueError(f"num_runs must be positive, got {num_runs}")

        contract = get_contract(query_id)
        _template, sql, placeholders = gen_query(
            query_name=contract.query_id,
            seed=seed,
            scale_factor=scale_factor,
        )
        return self.execute_sql_benchmark(
            sql,
            query_id=contract.query_id,
            query_type="tpch",
            params=placeholders,
            sorted_by=contract.sorted_by,
            num_runs=num_runs,
        )

    def execute_sql_benchmark(
        self,
        sql: str,
        *,
        query_id: str,
        query_type: str = "tpch",
        params: dict[str, Any] | None = None,
        sorted_by: tuple[str, ...] = (),
        num_runs: int = 3,
    ) -> tuple[TpchQueryResult, float]:
        """Run exact SQL repeatedly and return the last result plus median runtime."""
        if num_runs <= 0:
            raise ValueError(f"num_runs must be positive, got {num_runs}")

        resolved_params = {} if params is None else params
        connection = self.connect()
        try:
            self._execute_sql_on_connection(
                connection,
                sql,
                query_id=query_id,
                query_type=query_type,
                params=resolved_params,
                sorted_by=sorted_by,
            )
            runtimes_ms: list[float] = []
            result: TpchQueryResult | None = None
            for _ in range(num_runs):
                result = self._execute_sql_on_connection(
                    connection,
                    sql,
                    query_id=query_id,
                    query_type=query_type,
                    params=resolved_params,
                    sorted_by=sorted_by,
                )
                if result.exec_time_ms is None:
                    raise RuntimeError(f"Missing MonetDB runtime for {query_id}")
                runtimes_ms.append(result.exec_time_ms)
            if result is None:
                raise RuntimeError(f"MonetDB benchmark produced no result for {query_id}")
            result.raw_response = {"runtimes_ms": list(runtimes_ms)}
            return result, statistics.median(runtimes_ms)
        finally:
            connection.close()

    def _execute_sql_on_connection(
        self,
        connection: Any,
        sql: str,
        *,
        query_id: str,
        query_type: str,
        params: dict[str, Any],
        sorted_by: tuple[str, ...],
    ) -> TpchQueryResult:
        """Execute SQL on an existing DB-API connection and normalize the result."""
        cursor = connection.cursor()
        started = time.perf_counter()
        try:
            cursor.execute(sql)
            raw_rows = cursor.fetchall()
            exec_time_ms = (time.perf_counter() - started) * 1000.0
            columns = _cursor_columns(cursor)
            column_types = _infer_column_types(raw_rows, len(columns))
            rows = _normalize_rows(raw_rows)
            return TpchQueryResult(
                query_id=query_id,
                query_type=query_type,
                params=params,
                sql=sql,
                columns=columns,
                column_types=column_types,
                rows=rows,
                row_count=len(rows),
                sorted_by=sorted_by,
                source="monetdb",
                source_protocol="native-mapi",
                exec_time_ms=exec_time_ms,
            )
        finally:
            cursor.close()


def _cursor_columns(cursor: Any) -> list[str]:
    """Extract DB-API cursor column names."""
    if cursor.description is None:
        return []
    return [str(column[0]) for column in cursor.description]


def _normalize_cell(value: Any) -> Any:
    """Convert MonetDB DB-API values into JSON/comparator-friendly Python values."""
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, datetime | date):
        return value.isoformat()
    return value


def _normalize_rows(rows: list[Any]) -> list[list[Any]]:
    """Normalize DB-API row sequences to plain nested lists."""
    return [[_normalize_cell(value) for value in row] for row in rows]


def _infer_column_types(rows: list[Any], column_count: int) -> list[str]:
    """Infer coarse column types from the first non-null value in each result column."""
    inferred: list[str] = []
    for column_idx in range(column_count):
        sample = _first_non_null(rows, column_idx)
        inferred.append(_infer_value_type(sample))
    return inferred


def _first_non_null(rows: list[Any], column_idx: int) -> Any:
    """Return the first non-null value for a column, or None when no sample exists."""
    for row in rows:
        if column_idx < len(row) and row[column_idx] is not None:
            return row[column_idx]
    return None


def _infer_value_type(value: Any) -> str:
    """Map a Python DB-API value to the comparator's coarse type labels."""
    if value is None:
        return "UNKNOWN"
    if isinstance(value, bool):
        return "BOOLEAN"
    if isinstance(value, int):
        return "INTEGER"
    if isinstance(value, float | Decimal):
        return "DOUBLE"
    if isinstance(value, datetime):
        return "TIMESTAMP"
    if isinstance(value, date):
        return "DATE"
    return "STRING"
