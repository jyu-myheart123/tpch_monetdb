"""MonetDB TPC-H schema and fixture preparation helpers."""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tpch_monetdb.dataset.gen_tpch.tpch_queries import TPCH_TABLE_SCHEMAS, TPCH_TABLES


EXPECTED_TPCH_FIXTURE_FILES: tuple[str, ...] = tuple(f"{table}.tbl" for table in TPCH_TABLES)


@dataclass(frozen=True)
class TpchPrepareReport:
    """Structured report for one MonetDB TPC-H prepare run."""

    schema_tables: list[str]
    expected_row_counts: dict[str, int]
    actual_row_counts: dict[str, int]


def list_tpch_fixture_files(fixture_dir: Path) -> list[str]:
    """Return sorted fixture file names after validating all TPC-H tables exist."""
    missing = [name for name in EXPECTED_TPCH_FIXTURE_FILES if not (fixture_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing tiny TPC-H fixture files: {', '.join(missing)}")
    return sorted(path.name for path in fixture_dir.glob("*.tbl"))


def split_sql_statements(sql_text: str) -> list[str]:
    """Split semicolon-terminated schema SQL into executable statements."""
    return [statement.strip() for statement in sql_text.split(";") if statement.strip()]


def execute_statement(connection: Any, sql: str) -> None:
    """Execute one SQL statement and always close the DB-API cursor."""
    cursor = connection.cursor()
    try:
        cursor.execute(sql)
        return None
    finally:
        cursor.close()


def create_tpch_schema(connection: Any) -> list[str]:
    """Create the canonical TPC-H tables in MonetDB."""
    for table in TPCH_TABLES:
        for statement in split_sql_statements(TPCH_TABLE_SCHEMAS[table]):
            execute_statement(connection, statement)
    return list(TPCH_TABLES)


def quote_sql_path(path: Path) -> str:
    """Return a SQL string literal body for a server-local file path."""
    return str(path).replace("'", "''")


def build_copy_sql(table: str, fixture_dir: Path) -> str:
    """Build a MonetDB COPY statement for one TPC-H table fixture."""
    if table not in TPCH_TABLES:
        raise ValueError(f"Unknown TPC-H table: {table}")
    fixture_path = fixture_dir / f"{table}.tbl"
    return f"COPY INTO {table} FROM '{quote_sql_path(fixture_path)}' USING DELIMITERS '|', E'\\n'"


def sanitize_tpch_tbl_line(line: str) -> str:
    """Remove the standard TPC-H trailing delimiter for MonetDB COPY."""
    if line.endswith("|\n"):
        return f"{line[:-2]}\n"
    if line.endswith("|"):
        return line[:-1]
    return line


def create_monetdb_copy_fixture_dir(source_dir: Path, target_dir: Path) -> Path:
    """Create COPY-compatible fixture files without mutating source TPC-H data."""
    list_tpch_fixture_files(source_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    for table in TPCH_TABLES:
        source_path = source_dir / f"{table}.tbl"
        target_path = target_dir / f"{table}.tbl"
        with source_path.open("r", encoding="utf-8") as source_handle:
            with target_path.open("w", encoding="utf-8") as target_handle:
                for line in source_handle:
                    target_handle.write(sanitize_tpch_tbl_line(line))
    return target_dir


def count_fixture_rows(fixture_dir: Path) -> dict[str, int]:
    """Count non-empty rows for each TPC-H fixture file."""
    list_tpch_fixture_files(fixture_dir)
    expected: dict[str, int] = {}
    for table in TPCH_TABLES:
        fixture_path = fixture_dir / f"{table}.tbl"
        with fixture_path.open("r", encoding="utf-8") as handle:
            expected[table] = sum(1 for line in handle if line.strip())
    return expected


def fetch_table_count(connection: Any, table: str) -> int:
    """Read one table row count from MonetDB."""
    cursor = connection.cursor()
    try:
        cursor.execute(f"SELECT COUNT(*) FROM {table}")
        row = cursor.fetchone()
        if row is None:
            raise RuntimeError(f"COUNT(*) returned no row for {table}")
        return int(row[0])
    finally:
        cursor.close()


def import_tpch_fixture(connection: Any, fixture_dir: Path) -> dict[str, int]:
    """COPY all TPC-H fixture tables into MonetDB and return actual row counts."""
    for table in TPCH_TABLES:
        execute_statement(connection, build_copy_sql(table, fixture_dir))
    return {table: fetch_table_count(connection, table) for table in TPCH_TABLES}


def validate_row_counts(expected: dict[str, int], actual: dict[str, int]) -> None:
    """Raise when imported MonetDB row counts diverge from expected fixture rows."""
    mismatches = {
        table: {"expected": expected.get(table), "actual": actual.get(table)}
        for table in TPCH_TABLES
        if expected.get(table) != actual.get(table)
    }
    if mismatches:
        detail = json.dumps(mismatches, sort_keys=True)
        raise RuntimeError(f"TPC-H import row count mismatch: {detail}")
    return None


def prepare_tpch_database(connection: Any, fixture_dir: Path) -> TpchPrepareReport:
    """Create schema, import sanitized fixture files, and validate row counts."""
    schema_tables = create_tpch_schema(connection)
    expected_counts = count_fixture_rows(fixture_dir)
    with tempfile.TemporaryDirectory(prefix="tpch-monetdb-copy-") as copy_dir_name:
        copy_fixture_dir = create_monetdb_copy_fixture_dir(fixture_dir, Path(copy_dir_name))
        actual_counts = import_tpch_fixture(connection, copy_fixture_dir)
    validate_row_counts(expected_counts, actual_counts)
    return TpchPrepareReport(
        schema_tables=schema_tables,
        expected_row_counts=expected_counts,
        actual_row_counts=actual_counts,
    )
