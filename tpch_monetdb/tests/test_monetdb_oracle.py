from __future__ import annotations

import re
from datetime import date
from decimal import Decimal
from pathlib import Path
from subprocess import CompletedProcess
from typing import Any

from tpch_monetdb.dataset.gen_tpch.tpch_queries import TPCH_TABLES
from tpch_monetdb.benchmark.manifest import QueryInstantiation
from tpch_monetdb.benchmark.providers import (
    DockerMonetDBLifecycle,
    DockerMonetDBLifecycleConfig,
    MonetDBBaselineProvider,
)
from tpch_monetdb.oracle.monetdb_oracle import MonetDBOracle
from tpch_monetdb.oracle.monetdb_prepare import (
    build_copy_sql,
    create_monetdb_copy_fixture_dir,
    prepare_tpch_database,
    sanitize_tpch_tbl_line,
)
from tpch_monetdb.oracle.result import TpchQueryResult


REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_DIR = REPO_ROOT / "docker" / "tpch-monetdb" / "fixtures" / "tiny-tpch"


def test_oracle_public_api_does_not_export_questdb_oracle() -> None:
    """Oracle package public API should not expose the legacy QuestDB HTTP oracle."""
    import tpch_monetdb.oracle as oracle_pkg

    assert "QuestDBOracle" not in oracle_pkg.__all__
    assert not hasattr(oracle_pkg, "QuestDBOracle")
    return None


class FakeCursor:
    """Small DB-API cursor fake for MonetDB prepare and oracle tests."""

    def __init__(self, connection: "FakeConnection") -> None:
        self.connection = connection
        self.description: list[tuple[str, ...]] | None = None
        self.rows: list[tuple[Any, ...]] = []
        self.fetchone_row: tuple[Any, ...] | None = None
        self.closed = False
        return None

    def execute(self, sql: str) -> None:
        """Record SQL and emulate COPY, COUNT, and SELECT query responses."""
        self.connection.executed_sql.append(sql)
        if sql.startswith("COPY INTO "):
            table, fixture_path = _parse_copy_sql(sql)
            self.connection.table_counts[table] = _count_rows(Path(fixture_path))
            return None
        if sql.startswith("SELECT COUNT(*) FROM "):
            table = sql.rsplit(" ", 1)[1]
            self.fetchone_row = (self.connection.table_counts.get(table, 0),)
            return None
        self.description = [("amount",), ("shipdate",)]
        self.rows = [(Decimal("1.25"), date(1995, 3, 15))]
        return None

    def fetchone(self) -> tuple[Any, ...] | None:
        """Return the prepared single-row COUNT response."""
        return self.fetchone_row

    def fetchall(self) -> list[tuple[Any, ...]]:
        """Return the prepared query response rows."""
        return self.rows

    def close(self) -> None:
        """Record cursor closure."""
        self.closed = True
        return None


class FakeConnection:
    """Small DB-API connection fake that creates FakeCursor objects."""

    def __init__(self) -> None:
        self.executed_sql: list[str] = []
        self.table_counts: dict[str, int] = {}
        self.closed = False
        return None

    def cursor(self) -> FakeCursor:
        """Return a fresh fake cursor."""
        return FakeCursor(self)

    def close(self) -> None:
        """Record connection closure."""
        self.closed = True
        return None


class FakeBenchmarkOracle:
    """Fake MonetDBOracle object for provider tests."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        return None

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
        """Record exact benchmark inputs and return deterministic runtime evidence."""
        self.calls.append(
            {
                "sql": sql,
                "query_id": query_id,
                "query_type": query_type,
                "params": params,
                "sorted_by": sorted_by,
                "num_runs": num_runs,
            }
        )
        result = TpchQueryResult(
            query_id=query_id,
            query_type=query_type,
            params={} if params is None else params,
            sql=sql,
            columns=["revenue"],
            column_types=["DOUBLE"],
            rows=[[1.0]],
            row_count=1,
            sorted_by=sorted_by,
            source="monetdb",
            source_protocol="native-mapi",
            exec_time_ms=2.0,
            raw_response={"runtimes_ms": [3.0, 1.0, 2.0]},
        )
        return result, 2.0


class FakeDockerRunner:
    """Fake subprocess runner for Docker lifecycle tests."""

    def __init__(self, results: list[CompletedProcess[str]]) -> None:
        self.results = list(results)
        self.calls: list[list[str]] = []
        return None

    def __call__(self, command: list[str], **_kwargs: Any) -> CompletedProcess[str]:
        """Return the next prepared subprocess result."""
        self.calls.append(command)
        if not self.results:
            raise AssertionError("No fake Docker result prepared")
        return self.results.pop(0)


def _parse_copy_sql(sql: str) -> tuple[str, str]:
    """Extract table name and fixture path from a generated COPY statement."""
    match = re.match(r"COPY INTO (\w+) FROM '([^']+)'", sql)
    if match is None:
        raise AssertionError(f"Unexpected COPY SQL: {sql}")
    return match.group(1), match.group(2)


def _count_rows(path: Path) -> int:
    """Count non-empty rows in a temporary fixture file."""
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def test_sanitize_tpch_tbl_line_removes_only_trailing_delimiter() -> None:
    """Verify TPC-H trailing delimiters are removed for MonetDB COPY."""
    assert sanitize_tpch_tbl_line("a|b|\n") == "a|b\n"
    assert sanitize_tpch_tbl_line("a|b|") == "a|b"
    assert sanitize_tpch_tbl_line("a|b\n") == "a|b\n"
    return None


def test_create_monetdb_copy_fixture_dir_preserves_source_and_field_counts(tmp_path: Path) -> None:
    """Verify COPY fixtures are sanitized in a temp directory without mutating source files."""
    target_dir = create_monetdb_copy_fixture_dir(FIXTURE_DIR, tmp_path / "copy")

    for table in TPCH_TABLES:
        source_text = (FIXTURE_DIR / f"{table}.tbl").read_text(encoding="utf-8")
        target_text = (target_dir / f"{table}.tbl").read_text(encoding="utf-8")
        assert source_text.endswith("|\n")
        assert not target_text.endswith("|\n")
        assert _count_rows(target_dir / f"{table}.tbl") == 1
    return None


def test_build_copy_sql_rejects_unknown_table() -> None:
    """Verify COPY SQL generation only accepts canonical TPC-H tables."""
    assert build_copy_sql("lineitem", Path("/tmp/data")).startswith("COPY INTO lineitem")
    try:
        build_copy_sql("cpu", Path("/tmp/data"))
    except ValueError as exc:
        assert "Unknown TPC-H table" in str(exc)
        return None
    raise AssertionError("build_copy_sql accepted an unknown table")


def test_prepare_tpch_database_creates_schema_imports_and_validates_counts() -> None:
    """Verify prepare flow executes schema, COPY import, and row-count checks."""
    connection = FakeConnection()
    report = prepare_tpch_database(connection, FIXTURE_DIR)

    assert report.schema_tables == list(TPCH_TABLES)
    assert report.expected_row_counts == {table: 1 for table in TPCH_TABLES}
    assert report.actual_row_counts == {table: 1 for table in TPCH_TABLES}
    assert any(sql.startswith("create table lineitem") for sql in connection.executed_sql)
    assert any(sql.startswith("COPY INTO lineitem") for sql in connection.executed_sql)
    return None


def test_monetdb_oracle_execute_sql_normalizes_result_shape() -> None:
    """Verify raw MonetDB rows become TpchQueryResult with MonetDB provenance."""
    connection = FakeConnection()
    oracle = MonetDBOracle(connection_factory=lambda: connection)

    result = oracle.execute_sql(
        "select amount, shipdate from lineitem",
        query_id="QX",
        params={"seed": 7},
        sorted_by=("amount",),
    )

    assert connection.closed is True
    assert result.query_id == "QX"
    assert result.query_type == "tpch"
    assert result.source == "monetdb"
    assert result.source_protocol == "native-mapi"
    assert result.columns == ["amount", "shipdate"]
    assert result.column_types == ["DOUBLE", "DATE"]
    assert result.rows == [[1.25, "1995-03-15"]]
    assert result.row_count == 1
    assert result.sorted_by == ("amount",)
    assert result.exec_time_ms is not None
    return None


def test_monetdb_oracle_execute_benchmark_returns_median_runtime() -> None:
    """Verify benchmark mode runs warmup plus measured executions."""
    connection = FakeConnection()
    oracle = MonetDBOracle(connection_factory=lambda: connection)

    result, median_ms = oracle.execute_benchmark("Q6", num_runs=3)

    assert result.query_id == "Q6"
    assert median_ms >= 0
    assert connection.closed is True
    assert len(connection.executed_sql) == 4
    return None


def test_monetdb_baseline_provider_measures_exact_manifest_sql() -> None:
    """Verify provider records MonetDB runtime from the exact QueryInstantiation SQL."""
    fake_oracle = FakeBenchmarkOracle()
    provider = MonetDBBaselineProvider(oracle=fake_oracle, num_runs=3)
    instantiation = QueryInstantiation(
        query_id="Q6",
        scale_factor=1,
        instantiation_id="tpch_Q6_sf1_seed7",
        params_json={"DATE": "1995-01-01"},
        args_string="Q6 DATE=1995-01-01",
        sql="select sum(l_extendedprice) as revenue from lineitem",
        sql_hash="abc123",
    )

    measurement = provider.measure(instantiation)

    assert fake_oracle.calls[0]["sql"] == instantiation.sql
    assert fake_oracle.calls[0]["params"] == instantiation.params_json
    assert measurement.instantiation_id == instantiation.instantiation_id
    assert measurement.runtime_ms == 2.0
    assert measurement.all_runtimes_ms == [3.0, 1.0, 2.0]
    assert measurement.engine == "monetdb"
    assert measurement.output_row_count == 1
    assert measurement.provenance["baseline_backend"] == "monetdb-native-mapi"
    assert measurement.provenance["source_protocol"] == "native-mapi"
    assert measurement.provenance["sql_hash"] == "abc123"
    return None


def test_docker_monetdb_lifecycle_stops_after_failed_preflight(tmp_path: Path) -> None:
    """Verify workflow returns structured failure and does not build when Docker is unavailable."""
    dockerfile = tmp_path / "docker" / "tpch-monetdb" / "Dockerfile"
    dockerfile.parent.mkdir(parents=True)
    dockerfile.write_text("FROM monetdb/monetdb:Dec2025-SP2\n", encoding="utf-8")
    compose_file = tmp_path / "docker" / "tpch-monetdb" / "docker-compose.yml"
    compose_file.write_text("services:\n  tpch-monetdb: {}\n", encoding="utf-8")
    runner = FakeDockerRunner(
        [
            CompletedProcess(
                args=["docker", "compose", "version"],
                returncode=1,
                stdout="",
                stderr="Docker Compose unavailable",
            )
        ]
    )
    lifecycle = DockerMonetDBLifecycle(
        DockerMonetDBLifecycleConfig(repo_root=tmp_path),
        run_command=runner,
    )

    report = lifecycle.run_compose_workflow()

    assert report["status"] == "failed"
    assert report["preflight"]["report"]["reason"] == "docker_unavailable"
    assert report["build"] is None
    assert report["up"] is None
    assert report["init"] is None
    assert len(runner.calls) == 1
    return None


def test_docker_monetdb_lifecycle_builds_starts_and_parses_init_json(tmp_path: Path) -> None:
    """Verify lifecycle commands run compose build/up/init and parse init JSON."""
    dockerfile = tmp_path / "docker" / "tpch-monetdb" / "Dockerfile"
    dockerfile.parent.mkdir(parents=True)
    dockerfile.write_text("FROM monetdb/monetdb:Dec2025-SP2\n", encoding="utf-8")
    compose_file = tmp_path / "docker" / "tpch-monetdb" / "docker-compose.yml"
    compose_file.write_text("services:\n  tpch-monetdb: {}\n  tpch-monetdb-init: {}\n", encoding="utf-8")
    init_stdout = '{"actual_row_counts":{"lineitem":1},"schema_tables":["lineitem"]}\n'
    runner = FakeDockerRunner(
        [
            CompletedProcess(args=["docker", "compose", "version"], returncode=0, stdout="v2.27.0\n", stderr=""),
            CompletedProcess(args=["docker", "compose", "build"], returncode=0, stdout="built", stderr=""),
            CompletedProcess(args=["docker", "compose", "up"], returncode=0, stdout="started", stderr=""),
            CompletedProcess(args=["docker", "compose", "run"], returncode=0, stdout=init_stdout, stderr=""),
        ]
    )
    lifecycle = DockerMonetDBLifecycle(
        DockerMonetDBLifecycleConfig(repo_root=tmp_path, image_tag="tpch-monetdb:test"),
        run_command=runner,
    )

    report = lifecycle.run_compose_workflow()

    assert report["status"] == "ok"
    assert runner.calls[0] == ["docker", "compose", "version", "--short"]
    assert runner.calls[1] == [
        "docker",
        "compose",
        "-f",
        str(compose_file),
        "build",
        "tpch-monetdb",
    ]
    assert runner.calls[2] == [
        "docker",
        "compose",
        "-f",
        str(compose_file),
        "up",
        "-d",
        "tpch-monetdb",
    ]
    assert runner.calls[3] == [
        "docker",
        "compose",
        "-f",
        str(compose_file),
        "--profile",
        "init",
        "run",
        "--rm",
        "tpch-monetdb-init",
    ]
    assert report["up"]["report"]["service"] == "tpch-monetdb"
    assert report["init"]["report"]["tpch_prepare"]["actual_row_counts"]["lineitem"] == 1
    return None
