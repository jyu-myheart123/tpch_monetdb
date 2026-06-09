import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tpch_monetdb.utils.generated_query_checks import run_generated_code_checks


def test_q9_antipattern_check_drops_legacy_time_range_diagnostic(tmp_path: Path) -> None:
    """TPC-H Q9 不应继续要求 TPC-H MonetDB timestamp lower_bound 诊断."""
    (tmp_path / "query_q9.hpp").write_text(
        "void execute_q9(Engine& engine, const Q9Args& args);\n",
        encoding="utf-8",
    )
    (tmp_path / "query_q9.cpp").write_text(
        "\n".join(
            (
                '#include "query_q9.hpp"',
                "void execute_q9(Engine& engine, const Q9Args& args) {",
                "    (void)engine; (void)args;",
                "}",
            )
        ),
        encoding="utf-8",
    )

    violations = run_generated_code_checks(
        workspace_root=tmp_path,
        expected_query_id="9",
        checks=("query_antipatterns",),
    )

    assert all(v.code != "MISSING_TIME_RANGE_BINARY_SEARCH" for v in violations)
    assert all(v.severity != "error" for v in violations)


def test_query_antipatterns_never_emits_blocking_violations(tmp_path: Path) -> None:
    """Performance anti-patterns must never block base_impl; only diagnostic-severity violations are allowed."""
    (tmp_path / "query_q6.hpp").write_text(
        "void execute_q6(Engine& engine, const Q6Args& args);\n",
        encoding="utf-8",
    )
    (tmp_path / "query_q6.cpp").write_text(
        "\n".join(
            (
                '#include "query_q6.hpp"',
                "#include <unordered_map>",
                "void execute_q6(Engine& engine, const Q6Args& args) {",
                "    std::unordered_map<int, int> bucket_aggregator;",
                "    (void)bucket_aggregator; (void)engine; (void)args;",
                "}",
            )
        ),
        encoding="utf-8",
    )

    violations = run_generated_code_checks(
        workspace_root=tmp_path,
        expected_query_id="6",
        checks=("query_antipatterns",),
    )

    assert all(v.severity != "error" for v in violations), (
        "query_antipatterns must not block base_impl on performance heuristics; "
        f"got blocking violations: {[v for v in violations if v.severity == 'error']}"
    )


def test_removed_query_quality_check_is_rejected_as_unknown(tmp_path: Path) -> None:
    (tmp_path / "query_family_single_groupby.cpp").write_text(
        "\n".join(
            (
                "void kernel() {",
                "  for (int i = 0; i < n; ++i) { sum += a[i]; }",
                "  for (int i = 0; i < n; ++i) { out += b[i]; }",
                "}",
            )
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Unknown generated_code_check: query_quality"):
        run_generated_code_checks(
            workspace_root=tmp_path,
            expected_query_id=None,
            checks=("query_quality",),
            active_unit_files=("query_family_single_groupby.cpp",),
        )
