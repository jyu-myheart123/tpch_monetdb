import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tpch_monetdb.utils.design_evidence as design_evidence


def test_build_tpch_design_evidence_writes_tpc_h_sections(tmp_path) -> None:
    """TPC-H design evidence must describe the Dockerized MonetDB replacement path."""
    evidence_path = design_evidence.build_tpch_design_evidence(
        workspace_path=tmp_path,
        query_ids=["1", "9", "22"],
        benchmark_sf=1,
    )

    content = evidence_path.read_text(encoding="utf-8")

    for header in (
        "## Docker MonetDB Runtime Boundary",
        "## TPC-H Table Cardinality",
        "## TPC-H Query Join Graph",
        "## TPC-H Output Semantics",
        "## TPC-H Layout Decision Signals",
        "## Evidence Boundaries",
    ):
        assert content.count(header) == 1

    assert "Dockerized MonetDB daemon" in content
    assert "native/MAPI" in content
    assert "`lineitem`" in content
    assert "`estimated_row_count`" in content
    assert "`Q9`" in content
    assert "`tables`: ['part', 'supplier', 'lineitem', 'partsupp', 'orders', 'nation']" in content
    assert "`features`: ['join', 'like', 'aggregation', 'group_by', 'order_by']" in content
    assert "`result_ordered`: True" in content
    assert "`float_atol`: 0.01" in content
    assert "`pressure_class`" in content
    assert "removed HTTP/query-file runner and single-table ILP paths are not the TPC-H planning source" in content
    return None


def test_design_evidence_module_no_longer_exposes_questdb_builder() -> None:
    assert not hasattr(design_evidence, "build_design_evidence")
    assert not hasattr(design_evidence, "QuestDBOracle")
    return None
