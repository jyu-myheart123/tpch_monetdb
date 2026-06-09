from __future__ import annotations

from pathlib import Path


def build_tpch_design_evidence(
    *,
    workspace_path: Path,
    query_ids: list[str],
    benchmark_sf: int,
) -> Path:
    """Build deterministic TPC-H evidence for storage/base generation."""
    from tpch_monetdb.dataset.gen_tpch.tpch_queries import (
        TPCH_TABLES,
        get_contract as get_tpch_contract,
    )

    normalized_query_ids = tuple(
        dict.fromkeys(_normalize_design_query_id(query_id) for query_id in query_ids)
    )
    sections = (
        _render_tpch_runtime_boundary_section(),
        _render_tpch_table_cardinality_section(
            tables=TPCH_TABLES,
            benchmark_sf=benchmark_sf,
        ),
        _render_tpch_query_join_graph_section(
            query_ids=normalized_query_ids,
            get_contract=get_tpch_contract,
        ),
        _render_tpch_output_semantics_section(
            query_ids=normalized_query_ids,
            get_contract=get_tpch_contract,
        ),
        _render_tpch_layout_decision_signals_section(
            query_ids=normalized_query_ids,
            get_contract=get_tpch_contract,
        ),
        _render_tpch_evidence_boundaries_section(),
    )
    target_path = workspace_path / "design_evidence.md"
    target_path.write_text("\n\n".join(sections).strip() + "\n", encoding="utf-8")
    return target_path


def _normalize_design_query_id(query_id: object) -> str:
    """Normalize query ids for design-evidence rendering."""
    value = str(query_id).strip()
    if value.lower().startswith("q"):
        value = value[1:]
    return value


def _render_tpch_runtime_boundary_section() -> str:
    """Render the Dockerized MonetDB baseline boundary."""
    return "\n".join(
        [
            "## Docker MonetDB Runtime Boundary",
            "- Baseline engine: Dockerized MonetDB daemon reached through native/MAPI.",
            "- Baseline client: container-local `pymonetdb`, not HTTP SQL.",
            "- Data source: TPC-H `.tbl` files imported into 8 MonetDB tables with COPY.",
            "- Generated runtime source: the same 8-table TPC-H directory consumed by fasttest.",
            "- Legacy boundary: removed HTTP/query-file runner and single-table ILP paths are not the TPC-H planning source.",
        ]
    )


def _render_tpch_table_cardinality_section(
    *,
    tables: tuple[str, ...],
    benchmark_sf: int,
) -> str:
    """Render approximate TPC-H table-cardinality pressure signals."""
    base_counts = {
        "region": 5,
        "nation": 25,
        "supplier": 10_000,
        "customer": 150_000,
        "part": 200_000,
        "partsupp": 800_000,
        "orders": 1_500_000,
        "lineitem": 6_001_215,
    }
    lines = ["## TPC-H Table Cardinality"]
    for table in tables:
        base_count = base_counts.get(table, 0)
        scaled_count = base_count if table in {"region", "nation"} else base_count * benchmark_sf
        lines.append(f"- `{table}`")
        lines.append(f"  - `estimated_row_count`: {scaled_count}")
        lines.append(f"  - `scale_factor`: {benchmark_sf}")
    return "\n".join(lines)


def _render_tpch_query_join_graph_section(
    *,
    query_ids: tuple[str, ...],
    get_contract,
) -> str:
    """Render tables, features, and parameter names for each TPC-H query."""
    lines = ["## TPC-H Query Join Graph"]
    for query_id in query_ids:
        contract = get_contract(f"Q{query_id}")
        lines.append(f"- `Q{query_id}`")
        lines.append(f"  - `tables`: {list(contract.tables)}")
        lines.append(f"  - `features`: {list(contract.features)}")
        lines.append(f"  - `parameters`: {list(contract.parameter_names)}")
    return "\n".join(lines)


def _render_tpch_output_semantics_section(
    *,
    query_ids: tuple[str, ...],
    get_contract,
) -> str:
    """Render ordering and comparison policies for each TPC-H query."""
    lines = ["## TPC-H Output Semantics"]
    for query_id in query_ids:
        contract = get_contract(f"Q{query_id}")
        lines.append(f"- `Q{query_id}`")
        lines.append(f"  - `result_ordered`: {contract.result_ordered}")
        lines.append(f"  - `order_by`: {list(contract.sorted_by)}")
        lines.append(f"  - `comparison_strategy`: `{contract.comparison.strategy}`")
        lines.append(f"  - `float_atol`: {contract.float_atol}")
        lines.append(f"  - `float_rtol`: {contract.float_rtol}")
    return "\n".join(lines)


def _render_tpch_layout_decision_signals_section(
    *,
    query_ids: tuple[str, ...],
    get_contract,
) -> str:
    """Render storage-plan cost signals for requested TPC-H queries."""
    lines = ["## TPC-H Layout Decision Signals"]
    for query_id in query_ids:
        contract = get_contract(f"Q{query_id}")
        pressure = _classify_tpch_query_pressure(tuple(contract.features))
        lines.append(f"- `Q{query_id}`")
        lines.append(f"  - `pressure_class`: `{pressure}`")
        lines.append(f"  - `table_count`: {len(contract.tables)}")
        lines.append(f"  - `row_count`: cite table-level estimates above for scanned tables")
        lines.append(f"  - `output_cardinality`: derive from ORDER BY / single-row policy and measured baseline rows")
        lines.append(f"  - `semantic_traps`: preserve join predicates, anti/semi join semantics, output ordering, and numeric tolerance")
    return "\n".join(lines)


def _classify_tpch_query_pressure(features: tuple[str, ...]) -> str:
    """Classify a TPC-H query into a high-level planning pressure class."""
    feature_set = set(features)
    if "correlated_subquery" in feature_set or "nested_subquery" in feature_set:
        return "correlated-or-nested-subquery"
    if "anti_join" in feature_set or "not_exists" in feature_set or "not_in" in feature_set:
        return "anti-semi-join"
    if "ratio" in feature_set or "case" in feature_set:
        return "case-ratio-aggregation"
    if "join" in feature_set and "aggregation" in feature_set:
        return "join-aggregation"
    if "scan" in feature_set:
        return "lineitem-scan"
    return "relational-query"


def _render_tpch_evidence_boundaries_section() -> str:
    """Render what this deterministic evidence does and does not prove."""
    return "\n".join(
        [
            "## Evidence Boundaries",
            "- This artifact captures static TPC-H schema, contract, ordering, and table-cardinality pressure.",
            "- It does not prove a future layout or query kernel is optimal.",
            "- Treat missing measured row counts as `Unverified assumption:` until Dockerized MonetDB or runtime evidence is available.",
            "- A storage plan must cite `queries.txt`, `workload_objective.json`, and `data_law_contract.json` alongside this file.",
        ]
    )
