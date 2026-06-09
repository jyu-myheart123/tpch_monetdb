from __future__ import annotations

import re
from pathlib import Path

import pytest

from tpch_monetdb.benchmark.manifest import ReferenceManifest
from tpch_monetdb.dataset.dataset_tables_dict import (
    get_benchmark_schema,
    get_dataset_name,
    get_tables_for_benchmark,
)
from tpch_monetdb.dataset.gen_tpch.gen_tpch_query import instantiate_tpch_query
from tpch_monetdb.dataset.gen_tpch.tpch_queries import (
    QUERY_CONTRACTS,
    TPCH_TABLES,
    get_contract,
    list_all_contracts,
)
from tpch_monetdb.dataset.query_gen_factory import get_placeholders_fn, get_query_gen


def test_tpch_dataset_registry_declares_eight_tables_and_schema() -> None:
    """Verify the TPC-H dataset registry exposes the canonical eight-table set."""
    tables = get_tables_for_benchmark("tpch")
    schema = get_benchmark_schema("tpch").lower()

    assert get_dataset_name("tpch") == "tpch"
    assert tables == list(TPCH_TABLES)
    assert len(tables) == 8
    for table in TPCH_TABLES:
        assert f"create table {table}" in schema
    return None


def test_tpch_contracts_cover_q1_to_q22_with_required_metadata() -> None:
    """Verify each TPC-H query contract carries SQL, tables, features, and policies."""
    assert list_all_contracts() == [f"Q{query_id}" for query_id in range(1, 23)]
    assert set(QUERY_CONTRACTS) == set(list_all_contracts())

    for query_id in list_all_contracts():
        contract = get_contract(query_id)
        assert contract.sql_template.strip()
        assert contract.tables
        assert set(contract.tables).issubset(TPCH_TABLES)
        assert contract.features
        assert contract.ordering.strategy
        assert contract.comparison.strategy
        assert contract.container_profile == "smoke"
        assert "MonetDB native/MAPI" in contract.dialect_notes
        if contract.ordering.strategy == "order_by":
            assert contract.result_ordered is True
            assert contract.sorted_by
    return None


def test_tpch_query_factory_instantiates_all_queries_without_unresolved_placeholders() -> None:
    """Verify the public factory can instantiate Q1-Q22 and expose placeholders."""
    gen_query = get_query_gen("tpch")
    gen_placeholders = get_placeholders_fn("tpch")

    for query_id in list_all_contracts():
        template, sql, placeholders = gen_query(query_name=query_id, seed=7)
        placeholder_only = gen_placeholders(query_name=query_id, seed=7)

        assert template == get_contract(query_id).sql_template
        assert placeholders == placeholder_only
        assert placeholders.keys() == set(get_contract(query_id).parameter_names)
        assert re.search(r"\[[A-Z0-9_]+\]", sql) is None
        assert sql.strip().lower().startswith(("select", "with"))
    return None


def test_tpch_query_generation_accepts_numeric_ids_and_scale_factor() -> None:
    """Verify compatibility helpers accept numeric ids and scale-sensitive parameters."""
    gen_query = get_query_gen("tpch")
    _template, sql, placeholders = gen_query(query_name="11", seed=3, scale_factor=10)

    assert get_contract("11").query_id == "Q11"
    assert placeholders["FRACTION"] == "0.00001"
    assert "[FRACTION]" not in sql
    return None


def test_q18_uses_stable_aggregate_alias_for_monetdb() -> None:
    """Verify Q18 does not rely on MonetDB's generated aggregate column name."""
    instantiation = instantiate_tpch_query(
        query_id="Q18",
        scale_factor=1,
        seed=7,
    )

    assert "sum(l_quantity) as sum_l_quantity" in str(instantiation["sql"]).lower()
    return None


def test_tpch_manifest_instantiation_is_stable_and_tsbs_free() -> None:
    """Verify TPC-H manifest payloads do not carry TSBS host/time arguments."""
    first = instantiate_tpch_query(query_id="6", scale_factor=10, seed=7)
    second = instantiate_tpch_query(query_id="Q6", scale_factor=10, seed=7)
    spaced = instantiate_tpch_query(query_id="Q8", scale_factor=1, seed=7)

    assert first == second
    assert first["query_id"] == "Q6"
    assert first["scale_factor"] == 10
    assert len(first["sql_hash"]) == 16
    assert first["instantiation_id"].startswith("Q6_SF10_")
    assert first["args_string"].startswith("Q6 ")
    assert "DATE=" in first["args_string"]
    assert "DISCOUNT=" in first["args_string"]
    assert "QUANTITY=" in first["args_string"]
    assert "host" not in first["args_string"].lower()
    assert "time_start" not in first["params_json"]
    assert "cpu" not in first["sql"].lower()
    assert re.search(r"\[[A-Z0-9_]+\]", first["sql"]) is None
    assert " " in spaced["params_json"]["TYPE"]
    assert f'TYPE="{spaced["params_json"]["TYPE"]}"' in spaced["args_string"]
    return None


def test_reference_manifest_generates_tpch_instantiations_without_duplicates(
    tmp_path: Path,
) -> None:
    """Verify the replacement manifest path backfills exact TPC-H SQL only once."""
    manifest = ReferenceManifest(tmp_path / "manifest.json")

    added_count = manifest.ensure_tpch_instantiations(
        query_ids=["1", "Q6"],
        scale_factor=1,
        seed=7,
        num_instantiations=2,
    )
    duplicate_count = manifest.ensure_tpch_instantiations(
        query_ids=["1", "Q6"],
        scale_factor=1,
        seed=7,
        num_instantiations=2,
    )

    assert added_count == 4
    assert duplicate_count == 0
    for query_id in ("Q1", "Q6"):
        instantiations = manifest.get_instantiations_for_query(query_id, scale_factor=1)
        assert len(instantiations) == 2
        for instantiation in instantiations:
            contract = get_contract(query_id)
            assert instantiation.query_id == query_id
            assert instantiation.args_string.startswith(query_id)
            assert set(instantiation.params_json) == set(contract.parameter_names)
            assert instantiation.sql.strip().lower().startswith(("select", "with"))
            assert "cpu" not in instantiation.sql.lower()
            assert "time_start" not in instantiation.params_json
            assert re.search(r"\[[A-Z0-9_]+\]", instantiation.sql) is None
    return None


def test_reference_manifest_generate_from_tpch_builds_new_manifest(tmp_path: Path) -> None:
    """Verify the classmethod creates a TPC-H manifest through the new path."""
    manifest = ReferenceManifest.generate_from_tpch(
        query_ids=["Q6"],
        scale_factor=1,
        seed=9,
        manifest_path=tmp_path / "manifest.json",
        num_instantiations=1,
    )

    instantiations = manifest.get_instantiations_for_query("Q6", scale_factor=1)
    assert len(instantiations) == 1
    assert instantiations[0].query_id == "Q6"
    assert instantiations[0].instantiation_id.startswith("Q6_SF1_")
    return None


def test_unknown_tpch_query_and_benchmark_raise_clear_errors() -> None:
    """Verify unsupported query and benchmark names fail explicitly."""
    with pytest.raises(ValueError, match="Unknown TPC-H query name"):
        get_contract("Q23")

    with pytest.raises(ValueError, match="Unknown benchmark"):
        get_query_gen("unknown")
    return None
