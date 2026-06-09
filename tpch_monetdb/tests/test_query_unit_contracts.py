from pathlib import Path

from tpch_monetdb.run_gen_base_impl_tpch_monetdb import create_conversation
from tpch_monetdb.utils.pipeline_contracts import HOST_SEALED_MANIFEST_TRUST_MODE
from tpch_monetdb.utils.query_units import (
    build_manifest_for_requested_queries,
    build_query_units_for_requested_queries,
    manifest_path_for_conversation,
)


def test_query_unit_projection_keeps_only_requested_entrypoints() -> None:
    """Only requested TPC-H query entrypoints should appear in projected units."""
    units = build_query_units_for_requested_queries(["3"])
    assert len(units) == 1
    unit = units[0]
    assert unit.unit_kind == "query"
    assert unit.unit_id == "query:3"
    assert unit.query_ids == ("3",)
    assert unit.entrypoint_files == ("query_q3.cpp",)
    assert "query_q4.cpp" not in unit.entrypoint_files
    assert unit.kernel_files == ("query_q3.hpp", "query_q3.cpp")
    return None


def test_query_unit_projection_includes_tpch_queries_independently() -> None:
    """Q1/Q2/Q15 are independent TPC-H implementation units."""
    units = build_query_units_for_requested_queries(["1", "2", "15"])
    assert [unit.unit_id for unit in units] == ["query:1", "query:2", "query:15"]
    assert all(unit.unit_kind == "query" for unit in units)
    assert units[1].kernel_files == ("query_q2.hpp", "query_q2.cpp")
    return None


def test_manifest_is_host_sealed_and_does_not_expand_unrequested_queries() -> None:
    """The generated manifest must be host-sealed and respect requested-query projection."""
    manifest = build_manifest_for_requested_queries(
        benchmark="tpch",
        conversation_name="basef3v1",
        query_ids=["3"],
        storage_plan_snapshot="snap123",
    )
    assert manifest["trust_mode"] == HOST_SEALED_MANIFEST_TRUST_MODE
    assert manifest["requested_query_ids"] == ["3"]
    units = manifest["units"]
    assert len(units) == 1
    assert units[0]["entrypoint_files"] == ["query_q3.cpp"]
    return None


def test_create_conversation_writes_manifest_sidecar(tmp_path: Path) -> None:
    """Base implementation conversation creation must write the manifest sidecar up front."""
    artifacts_dir = tmp_path / "artifacts"
    conversation_dir = artifacts_dir / "conversations"
    create_conversation(
        short_name="basef3v1",
        query_ids=["3"],
        verify_sf_list=[1],
        max_scale_factor=1,
        artifacts_dir=artifacts_dir,
        benchmark="tpch",
        conversation_dir=conversation_dir,
        sample_query_args_dict={},
        base_data_dir=tmp_path / "data",
        validation_mode="strict",
        storage_plan_snapshot="snap123",
    )
    manifest_path = manifest_path_for_conversation(
        conversation_dir,
        benchmark="tpch",
        conversation_name="basef3v1",
    )
    assert manifest_path.exists()
    text = manifest_path.read_text(encoding="utf-8")
    assert HOST_SEALED_MANIFEST_TRUST_MODE in text
    assert '"requested_query_ids": [' in text
    return None
