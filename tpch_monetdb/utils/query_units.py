from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from tpch_monetdb.utils.pipeline_contracts import HOST_SEALED_MANIFEST_TRUST_MODE


@dataclass(frozen=True)
class QueryUnit:
    """Host-owned scope and validation unit for one query or a family subset."""

    unit_id: str
    unit_kind: str
    query_ids: tuple[str, ...]
    entrypoint_files: tuple[str, ...]
    kernel_files: tuple[str, ...]
    shared_helper_files: tuple[str, ...]
    validation_scope: tuple[str, ...]
    instrumentation_scope: tuple[str, ...]


def _normalize_query_ids(query_ids: Iterable[str]) -> tuple[str, ...]:
    """Normalize query ids to a stable ordered tuple of strings."""
    return tuple(str(query_id) for query_id in query_ids)


def build_query_units_for_requested_queries(
    query_ids: Iterable[str],
) -> tuple[QueryUnit, ...]:
    """Build TPC-H query implementation units for the requested query ids."""
    requested = _normalize_query_ids(query_ids)
    return tuple(
        QueryUnit(
            unit_id=f"query:{query_id}",
            unit_kind="query",
            query_ids=(query_id,),
            entrypoint_files=(f"query_q{query_id}.cpp",),
            kernel_files=(f"query_q{query_id}.hpp", f"query_q{query_id}.cpp"),
            shared_helper_files=(),
            validation_scope=(query_id,),
            instrumentation_scope=(query_id,),
        )
        for query_id in requested
    )


def build_typed_query_unit(
    query_id: str,
) -> QueryUnit:
    """Build one TPC-H query unit after normalizing the query id."""
    units = build_query_units_for_requested_queries([query_id])
    return units[0]


def build_family_prompt_context(unit: QueryUnit) -> dict[str, object]:
    """Reject old family prompts after TPC-H query-unit replacement."""
    if unit.unit_kind != "family":
        raise ValueError(f"Query unit is not a family: {unit.unit_id}")
    raise ValueError(
        "Family query-unit prompts were removed from the TPC-H replacement path."
    )


def build_query_unit_lookup(query_ids: Iterable[str]) -> dict[str, QueryUnit]:
    """Build a per-query lookup for the projected query/family units."""
    lookup: dict[str, QueryUnit] = {}
    for unit in build_query_units_for_requested_queries(query_ids):
        for query_id in unit.query_ids:
            lookup[query_id] = unit
    return lookup


def build_active_unit_metadata(
    query_ids: Iterable[str],
    *,
    query_id: str,
) -> dict[str, object]:
    """Build callback metadata for the active unit that owns one query."""
    unit = build_query_unit_lookup(query_ids)[str(query_id)]
    return {
        "active_unit_id": unit.unit_id,
        "active_unit_kind": unit.unit_kind,
        "active_unit_files": list(
            dict.fromkeys((*unit.entrypoint_files, *unit.kernel_files, *unit.shared_helper_files))
        ),
        "active_unit_query_ids": list(unit.query_ids),
    }


def build_manifest_for_requested_queries(
    *,
    benchmark: str,
    conversation_name: str,
    query_ids: Iterable[str],
    storage_plan_snapshot: str | None,
) -> dict[str, Any]:
    """Build a host-sealed manifest from trusted benchmark and query inputs."""
    requested = _normalize_query_ids(query_ids)
    units = build_query_units_for_requested_queries(requested)
    serialized_units = [
        {
            "unit_id": unit.unit_id,
            "unit_kind": unit.unit_kind,
            "query_ids": list(unit.query_ids),
            "entrypoint_files": list(unit.entrypoint_files),
            "kernel_files": list(unit.kernel_files),
            "shared_helper_files": list(unit.shared_helper_files),
            "validation_scope": list(unit.validation_scope),
            "instrumentation_scope": list(unit.instrumentation_scope),
        }
        for unit in units
    ]
    return {
        "version": 1,
        "benchmark": benchmark,
        "conversation_name": conversation_name,
        "trust_mode": HOST_SEALED_MANIFEST_TRUST_MODE,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "requested_query_ids": list(requested),
        "storage_plan_snapshot": storage_plan_snapshot,
        "units": serialized_units,
    }


def manifest_path_for_conversation(
    conversation_dir: Path,
    *,
    benchmark: str,
    conversation_name: str,
) -> Path:
    """Return the sidecar manifest path for one scripted conversation."""
    return conversation_dir / f"{benchmark}_{conversation_name}.implementation_manifest.json"


def write_manifest_for_conversation(
    conversation_dir: Path,
    *,
    benchmark: str,
    conversation_name: str,
    query_ids: Iterable[str],
    storage_plan_snapshot: str | None,
) -> Path:
    """Write the host-sealed manifest sidecar before scripted prompts are generated."""
    manifest = build_manifest_for_requested_queries(
        benchmark=benchmark,
        conversation_name=conversation_name,
        query_ids=query_ids,
        storage_plan_snapshot=storage_plan_snapshot,
    )
    manifest_path = manifest_path_for_conversation(
        conversation_dir,
        benchmark=benchmark,
        conversation_name=conversation_name,
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest_path
