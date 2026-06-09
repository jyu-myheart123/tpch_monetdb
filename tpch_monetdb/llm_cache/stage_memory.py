from __future__ import annotations

import hashlib
import re
from typing import Any, Iterable


STAGE_MEMORY_HEADER = "[Stage Memory v3]"
Q1_Q9_OBLIGATION_TEXTS: dict[str, str] = {
    "1": (
        "OBLIG_Q1_LINEITEM_SCAN_AGGREGATION: Engine must expose reusable lineitem "
        "columns/date filters/group keys; Q1 must not use a materialized answer cache"
    ),
    "9": (
        "OBLIG_Q9_JOIN_PROFIT_AGGREGATION: Engine must expose reusable part/supplier/"
        "partsupp/lineitem/orders/nation join support; Q9 must aggregate profit at query time"
    ),
}


def render_stage_memory(state: Any, *, artifact_refs: str | None = None) -> str:
    """Render a stable prompt memory block from StageState-like objects."""
    lines = [
        STAGE_MEMORY_HEADER,
        "schema_version: 3",
        "stage:",
        f"  profile_name: {_value(getattr(state, 'profile_name', None))}",
        f"  prompt_index: {_value(getattr(state, 'prompt_index', None))}",
        f"  prompt_descriptor: {_value(getattr(state, 'prompt_descriptor', None))}",
        "active_scope:",
        f"  query_ids: {_list_value(getattr(state, 'active_query_ids', ())) }",
        f"  unit_id: {_value(getattr(state, 'active_unit_id', None))}",
        f"  unit_kind: {_value(getattr(state, 'active_unit_kind', None))}",
        f"  files: {_list_value(getattr(state, 'active_unit_files', ())) }",
        f"  unit_query_ids: {_list_value(getattr(state, 'active_unit_query_ids', ())) }",
        "objectives:",
        f"  objective_ids: {_list_value(getattr(state, 'objective_ids', ())) }",
        f"  data_law_ids: {_list_value(getattr(state, 'data_law_ids', ())) }",
        f"  patch_scope_verdict: {_value(getattr(state, 'patch_scope_verdict', None))}",
        "latest_execution:",
        "  compile:",
        f"    succeeded: {_value(getattr(state, 'last_compile_succeeded', None))}",
        f"    summary: {_value(getattr(state, 'last_compile_summary', None))}",
        "  run:",
        f"    succeeded: {_value(getattr(state, 'last_run_succeeded', None))}",
        f"    summary: {_value(getattr(state, 'last_run_summary', None))}",
        "  validation:",
        f"    passed: {_value(getattr(state, 'validation_passed', None))}",
        f"    summary: {_value(getattr(state, 'last_validation_summary', None))}",
        f"last_failure_kind: {_value(getattr(state, 'last_failure_kind', None))}",
        f"open_failures: {_list_value(_open_failures(state))}",
        f"written_files: {_list_value(sorted(getattr(state, 'written_files', set())))}",
        f"control_artifacts_read: {_list_value(sorted(getattr(state, 'control_artifacts_read', set())))}",
        f"control_artifacts_injected: {_list_value(getattr(state, 'control_artifacts_injected', ())) }",
        f"required_control_artifacts: {_list_value(getattr(state, 'required_control_artifacts', ())) }",
        f"artifact_refs: {_list_value(_artifact_refs(artifact_refs))}",
        f"q1_q9_obligations: {_list_value(_q1_q9_obligations(state))}",
        f"snapshot_hash: {_value(_snapshot_hash(state, artifact_refs))}",
        f"next_required_action: {_next_required_action(state)}",
    ]
    return "\n".join(lines)


def q1_q9_obligations_for_query_ids(query_ids: Iterable[Any]) -> list[str]:
    """Return hard Q1/Q9 obligations for the active query ids."""
    normalized = {str(item) for item in query_ids}
    return [
        obligation
        for query_id, obligation in Q1_Q9_OBLIGATION_TEXTS.items()
        if query_id in normalized
    ]


def _open_failures(state: Any) -> list[str]:
    """Return current unresolved failure facts in stable order."""
    failures: list[str] = []
    compile_summary = getattr(state, "last_compile_summary", None)
    run_summary = getattr(state, "last_run_summary", None)
    validation_summary = getattr(state, "last_validation_summary", None)
    if getattr(state, "last_compile_succeeded", None) is False:
        failures.append(f"compile:{_plain(compile_summary)}")
    if getattr(state, "last_run_succeeded", None) is False:
        failures.append(f"run:{_plain(run_summary)}")
    if getattr(state, "validation_passed", None) is False:
        failures.append(f"validation:{_plain(validation_summary)}")
    return failures


def _artifact_refs(artifact_refs: str | None) -> list[str]:
    """Extract artifact ids from the bounded refs block."""
    if not artifact_refs:
        return []
    return re.findall(r"artifact_ref=([A-Za-z0-9_.-]+)", artifact_refs)


def _q1_q9_obligations(state: Any) -> list[str]:
    """Return active Q1/Q9 hard obligations for stage-memory carryover."""
    return q1_q9_obligations_for_query_ids(
        tuple(getattr(state, "active_query_ids", ()) or ())
        + tuple(getattr(state, "active_unit_query_ids", ()) or ())
    )


def _snapshot_hash(state: Any, artifact_refs: str | None) -> str:
    """Build a short hash for the current memory inputs."""
    payload = "|".join([
        str(getattr(state, "profile_name", "")),
        str(getattr(state, "prompt_index", "")),
        str(getattr(state, "last_compile_summary", "")),
        str(getattr(state, "last_run_summary", "")),
        str(getattr(state, "last_validation_summary", "")),
        artifact_refs or "",
    ])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _next_required_action(state: Any) -> str:
    """Infer the next required action from the latest stage failure state."""
    failure = getattr(state, "last_failure_kind", None)
    if failure == "compile":
        return "fix the latest compile failure before running again"
    if failure == "run":
        return "inspect the latest run failure or rollback the active unit"
    if getattr(state, "validation_passed", None) is False:
        return "repair validation failure before claiming completion"
    return "continue the current stage objective"


def _value(value: Any) -> str:
    """Render one scalar value with stable null handling."""
    if value is None:
        return "null"
    text = str(value).replace("\n", " ").strip()
    return f'"{text[:500]}"'


def _plain(value: Any) -> str:
    """Render a compact plain-text fact for list fields."""
    if value is None:
        return "null"
    return str(value).replace("\n", " ").strip()[:240]


def _list_value(values: Iterable[Any]) -> str:
    """Render a stable list value without relying on repr of containers."""
    rendered = [f'"{str(item)}"' for item in values]
    return "[" + ", ".join(rendered) + "]"
