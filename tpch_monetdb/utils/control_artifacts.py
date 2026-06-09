from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tpch_monetdb.tools.stage_tool_policy import TodoState
from tpch_monetdb.utils.large_data_objectives import (
    DATA_LAW_CONTRACT_FILE,
    STORAGE_PLAN_ALIGNMENT_FILE,
    STORAGE_PLAN_CONTRACT_FILE,
    WORKLOAD_OBJECTIVE_FILE,
    build_storage_plan_alignment as build_large_data_storage_plan_alignment,
    load_json_contract,
    write_storage_plan_alignment,
)
from tpch_monetdb.utils.pipeline_contracts import raise_pipeline_contract_error


TRACKED_CONTROL_ARTIFACTS: tuple[str, ...] = (
    WORKLOAD_OBJECTIVE_FILE,
    DATA_LAW_CONTRACT_FILE,
    STORAGE_PLAN_CONTRACT_FILE,
    STORAGE_PLAN_ALIGNMENT_FILE,
    "storage_plan.txt",
    "TODO.md",
    "optimization_hotspot_summary.md",
    "design_evidence.md",
    "implementation_manifest.json",
)


@dataclass(frozen=True)
class ControlArtifact:
    """One tracked planning/control artifact carried across stages."""

    artifact_id: str
    relative_path: str
    sha256: str | None


@dataclass(frozen=True)
class ControlArtifactEnvelope:
    """Host-owned audit envelope for tracked control artifacts."""

    artifact_hashes: dict[str, str] = field(default_factory=dict)
    artifacts: tuple[ControlArtifact, ...] = ()


def sha256_for_file(path: Path) -> str | None:
    """Return the SHA-256 hex digest for a file, or None when missing."""
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_control_artifact_hashes(workspace_path: Path) -> dict[str, str]:
    """Collect hashes for tracked control artifacts present in the workspace."""
    hashes: dict[str, str] = {}
    for relative_path in TRACKED_CONTROL_ARTIFACTS:
        digest = sha256_for_file(workspace_path / relative_path)
        if digest is not None:
            hashes[relative_path] = digest
    return hashes


def build_control_artifact_envelope(workspace_path: Path) -> ControlArtifactEnvelope:
    """Build an audit envelope from the tracked artifact set."""
    hashes = collect_control_artifact_hashes(workspace_path)
    artifacts = tuple(
        ControlArtifact(
            artifact_id=relative_path,
            relative_path=relative_path,
            sha256=hashes.get(relative_path),
        )
        for relative_path in TRACKED_CONTROL_ARTIFACTS
    )
    return ControlArtifactEnvelope(
        artifact_hashes=hashes,
        artifacts=artifacts,
    )


def build_todo_reconciliation(todo_path: Path) -> dict[str, Any]:
    """Summarize TODO status and expose semantic items for host-side gates."""
    state = TodoState.from_file(todo_path)
    if state is None:
        return {
            "status": "missing",
            "completed_count": 0,
            "in_progress_count": 0,
            "pending_count": 0,
            "semantic_items": {},
        }
    semantic_items = {
        item.active_form: {
            "content": item.content,
            "status": item.status,
            "category": _classify_todo_semantic_category(item.content),
        }
        for item in state.items
    }
    return {
        "status": "present",
        "completed_count": state.completed_count,
        "in_progress_count": state.in_progress_count,
        "pending_count": state.pending_count,
        "semantic_items": semantic_items,
    }


def _classify_todo_semantic_category(content: str) -> str:
    """Classify TODO text into a coarse objective category for evidence ledgers."""
    normalized = content.lower()
    if any(token in normalized for token in ("vector", "simd", "avx", "sse")):
        return "vectorization"
    if any(token in normalized for token in ("pmu", "perf", "counter", "hotspot")):
        return "instrumentation"
    if any(token in normalized for token in ("storage", "layout", "columnar", "contract")):
        return "storage_plan"
    if any(token in normalized for token in ("correct", "oracle", "validation")):
        return "correctness"
    return "general"


def _load_workspace_query_ids(workspace_path: Path) -> list[str]:
    """Load objective query IDs for host-side alignment evaluation."""
    objective = load_json_contract(workspace_path, WORKLOAD_OBJECTIVE_FILE)
    query_ids = objective.get("query_ids")
    if isinstance(query_ids, list):
        return [str(query_id) for query_id in query_ids]
    return []


def build_storage_plan_alignment(storage_plan_path: Path) -> dict[str, Any]:
    """Evaluate storage-plan alignment using host-owned contracts."""
    if not storage_plan_path.exists():
        return {"status": "missing", "departures": []}
    workspace_path = storage_plan_path.parent
    query_ids = _load_workspace_query_ids(workspace_path)
    if not query_ids:
        return {
            "status": "objective_missing",
            "departures": [f"{WORKLOAD_OBJECTIVE_FILE} missing or has no query_ids"],
        }
    return build_large_data_storage_plan_alignment(
        workspace_path,
        query_ids=query_ids,
    )


def ensure_required_control_artifacts_present(
    workspace_path: Path,
    required_artifacts: tuple[str, ...],
    *,
    stage: str | None = None,
) -> None:
    """Require each declared control artifact to exist before the stage runs."""
    missing = [
        relative_path
        for relative_path in required_artifacts
        if not (workspace_path / relative_path).exists()
    ]
    if missing:
        raise_pipeline_contract_error(
            code="CONTROL_ARTIFACT_MISSING",
            message="Missing required control artifacts: " + ", ".join(sorted(missing)),
            stage=stage,
        )
    return None


def ensure_required_control_artifacts_acknowledged(
    required_artifacts: tuple[str, ...],
    *,
    read_artifacts: set[str],
    injected_artifacts: tuple[str, ...],
    action: str,
    stage: str | None = None,
) -> None:
    """Require each gated artifact to be read or host-injected before action."""
    acknowledged = set(read_artifacts) | set(injected_artifacts)
    missing = [
        artifact for artifact in required_artifacts if artifact not in acknowledged
    ]
    if missing:
        raise_pipeline_contract_error(
            code="CONTROL_ARTIFACT_NOT_ACKNOWLEDGED",
            message=(
                f"Action {action} requires acknowledged control artifacts: "
                + ", ".join(sorted(missing))
            ),
            stage=stage,
        )
    return None


def write_control_artifact_audit_copy(workspace_path: Path) -> Path:
    """Write the host-owned control-artifact audit copy into the workspace."""
    query_ids = _load_workspace_query_ids(workspace_path)
    if query_ids:
        write_storage_plan_alignment(workspace_path, query_ids=query_ids)
    envelope = build_control_artifact_envelope(workspace_path)
    payload = {
        "artifact_hashes": envelope.artifact_hashes,
        "artifacts": [
            {
                "artifact_id": artifact.artifact_id,
                "relative_path": artifact.relative_path,
                "sha256": artifact.sha256,
            }
            for artifact in envelope.artifacts
        ],
        "todo_reconciliation": build_todo_reconciliation(workspace_path / "TODO.md"),
        "storage_plan_alignment": build_storage_plan_alignment(
            workspace_path / "storage_plan.txt"
        ),
    }
    target_path = workspace_path / "control_artifacts.json"
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return target_path
