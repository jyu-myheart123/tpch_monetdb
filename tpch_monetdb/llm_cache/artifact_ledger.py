from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


DEFAULT_DIGEST_PREVIEW_CHARS = 1200
DEFAULT_REFS_MAX_BYTES = 4_096
DEFAULT_REFS_MAX_ENTRIES = 12
DEFAULT_SCOPE_KEEP_MAX_ENTRIES = 40
DEFAULT_RETENTION_MAX_ARTIFACTS = 2_000
DEFAULT_RETENTION_MAX_TOTAL_BYTES = 512 * 1024 * 1024
RETENTION_MAX_ARTIFACTS_ENV = "TPCH_MONETDB_CONTEXT_ARTIFACT_MAX_COUNT"
RETENTION_MAX_TOTAL_BYTES_ENV = "TPCH_MONETDB_CONTEXT_ARTIFACT_MAX_BYTES"
PINNED_RETENTION_VALUES = {"keep", "permanent", "pinned"}


@dataclass(frozen=True)
class ContextArtifact:
    """Describe one large context artifact stored outside the model session."""

    artifact_id: str
    kind: str
    path: str
    sha256: str
    byte_size: int
    char_count: int
    created_at: str
    stage_name: str | None = None
    prompt_index: int | None = None
    prompt_descriptor: str | None = None
    tool_name: str | None = None
    call_id: str | None = None
    query_ids: tuple[str, ...] = ()
    scale_factor: int | None = None
    snapshot_hash: str | None = None
    success: bool | None = None
    summary: str = ""
    retention: str = "run"
    tags: tuple[str, ...] = ()

    def prompt_ref(self) -> str:
        """Return a pathless stable reference suitable for prompt inclusion."""
        kind = _safe_id_part(self.kind)
        tool = _safe_id_part(self.tool_name or self.kind)
        return f"{kind}_{tool}_{self.sha256[:12]}"

    def ref_line(self) -> str:
        """Render a short retrievable artifact reference for prompt inclusion."""
        query_part = ",".join(self.query_ids) if self.query_ids else "-"
        status = "unknown" if self.success is None else ("success" if self.success else "failure")
        return (
            f"- artifact_ref={self.prompt_ref()} kind={self.kind} tool={self.tool_name or '-'} "
            f"stage={self.stage_name or '-'} queries={query_part} status={status} "
            f"bytes={self.byte_size} summary={_single_line(self.summary)}"
        )


class ArtifactLedger:
    """Persist large evidence blobs and render compact prompt references."""

    def __init__(self, root_dir: Path) -> None:
        self.root_dir = root_dir
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root_dir / "ledger.jsonl"
        self._artifacts: list[ContextArtifact] = self._load_existing_index()
        self._sequence = self._max_sequence(self._artifacts)
        return None

    def record_text(
        self,
        *,
        kind: str,
        text: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> ContextArtifact:
        """Store text as an artifact and append a JSONL ledger record."""
        metadata_dict = dict(metadata or {})
        content = text if text.endswith("\n") else f"{text}\n"
        encoded = content.encode("utf-8")
        sha256 = hashlib.sha256(encoded).hexdigest()
        self._sequence += 1
        artifact_id = self._build_artifact_id(kind, metadata_dict, sha256)
        path = self.root_dir / f"{artifact_id}.txt"
        path.write_bytes(encoded)
        artifact = ContextArtifact(
            artifact_id=artifact_id,
            kind=kind,
            path=path.as_posix(),
            sha256=sha256,
            byte_size=len(encoded),
            char_count=len(content),
            created_at=datetime.now(timezone.utc).isoformat(),
            stage_name=_optional_str(metadata_dict.get("stage_name")),
            prompt_index=_optional_int(metadata_dict.get("prompt_index")),
            prompt_descriptor=_optional_str(metadata_dict.get("prompt_descriptor")),
            tool_name=_optional_str(metadata_dict.get("tool_name")),
            call_id=_optional_str(metadata_dict.get("call_id")),
            query_ids=_tuple_of_str(metadata_dict.get("query_ids")),
            scale_factor=_optional_int(metadata_dict.get("scale_factor")),
            snapshot_hash=_optional_str(metadata_dict.get("snapshot_hash")),
            success=_optional_bool(metadata_dict.get("success")),
            summary=_optional_str(metadata_dict.get("summary")) or _summarize_text(content),
            retention=_optional_str(metadata_dict.get("retention")) or "run",
            tags=_tuple_of_str(metadata_dict.get("tags")),
        )
        self._artifacts.append(artifact)
        self._append_index_record(artifact)
        return artifact

    def record_file(
        self,
        *,
        path: Path | str,
        kind: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> ContextArtifact:
        """Store an existing file as a ledger artifact and append its index record."""
        source_path = Path(path)
        if not source_path.exists():
            raise FileNotFoundError(source_path)
        metadata_dict = dict(metadata or {})
        content = source_path.read_bytes()
        sha256 = hashlib.sha256(content).hexdigest()
        self._sequence += 1
        artifact_id = self._build_artifact_id(kind, metadata_dict, sha256)
        suffix = source_path.suffix or ".bin"
        target_path = self.root_dir / f"{artifact_id}{suffix}"
        if source_path.resolve() != target_path.resolve():
            shutil.copyfile(source_path, target_path)
        text = content.decode("utf-8", errors="replace")
        artifact = ContextArtifact(
            artifact_id=artifact_id,
            kind=kind,
            path=target_path.as_posix(),
            sha256=sha256,
            byte_size=len(content),
            char_count=len(text),
            created_at=datetime.now(timezone.utc).isoformat(),
            stage_name=_optional_str(metadata_dict.get("stage_name")),
            prompt_index=_optional_int(metadata_dict.get("prompt_index")),
            prompt_descriptor=_optional_str(metadata_dict.get("prompt_descriptor")),
            tool_name=_optional_str(metadata_dict.get("tool_name")),
            call_id=_optional_str(metadata_dict.get("call_id")),
            query_ids=_tuple_of_str(metadata_dict.get("query_ids")),
            scale_factor=_optional_int(metadata_dict.get("scale_factor")),
            snapshot_hash=_optional_str(metadata_dict.get("snapshot_hash")),
            success=_optional_bool(metadata_dict.get("success")),
            summary=_optional_str(metadata_dict.get("summary")) or _summarize_text(text),
            retention=_optional_str(metadata_dict.get("retention")) or "run",
            tags=_tuple_of_str(metadata_dict.get("tags")),
        )
        self._artifacts.append(artifact)
        self._append_index_record(artifact)
        return artifact

    def render_digest(
        self,
        artifact: ContextArtifact,
        *,
        preview: str,
        omitted_chars: int,
    ) -> str:
        """Render a compact evidence digest suitable for a tool response."""
        lines = []
        if preview.strip():
            lines.append(preview.strip())
        lines.extend([
            "[Evidence Digest]",
            f"artifact_ref: {artifact.prompt_ref()}",
            f"kind: {artifact.kind}",
            f"tool: {artifact.tool_name or '-'}",
            f"stage: {artifact.stage_name or '-'}",
            f"query_ids: {json.dumps(list(artifact.query_ids), ensure_ascii=False)}",
            f"success: {artifact.success}",
            f"sha256: {artifact.sha256}",
            f"bytes: {artifact.byte_size}",
            f"summary: {_single_line(artifact.summary)}",
        ])
        if omitted_chars > 0:
            lines.append(f"omitted_chars: {omitted_chars}")
        lines.append("read_more: use read_artifact with artifact_ref and offset/limit")
        return "\n".join(lines)

    def refs_for_prompt(
        self,
        *,
        max_entries: int = DEFAULT_REFS_MAX_ENTRIES,
        max_bytes: int = DEFAULT_REFS_MAX_BYTES,
        query_ids: Sequence[str] = (),
        stage_name: str | None = None,
    ) -> str:
        """Render a bounded list of artifact references without full artifact metadata."""
        candidates = self._select_scope_artifacts(query_ids=query_ids, stage_name=stage_name)
        lines = ["[Artifact Refs]"]
        encoded_size = len(lines[0].encode("utf-8"))
        emitted = 0
        for artifact in candidates[:max_entries]:
            line = artifact.ref_line()
            next_size = encoded_size + len(line.encode("utf-8")) + 1
            if next_size > max_bytes:
                break
            lines.append(line)
            encoded_size = next_size
            emitted += 1
        if emitted == 0:
            return "[Artifact Refs]\n(no artifacts recorded)"
        remaining = max(0, len(candidates) - emitted)
        if remaining:
            lines.append(f"... {remaining} artifact(s) omitted by refs budget")
        lines.append("read_more: use read_artifact with artifact_ref and offset/limit")
        return "\n".join(lines)

    def artifact_ids_for_scope(
        self,
        *,
        max_entries: int = DEFAULT_SCOPE_KEEP_MAX_ENTRIES,
        query_ids: Sequence[str] = (),
        stage_name: str | None = None,
    ) -> tuple[str, ...]:
        """Return artifact ids selected by active stage/query relevance."""
        candidates = self._select_scope_artifacts(
            query_ids=query_ids,
            stage_name=stage_name,
        )
        return tuple(artifact.artifact_id for artifact in candidates[:max_entries])

    def artifact_ids_for_refs(self, artifact_refs: Sequence[str]) -> tuple[str, ...]:
        """Return artifact ids addressed by stable prompt refs."""
        ref_set = {str(ref) for ref in artifact_refs}
        resolved: list[str] = []
        for artifact in self._artifacts:
            if artifact.prompt_ref() in ref_set:
                resolved.append(artifact.artifact_id)
        return tuple(dict.fromkeys(resolved))

    def artifacts(self) -> tuple[ContextArtifact, ...]:
        """Return artifacts recorded by this ledger instance."""
        return tuple(self._artifacts)

    def lookup(self, artifact_id: str) -> ContextArtifact:
        """Return one artifact by id or raise KeyError when it is unknown."""
        for artifact in reversed(self._artifacts):
            if artifact.artifact_id == artifact_id:
                return artifact
        raise KeyError(artifact_id)

    def lookup_ref(self, artifact_ref: str) -> ContextArtifact:
        """Return one artifact by stable prompt ref."""
        for artifact in reversed(self._artifacts):
            if artifact.prompt_ref() == artifact_ref:
                return artifact
        raise KeyError(artifact_ref)

    def top_contributors(
        self,
        *,
        limit: int = 5,
        query_ids: Sequence[str] = (),
        stage_name: str | None = None,
    ) -> tuple[ContextArtifact, ...]:
        """Return the largest relevant artifacts for diagnostics."""
        if limit <= 0:
            return ()
        candidates = self._select_scope_artifacts(
            query_ids=query_ids,
            stage_name=stage_name,
        )
        ordered = sorted(candidates, key=lambda item: item.byte_size, reverse=True)
        return tuple(ordered[:limit])

    def cleanup_retention(
        self,
        *,
        keep_artifact_ids: Sequence[str] = (),
        max_artifacts: int | None = None,
        max_total_bytes: int | None = None,
    ) -> tuple[ContextArtifact, ...]:
        """Prune old run-retention artifacts while preserving active scope ids."""
        if max_artifacts is None and max_total_bytes is None:
            return ()
        keep_ids = {str(item) for item in keep_artifact_ids}
        retained = list(self._artifacts)
        pruned: list[ContextArtifact] = []

        def over_limits() -> bool:
            count_over = max_artifacts is not None and len(retained) > max_artifacts
            bytes_over = (
                max_total_bytes is not None
                and sum(artifact.byte_size for artifact in retained) > max_total_bytes
            )
            return count_over or bytes_over

        candidates = [
            artifact
            for artifact in retained
            if artifact.artifact_id not in keep_ids
            and artifact.retention not in PINNED_RETENTION_VALUES
        ]
        candidates.sort(key=lambda artifact: artifact.created_at)

        for artifact in candidates:
            if not over_limits():
                break
            retained.remove(artifact)
            pruned.append(artifact)
            self._delete_artifact_file(artifact)

        if pruned:
            self._artifacts = retained
            self._rewrite_index()
        return tuple(pruned)

    def cleanup_default_retention(
        self,
        *,
        keep_artifact_ids: Sequence[str] = (),
    ) -> tuple[ContextArtifact, ...]:
        """Apply the configured artifact retention limits with explicit keep ids."""
        return self.cleanup_retention(
            keep_artifact_ids=keep_artifact_ids,
            max_artifacts=_retention_limit(
                RETENTION_MAX_ARTIFACTS_ENV,
                DEFAULT_RETENTION_MAX_ARTIFACTS,
            ),
            max_total_bytes=_retention_limit(
                RETENTION_MAX_TOTAL_BYTES_ENV,
                DEFAULT_RETENTION_MAX_TOTAL_BYTES,
            ),
        )

    def _load_existing_index(self) -> list[ContextArtifact]:
        """Load previously recorded artifacts so resumed runs keep their ledger view."""
        if not self.index_path.exists():
            return []
        artifacts: list[ContextArtifact] = []
        with self.index_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    payload = json.loads(stripped)
                    artifacts.append(_artifact_from_payload(payload))
                except (TypeError, ValueError, KeyError):
                    continue
        return artifacts

    def _max_sequence(self, artifacts: Sequence[ContextArtifact]) -> int:
        """Return the largest numeric id prefix already present in the ledger."""
        max_seen = 0
        for artifact in artifacts:
            match = re.match(r"^(?P<seq>\d+)_", artifact.artifact_id)
            if match is None:
                continue
            max_seen = max(max_seen, int(match.group("seq")))
        return max_seen

    def _build_artifact_id(
        self,
        kind: str,
        metadata: Mapping[str, Any],
        sha256: str,
    ) -> str:
        """Build a deterministic-ish artifact id with a sequence suffix for uniqueness."""
        stage = _safe_id_part(_optional_str(metadata.get("stage_name")) or "stage")
        tool = _safe_id_part(_optional_str(metadata.get("tool_name")) or kind)
        return f"{self._sequence:05d}_{stage}_{tool}_{sha256[:12]}"

    def _append_index_record(self, artifact: ContextArtifact) -> None:
        """Append one artifact record to the JSONL index."""
        with self.index_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(artifact), ensure_ascii=False, sort_keys=True))
            handle.write("\n")
        return None

    def _rewrite_index(self) -> None:
        """Rewrite the JSONL ledger after retention cleanup."""
        with self.index_path.open("w", encoding="utf-8") as handle:
            for artifact in self._artifacts:
                handle.write(json.dumps(asdict(artifact), ensure_ascii=False, sort_keys=True))
                handle.write("\n")
        return None

    def _delete_artifact_file(self, artifact: ContextArtifact) -> None:
        """Delete an artifact payload file if it still lives under the ledger root."""
        path = Path(artifact.path)
        try:
            path.resolve().relative_to(self.root_dir.resolve())
        except ValueError:
            return None
        path.unlink(missing_ok=True)
        return None

    def _select_scope_artifacts(
        self,
        *,
        query_ids: Sequence[str],
        stage_name: str | None,
    ) -> list[ContextArtifact]:
        """Sort artifacts so active stage/query evidence appears first."""
        query_set = {str(item) for item in query_ids}

        def score(artifact: ContextArtifact) -> tuple[int, int, int, int, int]:
            stage_score = 0 if stage_name and artifact.stage_name == stage_name else 1
            query_score = 0 if query_set.intersection(artifact.query_ids) else 1
            failure_score = 0 if artifact.success is False else 1
            recency_score = _artifact_recency_score(artifact)
            return (
                stage_score,
                query_score,
                failure_score,
                -recency_score,
                -artifact.byte_size,
            )

        return sorted(self._artifacts, key=score)


def build_preview(text: str, limit: int = DEFAULT_DIGEST_PREVIEW_CHARS) -> tuple[str, int]:
    """Return a head/tail preview and omitted character count."""
    stripped = text.strip()
    if len(stripped) <= limit:
        return stripped, 0
    keep = max(256, (limit - 80) // 2)
    omitted = max(0, len(stripped) - (2 * keep))
    preview = (
        f"{stripped[:keep]}\n"
        f"... [{omitted} chars truncated into artifact] ...\n"
        f"{stripped[-keep:]}"
    )
    return preview, omitted


def _summarize_text(text: str, max_chars: int = 240) -> str:
    """Summarize text with a stable first-line-oriented fallback."""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped[:max_chars]
    return "(empty artifact)"


def _single_line(value: str) -> str:
    """Collapse a string into a single prompt-safe line."""
    return re.sub(r"\s+", " ", value.strip())


def _safe_id_part(value: str) -> str:
    """Return a filesystem-safe artifact id component."""
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return cleaned[:48] or "unknown"


def _artifact_recency_score(artifact: ContextArtifact) -> int:
    """Return a sortable recency score using prompt index and ledger sequence."""
    sequence_match = re.match(r"^(?P<seq>\d+)_", artifact.artifact_id)
    sequence = int(sequence_match.group("seq")) if sequence_match is not None else 0
    prompt_index = artifact.prompt_index if artifact.prompt_index is not None else 0
    return max(prompt_index, sequence)


def _optional_str(value: Any) -> str | None:
    """Normalize optional string metadata."""
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _optional_int(value: Any) -> int | None:
    """Normalize optional integer metadata."""
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_bool(value: Any) -> bool | None:
    """Normalize optional boolean metadata."""
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.lower()
        if lowered in {"true", "1", "yes"}:
            return True
        if lowered in {"false", "0", "no"}:
            return False
    return None


def _tuple_of_str(value: Any) -> tuple[str, ...]:
    """Normalize metadata sequence values to a tuple of strings."""
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Sequence):
        return tuple(str(item) for item in value)
    return (str(value),)


def _artifact_from_payload(payload: Mapping[str, Any]) -> ContextArtifact:
    """Deserialize one JSONL ledger record into a ContextArtifact."""
    return ContextArtifact(
        artifact_id=str(payload["artifact_id"]),
        kind=str(payload["kind"]),
        path=str(payload["path"]),
        sha256=str(payload["sha256"]),
        byte_size=int(payload["byte_size"]),
        char_count=int(payload.get("char_count", 0)),
        created_at=str(payload["created_at"]),
        stage_name=_optional_str(payload.get("stage_name")),
        prompt_index=_optional_int(payload.get("prompt_index")),
        prompt_descriptor=_optional_str(payload.get("prompt_descriptor")),
        tool_name=_optional_str(payload.get("tool_name")),
        call_id=_optional_str(payload.get("call_id")),
        query_ids=_tuple_of_str(payload.get("query_ids")),
        scale_factor=_optional_int(payload.get("scale_factor")),
        snapshot_hash=_optional_str(payload.get("snapshot_hash")),
        success=_optional_bool(payload.get("success")),
        summary=_optional_str(payload.get("summary")) or "",
        retention=_optional_str(payload.get("retention")) or "run",
        tags=_tuple_of_str(payload.get("tags")),
    )


def _retention_limit(env_name: str, default: int) -> int | None:
    """Read a positive retention limit; zero disables that limit."""
    raw = os.environ.get(env_name)
    if raw is None:
        return default
    try:
        parsed = int(raw)
    except ValueError:
        return default
    if parsed <= 0:
        return None
    return parsed
