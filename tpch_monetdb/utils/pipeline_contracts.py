from __future__ import annotations

from dataclasses import dataclass

HOST_SEALED_MANIFEST_TRUST_MODE = "host_sealed_read_only"
REQUIRED_RESUME_SNAPSHOT_FIELDS: tuple[str, ...] = (
    "implementation_manifest_sha256",
    "storage_plan_sha256",
    "todo_sha256",
    "todo_reconciliation",
    "control_artifact_hashes",
)


@dataclass(frozen=True)
class PipelineContractError(RuntimeError):
    """Structured runtime error for contract-gated pipeline failures."""

    code: str
    message: str
    stage: str | None = None

    def __str__(self) -> str:
        """Render the error in a stable, debuggable format."""
        stage_suffix = "" if self.stage is None else f" stage={self.stage}"
        return f"[ERROR:{self.code}]{stage_suffix} {self.message}"


def raise_pipeline_contract_error(
    *,
    code: str,
    message: str,
    stage: str | None = None,
) -> None:
    """Raise a structured pipeline contract error with a stable message."""
    raise PipelineContractError(code=code, message=message, stage=stage)
