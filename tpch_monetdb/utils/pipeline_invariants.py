from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from tpch_monetdb.utils.pipeline_contracts import (
    REQUIRED_RESUME_SNAPSHOT_FIELDS,
    raise_pipeline_contract_error,
)


def require_mapping_keys(
    mapping: Mapping[str, Any],
    *,
    required_keys: Iterable[str],
    code: str,
    stage: str | None = None,
) -> None:
    """Require a mapping to contain all required keys before continuing."""
    missing = [key for key in required_keys if key not in mapping]
    if missing:
        raise_pipeline_contract_error(
            code=code,
            message=f"Missing required keys: {', '.join(missing)}",
            stage=stage,
        )
    return None


def require_nonempty_value(
    value: Any,
    *,
    code: str,
    field_name: str,
    stage: str | None = None,
) -> None:
    """Require a value to be present and non-empty for a gated contract."""
    if value in (None, "", (), [], {}, set()):
        raise_pipeline_contract_error(
            code=code,
            message=f"Required field is empty: {field_name}",
            stage=stage,
        )
    return None


def require_resume_snapshot_fields(
    snapshot_fields: Mapping[str, Any],
    *,
    stage: str | None = None,
) -> None:
    """Require all new-contract snapshot fields before resume is allowed."""
    require_mapping_keys(
        snapshot_fields,
        required_keys=REQUIRED_RESUME_SNAPSHOT_FIELDS,
        code="RESUME_SNAPSHOT_INCOMPLETE",
        stage=stage,
    )
    for field_name in REQUIRED_RESUME_SNAPSHOT_FIELDS:
        require_nonempty_value(
            snapshot_fields.get(field_name),
            code="RESUME_SNAPSHOT_INCOMPLETE",
            field_name=field_name,
            stage=stage,
        )
    return None
