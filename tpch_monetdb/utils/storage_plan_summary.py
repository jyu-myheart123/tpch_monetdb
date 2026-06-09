"""Storage Plan Run Summary 管理工具."""

import dataclasses
import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from tpch_monetdb.config import DEFAULT_TPCH_MONETDB_ARTIFACTS_DIR

logger = logging.getLogger(__name__)


@dataclass
class StoragePlanRunSummary:
    """Storage plan run 成功摘要."""

    benchmark: str
    conv_name: str
    run_id: str
    query_list: list[str]
    final_snapshot_hash: str
    storage_plan_path: str
    completed_at: str
    conversation_json: str
    session_db_path: str
    success: bool
    storage_plan_excerpt: str = ""   # first 2000 chars of storage_plan.txt for prompt injection
    storage_plan_sha256: str | None = None
    storage_plan_size_bytes: int = 0
    wandb_primary_run_id: str | None = None
    wandb_final_run_id: str | None = None
    wandb_init_attempt_count: int = 0
    wandb_attempted_run_ids: list[str] = field(default_factory=list)
    wandb_retry_used: bool = False
    wandb_first_failure_excerpt: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "StoragePlanRunSummary":
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


def get_summary_dir(artifacts_dir: Path) -> Path:
    return artifacts_dir / "storage_plan_runs"


def _load_summary_file(file_path: Path) -> StoragePlanRunSummary:
    with open(file_path) as f:
        data = json.load(f)
    return StoragePlanRunSummary.from_dict(data)


def _load_latest_summary(conv_dir: Path) -> StoragePlanRunSummary | None:
    latest_path = conv_dir / "latest.json"
    if not latest_path.exists():
        return None
    with open(latest_path) as f:
        latest_data = json.load(f)
    embedded_summary = latest_data.get("summary")
    if isinstance(embedded_summary, dict):
        return StoragePlanRunSummary.from_dict(embedded_summary)
    latest_file = latest_data.get("latest_file")
    if not isinstance(latest_file, str) or not latest_file:
        raise KeyError("latest.json missing latest_file")
    return _load_summary_file(conv_dir / latest_file)


def write_storage_plan_run_summary(
    summary: StoragePlanRunSummary,
    artifacts_dir: Path,
) -> Path:
    if not summary.final_snapshot_hash:
        raise ValueError("final_snapshot_hash cannot be empty")

    summary_dir = get_summary_dir(artifacts_dir)
    conv_dir = summary_dir / summary.conv_name
    conv_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{summary.run_id}.json"

    file_path = conv_dir / filename
    with open(file_path, "w") as f:
        json.dump(summary.to_dict(), f, indent=2)

    latest_path = conv_dir / "latest.json"
    with open(latest_path, "w") as f:
        json.dump({
            "latest_file": filename,
            "summary": summary.to_dict(),
        }, f, indent=2)

    logger.info(f"Written storage plan run summary to {file_path}")
    return file_path


_EXCERPT_MAX_CHARS = 2000


def persist_successful_storage_plan_run(
    *,
    benchmark: str,
    conv_name: str,
    query_list: list[str],
    final_snapshot_hash: str,
    storage_plan_path: Path,
    conversation_json_path: Path,
    session_db_path: Path,
    artifacts_dir: Path,
    wandb_result: "Optional[Any]" = None,
) -> Path:
    # Read an excerpt of the plan for round-to-round injection
    plan_text = ""
    plan_sha256: str | None = None
    plan_size_bytes = 0
    if storage_plan_path.exists():
        try:
            plan_bytes = storage_plan_path.read_bytes()
            plan_text = plan_bytes.decode("utf-8")[:_EXCERPT_MAX_CHARS]
            plan_sha256 = hashlib.sha256(plan_bytes).hexdigest()
            plan_size_bytes = len(plan_bytes)
        except OSError:
            pass

    summary = StoragePlanRunSummary(
        benchmark=benchmark,
        conv_name=conv_name,
        run_id=conv_name,
        query_list=query_list,
        final_snapshot_hash=final_snapshot_hash,
        storage_plan_path=str(storage_plan_path),
        completed_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        conversation_json=str(conversation_json_path),
        session_db_path=str(session_db_path),
        success=True,
        storage_plan_excerpt=plan_text,
        storage_plan_sha256=plan_sha256,
        storage_plan_size_bytes=plan_size_bytes,
        wandb_primary_run_id=wandb_result.primary_run_id if wandb_result is not None else None,
        wandb_final_run_id=wandb_result.final_run_id if wandb_result is not None else None,
        wandb_init_attempt_count=wandb_result.attempt_count if wandb_result is not None else 0,
        wandb_attempted_run_ids=wandb_result.attempted_run_ids if wandb_result is not None else [],
        wandb_retry_used=wandb_result.used_fallback if wandb_result is not None else False,
        wandb_first_failure_excerpt=wandb_result.first_failure_excerpt if wandb_result is not None else None,
    )
    file_path = write_storage_plan_run_summary(summary, artifacts_dir)
    logger.info(f"Storage plan run completed successfully. Summary written to {file_path}")
    return file_path


def find_latest_successful_storage_plan_run(
    conv_name: str | None,
    query_list: list[str],
    benchmark: str = "tpch",
    artifacts_dir: Path | None = None,
) -> Optional[StoragePlanRunSummary]:
    if artifacts_dir is None:
        artifacts_dir = Path(DEFAULT_TPCH_MONETDB_ARTIFACTS_DIR)

    summary_dir = get_summary_dir(artifacts_dir)
    if conv_name is None:
        conv_dirs = [path for path in summary_dir.iterdir() if path.is_dir()] if summary_dir.exists() else []
    else:
        conv_dirs = [summary_dir / conv_name]

    if not conv_dirs:
        logger.info(f"No storage plan run directory found for {conv_name}")
        return None

    if conv_name is not None:
        conv_dir = conv_dirs[0]
        if conv_dir.exists():
            try:
                latest_summary = _load_latest_summary(conv_dir)
                if latest_summary is not None and latest_summary.success and latest_summary.benchmark == benchmark and set(latest_summary.query_list) == set(query_list):
                    return latest_summary
            except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                logger.warning(f"Failed to read latest summary pointer for {conv_name}: {exc}")

    summaries = []
    for conv_dir in conv_dirs:
        if not conv_dir.exists():
            continue
        for file_path in conv_dir.glob("*.json"):
            if file_path.name == "latest.json":
                continue
            try:
                summary = _load_summary_file(file_path)
                if summary.success and summary.benchmark == benchmark and set(summary.query_list) == set(query_list):
                    summaries.append(summary)
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                logger.warning(f"Failed to parse summary file {file_path}: {e}")
                continue

    if not summaries:
        logger.info(f"No compatible storage plan run found for {conv_name}")
        return None

    summaries.sort(key=lambda s: s.completed_at, reverse=True)
    latest = summaries[0]
    logger.info(f"Found latest storage plan run: {latest.conv_name} at {latest.completed_at}")
    return latest
