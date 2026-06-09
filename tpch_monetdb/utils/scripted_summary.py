"""Scripted Success Summary 管理工具.

提供 scripted run 成功后的摘要记录和 optimization 自动发现功能。
"""

import dataclasses
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from tpch_monetdb.config import DEFAULT_TPCH_MONETDB_ARTIFACTS_DIR
from tpch_monetdb.utils.control_artifacts import (
    build_storage_plan_alignment,
    build_todo_reconciliation,
    collect_control_artifact_hashes,
)

logger = logging.getLogger(__name__)

PROMOTABLE_STORAGE_PLAN_ALIGNMENT_STATUSES = frozenset({"contract_valid", "aligned"})


@dataclass
class ScriptedRunSummary:
    """Scripted run 成功摘要."""
    
    benchmark: str
    conv_name: str
    run_id: str
    query_list: list[str]
    is_bespoke_storage: bool
    final_snapshot_hash: str
    completed_at: str
    conversation_json: str
    session_db_path: str
    success: bool
    validation_mode: str = "strict"
    control_artifact_hashes: dict[str, str] = field(default_factory=dict)
    control_artifacts_read_by_stage: dict[str, list[str]] = field(default_factory=dict)
    control_artifacts_injected_by_stage: dict[str, list[str]] = field(default_factory=dict)
    storage_plan_sha256: str | None = None
    todo_sha256: str | None = None
    design_evidence_sha256: str | None = None
    implementation_manifest_sha256: str | None = None
    todo_reconciliation: dict[str, Any] = field(default_factory=dict)
    storage_plan_alignment: dict[str, Any] = field(default_factory=dict)
    wandb_primary_run_id: str | None = None
    wandb_final_run_id: str | None = None
    wandb_init_attempt_count: int = 0
    wandb_attempted_run_ids: list[str] = field(default_factory=list)
    wandb_retry_used: bool = False
    wandb_first_failure_excerpt: str | None = None

    def to_dict(self) -> dict:
        """转换为字典."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ScriptedRunSummary":
        """从字典创建（向后兼容旧格式）."""
        known = {f.name for f in dataclasses.fields(cls)}
        payload = {k: v for k, v in data.items() if k in known}
        payload["is_bespoke_storage"] = True
        return cls(**payload)


def get_summary_dir(artifacts_dir: Path) -> Path:
    """获取 summary 存储目录."""
    return artifacts_dir / "scripted_runs"


def _is_compatible_summary(
    summary: ScriptedRunSummary,
    query_list: list[str],
    benchmark: str,
    validation_mode: str | None,
    is_bespoke_storage: bool | None = None,
) -> bool:
    _ = is_bespoke_storage
    if not summary.success:
        return False
    if summary.benchmark != benchmark:
        return False
    if set(summary.query_list) != set(query_list):
        return False
    if validation_mode is not None and summary.validation_mode != validation_mode:
        return False
    return True


def _load_summary_file(file_path: Path) -> ScriptedRunSummary:
    with open(file_path) as f:
        data = json.load(f)
    return ScriptedRunSummary.from_dict(data)


def _load_latest_summary(conv_dir: Path) -> ScriptedRunSummary | None:
    latest_path = conv_dir / "latest.json"
    if not latest_path.exists():
        return None
    with open(latest_path) as f:
        latest_data = json.load(f)
    embedded_summary = latest_data.get("summary")
    if isinstance(embedded_summary, dict):
        return ScriptedRunSummary.from_dict(embedded_summary)
    latest_file = latest_data.get("latest_file")
    if not isinstance(latest_file, str) or not latest_file:
        raise KeyError("latest.json missing latest_file")
    return _load_summary_file(conv_dir / latest_file)


def write_scripted_run_summary(
    summary: ScriptedRunSummary,
    artifacts_dir: Path,
) -> Path:
    """写入 scripted run 成功摘要.
    
    Args:
        summary: 摘要数据
        artifacts_dir: artifacts 根目录
        
    Returns:
        写入的文件路径
        
    Raises:
        ValueError: 如果 final_snapshot_hash 为空
    """
    if not summary.final_snapshot_hash:
        raise ValueError("final_snapshot_hash cannot be empty")
    summary.is_bespoke_storage = True
    
    summary_dir = get_summary_dir(artifacts_dir)
    conv_dir = summary_dir / summary.conv_name
    conv_dir.mkdir(parents=True, exist_ok=True)
    
    # 使用时间戳和 run_id 创建唯一文件名
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    filename = f"{timestamp}_{summary.run_id}.json"
    
    file_path = conv_dir / filename
    with open(file_path, "w") as f:
        json.dump(summary.to_dict(), f, indent=2)
    
    # 更新 latest.json 指针
    latest_path = conv_dir / "latest.json"
    with open(latest_path, "w") as f:
        json.dump({
            "latest_file": filename,
            "summary": summary.to_dict(),
        }, f, indent=2)
    
    logger.info(f"Written scripted run summary to {file_path}")
    return file_path


def require_promotable_storage_plan_alignment(alignment: dict[str, Any]) -> None:
    """Reject base handoff summaries whose storage plan is not promotable."""
    status = str(alignment.get("status") or "")
    if status in PROMOTABLE_STORAGE_PLAN_ALIGNMENT_STATUSES:
        return None
    departures = alignment.get("departures") or alignment.get("missing_query_ids") or []
    raise ValueError(
        "storage_plan_alignment is not promotable: "
        f"status={status or 'missing'}, detail={departures}"
    )


def persist_successful_scripted_run(
    *,
    benchmark: str,
    conv_name: str,
    query_list: list[str],
    is_bespoke_storage: bool,
    final_snapshot_hash: str,
    conversation_json_path: Path,
    session_db_path: Path,
    artifacts_dir: Path,
    validation_mode: str,
    workspace_path: Path | None = None,
    stage_summaries: list[Any] | None = None,
    wandb_result: "Optional[Any]" = None,
    ) -> Path:
    """写入 scripted 成功交接摘要；失败时直接抛错."""
    if not final_snapshot_hash:
        raise ValueError("final_snapshot_hash cannot be empty")
    resolved_workspace_path = (
        workspace_path
        if workspace_path is not None
        else conversation_json_path.parent
    )
    control_artifact_hashes = collect_control_artifact_hashes(resolved_workspace_path)
    control_artifacts_read_by_stage: dict[str, list[str]] = {}
    control_artifacts_injected_by_stage: dict[str, list[str]] = {}
    if stage_summaries is not None:
        for summary in stage_summaries:
            descriptor = summary.prompt_descriptor or summary.profile_name
            control_artifacts_read_by_stage[descriptor] = list(summary.control_artifacts_read)
            control_artifacts_injected_by_stage[descriptor] = list(
                getattr(summary, "control_artifacts_injected", ())
            )
    storage_plan_alignment = build_storage_plan_alignment(
        resolved_workspace_path / "storage_plan.txt"
    )
    status = str(storage_plan_alignment.get("status") or "")
    if status not in PROMOTABLE_STORAGE_PLAN_ALIGNMENT_STATUSES:
        logger.warning(
            "Scripted run storage_plan_alignment is advisory-only and not promotable: status=%s, detail=%s",
            status or "missing",
            storage_plan_alignment.get("departures")
            or storage_plan_alignment.get("missing_query_ids")
            or [],
        )
    summary = ScriptedRunSummary(
        benchmark=benchmark,
        conv_name=conv_name,
        run_id=conv_name,
        query_list=query_list,
        is_bespoke_storage=True,
        final_snapshot_hash=final_snapshot_hash,
        completed_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        conversation_json=str(conversation_json_path),
        session_db_path=str(session_db_path),
        success=True,
        validation_mode=validation_mode,
        control_artifact_hashes=control_artifact_hashes,
        control_artifacts_read_by_stage=control_artifacts_read_by_stage,
        control_artifacts_injected_by_stage=control_artifacts_injected_by_stage,
        storage_plan_sha256=control_artifact_hashes.get("storage_plan.txt"),
        todo_sha256=control_artifact_hashes.get("TODO.md"),
        design_evidence_sha256=control_artifact_hashes.get("design_evidence.md"),
        implementation_manifest_sha256=control_artifact_hashes.get("implementation_manifest.json"),
        todo_reconciliation=build_todo_reconciliation(resolved_workspace_path / "TODO.md"),
        storage_plan_alignment=storage_plan_alignment,
        wandb_primary_run_id=wandb_result.primary_run_id if wandb_result is not None else None,
        wandb_final_run_id=wandb_result.final_run_id if wandb_result is not None else None,
        wandb_init_attempt_count=wandb_result.attempt_count if wandb_result is not None else 0,
        wandb_attempted_run_ids=wandb_result.attempted_run_ids if wandb_result is not None else [],
        wandb_retry_used=wandb_result.used_fallback if wandb_result is not None else False,
        wandb_first_failure_excerpt=wandb_result.first_failure_excerpt if wandb_result is not None else None,
    )
    file_path = write_scripted_run_summary(summary, artifacts_dir)
    logger.info(f"Scripted run completed successfully. Summary written to {file_path}")
    return file_path


def find_latest_successful_run(
    conv_name: str | None,
    query_list: list[str],
    benchmark: str = "tpch",
    artifacts_dir: Path | None = None,
    validation_mode: str | None = None,
    is_bespoke_storage: bool | None = None,
) -> Optional[ScriptedRunSummary]:
    """查找最新的成功 scripted run.
    
    Args:
        conv_name: Conversation 名称；None 时搜索全部 scripted run
        query_list: 查询列表（用于匹配兼容的 run）
        benchmark: Benchmark 名称
        artifacts_dir: artifacts 根目录，默认为 ./tpch_monetdb_artifacts
        validation_mode: 可选的 validation_mode 过滤
        
    Returns:
        最新的成功摘要，如果没有找到则返回 None
    """
    if artifacts_dir is None:
        artifacts_dir = Path(DEFAULT_TPCH_MONETDB_ARTIFACTS_DIR)
    
    summary_dir = get_summary_dir(artifacts_dir)
    if conv_name is None:
        conv_dirs = [path for path in summary_dir.iterdir() if path.is_dir()] if summary_dir.exists() else []
    else:
        conv_dirs = [summary_dir / conv_name]

    if not conv_dirs:
        logger.info(f"No scripted run directory found for {conv_name}")
        return None

    if conv_name is not None:
        conv_dir = conv_dirs[0]
        if conv_dir.exists():
            try:
                latest_summary = _load_latest_summary(conv_dir)
                if latest_summary is not None and _is_compatible_summary(
                    latest_summary,
                    query_list,
                    benchmark,
                    validation_mode,
                    is_bespoke_storage=is_bespoke_storage,
                ):
                    logger.info(
                        "Found latest scripted run via latest.json: %s at %s",
                        latest_summary.conv_name,
                        latest_summary.completed_at,
                    )
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
                if not _is_compatible_summary(
                    summary,
                    query_list,
                    benchmark,
                    validation_mode,
                    is_bespoke_storage=is_bespoke_storage,
                ):
                    continue
                summaries.append(summary)
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                logger.warning(f"Failed to parse summary file {file_path}: {e}")
                continue
    
    if not summaries:
        logger.info(f"No compatible scripted run found for {conv_name}")
        return None
    
    # 按 completed_at 排序，取最新
    summaries.sort(key=lambda s: s.completed_at, reverse=True)
    latest = summaries[0]
    
    logger.info(f"Found latest scripted run: {latest.conv_name} at {latest.completed_at}")
    return latest


def auto_discover_start_snapshot(
    conv_name: str | None,
    query_list: list[str],
    benchmark: str = "tpch",
    artifacts_dir: Path | None = None,
    explicit_snapshot: str | None = None,
    is_bespoke_storage: bool | None = None,
) -> str:
    """自动发现起始 snapshot.
    
    Args:
        conv_name: Conversation 名称
        query_list: 查询列表
        benchmark: Benchmark 名称
        artifacts_dir: artifacts 根目录
        explicit_snapshot: 显式指定的 snapshot（优先使用）
        
    Returns:
        snapshot hash
        
    Raises:
        ValueError: 如果无法发现有效的 snapshot
    """
    # 优先使用显式指定的 snapshot
    if explicit_snapshot is not None:
        logger.info(f"Using explicit start_snapshot: {explicit_snapshot}")
        return explicit_snapshot
    
    # 尝试从本地 summary 发现
    summary = find_latest_successful_run(
        conv_name=conv_name,
        query_list=query_list,
        benchmark=benchmark,
        artifacts_dir=artifacts_dir,
        validation_mode="strict",
        is_bespoke_storage=is_bespoke_storage,
    )
    
    if summary is not None:
        if summary.final_snapshot_hash:
            logger.info(f"Auto-discovered start_snapshot from scripted run: {summary.final_snapshot_hash}")
            return summary.final_snapshot_hash
        else:
            logger.warning("Found scripted run summary but final_snapshot_hash is empty")
    
    raise ValueError(
        f"Could not auto-discover start_snapshot for {conv_name}. "
        "Please run a scripted workflow first, or provide --start_snapshot explicitly."
    )
