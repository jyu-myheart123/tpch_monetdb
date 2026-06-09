"""Auto-compact implementation for automatic context compression.

This module provides automatic compaction based on token usage thresholds.
This is Layer 2 of the three-layer compression architecture (micro → auto → manual).
"""

import logging
import math
import os
import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any

from agents import TResponseInputItem
from tpch_monetdb.config import get_max_consecutive_failures

from .artifact_ledger import ArtifactLedger, build_preview
from .context_budget import RequestBudgetEstimate, build_request_budget_estimate
from .context_lifecycle_v3 import (
    LocalCompactResult,
    aggressive_compact_items as build_aggressive_compact_items,
    stage_memory_compact_items as build_stage_memory_compact_items,
)
from .models import get_context_window

logger = logging.getLogger(__name__)

MAX_OUTPUT_RESERVE = 20_000
AUTO_COMPACT_BUFFER_TOKENS = 13_000
WARNING_THRESHOLD_BUFFER_TOKENS = 20_000
BLOCKING_THRESHOLD_BUFFER_TOKENS = 3_000
MAX_CONSECUTIVE_FAILURES = get_max_consecutive_failures()
DISABLE_AUTO_COMPACT_ENV = "DISABLE_AUTO_COMPACT"
CHARS_PER_TOKEN = 4
TOKEN_ESTIMATE_SAFETY = 4 / 3
DEFAULT_CONTEXT_WINDOW = 200_000
BODY_BYTES_PER_CHAR = 1.5
DEFAULT_BODY_BLOCKING_BYTES = 8 * 1024 * 1024
_BODY_COMPACT_ENV = "TPCH_MONETDB_BODY_COMPACT_BYTES"
_BODY_BLOCKING_ENV = "TPCH_MONETDB_BODY_BLOCKING_BYTES"
DETERMINISTIC_TRIM_INLINE_BYTES = 16 * 1024
STAGE_PRESERVE_LIMITS = {
    "storage_plan": 12,
    "todo_plan": 12,
    "finish_skeleton": 12,
    "implement_queries_writeonly": 12,
    "correctness_queries_writeonly": 10,
    "correctness_foundation": 10,
    "correctness": 10,
    "all_queries_correctness": 10,
}
STAGE_WARNING_OFFSETS = {
    "implement_queries_writeonly": 15_000,
    "correctness_queries_writeonly": 30_000,
    "correctness_foundation": 30_000,
    "correctness": 20_000,
    "all_queries_correctness": 30_000,
}
DEFAULT_PRESERVE_LIMIT = 16


@dataclass(frozen=True)
class DeterministicTrimResult:
    """Describe one deterministic session trim attempt."""

    items: list[TResponseInputItem]
    changed_count: int
    bytes_before: int
    bytes_after: int


@dataclass(frozen=True)
class StageContextMaintenanceResult:
    """Describe a stage-end context maintenance pass."""

    stage_name: str | None
    profile_name: str | None
    deterministic_trimmed_items: int
    artifact_pruned_count: int
    pre_budget: RequestBudgetEstimate
    post_budget: RequestBudgetEstimate
    llm_compaction_attempted: bool
    llm_compaction_succeeded: bool
    should_fail: bool
    failure_detail: str | None = None


def estimate_token_count(text: str) -> int:
    """Estimate token count from character count.
    
    Uses a conservative estimate: ceil(chars / 4 * 4/3) ≈ chars / 3
    
    Args:
        text: Text to estimate tokens for
        
    Returns:
        Estimated token count
    """
    if not text:
        return 0
    chars = len(text)
    return math.ceil(chars / CHARS_PER_TOKEN * TOKEN_ESTIMATE_SAFETY)


def estimate_message_tokens(message: TResponseInputItem) -> int:
    """Estimate token count for a structured session item."""
    return sum(estimate_token_count(fragment) for fragment in collect_text_fragments(message))


def collect_text_fragments(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if value is None:
        return []
    if isinstance(value, list):
        fragments: list[str] = []
        for item in value:
            fragments.extend(collect_text_fragments(item))
        return fragments
    if isinstance(value, dict):
        fragments: list[str] = []
        for key in ("content", "text", "value", "message", "output", "name", "arguments", "summary"):
            if key in value:
                fragments.extend(collect_text_fragments(value[key]))
        return fragments
    return []


def estimate_session_tokens(items: list[TResponseInputItem]) -> int:
    """Estimate total token count for a list of session items."""
    return sum(estimate_message_tokens(item) for item in items)


def _json_bytes(value: Any) -> int:
    """Return UTF-8 JSON bytes for deterministic budget comparison."""
    return len(json.dumps(value, ensure_ascii=False, default=str).encode("utf-8"))


def _sha256_text(text: str) -> str:
    """Return SHA-256 for a UTF-8 text blob."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _collect_visible_artifact_refs(
    items: list[TResponseInputItem],
    *,
    stage_memory: str | None,
    artifact_context: str | None,
) -> tuple[str, ...]:
    """Return artifact refs that are still visible to the current session lifecycle."""
    texts = [json.dumps(item, ensure_ascii=False, default=str) for item in items]
    for extra in (stage_memory, artifact_context):
        if extra:
            texts.append(extra)
    refs: list[str] = []
    for text in texts:
        refs.extend(re.findall(r"artifact_ref[:=]\s*([A-Za-z0-9_.-]+)", text))
    return tuple(dict.fromkeys(refs))


def _build_call_id_to_name(items: list[TResponseInputItem]) -> dict[str, str]:
    """Map SDK function call ids to their tool names."""
    mapping: dict[str, str] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "function_call":
            call_id = item.get("call_id")
            name = item.get("name")
            if isinstance(call_id, str) and isinstance(name, str):
                mapping[call_id] = name
    return mapping


def _extract_tool_name(
    item: dict[str, Any],
    call_id_to_name: dict[str, str],
) -> str | None:
    """Extract the tool name for an SDK or chat-style tool result."""
    if item.get("type") == "function_call_output":
        call_id = item.get("call_id")
        return call_id_to_name.get(call_id) if isinstance(call_id, str) else None
    if item.get("role") == "tool":
        name = item.get("name")
        return name if isinstance(name, str) else None
    return None


def _tool_output_field(item: dict[str, Any]) -> str | None:
    """Return the mutable field that contains tool output text."""
    if isinstance(item.get("output"), str):
        return "output"
    if isinstance(item.get("content"), str):
        return "content"
    return None


class AutoCompactManager:
    """Manage auto-compact thresholds and stage-aware compaction attempts."""

    def __init__(self, model: str, artifact_ledger: ArtifactLedger | None = None):
        self.model = model
        self.artifact_ledger = artifact_ledger
        self.context_window = self._get_context_window(model)
        self.effective_context_window = self.context_window - min(
            self.context_window,
            MAX_OUTPUT_RESERVE,
        )
        self.threshold = self.effective_context_window - AUTO_COMPACT_BUFFER_TOKENS
        self.warning_threshold = (
            self.threshold - WARNING_THRESHOLD_BUFFER_TOKENS
        )
        self.blocking_threshold = (
            self.effective_context_window - BLOCKING_THRESHOLD_BUFFER_TOKENS
        )
        self.consecutive_failures = 0
        self.last_failure_info: dict[str, Any] | None = None
        self.disabled = os.environ.get(DISABLE_AUTO_COMPACT_ENV) is not None
        if self.disabled:
            logger.info("Auto-compact disabled via environment variable")

    def _get_context_window(self, model: str) -> int:
        try:
            return get_context_window(model)
        except KeyError:
            logger.warning(f"Unknown model {model}, using default 200K context window")
            return DEFAULT_CONTEXT_WINDOW

    def _stage_warning_offset(self, profile_name: str | None) -> int:
        if profile_name is None:
            return 0
        return STAGE_WARNING_OFFSETS.get(profile_name, 0)

    def should_compact(self, current_tokens: int, profile_name: str | None = None) -> bool:
        if self.disabled:
            return False
        if self.consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
            return False
        return current_tokens >= self.get_threshold(profile_name)

    def describe_usage(self, current_tokens: int, profile_name: str | None = None) -> dict[str, Any]:
        threshold = self.get_threshold(profile_name) if not self.disabled else self.effective_context_window
        warning_threshold = self.get_warning_threshold(profile_name)
        percent_left = max(
            0,
            math.floor(((threshold - current_tokens) / threshold) * 100),
        )
        return {
            "current_tokens": current_tokens,
            "percent_left": percent_left,
            "is_above_warning_threshold": current_tokens >= warning_threshold,
            "is_above_auto_compact_threshold": current_tokens >= threshold,
            "is_at_blocking_limit": current_tokens >= self.blocking_threshold,
        }

    def _get_preserve_limit(self, profile_name: str | None) -> int:
        if profile_name is None:
            return DEFAULT_PRESERVE_LIMIT
        return STAGE_PRESERVE_LIMITS.get(profile_name, DEFAULT_PRESERVE_LIMIT)

    def _log_circuit_breaker(self, current_tokens: int, non_system_item_count: int) -> None:
        logger.critical(
            "Circuit breaker triggered with context still above threshold: current_tokens=%s threshold=%s non_system_items=%s last_failure=%s",
            current_tokens,
            self.threshold,
            non_system_item_count,
            self.last_failure_info,
        )

    async def _replace_session_items(
        self,
        session: Any,
        items: list[TResponseInputItem],
    ) -> None:
        """Replace all session items with the supplied deterministic item list."""
        await session.clear_session()
        if items:
            await session.add_items(items)
        return None

    def deterministic_trim_items(
        self,
        items: list[TResponseInputItem],
        *,
        profile_name: str | None = None,
        max_inline_bytes: int = DETERMINISTIC_TRIM_INLINE_BYTES,
    ) -> DeterministicTrimResult:
        """Replace oversized historical tool outputs with stable artifact digests."""
        trimmed: list[TResponseInputItem] = []
        changed_count = 0
        bytes_before = 0
        bytes_after = 0
        call_id_to_name = _build_call_id_to_name(items)
        for item in items:
            before = _json_bytes(item)
            bytes_before += before
            replacement = self._trim_one_item(
                item,
                call_id_to_name=call_id_to_name,
                profile_name=profile_name,
                max_inline_bytes=max_inline_bytes,
            )
            after = _json_bytes(replacement)
            bytes_after += after
            if replacement != item:
                changed_count += 1
            trimmed.append(replacement)
        return DeterministicTrimResult(
            items=trimmed,
            changed_count=changed_count,
            bytes_before=bytes_before,
            bytes_after=bytes_after,
        )

    def _trim_one_item(
        self,
        item: TResponseInputItem,
        *,
        call_id_to_name: dict[str, str],
        profile_name: str | None,
        max_inline_bytes: int,
    ) -> TResponseInputItem:
        """Trim one oversized tool output item and preserve a retrievable digest."""
        if not isinstance(item, dict):
            return item
        tool_name = _extract_tool_name(item, call_id_to_name)
        if tool_name is None:
            return item
        field_name = _tool_output_field(item)
        if field_name is None:
            return item
        output_value = item.get(field_name)
        if not isinstance(output_value, str):
            return item
        if (
            "artifact_ref:" in output_value
            and len(output_value.encode("utf-8")) <= max_inline_bytes
        ):
            return item
        if len(output_value.encode("utf-8")) <= max_inline_bytes:
            return item
        replacement = self._build_trim_replacement(
            output_value,
            tool_name=tool_name,
            call_id=str(item.get("call_id") or ""),
            profile_name=profile_name,
        )
        trimmed = dict(item)
        trimmed[field_name] = replacement
        trimmed["_deterministic_trimmed"] = True
        return trimmed

    def _build_trim_replacement(
        self,
        text: str,
        *,
        tool_name: str,
        call_id: str,
        profile_name: str | None,
    ) -> str:
        """Build a compact digest for one trimmed raw tool output."""
        preview, omitted_chars = build_preview(text)
        if self.artifact_ledger is not None:
            artifact = self.artifact_ledger.record_text(
                kind="deterministic_trim_tool_output",
                text=text,
                metadata={
                    "stage_name": profile_name,
                    "tool_name": tool_name,
                    "call_id": call_id or None,
                    "summary": f"deterministically trimmed {tool_name} output",
                    "tags": ("deterministic_trim", tool_name),
                },
            )
            return self.artifact_ledger.render_digest(
                artifact,
                preview=preview,
                omitted_chars=omitted_chars,
            )
        sha256 = _sha256_text(text)
        return "\n".join([
            preview,
            "[Evidence Digest]",
            "artifact_ref: unavailable",
            "kind: deterministic_trim_tool_output",
            f"tool: {tool_name}",
            f"stage: {profile_name or '-'}",
            f"sha256: {sha256}",
            f"bytes: {len(text.encode('utf-8'))}",
            f"summary: deterministically trimmed {tool_name} output",
        ])

    def stage_memory_compact_items(
        self,
        items: list[TResponseInputItem],
        *,
        stage_memory: str | None = None,
        artifact_context: str | None = None,
        profile_name: str | None = None,
    ) -> LocalCompactResult:
        """Delegate local stage-memory compaction to Context Lifecycle v3."""
        return build_stage_memory_compact_items(
            items,
            stage_memory=stage_memory,
            artifact_context=artifact_context,
            profile_name=profile_name,
        )

    def aggressive_compact_items(
        self,
        items: list[TResponseInputItem],
        *,
        stage_memory: str | None = None,
        artifact_context: str | None = None,
        profile_name: str | None = None,
    ) -> LocalCompactResult:
        """Delegate aggressive local compaction to Context Lifecycle v3."""
        return build_aggressive_compact_items(
            items,
            stage_memory=stage_memory,
            artifact_context=artifact_context,
            profile_name=profile_name,
        )

    async def compact(
        self,
        session: Any,
        compaction_session: Any,
        current_tokens: int = 0,
        profile_name: str | None = None,
        stage_memory: str | None = None,
        artifact_context: str | None = None,
        force_aggressive: bool = False,
    ) -> bool:
        """Run stage-aware auto-compaction and update failure counters."""
        if self.disabled:
            return False
        non_system_items: list[TResponseInputItem] = []
        try:
            items = await session.get_items()
            trim_result = self.deterministic_trim_items(
                items,
                profile_name=profile_name,
            )
            if trim_result.changed_count > 0:
                await self._replace_session_items(session, trim_result.items)
                items = trim_result.items
                logger.info(
                    "Deterministic trim replaced %s tool output item(s): body_bytes %s -> %s",
                    trim_result.changed_count,
                    trim_result.bytes_before,
                    trim_result.bytes_after,
                )
            local_result = (
                self.aggressive_compact_items(
                    items,
                    stage_memory=stage_memory,
                    artifact_context=artifact_context,
                    profile_name=profile_name,
                )
                if force_aggressive
                else self.stage_memory_compact_items(
                    items,
                    stage_memory=stage_memory,
                    artifact_context=artifact_context,
                    profile_name=profile_name,
                )
            )
            if local_result.changed_count > 0:
                await self._replace_session_items(session, local_result.items)
                items = local_result.items
                logger.info(
                    "Local %s compact replaced session: tokens %s -> %s, body_bytes %s -> %s",
                    local_result.mode,
                    local_result.pre_tokens,
                    local_result.post_tokens,
                    local_result.pre_body_bytes,
                    local_result.post_body_bytes,
                )
                if force_aggressive or estimate_session_tokens(items) < self.get_threshold(profile_name):
                    self.consecutive_failures = 0
                    self.last_failure_info = None
                    return True
            elif force_aggressive:
                self.consecutive_failures += 1
                self.last_failure_info = {
                    "stage_name": profile_name,
                    "failure_count": self.consecutive_failures,
                    "candidate_count": 0,
                    "preserved_count": len(items),
                    "chunk_count": 0,
                    "estimated_candidate_tokens": 0,
                    "estimated_candidate_chars": 0,
                    "reason": "aggressive_compaction_noop",
                }
                return False
            elif trim_result.changed_count > 0 and estimate_session_tokens(items) < self.get_threshold(profile_name):
                self.consecutive_failures = 0
                self.last_failure_info = None
                return True
            non_system_items = [
                item
                for item in items
                if not (isinstance(item, dict) and item.get("role") == "system")
            ]
            preserve_limit = self._get_preserve_limit(profile_name)
            min_candidate_items = 0
            if len(non_system_items) > preserve_limit:
                min_candidate_items = max(8, math.ceil(len(non_system_items) * 0.2))
            diagnostics = self.describe_usage(current_tokens, profile_name=profile_name)
            logger.info(
                "Auto-compacting with stage_memory_v3: total_items=%s non_system=%s preserve_limit=%s min_candidate=%s",
                len(items),
                len(non_system_items),
                preserve_limit,
                min_candidate_items,
            )
            attempt = await compaction_session.run_compaction({
                "force_trigger": False,
                "selection_policy": "stage_memory_v3",
                "preserve_limit_items": preserve_limit,
                "min_candidate_items": min_candidate_items,
                "pre_compact_tokens": current_tokens,
                "trigger_threshold": self.get_threshold(profile_name),
                "warning_threshold": self.get_warning_threshold(profile_name),
                "context_diagnostics": diagnostics,
            })
            if attempt is None:
                self.consecutive_failures = 0
                self.last_failure_info = None
                logger.info("Auto-compact completed successfully")
                return True
            if attempt.status == "success" and getattr(attempt, "effective", True):
                self.consecutive_failures = 0
                self.last_failure_info = None
                logger.info("Auto-compact completed successfully")
                return True
            if attempt.status == "success":
                aggressive_items = await session.get_items()
                aggressive_result = self.aggressive_compact_items(
                    aggressive_items,
                    stage_memory=stage_memory,
                    artifact_context=artifact_context,
                    profile_name=profile_name,
                )
                if aggressive_result.changed_count > 0:
                    await self._replace_session_items(session, aggressive_result.items)
                    self.consecutive_failures = 0
                    self.last_failure_info = None
                    logger.warning(
                        "Auto-compact ineffective; aggressive local compact replaced session: tokens %s -> %s, body_bytes %s -> %s",
                        aggressive_result.pre_tokens,
                        aggressive_result.post_tokens,
                        aggressive_result.pre_body_bytes,
                        aggressive_result.post_body_bytes,
                    )
                    return True
            self.consecutive_failures += 1
            failure_reason = (
                "ineffective_compaction"
                if attempt.status == "success"
                else attempt.skip_reason
            )
            self.last_failure_info = {
                "stage_name": profile_name,
                "failure_count": self.consecutive_failures,
                "candidate_count": attempt.candidate_count,
                "preserved_count": attempt.preserved_count,
                "chunk_count": getattr(attempt, "chunk_count", 0),
                "estimated_candidate_tokens": getattr(attempt, "estimated_candidate_tokens", 0),
                "estimated_candidate_chars": getattr(attempt, "estimated_candidate_chars", 0),
                "reason": failure_reason,
            }
            if attempt.status == "skipped":
                logger.warning(
                    "Auto-compact skipped after threshold: reason=%s candidate=%s preserved=%s chunks=%s est_tokens=%s est_chars=%s stage=%s failure_count=%s",
                    attempt.skip_reason,
                    attempt.candidate_count,
                    attempt.preserved_count,
                    getattr(attempt, "chunk_count", 0),
                    getattr(attempt, "estimated_candidate_tokens", 0),
                    getattr(attempt, "estimated_candidate_chars", 0),
                    profile_name,
                    self.consecutive_failures,
                )
            elif attempt.status == "success":
                logger.warning(
                    "Auto-compact was ineffective: candidate=%s preserved=%s chunks=%s pre_tokens=%s post_tokens=%s pre_body=%s post_body=%s stage=%s failure_count=%s",
                    attempt.candidate_count,
                    attempt.preserved_count,
                    getattr(attempt, "chunk_count", 0),
                    getattr(attempt, "pre_tokens", 0),
                    getattr(attempt, "post_tokens", 0),
                    getattr(attempt, "pre_body_bytes", 0),
                    getattr(attempt, "post_body_bytes", 0),
                    profile_name,
                    self.consecutive_failures,
                )
            else:
                logger.warning(
                    "Auto-compact reported failure: reason=%s candidate=%s preserved=%s chunks=%s est_tokens=%s est_chars=%s stage=%s failure_count=%s",
                    attempt.skip_reason,
                    attempt.candidate_count,
                    attempt.preserved_count,
                    getattr(attempt, "chunk_count", 0),
                    getattr(attempt, "estimated_candidate_tokens", 0),
                    getattr(attempt, "estimated_candidate_chars", 0),
                    profile_name,
                    self.consecutive_failures,
                )
            if self.consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                self._log_circuit_breaker(
                    current_tokens=current_tokens,
                    non_system_item_count=len(non_system_items),
                )
            return False
        except Exception as e:
            self.consecutive_failures += 1
            self.last_failure_info = {
                "stage_name": profile_name,
                "failure_count": self.consecutive_failures,
                "candidate_count": 0,
                "preserved_count": 0,
                "chunk_count": 0,
                "estimated_candidate_tokens": 0,
                "estimated_candidate_chars": 0,
                "reason": str(e),
            }
            logger.warning(
                f"Auto-compact failed ({self.consecutive_failures}/{MAX_CONSECUTIVE_FAILURES}): {e}"
            )
            
            if self.consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                self._log_circuit_breaker(
                    current_tokens=current_tokens,
                    non_system_item_count=len(non_system_items),
                )
            
            return False

    async def maintain_after_stage(
        self,
        *,
        session: Any,
        compaction_session: Any,
        profile_name: str | None = None,
        stage_name: str | None = None,
        query_ids: tuple[str, ...] = (),
        allow_llm_compaction: bool = True,
        force_llm_compaction: bool = False,
        stage_memory: str | None = None,
        artifact_context: str | None = None,
    ) -> StageContextMaintenanceResult:
        """Run bounded stage-end context maintenance without unconditional LLM compaction."""
        pre_budget = await self.estimate_request_budget(session=session, new_input="")
        if self.disabled:
            return StageContextMaintenanceResult(
                stage_name=stage_name,
                profile_name=profile_name,
                deterministic_trimmed_items=0,
                artifact_pruned_count=0,
                pre_budget=pre_budget,
                post_budget=pre_budget,
                llm_compaction_attempted=False,
                llm_compaction_succeeded=False,
                should_fail=pre_budget.should_fail,
                failure_detail="auto_compact_disabled" if pre_budget.should_fail else None,
            )

        items = await session.get_items()
        trim_result = self.deterministic_trim_items(items, profile_name=profile_name)
        if trim_result.changed_count > 0:
            await self._replace_session_items(session, trim_result.items)
            items = trim_result.items
            logger.info(
                "Stage-end deterministic trim replaced %s item(s): body_bytes %s -> %s",
                trim_result.changed_count,
                trim_result.bytes_before,
                trim_result.bytes_after,
            )
        artifact_pruned_count = 0
        if self.artifact_ledger is not None:
            scope_keep_ids = self.artifact_ledger.artifact_ids_for_scope(
                query_ids=query_ids,
                stage_name=profile_name or stage_name,
            )
            visible_keep_ids = self.artifact_ledger.artifact_ids_for_refs(
                _collect_visible_artifact_refs(
                    items,
                    stage_memory=stage_memory,
                    artifact_context=artifact_context,
                )
            )
            keep_ids = tuple(dict.fromkeys((*scope_keep_ids, *visible_keep_ids)))
            pruned = self.artifact_ledger.cleanup_default_retention(
                keep_artifact_ids=keep_ids,
            )
            artifact_pruned_count = len(pruned)
            if artifact_pruned_count > 0:
                logger.info(
                    "Stage-end artifact retention pruned %s artifact(s)",
                    artifact_pruned_count,
                )

        budget_after_trim = await self.estimate_request_budget(session=session, new_input="")
        if force_llm_compaction or budget_after_trim.should_compact:
            local_stage_result = self.stage_memory_compact_items(
                items,
                stage_memory=stage_memory,
                artifact_context=artifact_context,
                profile_name=profile_name or stage_name,
            )
            if local_stage_result.changed_count > 0:
                await self._replace_session_items(session, local_stage_result.items)
                items = local_stage_result.items
                logger.info(
                    "Stage-end local stage-memory compact replaced session: tokens %s -> %s, body_bytes %s -> %s",
                    local_stage_result.pre_tokens,
                    local_stage_result.post_tokens,
                    local_stage_result.pre_body_bytes,
                    local_stage_result.post_body_bytes,
                )
                budget_after_trim = await self.estimate_request_budget(
                    session=session,
                    new_input="",
                )
        can_try_llm = self.consecutive_failures < MAX_CONSECUTIVE_FAILURES
        needs_llm = force_llm_compaction or (
            allow_llm_compaction and budget_after_trim.should_compact
        )
        llm_attempted = False
        llm_succeeded = False
        failure_detail = None
        if needs_llm and can_try_llm:
            llm_attempted = True
            llm_succeeded = await self.compact(
                session=session,
                compaction_session=compaction_session,
                current_tokens=budget_after_trim.token_estimate,
                profile_name=profile_name or stage_name,
                stage_memory=stage_memory,
                artifact_context=artifact_context,
            )
            if not llm_succeeded:
                failure_detail = str(self.last_failure_info or "llm_compaction_failed")
        elif needs_llm:
            failure_detail = "auto_compact_circuit_open"
            logger.warning(
                "Stage-end LLM compaction skipped because circuit is open: stage=%s failure=%s",
                stage_name,
                self.last_failure_info,
            )

        post_budget = await self.estimate_request_budget(session=session, new_input="")
        should_fail = post_budget.should_fail
        if should_fail and failure_detail is None:
            failure_detail = (
                f"tokens={post_budget.token_estimate}(level={post_budget.token_level}) "
                f"body={post_budget.body_bytes}(level={post_budget.body_level})"
            )
        logger.info(
            "Stage-end context maintenance: stage=%s profile=%s trim=%s pruned=%s "
            "pre_tokens=%s pre_body=%s post_tokens=%s post_body=%s llm_attempted=%s llm_ok=%s",
            stage_name,
            profile_name,
            trim_result.changed_count,
            artifact_pruned_count,
            pre_budget.token_estimate,
            pre_budget.body_bytes,
            post_budget.token_estimate,
            post_budget.body_bytes,
            llm_attempted,
            llm_succeeded,
        )
        return StageContextMaintenanceResult(
            stage_name=stage_name,
            profile_name=profile_name,
            deterministic_trimmed_items=trim_result.changed_count,
            artifact_pruned_count=artifact_pruned_count,
            pre_budget=pre_budget,
            post_budget=post_budget,
            llm_compaction_attempted=llm_attempted,
            llm_compaction_succeeded=llm_succeeded,
            should_fail=should_fail,
            failure_detail=failure_detail,
        )
    
    def get_threshold(self, profile_name: str | None = None) -> int:
        """Get the current threshold.
        
        Returns:
            Token threshold for triggering compaction
        """
        return self.threshold - self._stage_warning_offset(profile_name)

    def get_effective_context_window(self) -> int:
        return self.effective_context_window

    def get_warning_threshold(self, profile_name: str | None = None) -> int:
        return self.warning_threshold - self._stage_warning_offset(profile_name)

    def get_blocking_threshold(self) -> int:
        return self.blocking_threshold

    def get_body_blocking_threshold(self) -> int:
        """Return the legacy body compact threshold name for compatibility."""
        return self.get_body_compact_threshold()

    def get_body_compact_threshold(self) -> int:
        """Return serialized body bytes that should trigger pre-send compaction."""
        for env_name in (_BODY_COMPACT_ENV, _BODY_BLOCKING_ENV):
            env_val = os.environ.get(env_name)
            if env_val is None:
                continue
            try:
                parsed = int(env_val)
                if parsed > 0:
                    return parsed
            except ValueError:
                continue
        return DEFAULT_BODY_BLOCKING_BYTES

    async def estimate_request_tokens(
        self,
        session: Any,
        new_input: str,
    ) -> int:
        """Estimate total tokens for the next request.

        累加当前 session 中全部 items 的 token 估算，加上新输入文本的估算。
        """
        items = await session.get_items()
        session_tokens = estimate_session_tokens(items)
        input_tokens = estimate_token_count(new_input)
        return session_tokens + input_tokens

    async def estimate_request_body_bytes(
        self,
        session: Any,
        new_input: str,
    ) -> int:
        """Estimate serialized JSON bytes for the next request."""
        budget = await self.estimate_request_budget(session=session, new_input=new_input)
        return budget.body_bytes

    async def estimate_request_budget(
        self,
        session: Any,
        new_input: str,
    ) -> RequestBudgetEstimate:
        """Estimate token and body pressure with largest contributors."""
        items = await session.get_items()
        return build_request_budget_estimate(
            items,
            new_input=new_input,
            token_limit=self.get_blocking_threshold(),
            body_compact_bytes=self.get_body_compact_threshold(),
        )

    def get_stats(self) -> dict[str, Any]:
        return {
            "context_window": self.context_window,
            "effective_context_window": self.effective_context_window,
            "threshold": self.threshold,
            "warning_threshold": self.warning_threshold,
            "blocking_threshold": self.blocking_threshold,
            "consecutive_failures": self.consecutive_failures,
            "last_failure_info": self.last_failure_info,
            "disabled": self.disabled,
            "circuit_open": self.consecutive_failures >= MAX_CONSECUTIVE_FAILURES,
        }
