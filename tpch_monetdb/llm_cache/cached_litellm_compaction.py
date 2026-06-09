import hashlib
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import litellm

from agents import TResponseInputItem
from agents.memory.session import OpenAIResponsesCompactionArgs

from tpch_monetdb.conversations.compact_prompts import (
    COMPACT_MAX_OUTPUT_TOKENS,
    COMPACT_SYSTEM_PROMPT,
)
from tpch_monetdb.llm_cache.deepseek_reasoning_replay import repair_deepseek_input_items
from . import utils
from .auto_compact import collect_text_fragments, estimate_message_tokens
from .context_budget import BODY_COMPACT_BYTES, estimate_json_bytes
from .litellm_retry import run_with_transient_retry
from .models import get_context_window
from tpch_monetdb.tools.stage_tool_policy import looks_like_validation_text
from tpch_monetdb.utils.model_aliases import is_deepseek_model
from tpch_monetdb.utils.wandb_stats_logging import WandbRunHook

logger = logging.getLogger(__name__)

COMPACTION_V3_HEADER = "[Compaction Summary v3]"
COMPACTION_SUMMARY_INPUT_BODY_BYTES = BODY_COMPACT_BYTES // 2
COMPACTION_EFFECTIVE_RATIO = 0.85


class CompactCacheType:
    """Cache entry for compaction results."""
    
    def __init__(self, output_items: list[TResponseInputItem]):
        self.output_items = output_items


@dataclass(frozen=True)
class CompactSelectionResult:
    candidate_items: list[TResponseInputItem]
    preserved_items: list[TResponseInputItem]
    candidate_count: int
    preserved_count: int
    skip_reason: str | None
    selection_policy: str | None


@dataclass(frozen=True)
class CompactionAttemptResult:
    status: str
    candidate_count: int
    preserved_count: int
    skip_reason: str | None = None
    selection_policy: str | None = None
    chunk_count: int = 0
    estimated_candidate_tokens: int = 0
    estimated_candidate_chars: int = 0
    pre_tokens: int = 0
    post_tokens: int = 0
    pre_body_bytes: int = 0
    post_body_bytes: int = 0
    reduction_ratio: float = 0.0
    effective: bool = True


class CachedLitellmCompactionSession:
    """LiteLLM-based compaction session independent of OpenAI API.
    
    This session uses LiteLLM to call any supported model for generating
    structured summaries during compaction, eliminating the dependency on
    OpenAI's responses.compact API.
    
    Attributes:
        session_id: Unique identifier for this session
        model: The model name to use for compaction
        api_key: API key for the model provider
        cache_dir: Directory for caching compaction results
        wandb_metrics_hook: Optional hook for logging metrics
    """
    
    def __init__(
        self,
        session_id: str,
        model: str,
        api_key: str,
        base_url: Optional[str],
        cache_dir: Path,
        wandb_metrics_hook: Optional[WandbRunHook] = None,
        compaction_model_map: Optional[dict[str, str]] = None,
    ):
        self.session_id = session_id
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.wandb_metrics_hook = wandb_metrics_hook
        self.compaction_model_map = compaction_model_map or {}
        self._underlying_session: Any = None

    @staticmethod
    def _deepseek_compaction_kwargs(model: str) -> dict[str, Any]:
        """Return cache-friendly DeepSeek defaults for compaction requests."""
        if not is_deepseek_model(model):
            return {"temperature": 0.0}
        return {
            "thinking": {"type": "disabled"},
            "allowed_openai_params": ["thinking", "reasoning_effort"],
            "additional_drop_params": ["extra_body"],
        }
    
    def set_underlying_session(self, session: Any) -> None:
        """Set the underlying session for storing compacted items."""
        self._underlying_session = session
    
    async def get_items(self) -> list[TResponseInputItem]:
        """Get items from underlying session.
        
        Returns:
            List of session items
        """
        if self._underlying_session is not None:
            items = await self._underlying_session.get_items()
            return repair_deepseek_input_items(
                items,
                model_name=self.model,
                fail_on_unrecoverable=True,
                require_reasoning_for_tool_calls=True,
            )
        return []
    
    async def add_items(self, items: list[TResponseInputItem]) -> None:
        """Add items to underlying session.
        
        Args:
            items: Items to add
        """
        if self._underlying_session is not None:
            await self._underlying_session.add_items(items)
    
    def _get_compaction_model(self) -> str:
        """Get the model to use for compaction.
        
        Uses the mapped model if configured, otherwise defaults to main model.
        
        Returns:
            Model name for compaction
        """
        if self.model in self.compaction_model_map:
            mapped = self.compaction_model_map[self.model]
            logger.info(f"Using mapped compaction model: {mapped} (from {self.model})")
            return mapped
        return self.model
    
    def _compute_session_hash(self, items: list[TResponseInputItem]) -> str:
        """Compute hash of session items for cache key.
        
        Args:
            items: Session items to hash
            
        Returns:
            SHA256 hash string
        """
        content = json.dumps(items, sort_keys=True, default=str)
        return hashlib.sha256(content.encode()).hexdigest()
    
    def _get_cache_path(self, session_hash: str, model: str) -> Path:
        """Get cache file path for compaction result.
        
        Args:
            session_hash: Hash of session content
            model: Model name used for compaction
            
        Returns:
            Path to cache file
        """
        payload = {"session_hash": session_hash, "model": model}
        hash_str = utils.sha256(utils.stable_json(payload))
        return self.cache_dir / f"{hash_str}.pkl"
    
    def _split_items(
        self,
        items: list[TResponseInputItem],
        keep_recent: int = 0,
    ) -> tuple[list[TResponseInputItem], list[TResponseInputItem]]:
        """Split items into compaction candidates and preserved recent items.

        System messages at the beginning are excluded from compaction.
        The last `keep_recent` non-system items are preserved.

        Args:
            items: All session items
            keep_recent: Number of recent items to preserve (0 = compact all)

        Returns:
            Tuple of (items to compact, items to preserve)
        """
        # Separate leading system messages and preserve them
        leading_system: list[TResponseInputItem] = []
        non_system: list[TResponseInputItem] = []
        for item in items:
            if not non_system and isinstance(item, dict) and item.get("role") == "system":
                leading_system.append(item)
                continue
            non_system.append(item)

        if keep_recent <= 0:
            return non_system, leading_system
        if keep_recent >= len(non_system):
            return [], leading_system + non_system

        split_point = len(non_system) - keep_recent
        split_point = self._adjust_split_point_for_tool_pairs(non_system, split_point)
        return non_system[:split_point], leading_system + non_system[split_point:]

    def _adjust_split_point_for_tool_pairs(
        self,
        items: list[TResponseInputItem],
        split_point: int,
    ) -> int:
        """Move keep_recent boundary earlier when it would orphan a tool output."""
        if split_point <= 0 or split_point >= len(items):
            return split_point
        pair_lookup = self._build_call_pair_lookup(items)
        adjusted = split_point
        while adjusted < len(items):
            item = items[adjusted]
            if not isinstance(item, dict):
                return adjusted
            if item.get("type") != "function_call_output":
                return adjusted
            call_id = item.get("call_id")
            if not isinstance(call_id, str):
                adjusted -= 1
                if adjusted <= 0:
                    return 0
                continue
            record = pair_lookup.get(call_id, {})
            call_index = record.get("call_index")
            if isinstance(call_index, int) and call_index < adjusted:
                adjusted = call_index
                continue
            adjusted -= 1
            if adjusted <= 0:
                return 0
        return adjusted

    def _select_items(
        self,
        items: list[TResponseInputItem],
        keep_recent: int,
        selection_policy: str | None,
        preserve_limit_items: int,
        min_candidate_items: int,
    ) -> CompactSelectionResult:
        if selection_policy == "stage_memory_v3":
            return self._select_items_stage_memory_v3(
                items=items,
                preserve_limit_items=preserve_limit_items,
                min_candidate_items=min_candidate_items,
            )
        if selection_policy is not None:
            raise ValueError(f"Unsupported compaction selection_policy: {selection_policy}")
        candidate_items, preserved_items = self._split_items(items, keep_recent)
        return CompactSelectionResult(
            candidate_items=candidate_items,
            preserved_items=preserved_items,
            candidate_count=len(candidate_items),
            preserved_count=len(preserved_items),
            skip_reason=None,
            selection_policy=None,
        )

    def _select_items_stage_memory_v3(
        self,
        items: list[TResponseInputItem],
        preserve_limit_items: int,
        min_candidate_items: int,
    ) -> CompactSelectionResult:
        """Select compaction candidates using failure/control/artifact relevance."""
        leading_system, non_system = self._split_leading_system_items(items)
        if len(non_system) <= preserve_limit_items:
            return CompactSelectionResult(
                candidate_items=[],
                preserved_items=leading_system + non_system,
                candidate_count=0,
                preserved_count=len(leading_system) + len(non_system),
                skip_reason="too_few_items_total",
                selection_policy="stage_memory_v3",
            )

        pair_lookup = self._build_call_pair_lookup(non_system)
        groups = self._semantic_preservation_groups(non_system, pair_lookup)
        preserved_set: set[int] = set()
        hard_limit = max(0, preserve_limit_items)
        if min_candidate_items > 0:
            hard_limit = min(hard_limit, max(0, len(non_system) - min_candidate_items))
        required_groups = [
            group
            for group in groups
            if group[1] >= 90
        ]
        for indices, _priority, _recency in required_groups:
            new_indices = tuple(index for index in indices if index not in preserved_set)
            if len(preserved_set) + len(new_indices) > hard_limit:
                continue
            preserved_set.update(new_indices)

        optional_groups = [
            group
            for group in groups
            if group not in required_groups
        ]
        for indices, priority, _recency in optional_groups:
            if priority <= 0:
                continue
            if any(index in preserved_set for index in indices):
                continue
            if len(preserved_set) + len(indices) > hard_limit:
                continue
            preserved_set.update(indices)

        candidate_count = len(non_system) - len(preserved_set)
        releasable = [
            group
            for group in reversed(groups)
            if group[1] < 90 and all(index in preserved_set for index in group[0])
        ]
        for indices, _priority, _recency in releasable:
            if candidate_count >= min_candidate_items:
                break
            for index in indices:
                preserved_set.discard(index)
            candidate_count += len(indices)

        preserved_items = [
            item for index, item in enumerate(non_system) if index in preserved_set
        ]
        candidate_items = [
            item for index, item in enumerate(non_system) if index not in preserved_set
        ]
        skip_reason = None
        if len(candidate_items) < 2:
            skip_reason = "insufficient_items_after_selection"
        return CompactSelectionResult(
            candidate_items=candidate_items,
            preserved_items=leading_system + preserved_items,
            candidate_count=len(candidate_items),
            preserved_count=len(leading_system) + len(preserved_items),
            skip_reason=skip_reason,
            selection_policy="stage_memory_v3",
        )

    def _build_call_pair_lookup(
        self,
        items: list[TResponseInputItem],
    ) -> dict[str, dict[str, int | str]]:
        lookup: dict[str, dict[str, int | str]] = {}
        for index, item in enumerate(items):
            if not isinstance(item, dict):
                continue
            item_type = item.get("type")
            call_id = item.get("call_id")
            if item_type == "function_call" and isinstance(call_id, str):
                record = lookup.setdefault(call_id, {})
                record["call_index"] = index
                name = item.get("name")
                if isinstance(name, str):
                    record["name"] = name
            elif item_type == "function_call_output" and isinstance(call_id, str):
                record = lookup.setdefault(call_id, {})
                record["output_index"] = index
        return lookup

    def _find_latest_stage_summary_indices(
        self,
        items: list[TResponseInputItem],
    ) -> list[int]:
        for index in range(len(items) - 1, -1, -1):
            if self._is_stage_summary_item(items[index]):
                return [index]
        return []

    def _find_latest_validation_indices(
        self,
        items: list[TResponseInputItem],
        pair_lookup: dict[str, dict[str, int | str]],
    ) -> list[int]:
        for index in range(len(items) - 1, -1, -1):
            if self._is_stage_summary_item(items[index]):
                continue
            if not self._looks_like_validation_item(items[index]):
                continue
            return self._expand_pair_indices(index, items[index], pair_lookup)
        return []

    def _find_latest_prefixed_summary_indices(
        self,
        items: list[TResponseInputItem],
        prefix: str,
    ) -> list[int]:
        for index in range(len(items) - 1, -1, -1):
            if self._item_text(items[index]).startswith(prefix):
                return [index]
        return []

    def _find_latest_tool_pair_indices(
        self,
        items: list[TResponseInputItem],
        pair_lookup: dict[str, dict[str, int | str]],
        tool_name: str,
    ) -> list[int]:
        for index in range(len(items) - 1, -1, -1):
            if self._tool_name_for_item(items[index], pair_lookup) != tool_name:
                continue
            return self._expand_pair_indices(index, items[index], pair_lookup)
        return []

    def _expand_pair_indices(
        self,
        index: int,
        item: TResponseInputItem,
        pair_lookup: dict[str, dict[str, int | str]],
    ) -> list[int]:
        if not isinstance(item, dict):
            return [index]
        call_id = item.get("call_id")
        if not isinstance(call_id, str):
            return [index]
        record = pair_lookup.get(call_id, {})
        expanded = {
            int(record[key])
            for key in ("call_index", "output_index")
            if key in record and isinstance(record[key], int)
        }
        if not expanded:
            return [index]
        return sorted(expanded)

    def _split_leading_system_items(
        self,
        items: list[TResponseInputItem],
    ) -> tuple[list[TResponseInputItem], list[TResponseInputItem]]:
        """Split leading system prompts from mutable conversation items."""
        leading_system: list[TResponseInputItem] = []
        non_system: list[TResponseInputItem] = []
        for item in items:
            if not non_system and isinstance(item, dict) and item.get("role") == "system":
                leading_system.append(item)
                continue
            non_system.append(item)
        return leading_system, non_system

    def _semantic_preservation_groups(
        self,
        items: list[TResponseInputItem],
        pair_lookup: dict[str, dict[str, int | str]],
    ) -> list[tuple[tuple[int, ...], int, int]]:
        """Return grouped item indices ordered by semantic preservation priority."""
        seen: set[tuple[int, ...]] = set()
        groups: list[tuple[tuple[int, ...], int, int]] = []
        for index, item in enumerate(items):
            indices = tuple(self._expand_pair_indices(index, item, pair_lookup))
            if indices in seen:
                continue
            seen.add(indices)
            priority = max(
                self._semantic_item_priority(items[group_index], pair_lookup)
                for group_index in indices
            )
            groups.append((indices, priority, max(indices)))
        groups.sort(key=lambda group: (group[1], group[2]), reverse=True)
        return groups

    def _semantic_item_priority(
        self,
        item: TResponseInputItem,
        pair_lookup: dict[str, dict[str, int | str]],
    ) -> int:
        """Score an item by failure, control-artifact, and query relevance."""
        text = self._item_text(item)
        lowered = text.lower()
        if (
            text.startswith("[Stage Summary]")
            or text.startswith("[Stage Memory v3]")
        ):
            return 100
        if self._looks_like_validation_item(item) or "validation failed" in lowered:
            return 95
        if text.startswith("[Optimization Control Summary]"):
            return 92
        if any(
            marker in lowered
            for marker in (
                "workload_objective.json",
                "storage_plan_alignment.json",
                "control_artifacts",
                "required_control_artifacts",
            )
        ):
            return 92
        if text.startswith("[Global Hotspot Summary]") or "hotspot" in lowered:
            return 90
        if (
            "artifact_ref:" in text
            or "artifact_ref=" in text
            or "sha256:" in text
        ):
            return 85
        if re.search(r"\bq(?:uery)?[_ -]?(?:1|9)\b", lowered):
            return 80
        if any(marker in lowered for marker in ("regression", "failed", "failure", "error")):
            return 75
        tool_name = self._tool_name_for_item(item, pair_lookup)
        if tool_name in {"run", "compile", "read_file", "grep_repo"}:
            return 60
        return 10

    def _tool_name_for_item(
        self,
        item: TResponseInputItem,
        pair_lookup: dict[str, dict[str, int | str]],
    ) -> str | None:
        if not isinstance(item, dict):
            return None
        item_type = item.get("type")
        if item_type == "function_call":
            name = item.get("name")
            if isinstance(name, str):
                return name
            return None
        if item_type == "function_call_output":
            call_id = item.get("call_id")
            if isinstance(call_id, str):
                record = pair_lookup.get(call_id, {})
                name = record.get("name")
                if isinstance(name, str):
                    return name
            return None
        if item.get("role") == "tool":
            name = item.get("name")
            if isinstance(name, str):
                return name
        return None

    def _item_text(self, item: TResponseInputItem) -> str:
        fragments = collect_text_fragments(item)
        return "\n".join(fragment.strip() for fragment in fragments if fragment.strip())

    def _is_stage_summary_item(self, item: TResponseInputItem) -> bool:
        return self._item_text(item).startswith("[Stage Summary]")

    def _looks_like_validation_item(self, item: TResponseInputItem) -> bool:
        text = self._item_text(item).lower()
        if not text:
            return False
        return looks_like_validation_text(text)

    def _build_context_diagnostics(
        self,
        items: list[TResponseInputItem],
        context_diagnostics: dict[str, Any] | None,
    ) -> dict[str, int]:
        """Compute lightweight context diagnostics for logs and W&B."""
        total_estimated_tokens = sum(
            self._estimate_item_tokens(item)
            for item in items
            if not (isinstance(item, dict) and item.get("role") == "system")
        )
        pair_lookup = self._build_call_pair_lookup(items)
        noisy_tool_tokens = 0
        read_tool_tokens = 0
        for item in items:
            tool_name = self._tool_name_for_item(item, pair_lookup)
            if tool_name is None:
                continue
            item_tokens = self._estimate_item_tokens(item)
            if tool_name in {"compile", "run", "shell"}:
                noisy_tool_tokens += item_tokens
            if tool_name == "read_file":
                read_tool_tokens += item_tokens
        tool_bloat_tokens = 0
        read_bloat_tokens = 0
        if total_estimated_tokens > 0:
            if noisy_tool_tokens > 10_000 and noisy_tool_tokens / total_estimated_tokens > 0.15:
                tool_bloat_tokens = noisy_tool_tokens
            if read_tool_tokens > 10_000 and read_tool_tokens / total_estimated_tokens > 0.05:
                read_bloat_tokens = read_tool_tokens
        near_capacity = 0
        if context_diagnostics is not None and context_diagnostics.get("is_above_warning_threshold"):
            near_capacity = 1
        return {
            "context/near_capacity": near_capacity,
            "context/tool_bloat_tokens": tool_bloat_tokens,
            "context/read_bloat_tokens": read_bloat_tokens,
        }

    def _estimate_item_tokens(self, item: TResponseInputItem) -> int:
        return estimate_message_tokens(item)

    def _estimate_items_chars(self, items: list[TResponseInputItem]) -> int:
        return sum(len(fragment) for item in items for fragment in collect_text_fragments(item))

    def _max_summary_input_tokens(self, model: str) -> int:
        try:
            context_window = get_context_window(model)
        except KeyError:
            context_window = 200_000
        return max(12_000, min(60_000, context_window // 4))

    def _chunk_items_for_summary(
        self,
        items: list[TResponseInputItem],
        model: str,
    ) -> list[list[TResponseInputItem]]:
        del model
        body_budget = COMPACTION_SUMMARY_INPUT_BODY_BYTES
        chunks: list[list[TResponseInputItem]] = []
        current_chunk: list[TResponseInputItem] = []
        current_bytes = 0
        for item in items:
            item_bytes = estimate_json_bytes(self._summary_record_for_item(item, len(current_chunk)))
            if current_chunk and current_bytes + item_bytes > body_budget:
                chunks.append(current_chunk)
                current_chunk = []
                current_bytes = 0
            current_chunk.append(item)
            current_bytes += item_bytes
        if current_chunk:
            chunks.append(current_chunk)
        return chunks

    def _summary_record_for_item(
        self,
        item: TResponseInputItem,
        index: int,
    ) -> dict[str, Any]:
        """Build one bounded record for compaction input."""
        text = "\n".join(collect_text_fragments(item))
        artifact_refs = _extract_source_refs_from_text(text)
        record = {
            "source_ref": f"item:{index}",
            "item_type": item.get("type") if isinstance(item, dict) else type(item).__name__,
            "role": item.get("role") if isinstance(item, dict) else None,
            "tool_name": item.get("name") if isinstance(item, dict) else None,
            "call_id": item.get("call_id") if isinstance(item, dict) else None,
            "artifact_refs": artifact_refs,
            "text_preview": _bounded_text(text, 4_000),
        }
        return record

    def _summary_payload_for_items(
        self,
        items: list[TResponseInputItem],
    ) -> tuple[dict[str, Any], tuple[str, ...]]:
        """Build bounded compaction input and the required source refs."""
        records = [self._summary_record_for_item(item, index) for index, item in enumerate(items)]
        refs: list[str] = []
        for record in records:
            refs.append(str(record["source_ref"]))
            refs.extend(str(ref) for ref in record.get("artifact_refs", ()))
        unique_refs = tuple(dict.fromkeys(refs))
        payload = {
            "schema": "stale_dialogue_compaction_v3",
            "rules": [
                "Summarize stale dialogue only.",
                "Do not summarize raw evidence as facts unless a source_ref is present.",
                "Every fact must cite one of the supplied source_refs.",
                "Keep active scope, decisions, open failures, validation contracts, Q1/Q9 obligations, files, artifacts, and next action explicit.",
            ],
            "source_refs": list(unique_refs),
            "items": records,
        }
        return payload, unique_refs
    
    async def _generate_summary(
        self, items: list[TResponseInputItem], model: str
    ) -> str:
        """Generate structured summary via LiteLLM.
        
        Args:
            items: Items to summarize
            model: Model to use for summary generation
            
        Returns:
            Formatted summary string
            
        Raises:
            Exception: If LLM call fails
        """
        payload, source_refs = self._summary_payload_for_items(items)
        messages = [{
            "role": "system",
            "content": (
                f"{COMPACT_SYSTEM_PROMPT}\n\n"
                "Additional hard requirement: output a [Compaction Summary v3] block "
                "with fields covered_range, active_scope, decisions, open_failures, "
                "validation_contracts, q1_q9_obligations, files_touched, artifacts, "
                "next_required_action, and source_refs."
            ),
        }]
        items_text = json.dumps(payload, indent=2, ensure_ascii=False, default=str)
        messages.append({
            "role": "user",
            "content": (
                "Summarize these stale dialogue records. Use only the provided "
                f"source_refs and keep raw evidence out of the summary.\n\n{items_text}"
            ),
        })
        
        # Cap output tokens: use COMPACT_MAX_OUTPUT_TOKENS unless the model's
        # context window is small enough that 20K output would be excessive.
        # Heuristic: allow at most 1/5 of context window for summary output.
        try:
            context_window = get_context_window(model)
            max_output = min(COMPACT_MAX_OUTPUT_TOKENS, context_window // 5)
        except KeyError:
            max_output = COMPACT_MAX_OUTPUT_TOKENS
        
        logger.debug(f"Calling {model} for compaction summary (max_tokens={max_output})")
        
        raw_content: Any = None
        content_type: str = "unavailable"
        content_seen = False
        try:
            async def _request_summary() -> Any:
                return await litellm.acompletion(
                    model=model,
                    messages=messages,
                    api_key=self.api_key,
                    base_url=self.base_url,
                    max_tokens=max_output,
                    **self._deepseek_compaction_kwargs(model),
                )

            response = await run_with_transient_retry(
                operation_name=f"litellm compaction request ({model})",
                operation=_request_summary,
                logger=logger,
            )

            raw_content = response.choices[0].message.content
            content_type = type(raw_content).__name__
            content_seen = True
            normalized_content = self._normalize_compaction_content(raw_content)
            summary = format_compaction_summary_v3(
                normalized_content,
                source_refs=source_refs,
            )
            validate_compaction_summary_v3(summary)
            return summary

        except Exception as e:
            logger.warning(
                "LiteLLM compaction call failed for model=%s content_type=%s empty_content=%s: %s",
                model,
                content_type,
                content_seen and not self._has_text_content(raw_content),
                e,
            )
            raise

    def _normalize_compaction_content(self, content: Any) -> str:
        """Normalize LiteLLM compaction content into plain text ready for summary formatting."""
        fragments = collect_text_fragments(content)
        normalized = "\n".join(
            fragment.strip() for fragment in fragments if fragment.strip()
        ).strip()
        if normalized:
            return normalized
        raise RuntimeError("Compaction model returned empty content")

    def _has_text_content(self, value: Any) -> bool:
        """Return whether LiteLLM content contains any visible text."""
        return bool(collect_text_fragments(value))
    
    async def run_compaction(
        self, args: OpenAIResponsesCompactionArgs | None = None
    ) -> CompactionAttemptResult:
        """Run compaction using LiteLLM-based summary generation.

        This method:
        1. Splits items into compact-candidates and preserved-recent
        2. Checks cache for existing result
        3. Calls LLM via LiteLLM if needed
        4. Replaces old items with [summary] + [preserved recent]

        Args:
            args: Optional compaction arguments. Supported keys:
                - force_trigger (bool): Force compaction to run even with few items
                - force_regenerate (bool): Force summary regeneration, bypassing cache
                - keep_recent (int): Number of recent items to preserve
        """
        if self._underlying_session is None:
            raise RuntimeError("No underlying session set. Call set_underlying_session() first.")

        force_trigger = args.get("force_trigger", False) if args else False
        force_regenerate = args.get("force_regenerate", False) if args else False
        keep_recent = args.get("keep_recent", 0) if args else 0
        selection_policy = args.get("selection_policy") if args else None
        preserve_limit_items = args.get("preserve_limit_items", 12) if args else 12
        min_candidate_items = args.get("min_candidate_items", 0) if args else 0
        pre_compact_tokens = args.get("pre_compact_tokens", 0) if args else 0
        trigger_threshold = args.get("trigger_threshold", 0) if args else 0
        context_diagnostics = args.get("context_diagnostics") if args else None
        compaction_model = self._get_compaction_model()

        all_items = await self._underlying_session.get_items()
        pre_session_tokens = sum(self._estimate_item_tokens(item) for item in all_items)
        pre_session_body_bytes = estimate_json_bytes(all_items)
        selection = self._select_items(
            items=all_items,
            keep_recent=keep_recent,
            selection_policy=selection_policy,
            preserve_limit_items=preserve_limit_items,
            min_candidate_items=min_candidate_items,
        )
        candidate_items = selection.candidate_items
        preserved_items = selection.preserved_items
        estimated_candidate_tokens = sum(
            self._estimate_item_tokens(item) for item in candidate_items
        )
        estimated_candidate_chars = self._estimate_items_chars(candidate_items)
        chunk_count = 0

        if len(candidate_items) < 2 and not force_trigger:
            logger.debug("Skipping compaction: insufficient items")
            attempt = CompactionAttemptResult(
                status="skipped",
                candidate_count=selection.candidate_count,
                preserved_count=selection.preserved_count,
                skip_reason=selection.skip_reason or "insufficient_items",
                selection_policy=selection.selection_policy,
                estimated_candidate_tokens=estimated_candidate_tokens,
                estimated_candidate_chars=estimated_candidate_chars,
                pre_tokens=pre_session_tokens,
                post_tokens=pre_session_tokens,
                pre_body_bytes=pre_session_body_bytes,
                post_body_bytes=pre_session_body_bytes,
                reduction_ratio=0.0,
                effective=False,
            )
            self._log_compaction_metrics(
                attempt=attempt,
                output_items=None,
                source_items=all_items,
                model=compaction_model,
                pre_compact_tokens=pre_compact_tokens,
                trigger_threshold=trigger_threshold,
                context_diagnostics=context_diagnostics,
            )
            return attempt

        # Compute cache key
        session_hash = self._compute_session_hash(candidate_items)
        cache_path = self._get_cache_path(session_hash, compaction_model)

        # Try to load from cache (respect force_regenerate)
        summary_items: list[TResponseInputItem] = []

        if cache_path.exists() and not force_regenerate:
            logger.debug(f"Loading compaction from cache: {cache_path}")
            cached = utils.load_pickle(cache_path, CompactCacheType)
            if cached is not None:
                summary_items = cached.output_items

        # Generate summary if not cached or force_regenerate
        if not summary_items:
            logger.info(
                f"Running compaction with {compaction_model} "
                f"({len(candidate_items)} items, preserving {len(preserved_items)}, "
                f"policy={selection.selection_policy or 'keep_recent'})"
            )
            chunked_candidate_items = self._chunk_items_for_summary(
                candidate_items,
                compaction_model,
            )
            chunk_count = len(chunked_candidate_items)

            try:
                if chunk_count <= 1:
                    summary = await self._generate_summary(candidate_items, compaction_model)
                    summary_item: TResponseInputItem = {
                        "role": "user",
                        "content": summary,
                    }
                    summary_items = [summary_item]
                else:
                    summary_items = []
                    for index, chunk in enumerate(chunked_candidate_items, start=1):
                        summary = await self._generate_summary(chunk, compaction_model)
                        summary_items.append(
                            {
                                "role": "user",
                                "content": f"[Compaction Chunk {index}/{chunk_count}]\n{summary}",
                            }
                        )
            except Exception as e:
                logger.error(f"Compaction failed: {e}")
                attempt = CompactionAttemptResult(
                    status="failed",
                    candidate_count=selection.candidate_count,
                    preserved_count=selection.preserved_count,
                    skip_reason=str(e),
                    selection_policy=selection.selection_policy,
                    chunk_count=chunk_count,
                    estimated_candidate_tokens=estimated_candidate_tokens,
                    estimated_candidate_chars=estimated_candidate_chars,
                    pre_tokens=pre_session_tokens,
                    post_tokens=pre_session_tokens,
                    pre_body_bytes=pre_session_body_bytes,
                    post_body_bytes=pre_session_body_bytes,
                    reduction_ratio=0.0,
                    effective=False,
                )
                self._log_compaction_metrics(
                    attempt=attempt,
                    output_items=None,
                    source_items=all_items,
                    model=compaction_model,
                    pre_compact_tokens=pre_compact_tokens,
                    trigger_threshold=trigger_threshold,
                    context_diagnostics=context_diagnostics,
                )
                return attempt

        output_items = summary_items + preserved_items
        if not output_items:
            raise RuntimeError("Compaction produced no output items")

        post_session_tokens = sum(self._estimate_item_tokens(item) for item in output_items)
        post_session_body_bytes = estimate_json_bytes(output_items)
        reduction_ratio = 0.0
        if pre_session_tokens > 0:
            reduction_ratio = max(0.0, 1.0 - (post_session_tokens / pre_session_tokens))
        effective = (
            pre_session_tokens == 0
            or post_session_tokens <= int(pre_session_tokens * COMPACTION_EFFECTIVE_RATIO)
            or post_session_body_bytes <= int(pre_session_body_bytes * COMPACTION_EFFECTIVE_RATIO)
        )
        if not effective:
            logger.warning(
                "Compaction ineffective: tokens %s -> %s, body_bytes %s -> %s, policy=%s",
                pre_session_tokens,
                post_session_tokens,
                pre_session_body_bytes,
                post_session_body_bytes,
                selection.selection_policy or "keep_recent",
            )
            cache_path.unlink(missing_ok=True)

        attempt = CompactionAttemptResult(
            status="success",
            candidate_count=selection.candidate_count,
            preserved_count=selection.preserved_count,
            skip_reason=None,
            selection_policy=selection.selection_policy,
            chunk_count=max(chunk_count, len(summary_items)),
            estimated_candidate_tokens=estimated_candidate_tokens,
            estimated_candidate_chars=estimated_candidate_chars,
            pre_tokens=pre_session_tokens,
            post_tokens=post_session_tokens,
            pre_body_bytes=pre_session_body_bytes,
            post_body_bytes=post_session_body_bytes,
            reduction_ratio=reduction_ratio,
            effective=effective,
        )
        self._log_compaction_metrics(
            attempt=attempt,
            output_items=output_items,
            source_items=all_items,
            model=compaction_model,
            pre_compact_tokens=pre_compact_tokens,
            trigger_threshold=trigger_threshold,
            context_diagnostics=context_diagnostics,
        )
        if not effective:
            return attempt

        utils.dump_pickle(cache_path, CompactCacheType(summary_items))
        await self._underlying_session.clear_session()
        await self._underlying_session.add_items(output_items)
        logger.debug(
            f"Compaction complete: {len(candidate_items)} compacted, "
            f"{len(preserved_items)} preserved → {len(output_items)} total"
        )
        return attempt

    def _log_compaction_metrics(
        self,
        attempt: CompactionAttemptResult,
        output_items: list[TResponseInputItem] | None,
        source_items: list[TResponseInputItem],
        model: str,
        pre_compact_tokens: int,
        trigger_threshold: int,
        context_diagnostics: dict[str, Any] | None,
    ) -> None:
        """Log compaction attempt metrics with context diagnostics."""
        if self.wandb_metrics_hook is None:
            return None
        metrics = {
            "type": "compaction",
            "compaction/status": attempt.status,
            "compaction/output_items": len(output_items) if output_items is not None else 0,
            "compaction/candidate_items": attempt.candidate_count,
            "compaction/preserved_items": attempt.preserved_count,
            "compaction/chunk_count": attempt.chunk_count,
            "compaction/estimated_candidate_tokens": attempt.estimated_candidate_tokens,
            "compaction/estimated_candidate_chars": attempt.estimated_candidate_chars,
            "compaction/model": model,
            "compaction/pre_compact_tokens": pre_compact_tokens,
            "compaction/threshold": trigger_threshold,
            "compaction/fallback_applied": 0,
            "compaction/noop_after_threshold": int(
                attempt.status == "skipped"
                and pre_compact_tokens >= trigger_threshold > 0
            ),
            "compaction/skip_reason": attempt.skip_reason or "",
            "compaction/selection_policy": attempt.selection_policy or "keep_recent",
        }
        metrics.update(self._build_context_diagnostics(source_items, context_diagnostics))
        self.wandb_metrics_hook.log_metrics_callback(metrics, log_and_increment=True)
        return None


def format_compaction_summary_v3(
    content: str,
    *,
    source_refs: tuple[str, ...],
) -> str:
    """Normalize a compaction response into the required v3 memory block."""
    normalized = content.strip()
    if not normalized:
        raise ValueError("Compaction summary v3 requires non-empty content")
    if COMPACTION_V3_HEADER in normalized:
        summary = normalized
    else:
        summary = "\n".join([
            COMPACTION_V3_HEADER,
            "covered_range:",
            "  unknown",
            "active_scope:",
            "  unknown",
            "decisions:",
            _indent_lines(normalized, spaces=2),
            "open_failures:",
            "  []",
            "validation_contracts:",
            "  []",
            "q1_q9_obligations:",
            "  []",
            "files_touched:",
            "  []",
            "artifacts:",
            "  []",
            "next_required_action:",
            "  continue current stage objective",
        ])
    if "source_refs:" not in summary:
        summary = "\n".join([
            summary.rstrip(),
            "source_refs:",
            *[f"  - {ref}" for ref in source_refs[:40]],
        ])
    return summary


def validate_compaction_summary_v3(summary: str) -> None:
    """Raise when a compaction summary lacks the required v3 source refs."""
    if COMPACTION_V3_HEADER not in summary:
        raise ValueError("Compaction summary missing [Compaction Summary v3] header")
    if "source_refs:" not in summary:
        raise ValueError("Compaction summary missing source_refs field")
    required_fields = (
        "covered_range:",
        "active_scope:",
        "decisions:",
        "open_failures:",
        "validation_contracts:",
        "q1_q9_obligations:",
        "files_touched:",
        "artifacts:",
        "next_required_action:",
    )
    missing = [field for field in required_fields if field not in summary]
    if missing:
        raise ValueError(f"Compaction summary missing required v3 fields: {missing}")
    source_block = summary.split("source_refs:", 1)[1]
    if not any(line.strip().startswith("- ") for line in source_block.splitlines()):
        raise ValueError("Compaction summary source_refs field is empty")
    return None


def _bounded_text(text: str, limit: int) -> str:
    """Return a stable head/tail preview bounded by characters."""
    stripped = text.strip()
    if len(stripped) <= limit:
        return stripped
    keep = max(256, (limit - 80) // 2)
    omitted = max(0, len(stripped) - (2 * keep))
    return (
        f"{stripped[:keep]}\n"
        f"... [{omitted} chars omitted for compaction input] ...\n"
        f"{stripped[-keep:]}"
    )


def _extract_source_refs_from_text(text: str) -> list[str]:
    """Extract artifact and snapshot refs from text snippets."""
    refs: list[str] = []
    refs.extend(re.findall(r"artifact_ref[:=]\s*([A-Za-z0-9_.-]+)", text))
    refs.extend(re.findall(r"snapshot_hash[:=]\s*([A-Za-z0-9_.-]+)", text))
    refs.extend(f"query:{qid}" for qid in re.findall(r"\bQ(?:uery)?\s*([0-9]+)\b", text))
    return list(dict.fromkeys(refs))


def _indent_lines(text: str, *, spaces: int) -> str:
    """Indent text by a fixed number of spaces."""
    prefix = " " * spaces
    return "\n".join(prefix + line for line in text.splitlines())
