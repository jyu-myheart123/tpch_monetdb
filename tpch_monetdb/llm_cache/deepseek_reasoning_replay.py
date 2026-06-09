import logging
from collections.abc import Mapping, Sequence
from typing import Any, cast

from agents.items import TResponseInputItem, TResponseOutputItem
from agents.models.fake_id import FAKE_RESPONSES_ID
from openai.types.responses import ResponseFunctionToolCall, ResponseReasoningItem
from openai.types.responses.response_reasoning_item import Summary

from tpch_monetdb.utils.model_aliases import is_deepseek_model

logger = logging.getLogger(__name__)

_SYNTHETIC_REASONING_PLACEHOLDER = (
    "[Minimal thinking placeholder — DeepSeek API returned no "
    "reasoning_content for this turn]"
)


class DeepSeekReasoningReplayError(RuntimeError):
    pass


class DeepSeekReasoningReplayTransientError(DeepSeekReasoningReplayError):
    pass


def extract_reasoning_content_from_message(message: Any) -> str | None:
    for attr in ("reasoning_content", "reasoning", "thinking"):
        normalized = _extract_reasoning_text(getattr(message, attr, None))
        if normalized is not None:
            return normalized
    getter = getattr(message, "get", None)
    if callable(getter):
        provider_specific_fields = getter("provider_specific_fields", None)
        if isinstance(provider_specific_fields, Mapping):
            for key in ("reasoning_content", "reasoning", "thinking"):
                normalized = _extract_reasoning_text(provider_specific_fields.get(key))
                if normalized is not None:
                    return normalized
        for key in ("reasoning_content", "reasoning", "thinking"):
            normalized = _extract_reasoning_text(getter(key, None))
            if normalized is not None:
                return normalized
    return None


def ensure_deepseek_response_output(
    output_items: Sequence[TResponseOutputItem],
    *,
    model_name: str,
    fallback_reasoning_content: str | None,
    response_id: str | None,
    require_reasoning_content: bool,
) -> list[TResponseOutputItem]:
    """Normalize DeepSeek tool-call output so continuation can always replay reasoning."""
    items = list(output_items)
    if not is_deepseek_model(model_name):
        return items
    if not _has_function_call_output_item(items):
        return items
    reasoning_content = _extract_reasoning_content_from_output_items(items)
    if reasoning_content is None:
        reasoning_content = _extract_reasoning_backup_from_output_items(items)
    if reasoning_content is None:
        reasoning_content = _normalize_reasoning_content(fallback_reasoning_content)
    if reasoning_content is None:
        if not require_reasoning_content:
            return items
        logger.warning(
            "DeepSeek tool-call response missing reasoning_content; "
            "injecting synthetic placeholder. model=%s response_id=%s",
            model_name,
            response_id,
        )
        reasoning_content = _SYNTHETIC_REASONING_PLACEHOLDER
    if not _has_reasoning_output_item(items):
        items = [
            _build_reasoning_output_item(
                reasoning_content,
                model_name=model_name,
                response_id=response_id,
            )
        ] + items
    return _attach_reasoning_backup_to_output_items(
        items,
        reasoning_content=reasoning_content,
        model_name=model_name,
        response_id=response_id,
    )


def repair_deepseek_input_items(
    items: Sequence[TResponseInputItem],
    *,
    model_name: str,
    fail_on_unrecoverable: bool = False,
    require_reasoning_for_tool_calls: bool = False,
) -> list[TResponseInputItem]:
    """Repair persisted DeepSeek assistant tool-call turns that lost their reasoning item."""
    if not is_deepseek_model(model_name):
        return list(items)
    repaired: list[TResponseInputItem] = []
    history_requires_reasoning = require_reasoning_for_tool_calls or (
        fail_on_unrecoverable and _history_contains_reasoning_artifacts(items)
    )
    assistant_segment: list[TResponseInputItem] = []
    for item in items:
        if _is_assistant_segment_item(item):
            assistant_segment.append(item)
            continue
        repaired.extend(
            _repair_deepseek_assistant_segment(
                assistant_segment,
                model_name=model_name,
                fail_on_unrecoverable=history_requires_reasoning,
            )
        )
        assistant_segment = []
        repaired.append(item)
    repaired.extend(
        _repair_deepseek_assistant_segment(
            assistant_segment,
            model_name=model_name,
            fail_on_unrecoverable=history_requires_reasoning,
        )
    )
    return repaired


def _repair_deepseek_assistant_segment(
    segment: Sequence[TResponseInputItem],
    *,
    model_name: str,
    fail_on_unrecoverable: bool,
) -> list[TResponseInputItem]:
    items = list(segment)
    if not items or not _has_function_call_input_item(items):
        return items
    if _has_reasoning_input_item(items):
        return items
    reasoning_content = _extract_reasoning_backup_from_input_items(items)
    response_id = _extract_response_id_from_input_items(items)
    if reasoning_content is None:
        if fail_on_unrecoverable:
            logger.warning(
                "DeepSeek persisted tool-call turn missing reasoning backup; "
                "injecting synthetic placeholder. model=%s response_id=%s source=%s",
                model_name,
                response_id,
                "injected_placeholder",
            )
            reasoning_content = _SYNTHETIC_REASONING_PLACEHOLDER
        else:
            return items
    else:
        logger.warning(
            "DeepSeek persisted tool-call turn missing reasoning item; "
            "restoring from provider backup. model=%s response_id=%s source=%s",
            model_name,
            response_id,
            "recovered_from_provider_backup",
        )
    return [
        _build_reasoning_input_item(
            reasoning_content,
            model_name=model_name,
            response_id=response_id,
        )
    ] + items


def ensure_deepseek_assistant_messages_have_reasoning_content(
    messages: Sequence[dict[str, Any]],
    *,
    model_name: str,
) -> list[dict[str, Any]]:
    """Patch outgoing DeepSeek assistant tool-call messages before provider submission."""
    if not is_deepseek_model(model_name):
        return list(messages)

    repaired: list[dict[str, Any]] = []
    for message in messages:
        if not _assistant_tool_call_message_needs_reasoning_content(message):
            repaired.append(message)
            continue

        updated = dict(message)
        updated["reasoning_content"] = _SYNTHETIC_REASONING_PLACEHOLDER
        logger.warning(
            "DeepSeek outgoing assistant tool-call message missing reasoning_content; "
            "injecting synthetic placeholder before provider request. model=%s source=%s",
            model_name,
            "injected_placeholder",
        )
        repaired.append(updated)

    return repaired


def _attach_reasoning_backup_to_output_items(
    output_items: Sequence[TResponseOutputItem],
    *,
    reasoning_content: str,
    model_name: str,
    response_id: str | None,
) -> list[TResponseOutputItem]:
    updated_items: list[TResponseOutputItem] = []
    for item in output_items:
        if isinstance(item, ResponseFunctionToolCall):
            provider_data = _merge_provider_data(
                getattr(item, "provider_data", None),
                reasoning_content=reasoning_content,
                model_name=model_name,
                response_id=response_id,
            )
            updated_items.append(item.model_copy(update={"provider_data": provider_data}))
            continue
        updated_items.append(item)
    return updated_items


def _build_reasoning_output_item(
    reasoning_content: str,
    *,
    model_name: str,
    response_id: str | None,
) -> ResponseReasoningItem:
    provider_data = _merge_provider_data(
        None,
        reasoning_content=None,
        model_name=model_name,
        response_id=response_id,
    )
    return ResponseReasoningItem(
        id=FAKE_RESPONSES_ID,
        summary=[Summary(text=reasoning_content, type="summary_text")],
        type="reasoning",
        provider_data=provider_data,
    )


def _build_reasoning_input_item(
    reasoning_content: str,
    *,
    model_name: str,
    response_id: str | None,
) -> TResponseInputItem:
    provider_data = _merge_provider_data(
        None,
        reasoning_content=None,
        model_name=model_name,
        response_id=response_id,
    )
    return cast(
        TResponseInputItem,
        {
            "id": FAKE_RESPONSES_ID,
            "summary": [{"text": reasoning_content, "type": "summary_text"}],
            "type": "reasoning",
            "provider_data": provider_data,
        },
    )


def _merge_provider_data(
    provider_data: Any,
    *,
    reasoning_content: str | None,
    model_name: str,
    response_id: str | None,
) -> dict[str, Any]:
    merged = dict(provider_data) if isinstance(provider_data, Mapping) else {}
    merged["model"] = merged.get("model") or model_name
    if response_id is not None and "response_id" not in merged:
        merged["response_id"] = response_id
    if reasoning_content is not None:
        merged["reasoning_content"] = reasoning_content
    return merged


def _extract_reasoning_content_from_output_items(
    output_items: Sequence[TResponseOutputItem],
) -> str | None:
    for item in output_items:
        if not isinstance(item, ResponseReasoningItem):
            continue
        texts = [
            summary.text
            for summary in item.summary
            if getattr(summary, "text", None)
        ]
        if texts:
            return "\n".join(texts)
    return None


def _extract_reasoning_backup_from_output_items(
    output_items: Sequence[TResponseOutputItem],
) -> str | None:
    for item in output_items:
        if not isinstance(item, ResponseFunctionToolCall):
            continue
        provider_data = getattr(item, "provider_data", None)
        if isinstance(provider_data, Mapping):
            for key in ("reasoning_content", "reasoning", "thinking"):
                reasoning_content = _normalize_reasoning_content(
                    provider_data.get(key)
                )
                if reasoning_content is not None:
                    return reasoning_content
    return None


def _extract_reasoning_backup_from_input_items(
    items: Sequence[TResponseInputItem],
) -> str | None:
    for item in items:
        if not isinstance(item, dict):
            continue
        if item.get("type") != "function_call":
            continue
        provider_data = item.get("provider_data")
        if isinstance(provider_data, Mapping):
            for key in ("reasoning_content", "reasoning", "thinking"):
                reasoning_content = _normalize_reasoning_content(
                    provider_data.get(key)
                )
                if reasoning_content is not None:
                    return reasoning_content
    return None


def _extract_response_id_from_input_items(
    items: Sequence[TResponseInputItem],
) -> str | None:
    for item in items:
        if not isinstance(item, dict):
            continue
        provider_data = item.get("provider_data")
        if isinstance(provider_data, Mapping):
            response_id = provider_data.get("response_id")
            if isinstance(response_id, str) and response_id:
                return response_id
    return None


def _has_function_call_output_item(output_items: Sequence[TResponseOutputItem]) -> bool:
    return any(isinstance(item, ResponseFunctionToolCall) for item in output_items)


def _has_function_call_input_item(items: Sequence[TResponseInputItem]) -> bool:
    for item in items:
        if isinstance(item, dict) and item.get("type") == "function_call":
            return True
    return False


def _has_reasoning_output_item(output_items: Sequence[TResponseOutputItem]) -> bool:
    return any(isinstance(item, ResponseReasoningItem) for item in output_items)


def _has_reasoning_input_item(items: Sequence[TResponseInputItem]) -> bool:
    for item in items:
        if isinstance(item, dict) and item.get("type") == "reasoning":
            return True
    return False


def _assistant_tool_call_message_needs_reasoning_content(message: Mapping[str, Any]) -> bool:
    if message.get("role") != "assistant":
        return False
    tool_calls = message.get("tool_calls")
    if not isinstance(tool_calls, Sequence) or isinstance(
        tool_calls, (str, bytes, bytearray)
    ):
        return False
    if not tool_calls:
        return False
    return _normalize_reasoning_content(message.get("reasoning_content")) is None


def _history_contains_reasoning_artifacts(items: Sequence[TResponseInputItem]) -> bool:
    for item in items:
        if isinstance(item, dict) and item.get("type") == "reasoning":
            return True
        if not isinstance(item, dict) or item.get("type") != "function_call":
            continue
        provider_data = item.get("provider_data")
        if not isinstance(provider_data, Mapping):
            continue
        reasoning_content = _normalize_reasoning_content(
            provider_data.get("reasoning_content")
        )
        if reasoning_content is not None:
            return True
    return False


def _is_assistant_segment_item(item: TResponseInputItem) -> bool:
    if not isinstance(item, dict):
        return False
    item_type = item.get("type")
    if item_type in {"reasoning", "function_call"}:
        return True
    if item_type == "message" and item.get("role") == "assistant":
        return True
    return item.get("role") == "assistant"


def _normalize_reasoning_content(reasoning_content: Any) -> str | None:
    if not isinstance(reasoning_content, str):
        return None
    normalized = reasoning_content.strip()
    if not normalized:
        return None
    return normalized


def _extract_reasoning_text(value: Any) -> str | None:
    normalized = _normalize_reasoning_content(value)
    if normalized is not None:
        return normalized
    if value is None:
        return None
    if isinstance(value, Mapping):
        for key in (
            "reasoning_content",
            "reasoning",
            "thinking",
            "content",
            "text",
        ):
            normalized = _extract_reasoning_text(value.get(key))
            if normalized is not None:
                return normalized
        return None
    for attr in ("reasoning", "thinking", "content", "text"):
        normalized = _extract_reasoning_text(getattr(value, attr, None))
        if normalized is not None:
            return normalized
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        parts = [_extract_reasoning_text(item) for item in value]
        non_empty = [part for part in parts if part]
        if non_empty:
            return "\n".join(non_empty)
    return None
