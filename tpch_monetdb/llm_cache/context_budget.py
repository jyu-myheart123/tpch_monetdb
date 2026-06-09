from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from agents import TResponseInputItem


BODY_WARN_BYTES = 6 * 1024 * 1024
BODY_COMPACT_BYTES = 8 * 1024 * 1024
BODY_FAIL_BYTES = int(9.5 * 1024 * 1024)
TOKEN_WARN_RATIO = 0.70
TOKEN_COMPACT_RATIO = 0.85
TOKEN_FAIL_RATIO = 0.95
CHARS_PER_TOKEN = 4
TOKEN_ESTIMATE_SAFETY = 4 / 3


@dataclass(frozen=True)
class ContextContributor:
    """Describe one large request-body contributor."""

    source: str
    item_index: int | None
    artifact_ref: str | None
    byte_size: int
    summary: str


@dataclass(frozen=True)
class RequestBudgetEstimate:
    """Report token/body budget state for one prospective model request."""

    session_item_count: int
    token_estimate: int
    token_limit: int
    token_level: str
    body_bytes: int
    body_warn_bytes: int
    body_compact_bytes: int
    body_fail_bytes: int
    body_level: str
    top_contributors: tuple[ContextContributor, ...]

    @property
    def body_warn(self) -> bool:
        """Return whether the serialized body crossed the warning threshold."""
        return self.body_level in {"yellow", "orange", "red"}

    @property
    def body_compact(self) -> bool:
        """Return whether deterministic trim should run before sending."""
        return self.body_level in {"orange", "red"}

    @property
    def body_fail(self) -> bool:
        """Return whether the provider request must fail closed."""
        return self.body_level == "red"

    @property
    def should_warn(self) -> bool:
        """Return whether this request should emit budget diagnostics."""
        return self.body_level in {"yellow", "orange", "red"} or self.token_level in {
            "yellow",
            "orange",
            "red",
        }

    @property
    def should_compact(self) -> bool:
        """Return whether deterministic trim or compaction should run first."""
        return self.body_level in {"orange", "red"} or self.token_level in {"orange", "red"}

    @property
    def should_fail(self) -> bool:
        """Return whether sending should be blocked until the request shrinks."""
        return self.body_level == "red" or self.token_level == "red"


class ContextBudgetManager:
    """Small reusable manager for token/body budget estimates."""

    def __init__(
        self,
        *,
        token_limit: int,
        body_warn_bytes: int = BODY_WARN_BYTES,
        body_compact_bytes: int = BODY_COMPACT_BYTES,
        body_fail_bytes: int = BODY_FAIL_BYTES,
    ) -> None:
        self.token_limit = token_limit
        self.body_warn_bytes = body_warn_bytes
        self.body_compact_bytes = body_compact_bytes
        self.body_fail_bytes = body_fail_bytes
        return None

    def estimate_session_request(
        self,
        items: list[TResponseInputItem],
        *,
        new_input: str,
    ) -> RequestBudgetEstimate:
        """Estimate a session request before SDK provider conversion."""
        return build_request_budget_estimate(
            items,
            new_input=new_input,
            token_limit=self.token_limit,
            body_warn_bytes=self.body_warn_bytes,
            body_compact_bytes=self.body_compact_bytes,
            body_fail_bytes=self.body_fail_bytes,
        )

    def estimate_provider_request(
        self,
        payload: dict[str, Any],
    ) -> RequestBudgetEstimate:
        """Estimate a provider request after message/tool conversion."""
        return build_provider_request_budget_estimate(
            payload,
            token_limit=self.token_limit,
            body_warn_bytes=self.body_warn_bytes,
            body_compact_bytes=self.body_compact_bytes,
            body_fail_bytes=self.body_fail_bytes,
        )


def estimate_json_bytes(value: Any) -> int:
    """Estimate request body bytes using actual JSON serialization."""
    return len(json.dumps(value, ensure_ascii=False, default=str).encode("utf-8"))


def estimate_token_count(text: str) -> int:
    """Estimate token count using the existing conservative chars/3 heuristic."""
    if not text:
        return 0
    return int((len(text) / CHARS_PER_TOKEN * TOKEN_ESTIMATE_SAFETY) + 0.999)


def estimate_item_tokens(item: TResponseInputItem) -> int:
    """Estimate tokens for one structured session item."""
    return sum(estimate_token_count(fragment) for fragment in _collect_text_fragments(item))


def estimate_session_tokens(items: list[TResponseInputItem]) -> int:
    """Estimate tokens for a list of session items."""
    return sum(estimate_item_tokens(item) for item in items)


def estimate_session_body_bytes(
    items: list[TResponseInputItem],
    *,
    new_input: str,
) -> int:
    """Estimate serialized bytes for session items plus the next input."""
    payload = {
        "input": items,
        "new_input": new_input,
    }
    return estimate_json_bytes(payload)


def estimate_provider_body_bytes(payload: dict[str, Any]) -> int:
    """Estimate provider HTTP body bytes from the prepared LiteLLM payload."""
    return estimate_json_bytes(payload)


def build_provider_request_budget_estimate(
    payload: dict[str, Any],
    *,
    token_limit: int,
    body_warn_bytes: int = BODY_WARN_BYTES,
    body_compact_bytes: int = BODY_COMPACT_BYTES,
    body_fail_bytes: int = BODY_FAIL_BYTES,
) -> RequestBudgetEstimate:
    """Build a budget estimate from the actual provider request payload."""
    messages = payload.get("messages")
    message_items = messages if isinstance(messages, list) else []
    payload_text = json.dumps(payload, ensure_ascii=False, default=str)
    token_estimate = estimate_token_count(payload_text)
    body_bytes = estimate_provider_body_bytes(payload)
    return RequestBudgetEstimate(
        session_item_count=len(message_items),
        token_estimate=token_estimate,
        token_limit=token_limit,
        token_level=_token_level(token_estimate, token_limit),
        body_bytes=body_bytes,
        body_warn_bytes=body_warn_bytes,
        body_compact_bytes=body_compact_bytes,
        body_fail_bytes=body_fail_bytes,
        body_level=_body_level(
            body_bytes,
            warn_bytes=body_warn_bytes,
            compact_bytes=body_compact_bytes,
            fail_bytes=body_fail_bytes,
        ),
        top_contributors=collect_provider_contributors(payload),
    )


def build_request_budget_estimate(
    items: list[TResponseInputItem],
    *,
    new_input: str,
    token_limit: int,
    body_warn_bytes: int = BODY_WARN_BYTES,
    body_compact_bytes: int = BODY_COMPACT_BYTES,
    body_fail_bytes: int = BODY_FAIL_BYTES,
) -> RequestBudgetEstimate:
    """Build a token/body budget estimate from session items and new input."""
    token_estimate = estimate_session_tokens(items) + estimate_token_count(new_input)
    body_bytes = estimate_session_body_bytes(items, new_input=new_input)
    return RequestBudgetEstimate(
        session_item_count=len(items),
        token_estimate=token_estimate,
        token_limit=token_limit,
        token_level=_token_level(token_estimate, token_limit),
        body_bytes=body_bytes,
        body_warn_bytes=body_warn_bytes,
        body_compact_bytes=body_compact_bytes,
        body_fail_bytes=body_fail_bytes,
        body_level=_body_level(
            body_bytes,
            warn_bytes=body_warn_bytes,
            compact_bytes=body_compact_bytes,
            fail_bytes=body_fail_bytes,
        ),
        top_contributors=collect_context_contributors(items, new_input=new_input),
    )


def collect_context_contributors(
    items: list[TResponseInputItem],
    *,
    new_input: str,
    limit: int = 5,
) -> tuple[ContextContributor, ...]:
    """Return the largest serialized contributors in the next request."""
    contributors: list[ContextContributor] = []
    for index, item in enumerate(items):
        byte_size = estimate_json_bytes(item)
        contributors.append(
            ContextContributor(
                source=_item_source(item),
                item_index=index,
                artifact_ref=_artifact_ref_from_item(item),
                byte_size=byte_size,
                summary=_summarize_item(item),
            )
        )
    if new_input:
        contributors.extend(
            ContextContributor(
                source=source,
                item_index=None,
                artifact_ref=_artifact_ref_from_text(block),
                byte_size=len(block.encode("utf-8")),
                summary=block.strip().splitlines()[0][:160] if block.strip() else "",
            )
            for source, block in _split_new_input_blocks(new_input)
        )
    ordered = sorted(contributors, key=lambda item: item.byte_size, reverse=True)
    return tuple(ordered[:limit])


def collect_provider_contributors(
    payload: dict[str, Any],
    *,
    limit: int = 5,
) -> tuple[ContextContributor, ...]:
    """Return the largest contributors inside the provider request payload."""
    contributors: list[ContextContributor] = []
    messages = payload.get("messages")
    if isinstance(messages, list):
        for index, message in enumerate(messages):
            contributors.append(
                ContextContributor(
                    source=f"provider.messages.{_item_source(message)}",
                    item_index=index,
                    artifact_ref=_artifact_ref_from_item(message),
                    byte_size=estimate_json_bytes(message),
                    summary=_summarize_item(message),
                )
            )
    tools = payload.get("tools")
    if tools is not None:
        contributors.append(
            ContextContributor(
                source="provider.tools",
                item_index=None,
                artifact_ref=None,
                byte_size=estimate_json_bytes(tools),
                summary="serialized tool schema",
            )
        )
    headers = payload.get("__http_headers__")
    if headers is not None:
        contributors.append(
            ContextContributor(
                source="provider.headers",
                item_index=None,
                artifact_ref=None,
                byte_size=estimate_json_bytes(headers),
                summary="serialized HTTP headers for wire-budget estimate",
            )
        )
    settings_payload = {
        key: value
        for key, value in payload.items()
        if key not in {"messages", "tools", "__http_headers__"}
    }
    if settings_payload:
        contributors.append(
            ContextContributor(
                source="provider.settings",
                item_index=None,
                artifact_ref=None,
                byte_size=estimate_json_bytes(settings_payload),
                summary=_summarize_item(settings_payload),
            )
        )
    ordered = sorted(contributors, key=lambda item: item.byte_size, reverse=True)
    return tuple(ordered[:limit])


def _body_level(
    body_bytes: int,
    *,
    warn_bytes: int,
    compact_bytes: int,
    fail_bytes: int,
) -> str:
    """Classify body budget pressure into green/yellow/orange/red."""
    if body_bytes >= fail_bytes:
        return "red"
    if body_bytes >= compact_bytes:
        return "orange"
    if body_bytes >= warn_bytes:
        return "yellow"
    return "green"


def _token_level(token_estimate: int, token_limit: int) -> str:
    """Classify token budget pressure into green/yellow/orange/red."""
    if token_limit <= 0:
        return "green"
    ratio = token_estimate / token_limit
    if ratio >= TOKEN_FAIL_RATIO:
        return "red"
    if ratio >= TOKEN_COMPACT_RATIO:
        return "orange"
    if ratio >= TOKEN_WARN_RATIO:
        return "yellow"
    return "green"


def _item_source(item: TResponseInputItem) -> str:
    """Return a stable contributor source label for one session item."""
    text = json.dumps(item, ensure_ascii=False, default=str)
    if "[Stage Memory v3]" in text:
        return "stage_memory"
    if "[Artifact Refs]" in text or "artifact_ref=" in text:
        return "artifact_refs"
    if "[Evidence Digest]" in text or "artifact_ref:" in text:
        return "tool_output_digest"
    if isinstance(item, dict):
        if item.get("type") == "function_call_output" or item.get("role") == "tool":
            return "tool_output"
        if item.get("type"):
            return str(item.get("type"))
        if item.get("role"):
            return str(item.get("role"))
    return type(item).__name__


def _artifact_ref_from_item(item: TResponseInputItem) -> str | None:
    """Extract an artifact ref from an evidence digest item when present."""
    text = json.dumps(item, ensure_ascii=False, default=str)
    return _artifact_ref_from_text(text)


def _artifact_ref_from_text(text: str) -> str | None:
    """Extract an artifact ref from digest/ref text when present."""
    marker_pairs = (
        ("artifact_ref:", "artifact_ref="),
    )
    after = None
    for marker, equals_marker in marker_pairs:
        if marker in text:
            after = text.split(marker, 1)[1].strip()
            break
        if equals_marker in text:
            after = text.split(equals_marker, 1)[1].strip()
            break
    if after is None:
        return None
    return after.split()[0] if after else None


def _split_new_input_blocks(new_input: str) -> list[tuple[str, str]]:
    """Split contextual prompt input into diagnostic contributor blocks."""
    markers = (
        ("runtime_stage_hint", "[Runtime Stage Hint]"),
        ("scoped_stage_rules", "[Scoped Stage Rules]"),
        ("new_input", "[Current Task]"),
        ("stage_memory", "[Stage Memory v3]"),
        ("artifact_refs", "[Artifact Refs]"),
    )
    positions = [
        (index, source, marker)
        for source, marker in markers
        if (index := new_input.find(marker)) >= 0
    ]
    if not positions:
        return [("new_input", new_input)]
    ordered = sorted(positions, key=lambda item: item[0])
    blocks: list[tuple[str, str]] = []
    for pos, source, _marker in ordered:
        next_positions = [candidate[0] for candidate in ordered if candidate[0] > pos]
        end = min(next_positions) if next_positions else len(new_input)
        block = new_input[pos:end].strip()
        if block:
            blocks.append((source, block))
    return blocks


def _summarize_item(item: TResponseInputItem) -> str:
    """Return a short contributor summary for diagnostics."""
    text = json.dumps(item, ensure_ascii=False, default=str)
    compact = " ".join(text.split())
    return compact[:180]


def _collect_text_fragments(value: Any) -> list[str]:
    """Collect text fragments from SDK-style session items."""
    if isinstance(value, str):
        return [value]
    if value is None:
        return []
    if isinstance(value, list):
        fragments: list[str] = []
        for item in value:
            fragments.extend(_collect_text_fragments(item))
        return fragments
    if isinstance(value, dict):
        fragments = []
        for key in ("content", "text", "value", "message", "output", "name", "arguments", "summary"):
            if key in value:
                fragments.extend(_collect_text_fragments(value[key]))
        return fragments
    return []
