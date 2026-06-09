from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agents import TResponseInputItem

from .context_budget import estimate_item_tokens, estimate_json_bytes, estimate_session_tokens


V3_LIFECYCLE_NAME = "TPC-H MonetDB Context Lifecycle v3"
STAGE_MEMORY_TAIL_ITEM_LIMIT = 64
STAGE_MEMORY_TAIL_TOKEN_LIMIT = 40_000
AGGRESSIVE_TAIL_ITEM_LIMIT = 24
AGGRESSIVE_TAIL_TOKEN_LIMIT = 16_000


@dataclass(frozen=True)
class LocalCompactResult:
    """Describe one local deterministic context lifecycle compaction pass."""

    items: list[TResponseInputItem]
    changed_count: int
    pre_tokens: int
    post_tokens: int
    pre_body_bytes: int
    post_body_bytes: int
    mode: str


def stage_memory_compact_items(
    items: list[TResponseInputItem],
    *,
    stage_memory: str | None = None,
    artifact_context: str | None = None,
    profile_name: str | None = None,
) -> LocalCompactResult:
    """Replace stale dialogue with v3 stage memory plus a bounded recent tail."""
    pre_tokens = estimate_session_tokens(items)
    pre_body_bytes = estimate_json_bytes(items)
    leading_system, non_system = _split_leading_system_items(items)
    memory_text = _select_stage_memory_text(non_system, stage_memory)
    if memory_text is None:
        return LocalCompactResult(
            items=items,
            changed_count=0,
            pre_tokens=pre_tokens,
            post_tokens=pre_tokens,
            pre_body_bytes=pre_body_bytes,
            post_body_bytes=pre_body_bytes,
            mode="stage_memory",
        )

    tail_items = _bounded_tail_items(
        non_system,
        max_items=STAGE_MEMORY_TAIL_ITEM_LIMIT,
        max_tokens=STAGE_MEMORY_TAIL_TOKEN_LIMIT,
        exclude_context_wrappers=True,
    )
    compacted_items = [
        *leading_system,
        _compact_boundary_item("stage_memory", profile_name),
        {"role": "user", "content": memory_text},
    ]
    artifact_item = _artifact_refs_item(artifact_context)
    if artifact_item is not None:
        compacted_items.append(artifact_item)
    compacted_items.extend(tail_items)
    return _local_compact_result(
        original=items,
        compacted=compacted_items,
        pre_tokens=pre_tokens,
        pre_body_bytes=pre_body_bytes,
        mode="stage_memory",
    )


def aggressive_compact_items(
    items: list[TResponseInputItem],
    *,
    stage_memory: str | None = None,
    artifact_context: str | None = None,
    profile_name: str | None = None,
) -> LocalCompactResult:
    """Keep only current state, latest validation/tool evidence, and minimal tail."""
    pre_tokens = estimate_session_tokens(items)
    pre_body_bytes = estimate_json_bytes(items)
    leading_system, non_system = _split_leading_system_items(items)
    memory_text = _select_stage_memory_text(non_system, stage_memory)
    selected_indices: set[int] = set()
    selected_indices.update(_latest_control_indices(non_system))
    selected_indices.update(_latest_validation_indices(non_system))
    selected_indices.update(_latest_tool_pair_indices(non_system, {"compile", "run"}))
    selected_indices.update(_recent_user_task_indices(non_system, limit=3))
    selected_items = [
        _compact_contextual_user_item(item)
        for index, item in enumerate(non_system)
        if index in selected_indices and not _is_stage_memory_item(item)
    ]
    selected_items = _bounded_tail_items(
        selected_items,
        max_items=AGGRESSIVE_TAIL_ITEM_LIMIT,
        max_tokens=AGGRESSIVE_TAIL_TOKEN_LIMIT,
        exclude_context_wrappers=False,
    )
    compacted_items = [
        *leading_system,
        _compact_boundary_item("aggressive", profile_name),
    ]
    if memory_text is not None:
        compacted_items.append({"role": "user", "content": memory_text})
    artifact_item = _artifact_refs_item(artifact_context)
    if artifact_item is not None:
        compacted_items.append(artifact_item)
    compacted_items.extend(selected_items)
    return _local_compact_result(
        original=items,
        compacted=compacted_items,
        pre_tokens=pre_tokens,
        pre_body_bytes=pre_body_bytes,
        mode="aggressive",
    )


def _split_leading_system_items(
    items: list[TResponseInputItem],
) -> tuple[list[TResponseInputItem], list[TResponseInputItem]]:
    """Split leading system messages from mutable dialogue history."""
    leading_system: list[TResponseInputItem] = []
    non_system: list[TResponseInputItem] = []
    for item in items:
        if not non_system and isinstance(item, dict) and item.get("role") == "system":
            leading_system.append(item)
            continue
        non_system.append(item)
    return leading_system, non_system


def _item_text(item: TResponseInputItem) -> str:
    """Return normalized visible text for a session item."""
    return "\n".join(fragment.strip() for fragment in _collect_text_fragments(item) if fragment.strip())


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
        fragments: list[str] = []
        for key in ("content", "text", "value", "message", "output", "name", "arguments", "summary"):
            if key in value:
                fragments.extend(_collect_text_fragments(value[key]))
        return fragments
    return []


def _is_stage_memory_item(item: TResponseInputItem) -> bool:
    """Return whether an item carries generated stage memory."""
    return "[Stage Memory v3]" in _item_text(item)


def _select_stage_memory_text(
    items: list[TResponseInputItem],
    stage_memory: str | None,
) -> str | None:
    """Select the explicit or latest session stage-memory block."""
    if stage_memory and stage_memory.strip():
        return stage_memory.strip()
    for item in reversed(items):
        text = _item_text(item)
        marker = "[Stage Memory v3]"
        if marker not in text:
            continue
        memory_block = text[text.index(marker):].strip()
        if "[Artifact Refs]" in memory_block:
            memory_block = memory_block.split("[Artifact Refs]", 1)[0].strip()
        return memory_block
    return None


def _compact_boundary_item(mode: str, profile_name: str | None) -> TResponseInputItem:
    """Build a local compact boundary marker for the replacement session."""
    return {
        "role": "user",
        "content": "\n".join([
            "[Compact Boundary]",
            f"lifecycle: {V3_LIFECYCLE_NAME}",
            f"mode: {mode}",
            f"profile: {profile_name or '-'}",
        ]),
    }


def _artifact_refs_item(artifact_context: str | None) -> TResponseInputItem | None:
    """Return an artifact refs item when concrete refs are present."""
    if not artifact_context or "(no artifacts recorded)" in artifact_context:
        return None
    if "[Artifact Refs]" not in artifact_context:
        return None
    return {"role": "user", "content": artifact_context.strip()}


def _compact_contextual_user_item(item: TResponseInputItem) -> TResponseInputItem:
    """Strip generated context wrappers from a user item and keep only task text."""
    if not isinstance(item, dict) or item.get("role") != "user":
        return item
    text = _item_text(item)
    if "[Current Task]" not in text:
        return item
    task = text.split("[Current Task]", 1)[1]
    for marker in ("[Stage Memory v3]", "[Artifact Refs]"):
        if marker in task:
            task = task.split(marker, 1)[0]
    compacted = dict(item)
    compacted["content"] = "[Recent Task]\n" + task.strip()[:8_000]
    return compacted


def _bounded_tail_items(
    items: list[TResponseInputItem],
    *,
    max_items: int,
    max_tokens: int,
    exclude_context_wrappers: bool,
) -> list[TResponseInputItem]:
    """Return a bounded latest tail without orphaning tool call/output pairs."""
    selected_indices: set[int] = set()
    used_tokens = 0
    pair_lookup = _build_call_pair_lookup(items)
    for index in range(len(items) - 1, -1, -1):
        if len(selected_indices) >= max_items:
            break
        if index in selected_indices:
            continue
        group_indices = _expand_pair_indices(index, items[index], pair_lookup)
        if any(group_index in selected_indices for group_index in group_indices):
            continue
        if len(selected_indices) + len(group_indices) > max_items:
            continue
        group_items = [items[group_index] for group_index in group_indices]
        group_text = "\n".join(_item_text(item) for item in group_items)
        if exclude_context_wrappers and (
            "[Runtime Stage Hint]" in group_text
            or "[Stage Memory v3]" in group_text
            or "[Artifact Refs]" in group_text
        ):
            continue
        compacted_group = [
            _compact_contextual_user_item(item)
            for item in group_items
        ]
        group_tokens = sum(estimate_item_tokens(item) for item in compacted_group)
        if group_tokens > max_tokens:
            continue
        if used_tokens + group_tokens > max_tokens:
            continue
        selected_indices.update(group_indices)
        used_tokens += group_tokens
    return [
        _compact_contextual_user_item(item)
        for index, item in enumerate(items)
        if index in selected_indices
    ]


def _latest_control_indices(items: list[TResponseInputItem]) -> set[int]:
    """Return latest control-artifact related item indices."""
    pair_lookup = _build_call_pair_lookup(items)
    markers = (
        "workload_objective.json",
        "storage_plan_alignment.json",
        "control_artifacts",
        "required_control_artifacts",
        "[Optimization Control Summary]",
    )
    selected: set[int] = set()
    for index in range(len(items) - 1, -1, -1):
        text = _item_text(items[index]).lower()
        if any(marker.lower() in text for marker in markers):
            selected.update(_expand_pair_indices(index, items[index], pair_lookup))
            if len(selected) >= 4:
                break
    return selected


def _latest_validation_indices(items: list[TResponseInputItem]) -> set[int]:
    """Return latest validation/failure item indices."""
    pair_lookup = _build_call_pair_lookup(items)
    markers = ("validation failed", "row count mismatch", "column check failed", "strict validation")
    for index in range(len(items) - 1, -1, -1):
        text = _item_text(items[index]).lower()
        if any(marker in text for marker in markers):
            return set(_expand_pair_indices(index, items[index], pair_lookup))
    return set()


def _latest_tool_pair_indices(
    items: list[TResponseInputItem],
    tool_names: set[str],
) -> set[int]:
    """Return latest function call/output pair indices for named tools."""
    call_index_by_id: dict[str, int] = {}
    tool_name_by_id: dict[str, str] = {}
    output_index_by_id: dict[str, int] = {}
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        call_id = item.get("call_id")
        if not isinstance(call_id, str):
            continue
        if item.get("type") == "function_call":
            name = item.get("name")
            if isinstance(name, str):
                call_index_by_id[call_id] = index
                tool_name_by_id[call_id] = name
        elif item.get("type") == "function_call_output":
            output_index_by_id[call_id] = index
    selected: set[int] = set()
    for call_id, output_index in sorted(output_index_by_id.items(), key=lambda pair: pair[1], reverse=True):
        if tool_name_by_id.get(call_id) not in tool_names:
            continue
        if call_id in call_index_by_id:
            selected.add(call_index_by_id[call_id])
        selected.add(output_index)
        if len(selected) >= 4:
            break
    return selected


def _build_call_pair_lookup(
    items: list[TResponseInputItem],
) -> dict[str, dict[str, int | str]]:
    """Return function-call pair locations for SDK session items."""
    lookup: dict[str, dict[str, int | str]] = {}
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        call_id = item.get("call_id")
        if not isinstance(call_id, str):
            continue
        if item.get("type") == "function_call":
            record = lookup.setdefault(call_id, {})
            record["call_index"] = index
            name = item.get("name")
            if isinstance(name, str):
                record["name"] = name
        elif item.get("type") == "function_call_output":
            lookup.setdefault(call_id, {})["output_index"] = index
    return lookup


def _expand_pair_indices(
    index: int,
    item: TResponseInputItem,
    pair_lookup: dict[str, dict[str, int | str]],
) -> tuple[int, ...]:
    """Return the atomic item group containing a tool call and its output."""
    if not isinstance(item, dict):
        return (index,)
    call_id = item.get("call_id")
    if not isinstance(call_id, str):
        return (index,)
    record = pair_lookup.get(call_id, {})
    expanded = {
        int(record[key])
        for key in ("call_index", "output_index")
        if key in record and isinstance(record[key], int)
    }
    if not expanded:
        return (index,)
    return tuple(sorted(expanded))


def _recent_user_task_indices(items: list[TResponseInputItem], *, limit: int) -> set[int]:
    """Return recent user task item indices."""
    selected: set[int] = set()
    for index in range(len(items) - 1, -1, -1):
        item = items[index]
        if isinstance(item, dict) and item.get("role") == "user":
            selected.add(index)
            if len(selected) >= limit:
                break
    return selected


def _local_compact_result(
    *,
    original: list[TResponseInputItem],
    compacted: list[TResponseInputItem],
    pre_tokens: int,
    pre_body_bytes: int,
    mode: str,
) -> LocalCompactResult:
    """Build local compact metrics and report whether the session changed."""
    post_tokens = estimate_session_tokens(compacted)
    post_body_bytes = estimate_json_bytes(compacted)
    changed_count = 1 if compacted != original and (
        post_tokens < pre_tokens or post_body_bytes < pre_body_bytes
    ) else 0
    return LocalCompactResult(
        items=compacted if changed_count else original,
        changed_count=changed_count,
        pre_tokens=pre_tokens,
        post_tokens=post_tokens if changed_count else pre_tokens,
        pre_body_bytes=pre_body_bytes,
        post_body_bytes=post_body_bytes if changed_count else pre_body_bytes,
        mode=mode,
    )
