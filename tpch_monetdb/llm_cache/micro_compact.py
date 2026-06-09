"""Micro-compact implementation for cleaning up old tool results.

This module provides functionality to reduce context size by clearing
old tool results while preserving recent ones. This is Layer 1 of the
three-layer compression architecture (micro → auto → manual).
"""

import os
from typing import Any

from agents import TResponseInputItem


# Tools that can have their results compacted
# P0: Only compact high-noise validation tools (shell/compile/run)
# Read/edit/write tools are preserved in code-generation stages for context retention
COMPACTABLE_TOOLS = {
    "shell",
    "compile",
    "run",
}

# Default number of recent tool results to keep
KEEP_RECENT = 5

# Number of recent validation tool results to keep (compile/run contain important validation info)
KEEP_RECENT_VALIDATION = 10

# Environment variable for custom keep count
KEEP_RECENT_ENV_VAR = "MICRO_COMPACT_KEEP_RECENT"

# Placeholder text for cleared tool results
CLEARED_PLACEHOLDER = "[tool result cleared]"


def _get_keep_count(tool_name: str) -> int:
    """Get the number of recent results to keep for a tool.
    
    Args:
        tool_name: Name of the tool
        
    Returns:
        Number of recent results to preserve
    """
    # Check for custom configuration
    custom = os.environ.get(KEEP_RECENT_ENV_VAR)
    if custom is not None:
        try:
            return int(custom)
        except ValueError:
            pass
    
    # Validation tools keep more history
    if tool_name in {"compile", "run"}:
        return KEEP_RECENT_VALIDATION
    
    return KEEP_RECENT


def _build_call_id_to_name(items: list[TResponseInputItem]) -> dict[str, str]:
    """Build mapping from call_id to tool name by scanning function_call items.

    In the OpenAI Agents SDK format, function_call items carry the tool name
    while function_call_output items only carry call_id. This mapping bridges
    the two so we can identify which tool produced each output.

    Args:
        items: List of session items

    Returns:
        Mapping of call_id to tool name
    """
    mapping: dict[str, str] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "function_call" and "call_id" in item and "name" in item:
            mapping[item["call_id"]] = item["name"]
    return mapping


def micro_compact_tool_results(
    items: list[TResponseInputItem],
    compactable_tools: set[str] | None = None,
    cleared_placeholder: str = CLEARED_PLACEHOLDER,
) -> list[TResponseInputItem]:
    """Compact old tool results in session items.

    Clears content of old tool results while preserving the most recent N
    for each tool type. Tool use/result pairings are preserved.

    Works with OpenAI Agents SDK format where:
    - function_call items have {type, call_id, name, arguments}
    - function_call_output items have {type, call_id, output} (no name)

    Args:
        items: List of session items (messages)
        compactable_tools: Set of tool names to compact (default: COMPACTABLE_TOOLS)
        cleared_placeholder: Text to replace cleared content with

    Returns:
        List of items with old tool results cleared
    """
    if compactable_tools is None:
        compactable_tools = COMPACTABLE_TOOLS

    if not items:
        return items

    # Build call_id → tool_name mapping from function_call items
    call_id_to_name = _build_call_id_to_name(items)

    # Track tool result indices by tool type
    tool_indices: dict[str, list[int]] = {name: [] for name in compactable_tools}

    for i, item in enumerate(items):
        if not isinstance(item, dict):
            continue

        tool_name = _extract_tool_name(item, call_id_to_name)

        if tool_name is not None and tool_name in compactable_tools:
            tool_indices[tool_name].append(i)

    # Clear old results, keep recent ones
    result = list(items)

    for tool_name, indices in tool_indices.items():
        keep_count = _get_keep_count(tool_name)
        if len(indices) <= keep_count:
            continue

        to_clear = indices[:-keep_count] if keep_count > 0 else indices

        for idx in to_clear:
            item = result[idx]
            if isinstance(item, dict):
                result[idx] = _clear_tool_result(item, cleared_placeholder)

    return result


def _extract_tool_name(
    item: dict[str, Any],
    call_id_to_name: dict[str, str],
) -> str | None:
    """Extract tool name from a tool result item.

    Handles both formats:
    1. SDK format: function_call_output with call_id → lookup in mapping
    2. Chat format: tool result with name field directly

    Args:
        item: Tool result item
        call_id_to_name: Mapping from call_id to tool name

    Returns:
        Tool name if this is a compactable tool result, None otherwise
    """
    item_type = item.get("type", "")

    # SDK Responses API format: function_call_output
    if item_type == "function_call_output":
        call_id = item.get("call_id", "")
        return call_id_to_name.get(call_id)

    # Chat completions format: role=tool with name field
    if item.get("role") == "tool":
        return item.get("name")

    return None


def _clear_tool_result(
    item: dict[str, Any],
    placeholder: str = CLEARED_PLACEHOLDER,
) -> dict[str, Any]:
    """Clear content from a tool result item while preserving structure.
    
    Args:
        item: Tool result item to clear
        placeholder: Text to use as replacement content
        
    Returns:
        Item with cleared content
    """
    cleared = dict(item)
    replacement = _preserved_artifact_placeholder(item, placeholder)
    
    # Clear main content fields
    if "content" in cleared:
        if isinstance(cleared["content"], str):
            cleared["content"] = replacement
        elif isinstance(cleared["content"], dict):
            cleared["content"] = {"cleared": True, "placeholder": replacement}
        else:
            cleared["content"] = replacement
    
    # Clear output field (common in tool results)
    if "output" in cleared:
        cleared["output"] = replacement
    
    # Add metadata to indicate clearing
    cleared["_compacted"] = True
    
    return cleared


def _preserved_artifact_placeholder(item: dict[str, Any], placeholder: str) -> str:
    """Preserve stable artifact refs and hashes when compacting evidence digests."""
    text = str(item.get("output") or item.get("content") or "")
    artifact_ref = _extract_digest_field(text, "artifact_ref")
    if artifact_ref is None:
        return placeholder
    fields = [f"{placeholder}; artifact_ref={artifact_ref}"]
    sha256 = _extract_digest_field(text, "sha256")
    if sha256:
        fields.append(f"sha256={sha256}")
    return "; ".join(fields)


def _extract_digest_field(text: str, field_name: str) -> str | None:
    """Extract one field from a line-oriented evidence digest."""
    marker = f"{field_name}:"
    if marker not in text:
        return None
    value = text.split(marker, 1)[1].strip().splitlines()[0].strip()
    return value or None


def should_compact_tool(tool_name: str) -> bool:
    """Check if a tool should be compacted.
    
    Args:
        tool_name: Name of the tool
        
    Returns:
        True if the tool is in the compactable set
    """
    return tool_name in COMPACTABLE_TOOLS
