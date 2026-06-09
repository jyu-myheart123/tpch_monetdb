import json
from typing import Any


def _unwrap_raw_arguments(payload: Any) -> dict[str, Any]:
    """Unwrap supported raw_arguments wrappers."""
    current = payload
    depth = 0
    while isinstance(current, dict) and "raw_arguments" in current:
        raw_arguments = current["raw_arguments"]
        if isinstance(raw_arguments, str):
            current = json.loads(raw_arguments)
        elif isinstance(raw_arguments, dict):
            current = raw_arguments
        else:
            raise ValueError("raw_arguments must be a JSON string or object.")
        depth += 1
        if depth > 3:
            raise ValueError("raw_arguments nesting is too deep.")
    if not isinstance(current, dict):
        raise ValueError("Tool arguments must decode to a JSON object.")
    return current


def load_function_tool_args(args_json: str) -> dict[str, Any]:
    payload = json.loads(args_json)
    parsed = _unwrap_raw_arguments(payload)
    return parsed

