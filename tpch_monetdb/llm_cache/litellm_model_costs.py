from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

_OVERRIDES_PATH = Path(__file__).with_name("litellm_model_cost_overrides.json")
_REGISTERED = False


def force_litellm_local_model_cost_map() -> None:
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "true"
    return None


def load_tpch_monetdb_litellm_model_cost_overrides() -> dict[str, dict[str, Any]]:
    """Load and validate the local LiteLLM model-cost override JSON."""
    raise NotImplementedError("TODO(student): load and validate LiteLLM cost overrides")


def register_tpch_monetdb_litellm_model_costs() -> None:
    """Register local LiteLLM model-cost overrides idempotently."""
    global _REGISTERED
    force_litellm_local_model_cost_map()
    if _REGISTERED:
        return None
    # TODO(student): update litellm.model_cost and refresh LiteLLM's lowercase cache.
    _REGISTERED = True
    return None


def validate_gpt55_xhigh_model_cost() -> None:
    register_tpch_monetdb_litellm_model_costs()
    import litellm

    model_info = litellm.model_cost.get("gpt-5.5")
    if not isinstance(model_info, dict):
        raise RuntimeError("LiteLLM local model cost map is missing gpt-5.5.")
    if model_info.get("mode") != "responses":
        raise RuntimeError("LiteLLM gpt-5.5 local model cost mode must be responses.")
    if model_info.get("supports_xhigh_reasoning_effort") is not True:
        raise RuntimeError("LiteLLM gpt-5.5 local model cost map must support xhigh.")
    return None
