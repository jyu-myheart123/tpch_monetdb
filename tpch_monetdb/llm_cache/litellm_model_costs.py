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
    # 从文件系统读取 JSON 配置文件
    if not _OVERRIDES_PATH.exists():
        # 如果文件不存在，返回空字典（没有覆盖）
        return {}
    
    # 读取 JSON 内容
    json_text = _OVERRIDES_PATH.read_text(encoding="utf-8")
    overrides = json.loads(json_text)
    
    # 校验：必须是非空 dict
    if not isinstance(overrides, dict):
        raise TypeError(
            f"litellm_model_cost_overrides.json must contain a dict, "
            f"but got {type(overrides).__name__}"
        )
    
    # 校验：每个 key 是 str，每个 value 是 dict
    for model_name, model_info in overrides.items():
        if not isinstance(model_name, str):
            raise TypeError(
                f"Model name must be str, but got {type(model_name).__name__}: {model_name}"
            )
        if not isinstance(model_info, dict):
            raise TypeError(
                f"Model info for {model_name} must be dict, "
                f"but got {type(model_info).__name__}"
            )
    
    return overrides


def register_tpch_monetdb_litellm_model_costs() -> None:
    """Register local LiteLLM model-cost overrides idempotently."""
    global _REGISTERED
    force_litellm_local_model_cost_map()
    if _REGISTERED:
        return None
    
    # 导入 LiteLLM 库并加载覆盖配置
    import litellm
    
    overrides = load_tpch_monetdb_litellm_model_cost_overrides()
    
    # 更新 litellm.model_cost 字典，添加或覆盖现有配置
    # litellm.model_cost 是一个全局字典，存储所有模型的成本信息
    for model_name, model_info in overrides.items():
        litellm.model_cost[model_name] = model_info
    
    # 刷新 LiteLLM 的小写字母映射缓存
    # LiteLLM 内部维护一个 lowercase_model_cost_map，用于快速查询
    # 添加新模型后需要手动刷新这个缓存
    try:
        import importlib

        litellm_utils = importlib.import_module("litellm.utils")
    except ImportError:
        litellm_utils = getattr(litellm, "utils", None)

    invalidate = getattr(litellm_utils, "_invalidate_model_cost_lowercase_map", None)
    if callable(invalidate):
        invalidate()
    elif hasattr(litellm, "refresh_local_model_cost_map"):
        litellm.refresh_local_model_cost_map()
    
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
