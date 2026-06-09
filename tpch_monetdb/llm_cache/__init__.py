from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "AutoCompactManager": ("tpch_monetdb.llm_cache.auto_compact", "AutoCompactManager"),
    "CachedOpenAIResponsesCompactionSession": (
        "tpch_monetdb.llm_cache.cached_compaction_session",
        "CachedOpenAIResponsesCompactionSession",
    ),
    "CachedOpenAIResponsesModel": (
        "tpch_monetdb.llm_cache.cached_openai",
        "CachedOpenAIResponsesModel",
    ),
    "GitSnapshotter": ("tpch_monetdb.llm_cache.git_snapshotter", "GitSnapshotter"),
    "ModelPricing": ("tpch_monetdb.llm_cache.models", "ModelPricing"),
    "context_window_usage": ("tpch_monetdb.llm_cache.models", "context_window_usage"),
    "get_context_window": ("tpch_monetdb.llm_cache.models", "get_context_window"),
    "get_model_pricing": ("tpch_monetdb.llm_cache.models", "get_model_pricing"),
    "micro_compact_tool_results": (
        "tpch_monetdb.llm_cache.micro_compact",
        "micro_compact_tool_results",
    ),
    "request_cost_usd": ("tpch_monetdb.llm_cache.models", "request_cost_usd"),
    "send_notification": ("tpch_monetdb.llm_cache.notify", "send_notification"),
    "setup_logging": ("tpch_monetdb.llm_cache.logger", "setup_logging"),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    """Load LLM cache exports lazily to keep lightweight imports dependency-free."""
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = _EXPORTS[name]
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value
