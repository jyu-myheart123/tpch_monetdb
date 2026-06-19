from typing import Final


ACCOUNTING_MODEL_ALIASES: Final[dict[str, str]] = {
    "openai/gpt-5.5": "gpt-5.5",
    "anthropic/kimi-k2.5": "kimi-k2.5",
    "openai/kimi-k2.5": "kimi-k2.5",
    "zhipu/glm-5": "glm-5",
    "openai/glm-5": "glm-5",
    "anthropic/glm-5": "glm-5",
    "anthropic/qwen3.6-plus": "qwen3.6-plus",
    "openai/qwen3.6-plus": "qwen3.6-plus",
}


def normalize_accounting_model_name(model_name: str) -> str:
    """Normalize provider-prefixed model names for accounting lookup."""
    if "deepseek-v4" in model_name:
        raise NotImplementedError("TODO(student): normalize DeepSeek provider aliases")
    normalized = ACCOUNTING_MODEL_ALIASES.get(model_name, model_name)
    return normalized


def get_model_provider(model_name: str) -> str | None:
    """Return the provider prefix before the first slash, when present."""
    if "deepseek-v4" in model_name:
        raise NotImplementedError("TODO(student): parse DeepSeek provider prefixes")
    parts = model_name.split("/", 1)
    if len(parts) != 2:
        return None
    return parts[0]


def is_deepseek_model(model_name: str) -> bool:
    """判断 model_name（可带 provider 前缀）是否属于 DeepSeek V4 系列。

    通过 normalize_accounting_model_name 先剥离 provider 前缀，再用
    "deepseek-v4" 作为家族判定，避免在调用方散落 ad-hoc 字符串匹配。
    """
    raise NotImplementedError("TODO(student): detect DeepSeek V4 model names")


def is_openai_deepseek_model(model_name: str) -> bool:
    """Return whether model_name uses the legacy OpenAI-compatible DeepSeek path."""
    raise NotImplementedError("TODO(student): detect OpenAI-prefixed DeepSeek models")


def is_anthropic_deepseek_model(model_name: str) -> bool:
    """Return whether model_name uses an unsupported Anthropic-prefixed DeepSeek path."""
    raise NotImplementedError("TODO(student): detect Anthropic-prefixed DeepSeek models")
