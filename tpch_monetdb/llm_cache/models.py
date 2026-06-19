import logging
from dataclasses import dataclass

from tpch_monetdb.utils.model_aliases import normalize_accounting_model_name


@dataclass
class ModelPricing:
    """Pricing information for a model.
    
    Attributes:
        input: Cost per input token in USD
        cached_input: Cost per cached input token in USD
        output: Cost per output token in USD
        context_window: Maximum context window size in tokens
        max_output_tokens: Maximum output tokens allowed by the model (optional)
        tier_threshold: Input token threshold for tiered pricing (optional)
        long_input: Cost per input token above threshold in USD (optional)
        long_cached_input: Cost per cached input token above threshold in USD (optional)
        long_output: Cost per output token above threshold in USD (optional)
    """
    input: float
    cached_input: float
    output: float
    context_window: int
    max_output_tokens: int | None = None
    tier_threshold: int | None = None
    long_input: float | None = None
    long_cached_input: float | None = None
    long_output: float | None = None


# Model registry with consistent structure
# Supports prefix matching for versioned models (e.g., "anthropic/claude-opus-4-6-20250514"
# matches "anthropic/claude-opus-4-6")
MODEL_REGISTRY: dict[str, ModelPricing] = {
    # GPT-5.1 series
    "gpt-5.1": ModelPricing(
        input=1.25 / 1_000_000,
        cached_input=0.125 / 1_000_000,
        output=10.00 / 1_000_000,
        context_window=400_000,
    ),
    "gpt-5.1-codex": ModelPricing(
        input=1.25 / 1_000_000,
        cached_input=0.125 / 1_000_000,
        output=10.00 / 1_000_000,
        context_window=400_000,
    ),
    "gpt-5.1-codex-max": ModelPricing(
        input=1.25 / 1_000_000,
        cached_input=0.125 / 1_000_000,
        output=10.00 / 1_000_000,
        context_window=400_000,
    ),
    # GPT-5.2 series
    "gpt-5.2": ModelPricing(
        input=1.75 / 1_000_000,
        cached_input=0.17 / 1_000_000,
        output=14.00 / 1_000_000,
        context_window=400_000,
    ),
    "gpt-5.2-codex": ModelPricing(
        input=1.75 / 1_000_000,
        cached_input=0.17 / 1_000_000,
        output=14.00 / 1_000_000,
        context_window=400_000,
    ),
    # GPT-5.5 series
    "gpt-5.5": ModelPricing(
        input=5.00 / 1_000_000,
        cached_input=0.50 / 1_000_000,
        output=30.00 / 1_000_000,
        context_window=272_000,
        max_output_tokens=128_000,
    ),
    # Anthropic Claude Opus 4 series
    "anthropic/claude-opus-4": ModelPricing(
        input=15.00 / 1_000_000,
        cached_input=1.50 / 1_000_000,
        output=75.00 / 1_000_000,
        context_window=200_000,
    ),
    "anthropic/claude-opus-4-1": ModelPricing(
        input=15.00 / 1_000_000,
        cached_input=1.50 / 1_000_000,
        output=75.00 / 1_000_000,
        context_window=200_000,
    ),
    "anthropic/claude-opus-4-5": ModelPricing(
        input=5.00 / 1_000_000,
        cached_input=0.50 / 1_000_000,
        output=25.00 / 1_000_000,
        context_window=200_000,
    ),
    "anthropic/claude-opus-4-6": ModelPricing(
        input=5.00 / 1_000_000,
        cached_input=0.50 / 1_000_000,
        output=25.00 / 1_000_000,
        context_window=200_000,
    ),
    # Anthropic Claude Sonnet 4 series
    "anthropic/claude-sonnet-4": ModelPricing(
        input=3.00 / 1_000_000,
        cached_input=0.30 / 1_000_000,
        output=15.00 / 1_000_000,
        context_window=200_000,
    ),
    "anthropic/claude-sonnet-4-5": ModelPricing(
        input=3.00 / 1_000_000,
        cached_input=0.30 / 1_000_000,
        output=15.00 / 1_000_000,
        context_window=200_000,
    ),
    # Moonshot AI Kimi K2.5 series
    "kimi-k2.5": ModelPricing(
        input=0.60 / 1_000_000,
        cached_input=0.10 / 1_000_000,
        output=3.00 / 1_000_000,
        context_window=262_144,
    ),
    # Zhipu AI GLM-5 series
    "glm-5": ModelPricing(
        input=1.00 / 1_000_000,
        cached_input=0.20 / 1_000_000,
        output=3.20 / 1_000_000,
        context_window=200_000,
    ),
    # Qwen 3.6 Plus series
    "qwen3.6-plus": ModelPricing(
        input=0.276 / 1_000_000,
        cached_input=0.276 / 1_000_000,
        output=1.651 / 1_000_000,
        context_window=1_000_000,
        tier_threshold=256_000,
        long_input=1.101 / 1_000_000,
        long_cached_input=1.101 / 1_000_000,
        long_output=6.602 / 1_000_000,
    ),
    # DeepSeek V4 Flash - 平衡性能和成本的版本
    "deepseek-v4-flash": ModelPricing(
        input=0.14 / 1_000_000,           # 单位：美元/百万 token
        cached_input=0.0028 / 1_000_000,  # 缓存 token 价格更便宜（2%）
        output=0.28 / 1_000_000,          # 输出 token 价格是输入的 2 倍
        context_window=1_000_000,         # 最大上下文窗口 100 万 token
    ),
    # DeepSeek V4 Pro - 高质量版本，价格更高
    "deepseek-v4-pro": ModelPricing(
        input=0.435 / 1_000_000,          # Flash 的 3 倍价格
        cached_input=0.003625 / 1_000_000,  # 缓存 token 价格（约 0.83%）
        output=0.87 / 1_000_000,          # 输出 token 价格是输入的 2 倍
        context_window=1_000_000,         # 最大上下文窗口 100 万 token
    ),
}

logger = logging.getLogger(__name__)


def get_model_pricing(model_name: str) -> ModelPricing:
    """Get pricing information for a model.
    
    Supports exact matches and prefix matching for versioned models.
    For prefix matching, the longest matching prefix wins.
    
    Args:
        model_name: The model name to look up
        
    Returns:
        ModelPricing for the matched model
        
    Raises:
        KeyError: If no matching model is found in the registry
        
    Examples:
        >>> get_model_pricing("gpt-5.1")
        ModelPricing(...)
        >>> get_model_pricing("anthropic/claude-opus-4-6-20250514")
        # Matches "anthropic/claude-opus-4-6" (longest prefix)
        ModelPricing(...)
    """
    normalized_name = normalize_accounting_model_name(model_name)

    if normalized_name in MODEL_REGISTRY:
        return MODEL_REGISTRY[normalized_name]
    
    # Prefix match: longest match wins
    candidates = [key for key in MODEL_REGISTRY if normalized_name.startswith(key)]
    if candidates:
        best = max(candidates, key=len)
        return MODEL_REGISTRY[best]
    
    raise KeyError(
        f"Unknown model: {normalized_name}. "
        f"Add to MODEL_REGISTRY in tpch_monetdb/llm_cache/models.py."
    )


def get_context_window(model_name: str) -> int:
    """Get context window size for a model.
    
    Args:
        model_name: The model name to look up
        
    Returns:
        Context window size in tokens
        
    Raises:
        KeyError: If no matching model is found
    """
    return get_model_pricing(model_name).context_window


def request_cost_usd(
    model: str, input_tokens: int, cached_tokens: int, output_tokens: int
) -> float:
    """Calculate request cost in USD.
    
    Supports tiered pricing based on input token count.
    
    Args:
        model: The model name
        input_tokens: Total input tokens
        cached_tokens: Cached input tokens (charged at lower rate)
        output_tokens: Output tokens
        
    Returns:
        Total cost in USD
    """
    if "deepseek-v4" in str(model):
        # DeepSeek 计费逻辑：缓存 token 和新增 token 按不同价格计费
        # 这体现了 DeepSeek 的 prompt cache 优势
        pricing = get_model_pricing(str(model))
        # 计算非缓存的 token 数（新增 token）
        billable_input_tokens = max(0, input_tokens - cached_tokens)
        
        # 成本 = 缓存 token * 缓存价格 + 新增 token * 正常价格 + 输出 token * 输出价格
        cost = (
            cached_tokens * pricing.cached_input
            + billable_input_tokens * pricing.input
            + output_tokens * pricing.output
        )
        return max(0, cost)  # 确保不产生负成本
    
    pricing = get_model_pricing(str(model))
    billable_input_tokens = max(0, input_tokens - cached_tokens)

    if pricing.tier_threshold is not None and input_tokens >= pricing.tier_threshold:
        input_price = pricing.long_input if pricing.long_input is not None else pricing.input
        cached_price = pricing.long_cached_input if pricing.long_cached_input is not None else pricing.cached_input
        output_price = pricing.long_output if pricing.long_output is not None else pricing.output
    else:
        input_price = pricing.input
        cached_price = pricing.cached_input
        output_price = pricing.output

    return (
        billable_input_tokens * input_price
        + cached_tokens * cached_price
        + output_tokens * output_price
    )


def context_window_usage(model: str, used_tokens: int) -> tuple[str, float]:
    """Calculate context window usage statistics.
    
    Args:
        model: The model name
        used_tokens: Number of tokens used
        
    Returns:
        Tuple of (formatted string, usage ratio)
    """
    window_size = get_context_window(str(model))
    used_pct = (used_tokens / window_size) * 100
    left_pct = 100 - used_pct
    
    def fmt_k(n: int) -> str:
        return f"{n / 1000:.1f}K" if n >= 1000 else str(n)
    
    return (
        f"{left_pct:.0f}% left ({fmt_k(used_tokens)} used / {fmt_k(window_size)})",
        used_pct / 100,
    )
