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
    """Normalize provider-prefixed model names for accounting lookup.
    
    将带 provider 前缀的模型名称去除前缀，只保留核心模型名称。
    例如：
    - "openai/deepseek-v4-flash" -> "deepseek-v4-flash"
    - "deepseek/deepseek-v4-flash" -> "deepseek-v4-flash"
    - "anthropic/deepseek-v4-flash" -> "deepseek-v4-flash"
    
    这样可以用一个模型名称 key 代表多个 provider 的同一个模型，
    简化计费和配置查询的逻辑。
    """
    # 第1步：检查是否包含 deepseek-v4，如果有则去除任何 provider 前缀
    if "deepseek-v4" in model_name:
        # 如果模型名包含斜杠（即有 provider 前缀），提取斜杠后面的部分
        if "/" in model_name:
            # "openai/deepseek-v4-flash" -> ["openai", "deepseek-v4-flash"]
            # 取第二部分
            return model_name.split("/", 1)[1]
        # 如果没有斜杠，说明已经是无前缀的形式，直接返回
        return model_name
    
    # 第2步：对于非 DeepSeek 模型，使用预定义的别名映射表
    # ACCOUNTING_MODEL_ALIASES 中存储了其他模型的别名转换规则
    normalized = ACCOUNTING_MODEL_ALIASES.get(model_name, model_name)
    return normalized


def get_model_provider(model_name: str) -> str | None:
    """Return the provider prefix before the first slash, when present.
    
    从模型名称中提取 provider 前缀。
    例如：
    - "openai/gpt-5.5" -> "openai"
    - "deepseek/deepseek-v4-flash" -> "deepseek"
    - "gpt-5.5" -> None （没有前缀）
    
    这对于判断模型如何调用 API 很关键：
    - "openai/gpt-5.5" 说明要用 OpenAI 的 API 调用这个模型
    - "deepseek-v4-flash" 说明这是本地模型或需要默认 provider
    """
    # DeepSeek 模型的特殊处理
    if "deepseek-v4" in model_name:
        # 从 "openai/deepseek-v4-flash" 或类似的名称中提取前缀
        parts = model_name.split("/", 1)
        # 如果有斜杠，返回前缀；否则返回 None
        if len(parts) == 2:
            return parts[0]
        return None
    
    # 非 DeepSeek 模型：通用逻辑
    parts = model_name.split("/", 1)
    if len(parts) != 2:
        return None
    return parts[0]


def is_deepseek_model(model_name: str) -> bool:
    """判断 model_name（可带 provider 前缀）是否属于 DeepSeek V4 系列。

    通过 normalize_accounting_model_name 先剥离 provider 前缀，再用
    "deepseek-v4" 作为家族判定，避免在调用方散落 ad-hoc 字符串匹配。
    
    示例：
    - "deepseek-v4-flash" -> True
    - "openai/deepseek-v4-pro" -> True（先被规范化为 "deepseek-v4-pro"）
    - "gpt-5.5" -> False
    """
    # 使用已有的规范化函数去除 provider 前缀
    normalized = normalize_accounting_model_name(model_name)
    # 检查规范化后的名称是否包含 "deepseek-v4"
    return "deepseek-v4" in normalized


def is_openai_deepseek_model(model_name: str) -> bool:
    """Return whether model_name uses the legacy OpenAI-compatible DeepSeek path.
    
    判断模型是否使用 "openai/deepseek-v4-*" 的遗留路径。
    这种路径用 OpenAI 兼容的 API 调用 DeepSeek 模型，但存在兼容性问题，
    所以被标记为 "legacy"。
    
    示例：
    - "openai/deepseek-v4-flash" -> True
    - "deepseek/deepseek-v4-flash" -> False
    - "openai/gpt-5.5" -> False
    """
    # 首先检查是不是 DeepSeek 模型
    if not is_deepseek_model(model_name):
        return False
    # 然后检查 provider 是否是 "openai"
    provider = get_model_provider(model_name)
    return provider == "openai"


def is_anthropic_deepseek_model(model_name: str) -> bool:
    """Return whether model_name uses an unsupported Anthropic-prefixed DeepSeek path.
    
    判断模型是否使用 "anthropic/deepseek-v4-*" 的路径。
    这种路径**不被支持**，因为 Anthropic 的 API 无法调用 DeepSeek 模型。
    使用这种路径时应该抛错。
    
    示例：
    - "anthropic/deepseek-v4-pro" -> True（不支持）
    - "deepseek/deepseek-v4-pro" -> False（正确的方式）
    - "anthropic/claude-opus-4" -> False
    """
    # 首先检查是不是 DeepSeek 模型
    if not is_deepseek_model(model_name):
        return False
    # 然后检查 provider 是否是 "anthropic"
    provider = get_model_provider(model_name)
    return provider == "anthropic"
