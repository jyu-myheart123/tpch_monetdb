import logging
import os
from dataclasses import dataclass
from typing import Optional

from openai import AsyncOpenAI

from tpch_monetdb.utils.model_aliases import (
    is_anthropic_deepseek_model,
    is_deepseek_model,
    is_openai_deepseek_model,
    normalize_accounting_model_name,
)

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for model initialization.
    
    Attributes:
        use_litellm: Whether to use LiteLLM path instead of OpenAI
        model_name: The model name (with litellm/ prefix stripped if applicable)
        api_key: API key for the selected provider
        base_url: Base URL for LiteLLM proxy
        openai_client: AsyncOpenAI client (only created for OpenAI path)
    """
    use_litellm: bool
    model_name: str
    accounting_model_name: str
    api_key: Optional[str]
    base_url: Optional[str] = None
    openai_client: Optional[AsyncOpenAI] = None


def setup_model_config(model_arg: str) -> ModelConfig:
    """Setup model configuration based on model argument.
    
    For LiteLLM path, no OpenAI client is created, ensuring path independence.
    For OpenAI path, an AsyncOpenAI client is created and returned.
    
    Args:
        model_arg: The model argument, potentially with 'litellm/' prefix
        
    Returns:
        ModelConfig with all necessary configuration
        
    Raises:
        RuntimeError: If required API keys are not set
    """
    model_name = model_arg
    litellm_prefix = "litellm/"
    use_litellm = model_name.startswith(litellm_prefix)
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    
    if use_litellm:
        model_name = model_name[len(litellm_prefix):]
        api_key = (
            os.environ.get("LITELLM_API_KEY")
            or os.environ.get("ANTHROPIC_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
        )
        if not api_key:
            raise RuntimeError(
                "LITELLM_API_KEY (or provider API key) must be set for litellm models."
            )
        base_url = os.environ.get("LITELLM_BASE_URL")
        if is_deepseek_model(model_name):
            if is_anthropic_deepseek_model(model_name):
                raise RuntimeError("DeepSeek models with anthropic prefix are not supported. Use litellm/deepseek/deepseek-v4-* or litellm/openai/deepseek-v4-* instead.")
            if is_openai_deepseek_model(model_name):
                if not base_url:
                    base_url = "https://api.deepseek.com"
                    logger.warning("Deprecated: Using legacy OpenAI-prefixed DeepSeek path without base_url, defaulting to %s", base_url)
            return ModelConfig(
                use_litellm=True,
                model_name=model_name,
                accounting_model_name=normalize_accounting_model_name(model_name),
                api_key=api_key,
                base_url=base_url,
                openai_client=None,
            )
        return ModelConfig(
            use_litellm=True,
            model_name=model_name,
            accounting_model_name=normalize_accounting_model_name(model_name),
            api_key=api_key,
            base_url=base_url,
            openai_client=None,
        )
    else:
        if not openai_api_key:
            raise RuntimeError("OPENAI_API_KEY must be set for OpenAI models.")
        api_key = openai_api_key
        client = AsyncOpenAI(api_key=openai_api_key)
        return ModelConfig(
            use_litellm=False,
            model_name=model_name,
            accounting_model_name=normalize_accounting_model_name(model_name),
            api_key=api_key,
            base_url=None,
            openai_client=client,
        )
