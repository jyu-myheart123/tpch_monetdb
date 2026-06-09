from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

import tpch_monetdb.main_tpch_monetdb as main_tpch_monetdb
from tpch_monetdb.llm_cache import litellm_model_costs
from tpch_monetdb.llm_cache.models import (
    get_context_window,
    get_model_pricing,
    request_cost_usd,
)
from tpch_monetdb.utils.model_aliases import (
    get_model_provider,
    is_anthropic_deepseek_model,
    is_deepseek_model,
    is_openai_deepseek_model,
    normalize_accounting_model_name,
)
from tpch_monetdb.utils.model_setup import setup_model_config


def test_deepseek_aliases_are_recognized() -> None:
    """DeepSeek V4 names should be detected with and without provider prefixes."""
    assert is_deepseek_model("deepseek-v4-flash") is True
    assert is_deepseek_model("deepseek/deepseek-v4-pro") is True
    assert is_deepseek_model("openai/deepseek-v4-pro") is True
    assert is_deepseek_model("anthropic/deepseek-v4-flash") is True
    assert is_deepseek_model("kimi-k2.5") is False
    assert is_deepseek_model("") is False
    return None


def test_deepseek_accounting_aliases_normalize_to_base_model() -> None:
    """Provider-prefixed DeepSeek names should share one accounting model name."""
    assert normalize_accounting_model_name("deepseek/deepseek-v4-pro") == "deepseek-v4-pro"
    assert normalize_accounting_model_name("openai/deepseek-v4-flash") == "deepseek-v4-flash"
    assert get_model_provider("openai/deepseek-v4-pro") == "openai"
    assert is_openai_deepseek_model("openai/deepseek-v4-pro") is True
    assert is_anthropic_deepseek_model("anthropic/deepseek-v4-pro") is True
    return None


def test_setup_model_config_allows_native_deepseek_without_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    """The native LiteLLM DeepSeek provider should not require LITELLM_BASE_URL."""
    monkeypatch.setenv("LITELLM_API_KEY", "test-key")
    monkeypatch.delenv("LITELLM_BASE_URL", raising=False)

    config = setup_model_config("litellm/deepseek/deepseek-v4-pro")

    assert config.use_litellm is True
    assert config.model_name == "deepseek/deepseek-v4-pro"
    assert config.accounting_model_name == "deepseek-v4-pro"
    assert config.base_url is None
    return None


def test_setup_model_config_handles_legacy_openai_deepseek(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    """The legacy OpenAI-compatible DeepSeek path should warn and fill the default base URL."""
    monkeypatch.setenv("LITELLM_API_KEY", "test-key")
    monkeypatch.delenv("LITELLM_BASE_URL", raising=False)
    caplog.set_level("WARNING", logger="tpch_monetdb.utils.model_setup")

    config = setup_model_config("litellm/openai/deepseek-v4-pro")

    assert config.base_url == "https://api.deepseek.com"
    assert config.accounting_model_name == "deepseek-v4-pro"
    assert "Deprecated" in caplog.text
    return None


def test_setup_model_config_rejects_anthropic_deepseek(monkeypatch: pytest.MonkeyPatch) -> None:
    """Anthropic-prefixed DeepSeek aliases should fail closed."""
    monkeypatch.setenv("LITELLM_API_KEY", "test-key")

    with pytest.raises(RuntimeError, match="DeepSeek"):
        setup_model_config("litellm/anthropic/deepseek-v4-pro")
    return None


def test_deepseek_pricing_and_context_window_are_registered() -> None:
    """Both DeepSeek V4 variants should expose pricing and 1M-token context windows."""
    flash = get_model_pricing("deepseek-v4-flash")
    pro = get_model_pricing("deepseek/deepseek-v4-pro")

    assert flash.input == pytest.approx(0.14 / 1_000_000)
    assert flash.cached_input == pytest.approx(0.0028 / 1_000_000)
    assert flash.output == pytest.approx(0.28 / 1_000_000)
    assert pro.input == pytest.approx(0.435 / 1_000_000)
    assert pro.cached_input == pytest.approx(0.003625 / 1_000_000)
    assert pro.output == pytest.approx(0.87 / 1_000_000)
    assert get_context_window("openai/deepseek-v4-flash") == 1_000_000
    return None


def test_deepseek_cached_and_uncached_tokens_are_billed_separately() -> None:
    """DeepSeek billing should charge cache hits and misses at different rates."""
    cost = request_cost_usd(
        "deepseek/deepseek-v4-pro",
        input_tokens=1_000_000,
        cached_tokens=100_000,
        output_tokens=2_000_000,
    )

    expected = (900_000 * 0.435 + 100_000 * 0.003625 + 2_000_000 * 0.87) / 1_000_000
    assert cost == pytest.approx(expected)
    assert request_cost_usd("deepseek-v4-flash", 10, 20, 0) >= 0.0
    return None


def test_litellm_cost_overrides_load_and_register_idempotently(monkeypatch: pytest.MonkeyPatch) -> None:
    """Local LiteLLM model-cost overrides should be loaded, validated, and registered once."""
    calls = {"invalidate": 0}
    fake_litellm = SimpleNamespace(
        model_cost={},
        open_ai_chat_completion_models=set(),
    )
    fake_utils = SimpleNamespace(
        _invalidate_model_cost_lowercase_map=lambda: calls.__setitem__("invalidate", calls["invalidate"] + 1)
    )
    monkeypatch.setitem(sys.modules, "litellm", fake_litellm)
    monkeypatch.setitem(sys.modules, "litellm.utils", fake_utils)
    monkeypatch.setattr(litellm_model_costs, "_REGISTERED", False)

    overrides = litellm_model_costs.load_tpch_monetdb_litellm_model_cost_overrides()
    litellm_model_costs.register_tpch_monetdb_litellm_model_costs()
    litellm_model_costs.register_tpch_monetdb_litellm_model_costs()

    assert "gpt-5.5" in overrides
    assert "gpt-5.5" in fake_litellm.model_cost
    assert calls["invalidate"] == 1
    return None


def test_deepseek_reasoning_effort_maps_to_provider_values() -> None:
    """Harness reasoning effort values should map to the DeepSeek provider enum."""
    assert main_tpch_monetdb._normalize_deepseek_reasoning_effort("xhigh") == "max"
    assert main_tpch_monetdb._normalize_deepseek_reasoning_effort("max") == "max"
    assert main_tpch_monetdb._normalize_deepseek_reasoning_effort("medium") == "high"
    return None


def test_build_model_settings_injects_deepseek_thinking_body() -> None:
    """DeepSeek model settings should use extra_body thinking instead of SDK reasoning."""
    args = SimpleNamespace(reasoning_effort="xhigh", tool_parallelism=None)
    config = SimpleNamespace(
        use_litellm=True,
        accounting_model_name="deepseek-v4-pro",
        model_name="deepseek/deepseek-v4-pro",
    )

    settings = main_tpch_monetdb._build_model_settings(args, config)

    assert settings.reasoning is None
    assert settings.extra_body == {
        "thinking": {"type": "enabled"},
        "reasoning_effort": "max",
    }
    assert "thinking" in settings.extra_args["allowed_openai_params"]
    assert "reasoning_effort" in settings.extra_args["allowed_openai_params"]
    assert settings.extra_args["additional_drop_params"] == ["extra_body"]
    return None


def test_build_model_settings_disables_deepseek_thinking_when_effort_none() -> None:
    """reasoning_effort=none should disable DeepSeek thinking explicitly."""
    args = SimpleNamespace(reasoning_effort="none", tool_parallelism=None)
    config = SimpleNamespace(
        use_litellm=True,
        accounting_model_name="deepseek-v4-flash",
        model_name="deepseek/deepseek-v4-flash",
    )

    settings = main_tpch_monetdb._build_model_settings(args, config)

    assert settings.reasoning is None
    assert settings.extra_body == {"thinking": {"type": "disabled"}}
    assert settings.extra_args == {
        "allowed_openai_params": ["thinking"],
        "additional_drop_params": ["extra_body"],
    }
    return None


def test_build_model_settings_does_not_pollute_non_deepseek_models() -> None:
    """DeepSeek-only request parameters should not leak into other LiteLLM models."""
    args = SimpleNamespace(reasoning_effort="high", tool_parallelism=None)
    config = SimpleNamespace(
        use_litellm=True,
        accounting_model_name="kimi-k2.5",
        model_name="anthropic/kimi-k2.5",
    )

    settings = main_tpch_monetdb._build_model_settings(args, config)

    assert settings.extra_body is None
    assert settings.extra_args == {"allowed_openai_params": ["reasoning_effort"]}
    return None
