import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from agents import FunctionTool
from agents.items import ModelResponse
from agents.model_settings import ModelSettings
from agents.usage import Usage
from litellm.types.utils import Choices, Message as LitellmMessage, ModelResponse as LitellmModelResponse, Usage as LitellmUsage
from openai.types.responses import ResponseFunctionToolCall, ResponseReasoningItem
from openai.types.responses.response_reasoning_item import Summary

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tpch_monetdb.llm_cache.cached_litellm import CacheType, CachedLitellmModel
from tpch_monetdb.llm_cache.cached_litellm_compaction import CachedLitellmCompactionSession
from tpch_monetdb.llm_cache.deepseek_reasoning_replay import (
    ensure_deepseek_assistant_messages_have_reasoning_content,
    ensure_deepseek_response_output,
    extract_reasoning_content_from_message,
    repair_deepseek_input_items,
)


class FakeTracing:
    def is_disabled(self) -> bool:
        return True

    def include_data(self) -> bool:
        return False


class FakeSpanData:
    def __init__(self) -> None:
        self.output = []
        self.usage = {}


class FakeSpan:
    def __init__(self) -> None:
        self.span_data = FakeSpanData()


class FakeSpanContext:
    def __enter__(self) -> FakeSpan:
        return FakeSpan()

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


class FakeSnapshotter:
    def __init__(self, working_dir: Path) -> None:
        self.working_dir = working_dir

    def snapshot(self, _req_hash: str) -> tuple[str, str]:
        return "", "commit"

    def push_snapshots(self) -> None:
        return None


def _function_call_item(
    *,
    provider_data: dict[str, object] | None = None,
) -> ResponseFunctionToolCall:
    return ResponseFunctionToolCall(
        id="fc_1",
        call_id="call_1",
        arguments="{}",
        name="read_file",
        type="function_call",
        provider_data=provider_data,
    )


def _reasoning_item(text: str) -> ResponseReasoningItem:
    return ResponseReasoningItem(
        id="rs_1",
        summary=[Summary(text=text, type="summary_text")],
        type="reasoning",
        provider_data={"model": "openai/deepseek-v4-pro"},
    )


def test_ensure_deepseek_response_output_injects_reasoning_item_from_fallback() -> None:
    items = [_function_call_item(provider_data={"model": "openai/deepseek-v4-pro"})]

    repaired = ensure_deepseek_response_output(
        items,
        model_name="openai/deepseek-v4-pro",
        fallback_reasoning_content="Inspect the evidence first.",
        response_id="resp_123",
        require_reasoning_content=True,
    )

    assert repaired[0].type == "reasoning"
    assert repaired[0].summary[0].text == "Inspect the evidence first."
    assert repaired[1].provider_data["reasoning_content"] == "Inspect the evidence first."
    assert repaired[1].provider_data["response_id"] == "resp_123"
    return None


def test_ensure_deepseek_response_output_injects_placeholder_when_reasoning_missing() -> None:
    items = [_function_call_item(provider_data={"model": "openai/deepseek-v4-pro"})]

    repaired = ensure_deepseek_response_output(
        items,
        model_name="openai/deepseek-v4-pro",
        fallback_reasoning_content=None,
        response_id=None,
        require_reasoning_content=True,
    )

    assert len(repaired) == 2
    assert repaired[0].type == "reasoning"
    assert "[Minimal thinking placeholder" in repaired[0].summary[0].text
    assert repaired[1].type == "function_call"
    assert repaired[1].provider_data["reasoning_content"].startswith("[Minimal thinking placeholder")
    return None


def test_ensure_deepseek_response_output_allows_non_thinking_tool_call_without_reasoning() -> None:
    items = [_function_call_item(provider_data={"model": "openai/deepseek-v4-pro"})]

    repaired = ensure_deepseek_response_output(
        items,
        model_name="openai/deepseek-v4-pro",
        fallback_reasoning_content=None,
        response_id=None,
        require_reasoning_content=False,
    )

    assert len(repaired) == 1
    assert repaired[0].type == "function_call"
    assert repaired[0].provider_data == {"model": "openai/deepseek-v4-pro"}
    return None


def test_extract_reasoning_content_from_message_reads_reasoning_payload() -> None:
    message = SimpleNamespace(
        reasoning_content="",
        reasoning={"content": [{"text": "Inspect the evidence first."}]},
        thinking=None,
        get=lambda key, default=None: default,
    )

    assert (
        extract_reasoning_content_from_message(message)
        == "Inspect the evidence first."
    )
    return None


def test_extract_reasoning_content_from_message_reads_provider_thinking_payload() -> None:
    message = SimpleNamespace(
        reasoning_content="",
        reasoning=None,
        thinking=None,
        get=lambda key, default=None: (
            {"thinking": [{"text": "Check the schema before calling tools."}]}
            if key == "provider_specific_fields"
            else default
        ),
    )

    assert (
        extract_reasoning_content_from_message(message)
        == "Check the schema before calling tools."
    )
    return None


def test_repair_deepseek_input_items_restores_reasoning_before_tool_call() -> None:
    items = [
        {
            "type": "function_call",
            "call_id": "call_1",
            "name": "read_file",
            "arguments": "{}",
            "provider_data": {
                "model": "openai/deepseek-v4-pro",
                "reasoning_content": "Inspect the evidence first.",
                "response_id": "resp_123",
            },
        },
        {
            "type": "function_call_output",
            "call_id": "call_1",
            "output": "ok",
        },
    ]

    repaired = repair_deepseek_input_items(
        items,
        model_name="openai/deepseek-v4-pro",
    )

    assert repaired[0]["type"] == "reasoning"
    assert repaired[0]["summary"][0]["text"] == "Inspect the evidence first."
    assert repaired[1]["type"] == "function_call"
    return None


def test_repair_deepseek_input_items_does_not_leak_past_new_user_turn() -> None:
    items = [
        {
            "type": "function_call",
            "call_id": "call_1",
            "name": "read_file",
            "arguments": "{}",
            "provider_data": {
                "model": "openai/deepseek-v4-pro",
                "reasoning_content": "Inspect the evidence first.",
            },
        },
        {
            "type": "function_call_output",
            "call_id": "call_1",
            "output": "ok",
        },
        {
            "role": "user",
            "content": "new request",
        },
        {
            "type": "function_call",
            "call_id": "call_2",
            "name": "list_files",
            "arguments": "{}",
            "provider_data": {
                "model": "openai/deepseek-v4-pro",
            },
        },
    ]

    repaired = repair_deepseek_input_items(
        items,
        model_name="openai/deepseek-v4-pro",
    )

    reasoning_indexes = [
        index
        for index, item in enumerate(repaired)
        if isinstance(item, dict) and item.get("type") == "reasoning"
    ]
    assert reasoning_indexes == [0]
    return None


def test_repair_deepseek_input_items_injects_placeholder_for_unrecoverable_turn() -> None:
    items = [
        {
            "id": "rs_1",
            "summary": [{"text": "Inspect the evidence first.", "type": "summary_text"}],
            "type": "reasoning",
            "provider_data": {"model": "openai/deepseek-v4-pro"},
        },
        {
            "type": "function_call",
            "call_id": "call_1",
            "name": "read_file",
            "arguments": "{}",
            "provider_data": {"model": "openai/deepseek-v4-pro"},
        },
        {
            "type": "function_call_output",
            "call_id": "call_1",
            "output": "ok",
        },
        {
            "type": "function_call",
            "call_id": "call_2",
            "name": "list_files",
            "arguments": "{}",
            "provider_data": {"model": "openai/deepseek-v4-pro"},
        },
    ]

    repaired = repair_deepseek_input_items(
        items,
        model_name="openai/deepseek-v4-pro",
        fail_on_unrecoverable=True,
    )
    # 第二个 function_call (call_2) 前应注入占位 reasoning
    reasoning_indexes = [
        index
        for index, item in enumerate(repaired)
        if isinstance(item, dict) and item.get("type") == "reasoning"
    ]
    assert len(reasoning_indexes) == 2
    assert "[Minimal thinking placeholder" in repaired[reasoning_indexes[1]]["summary"][0]["text"]
    return None


def test_repair_deepseek_input_items_allows_non_thinking_history_without_reasoning() -> None:
    items = [
        {
            "type": "function_call",
            "call_id": "call_1",
            "name": "read_file",
            "arguments": "{}",
            "provider_data": {"model": "openai/deepseek-v4-pro"},
        },
        {
            "type": "function_call_output",
            "call_id": "call_1",
            "output": "ok",
        },
    ]

    repaired = repair_deepseek_input_items(
        items,
        model_name="openai/deepseek-v4-pro",
        fail_on_unrecoverable=True,
    )

    assert repaired == items
    return None


def test_repair_deepseek_input_items_injects_placeholder_when_thinking_required_without_history() -> None:
    items = [
        {
            "type": "function_call",
            "call_id": "call_1",
            "name": "read_file",
            "arguments": "{}",
            "provider_data": {"model": "openai/deepseek-v4-pro"},
        },
        {
            "type": "function_call_output",
            "call_id": "call_1",
            "output": "ok",
        },
    ]

    repaired = repair_deepseek_input_items(
        items,
        model_name="openai/deepseek-v4-pro",
        require_reasoning_for_tool_calls=True,
    )

    assert repaired[0]["type"] == "reasoning"
    assert "[Minimal thinking placeholder" in repaired[0]["summary"][0]["text"]
    return None


def test_ensure_deepseek_assistant_messages_have_reasoning_content_injects_placeholder() -> None:
    messages = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "read_file", "arguments": "{}"},
                }
            ],
        }
    ]

    repaired = ensure_deepseek_assistant_messages_have_reasoning_content(
        messages,
        model_name="openai/deepseek-v4-pro",
    )

    assert repaired[0]["reasoning_content"].startswith(
        "[Minimal thinking placeholder"
    )
    return None


def test_litellm_deepseek_allowed_params_preserve_reasoning_effort() -> None:
    from litellm import DeepSeekChatConfig
    from litellm.utils import get_optional_params

    optional_params = get_optional_params(
        model="deepseek-v4-pro",
        custom_llm_provider="deepseek",
        reasoning_effort="max",
        thinking={"type": "enabled"},
        allowed_openai_params=["thinking", "reasoning_effort"],
        additional_drop_params=["extra_body"],
    )
    request_body = DeepSeekChatConfig().transform_request(
        model="deepseek-v4-pro",
        messages=[{"role": "user", "content": "hello"}],
        optional_params=optional_params,
        litellm_params={},
        headers={},
    )

    assert optional_params["thinking"] == {"type": "enabled"}
    assert optional_params["reasoning_effort"] == "max"
    assert "output_config" not in optional_params
    assert "extra_body" not in request_body
    assert request_body["reasoning_effort"] == "max"
    return None


@pytest.mark.asyncio
async def test_cached_litellm_deepseek_thinking_request_preserves_effort_and_drops_sampling(
    tmp_path,
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    async def fake_tool_handler(_context, _input: str) -> str:
        return "ok"

    def _build_fake_message() -> LitellmMessage:
        return LitellmMessage(content="done", role="assistant")

    monkeypatch.setattr(
        "agents.extensions.models.litellm_model.generation_span",
        lambda **_kwargs: FakeSpanContext(),
    )

    async def fake_acompletion(*args, **kwargs):
        del args
        captured.update(kwargs)
        return LitellmModelResponse(
            id="resp_123",
            choices=[Choices(finish_reason="stop", index=0, message=_build_fake_message())],
            usage=LitellmUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )

    monkeypatch.setattr(
        "tpch_monetdb.llm_cache.cached_litellm.litellm.acompletion",
        fake_acompletion,
    )
    monkeypatch.setattr(
        "tpch_monetdb.llm_cache.cached_litellm.get_tokens_context_and_dollar_info",
        lambda *_args, **_kwargs: {"cost": None},
    )
    monkeypatch.setattr(
        "tpch_monetdb.llm_cache.cached_litellm.utils.dump_pickle",
        lambda *_args, **_kwargs: None,
    )

    tool = FunctionTool(
        name="read_file",
        description="read",
        params_json_schema={
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
        on_invoke_tool=fake_tool_handler,
    )
    model = CachedLitellmModel(
        model="deepseek/deepseek-v4-pro",
        llm_cache_dir=tmp_path / "cache",
        snapshotter=FakeSnapshotter(tmp_path),
    )

    response = await model.get_response(
        system_instructions=None,
        input="hello",
        model_settings=ModelSettings(
            temperature=0.7,
            top_p=0.8,
            frequency_penalty=0.1,
            presence_penalty=0.2,
            extra_body={
                "thinking": {"type": "enabled"},
                "reasoning_effort": "max",
            },
            extra_args={
                "allowed_openai_params": ["thinking", "reasoning_effort"],
                "additional_drop_params": ["extra_body"],
            },
        ),
        tools=[tool],
        output_schema=None,
        handoffs=[],
        previous_response_id=None,
        conversation_id=None,
        prompt=None,
        tracing=FakeTracing(),
    )

    assert response.output[0].content[0].text == "done"
    assert captured["reasoning_effort"] == "max"
    assert captured["thinking"] == {"type": "enabled"}
    assert "reasoning_effort" not in captured.get("extra_body", {})
    assert captured["allowed_openai_params"] == ["thinking", "reasoning_effort"]
    assert captured["additional_drop_params"] == ["extra_body"]
    assert captured["temperature"] is None
    assert captured["top_p"] is None
    assert captured["frequency_penalty"] is None
    assert captured["presence_penalty"] is None
    assert captured["parallel_tool_calls"] is None
    return None


@pytest.mark.asyncio
async def test_cached_litellm_get_response_rehydrates_deepseek_reasoning_from_live_message(
    tmp_path,
    monkeypatch,
) -> None:
    class FakeTracing:
        def is_disabled(self) -> bool:
            return True

        def include_data(self) -> bool:
            return False

    class FakeSpanData:
        def __init__(self) -> None:
            self.output = []
            self.usage = {}

    class FakeSpan:
        def __init__(self) -> None:
            self.span_data = FakeSpanData()

    class FakeSpanContext:
        def __enter__(self) -> FakeSpan:
            return FakeSpan()

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

    class FakeSnapshotter:
        def __init__(self, working_dir: Path) -> None:
            self.working_dir = working_dir

        def snapshot(self, _req_hash: str) -> tuple[str, str]:
            return "", "commit"

        def push_snapshots(self) -> None:
            return None

    def _build_fake_litellm_message():
        from litellm.types.utils import ChatCompletionMessageToolCall
        from openai.types.chat.chat_completion_message_tool_call import Function as LiteLLMFunction
        return LitellmMessage(
            content="",
            role="assistant",
            tool_calls=[
                ChatCompletionMessageToolCall(
                    type="function",
                    id="fc_1",
                    function=LiteLLMFunction(name="read_file", arguments="{}"),
                )
            ],
            reasoning_content="Inspect the evidence first.",
        )

    monkeypatch.setattr(
        "agents.extensions.models.litellm_model.generation_span",
        lambda **_kwargs: FakeSpanContext(),
    )

    async def fake_acompletion(*args, **kwargs):
        del args, kwargs
        return LitellmModelResponse(
            id="resp_123",
            choices=[Choices(finish_reason="tool_calls", index=0, message=_build_fake_litellm_message())],
            usage=LitellmUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )

    monkeypatch.setattr(
        "tpch_monetdb.llm_cache.cached_litellm.litellm.acompletion",
        fake_acompletion,
    )
    monkeypatch.setattr(
        "tpch_monetdb.llm_cache.cached_litellm.get_tokens_context_and_dollar_info",
        lambda *_args, **_kwargs: {"cost": None},
    )
    monkeypatch.setattr(
        "tpch_monetdb.llm_cache.cached_litellm.utils.dump_pickle",
        lambda *_args, **_kwargs: None,
    )

    model = CachedLitellmModel(
        model="openai/deepseek-v4-pro",
        llm_cache_dir=tmp_path / "cache",
        snapshotter=FakeSnapshotter(tmp_path),
    )

    response = await model.get_response(
        system_instructions=None,
        input="hello",
        model_settings=ModelSettings(
            extra_args={"extra_body": {"thinking": {"type": "enabled"}}},
        ),
        tools=[],
        output_schema=None,
        handoffs=[],
        previous_response_id=None,
        conversation_id=None,
        prompt=None,
        tracing=FakeTracing(),
    )

    input_items = response.to_input_items()
    assert input_items[0]["type"] == "reasoning"
    assert input_items[0]["summary"][0]["text"] == "Inspect the evidence first."
    assert input_items[1]["provider_data"]["reasoning_content"] == "Inspect the evidence first."
    return None


@pytest.mark.asyncio
async def test_cached_litellm_injects_placeholder_when_reasoning_payload_is_missing(
    tmp_path,
    monkeypatch,
) -> None:
    class FakeTracing:
        def is_disabled(self) -> bool:
            return True

        def include_data(self) -> bool:
            return False

    class FakeSpanData:
        def __init__(self) -> None:
            self.output = []
            self.usage = {}

    class FakeSpan:
        def __init__(self) -> None:
            self.span_data = FakeSpanData()

    class FakeSpanContext:
        def __enter__(self) -> FakeSpan:
            return FakeSpan()

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

    class FakeSnapshotter:
        def __init__(self, working_dir: Path) -> None:
            self.working_dir = working_dir

        def snapshot(self, _req_hash: str) -> tuple[str, str]:
            return "", "commit"

        def push_snapshots(self) -> None:
            return None

    call_count = {"value": 0}

    def _build_fake_litellm_message_retry():
        from litellm.types.utils import ChatCompletionMessageToolCall
        from openai.types.chat.chat_completion_message_tool_call import Function as LiteLLMFunction
        call_count["value"] += 1
        reasoning = "" if call_count["value"] < 2 else "Inspect the evidence first."
        return LitellmMessage(
            content="",
            role="assistant",
            tool_calls=[
                ChatCompletionMessageToolCall(
                    type="function",
                    id="fc_1",
                    function=LiteLLMFunction(name="read_file", arguments="{}"),
                )
            ],
            reasoning_content=reasoning,
        )

    monkeypatch.setattr(
        "agents.extensions.models.litellm_model.generation_span",
        lambda **_kwargs: FakeSpanContext(),
    )

    async def fake_acompletion(*args, **kwargs):
        del args, kwargs
        msg = _build_fake_litellm_message_retry()
        return LitellmModelResponse(
            id=f"resp_{call_count['value']}",
            choices=[Choices(finish_reason="tool_calls", index=0, message=msg)],
            usage=LitellmUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )

    monkeypatch.setattr(
        "tpch_monetdb.llm_cache.cached_litellm.litellm.acompletion",
        fake_acompletion,
    )
    monkeypatch.setattr(
        "tpch_monetdb.llm_cache.cached_litellm.get_tokens_context_and_dollar_info",
        lambda *_args, **_kwargs: {"cost": None},
    )
    monkeypatch.setattr(
        "tpch_monetdb.llm_cache.cached_litellm.utils.dump_pickle",
        lambda *_args, **_kwargs: None,
    )

    model = CachedLitellmModel(
        model="openai/deepseek-v4-pro",
        llm_cache_dir=tmp_path / "cache",
        snapshotter=FakeSnapshotter(tmp_path),
        config_kwargs={"network_retry_attempts": 2, "network_retry_base_delay_s": 0.0},
    )

    response = await model.get_response(
        system_instructions=None,
        input="hello",
        model_settings=ModelSettings(
            extra_args={"extra_body": {"thinking": {"type": "enabled"}}},
        ),
        tools=[],
        output_schema=None,
        handoffs=[],
        previous_response_id=None,
        conversation_id=None,
        prompt=None,
        tracing=FakeTracing(),
    )

    input_items = response.to_input_items()
    assert call_count["value"] == 1
    assert input_items[0]["type"] == "reasoning"
    assert "[Minimal thinking placeholder" in input_items[0]["summary"][0]["text"]
    return None


@pytest.mark.asyncio
async def test_cached_litellm_preflight_repair_preserves_outgoing_reasoning_content(
    tmp_path,
    monkeypatch,
) -> None:
    class FakeTracing:
        def is_disabled(self) -> bool:
            return True

        def include_data(self) -> bool:
            return False

    class FakeSpanData:
        def __init__(self) -> None:
            self.output = []
            self.usage = {}

    class FakeSpan:
        def __init__(self) -> None:
            self.span_data = FakeSpanData()

    class FakeSpanContext:
        def __enter__(self) -> FakeSpan:
            return FakeSpan()

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

    class FakeSnapshotter:
        def __init__(self, working_dir: Path) -> None:
            self.working_dir = working_dir

        def snapshot(self, _req_hash: str) -> tuple[str, str]:
            return "", "commit"

        def push_snapshots(self) -> None:
            return None

    captured_messages: dict[str, object] = {}

    def _build_fake_litellm_message():
        from litellm.types.utils import ChatCompletionMessageToolCall
        from openai.types.chat.chat_completion_message_tool_call import Function as LiteLLMFunction

        return LitellmMessage(
            content="",
            role="assistant",
            tool_calls=[
                ChatCompletionMessageToolCall(
                    type="function",
                    id="fc_1",
                    function=LiteLLMFunction(name="read_file", arguments="{}"),
                )
            ],
            reasoning_content="Inspect the evidence first.",
        )

    monkeypatch.setattr(
        "agents.extensions.models.litellm_model.generation_span",
        lambda **_kwargs: FakeSpanContext(),
    )

    async def fake_acompletion(*args, **kwargs):
        del args
        captured_messages["messages"] = kwargs["messages"]
        return LitellmModelResponse(
            id="resp_123",
            choices=[Choices(finish_reason="tool_calls", index=0, message=_build_fake_litellm_message())],
            usage=LitellmUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )

    monkeypatch.setattr(
        "tpch_monetdb.llm_cache.cached_litellm.litellm.acompletion",
        fake_acompletion,
    )
    monkeypatch.setattr(
        "tpch_monetdb.llm_cache.cached_litellm.get_tokens_context_and_dollar_info",
        lambda *_args, **_kwargs: {"cost": None},
    )
    monkeypatch.setattr(
        "tpch_monetdb.llm_cache.cached_litellm.utils.dump_pickle",
        lambda *_args, **_kwargs: None,
    )

    model = CachedLitellmModel(
        model="openai/deepseek-v4-pro",
        llm_cache_dir=tmp_path / "cache",
        snapshotter=FakeSnapshotter(tmp_path),
    )

    await model.get_response(
        system_instructions=None,
        input=[
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "read_file",
                "arguments": "{}",
                "provider_data": {
                    "model": "openai/deepseek-v4-pro",
                    "reasoning_content": "Inspect the evidence first.",
                },
            }
        ],
        model_settings=ModelSettings(
            extra_args={"extra_body": {"thinking": {"type": "enabled"}}},
        ),
        tools=[],
        output_schema=None,
        handoffs=[],
        previous_response_id=None,
        conversation_id=None,
        prompt=None,
        tracing=FakeTracing(),
    )

    messages = captured_messages["messages"]
    assistant_messages = [
        message
        for message in messages
        if message.get("role") == "assistant" and message.get("tool_calls")
    ]
    assert len(assistant_messages) == 1
    assert assistant_messages[0]["reasoning_content"] == "Inspect the evidence first."
    return None


@pytest.mark.asyncio
async def test_cached_litellm_bypasses_invalid_deepseek_cache_without_restoring_snapshot(
    tmp_path,
    monkeypatch,
) -> None:
    class FakeSnapshotter:
        def __init__(self, working_dir: Path) -> None:
            self.working_dir = working_dir
            self.clear_calls = 0
            self.reset_calls = 0
            self.restore_calls = 0
            self.snapshot_calls = 0

        def has_snapshot(self, _commit_hash: str) -> bool:
            return True

        def clear_untracked(self, include_ignored: bool = True) -> None:
            del include_ignored
            self.clear_calls += 1
            return None

        def reset_changes(self) -> None:
            self.reset_calls += 1
            return None

        def restore(self, _commit_hash: str) -> None:
            self.restore_calls += 1
            return None

        def snapshot(self, _req_hash: str) -> tuple[str, str]:
            self.snapshot_calls += 1
            return "", "commit"

        def push_snapshots(self) -> None:
            return None

    async def fake_live_response(self, *args, **kwargs):
        del self, args, kwargs
        return ModelResponse(
            output=[
                _reasoning_item("Fresh reasoning"),
                _function_call_item(
                    provider_data={
                        "model": "openai/deepseek-v4-pro",
                        "reasoning_content": "Fresh reasoning",
                    }
                ),
            ],
            usage=Usage(),
            response_id=None,
        )

    cache_file = tmp_path / "cache" / "entry.pkl"
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_bytes(b"placeholder")
    snapshotter = FakeSnapshotter(tmp_path)
    cached_response = CacheType(
        response=ModelResponse(
            output=[_function_call_item(provider_data={"model": "openai/deepseek-v4-pro"})],
            usage=Usage(),
            response_id=None,
        ),
        parent_hash="deadbeef",
    )

    monkeypatch.setattr(
        "tpch_monetdb.llm_cache.cached_litellm.utils.load_pickle",
        lambda *_args, **_kwargs: cached_response,
    )
    monkeypatch.setattr(
        "tpch_monetdb.llm_cache.cached_litellm.CachedLitellmModel._fetch_uncached_live_response",
        fake_live_response,
    )
    monkeypatch.setattr(
        "tpch_monetdb.llm_cache.cached_litellm.get_tokens_context_and_dollar_info",
        lambda *_args, **_kwargs: {"cost": None},
    )
    monkeypatch.setattr(
        "tpch_monetdb.llm_cache.cached_litellm.utils.dump_pickle",
        lambda *_args, **_kwargs: None,
    )

    model = CachedLitellmModel(
        model="openai/deepseek-v4-pro",
        llm_cache_dir=tmp_path / "cache",
        snapshotter=snapshotter,
    )
    monkeypatch.setattr(model, "_cache_path_for", lambda _hash: cache_file)

    response = await model.get_response(
        system_instructions=None,
        input="hello",
        model_settings=ModelSettings(
            extra_args={"extra_body": {"thinking": {"type": "enabled"}}},
        ),
        tools=[],
        output_schema=None,
        handoffs=[],
        previous_response_id=None,
        conversation_id=None,
        prompt=None,
    )

    assert snapshotter.clear_calls == 1
    assert snapshotter.reset_calls == 1
    assert snapshotter.restore_calls == 1
    assert snapshotter.snapshot_calls == 0
    assert cache_file.exists() is True
    assert response.output[0].type == "reasoning"
    assert "[Minimal thinking placeholder" in response.output[0].summary[0].text
    return None


@pytest.mark.asyncio
async def test_litellm_compaction_session_get_items_repairs_backup_only_turns(
    tmp_path,
) -> None:
    class FakeUnderlyingSession:
        async def get_items(self) -> list[dict[str, object]]:
            return [
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "read_file",
                    "arguments": "{}",
                    "provider_data": {
                        "model": "openai/deepseek-v4-pro",
                        "reasoning_content": "Inspect the evidence first.",
                    },
                }
            ]

    session = CachedLitellmCompactionSession(
        session_id="conv-test",
        model="openai/deepseek-v4-pro",
        api_key="test-key",
        base_url="https://api.deepseek.com",
        cache_dir=tmp_path,
        wandb_metrics_hook=None,
        compaction_model_map=None,
    )
    session.set_underlying_session(FakeUnderlyingSession())

    items = await session.get_items()

    assert items[0]["type"] == "reasoning"
    assert items[0]["summary"][0]["text"] == "Inspect the evidence first."
    return None


@pytest.mark.asyncio
async def test_litellm_compaction_session_get_items_injects_placeholder_for_old_broken_turn(
    tmp_path,
) -> None:
    class FakeUnderlyingSession:
        async def get_items(self) -> list[dict[str, object]]:
            return [
                {
                    "id": "rs_1",
                    "summary": [
                        {"text": "Inspect the evidence first.", "type": "summary_text"}
                    ],
                    "type": "reasoning",
                    "provider_data": {"model": "openai/deepseek-v4-pro"},
                },
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "read_file",
                    "arguments": "{}",
                    "provider_data": {
                        "model": "openai/deepseek-v4-pro",
                        "reasoning_content": "Inspect the evidence first.",
                    },
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": "ok",
                },
                {
                    "type": "function_call",
                    "call_id": "call_2",
                    "name": "list_files",
                    "arguments": "{}",
                    "provider_data": {"model": "openai/deepseek-v4-pro"},
                },
            ]

    session = CachedLitellmCompactionSession(
        session_id="conv-test",
        model="openai/deepseek-v4-pro",
        api_key="test-key",
        base_url="https://api.deepseek.com",
        cache_dir=tmp_path,
        wandb_metrics_hook=None,
        compaction_model_map=None,
    )
    session.set_underlying_session(FakeUnderlyingSession())

    items = await session.get_items()
    reasoning_indexes = [
        index
        for index, item in enumerate(items)
        if isinstance(item, dict) and item.get("type") == "reasoning"
    ]
    assert len(reasoning_indexes) == 2
    assert "[Minimal thinking placeholder" in items[reasoning_indexes[1]]["summary"][0]["text"]
    return None
