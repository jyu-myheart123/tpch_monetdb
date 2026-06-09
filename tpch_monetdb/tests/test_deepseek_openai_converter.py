import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import litellm
from agents.models.chatcmpl_converter import Converter
from agents.extensions.models.litellm_model import LitellmConverter


def test_deepseek_tool_call_round_trains_reasoning_content_on_assistant_message() -> None:
    items = [
        {
            "type": "reasoning",
            "summary": [{"text": "Need to inspect the evidence before calling tools.", "type": "summary_text"}],
            "provider_data": {"model": "openai/deepseek-v4-pro"},
        },
        {
            "type": "function_call",
            "call_id": "call_1",
            "name": "read_file",
            "arguments": "{\"file_path\": \"design_evidence.md\"}",
        },
    ]

    messages = Converter.items_to_messages(
        items,
        model="openai/deepseek-v4-pro",
    )

    assert len(messages) == 1
    assistant_message = messages[0]
    assert assistant_message["role"] == "assistant"
    assert assistant_message["reasoning_content"] == (
        "Need to inspect the evidence before calling tools."
    )
    assert assistant_message["tool_calls"][0]["function"]["name"] == "read_file"


def test_deepseek_output_message_then_tool_call_keeps_reasoning_content() -> None:
    items = [
        {
            "type": "reasoning",
            "summary": [{"text": "Read evidence, then inspect APIs.", "type": "summary_text"}],
            "provider_data": {"model": "openai/deepseek-v4-pro"},
        },
        {
            "type": "message",
            "role": "assistant",
            "content": [
                {
                    "type": "output_text",
                    "text": "Now let me read the builder and loader APIs.",
                    "annotations": [],
                    "logprobs": [],
                }
            ],
        },
        {
            "type": "function_call",
            "call_id": "call_2",
            "name": "read_file",
            "arguments": "{\"file_path\": \"builder_api.hpp\"}",
        },
    ]

    messages = Converter.items_to_messages(
        items,
        model="openai/deepseek-v4-pro",
    )

    assert len(messages) == 1
    assistant_message = messages[0]
    assert assistant_message["role"] == "assistant"
    assert assistant_message["content"] == "Now let me read the builder and loader APIs."
    assert assistant_message["reasoning_content"] == "Read evidence, then inspect APIs."
    assert assistant_message["tool_calls"][0]["function"]["name"] == "read_file"


def test_deepseek_function_call_provider_data_alone_does_not_restore_reasoning_content() -> None:
    items = [
        {
            "type": "function_call",
            "call_id": "call_3",
            "name": "read_file",
            "arguments": "{\"file_path\": \"design_evidence.md\"}",
            "provider_data": {
                "model": "openai/deepseek-v4-pro",
                "reasoning_content": "Keep using the current reasoning for tool continuation.",
            },
        },
    ]

    messages = Converter.items_to_messages(
        items,
        model="openai/deepseek-v4-pro",
    )

    assert len(messages) == 1
    assistant_message = messages[0]
    assert assistant_message["role"] == "assistant"
    assert "reasoning_content" not in assistant_message
    assert assistant_message["tool_calls"][0]["function"]["name"] == "read_file"


def test_deepseek_reasoning_content_does_not_leak_past_new_user_turn() -> None:
    items = [
        {
            "type": "reasoning",
            "summary": [{"text": "Old reasoning should not leak.", "type": "summary_text"}],
            "provider_data": {"model": "openai/deepseek-v4-pro"},
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
        },
    ]

    messages = Converter.items_to_messages(
        items,
        model="openai/deepseek-v4-pro",
    )

    assert len(messages) == 2
    assert messages[0]["role"] == "user"
    assistant_message = messages[1]
    assert assistant_message["role"] == "assistant"
    assert "reasoning_content" not in assistant_message


def test_litellm_converter_uses_explicit_reasoning_content_field() -> None:
    message = litellm.types.utils.Message(
        content="",
        role="assistant",
        tool_calls=[],
        reasoning_content="explicit reasoning",
    )

    converted = LitellmConverter.convert_message_to_openai(
        message,
        model="openai/deepseek-v4-pro",
    )

    assert converted.reasoning_content == "explicit reasoning"
