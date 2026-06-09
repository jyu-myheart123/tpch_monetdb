from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any

import pytest
from agents.items import ModelResponse
from agents.model_settings import ModelSettings
from agents.usage import InputTokensDetails, OutputTokensDetails, Usage
from openai.types.responses import (
    Response,
    ResponseCompletedEvent,
    ResponseCreatedEvent,
    ResponseUsage,
)

import run_outer_loop_tpch_monetdb
import tpch_monetdb.main_tpch_monetdb
from tpch_monetdb.llm_cache.cached_litellm import CacheType, CachedLitellmModel
from tpch_monetdb.llm_cache.litellm_retry import run_stream_with_transient_retry
from tpch_monetdb.llm_cache import utils
from tpch_monetdb.utils.cli_config import add_common_args, build_run_config


class FakeSnapshotter:
    def __init__(self, working_dir: Any) -> None:
        self.working_dir = working_dir
        self.restored: str | None = None
        self.snapshots: list[str] = []
        self.pushed = False
        return None

    def has_snapshot(self, _commit_hash: str) -> bool:
        return True

    def fetch_snapshots(self) -> None:
        return None

    def clear_untracked(self, include_ignored: bool = False) -> None:
        return None

    def reset_changes(self) -> None:
        return None

    def restore(self, commit_hash: str) -> None:
        self.restored = commit_hash
        return None

    def is_dirty(self) -> bool:
        return False

    def snapshot(self, req_hash: str) -> tuple[str, str]:
        self.snapshots.append(req_hash)
        return "", "commit-miss"

    def push_snapshots(self) -> None:
        self.pushed = True
        return None


def _outer_args() -> SimpleNamespace:
    return SimpleNamespace(
        conv="outer1-15v1", benchmark="tpch", artifacts_dir="/tmp/artifacts",
        validation_mode="strict", base_data_dir=None,
        model="litellm/openai/gpt-5.5", reasoning_effort="xhigh", notify=False,
        disable_repo_sync=False, replay_cache=False, auto_u=False, auto_finish=False,
        disable_wandb=False, disable_tracing=False, replay=False,
        only_from_llm_cache=False, only_from_cache=False, enable_auto_compact=False,
        baseline_backend=None, baseline_query_file_dir=None,
        benchmark_mode="system-parity", storage_mode="persistent", stream_llm=True,
    )


def test_stream_llm_cli_config_is_preserved() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser, include_stream_llm=True)
    parsed = parser.parse_args(["--stream_llm"])
    config = build_run_config(
        benchmark="tpch", conv_name="run1", query_list="1", notify=False,
        conv_mode="manual", stream_llm=parsed.stream_llm,
    )
    assert config.stream_llm is True
    return None


def test_outer_loop_forwards_stream_llm_to_child_commands() -> None:
    args = _outer_args()
    storage_cmd = run_outer_loop_tpch_monetdb._build_storage_plan_cmd(args, 1)
    base_cmd = run_outer_loop_tpch_monetdb._build_base_impl_cmd(args, 1, "snapshot", True)
    optim_cmd = run_outer_loop_tpch_monetdb._build_optimization_cmd(args, 1, "snapshot", True)
    assert "--stream_llm" in storage_cmd
    assert "--stream_llm" in base_cmd
    assert "--stream_llm" in optim_cmd
    return None


def test_gpt55_xhigh_uses_registered_local_cost_map(monkeypatch) -> None:
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "false")
    model_config = SimpleNamespace(use_litellm=True, model_name="openai/gpt-5.5")
    reasoning = SimpleNamespace(effort="xhigh")
    tpch_monetdb.main_tpch_monetdb._configure_litellm_cost_map_for_reasoning(model_config, reasoning)
    assert tpch_monetdb.main_tpch_monetdb.os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] == "true"
    import litellm

    model_info = litellm.model_cost["gpt-5.5"]
    assert model_info["mode"] == "responses"
    assert model_info["supports_xhigh_reasoning_effort"] is True
    return None


def test_token_usage_handles_empty_request_usage_entries() -> None:
    """streaming provider usage 为空时，计费统计应降级为 0 而不是抛异常。"""
    from tpch_monetdb.utils.token_usage import get_tokens_context_and_dollar_info

    info = get_tokens_context_and_dollar_info(
        Usage(),
        model="gpt-5.5",
        last_entry_only=True,
    )

    assert info["input_tokens"] == 0
    assert info["billed_output_tokens"] == 0
    assert info["num_llm_request"] == 0
    return None


def test_main_import_does_not_preload_litellm() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import tpch_monetdb.main_tpch_monetdb; print('litellm' in sys.modules)",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "False"
    return None


def test_cached_response_defaults_omitted_tool_choice_to_auto(tmp_path) -> None:
    model = CachedLitellmModel(
        model="openai/test", llm_cache_dir=tmp_path / "cache",
        snapshotter=FakeSnapshotter(tmp_path),
    )
    cached = ModelResponse(output=[], usage=Usage(), response_id="resp_cached")
    response = model._response_from_model_response(cached, ModelSettings())
    assert response.tool_choice == "auto"
    return None


@pytest.mark.asyncio
async def test_run_agent_turn_streamed_drains_events(monkeypatch) -> None:
    class FakeResult:
        def __init__(self) -> None:
            self.drained = False
            return None

        async def stream_events(self) -> AsyncIterator[Any]:
            yield SimpleNamespace(type="run_item_stream_event", name="tool_called")
            self.drained = True
            return

    result = FakeResult()
    monkeypatch.setattr(tpch_monetdb.main_tpch_monetdb.Runner, "run_streamed", lambda *_a, **_k: result)
    returned = await tpch_monetdb.main_tpch_monetdb._run_agent_turn(
        object(), input="x", session=None, max_turns=1, hooks=None, stream_llm=True,
    )
    assert returned is result
    assert result.drained is True
    return None


@pytest.mark.asyncio
async def test_stream_retry_retries_only_before_first_event(monkeypatch) -> None:
    attempts = 0
    monkeypatch.setattr("tpch_monetdb.llm_cache.litellm_retry.asyncio.sleep", _no_sleep)

    async def operation() -> AsyncIterator[str]:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise TimeoutError("timeout")
        yield "ok"
        return

    events = [
        event async for event in run_stream_with_transient_retry(
            operation_name="stream", operation=operation,
            logger=logging.getLogger(__name__), max_attempts=2, base_delay_s=0.0,
        )
    ]
    assert events == ["ok"]
    assert attempts == 2
    return None


async def _no_sleep(_delay: float) -> None:
    return None


@pytest.mark.asyncio
async def test_stream_retry_does_not_retry_after_first_event() -> None:
    attempts = 0
    events: list[str] = []

    async def operation() -> AsyncIterator[str]:
        nonlocal attempts
        attempts += 1
        yield "first"
        raise TimeoutError("timeout")

    with pytest.raises(TimeoutError):
        async for event in run_stream_with_transient_retry(
            operation_name="stream", operation=operation,
            logger=logging.getLogger(__name__), max_attempts=2, base_delay_s=0.0,
        ):
            events.append(event)
    assert events == ["first"]
    assert attempts == 1
    return None


@pytest.mark.asyncio
async def test_cached_litellm_stream_cache_hit_replays_completed_event(tmp_path) -> None:
    snapshotter = FakeSnapshotter(tmp_path)
    model = CachedLitellmModel(
        model="openai/test", llm_cache_dir=tmp_path / "cache",
        snapshotter=snapshotter,
    )
    settings = ModelSettings(tool_choice="auto", include_usage=True)
    req_hash = model._hash_payload(
        None, "hello", settings, [], None, [], None, None, None, stream=True,
    )
    cached = ModelResponse(output=[], usage=Usage(requests=1, input_tokens=1, output_tokens=2, total_tokens=3), response_id="resp_cached")
    utils.dump_pickle(model._cache_path_for(req_hash), CacheType(cached, parent_hash="commit-hit"))
    events = [
        event async for event in model.stream_response(None, "hello", settings, [], None, [], object())
    ]
    assert [event.type for event in events] == ["response.created", "response.completed"]
    assert events[-1].response.id == "resp_cached"
    assert snapshotter.restored == "commit-hit"
    assert model.llm_was_cached is True
    return None


@pytest.mark.asyncio
async def test_cached_litellm_stream_cache_miss_writes_snapshot(tmp_path, monkeypatch) -> None:
    async def fake_parent_stream(self, *_args, **_kwargs) -> AsyncIterator[Any]:
        response = _completed_response(str(self.model))
        yield ResponseCreatedEvent(
            response=response.model_copy(update={"output": []}),
            sequence_number=0, type="response.created",
        )
        yield ResponseCompletedEvent(
            response=response, sequence_number=1, type="response.completed",
        )
        return

    monkeypatch.setattr("tpch_monetdb.llm_cache.cached_litellm.LitellmModel.stream_response", fake_parent_stream)
    monkeypatch.setattr("tpch_monetdb.llm_cache.cached_litellm.get_tokens_context_and_dollar_info", lambda *_a, **_k: {"cost": None})
    snapshotter = FakeSnapshotter(tmp_path)
    model = CachedLitellmModel(
        model="openai/test", llm_cache_dir=tmp_path / "cache", snapshotter=snapshotter,
    )
    events = [
        event async for event in model.stream_response(None, "hello", ModelSettings(), [], None, [], object())
    ]
    assert events[-1].type == "response.completed"
    assert len(snapshotter.snapshots) == 1
    assert snapshotter.pushed is True
    assert len(list((tmp_path / "cache").glob("*.pkl"))) == 1
    return None


def _completed_response(model_name: str) -> Response:
    usage = ResponseUsage(
        input_tokens=1, input_tokens_details=InputTokensDetails(cached_tokens=0),
        output_tokens=2, output_tokens_details=OutputTokensDetails(reasoning_tokens=0),
        total_tokens=3,
    )
    return Response(
        id="resp_live", created_at=1.0, model=model_name, object="response",
        output=[], tool_choice="auto", tools=[], parallel_tool_calls=False,
        usage=usage,
    )
