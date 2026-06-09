"""Phase10 LiteLLM readonly parallel tool runtime regression tests.

锁定 Section 5：tool parallelism 配置、readonly 模式 verified/unverified 分支、
只读 batch 并发、mixed batch 写独占、_state 原子 bookkeeping。
"""

import asyncio
import sys
import threading
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tpch_monetdb.tools.tool_parallelism import (
    AsyncRWLock,
    ParallelismConfig,
    ToolParallelismMode,
    UnverifiedLiteLLMModelError,
    VERIFIED_LITELLM_MODELS,
    is_exclusive_tool,
    is_read_only_tool,
    resolve_mode,
    resolve_parallelism_config,
)


def test_resolve_mode_defaults_to_readonly_for_none_or_blank() -> None:
    """未传值默认启用 readonly 并发."""
    assert resolve_mode(None) is ToolParallelismMode.READONLY
    assert resolve_mode("") is ToolParallelismMode.READONLY
    assert resolve_mode("off") is ToolParallelismMode.OFF
    assert resolve_mode("readonly") is ToolParallelismMode.READONLY


def test_resolve_mode_rejects_unknown_tokens() -> None:
    with pytest.raises(ValueError):
        resolve_mode("shared_write")


def test_off_mode_keeps_parallel_tool_calls_false() -> None:
    """off: parallel_tool_calls 强制 False，LiteLLM 主链也不绕开."""
    cfg = resolve_parallelism_config("off", use_litellm=True, model_name="any")
    assert isinstance(cfg, ParallelismConfig)
    assert cfg.mode is ToolParallelismMode.OFF
    assert cfg.parallel_tool_calls is False


def test_readonly_enables_parallel_for_verified_litellm_model() -> None:
    """verified LiteLLM 模型在 readonly 下启用 parallel_tool_calls."""
    model = next(iter(VERIFIED_LITELLM_MODELS))
    cfg = resolve_parallelism_config(
        "readonly", use_litellm=True, model_name=model
    )
    assert cfg.parallel_tool_calls is True


def test_readonly_fails_fast_for_unverified_litellm_model() -> None:
    """unverified LiteLLM 模型在 readonly 下必须 fail-fast."""
    with pytest.raises(UnverifiedLiteLLMModelError):
        resolve_parallelism_config(
            "readonly", use_litellm=True, model_name="anthropic/unknown-model"
        )


def test_default_mode_enables_parallel_for_verified_litellm_model() -> None:
    """未显式传 tool_parallelism 时，LiteLLM verified model 默认走 readonly."""
    model = next(iter(VERIFIED_LITELLM_MODELS))
    cfg = resolve_parallelism_config(None, use_litellm=True, model_name=model)
    assert cfg.mode is ToolParallelismMode.READONLY
    assert cfg.parallel_tool_calls is True


def test_openai_deepseek_models_are_verified_for_readonly_parallelism() -> None:
    for model in (
        "openai/deepseek-v4-pro", "openai/deepseek-v4-flash",
        "deepseek/deepseek-v4-pro", "deepseek/deepseek-v4-flash",
    ):
        cfg = resolve_parallelism_config(
            "readonly", use_litellm=True, model_name=model
        )
        assert cfg.parallel_tool_calls is True


def test_openai_gpt55_is_verified_for_readonly_parallelism() -> None:
    cfg = resolve_parallelism_config(
        "readonly", use_litellm=True, model_name="openai/gpt-5.5"
    )
    assert cfg.parallel_tool_calls is True


def test_anthropic_deepseek_models_fail_fast_in_readonly_parallelism() -> None:
    for model in ("anthropic/deepseek-v4-pro", "anthropic/deepseek-v4-flash"):
        with pytest.raises(UnverifiedLiteLLMModelError):
            resolve_parallelism_config(
                "readonly", use_litellm=True, model_name=model
            )


def test_readonly_openai_path_leaves_parallel_none() -> None:
    """OpenAI 路径走 SDK 默认并发，parallel_tool_calls 置为 None."""
    cfg = resolve_parallelism_config(
        "readonly", use_litellm=False, model_name="gpt-x"
    )
    assert cfg.parallel_tool_calls is None


def test_tool_concurrency_classification_matches_decision() -> None:
    """read-only / exclusive tool 分类与 Decision 1 一致."""
    for name in ("read_file", "list_files", "grep_repo", "shell", "cpu_info"):
        assert is_read_only_tool(name)
        assert not is_exclusive_tool(name)
    for name in ("edit_file", "write_file", "apply_patch", "compile", "run"):
        assert not is_read_only_tool(name)
        assert is_exclusive_tool(name)


def test_async_rw_lock_allows_concurrent_readers() -> None:
    """AsyncRWLock shared 区允许多读并发."""
    lock = AsyncRWLock()
    active = 0
    peak = 0
    barrier = asyncio.Event()

    async def reader() -> None:
        nonlocal active, peak
        async with lock.shared():
            active += 1
            peak = max(peak, active)
            # 让所有 reader 都先拿到锁再退出
            await barrier.wait()
            active -= 1

    async def runner() -> None:
        tasks = [asyncio.create_task(reader()) for _ in range(4)]
        # 给所有 reader 一点调度时间
        await asyncio.sleep(0.05)
        barrier.set()
        await asyncio.gather(*tasks)

    asyncio.run(runner())
    assert peak == 4


def test_async_rw_lock_writer_blocks_readers() -> None:
    """AsyncRWLock exclusive 区阻塞所有 reader/writer."""
    lock = AsyncRWLock()
    log: list[str] = []

    async def writer() -> None:
        async with lock.exclusive():
            log.append("w_start")
            await asyncio.sleep(0.05)
            log.append("w_end")

    async def reader(tag: str) -> None:
        async with lock.shared():
            log.append(f"r_{tag}")

    async def runner() -> None:
        w = asyncio.create_task(writer())
        await asyncio.sleep(0.01)  # writer 先拿到锁
        r1 = asyncio.create_task(reader("a"))
        r2 = asyncio.create_task(reader("b"))
        await asyncio.gather(w, r1, r2)

    asyncio.run(runner())
    assert log[0] == "w_start"
    assert log[1] == "w_end"
    assert {log[2], log[3]} == {"r_a", "r_b"}


def test_stage_tool_runtime_tool_guard_picks_shared_or_exclusive(tmp_path: Path) -> None:
    """runtime.tool_guard 对读工具返回 shared guard，对写工具返回 exclusive guard."""
    from tpch_monetdb.tools.tool_parallelism import _ExclusiveGuard, _SharedGuard
    from tpch_monetdb.tools.tpch_monetdb_agent_tools import StageToolRuntime

    runtime = StageToolRuntime(tmp_path)
    assert isinstance(runtime.tool_guard("read_file"), _SharedGuard)
    assert isinstance(runtime.tool_guard("grep_repo"), _SharedGuard)
    assert isinstance(runtime.tool_guard("cpu_info"), _SharedGuard)
    assert isinstance(runtime.tool_guard("edit_file"), _ExclusiveGuard)
    assert isinstance(runtime.tool_guard("compile"), _ExclusiveGuard)
    assert isinstance(runtime.tool_guard("run"), _ExclusiveGuard)


def test_stage_tool_runtime_observation_state_is_threadsafe(tmp_path: Path) -> None:
    """并发 record_observation 下 tool_counts 结果必须确定性."""
    from tpch_monetdb.tools.tpch_monetdb_agent_tools import StageToolRuntime

    runtime = StageToolRuntime(tmp_path)
    runtime.activate("default_general", prompt_index=0, prompt_descriptor="x")

    def worker() -> None:
        for _ in range(200):
            try:
                runtime.record_observation("read_file")
            except Exception:
                # observation 可能触发 soft limit 提示，在本测试中忽略
                pass

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # 4 threads × 200 次 read_file：tool_counts 必须正好 800，不允许 race 丢数。
    assert runtime._state.tool_counts["read_file"] == 4 * 200
