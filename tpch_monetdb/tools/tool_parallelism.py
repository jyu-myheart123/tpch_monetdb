"""TPC-H MonetDB tool parallelism 配置模块.

phase10 将 LiteLLM 主链上的 tool concurrency 由隐式常量改为显式配置，
并提供一个 verified LiteLLM model matrix，使得 readonly 并发只对已验证过的
模型开启，未验证模型立即 fail-fast。
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from enum import StrEnum
from typing import Iterable, Optional


class ToolParallelismMode(StrEnum):
    """TPC-H MonetDB 工具并发模式."""

    OFF = "off"
    READONLY = "readonly"


# phase10 第一批 verified LiteLLM 模型：这些模型已在 read-only 并发路径上做过
# smoke 验证。扩展时直接在 allowlist 中补齐并同步更新测试。
VERIFIED_LITELLM_MODELS: frozenset[str] = frozenset(
    {
        "anthropic/kimi-k2.5",
        "anthropic/glm-5",
        "anthropic/qwen3.6-plus",
        "anthropic/claude-opus-4-7",
        "anthropic/claude-sonnet-4-6",
        "openai/gpt-5.5",
        "openai/deepseek-v4-flash",
        "openai/deepseek-v4-pro",
        "deepseek/deepseek-v4-flash",
        "deepseek/deepseek-v4-pro",
    }
)


# 读工具：在 readonly 模式下允许并发。
READ_ONLY_TOOLS: frozenset[str] = frozenset(
    {"read_file", "read_artifact", "list_files", "grep_repo", "shell", "cpu_info"}
)

# 写 / 执行工具：任何模式下都必须独占执行。
EXCLUSIVE_TOOLS: frozenset[str] = frozenset(
    {"edit_file", "write_file", "apply_patch", "compile", "run"}
)


class UnverifiedLiteLLMModelError(RuntimeError):
    """readonly 并发启用时，选中的 LiteLLM 模型不在 verified matrix 中."""


@dataclass(frozen=True)
class ParallelismConfig:
    """解析后的并发决策结果，供 ModelSettings 使用."""

    mode: ToolParallelismMode
    parallel_tool_calls: Optional[bool]


def resolve_mode(raw: Optional[str]) -> ToolParallelismMode:
    """将 CLI / 环境变量字符串解析为 ToolParallelismMode（默认 readonly）."""
    if raw is None or raw == "":
        return ToolParallelismMode.READONLY
    try:
        return ToolParallelismMode(raw)
    except ValueError as exc:
        raise ValueError(
            f"Unknown tool_parallelism mode: {raw!r}. "
            f"Valid modes: {', '.join(m.value for m in ToolParallelismMode)}"
        ) from exc


def resolve_parallelism_config(
    mode_value: Optional[str],
    *,
    use_litellm: bool,
    model_name: Optional[str],
    verified_models: Optional[Iterable[str]] = None,
) -> ParallelismConfig:
    """根据 mode 与当前模型决定 parallel_tool_calls 设置.

    - 默认/readonly + LiteLLM + verified model: parallel_tool_calls=True
    - readonly + LiteLLM + verified model: parallel_tool_calls=True
    - readonly + LiteLLM + unverified model: 抛 UnverifiedLiteLLMModelError
    - readonly + OpenAI 路径: parallel_tool_calls=None（让 SDK 默认并发）
    """
    mode = resolve_mode(mode_value)
    if mode is ToolParallelismMode.OFF:
        return ParallelismConfig(mode=mode, parallel_tool_calls=False)

    # readonly 分支
    if use_litellm:
        verified = frozenset(verified_models) if verified_models is not None else VERIFIED_LITELLM_MODELS
        if model_name is None or model_name not in verified:
            raise UnverifiedLiteLLMModelError(
                f"readonly tool parallelism is not verified for LiteLLM model "
                f"{model_name!r}. Either pick a verified model "
                f"({', '.join(sorted(verified))}) or set --tool_parallelism off."
            )
        return ParallelismConfig(mode=mode, parallel_tool_calls=True)

    # 非 LiteLLM 路径：让 SDK 默认 None（即并发）
    return ParallelismConfig(mode=mode, parallel_tool_calls=None)


def is_read_only_tool(tool_name: str) -> bool:
    """phase10 并发锁决策：名字匹配 READ_ONLY_TOOLS 视为读工具."""
    return tool_name in READ_ONLY_TOOLS


def is_exclusive_tool(tool_name: str) -> bool:
    """phase10 并发锁决策：名字匹配 EXCLUSIVE_TOOLS 视为独占工具."""
    return tool_name in EXCLUSIVE_TOOLS


class AsyncRWLock:
    """最小可行的 asyncio 读写锁.

    语义：
    - `acquire_shared()`：允许多个 reader 同时持有；writer 等待时，新 reader 也需要排队，
      避免 reader 饥饿 writer。
    - `acquire_exclusive()`：writer 独占；持锁期间 reader/writer 都阻塞。

    实现上只依赖标准库的 `asyncio.Lock` 与 `asyncio.Condition`，不做 reentrancy。
    """

    def __init__(self) -> None:
        self._cond = asyncio.Condition()
        self._active_readers = 0
        self._active_writer = False
        self._waiting_writers = 0

    async def acquire_shared(self) -> None:
        async with self._cond:
            while self._active_writer or self._waiting_writers > 0:
                await self._cond.wait()
            self._active_readers += 1

    async def release_shared(self) -> None:
        async with self._cond:
            assert self._active_readers > 0
            self._active_readers -= 1
            if self._active_readers == 0:
                self._cond.notify_all()

    async def acquire_exclusive(self) -> None:
        async with self._cond:
            self._waiting_writers += 1
            try:
                while self._active_writer or self._active_readers > 0:
                    await self._cond.wait()
                self._active_writer = True
            finally:
                self._waiting_writers -= 1

    async def release_exclusive(self) -> None:
        async with self._cond:
            assert self._active_writer
            self._active_writer = False
            self._cond.notify_all()

    def shared(self) -> "_SharedGuard":
        return _SharedGuard(self)

    def exclusive(self) -> "_ExclusiveGuard":
        return _ExclusiveGuard(self)


class _SharedGuard:
    def __init__(self, lock: AsyncRWLock) -> None:
        self._lock = lock

    async def __aenter__(self) -> None:
        await self._lock.acquire_shared()

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self._lock.release_shared()


class _ExclusiveGuard:
    def __init__(self, lock: AsyncRWLock) -> None:
        self._lock = lock

    async def __aenter__(self) -> None:
        await self._lock.acquire_exclusive()

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self._lock.release_exclusive()
