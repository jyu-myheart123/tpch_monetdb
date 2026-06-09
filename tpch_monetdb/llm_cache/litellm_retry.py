from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from typing import Awaitable, Callable, TypeVar

from .deepseek_reasoning_replay import DeepSeekReasoningReplayTransientError

T = TypeVar("T")

_RETRYABLE_STATUS_CODES = frozenset({408, 409, 425, 429, 500, 502, 503, 504})
_TRANSIENT_CLASS_TOKENS = (
    "apiconnectionerror",
    "internalservererror",
    "serviceunavailableerror",
    "ratelimiterror",
    "timeout",
    "connecterror",
    "readtimeout",
    "remoteprotocolerror",
)
_TRANSIENT_MESSAGE_TOKENS = (
    "server disconnected",
    "connection aborted",
    "temporarily unavailable",
    "service unavailable",
    "timed out",
    "timeout",
    "rate limit",
    "429",
    "502",
    "503",
    "504",
)

_PERSISTENT_CONNECTION_TOKENS = (
    "connection reset by peer",
    "connection refused",
)


def _iter_exception_chain(exc: BaseException) -> tuple[BaseException, ...]:
    chain: list[BaseException] = []
    current: BaseException | None = exc
    for _ in range(6):
        if current is None:
            break
        chain.append(current)
        current = current.__cause__ or current.__context__
    return tuple(chain)


def is_persistent_connection_error(exc: BaseException) -> bool:
    """Return True if the error indicates a persistent connection-level failure.

    Unlike transient errors (rate limits, timeouts, 5xx), these indicate
    the request itself is too large or malformed for the server's edge proxy.
    Retrying with the same payload will produce the same TCP RST.
    """
    for candidate in _iter_exception_chain(exc):
        message = str(candidate).lower()
        if any(token in message for token in _PERSISTENT_CONNECTION_TOKENS):
            return True
    return False


def is_transient_litellm_error(exc: BaseException) -> bool:
    """Return whether an exception chain looks like a retryable transport/provider failure."""
    for candidate in _iter_exception_chain(exc):
        if isinstance(candidate, DeepSeekReasoningReplayTransientError):
            return True
        if isinstance(candidate, (asyncio.TimeoutError, TimeoutError)):
            return True
        status_code = getattr(candidate, "status_code", None)
        if isinstance(status_code, int) and status_code in _RETRYABLE_STATUS_CODES:
            return True
        class_name = candidate.__class__.__name__.lower()
        if any(token in class_name for token in _TRANSIENT_CLASS_TOKENS):
            return True
        message = str(candidate).lower()
        if any(token in message for token in _TRANSIENT_MESSAGE_TOKENS):
            return True
    return False


async def run_with_transient_retry(
    *,
    operation_name: str,
    operation: Callable[[], Awaitable[T]],
    logger: logging.Logger,
    max_attempts: int = 3,
    base_delay_s: float = 1.0,
) -> T:
    """Run one async LLM operation with retry-on-transient-network/provider failures."""
    attempts = max(1, max_attempts)
    last_error: BaseException | None = None
    persistent_count = 0
    for attempt in range(1, attempts + 1):
        try:
            result = await operation()
            return result
        except Exception as exc:
            last_error = exc
            if is_persistent_connection_error(exc):
                persistent_count += 1
                if persistent_count >= 2:
                    logger.error(
                        "Persistent connection error during %s after %d attempts: %s; "
                        "request body likely exceeds server proxy limit — fast-failing",
                        operation_name, attempt, exc,
                    )
                    raise
            is_retryable = is_transient_litellm_error(exc)
            if attempt >= attempts or not is_retryable:
                raise
            delay_s = base_delay_s * (2 ** (attempt - 1))
            logger.warning(
                "Transient LLM error during %s (attempt %d/%d): %s; retrying in %.1fs",
                operation_name,
                attempt,
                attempts,
                exc,
                delay_s,
            )
            await asyncio.sleep(delay_s)
    if last_error is not None:
        raise last_error
    raise RuntimeError(f"{operation_name} retry loop ended without result")


async def run_stream_with_transient_retry(
    *,
    operation_name: str,
    operation: Callable[[], AsyncIterator[T]],
    logger: logging.Logger,
    max_attempts: int = 3,
    base_delay_s: float = 1.0,
) -> AsyncIterator[T]:
    """Run a streaming LLM operation with retry only before the first yielded event."""
    attempts = max(1, max_attempts)
    last_error: BaseException | None = None
    persistent_count = 0
    for attempt in range(1, attempts + 1):
        emitted = False
        try:
            async for item in operation():
                emitted = True
                yield item
            return
        except Exception as exc:
            last_error = exc
            if emitted:
                raise
            if is_persistent_connection_error(exc):
                persistent_count += 1
                if persistent_count >= 2:
                    logger.error(
                        "Persistent connection error during %s after %d attempts: %s; "
                        "request body likely exceeds server proxy limit — fast-failing",
                        operation_name, attempt, exc,
                    )
                    raise
            is_retryable = is_transient_litellm_error(exc)
            if attempt >= attempts or not is_retryable:
                raise
            delay_s = base_delay_s * (2 ** (attempt - 1))
            logger.warning(
                "Transient LLM stream error during %s before first event "
                "(attempt %d/%d): %s; retrying in %.1fs",
                operation_name,
                attempt,
                attempts,
                exc,
                delay_s,
            )
            await asyncio.sleep(delay_s)
    if last_error is not None:
        raise last_error
    raise RuntimeError(f"{operation_name} stream retry loop ended without result")
