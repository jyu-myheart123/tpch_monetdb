import logging
from dataclasses import dataclass, field
from typing import Any, Mapping

from . import utils

logger = logging.getLogger(__name__)

MIN_CACHE_MISS_TOKENS = 2_000
MIN_CACHE_RETAIN_RATIO = 0.95
LOW_CACHE_READ_RATIO = 0.05
INPUT_PREFIX_ITEMS = 4
INPUT_PREFIX_CHARS = 4_000

COMPONENT_ORDER = (
    "model",
    "system_instructions",
    "tools",
    "model_settings",
    "output_schema",
    "prompt",
    "query_gen_list",
    "artifacts_in_context",
    "config_kwargs",
    "conversation_id",
    "previous_response_id",
    "input_prefix",
    "input",
    "stream",
)


@dataclass(frozen=True)
class PromptCachePendingRequest:
    request_hash: str
    component_hashes: Mapping[str, str]
    changed_components: tuple[str, ...]
    request_number: int
    stream: bool


@dataclass(frozen=True)
class PromptCacheObservation:
    request_hash: str
    component_hashes: Mapping[str, str]
    cached_tokens: int
    input_tokens: int
    stream: bool


@dataclass
class PromptCacheDiagnostics:
    min_cache_miss_tokens: int = MIN_CACHE_MISS_TOKENS
    min_cache_retain_ratio: float = MIN_CACHE_RETAIN_RATIO
    low_cache_read_ratio: float = LOW_CACHE_READ_RATIO
    _request_count: int = 0
    _last_by_stream: dict[bool, PromptCacheObservation] = field(default_factory=dict)
    _near_zero_warning_keys: set[tuple[bool, tuple[str, ...], str]] = field(
        default_factory=set
    )

    def begin_request(
        self,
        *,
        request_hash: str,
        payload: Mapping[str, Any],
        stream: bool,
    ) -> PromptCachePendingRequest:
        """Record the prompt state before one live provider request."""
        self._request_count += 1
        component_hashes = summarize_prompt_cache_payload(payload, stream=stream)
        previous = self._last_by_stream.get(stream)
        changed = (
            changed_prompt_components(previous.component_hashes, component_hashes)
            if previous is not None
            else tuple()
        )
        if changed:
            logger.debug(
                "LLM provider prompt state changed before request %s: changed=%s",
                request_hash,
                ",".join(changed),
            )
        return PromptCachePendingRequest(
            request_hash=request_hash,
            component_hashes=component_hashes,
            changed_components=changed,
            request_number=self._request_count,
            stream=stream,
        )

    def complete_request(
        self,
        pending: PromptCachePendingRequest,
        usage: Any,
        *,
        model: str,
    ) -> None:
        """Compare provider cache tokens after one live request and log likely breaks."""
        input_tokens, cached_tokens = extract_provider_cache_tokens(usage)
        previous = self._last_by_stream.get(pending.stream)
        if previous is not None and input_tokens >= self.min_cache_miss_tokens:
            self._log_provider_cache_signal(
                pending=pending,
                previous=previous,
                input_tokens=input_tokens,
                cached_tokens=cached_tokens,
                model=model,
            )
        self._last_by_stream[pending.stream] = PromptCacheObservation(
            request_hash=pending.request_hash,
            component_hashes=pending.component_hashes,
            cached_tokens=cached_tokens,
            input_tokens=input_tokens,
            stream=pending.stream,
        )
        return None

    def _log_provider_cache_signal(
        self,
        *,
        pending: PromptCachePendingRequest,
        previous: PromptCacheObservation,
        input_tokens: int,
        cached_tokens: int,
        model: str,
    ) -> None:
        """Emit a concise prompt-cache warning when provider reuse collapses."""
        token_drop = previous.cached_tokens - cached_tokens
        changed = ",".join(pending.changed_components) or "none"
        if (
            token_drop >= self.min_cache_miss_tokens
            and cached_tokens < previous.cached_tokens * self.min_cache_retain_ratio
        ):
            logger.warning(
                "LLM provider prompt cache break: cached_tokens %s -> %s, "
                "input_tokens=%s, changed=%s, model=%s, request=%s",
                previous.cached_tokens,
                cached_tokens,
                input_tokens,
                changed,
                model,
                pending.request_hash,
            )
            return None
        if not _model_is_cache_diagnostic_target(model):
            return None
        cached_ratio = cached_tokens / input_tokens if input_tokens > 0 else 0.0
        if cached_ratio < self.low_cache_read_ratio:
            warning_key = (pending.stream, pending.changed_components, model)
            if warning_key in self._near_zero_warning_keys:
                return None
            self._near_zero_warning_keys.add(warning_key)
            logger.warning(
                "LLM provider prompt cache read is near zero: cached_tokens=%s, "
                "input_tokens=%s, changed=%s, model=%s, request=%s",
                cached_tokens,
                input_tokens,
                changed,
                model,
                pending.request_hash,
            )
        return None


def summarize_prompt_cache_payload(
    payload: Mapping[str, Any],
    *,
    stream: bool,
) -> dict[str, str]:
    """Return redacted component hashes for provider prompt-cache diagnosis."""
    component_values = {
        key: payload.get(key)
        for key in COMPONENT_ORDER
        if key not in {"input_prefix", "stream"}
    }
    component_values["input_prefix"] = _input_prefix(payload.get("input"))
    component_values["stream"] = stream
    return {
        key: _stable_component_hash(component_values.get(key))
        for key in COMPONENT_ORDER
    }


def changed_prompt_components(
    previous_hashes: Mapping[str, str],
    current_hashes: Mapping[str, str],
) -> tuple[str, ...]:
    """Return ordered component names whose redacted hashes changed."""
    return tuple(
        key
        for key in COMPONENT_ORDER
        if previous_hashes.get(key) != current_hashes.get(key)
    )


def extract_provider_cache_tokens(usage: Any) -> tuple[int, int]:
    """Extract input and cached-token counts from an Agents SDK Usage object."""
    entries = getattr(usage, "request_usage_entries", None)
    elem = entries[-1] if entries else usage
    input_tokens = _int_value(_get_usage_value(elem, "input_tokens"))
    details = getattr(elem, "input_tokens_details", None)
    cached_tokens = _int_value(getattr(details, "cached_tokens", 0))
    deepseek_hit = _int_value(_get_usage_value(elem, "prompt_cache_hit_tokens"))
    deepseek_miss = _int_value(_get_usage_value(elem, "prompt_cache_miss_tokens"))
    if deepseek_hit > cached_tokens:
        cached_tokens = deepseek_hit
    if input_tokens <= 0 and (deepseek_hit > 0 or deepseek_miss > 0):
        input_tokens = deepseek_hit + deepseek_miss
    return input_tokens, cached_tokens


def _get_usage_value(value: Any, key: str) -> Any:
    """Read a usage value from object or dict-like provider payloads."""
    if isinstance(value, Mapping):
        return value.get(key, 0)
    return getattr(value, key, 0)


def _input_prefix(value: Any) -> Any:
    """Return a bounded prefix that approximates the provider-cache-sensitive input."""
    if isinstance(value, str):
        return value[:INPUT_PREFIX_CHARS]
    if isinstance(value, list):
        return value[:INPUT_PREFIX_ITEMS]
    return value


def _stable_component_hash(value: Any) -> str:
    """Hash one redacted component with the same stable JSON rules as the cache key."""
    return utils.sha256(utils.stable_json(value))


def _int_value(value: Any) -> int:
    """Coerce numeric SDK fields to int without raising on missing values."""
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _model_is_cache_diagnostic_target(model: str) -> bool:
    """Return whether near-zero provider cache reads should be surfaced for this model."""
    normalized = model.lower()
    return (
        "gpt" in normalized
        or "openai" in normalized
        or "claude" in normalized
        or "anthropic" in normalized
        or "deepseek" in normalized
    )
