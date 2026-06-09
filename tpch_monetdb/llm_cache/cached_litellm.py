import logging
import time
from collections.abc import AsyncIterator
from copy import copy
from dataclasses import is_dataclass
from pathlib import Path
from typing import Any, Dict

from tpch_monetdb.bootstrap_env import bootstrap_runtime_env

bootstrap_runtime_env()

from agents import ApplyPatchTool, ShellTool
from agents.agent_output import AgentOutputSchemaBase
from agents.extensions.models.litellm_model import (
    Converter,
    LitellmModel,
    OpenAIResponsesConverter,
    _to_dump_compatible,
    litellm,
)
from agents.handoffs import Handoff
from agents.items import ModelResponse, TResponseStreamEvent
from agents.model_settings import ModelSettings
from agents.models._retry_runtime import should_disable_provider_managed_retries
from agents.tool import Tool
from agents.usage import (
    InputTokensDetails,
    OutputTokensDetails,
    RequestUsage,
    Usage,
)
from openai import AsyncStream, BaseModel
from openai.types.chat import ChatCompletionChunk
from openai.types.responses import (
    Response,
    ResponseCompletedEvent,
    ResponseCreatedEvent,
    ResponseUsage,
)
from tpch_monetdb.utils.token_usage import get_tokens_context_and_dollar_info
from tpch_monetdb.utils.truncate_csv import truncate_csvs_recursively
from tpch_monetdb.utils.model_aliases import is_deepseek_model

from . import utils
from .context_budget import BODY_FAIL_BYTES, build_provider_request_budget_estimate
from .deepseek_reasoning_replay import (
    DeepSeekReasoningReplayError,
    ensure_deepseek_response_output,
    ensure_deepseek_assistant_messages_have_reasoning_content,
    repair_deepseek_input_items,
)
from .git_snapshotter import GitSnapshotter
from .litellm_model_costs import register_tpch_monetdb_litellm_model_costs
from .litellm_retry import run_stream_with_transient_retry, run_with_transient_retry
from .micro_compact import micro_compact_tool_results
from .models import get_context_window
from .prompt_cache_diagnostics import PromptCacheDiagnostics

logger = logging.getLogger(__name__)

register_tpch_monetdb_litellm_model_costs()


def _is_sensitive_header(header_name: str) -> bool:
    """Return whether a header value should be masked in budget diagnostics."""
    lowered = header_name.lower()
    return any(
        marker in lowered
        for marker in ("authorization", "api-key", "api_key", "token", "secret")
    )


class CacheType:
    def __init__(self, response, parent_hash: str | None = None) -> None:
        self.response = response
        self.parent_hash = parent_hash
        return None


class CachedLitellmModel(LitellmModel):
    def __init__(
        self,
        *args,
        llm_cache_dir: Path,
        snapshotter: GitSnapshotter | None = None,
        stop_on_cache_miss: bool = False,
        query_gen_list: list[str] | None = None,
        artifacts_in_context: str | None = None,
        config_kwargs: Dict[str, Any] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.cache_dir = llm_cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.snapshotter = snapshotter
        self.stop_on_cache_miss = stop_on_cache_miss
        self.query_gen_list = query_gen_list
        self.artifacts_in_context = artifacts_in_context
        self.total_saved = 0.0
        self.llm_was_cached = False
        self.config_kwargs = config_kwargs if config_kwargs is not None else {}
        self.prompt_cache_diagnostics = PromptCacheDiagnostics()
        return None

    def _serialize_tools_for_hash(self, tools: list[Tool]) -> list[Any]:
        """Return deterministic tool representations for cache hashing."""
        tools_serialized = []
        current_tool: Tool | None = None
        try:
            for current_tool in tools:
                if isinstance(current_tool, ApplyPatchTool) or isinstance(
                    current_tool, ShellTool
                ):
                    data = current_tool.name
                elif isinstance(current_tool, BaseModel):
                    data = utils.stable_json(current_tool.to_dict())
                elif is_dataclass(current_tool):
                    data = current_tool.__dict__.copy()
                    data.pop("on_invoke_tool", None)
                    data = utils.stable_json(data)
                else:
                    raise Exception(f"Cannot hash tool of type {type(current_tool)}")

                assert "0x" not in data, (
                    "Cannot hash tool with non-deterministic data. "
                    f"Discovered likely a function or object reference in the tool definition: {data}"
                )

                tools_serialized.append(data)
        except Exception as e:
            logger.debug(
                f"Error serializing tools for hashing: {e}\n{str(current_tool)}"
            )
            raise Exception(f"Error serializing tools for hashing: {e}")
        return tools_serialized

    def _build_hash_payload(
        self,
        system_instructions: str | None,
        input: Any,
        model_settings: ModelSettings,
        tools: list[Tool],
        output_schema: AgentOutputSchemaBase | None,
        handoffs: list[Handoff],
        previous_response_id: str | None,
        conversation_id: str | None,
        prompt: Any | None,
        stream: bool = False,
    ) -> dict[str, Any]:
        """Build the exact payload used by the LiteLLM local cache key."""
        if handoffs:
            raise RuntimeError("Handoffs are not supported with caching.")

        config_kwargs_serialized = ",".join(
            f"{k}={v}" for k, v in sorted(self.config_kwargs.items())
        )

        payload = {
            "model": str(self.model),
            "system_instructions": system_instructions,
            "input": input,
            "model_settings": model_settings.to_json_dict(),
            "tools": self._serialize_tools_for_hash(tools),
            "output_schema": (
                output_schema.json_schema() if output_schema is not None else None
            ),
            "conversation_id": conversation_id,
            "previous_response_id": previous_response_id,
            "prompt": prompt,
            "query_gen_list": self.query_gen_list,
            "artifacts_in_context": self.artifacts_in_context,
            "config_kwargs": config_kwargs_serialized,
        }
        if stream:
            payload["stream"] = True
        return payload

    @staticmethod
    def _hash_cache_payload(payload: dict[str, Any]) -> str:
        """Return the SHA-256 key for a prepared LiteLLM cache payload."""
        return utils.sha256(utils.stable_json(payload))

    def _hash_payload(
        self,
        system_instructions: str | None,
        input: Any,
        model_settings: ModelSettings,
        tools: list[Tool],
        output_schema: AgentOutputSchemaBase | None,
        handoffs: list[Handoff],
        previous_response_id: str | None,
        conversation_id: str | None,
        prompt: Any | None,
        stream: bool = False,
    ) -> str:
        """Return a stable cache key for one LiteLLM model request."""
        payload = self._build_hash_payload(
            system_instructions,
            input,
            model_settings,
            tools,
            output_schema,
            handoffs,
            previous_response_id,
            conversation_id,
            prompt,
            stream=stream,
        )
        return self._hash_cache_payload(payload)

    @staticmethod
    def _provider_body_payload(completion_kwargs: dict[str, Any]) -> dict[str, Any]:
        """Return a conservative wire-budget payload sent through LiteLLM."""
        excluded = {"api_key", "base_url", "extra_headers"}
        payload = {
            key: value
            for key, value in completion_kwargs.items()
            if key not in excluded and value is not None
        }
        header_payload = CachedLitellmModel._provider_header_budget_payload(
            completion_kwargs.get("extra_headers")
        )
        if header_payload:
            payload["__http_headers__"] = header_payload
        return payload

    @staticmethod
    def _provider_header_budget_payload(extra_headers: Any) -> dict[str, str]:
        """Return header-sized budget payload without exposing sensitive values."""
        if not isinstance(extra_headers, dict):
            return {}
        rendered: dict[str, str] = {}
        for key, value in extra_headers.items():
            key_text = str(key)
            value_text = str(value)
            if _is_sensitive_header(key_text):
                rendered[key_text] = "*" * len(value_text)
            else:
                rendered[key_text] = value_text
        return rendered

    def _provider_token_limit(self) -> int:
        """Return the best known context window for provider budget diagnostics."""
        try:
            return get_context_window(str(self.model))
        except KeyError:
            return 0

    def _enforce_provider_body_budget(self, payload: dict[str, Any]) -> None:
        """Fail closed before LiteLLM sends an oversized provider request body."""
        budget = build_provider_request_budget_estimate(
            payload,
            token_limit=self._provider_token_limit(),
        )
        if budget.should_warn:
            contributors = " | ".join(
                f"{item.source}[{item.item_index if item.item_index is not None else '-'}]="
                f"{item.byte_size}B:{item.summary}"
                for item in budget.top_contributors
            )
            logger.warning(
                "Provider request budget: body=%s bytes(level=%s) tokens=%s(level=%s) contributors=%s",
                budget.body_bytes,
                budget.body_level,
                budget.token_estimate,
                budget.token_level,
                contributors or "(none)",
            )
        if budget.body_fail:
            raise RuntimeError(
                "Provider request body exceeds fail-closed threshold: "
                f"{budget.body_bytes} bytes >= {BODY_FAIL_BYTES} bytes. "
                "Run deterministic trim/compaction before sending to DeepSeek."
            )
        return None

    def _cache_path_for(self, hash: str) -> Path:
        return self.cache_dir / f"{hash}.pkl"

    def __str__(self) -> str:
        return str(self.model)

    @staticmethod
    def _ensure_usage_entries(usage: Usage) -> None:
        if usage.request_usage_entries:
            return None
        if usage.total_tokens <= 0:
            return None
        request = RequestUsage(
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            total_tokens=usage.total_tokens,
            input_tokens_details=usage.input_tokens_details,
            output_tokens_details=usage.output_tokens_details,
        )
        usage.request_usage_entries.append(request)
        return None

    def _prepare_input_for_request(
        self,
        input: Any,
        require_reasoning_content: bool,
    ) -> Any:
        if not input or not isinstance(input, list):
            return input
        prepared_input = micro_compact_tool_results(input)
        if not require_reasoning_content:
            return prepared_input
        return repair_deepseek_input_items(
            prepared_input,
            model_name=str(self.model),
            fail_on_unrecoverable=True,
            require_reasoning_for_tool_calls=True,
        )

    @staticmethod
    def _usage_from_response_usage(response_usage: Any | None) -> Usage:
        if response_usage is None:
            return Usage()
        input_details = response_usage.input_tokens_details or InputTokensDetails(
            cached_tokens=0
        )
        output_details = response_usage.output_tokens_details or OutputTokensDetails(
            reasoning_tokens=0
        )
        return Usage(
            requests=1,
            input_tokens=response_usage.input_tokens,
            input_tokens_details=input_details,
            output_tokens=response_usage.output_tokens,
            output_tokens_details=output_details,
            total_tokens=response_usage.total_tokens,
        )

    @staticmethod
    def _response_usage_from_usage(usage: Usage) -> ResponseUsage:
        return ResponseUsage(
            input_tokens=usage.input_tokens,
            input_tokens_details=usage.input_tokens_details,
            output_tokens=usage.output_tokens,
            output_tokens_details=usage.output_tokens_details,
            total_tokens=usage.total_tokens,
        )

    def _model_response_from_response(self, response: Response) -> ModelResponse:
        usage = self._usage_from_response_usage(response.usage)
        return ModelResponse(
            output=list(response.output or []),
            usage=usage,
            response_id=response.id,
        )

    def _response_from_model_response(
        self,
        response: ModelResponse,
        model_settings: ModelSettings,
    ) -> Response:
        responses_tool_choice = OpenAIResponsesConverter.convert_tool_choice(
            model_settings.tool_choice
        )
        responses_tool_choice = self._remove_not_given(responses_tool_choice)
        if responses_tool_choice is None:
            responses_tool_choice = "auto"
        parallel_tool_calls = bool(
            model_settings.parallel_tool_calls and response.output
        )
        return Response(
            id=response.response_id or "cached-resp",
            created_at=time.time(),
            model=self.model,
            object="response",
            output=response.output,
            tool_choice=responses_tool_choice,  # type: ignore[arg-type]
            top_p=model_settings.top_p,
            temperature=model_settings.temperature,
            tools=[],
            parallel_tool_calls=parallel_tool_calls,
            reasoning=model_settings.reasoning,
            usage=self._response_usage_from_usage(response.usage),
        )

    async def _stream_cached_response(
        self,
        response: ModelResponse,
        model_settings: ModelSettings,
    ) -> AsyncIterator[TResponseStreamEvent]:
        completed_response = self._response_from_model_response(
            response,
            model_settings,
        )
        yield ResponseCreatedEvent(
            response=completed_response.model_copy(update={"output": []}),
            sequence_number=0,
            type="response.created",
        )
        yield ResponseCompletedEvent(
            response=completed_response,
            sequence_number=1,
            type="response.completed",
        )
        return

    def _prepare_cached_response(self, path: Path, cached: CacheType) -> bool:
        assert self.snapshotter is not None
        if cached.parent_hash:
            exists = self.snapshotter.has_snapshot(cached.parent_hash)
            if not exists:
                self.snapshotter.fetch_snapshots()
            exists = self.snapshotter.has_snapshot(cached.parent_hash)
            if not exists:
                logger.warning(
                    "Ignoring cached response %s because snapshot %s is missing.",
                    path,
                    cached.parent_hash,
                )
                path.unlink(missing_ok=True)
                return False

            self.snapshotter.clear_untracked(include_ignored=True)
            self.snapshotter.reset_changes()
            self.snapshotter.restore(cached.parent_hash)
            return True

        if self.snapshotter.is_dirty():
            logger.warning(
                "Ignoring cached response %s because it has no parent hash and the workspace is dirty.",
                path,
            )
            path.unlink(missing_ok=True)
            return False
        return True

    @staticmethod
    def _requires_deepseek_reasoning_replay(model_settings: Any) -> bool:
        # Check extra_body at ModelSettings top level (native deepseek/ path)
        extra_body = getattr(model_settings, "extra_body", None)
        if isinstance(extra_body, dict):
            thinking = extra_body.get("thinking")
            if isinstance(thinking, dict) and thinking.get("type") == "enabled":
                return True
        # Check extra_body nested inside extra_args (legacy openai/ path)
        extra_args = getattr(model_settings, "extra_args", None)
        if isinstance(extra_args, dict):
            nested_extra_body = extra_args.get("extra_body")
            if isinstance(nested_extra_body, dict):
                thinking = nested_extra_body.get("thinking")
                if isinstance(thinking, dict) and thinking.get("type") == "enabled":
                    return True
            # Check thinking directly in extra_args (alternate legacy format)
            thinking = extra_args.get("thinking")
            if isinstance(thinking, dict) and thinking.get("type") == "enabled":
                return True
        return False

    @staticmethod
    def _is_deepseek_thinking_request(model: str, model_settings: Any) -> bool:
        """Return whether the current request targets DeepSeek thinking mode."""
        return is_deepseek_model(model) and CachedLitellmModel._requires_deepseek_reasoning_replay(
            model_settings
        )

    @staticmethod
    def _drop_deepseek_thinking_sampling_params(
        *,
        is_deepseek_thinking: bool,
        temperature: Any,
        top_p: Any,
        frequency_penalty: Any,
        presence_penalty: Any,
    ) -> tuple[Any, Any, Any, Any]:
        """Drop sampling params that DeepSeek ignores in thinking mode."""
        if not is_deepseek_thinking:
            return temperature, top_p, frequency_penalty, presence_penalty
        return None, None, None, None

    def _normalize_deepseek_response(
        self,
        response: Any,
        *,
        fallback_reasoning_content: str | None = None,
        response_id: str | None = None,
        require_reasoning_content: bool,
    ) -> Any:
        if not is_deepseek_model(str(self.model)):
            return response
        output = getattr(response, "output", None)
        if not isinstance(output, list):
            return response
        response.output = ensure_deepseek_response_output(
            output,
            model_name=str(self.model),
            fallback_reasoning_content=fallback_reasoning_content,
            response_id=response_id,
            require_reasoning_content=require_reasoning_content,
        )
        return response

    async def _fetch_response(
        self,
        system_instructions: str | None,
        input: str | list[Any],
        model_settings: ModelSettings,
        tools: list[Tool],
        output_schema: AgentOutputSchemaBase | None,
        handoffs: list[Handoff],
        span: Any,
        tracing: Any,
        stream: bool = False,
        prompt: Any | None = None,
    ) -> litellm.types.utils.ModelResponse | tuple[Response, AsyncStream[ChatCompletionChunk]]:
        """Wrap upstream message conversion so DeepSeek tool continuations get a final guard."""
        preserve_thinking_blocks = (
            model_settings.reasoning is not None
            and model_settings.reasoning.effort is not None
        )

        converted_messages = Converter.items_to_messages(
            input,
            base_url=self.base_url,
            preserve_thinking_blocks=preserve_thinking_blocks,
            preserve_tool_output_all_content=True,
            model=self.model,
            should_replay_reasoning_content=self.should_replay_reasoning_content,
        )
        converted_messages = ensure_deepseek_assistant_messages_have_reasoning_content(
            converted_messages,
            model_name=str(self.model),
        )

        if any(
            model.lower() in self.model.lower()
            for model in ["anthropic", "claude", "gemini"]
        ):
            converted_messages = self._fix_tool_message_ordering(converted_messages)

        if "gemini" in self.model.lower():
            converted_messages = (
                self._convert_gemini_extra_content_to_provider_specific_fields(
                    converted_messages
                )
            )

        if system_instructions:
            converted_messages.insert(
                0,
                {
                    "content": system_instructions,
                    "role": "system",
                },
            )
        converted_messages = _to_dump_compatible(converted_messages)

        if tracing.include_data():
            span.span_data.input = converted_messages

        parallel_tool_calls = (
            True
            if model_settings.parallel_tool_calls and tools and len(tools) > 0
            else False
            if model_settings.parallel_tool_calls is False
            else None
        )
        tool_choice = Converter.convert_tool_choice(model_settings.tool_choice)
        response_format = Converter.convert_response_format(output_schema)

        converted_tools = [Converter.tool_to_openai(tool) for tool in tools] if tools else []
        for handoff in handoffs:
            converted_tools.append(Converter.convert_handoff_tool(handoff))
        converted_tools = _to_dump_compatible(converted_tools)

        if model_settings.extra_args:
            extra_kwargs = dict(model_settings.extra_args)
        else:
            extra_kwargs = {}
        if model_settings.extra_query:
            extra_kwargs["extra_query"] = copy(model_settings.extra_query)
        if model_settings.metadata:
            extra_kwargs["metadata"] = copy(model_settings.metadata)
        if model_settings.extra_body and isinstance(model_settings.extra_body, dict):
            extra_kwargs.update(model_settings.extra_body)

        if should_disable_provider_managed_retries():
            extra_kwargs["num_retries"] = 0
            extra_kwargs["max_retries"] = 0

        is_deepseek_thinking = self._is_deepseek_thinking_request(
            str(self.model),
            model_settings,
        )
        reasoning_effort = self._get_reasoning_effort(model_settings)
        extra_kwargs.pop("reasoning_effort", None)
        temperature, top_p, frequency_penalty, presence_penalty = (
            self._drop_deepseek_thinking_sampling_params(
                is_deepseek_thinking=is_deepseek_thinking,
                temperature=model_settings.temperature,
                top_p=model_settings.top_p,
                frequency_penalty=model_settings.frequency_penalty,
                presence_penalty=model_settings.presence_penalty,
            )
        )

        stream_options = None
        if stream and model_settings.include_usage is not None:
            stream_options = {"include_usage": model_settings.include_usage}

        completion_kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": converted_messages,
            "tools": converted_tools or None,
            "temperature": temperature,
            "top_p": top_p,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
            "max_tokens": model_settings.max_tokens,
            "tool_choice": self._remove_not_given(tool_choice),
            "response_format": self._remove_not_given(response_format),
            "parallel_tool_calls": parallel_tool_calls,
            "stream": stream,
            "stream_options": stream_options,
            "reasoning_effort": reasoning_effort,
            "top_logprobs": model_settings.top_logprobs,
            "extra_headers": self._merge_headers(model_settings),
            "api_key": self.api_key,
            "base_url": self.base_url,
        }
        completion_kwargs.update(extra_kwargs)
        self._enforce_provider_body_budget(
            self._provider_body_payload(completion_kwargs)
        )
        ret = await litellm.acompletion(**completion_kwargs)

        if isinstance(ret, litellm.types.utils.ModelResponse):
            return ret

        responses_tool_choice = OpenAIResponsesConverter.convert_tool_choice(
            model_settings.tool_choice
        )
        responses_tool_choice = self._remove_not_given(responses_tool_choice)
        if responses_tool_choice is None:
            responses_tool_choice = "auto"

        response = Response(
            id="fake-resp",
            created_at=time.time(),
            model=self.model,
            object="response",
            output=[],
            tool_choice=responses_tool_choice,  # type: ignore[arg-type]
            top_p=model_settings.top_p,
            temperature=model_settings.temperature,
            tools=[],
            parallel_tool_calls=parallel_tool_calls or False,
            reasoning=model_settings.reasoning,
        )
        return response, ret

    async def _fetch_uncached_live_response(self, *args, **kwargs) -> Any:
        """Fetch one live LiteLLM response, preserving DeepSeek reasoning replay state."""
        model_settings = kwargs.get("model_settings")
        if not is_deepseek_model(str(self.model)) or model_settings is None:
            return await super(CachedLitellmModel, self).get_response(*args, **kwargs)

        # Delegate to parent's standard pipeline, then normalize DeepSeek reasoning
        response = await super(CachedLitellmModel, self).get_response(*args, **kwargs)
        require_reasoning_content = self._requires_deepseek_reasoning_replay(model_settings)
        response.output = ensure_deepseek_response_output(
            response.output,
            model_name=str(self.model),
            fallback_reasoning_content=None,
            response_id=None,
            require_reasoning_content=require_reasoning_content,
        )
        return response

    async def get_response(self, *args, **kwargs) -> Any:
        """Return a cached or live LiteLLM response, normalizing DeepSeek continuation items."""
        system_instructions = kwargs.get("system_instructions")
        input = kwargs.get("input")
        model_settings = kwargs.get("model_settings")
        require_reasoning_content = self._requires_deepseek_reasoning_replay(
            model_settings
        )

        input = self._prepare_input_for_request(input, require_reasoning_content)
        kwargs["input"] = input

        tools = kwargs.get("tools") or []
        output_schema = kwargs.get("output_schema")
        handoffs = kwargs.get("handoffs") or []
        previous_response_id = kwargs.get("previous_response_id")
        conversation_id = kwargs.get("conversation_id")
        prompt = kwargs.get("prompt")

        payload = self._build_hash_payload(
            system_instructions,
            input,
            model_settings,
            tools,
            output_schema,
            handoffs,
            previous_response_id,
            conversation_id,
            prompt,
        )
        req_hash = self._hash_cache_payload(payload)

        path = self._cache_path_for(req_hash)

        if path.exists():
            cached = utils.load_pickle(path, CacheType)
            if cached is not None:
                try:
                    resp = self._normalize_deepseek_response(
                        cached.response,
                        require_reasoning_content=require_reasoning_content,
                    )
                except DeepSeekReasoningReplayError:
                    logger.warning(
                        "Bypassing cached DeepSeek response %s because reasoning replay state is unrecoverable.",
                        path,
                    )
                    cached = None
                else:
                    if not self._prepare_cached_response(path, cached):
                        cached = None
                    else:
                        self._ensure_usage_entries(resp.usage)
                        cost = get_tokens_context_and_dollar_info(
                            resp.usage, self.model, last_entry_only=True, log=False
                        )["cost"]
                        if cost is not None:
                            logger.debug(f"Saved: ${cost:0.6f}")
                            self.total_saved += cost

                        self.llm_was_cached = True
                        return resp

        if self.stop_on_cache_miss:
            raise Exception("Stop on cache miss. Did not found in cache: " + str(path))

        retry_attempts = int(self.config_kwargs.get("network_retry_attempts", 3))
        retry_delay_s = float(self.config_kwargs.get("network_retry_base_delay_s", 1.0))
        prompt_cache_pending = self.prompt_cache_diagnostics.begin_request(
            request_hash=req_hash,
            payload=payload,
            stream=False,
        )

        async def _fetch_uncached_response() -> Any:
            return await self._fetch_uncached_live_response(*args, **kwargs)

        resp = await run_with_transient_retry(
            operation_name=f"litellm model request ({self.model})",
            operation=_fetch_uncached_response,
            logger=logger,
            max_attempts=retry_attempts,
            base_delay_s=retry_delay_s,
        )
        resp = self._normalize_deepseek_response(
            resp,
            require_reasoning_content=require_reasoning_content,
        )
        self._ensure_usage_entries(resp.usage)
        self.prompt_cache_diagnostics.complete_request(
            prompt_cache_pending,
            resp.usage,
            model=str(self.model),
        )
        cost = get_tokens_context_and_dollar_info(
            resp.usage, self.model, last_entry_only=True, log=False
        )["cost"]

        if cost is not None:
            logger.debug(f"Cost: ${cost:0.6f}")

        assert self.snapshotter is not None

        if self.config_kwargs.get("max_snapshot_csv_size_mb") is not None:
            truncate_csvs_recursively(
                self.snapshotter.working_dir,
                max_size_mb=self.config_kwargs["max_snapshot_csv_size_mb"],
            )

        _, commit = self.snapshotter.snapshot(req_hash)

        utils.dump_pickle(path, CacheType(resp, parent_hash=commit))

        self.snapshotter.push_snapshots()

        self.llm_was_cached = False
        return resp

    async def stream_response(
        self,
        system_instructions: str | None,
        input: str | list[Any],
        model_settings: ModelSettings,
        tools: list[Tool],
        output_schema: AgentOutputSchemaBase | None,
        handoffs: list[Handoff],
        tracing: Any,
        previous_response_id: str | None = None,
        conversation_id: str | None = None,
        prompt: Any | None = None,
    ) -> AsyncIterator[TResponseStreamEvent]:
        """Stream a cached or live LiteLLM response while preserving TPC-H MonetDB snapshots."""
        require_reasoning_content = self._requires_deepseek_reasoning_replay(
            model_settings
        )
        input = self._prepare_input_for_request(input, require_reasoning_content)
        payload = self._build_hash_payload(
            system_instructions,
            input,
            model_settings,
            tools,
            output_schema,
            handoffs,
            previous_response_id,
            conversation_id,
            prompt,
            stream=True,
        )
        req_hash = self._hash_cache_payload(payload)
        path = self._cache_path_for(req_hash)
        if path.exists():
            cached = utils.load_pickle(path, CacheType)
            if cached is not None:
                try:
                    resp = self._normalize_deepseek_response(
                        cached.response,
                        require_reasoning_content=require_reasoning_content,
                    )
                except DeepSeekReasoningReplayError:
                    logger.warning(
                        "Bypassing cached DeepSeek stream response %s because reasoning replay state is unrecoverable.",
                        path,
                    )
                    cached = None
                else:
                    if not self._prepare_cached_response(path, cached):
                        cached = None
                    else:
                        self._ensure_usage_entries(resp.usage)
                        cost = get_tokens_context_and_dollar_info(
                            resp.usage, self.model, last_entry_only=True, log=False
                        )["cost"]
                        if cost is not None:
                            logger.debug(f"Saved: ${cost:0.6f}")
                            self.total_saved += cost
                        self.llm_was_cached = True
                        async for event in self._stream_cached_response(
                            resp,
                            model_settings,
                        ):
                            yield event
                        return
        if self.stop_on_cache_miss:
            raise Exception("Stop on cache miss. Did not found in cache: " + str(path))

        retry_attempts = int(self.config_kwargs.get("network_retry_attempts", 3))
        retry_delay_s = float(self.config_kwargs.get("network_retry_base_delay_s", 1.0))
        final_response: Response | None = None
        prompt_cache_pending = self.prompt_cache_diagnostics.begin_request(
            request_hash=req_hash,
            payload=payload,
            stream=True,
        )

        async def _fetch_uncached_stream_response() -> AsyncIterator[TResponseStreamEvent]:
            async for event in super(CachedLitellmModel, self).stream_response(
                system_instructions,
                input,
                model_settings,
                tools,
                output_schema,
                handoffs,
                tracing,
                previous_response_id=previous_response_id,
                conversation_id=conversation_id,
                prompt=prompt,
            ):
                yield event
            return

        async for event in run_stream_with_transient_retry(
            operation_name=f"litellm model stream request ({self.model})",
            operation=_fetch_uncached_stream_response,
            logger=logger,
            max_attempts=retry_attempts,
            base_delay_s=retry_delay_s,
        ):
            if isinstance(event, ResponseCompletedEvent):
                final_response = event.response
            yield event
        if final_response is None:
            raise RuntimeError("Streaming LiteLLM response ended without response.completed")

        resp = self._model_response_from_response(final_response)
        resp = self._normalize_deepseek_response(
            resp,
            require_reasoning_content=require_reasoning_content,
        )
        self._ensure_usage_entries(resp.usage)
        self.prompt_cache_diagnostics.complete_request(
            prompt_cache_pending,
            resp.usage,
            model=str(self.model),
        )
        cost = get_tokens_context_and_dollar_info(
            resp.usage, self.model, last_entry_only=True, log=False
        )["cost"]
        if cost is not None:
            logger.debug(f"Cost: ${cost:0.6f}")
        assert self.snapshotter is not None
        if self.config_kwargs.get("max_snapshot_csv_size_mb") is not None:
            truncate_csvs_recursively(
                self.snapshotter.working_dir,
                max_size_mb=self.config_kwargs["max_snapshot_csv_size_mb"],
            )
        _, commit = self.snapshotter.snapshot(req_hash)
        utils.dump_pickle(path, CacheType(resp, parent_hash=commit))
        self.snapshotter.push_snapshots()
        self.llm_was_cached = False
        return
