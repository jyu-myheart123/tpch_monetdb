import logging

from agents import Usage
from openai.types.responses import ResponseUsage

from tpch_monetdb.llm_cache.models import context_window_usage, request_cost_usd

logger = logging.getLogger(__name__)


def _detail_value(details: object, field_name: str) -> int:
    """Read one token detail field while tolerating missing provider usage."""
    return int(getattr(details, field_name, 0) or 0)


def get_tokens_context_and_dollar_info(
    usage: Usage | ResponseUsage,
    model: str,
    last_entry_only: bool = True,
    log: bool = False,
) -> dict[str, float | int | str | None]:
    if isinstance(usage, ResponseUsage):
        assert last_entry_only, "last_entry_only must be True for ResponseUsage"
        num_llm_request = 1
        elem = usage
        last_entry = usage
    else:
        if usage.request_usage_entries:
            last_entry = usage.request_usage_entries[-1]
        else:
            last_entry = usage
        if last_entry_only or not usage.request_usage_entries:
            elem = last_entry
            num_llm_request = 1 if usage.request_usage_entries else 0
        else:
            elem = usage
            num_llm_request = len(usage.request_usage_entries)

    input_tokens = elem.input_tokens
    output_tokens = elem.output_tokens
    cached_tokens = _detail_value(elem.input_tokens_details, "cached_tokens")
    reasoning_tokens = _detail_value(
        elem.output_tokens_details,
        "reasoning_tokens",
    )
    last_request_input = last_entry.input_tokens
    last_request_output = last_entry.output_tokens

    try:
        usage_str, usage_float = context_window_usage(
            model, last_request_input + last_request_output
        )
        cost = request_cost_usd(model, input_tokens, cached_tokens, output_tokens)
        pricing_missing = False
    except KeyError as exc:
        logger.error("Missing pricing for model %s: %s", model, exc)
        usage_str = "n/a"
        usage_float = 0.0
        cost = None
        pricing_missing = True

    if log:
        cost_str = f"${cost:0.6f}" if cost is not None else "n/a"
        logger.info(
            f"Context window usage: {usage_str} | Input tokens: {input_tokens} "
            f"(cached: {cached_tokens}), Output tokens: {output_tokens} "
            f"(reasoning: {reasoning_tokens}) | Estimated cost: {cost_str} | "
            f"LLM requests: {num_llm_request}"
        )

    return {
        "input_tokens": input_tokens,
        "visible_output_tokens": output_tokens - reasoning_tokens,
        "billed_output_tokens": output_tokens,
        "cached_tokens": cached_tokens,
        "reasoning_tokens": reasoning_tokens,
        "context_window_usage_str": usage_str,
        "context_window_usage": usage_float,
        "cost": cost,
        "pricing_missing": pricing_missing,
        "num_llm_request": num_llm_request,
    }
