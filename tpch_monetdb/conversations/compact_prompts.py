"""Structured summary prompts for context compaction.

This module provides prompts and utilities for generating structured summaries
during context compaction. The prompt is intentionally scoped to L2 history:
it preserves current work state and recent evidence, but it is not used to
re-inject project rules.
"""

import re

from tpch_monetdb.conversations.agent_text_registry import load_agent_text_asset

COMPACT_MAX_OUTPUT_TOKENS = 20_000

COMPACT_SYSTEM_PROMPT = load_agent_text_asset("compaction.system")


def format_compact_summary(llm_response: str) -> str:
    """Extract and format the summary block from LLM response.
    
    Extracts content from <summary> tags and formats it with a "Summary:" header.
    If no summary tags are found, uses the full response with a warning.
    
    Args:
        llm_response: The raw response from the LLM
        
    Returns:
        Formatted summary string ready for context storage
        
    Example:
        >>> response = "<analysis>...</analysis>\\n<summary>Key points...</summary>"
        >>> format_compact_summary(response)
        'Summary:\\nKey points...'
    """
    if not isinstance(llm_response, str):
        raise TypeError(
            "Compaction summary formatter requires a string response."
        )
    normalized_response = llm_response.strip()
    if not normalized_response:
        raise ValueError(
            "Compaction summary formatter requires a non-empty string response."
        )
    summary_match = re.search(
        r"<summary>(.*?)</summary>",
        normalized_response,
        re.DOTALL | re.IGNORECASE
    )
    
    if summary_match:
        summary_content = summary_match.group(1).strip()
        if not summary_content:
            raise ValueError("Compaction summary formatter extracted an empty <summary> block.")
        return f"Summary:\n{summary_content}"
    
    # No summary tags found - use full response
    return f"Summary:\n{normalized_response}"


def extract_analysis(llm_response: str) -> str | None:
    """Extract the analysis block from LLM response.
    
    This is primarily for debugging purposes. The analysis block
    is discarded from the final context but can be logged.
    
    Args:
        llm_response: The raw response from the LLM
        
    Returns:
        Analysis content if found, None otherwise
    """
    analysis_match = re.search(
        r"<analysis>(.*?)</analysis>",
        llm_response,
        re.DOTALL | re.IGNORECASE
    )
    
    if analysis_match:
        return analysis_match.group(1).strip()
    return None
