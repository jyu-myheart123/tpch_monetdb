"""Declarative stage runtime policy for scripted execution.

Defines turn budgets, proactive-compact rules, and progress-gated budget
extensions for high-risk scripted stages.
"""

from dataclasses import dataclass
from typing import Any

from tpch_monetdb.config import (
    PHASE10_OPTIM_STAGE_MAX_TURNS,
    PHASE10_STAGE_TURN_BUDGETS,
    get_stage_turn_budget,
)
from tpch_monetdb.tools.stage_tool_policy import StageRunSummary


@dataclass(frozen=True)
class StageRuntimePolicy:
    """Frozen policy for one stage type."""

    base_turns: int
    extra_turns: int
    max_extensions: int
    proactive_compact_on_warning: bool
    block_on_context_saturation: bool
    diagnostic_summary_budget: int = 800
    stage_end_maintenance: bool = True
    stage_end_llm_compact_on_orange: bool = True
    stage_end_force_llm_compact: bool = False


STAGE_RUNTIME_POLICIES: dict[str, StageRuntimePolicy] = {
    "todo_plan": StageRuntimePolicy(
        base_turns=PHASE10_STAGE_TURN_BUDGETS["todo_plan"],
        extra_turns=60,
        max_extensions=2,
        proactive_compact_on_warning=True,
        block_on_context_saturation=True,
    ),
    "compile_fix": StageRuntimePolicy(
        base_turns=PHASE10_STAGE_TURN_BUDGETS["compile_fix"],
        extra_turns=128,
        max_extensions=2,
        proactive_compact_on_warning=True,
        block_on_context_saturation=True,
    ),
    "correctness_primary_query": StageRuntimePolicy(
        base_turns=PHASE10_STAGE_TURN_BUDGETS["correctness_primary_query"],
        extra_turns=64,
        max_extensions=2,
        proactive_compact_on_warning=True,
        block_on_context_saturation=True,
    ),
    "correctness_single_query": StageRuntimePolicy(
        base_turns=PHASE10_STAGE_TURN_BUDGETS["correctness_single_query"],
        extra_turns=64,
        max_extensions=2,
        proactive_compact_on_warning=True,
        block_on_context_saturation=True,
    ),
    "all_queries_correctness": StageRuntimePolicy(
        base_turns=PHASE10_STAGE_TURN_BUDGETS["all_queries_correctness"],
        extra_turns=64,
        max_extensions=2,
        proactive_compact_on_warning=True,
        block_on_context_saturation=True,
    ),
    "trace_expert": StageRuntimePolicy(
        base_turns=PHASE10_OPTIM_STAGE_MAX_TURNS["trace_expert"],
        extra_turns=96,
        max_extensions=2,
        proactive_compact_on_warning=True,
        block_on_context_saturation=True,
        diagnostic_summary_budget=1200,
    ),
    "global_human_reference": StageRuntimePolicy(
        base_turns=PHASE10_OPTIM_STAGE_MAX_TURNS["global_human_reference"],
        extra_turns=96,
        max_extensions=2,
        proactive_compact_on_warning=True,
        block_on_context_saturation=True,
        diagnostic_summary_budget=1600,
    ),
}

# Alias to phase10 declarative budget — kept for backward compatibility
DEFAULT_STAGE_TURN_BUDGET: dict[str, int] = dict(PHASE10_STAGE_TURN_BUDGETS)


class StageBudgetTracker:
    """Tracks per-stage execution history for progress-based budget extension."""

    def __init__(self) -> None:
        self._last_summary_by_stage: dict[str, StageRunSummary | None] = {}
        self._extensions_used: dict[str, int] = {}
        return None

    def compute_effective_max_turns(
        self,
        stage_descriptor: str,
        static_budget: int | None = None,
    ) -> int:
        """Return effective max_turns for a stage, accounting for extensions."""
        policy = get_policy_for_stage(stage_descriptor)
        if policy is None:
            return static_budget if static_budget is not None else 75
        base = policy.base_turns
        extensions = self._extensions_used.get(stage_descriptor, 0)
        return base + extensions * policy.extra_turns

    def record_stage_result(
        self,
        stage_descriptor: str,
        summary: StageRunSummary | None,
    ) -> None:
        """Record a stage result and unlock an extension if progress is detected."""
        last_summary = self._last_summary_by_stage.get(stage_descriptor)
        self._last_summary_by_stage[stage_descriptor] = summary

        policy = get_policy_for_stage(stage_descriptor)
        if policy is None:
            return None
        if summary is None:
            return None

        extensions_used = self._extensions_used.get(stage_descriptor, 0)
        if extensions_used >= policy.max_extensions:
            return None

        if self._has_progress(last_summary, summary):
            self._extensions_used[stage_descriptor] = extensions_used + 1
        return None

    @staticmethod
    def _has_progress(
        previous: StageRunSummary | None,
        current: StageRunSummary | None,
    ) -> bool:
        """Detect progress using only the 4 stable signals."""
        if current is None:
            return False
        if current.has_writes:
            return True
        if current.todo_progressed:
            return True
        if previous is None:
            return False
        if current.last_compile_summary != previous.last_compile_summary:
            return True
        if current.last_validation_summary != previous.last_validation_summary:
            return True
        return False

    def get_extensions_used(self, stage_descriptor: str) -> int:
        return self._extensions_used.get(stage_descriptor, 0)

    def reset_stage(self, stage_descriptor: str) -> None:
        self._last_summary_by_stage.pop(stage_descriptor, None)
        self._extensions_used.pop(stage_descriptor, None)
        return None


# Maps descriptor prefix → canonical policy key.
# Longer prefixes must appear before shorter ones so the more specific wins.
_STAGE_POLICY_PREFIX_MAP: tuple[tuple[str, str], ...] = (
    ("optim_global", "global_human_reference"),
    ("global_", "global_human_reference"),
    ("trace_expert", "trace_expert"),
    ("optimization", "trace_expert"),
    ("correctness_query_", "correctness_single_query"),  # correctness_query_3, _4 …
    ("correctness_q", "correctness_primary_query"),       # correctness_q1, correctness_q2
)


def get_policy_for_stage(stage_descriptor: str) -> StageRuntimePolicy | None:
    policy = STAGE_RUNTIME_POLICIES.get(stage_descriptor)
    if policy is not None:
        return policy
    for prefix, canonical_key in _STAGE_POLICY_PREFIX_MAP:
        if stage_descriptor.startswith(prefix):
            return STAGE_RUNTIME_POLICIES.get(canonical_key)
    return None


def get_default_turn_budget(stage_descriptor: str) -> int:
    return get_stage_turn_budget(stage_descriptor)
