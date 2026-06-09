from pathlib import Path
from types import SimpleNamespace

import pytest

from tpch_monetdb.conversations import optimization_conversation_tpch_monetdb as optimization_module
from tpch_monetdb.conversations.conversation import (
    VALIDATE_ON,
    VALIDATE_OUTPUT_STDOUT_OFF,
)
from tpch_monetdb.conversations.optimization_validation import CorrectnessCheckSummary
from tpch_monetdb.conversations.optimization_conversation_tpch_monetdb import (
    TpchMonetdbOptimizationConversation,
)


class _StopAfterPinning(Exception):
    pass


@pytest.mark.asyncio
async def test_optimization_run_uses_expanded_pinning_turn_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    queries_path = tmp_path / "queries.txt"
    queries_path.write_text("SELECT 1;\n", encoding="utf-8")

    conversation = TpchMonetdbOptimizationConversation.__new__(
        TpchMonetdbOptimizationConversation
    )
    conversation.query_ids = ["1"]
    conversation.bespoke_storage = True
    conversation.run_tool = SimpleNamespace(cwd=tmp_path)

    captured: dict[str, int | None] = {}
    observed_trace_modes: list[bool] = []

    async def fake_check_correctness(_query_ids, trace_mode: bool = False) -> bool:
        observed_trace_modes.append(trace_mode)
        return True

    async def fake_exec(
        _prompt: str,
        descriptor: str | None,
        max_turns: int | None = None,
        tool_profile: str | None = None,
    ) -> None:
        if descriptor == "Pinning":
            captured["max_turns"] = max_turns
            captured["tool_profile"] = tool_profile
            raise _StopAfterPinning()
        raise AssertionError(f"unexpected descriptor: {descriptor}")

    monkeypatch.setattr(conversation, "_check_correctness", fake_check_correctness)
    monkeypatch.setattr(conversation, "_exec", fake_exec)

    with pytest.raises(_StopAfterPinning):
        await conversation.run()

    assert captured["max_turns"] == optimization_module.PINNING_PROMPT_MAX_TURNS
    assert captured["max_turns"] == 600
    assert captured["tool_profile"] == "optimization_general"
    return None


@pytest.mark.asyncio
async def test_optimization_run_uses_declared_budget_for_add_timings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    queries_path = tmp_path / "queries.txt"
    queries_path.write_text("SELECT 1;\n", encoding="utf-8")

    conversation = TpchMonetdbOptimizationConversation.__new__(
        TpchMonetdbOptimizationConversation
    )
    conversation.query_ids = ["1", "2", "3"]
    conversation.bespoke_storage = True
    conversation.run_tool = SimpleNamespace(cwd=tmp_path)
    conversation.required_validation_sf_list = [1, 10, 100]
    conversation.benchmark_sf = 100
    conversation.best_rt_log = {}
    conversation.query_rt_log = {}
    conversation.revert_on_regression = True
    conversation.regression_tolerance = 0.05

    captured: dict[str, int | None] = {}
    observed_smoke_checks: list[tuple[tuple[str, ...], bool, tuple[int, ...]]] = []

    async def fake_check_correctness(_query_ids, trace_mode: bool = False) -> bool:
        return True

    async def fake_check_correctness_with_scale_factors(
        qids: list[str],
        trace_mode: bool,
        scale_factors: tuple[int, ...],
    ) -> bool:
        observed_smoke_checks.append((tuple(qids), trace_mode, scale_factors))
        return True

    def fake_collect_baselines() -> None:
        return None

    def fake_delete_result_csvs(_workspace: Path) -> None:
        return None

    async def fake_exec(
        _prompt: str,
        descriptor: str | None,
        max_turns: int | None = None,
        tool_profile: str | None = None,
        prompt_metadata=None,
    ) -> None:
        if descriptor == "Add Timings for Queries 1, 2, 3":
            captured["max_turns"] = max_turns
            captured["tool_profile"] = tool_profile
            captured["prompt_metadata"] = prompt_metadata
            raise _StopAfterPinning()
        return None

    monkeypatch.setattr(conversation, "_check_correctness", fake_check_correctness)
    monkeypatch.setattr(
        conversation,
        "_check_correctness_with_scale_factors",
        fake_check_correctness_with_scale_factors,
    )
    monkeypatch.setattr(conversation, "_exec", fake_exec)
    monkeypatch.setattr(conversation, "_collect_baselines_at_checkpoint", fake_collect_baselines)
    monkeypatch.setattr(conversation, "_delete_result_csvs", fake_delete_result_csvs)
    monkeypatch.setattr(
        optimization_module,
        "run_required_correctness_checks",
        lambda *_args, **_kwargs: CorrectnessCheckSummary(
            success=True,
            message="ok",
            metrics={"validation/correct": True},
            failed_scale_factor=None,
        ),
    )

    with pytest.raises(_StopAfterPinning):
        await conversation.run()

    assert captured["max_turns"] == 160
    assert captured["tool_profile"] == "optimization_instrumentation"
    assert captured["prompt_metadata"] == {"active_query_ids": ["1", "2", "3"]}
    return None


@pytest.mark.asyncio
async def test_optimization_exec_rejects_missing_explicit_max_turns() -> None:
    conversation = TpchMonetdbOptimizationConversation.__new__(
        TpchMonetdbOptimizationConversation
    )

    with pytest.raises(RuntimeError, match="missing explicit max_turns"):
        await conversation._exec(
            "some optimization prompt",
            "Custom Optimization Prompt",
            max_turns=None,
            tool_profile="optimization_general",
        )


@pytest.mark.asyncio
async def test_optimization_run_uses_explicit_budget_for_trace_to_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    queries_path = tmp_path / "queries.txt"
    queries_path.write_text("SELECT 1;\n", encoding="utf-8")

    conversation = TpchMonetdbOptimizationConversation.__new__(
        TpchMonetdbOptimizationConversation
    )
    conversation.query_ids = ["1"]
    conversation.bespoke_storage = True
    conversation.run_tool = SimpleNamespace(cwd=tmp_path)
    conversation.required_validation_sf_list = [1, 10, 100]
    conversation.benchmark_sf = 100
    conversation.best_rt_log = {}
    conversation.query_rt_log = {}
    conversation.revert_on_regression = True
    conversation.regression_tolerance = 0.05

    captured: dict[str, int | None] = {}
    observed_smoke_checks: list[tuple[tuple[str, ...], bool, tuple[int, ...]]] = []

    async def fake_check_correctness(_query_ids, trace_mode: bool = False) -> bool:
        return True

    async def fake_check_correctness_with_scale_factors(
        qids: list[str],
        trace_mode: bool,
        scale_factors: tuple[int, ...],
    ) -> bool:
        observed_smoke_checks.append((tuple(qids), trace_mode, scale_factors))
        return True

    def fake_collect_baselines() -> None:
        return None

    def fake_delete_result_csvs(_workspace: Path) -> None:
        return None

    async def fake_exec(
        _prompt: str,
        descriptor: str | None,
        max_turns: int | None = None,
        tool_profile: str | None = None,
        prompt_metadata=None,
    ) -> None:
        if descriptor == "Trace->File":
            captured["max_turns"] = max_turns
            captured["tool_profile"] = tool_profile
            captured["prompt_metadata"] = prompt_metadata
            raise _StopAfterPinning()
        return None

    monkeypatch.setattr(conversation, "_check_correctness", fake_check_correctness)
    monkeypatch.setattr(
        conversation,
        "_check_correctness_with_scale_factors",
        fake_check_correctness_with_scale_factors,
    )
    monkeypatch.setattr(conversation, "_exec", fake_exec)
    monkeypatch.setattr(conversation, "_collect_baselines_at_checkpoint", fake_collect_baselines)
    monkeypatch.setattr(conversation, "_delete_result_csvs", fake_delete_result_csvs)
    monkeypatch.setattr(
        optimization_module,
        "run_required_correctness_checks",
        lambda *_args, **_kwargs: CorrectnessCheckSummary(
            success=True,
            message="ok",
            metrics={"validation/correct": True},
            failed_scale_factor=None,
        ),
    )

    with pytest.raises(_StopAfterPinning):
        await conversation.run()

    assert captured["max_turns"] == 160
    assert captured["tool_profile"] == "optimization_instrumentation"
    assert captured["prompt_metadata"] == {"active_query_ids": ["1"]}
    assert observed_smoke_checks == []
    return None


@pytest.mark.asyncio
async def test_optimization_run_disables_validation_stdout_before_trace_batches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    queries_path = tmp_path / "queries.txt"
    queries_path.write_text("SELECT 1;\n", encoding="utf-8")

    conversation = TpchMonetdbOptimizationConversation.__new__(
        TpchMonetdbOptimizationConversation
    )
    conversation.query_ids = ["1"]
    conversation.bespoke_storage = True
    conversation.run_tool = SimpleNamespace(cwd=tmp_path)
    conversation.required_validation_sf_list = [1]
    conversation.benchmark_sf = 100
    conversation.best_rt_log = {}
    conversation.query_rt_log = {}
    conversation.revert_on_regression = True
    conversation.regression_tolerance = 0.05

    control_prompts: list[str] = []

    async def fake_check_correctness(_query_ids, trace_mode: bool = False) -> bool:
        return True

    async def fake_exec(
        prompt: str,
        descriptor: str | None,
        max_turns: int | None = None,
        tool_profile: str | None = None,
        prompt_metadata=None,
    ) -> None:
        del max_turns, tool_profile, prompt_metadata
        if descriptor is None and prompt in (VALIDATE_ON, VALIDATE_OUTPUT_STDOUT_OFF):
            control_prompts.append(prompt)
            return None
        if descriptor == "Trace->File":
            raise _StopAfterPinning()
        return None

    monkeypatch.setattr(conversation, "_check_correctness", fake_check_correctness)
    monkeypatch.setattr(conversation, "_exec", fake_exec)
    monkeypatch.setattr(conversation, "_collect_baselines_at_checkpoint", lambda: None)
    monkeypatch.setattr(conversation, "_delete_result_csvs", lambda _workspace: None)
    monkeypatch.setattr(
        optimization_module,
        "run_required_correctness_checks",
        lambda *_args, **_kwargs: CorrectnessCheckSummary(
            success=True,
            message="ok",
            metrics={"validation/correct": True},
            failed_scale_factor=None,
        ),
    )

    with pytest.raises(_StopAfterPinning):
        await conversation.run()

    assert control_prompts == [VALIDATE_ON, VALIDATE_OUTPUT_STDOUT_OFF]
    return None
