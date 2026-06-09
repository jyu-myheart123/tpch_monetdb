from pathlib import Path

import pytest

from tpch_monetdb.conversations.scripted_prompts_gen import (
    get_scripted_prompts_root,
    list_scripted_prompt_assets,
    load_scripted_prompt_asset,
    render_scripted_prompt_asset,
)
from tpch_monetdb.conversations.agent_text_registry import (
    list_agent_text_assets,
    load_agent_text_asset,
    render_agent_text_asset,
)


def test_scripted_prompt_asset_inventory_has_storage_and_base_impl_assets() -> None:
    """The scripted prompt catalog must expose the seeded storage/base assets."""
    asset_names = {
        path.relative_to(get_scripted_prompts_root()).as_posix()
        for path in list_scripted_prompt_assets()
    }
    assert "storage_plan/todo_plan.txt" in asset_names
    assert "storage_plan/storage_plan.txt" in asset_names
    assert "storage_plan/prev_feedback_section.txt" in asset_names
    assert "base_impl/finish_skeleton.txt" in asset_names
    return None


def test_scripted_prompt_asset_loader_reads_nonempty_text() -> None:
    """Seeded scripted prompt assets must be readable and non-empty."""
    text = load_scripted_prompt_asset("storage_plan", "todo_plan.txt")
    assert "sealed manifest" in text
    return None


def test_scripted_prompt_assets_frontload_vectorization_readiness() -> None:
    """Scripted early-stage assets must treat vectorization-readiness as a design input."""
    storage_plan = load_scripted_prompt_asset("storage_plan", "storage_plan.txt")
    todo_plan = load_scripted_prompt_asset("storage_plan", "todo_plan.txt")
    finish_skeleton = load_scripted_prompt_asset("base_impl", "finish_skeleton.txt")
    assert "Vectorization-readiness" in storage_plan
    assert "Treat vectorization-readiness as a first-class layout property" in storage_plan
    assert "Turn vectorization-readiness from the storage plan into explicit implementation" in todo_plan
    assert "preserve the storage-plan" in finish_skeleton
    assert "compiler-friendly" in finish_skeleton
    return None


def test_optimization_prompt_assets_require_objective_routing() -> None:
    """Optimization prompts must make objective and route constraints explicit."""
    trace_prompt = (
        Path(__file__).resolve().parents[1]
        / "conversations"
        / "prompts"
        / "optimization"
        / "stages"
        / "tpch_monetdb_optim_trace_expert.txt"
    ).read_text(encoding="utf-8")
    global_prompt = (
        Path(__file__).resolve().parents[1]
        / "conversations"
        / "prompts"
        / "optimization"
        / "stages"
        / "tpch_monetdb_optim_global_human_reference.txt"
    ).read_text(encoding="utf-8")
    assert "Objective routing is mandatory" in trace_prompt
    assert "storage_plan_contract.json" in trace_prompt
    assert "PMU_REQUIRED_BUT_MISSING" in trace_prompt
    assert "Vectorization-first repair for critical queries" in trace_prompt
    assert "requires_vectorization=true" in trace_prompt
    assert "contiguous-row or contiguous-column SIMD" in trace_prompt
    assert "prompt claims or token matches do not" in trace_prompt
    assert "STORAGE_PLAN_REFINEMENT_REQUIRED" in trace_prompt
    assert "workload_objective.json" in global_prompt
    assert "storage_plan_alignment.json" in global_prompt
    assert "Do not use a fixed direction queue" in global_prompt
    assert "evidence-backed hypothesis" in global_prompt
    assert "posterior labels" in global_prompt
    assert "storage layout alignment with storage_plan.txt" not in global_prompt
    assert "objective failures" in global_prompt
    return None


def test_scripted_prompt_asset_loader_fails_for_missing_asset() -> None:
    """Missing assets must fail fast instead of falling back silently."""
    with pytest.raises(FileNotFoundError):
        load_scripted_prompt_asset("missing", "asset.txt")
    return None


def test_scripted_prompt_renderer_requires_all_placeholders() -> None:
    """Template rendering must fail if a placeholder value is missing."""
    with pytest.raises(ValueError, match="Missing placeholder"):
        render_scripted_prompt_asset(
            "base_impl",
            "finish_skeleton.txt",
            variables={"qid": "3"},
        )
    return None


def test_agent_text_registry_covers_runtime_repair_and_compaction_assets() -> None:
    """Agent-facing behavior text must be registered, not hidden in code."""
    asset_ids = {asset.asset_id for asset in list_agent_text_assets()}
    assert "compaction.system" in asset_ids
    assert "runtime.tool_correction" in asset_ids
    assert "runtime.tool_correction_continue" in asset_ids
    assert "runtime.base_agent_instructions" in asset_ids
    assert "runtime.litellm_tool_guidance" in asset_ids
    assert "optimization.repair.precheck_correctness" in asset_ids
    assert "optimization.repair.rollback_correctness" in asset_ids
    assert "optimization.instrumentation.trace_to_file" in asset_ids
    assert "scripted.remediation.file_contract" in asset_ids
    assert "scripted.remediation.validation_rerun" in asset_ids
    assert "runtime.stage_hint.header" in asset_ids
    assert "runtime.stage_hint.stop_condition" in asset_ids
    assert "runtime.stage_hint.write_tools" in asset_ids
    assert "runtime.stage_hint_label_editable_files" in asset_ids
    assert "runtime.stage_hint_condition_todo_plan" in asset_ids
    assert "runtime.stage_hint_condition_correctness_queries_writeonly" in asset_ids
    assert "runtime.stage_hint_condition_correctness_foundation" in asset_ids
    assert "runtime.policy_no_write_tools_available" in asset_ids
    assert "runtime.policy_creatable_scope_suffix" in asset_ids
    assert "runtime.policy_observation_guidance_correctness_foundation" in asset_ids
    assert "runtime.policy_observation_limit_message" in asset_ids
    assert "runtime.policy_must_write_first_message" in asset_ids
    assert "runtime.execution_compile_failed_next_action" in asset_ids
    assert "### 1. Current Goal" in load_agent_text_asset("compaction.system")
    rendered = render_agent_text_asset(
        "runtime.tool_correction",
        {"error_message": "bad tool", "available_tools": "read_file"},
    )
    assert "bad tool" in rendered
    assert "read_file" in rendered
    base = render_agent_text_asset(
        "runtime.base_agent_instructions",
        {"workspace_path": "/tmp/work", "litellm_tool_guidance": ""},
    )
    assert "/tmp/work" in base
    policy = render_agent_text_asset(
        "runtime.policy_observation_limit_message",
        {
            "current_count": 3,
            "profile_name": "todo_plan",
            "hard_limit_text": "6",
            "progress_tools": "write_file",
            "editable_scope": "TODO.md",
            "create_scope": "",
            "stage_guidance": "",
        },
    )
    assert "todo_plan" in policy
    assert "write_file" in policy
    return None


def test_agent_text_registry_assets_are_nonempty_and_under_prompts() -> None:
    """Registered agent-facing text assets must live under conversations/prompts."""
    for asset in list_agent_text_assets():
        assert "prompts" in asset.path.parts
        assert load_agent_text_asset(asset.asset_id).strip()
    return None
