"""Agent-facing text registry for prompt and runtime message assets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from string import Template
from typing import Mapping


PROMPTS_ROOT = Path(__file__).parent / "prompts"


@dataclass(frozen=True)
class AgentTextAsset:
    """Registered agent-facing text asset with provenance metadata."""

    asset_id: str
    relative_path: str
    owner: str
    source_type: str

    @property
    def path(self) -> Path:
        return PROMPTS_ROOT / self.relative_path


_RUNTIME_STAGE_HINT_TEXT_ASSETS: tuple[tuple[str, str], ...] = (
    ("runtime.stage_hint_label_editable_files", "stage_hint_label_editable_files.txt"),
    ("runtime.stage_hint_label_creatable_files", "stage_hint_label_creatable_files.txt"),
    ("runtime.stage_hint_label_writable_files", "stage_hint_label_writable_files.txt"),
    ("runtime.stage_hint_condition_todo_plan", "stage_hint_condition_todo_plan.txt"),
    ("runtime.stage_hint_condition_storage_plan", "stage_hint_condition_storage_plan.txt"),
    ("runtime.stage_hint_condition_finish_skeleton", "stage_hint_condition_finish_skeleton.txt"),
    (
        "runtime.stage_hint_condition_implement_queries_writeonly",
        "stage_hint_condition_implement_queries_writeonly.txt",
    ),
    (
        "runtime.stage_hint_condition_correctness_queries_writeonly",
        "stage_hint_condition_correctness_queries_writeonly.txt",
    ),
    (
        "runtime.stage_hint_condition_correctness_foundation",
        "stage_hint_condition_correctness_foundation.txt",
    ),
)


_RUNTIME_POLICY_ASSETS: tuple[tuple[str, str], ...] = (
    ("runtime.policy_no_write_tools_available", "policy_no_write_tools_available.txt"),
    ("runtime.policy_no_editable_files", "policy_no_editable_files.txt"),
    ("runtime.policy_no_hard_limit", "policy_no_hard_limit.txt"),
    ("runtime.policy_creatable_scope_suffix", "policy_creatable_scope_suffix.txt"),
    ("runtime.policy_tool_not_allowed_message", "policy_tool_not_allowed_message.txt"),
    ("runtime.policy_tool_not_allowed_next_action", "policy_tool_not_allowed_next_action.txt"),
    ("runtime.policy_observation_guidance_todo_plan", "policy_observation_guidance_todo_plan.txt"),
    ("runtime.policy_observation_guidance_storage_plan", "policy_observation_guidance_storage_plan.txt"),
    ("runtime.policy_observation_guidance_finish_skeleton", "policy_observation_guidance_finish_skeleton.txt"),
    (
        "runtime.policy_observation_guidance_implement_queries_writeonly",
        "policy_observation_guidance_implement_queries_writeonly.txt",
    ),
    (
        "runtime.policy_observation_guidance_correctness_queries_writeonly",
        "policy_observation_guidance_correctness_queries_writeonly.txt",
    ),
    (
        "runtime.policy_observation_guidance_correctness_foundation",
        "policy_observation_guidance_correctness_foundation.txt",
    ),
    ("runtime.policy_observation_limit_message", "policy_observation_limit_message.txt"),
    ("runtime.policy_observation_limit_next_action", "policy_observation_limit_next_action.txt"),
    ("runtime.policy_must_write_first_message", "policy_must_write_first_message.txt"),
    ("runtime.policy_must_write_first_next_action", "policy_must_write_first_next_action.txt"),
    (
        "runtime.policy_run_query_batch_scope_denied_message",
        "policy_run_query_batch_scope_denied_message.txt",
    ),
    (
        "runtime.policy_run_query_batch_scope_denied_next_action",
        "policy_run_query_batch_scope_denied_next_action.txt",
    ),
    ("runtime.policy_run_query_scope_denied_message", "policy_run_query_scope_denied_message.txt"),
    (
        "runtime.policy_run_query_scope_denied_next_action",
        "policy_run_query_scope_denied_next_action.txt",
    ),
    ("runtime.policy_single_file_rebuild_default_reason", "policy_single_file_rebuild_default_reason.txt"),
    ("runtime.policy_single_file_rebuild_next_action", "policy_single_file_rebuild_next_action.txt"),
    (
        "runtime.policy_query_file_structural_corruption_reason",
        "policy_query_file_structural_corruption_reason.txt",
    ),
    ("runtime.policy_patch_op_denied_message", "policy_patch_op_denied_message.txt"),
    ("runtime.policy_patch_op_denied_next_action", "policy_patch_op_denied_next_action.txt"),
    ("runtime.policy_patch_create_exists_message", "policy_patch_create_exists_message.txt"),
    ("runtime.policy_patch_create_exists_next_action", "policy_patch_create_exists_next_action.txt"),
    ("runtime.policy_stalled_execution_message", "policy_stalled_execution_message.txt"),
    ("runtime.policy_stalled_execution_next_action", "policy_stalled_execution_next_action.txt"),
    ("runtime.policy_path_outside_workspace_message", "policy_path_outside_workspace_message.txt"),
    ("runtime.policy_path_outside_workspace_next_action", "policy_path_outside_workspace_next_action.txt"),
    ("runtime.policy_read_denied_message", "policy_read_denied_message.txt"),
    ("runtime.policy_read_denied_next_action", "policy_read_denied_next_action.txt"),
    ("runtime.policy_edit_denied_message", "policy_edit_denied_message.txt"),
    ("runtime.policy_edit_denied_next_action", "policy_edit_denied_next_action.txt"),
    ("runtime.policy_create_denied_message", "policy_create_denied_message.txt"),
    ("runtime.policy_create_denied_next_action", "policy_create_denied_next_action.txt"),
    ("runtime.policy_write_denied_message", "policy_write_denied_message.txt"),
    ("runtime.policy_write_denied_next_action", "policy_write_denied_next_action.txt"),
    ("runtime.policy_path_not_found_message", "policy_path_not_found_message.txt"),
    ("runtime.policy_path_not_found_next_action", "policy_path_not_found_next_action.txt"),
    (
        "runtime.policy_instrumentation_scope_denied_message",
        "policy_instrumentation_scope_denied_message.txt",
    ),
    (
        "runtime.policy_instrumentation_scope_denied_next_action",
        "policy_instrumentation_scope_denied_next_action.txt",
    ),
    ("runtime.execution_compile_failed_message", "execution_compile_failed_message.txt"),
    ("runtime.execution_compile_failed_next_action", "execution_compile_failed_next_action.txt"),
    ("runtime.execution_run_failed_message", "execution_run_failed_message.txt"),
    ("runtime.execution_run_failed_next_action", "execution_run_failed_next_action.txt"),
)


_ASSETS: dict[str, AgentTextAsset] = {
    "compaction.system": AgentTextAsset(
        asset_id="compaction.system",
        relative_path="compaction/system/compact_system_prompt.txt",
        owner="compaction",
        source_type="system_prompt",
    ),
    "runtime.tool_correction": AgentTextAsset(
        asset_id="runtime.tool_correction",
        relative_path="shared/runtime/tool_correction.txt",
        owner="runtime",
        source_type="tool_correction",
    ),
    "runtime.tool_correction_continue": AgentTextAsset(
        asset_id="runtime.tool_correction_continue",
        relative_path="shared/runtime/tool_correction_continue.txt",
        owner="runtime",
        source_type="tool_correction",
    ),
    "runtime.base_agent_instructions": AgentTextAsset(
        asset_id="runtime.base_agent_instructions",
        relative_path="shared/runtime/base_agent_instructions.txt",
        owner="runtime",
        source_type="base_instructions",
    ),
    "runtime.litellm_tool_guidance": AgentTextAsset(
        asset_id="runtime.litellm_tool_guidance",
        relative_path="shared/runtime/litellm_tool_guidance.txt",
        owner="runtime",
        source_type="base_instructions",
    ),
    "runtime.stage_hint.header": AgentTextAsset(
        asset_id="runtime.stage_hint.header",
        relative_path="shared/runtime/stage_hint_header.txt",
        owner="runtime",
        source_type="stage_hint",
    ),
    "runtime.stage_hint.scope": AgentTextAsset(
        asset_id="runtime.stage_hint.scope",
        relative_path="shared/runtime/stage_hint_scope.txt",
        owner="runtime",
        source_type="stage_hint",
    ),
    "runtime.stage_hint.stop_condition": AgentTextAsset(
        asset_id="runtime.stage_hint.stop_condition",
        relative_path="shared/runtime/stage_hint_stop_condition.txt",
        owner="runtime",
        source_type="stage_hint",
    ),
    "runtime.stage_hint.compile_run_unavailable": AgentTextAsset(
        asset_id="runtime.stage_hint.compile_run_unavailable",
        relative_path="shared/runtime/stage_hint_compile_run_unavailable.txt",
        owner="runtime",
        source_type="stage_hint",
    ),
    "runtime.stage_hint.active_query_batch": AgentTextAsset(
        asset_id="runtime.stage_hint.active_query_batch",
        relative_path="shared/runtime/stage_hint_active_query_batch.txt",
        owner="runtime",
        source_type="stage_hint",
    ),
    "runtime.stage_hint.active_unit": AgentTextAsset(
        asset_id="runtime.stage_hint.active_unit",
        relative_path="shared/runtime/stage_hint_active_unit.txt",
        owner="runtime",
        source_type="stage_hint",
    ),
    "runtime.stage_hint.active_unit_kind": AgentTextAsset(
        asset_id="runtime.stage_hint.active_unit_kind",
        relative_path="shared/runtime/stage_hint_active_unit_kind.txt",
        owner="runtime",
        source_type="stage_hint",
    ),
    "runtime.stage_hint.active_unit_queries": AgentTextAsset(
        asset_id="runtime.stage_hint.active_unit_queries",
        relative_path="shared/runtime/stage_hint_active_unit_queries.txt",
        owner="runtime",
        source_type="stage_hint",
    ),
    "runtime.stage_hint.single_file_rebuild": AgentTextAsset(
        asset_id="runtime.stage_hint.single_file_rebuild",
        relative_path="shared/runtime/stage_hint_single_file_rebuild.txt",
        owner="runtime",
        source_type="stage_hint",
    ),
    "runtime.stage_hint.write_tools": AgentTextAsset(
        asset_id="runtime.stage_hint.write_tools",
        relative_path="shared/runtime/stage_hint_write_tools.txt",
        owner="runtime",
        source_type="stage_hint",
    ),
    "optimization.repair.precheck_correctness": AgentTextAsset(
        asset_id="optimization.repair.precheck_correctness",
        relative_path="optimization/repair/precheck_correctness.txt",
        owner="optimization",
        source_type="repair_prompt",
    ),
    "optimization.repair.rollback_correctness": AgentTextAsset(
        asset_id="optimization.repair.rollback_correctness",
        relative_path="optimization/repair/rollback_correctness.txt",
        owner="optimization",
        source_type="repair_prompt",
    ),
    "optimization.instrumentation.trace_to_file": AgentTextAsset(
        asset_id="optimization.instrumentation.trace_to_file",
        relative_path="optimization/instrumentation/trace_to_file.txt",
        owner="optimization",
        source_type="instrumentation_prompt",
    ),
    "scripted.remediation.file_contract": AgentTextAsset(
        asset_id="scripted.remediation.file_contract",
        relative_path="scripted/remediation/file_contract.txt",
        owner="scripted",
        source_type="remediation_prompt",
    ),
    "scripted.remediation.validation_rerun": AgentTextAsset(
        asset_id="scripted.remediation.validation_rerun",
        relative_path="scripted/remediation/validation_rerun.txt",
        owner="scripted",
        source_type="remediation_prompt",
    ),
    "scripted.remediation.storage_plan_contract": AgentTextAsset(
        asset_id="scripted.remediation.storage_plan_contract",
        relative_path="scripted/remediation/storage_plan_contract.txt",
        owner="scripted",
        source_type="remediation_prompt",
    ),
}

_ASSETS.update(
    {
        asset_id: AgentTextAsset(
            asset_id=asset_id,
            relative_path=f"shared/runtime/{filename}",
            owner="runtime",
            source_type="stage_hint",
        )
        for asset_id, filename in _RUNTIME_STAGE_HINT_TEXT_ASSETS
    }
)

_ASSETS.update(
    {
        asset_id: AgentTextAsset(
            asset_id=asset_id,
            relative_path=f"shared/runtime/{filename}",
            owner="runtime",
            source_type="stage_policy",
        )
        for asset_id, filename in _RUNTIME_POLICY_ASSETS
    }
)


def list_agent_text_assets() -> tuple[AgentTextAsset, ...]:
    """Return registered agent-facing text assets in stable order."""
    return tuple(_ASSETS[key] for key in sorted(_ASSETS))


def get_agent_text_asset(asset_id: str) -> AgentTextAsset:
    """Return one registered agent-facing text asset by id."""
    try:
        return _ASSETS[asset_id]
    except KeyError as exc:
        raise KeyError(f"Unknown agent-facing text asset: {asset_id}") from exc


def load_agent_text_asset(asset_id: str) -> str:
    """Load a registered agent-facing text asset and fail on missing/empty files."""
    asset = get_agent_text_asset(asset_id)
    if not asset.path.exists():
        raise FileNotFoundError(f"Missing agent-facing text asset: {asset.path}")
    text = asset.path.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError(f"Agent-facing text asset is empty: {asset.path}")
    return text


def render_agent_text_asset(
    asset_id: str,
    variables: Mapping[str, object] | None = None,
) -> str:
    """Render a registered agent-facing text asset with strict placeholders."""
    template = Template(load_agent_text_asset(asset_id))
    mapping = {} if variables is None else {key: str(value) for key, value in variables.items()}
    try:
        return template.substitute(mapping)
    except KeyError as exc:
        raise ValueError(
            f"Missing placeholder for agent-facing text asset {asset_id}: {exc.args[0]}"
        ) from exc
