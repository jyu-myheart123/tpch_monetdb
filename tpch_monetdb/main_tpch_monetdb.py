import argparse
import asyncio
import json
import inspect
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from tpch_monetdb.bootstrap_env import bootstrap_runtime_env

bootstrap_runtime_env()

from agents import Agent, ModelSettings, Runner, trace
from agents.exceptions import ModelBehaviorError
from agents.extensions.memory import AdvancedSQLiteSession
from agents.models.reasoning_content_replay import (
    ShouldReplayReasoningContent,
    default_should_replay_reasoning_content,
)
from agents.tracing import set_tracing_disabled
from dotenv import load_dotenv
from openai.types.shared.reasoning import Reasoning

import wandb
from tpch_monetdb.conversations.conversation import (
    COMPACTION_MARKER,
    VALIDATE_OFF,
    VALIDATE_ON,
    VALIDATE_OUTPUT_STDOUT_OFF,
    VALIDATE_OUTPUT_STDOUT_ON,
)
from tpch_monetdb.conversations.agent_text_registry import render_agent_text_asset
from tpch_monetdb.conversations.scripted_conversation import ScriptedConversation
from tpch_monetdb.conversations.optimization_validation import (
    required_validation_scale_factors,
    run_required_correctness_checks,
)
from tpch_monetdb.dataset.dataset_tables_dict import get_dataset_name
from tpch_monetdb.dataset.query_gen_factory import get_placeholders_fn, get_query_gen
from tpch_monetdb.llm_cache.auto_compact import AutoCompactManager, MAX_CONSECUTIVE_FAILURES
from tpch_monetdb.llm_cache.artifact_ledger import ArtifactLedger
from tpch_monetdb.llm_cache.git_snapshotter import GitSnapshotter
from tpch_monetdb.llm_cache.logger import setup_logging
from tpch_monetdb.llm_cache.stage_memory import q1_q9_obligations_for_query_ids
from tpch_monetdb.llm_cache.utils import ask_yes_no
from tpch_monetdb.config import (
    get_default_verify_scale_factors,
    resolve_active_verify_scale_factors,
    resolve_workflow_scale_factors,
)
from tpch_monetdb.tools.tool_factory import build_tools
from tpch_monetdb.tools.error_envelope import ErrorEnvelope
from tpch_monetdb.tools.tpch import copy_template_to, make_run_tool
from tpch_monetdb.tools.tpch.hardware_counters import build_hardware_counter_preflight
from tpch_monetdb.oracle.tpch_runtime_validator import TpchRuntimeValidator
from tpch_monetdb.runtime_stage_policy import get_policy_for_stage
from tpch_monetdb.runtime_workspace import (
    _prepare_runtime_workspace,
    _snapshot_final_workspace_state,
    build_runtime_snapshotter,
    resolve_runtime_workspace_path,
)
from tpch_monetdb.utils.agent_rules import (
    DEFAULT_RULE_TOKEN_BUDGET,
    RuleAssembly,
    RuleScope,
    load_agent_rules,
    log_rule_assembly,
)
from tpch_monetdb.utils.cli_config import add_common_args
from tpch_monetdb.utils.general_utils import write_query_and_args_file
from tpch_monetdb.utils.model_aliases import is_deepseek_model
from tpch_monetdb.utils.model_setup import setup_model_config
from tpch_monetdb.utils.outer_loop_supervisor import classify_model_failure, should_reactive_compact
from tpch_monetdb.utils.scripted_summary import persist_successful_scripted_run
from tpch_monetdb.utils.storage_plan_summary import persist_successful_storage_plan_run
from tpch_monetdb.utils.control_artifacts import (
    build_todo_reconciliation,
    collect_control_artifact_hashes,
    write_control_artifact_audit_copy,
)
from tpch_monetdb.utils.design_evidence import build_tpch_design_evidence
from tpch_monetdb.utils.large_data_objectives import (
    WORKLOAD_OBJECTIVE_FILE,
    load_json_contract,
    write_data_law_contract,
    write_workload_objective,
)
from tpch_monetdb.utils.pipeline_evidence import (
    PipelineEvidenceStage,
    build_pipeline_evidence_ledger,
    require_base_impl_promotable,
)
from tpch_monetdb.utils.pipeline_contracts import PipelineContractError
from tpch_monetdb.utils.pipeline_invariants import require_resume_snapshot_fields
from tpch_monetdb.utils.stage_summary import render_stage_summary
from tpch_monetdb.utils.snapshot_utils import load_storage_plan_from_snapshot
from tpch_monetdb.utils.token_usage import get_tokens_context_and_dollar_info
from tpch_monetdb.utils.truncate_model_log import truncate_model_final_output
from tpch_monetdb.utils.wandb_init import init_wandb_run_with_retry
from tpch_monetdb.utils.wandb_runtime_guard import (
    DEFAULT_WANDB_FINISH_RETRIES,
    DEFAULT_WANDB_FINISH_TIMEOUT_S,
    DEFAULT_WANDB_INIT_TIMEOUT_S,
    DEFAULT_WANDB_UPLOAD_TIMEOUT_S,
    finish_wandb_with_guard,
    run_callable_with_timeout,
    upload_workspace_code_with_guard,
)
from tpch_monetdb.utils.wandb_stats_logging import WandbRunHook
from tpch_monetdb.utils.weave_cache import configure_weave_cache_dirs

logger = logging.getLogger(__name__)

LITELLM_MODEL_ALLOWED_OPENAI_PARAMS: dict[str, tuple[str, ...]] = {
    "openai/gpt-5.5": ("tool_choice",),
}
DEFAULT_REASONING_EFFORT = "xhigh"
DEEPSEEK_DEFAULT_REASONING_EFFORT = "xhigh"


def _normalize_deepseek_reasoning_effort(effort: str | None) -> str:
    """将 harness effort 归一化为 DeepSeek OpenAI Chat Completions 支持的值."""
    raise NotImplementedError("TODO(student): map harness reasoning effort to DeepSeek values")


def _normalize_agents_reasoning_effort(
    effort: str | None,
    model_config: Any,
) -> str | None:
    """将 provider-only effort 映射为 OpenAI Agents SDK 支持的枚举值."""
    if effort == "max" and is_deepseek_model(
        getattr(model_config, "accounting_model_name", "") or ""
    ):
        return "xhigh"
    return effort


def _resolve_reasoning(
    args: argparse.Namespace,
    model_config: Any,
) -> Reasoning | None:
    """解析 reasoning effort.

    优先级：用户显式 CLI 传入 > 模型族默认 > 全局默认。
    deepseek-v4 家族单独维护默认值，与全局 DEFAULT_REASONING_EFFORT 解耦，
    便于未来调整全局默认时不波及 deepseek 的预期行为。
    """
    effort = getattr(args, "reasoning_effort", None)
    if effort == "none":
        return None
    if effort is None:
        accounting_name = getattr(model_config, "accounting_model_name", "") or ""
        if is_deepseek_model(accounting_name):
            effort = DEEPSEEK_DEFAULT_REASONING_EFFORT
        else:
            effort = DEFAULT_REASONING_EFFORT
    effort = _normalize_agents_reasoning_effort(effort, model_config)
    return Reasoning(effort=effort)


def _build_model_settings(
    args: argparse.Namespace,
    model_config: Any,
) -> ModelSettings:
    """组装 ModelSettings；phase10 起由 tool_parallelism 决定并发开关."""
    from tpch_monetdb.tools.tool_parallelism import resolve_parallelism_config

    reasoning = _resolve_reasoning(args, model_config)
    model_is_deepseek = is_deepseek_model(
        getattr(model_config, "accounting_model_name", "") or ""
    )
    extra_args: dict[str, Any] | None = None
    if model_config.use_litellm and reasoning is not None and not model_is_deepseek:
        extra_args = {"allowed_openai_params": ["reasoning_effort"]}

    extra_body: dict[str, Any] | None = None
    reasoning_for_model = reasoning
    if model_is_deepseek and reasoning is not None and reasoning.effort != "none":
        raise NotImplementedError("TODO(student): inject DeepSeek thinking extra_body")
    elif model_is_deepseek and reasoning is None:
        raise NotImplementedError("TODO(student): disable DeepSeek thinking when effort is none")

    model_specific_allowed = LITELLM_MODEL_ALLOWED_OPENAI_PARAMS.get(
        getattr(model_config, "model_name", "") or ""
    )
    if model_config.use_litellm and model_specific_allowed:
        if extra_args is None:
            extra_args = {}
        extra_args = dict(extra_args)
        allowed = list(extra_args.get("allowed_openai_params") or [])
        allowed.extend(
            param for param in model_specific_allowed if param not in allowed
        )
        extra_args["allowed_openai_params"] = allowed

    parallelism = resolve_parallelism_config(
        getattr(args, "tool_parallelism", None),
        use_litellm=model_config.use_litellm,
        model_name=getattr(model_config, "model_name", None) or getattr(args, "model", None),
    )
    return ModelSettings(
        tool_choice="auto",
        include_usage=model_config.use_litellm,
        parallel_tool_calls=parallelism.parallel_tool_calls,
        reasoning=reasoning_for_model,
        extra_args=extra_args,
        extra_body=extra_body,
    )


def _build_stage_model_settings(
    args: argparse.Namespace,
    model_config: Any,
    profile_key: str,
) -> ModelSettings:
    """Build model settings for one runtime stage without changing user effort."""
    del profile_key
    return _build_model_settings(args, model_config)


def _configure_litellm_cost_map_for_reasoning(
    model_config: Any,
    reasoning: Reasoning | None,
) -> None:
    if not model_config.use_litellm:
        return None
    from tpch_monetdb.llm_cache.litellm_model_costs import (
        force_litellm_local_model_cost_map,
        register_tpch_monetdb_litellm_model_costs,
        validate_gpt55_xhigh_model_cost,
    )

    force_litellm_local_model_cost_map()
    register_tpch_monetdb_litellm_model_costs()
    if getattr(model_config, "model_name", "") != "openai/gpt-5.5":
        return None
    if getattr(reasoning, "effort", None) != "xhigh":
        return None
    validate_gpt55_xhigh_model_cost()
    return None


def _log_stream_event(event: Any) -> None:
    event_type = getattr(event, "type", "")
    if event_type == "raw_response_event":
        data = getattr(event, "data", None)
        data_type = getattr(data, "type", "")
        if data_type == "response.output_text.delta":
            delta = getattr(data, "delta", "")
            if delta:
                logger.info("[stream.delta] %s", delta)
        elif data_type in {"response.completed", "response.incomplete", "response.failed"}:
            logger.info("[stream.%s]", data_type)
        return None
    if event_type == "run_item_stream_event":
        name = getattr(event, "name", "")
        if name in {"tool_called", "tool_output", "message_output_created"}:
            logger.info("[stream.%s]", name)
    return None


async def _run_agent_turn(
    agent: Any,
    *,
    input: Any,
    session: Any,
    max_turns: int | None,
    hooks: Any,
    stream_llm: bool,
) -> Any:
    if not stream_llm:
        return await Runner.run(
            agent,
            input=input,
            session=session,
            max_turns=max_turns,
            hooks=hooks,
        )
    result = Runner.run_streamed(
        agent,
        input=input,
        session=session,
        max_turns=max_turns,
        hooks=hooks,
    )
    async for event in result.stream_events():
        _log_stream_event(event)
    return result


def _context_too_large_envelope(
    *,
    prompt_index: int,
    descriptor: str,
    detail: str,
) -> ErrorEnvelope:
    return ErrorEnvelope(
        code="CONTEXT_TOO_LARGE",
        category="model_context",
        stage=descriptor,
        message=(
            f"Prompt {prompt_index} ({descriptor}) exceeded the model/provider "
            f"context or request-size limit. Detail: {detail}"
        ),
        recoverable=False,
        recommended_next_action=(
            "Force compaction or reduce instrumentation feedback before resuming."
        ),
    )


def _run_final_correctness_gate(
    *,
    query_validator: Any | None,
    run_tool: Any,
    active_verify_sf_list: list[int],
    max_scale_factor: int,
    query_list: list[str],
    only_from_cache: bool,
) -> None:
    """Run a final scripted correctness gate before persisting success."""
    if only_from_cache:
        raise RuntimeError(
            str(
                ErrorEnvelope(
                    code="FINAL_CORRECTNESS_GATE_FAILED",
                    category="correctness",
                    stage="scripted_finalize",
                    message=(
                        "Final correctness gate requires fresh validation and is incompatible "
                        "with --only_from_cache."
                    ),
                    recoverable=False,
                    recommended_next_action=(
                        "Rerun scripted handoff without --only_from_cache to allow fresh correctness validation."
                    ),
                )
            )
        )
    if query_validator is None:
        raise RuntimeError(
            str(
                ErrorEnvelope(
                    code="FINAL_CORRECTNESS_GATE_FAILED",
                    category="correctness",
                    stage="scripted_finalize",
                    message=(
                        "Final correctness gate requires query_validator; "
                        "--disable_valtool is incompatible with scripted handoff."
                    ),
                    recoverable=False,
                    recommended_next_action=(
                        "Enable validator-backed scripted runs and rerun the base implementation phase."
                    ),
                )
            )
        )
    gate_sf_list = required_validation_scale_factors(
        active_verify_sf_list,
        max_scale_factor,
    )
    summary = run_required_correctness_checks(
        run_tool,
        gate_sf_list,
        query_list,
        optimize=True,
        external_call=True,
        fail_fast=False,
        force_fresh_validation=True,
    )
    if summary.success:
        return None
    raise RuntimeError(
        str(
            ErrorEnvelope(
                code="FINAL_CORRECTNESS_GATE_FAILED",
                category="correctness",
                stage="scripted_finalize",
                message=(
                    "Final correctness gate failed after scripted run completion. "
                    f"failure_code={summary.failure_code}, "
                    f"failed_scale_factor={summary.failed_scale_factor}. "
                    f"Detail: {summary.failure_detail or summary.message}"
                ),
                recoverable=False,
                recommended_next_action=(
                    "Run the next outer-loop round to regenerate the base implementation."
                ),
            )
        )
    )


def _run_base_impl_promotion_gate(
    *,
    workspace_path: Path,
) -> None:
    """Run the host-owned base implementation promotion gate."""
    workload_objective = load_json_contract(workspace_path, WORKLOAD_OBJECTIVE_FILE)
    if not workload_objective:
        return None
    ledger = build_pipeline_evidence_ledger(
        workspace_path=workspace_path,
        workload_objective=workload_objective,
        stage=PipelineEvidenceStage.BASE_PROMOTION,
        todo_reconciliation=build_todo_reconciliation(workspace_path / "TODO.md"),
    )
    require_base_impl_promotable(ledger)
    return None


def _validate_resume_contract_fields(
    *,
    workspace_path: Path,
    conv_mode: str | None,
    continue_run: bool,
    start_snapshot: str | None,
) -> None:
    """Reject scripted/optimization resumes that lack the new contract fields."""
    if conv_mode not in {"scripted", "optimization"}:
        return None
    if not continue_run and start_snapshot is None:
        return None
    hashes = collect_control_artifact_hashes(workspace_path)
    snapshot_fields = {
        "implementation_manifest_sha256": hashes.get("implementation_manifest.json"),
        "storage_plan_sha256": hashes.get("storage_plan.txt"),
        "todo_sha256": hashes.get("TODO.md"),
        "todo_reconciliation": build_todo_reconciliation(workspace_path / "TODO.md"),
        "control_artifact_hashes": hashes,
    }
    try:
        require_resume_snapshot_fields(
            snapshot_fields,
            stage="resume_startup",
        )
    except PipelineContractError as exc:
        raise RuntimeError(
            str(
                ErrorEnvelope(
                    code="RESUME_SNAPSHOT_INCOMPLETE",
                    category="handoff",
                    stage="resume_startup",
                    message=str(exc),
                    recoverable=False,
                    recommended_next_action=(
                        "Regenerate the upstream scripted chain so the resumed snapshot contains manifest, storage-plan, TODO, and control-artifact lineage fields."
                    ),
                )
            )
        ) from exc
    return None


def _refresh_control_artifact_audit_copy(workspace_path: Path) -> None:
    """Rewrite the host-owned control-artifact audit copy for the workspace."""
    write_control_artifact_audit_copy(workspace_path)
    return None


def _validate_hardware_counter_preflight(args: argparse.Namespace) -> None:
    """Validate explicit PMU/runtime inputs before optimization starts."""
    backend = getattr(args, "hardware_counter_backend", None)
    if backend in (None, ""):
        return None
    preflight = build_hardware_counter_preflight(
        backend=backend,
        target_cpu=getattr(args, "target_cpu", None),
        runner_cmd=getattr(args, "hardware_counter_runner_cmd", None),
        host_kernel=getattr(args, "host_kernel", None),
        perf_event_paranoid=getattr(args, "perf_event_paranoid", None),
        large_sf=getattr(args, "large_sf", None),
    )
    args.hardware_counter_preflight = preflight
    return None


def _resolve_litellm_reasoning_replay_hook(
    model_name: str,
) -> ShouldReplayReasoningContent | None:
    if is_deepseek_model(model_name):
        return default_should_replay_reasoning_content
    return None


def _activate_tool_runtime(
    runtime: Any,
    *,
    profile_name: str | None,
    prompt_index: int,
    prompt_descriptor: str | None,
    prompt_metadata: dict[str, Any] | None = None,
) -> None:
    activate = runtime.activate
    try:
        parameters = inspect.signature(activate).parameters
    except (TypeError, ValueError):
        activate(
            profile_name=profile_name,
            prompt_index=prompt_index,
            prompt_descriptor=prompt_descriptor,
            prompt_metadata=prompt_metadata,
        )
        return None

    supports_prompt_metadata = "prompt_metadata" in parameters
    supports_kwargs = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )
    if supports_prompt_metadata or supports_kwargs:
        activate(
            profile_name=profile_name,
            prompt_index=prompt_index,
            prompt_descriptor=prompt_descriptor,
            prompt_metadata=prompt_metadata,
        )
        return None

    activate(profile_name, prompt_index, prompt_descriptor)
    return None


def _load_agent_rules(
    tpch_monetdb_root: Path,
    *,
    stage_name: str | None,
    area_name: str | None,
    candidate_paths: tuple[str, ...] = (),
    include_global: bool,
    token_budget: int = DEFAULT_RULE_TOKEN_BUDGET,
) -> RuleAssembly:
    return load_agent_rules(
        tpch_monetdb_root / "agent_rules",
        scope=RuleScope(
            stage_name=stage_name,
            area_name=area_name,
            candidate_paths=candidate_paths,
        ),
        include_global=include_global,
        token_budget=token_budget,
    )


def _log_pending_exception(
    pending_error: BaseException,
    finalize_error: Exception | None,
    teardown_error: Exception | None,
) -> None:
    logger.error(
        "Primary exception while finalizing TPC-H MonetDB run",
        exc_info=pending_error,
    )
    if finalize_error is not None:
        logger.error(
            "W&B finalize failed while propagating primary exception",
            exc_info=finalize_error,
        )
    if teardown_error is not None:
        logger.error(
            "W&B teardown failed while propagating primary exception",
            exc_info=teardown_error,
        )
    return None


def _compose_agent_instructions(
    base_instructions: str,
    global_rules: RuleAssembly,
    scoped_rules: RuleAssembly | None = None,
) -> str:
    parts = [base_instructions.strip()]
    if global_rules.text:
        parts.append(global_rules.text)
    if scoped_rules is not None and scoped_rules.text:
        parts.append(scoped_rules.text)
    return "\n\n".join(part for part in parts if part).strip()


def _create_compaction_session(
    use_litellm: bool,
    session_id: str,
    model_name: str,
    api_key: str,
    base_url: Optional[str],
    client: Any,
    underlying_session: AdvancedSQLiteSession,
    cache_path: Path,
    wandb_metrics_hook: Optional[WandbRunHook],
    compaction_model_map: Optional[dict[str, str]] = None,
) -> Any:
    """Create appropriate compaction session based on path.
    
    Args:
        use_litellm: Whether to use LiteLLM path
        session_id: Session identifier
        model_name: Model name for compaction
        api_key: API key for LiteLLM path
        base_url: Base URL for LiteLLM-compatible endpoints
        client: OpenAI client for OpenAI path
        underlying_session: SQLite session for storage
        cache_path: Path for cache directory
        wandb_metrics_hook: Optional metrics hook
        compaction_model_map: Optional mapping of main model to compaction model
        
    Returns:
        Compaction session instance
    """
    if use_litellm:
        from tpch_monetdb.llm_cache.cached_litellm_compaction import CachedLitellmCompactionSession
        session = CachedLitellmCompactionSession(
            session_id=session_id,
            model=model_name,
            api_key=api_key,
            base_url=base_url,
            cache_dir=cache_path / "compaction",
            wandb_metrics_hook=wandb_metrics_hook,
            compaction_model_map=compaction_model_map or {},
        )
        session.set_underlying_session(underlying_session)
        return session
    else:
        def log_should_trigger_compaction(context: dict[str, Any]) -> bool:
            return False
        
        from tpch_monetdb.llm_cache.cached_compaction_session import CachedOpenAIResponsesCompactionSession
        return CachedOpenAIResponsesCompactionSession(
            session_id=session_id,
            client=client,
            underlying_session=underlying_session,
            should_trigger_compaction=log_should_trigger_compaction,
            cache_dir=cache_path / "compaction",
            model=model_name,
            wandb_metrics_hook=wandb_metrics_hook,
        )


def _wandb_code_include_fn(path: str, root: str) -> bool:
    try:
        rel_path = Path(path).relative_to(root)
    except ValueError:
        rel_path = Path(path)
    excluded_parts = {"build", "db", ".git", "__pycache__"}
    excluded_suffixes = {".o", ".d", ".csv", ".log", ".tmp", ".sqlite"}
    allowed_suffixes = {
        ".c",
        ".cc",
        ".cpp",
        ".cmake",
        ".h",
        ".hpp",
        ".json",
        ".md",
        ".py",
        ".sh",
        ".txt",
    }
    allowed_names = {"CMakeLists.txt", "Makefile", "TODO.md", "queries.txt", "storage_plan.txt"}
    return not any(part in excluded_parts for part in rel_path.parts) and (
        rel_path.name in allowed_names or rel_path.suffix in allowed_suffixes
    ) and rel_path.suffix not in excluded_suffixes


def _upload_workspace_code_to_wandb(
    wandb_run: Any,
    workspace_path: Path,
    *,
    timeout_s: float = DEFAULT_WANDB_UPLOAD_TIMEOUT_S,
) -> None:
    if wandb_run is None or not workspace_path.exists():
        return None
    upload_workspace_code_with_guard(
        wandb_run=wandb_run,
        workspace_path=workspace_path,
        include_fn=_wandb_code_include_fn,
        timeout_s=timeout_s,
    )
    return None


def _log_final_wandb_summary(wandb_metrics_hook: WandbRunHook | None) -> None:
    if wandb_metrics_hook is None:
        return None
    total_tokens = (
        wandb_metrics_hook.total_stats["billed_output_tokens"]
        + wandb_metrics_hook.total_stats["input_tokens"]
    )
    metrics = {
        "final/total_turns": wandb_metrics_hook.last_turn,
        "final/total_tokens": total_tokens,
        "final/total_input_tokens": wandb_metrics_hook.total_stats["input_tokens"],
        "final/total_cached_tokens": wandb_metrics_hook.total_stats["cached_tokens"],
        "final/total_visible_output_tokens": wandb_metrics_hook.total_stats["visible_output_tokens"],
        "final/total_billed_output_tokens": wandb_metrics_hook.total_stats["billed_output_tokens"],
        "final/total_reasoning_tokens": wandb_metrics_hook.total_stats["reasoning_tokens"],
        "final/pricing_missing": int(
            getattr(wandb_metrics_hook, "pricing_missing_seen", False)
        ),
        "final/num_prompts": wandb_metrics_hook.prompt_idx + 1,
    }
    if getattr(wandb_metrics_hook, "known_cost_seen", False):
        metrics["final/total_cost_usd"] = wandb_metrics_hook.total_stats["cost_usd"]
    wandb.log(
        metrics,
        step=wandb_metrics_hook.last_turn,
        commit=True,
    )
    return None


def _resolve_artifacts_context_mode(args: argparse.Namespace) -> str:
    """Resolve artifact context mode while preserving the legacy disable flag."""
    if getattr(args, "disable_artifacts_context", False):
        return "off"
    mode = str(getattr(args, "artifacts_context_mode", "refs") or "refs").lower()
    if mode not in {"refs", "full", "off"}:
        raise ValueError(f"Unsupported artifacts_context_mode: {mode}")
    return mode


def _append_artifacts_context(
    *,
    ledger: ArtifactLedger,
    current: str,
    artifact_text: str,
    kind: str,
    mode: str,
) -> str:
    """Append generated workspace artifacts according to the context mode."""
    if not artifact_text or mode == "off":
        return current
    if mode == "full":
        return current + artifact_text
    ledger.record_text(
        kind=kind,
        text=artifact_text,
        metadata={
            "tool_name": "artifact_context",
            "summary": f"{kind} generated workspace artifact context",
            "tags": ("artifact_context", kind),
        },
    )
    return ledger.refs_for_prompt()


def _build_contextual_input(
    *,
    stage_hint: str,
    stage_memory: str,
    artifact_context: str,
    current_task: str,
    scoped_stage_rules: str = "",
    stage_contract: str = "",
) -> str:
    """Build the per-turn user input with stable context block ordering."""
    blocks = [
        "[Runtime Stage Hint]\n" + stage_hint,
    ]
    if scoped_stage_rules.strip():
        blocks.append("[Scoped Stage Rules]\n" + scoped_stage_rules.strip())
    if stage_contract.strip():
        blocks.append("[Stage Contract]\n" + stage_contract.strip())
    blocks.append("[Current Task]\n" + current_task)
    if stage_memory.strip():
        blocks.append(stage_memory)
    if artifact_context:
        blocks.append(artifact_context)
    return "\n\n".join(blocks)


async def _session_has_context_lifecycle_state(session: Any) -> bool:
    """Return whether compacted state already carries memory/artifact context."""
    try:
        items = await session.get_items()
    except Exception:
        return False
    markers = (
        "[Stage Memory v3]",
        "[Compaction Summary v3]",
        "[Artifact Refs]",
    )
    for item in items:
        text = json.dumps(item, ensure_ascii=False, default=str)
        if any(marker in text for marker in markers):
            return True
    return False


def _build_stage_contract(query_ids: tuple[str, ...]) -> str:
    """Render stable per-query hard constraints before dynamic task context."""
    obligations = q1_q9_obligations_for_query_ids(query_ids)
    if not obligations:
        return ""
    return "\n".join(f"- {obligation}" for obligation in obligations)


def _format_budget_contributors(budget: Any) -> str:
    """Format top context contributors for pre-send budget diagnostics."""
    contributors = getattr(budget, "top_contributors", ()) or ()
    rendered: list[str] = []
    for item in contributors:
        rendered.append(
            f"{getattr(item, 'source', '?')}[{getattr(item, 'item_index', '-')}]"
            f"={getattr(item, 'byte_size', 0)}B:{getattr(item, 'summary', '')}"
        )
    return " | ".join(rendered) if rendered else "(none)"


async def main(args: argparse.Namespace, wandb_result: Any = None) -> None:
    tpch_monetdb_root = Path(__file__).resolve().parent
    workspace_path = resolve_runtime_workspace_path(tpch_monetdb_root)

    cache_path = Path(args.artifacts_dir) / "cache"
    artifact_ledger = ArtifactLedger(cache_path / "context_artifacts")
    artifacts_context_mode = _resolve_artifacts_context_mode(args)

    conversations_dir = Path(args.artifacts_dir) / "conversations"

    dataset_version = None

    snapshotter = build_runtime_snapshotter(
        tpch_monetdb_root,
        disable_repo_sync=args.disable_repo_sync,
        keep_csv=args.keep_csv,
    )
    _prepare_runtime_workspace(
        snapshotter,
        workspace_path,
        continue_run=args.continue_run,
        reset_git_history=(
            not args.continue_run
            and args.start_snapshot is None
            and args.storage_plan_snapshot is None
        ),
    )

    # Prepare query gen
    gen_query_fn = get_query_gen(args.benchmark)
    gen_placeholders_fn = get_placeholders_fn(
        args.benchmark, cache_path / "placeholders_cache"
    )

    query_list = [q.strip() for q in args.query_list.split(",")]
    base_data_dir = getattr(
        args,
        "base_data_dir",
        getattr(args, "base_parquet_dir", args.artifacts_dir),
    )

    restore_storage_plan_snapshot = args.storage_plan_snapshot is not None
    if restore_storage_plan_snapshot:
        assert args.start_snapshot is None, (
            "loading a storage plan snapshot, but also providing a start snapshot is not supported."
        )
    storage_plan = None

    artifacts_in_context = ""
    
    if args.start_snapshot is None:
        if restore_storage_plan_snapshot:
            storage_plan = load_storage_plan_from_snapshot(
                args, snapshotter, workspace_path
            )
            logger.info(
                "Using restored storage plan snapshot %s as base workspace for %s",
                args.storage_plan_snapshot,
                args.conv_name,
            )
            query_artifacts = write_query_and_args_file(
                benchmark_name=args.benchmark,
                gen_placeholders_fn=gen_placeholders_fn,
                query_list=query_list,
                out_dir=workspace_path.as_posix(),
                use_fasttest_format=True,
                storage_plan=storage_plan,
            )
            artifacts_in_context = _append_artifacts_context(
                ledger=artifact_ledger,
                current=artifacts_in_context,
                artifact_text=query_artifacts,
                kind="query_artifacts",
                mode=artifacts_context_mode,
            )
        elif not args.continue_run:
            snapshotter.create_empty_snapshot(args.conv_name)
            template_artifacts = copy_template_to(workspace_path, args.benchmark)
            artifacts_in_context = _append_artifacts_context(
                ledger=artifact_ledger,
                current=artifacts_in_context,
                artifact_text=template_artifacts,
                kind="template_artifacts",
                mode=artifacts_context_mode,
            )

            logger.info(
                f"Generating query and args files for queries: {args.benchmark}/{query_list}"
            )
            query_artifacts = write_query_and_args_file(
                benchmark_name=args.benchmark,
                gen_placeholders_fn=gen_placeholders_fn,
                query_list=query_list,
                out_dir=workspace_path.as_posix(),
                use_fasttest_format=True,
                storage_plan=storage_plan,
            )
            artifacts_in_context = _append_artifacts_context(
                ledger=artifact_ledger,
                current=artifacts_in_context,
                artifact_text=query_artifacts,
                kind="query_artifacts",
                mode=artifacts_context_mode,
            )
    else:
        assert not args.continue_run
        assert snapshotter.has_snapshot(args.start_snapshot), (
            f"Snapshot {args.start_snapshot} not found in repo."
        )
        logger.info(f"Restoring snapshot {args.start_snapshot}")
        snapshotter.restore(args.start_snapshot)
        csv_files = list(workspace_path.rglob("result*.csv"))
        logger.info(f"Deleting existing result-csv files ({len(csv_files)} files).")
        for csv_file in csv_files:
            csv_file.unlink()

        # 确保 queries.txt / args_parser.hpp 存在且与当前 query_list 一致
        # start_snapshot 中这些文件可能为空或过期
        logger.info(
            f"Regenerating query and args files for queries: {args.benchmark}/{query_list}"
        )
        query_artifacts = write_query_and_args_file(
            benchmark_name=args.benchmark,
            gen_placeholders_fn=gen_placeholders_fn,
            query_list=query_list,
            out_dir=workspace_path.as_posix(),
            use_fasttest_format=True,
            storage_plan=storage_plan,
        )
        artifacts_in_context = _append_artifacts_context(
            ledger=artifact_ledger,
            current=artifacts_in_context,
            artifact_text=query_artifacts,
            kind="query_artifacts",
            mode=artifacts_context_mode,
        )

    _validate_resume_contract_fields(
        workspace_path=workspace_path,
        conv_mode=getattr(args, "conv_mode", None),
        continue_run=args.continue_run,
        start_snapshot=args.start_snapshot,
    )

    # Write manifest AFTER snapshot restore to avoid dirtying the workspace
    # before git switch --detach.
    if getattr(args, "conv_mode", None) == "scripted":
        manifest_sidecar = (
            conversations_dir / f"{args.conv_name}.implementation_manifest.json"
        )
        if manifest_sidecar.exists():
            manifest_payload = json.loads(manifest_sidecar.read_text(encoding="utf-8"))
            (workspace_path / "implementation_manifest.json").write_text(
                json.dumps(manifest_payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

    max_scale_factor = (
        args.max_scale_factor if hasattr(args, "max_scale_factor") else 20
    )
    assert max_scale_factor is not None, "max_scale_factor must be set and not None."

    write_workload_objective(
        workspace_path=workspace_path,
        query_ids=query_list,
        benchmark_sf=int(max_scale_factor),
        large_sf=getattr(args, "large_sf", None),
        hardware_counter_backend=getattr(args, "hardware_counter_backend", None),
        target_cpu=getattr(args, "target_cpu", None),
        benchmark=args.benchmark,
    )

    if args.benchmark == "tpch":
        evidence_path = build_tpch_design_evidence(
            workspace_path=workspace_path,
            query_ids=query_list,
            benchmark_sf=int(max_scale_factor),
        )
        write_data_law_contract(
            workspace_path=workspace_path,
            query_ids=query_list,
            benchmark_sf=int(max_scale_factor),
            design_evidence_path=evidence_path,
            benchmark=args.benchmark,
        )
        logger.info("Generated TPC-H design evidence at %s", evidence_path)
    _refresh_control_artifact_audit_copy(workspace_path)
    _validate_hardware_counter_preflight(args)

    model_config = setup_model_config(args.model)
    reasoning = _resolve_reasoning(args, model_config)
    _configure_litellm_cost_map_for_reasoning(model_config, reasoning)

    # Misc setup
    data_path = base_data_dir

    # Create hooks instance for tracking metrics
    wandb_metrics_hook: WandbRunHook | None = None
    if not args.disable_wandb:
        wandb_metrics_hook = WandbRunHook(
            model=model_config.accounting_model_name,
            git_snapshotter=snapshotter,
            cloc_cache_dir=cache_path / "cloc_cache",
        )

    default_verify_sf_list, _default_max_sf = get_default_verify_scale_factors(
        args.benchmark,
        "verify",
    )
    active_verify_sf_list = resolve_active_verify_scale_factors(
        benchmark_sf=max_scale_factor,
        verify_sf_list=default_verify_sf_list,
    )
    validator_sf_list = resolve_workflow_scale_factors(
        benchmark_sf=max_scale_factor,
        verify_sf_list=default_verify_sf_list,
    )

    compile_cache_dir = cache_path / "compile"
    
    # Benchmark-specific validator.
    query_validator = None
    if not args.disable_valtool:
        if args.benchmark != "tpch":
            raise ValueError(
                "Validation tool only supports benchmark='tpch' after "
                "legacy validator removal. Use --benchmark tpch or "
                "--disable_valtool for legacy manual experiments."
            )
        query_validator = TpchRuntimeValidator(
            workspace_path=workspace_path,
            sf_list=validator_sf_list,
            allowed_query_ids=query_list,
            cache_dir=cache_path / "validate_cache",
        )

    run_tool_wrapper, run_tool = make_run_tool(
        cwd=workspace_path,
        query_validator=query_validator,
        wandb_metrics_hook=wandb_metrics_hook,
        compile_cache_dir=compile_cache_dir,
        git_snapshotter=snapshotter,
        dataset_name=get_dataset_name(args.benchmark),
        base_data_dir=base_data_dir,
        run_tool_offer_trace_option=args.run_tool_offer_trace_option,
        only_from_cache=args.only_from_cache,
        target_cpu=getattr(args, "target_cpu", None),
        emit_vectorization_reports=True,
    )

    session_db_path = cache_path / "session" / f"{args.conv_name}.sqlite"
    session_db_path.parent.mkdir(parents=True, exist_ok=True)
    underlying_session = AdvancedSQLiteSession(
        session_id=args.conv_name,
        db_path=session_db_path,
        create_tables=True,
    )

    # Create compaction session (model mapping handled internally by session)
    compaction_model_map = getattr(args, "compaction_model_map", None)

    session = _create_compaction_session(
        use_litellm=model_config.use_litellm,
        session_id=args.conv_name,
        model_name=model_config.model_name,
        api_key=model_config.api_key or "",
        base_url=model_config.base_url,
        client=model_config.openai_client,
        underlying_session=underlying_session,
        cache_path=cache_path,
        wandb_metrics_hook=wandb_metrics_hook,
        compaction_model_map=compaction_model_map,
    )

    # Build tools using factory
    tool_bundle = build_tools(
        use_litellm=model_config.use_litellm,
        workspace_path=workspace_path,
        cache_path=cache_path,
        extra_read_roots=(Path(base_data_dir).resolve(),),
        shell_executor=None,
        workspace_editor=None,
        compile_cache_dir=compile_cache_dir,
        run_tool_wrapper=run_tool_wrapper,
        git_snapshotter=snapshotter,
        wandb_metrics_hook=wandb_metrics_hook,
        artifact_ledger=artifact_ledger,
    )
    tools = tool_bundle.all_tools

    # Prepare dict to be included in hash
    config_kwargs: Dict[str, Any] = {"max_snapshot_csv_size_mb": 5.0}
    if args.start_snapshot is not None:
        config_kwargs["start_snapshot"] = args.start_snapshot
    if dataset_version is not None:
        config_kwargs["dataset_version"] = dataset_version
    if getattr(args, "stream_llm", False):
        config_kwargs["stream_llm"] = True

    if model_config.use_litellm:
        from tpch_monetdb.llm_cache.cached_litellm import CachedLitellmModel
        model = CachedLitellmModel(
            model=model_config.model_name,
            api_key=model_config.api_key,
            base_url=model_config.base_url,
            should_replay_reasoning_content=_resolve_litellm_reasoning_replay_hook(
                model_config.model_name
            ),
            llm_cache_dir=cache_path / "llm_cache",
            snapshotter=snapshotter,
            stop_on_cache_miss=args.replay,
            query_gen_list=query_list,
            artifacts_in_context=artifacts_in_context,
            config_kwargs=config_kwargs,
        )
    else:
        from tpch_monetdb.llm_cache.cached_openai import CachedOpenAIResponsesModel
        model = CachedOpenAIResponsesModel(
            model=model_config.model_name,
            openai_client=model_config.openai_client,
            llm_cache_dir=cache_path / "llm_cache",
            snapshotter=snapshotter,
            stop_on_cache_miss=args.replay or args.only_from_llm_cache or args.only_from_cache,
            query_gen_list=query_list,
            artifacts_in_context=artifacts_in_context,
            config_kwargs=config_kwargs,
        )

    litellm_tool_guidance = ""
    if model_config.use_litellm:
        litellm_tool_guidance = render_agent_text_asset("runtime.litellm_tool_guidance")
    base_instructions = render_agent_text_asset(
        "runtime.base_agent_instructions",
        {
            "workspace_path": workspace_path,
            "litellm_tool_guidance": litellm_tool_guidance,
        },
    )
    global_rule_assembly = _load_agent_rules(
        tpch_monetdb_root,
        stage_name=None,
        area_name=None,
        include_global=True,
    )
    log_rule_assembly("global", global_rule_assembly)
    base_agent_instructions = _compose_agent_instructions(
        base_instructions,
        global_rule_assembly,
    )

    if reasoning is not None:
        logger.info("Using reasoning effort: %s", reasoning.effort)

    model_settings = _build_model_settings(args, model_config)
    
    default_agent_name = "Generated TPC-H Assistant"
    agent = Agent(
        name=default_agent_name,
        model=model,
        instructions=base_agent_instructions,
        tools=tools,
        model_settings=model_settings,
    )

    logger.info(f"Workspace root: {workspace_path}")
    logger.info(f"Using model: {model}")

    # Initialize auto-compact manager
    auto_compact_manager: AutoCompactManager | None = None
    if hasattr(args, "enable_auto_compact") and args.enable_auto_compact:
        auto_compact_manager = AutoCompactManager(
            model_config.accounting_model_name,
            artifact_ledger=artifact_ledger,
        )
        logger.info(f"Auto-compact enabled with threshold: {auto_compact_manager.get_threshold()}")

    last_token_usage = 0

    async def handle_prompt(
        text: str,
        short_desc: Optional[str],
        idx: int,
        max_turns: Optional[int] = None,
        prompt_metadata: Optional[dict[str, Any]] = None,
    ) -> Any:
        nonlocal last_token_usage
        if max_turns is None:
            max_turns = 75

        # Stage-aware proactive compact and circuit breaker
        stage_policy = get_policy_for_stage(short_desc or "")
        compact_profile = None if prompt_metadata is None else prompt_metadata.get("tool_profile")
        if auto_compact_manager is not None and stage_policy is not None:
            if auto_compact_manager.consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                if stage_policy.block_on_context_saturation:
                    logger.warning(
                        "Stage '%s' continues despite auto-compact circuit breaker being open after %s consecutive failures: %s",
                        short_desc,
                        MAX_CONSECUTIVE_FAILURES,
                        auto_compact_manager.last_failure_info,
                    )
            if stage_policy.proactive_compact_on_warning:
                proactive_estimate = await auto_compact_manager.estimate_request_tokens(
                    session=underlying_session, new_input=""
                )
                if proactive_estimate >= auto_compact_manager.get_warning_threshold(compact_profile):
                    logger.info(
                        "Proactive compact triggered for stage='%s' at %s tokens (warning=%s)",
                        short_desc,
                        proactive_estimate,
                        auto_compact_manager.get_warning_threshold(compact_profile),
                    )
                    compact_ok = await auto_compact_manager.compact(
                        session=underlying_session,
                        compaction_session=session,
                        current_tokens=proactive_estimate,
                        profile_name=compact_profile or short_desc,
                    )
                    if not compact_ok and stage_policy.block_on_context_saturation:
                        logger.warning(
                            "Stage '%s' continues after proactive compact failure: %s",
                            short_desc,
                            auto_compact_manager.last_failure_info,
                        )

        # Check for compaction marker
        if text == COMPACTION_MARKER:
            logger.info(f"Triggering compaction at prompt index {idx}")
            # Use force_trigger to ensure compaction runs, but don't force_regenerate
            # so that replay cache can reuse the summary
            attempt = await session.run_compaction(
                {
                    "force_trigger": True,
                    "compaction_mode": "input",
                    "selection_policy": "stage_memory_v3",
                    "preserve_limit_items": 12,
                    "min_candidate_items": 0,
                }
            )
            if attempt is not None and hasattr(attempt, "status") and attempt.status != "success":
                logger.warning(
                    "Manual compaction finished with status=%s reason=%s",
                    attempt.status,
                    attempt.skip_reason,
                )
            return None
        
        # Check for validation markers
        if text == VALIDATE_ON:
            run_tool.parse_out_and_validate_output = True
            logger.info(f"Enabled output parsing and validation at prompt index {idx}")
            return None
        if text == VALIDATE_OFF:
            run_tool.parse_out_and_validate_output = False
            logger.info(f"Disabled output parsing and validation at prompt index {idx}")
            return None
        if text == VALIDATE_OUTPUT_STDOUT_ON:
            if query_validator is not None:
                query_validator.output_stdout_stderr = True
                logger.info(f"Enabled output stdout in validation results at prompt index {idx}")
            return None
        if text == VALIDATE_OUTPUT_STDOUT_OFF:
            if query_validator is not None:
                query_validator.output_stdout_stderr = False
                logger.info(f"Disabled output stdout in validation results at prompt index {idx}")
            return None

        logger.info("=" * 80)
        logger.info(text)
        logger.info("=" * 80)

        # Update prompt index in hooks
        if wandb_metrics_hook is not None:
            wandb_metrics_hook.prompt_idx = idx
            wandb_metrics_hook.current_prompt = text
            wandb_metrics_hook.current_prompt_descriptor = short_desc

        # Rename the agent for each stage
        if short_desc is None:
            agent.name = default_agent_name
        else:
            agent.name = f"{default_agent_name} ({short_desc})"

        # Run with hooks for automatic metric tracking
        original_tools = agent.tools
        original_instructions = base_agent_instructions
        tool_profile = None if prompt_metadata is None else prompt_metadata.get("tool_profile")
        _activate_tool_runtime(
            tool_bundle.runtime,
            profile_name=tool_profile,
            prompt_index=idx,
            prompt_descriptor=short_desc,
            prompt_metadata=prompt_metadata,
        )
        profile_key = tool_profile or "default_general"
        rule_area = None if prompt_metadata is None else prompt_metadata.get("rule_area")
        rule_paths_raw = () if prompt_metadata is None else prompt_metadata.get("rule_paths", ())
        if isinstance(rule_paths_raw, (list, tuple)):
            rule_paths = tuple(str(item) for item in rule_paths_raw)
        else:
            rule_paths = ()
        scoped_rule_assembly = _load_agent_rules(
            tpch_monetdb_root,
            stage_name=profile_key,
            area_name=rule_area,
            candidate_paths=rule_paths,
            include_global=False,
        )
        scoped_scope_label = f"stage={profile_key},area={rule_area or '(none)'}"
        log_rule_assembly(scoped_scope_label, scoped_rule_assembly)
        agent.instructions = base_agent_instructions
        agent.tools = tool_bundle.all_tools
        agent.model_settings = _build_stage_model_settings(args, model_config, profile_key)
        if wandb_metrics_hook is not None:
            wandb_metrics_hook.set_current_stage(profile_key)

        # Inject stage hint into input
        stage_hint = tool_bundle.runtime.generate_stage_hint()
        active_query_ids = (
            tuple(str(item) for item in prompt_metadata.get("active_query_ids", ()))
            if isinstance(prompt_metadata, dict)
            else ()
        )
        active_unit_query_ids = (
            tuple(str(item) for item in prompt_metadata.get("active_unit_query_ids", ()))
            if isinstance(prompt_metadata, dict)
            else ()
        )
        scoped_query_ids = tuple(dict.fromkeys((*active_query_ids, *active_unit_query_ids)))
        artifact_context = artifact_ledger.refs_for_prompt(
            query_ids=scoped_query_ids,
            stage_name=profile_key,
        )
        generate_stage_memory = getattr(tool_bundle.runtime, "generate_stage_memory", None)
        stage_memory = (
            generate_stage_memory(artifact_refs=artifact_context)
            if callable(generate_stage_memory)
            else "[Stage Memory v3]\n(unavailable for this runtime)"
        )
        input_with_hint = _build_contextual_input(
            stage_hint=stage_hint,
            scoped_stage_rules=scoped_rule_assembly.text,
            stage_contract=_build_stage_contract(scoped_query_ids),
            stage_memory=stage_memory,
            artifact_context=artifact_context,
            current_task=text,
        )
        input_for_model = input_with_hint

        # Pre-send guard: 发送前同时估算 token 和 serialized session body。
        if auto_compact_manager is not None:
            estimate_budget = getattr(auto_compact_manager, "estimate_request_budget", None)
            pre_send_budget = (
                await estimate_budget(session=underlying_session, new_input=input_for_model)
                if callable(estimate_budget)
                else None
            )
            if pre_send_budget is not None:
                pre_send_estimate = pre_send_budget.token_estimate
                if pre_send_budget.should_warn:
                    logger.warning(
                        "Pre-send context budget: tokens=%s(level=%s) body=%s bytes(level=%s) contributors=%s",
                        pre_send_budget.token_estimate,
                        pre_send_budget.token_level,
                        pre_send_budget.body_bytes,
                        pre_send_budget.body_level,
                        _format_budget_contributors(pre_send_budget),
                    )
                pre_send_should_compact = pre_send_budget.should_compact
            else:
                pre_send_estimate = await auto_compact_manager.estimate_request_tokens(
                    session=underlying_session, new_input=input_for_model
                )
                pre_send_should_compact = False
            blocking = auto_compact_manager.get_blocking_threshold()
            if pre_send_should_compact or pre_send_estimate >= blocking:
                logger.warning(
                    "Pre-send guard forcing compaction: estimated %s tokens, blocking threshold %s, body_compact=%s (stage='%s' prompt=%s)",
                    pre_send_estimate,
                    blocking,
                    getattr(pre_send_budget, "body_compact", False),
                    short_desc,
                    idx,
                )
                compact_ok = await auto_compact_manager.compact(
                    session=underlying_session,
                    compaction_session=session,
                    current_tokens=pre_send_estimate,
                    profile_name=compact_profile or short_desc,
                    stage_memory=stage_memory,
                    artifact_context=artifact_context,
                )
                if compact_ok and await _session_has_context_lifecycle_state(underlying_session):
                    input_for_model = _build_contextual_input(
                        stage_hint=stage_hint,
                        scoped_stage_rules=scoped_rule_assembly.text,
                        stage_contract=_build_stage_contract(scoped_query_ids),
                        stage_memory="",
                        artifact_context="",
                        current_task=text,
                    )
                post_compact_budget = (
                    await estimate_budget(session=underlying_session, new_input=input_for_model)
                    if callable(estimate_budget)
                    else None
                )
                post_compact_estimate = (
                    post_compact_budget.token_estimate
                    if post_compact_budget is not None
                    else await auto_compact_manager.estimate_request_tokens(
                        session=underlying_session,
                        new_input=input_for_model,
                    )
                )
                if post_compact_budget is not None and post_compact_budget.should_warn:
                    logger.warning(
                        "Post-compact context budget: tokens=%s(level=%s) body=%s bytes(level=%s) contributors=%s",
                        post_compact_budget.token_estimate,
                        post_compact_budget.token_level,
                        post_compact_budget.body_bytes,
                        post_compact_budget.body_level,
                        _format_budget_contributors(post_compact_budget),
                    )
                post_compact_failed = (
                    post_compact_budget.should_fail
                    if post_compact_budget is not None
                    else post_compact_estimate >= blocking
                )
                if post_compact_failed:
                    descriptor = short_desc or "prompt"
                    if post_compact_budget is None:
                        detail = (
                            f"estimated {post_compact_estimate} tokens still "
                            f"exceeds blocking threshold {blocking} after forced compaction"
                        )
                    else:
                        detail = (
                            f"estimated {post_compact_budget.token_estimate} tokens "
                            f"(level={post_compact_budget.token_level}) and "
                            f"{post_compact_budget.body_bytes} body bytes "
                            f"(level={post_compact_budget.body_level}) after forced compaction; "
                            f"contributors={_format_budget_contributors(post_compact_budget)}"
                        )
                    raise RuntimeError(
                        str(
                            _context_too_large_envelope(
                                prompt_index=idx,
                                descriptor=descriptor,
                                detail=detail,
                            )
                        )
                    )

        # 模型可能幻觉出不存在的工具名，最多重试 N 次，每次告诉 AI 可用工具有哪些
        MAX_TOOL_RETRIES = 3
        MAX_REACTIVE_COMPACT_RETRIES = 3
        retries_left = MAX_TOOL_RETRIES
        reactive_retries_left = MAX_REACTIVE_COMPACT_RETRIES
        current_input = input_for_model
        current_max_turns = max_turns

        try:
            while True:
                try:
                    result = await _run_agent_turn(
                        agent,
                        input=current_input,
                        session=session,
                        max_turns=current_max_turns,
                        hooks=wandb_metrics_hook,
                        stream_llm=bool(getattr(args, "stream_llm", False)),
                    )
                    break
                except ModelBehaviorError as exc:
                    err_msg = str(exc)
                    if retries_left <= 0:
                        raise RuntimeError(
                            f"Prompt {idx} ({short_desc or 'prompt'}): model repeatedly "
                            f"called non-existent tools after {MAX_TOOL_RETRIES} retries. "
                            f"Last error: {err_msg}"
                        ) from exc
                    retries_left -= 1
                    logger.warning(
                        "Prompt %s (%s): model called non-existent tool: %s. "
                        "Retry %s/%s — injecting tool-list correction.",
                        idx, short_desc or "prompt", err_msg,
                        MAX_TOOL_RETRIES - retries_left,
                        MAX_TOOL_RETRIES,
                    )
                    available_names = [t.name for t in agent.tools]
                    correction = render_agent_text_asset(
                        "runtime.tool_correction",
                        {
                            "error_message": err_msg,
                            "available_tools": ", ".join(available_names),
                        },
                    )
                    await underlying_session.add_items([
                        {"role": "user", "content": correction}
                    ])
                    current_input = render_agent_text_asset(
                        "runtime.tool_correction_continue"
                    )
                    current_max_turns = min(max_turns, 20)
                except Exception as exc:
                    if (
                        auto_compact_manager is not None
                        and reactive_retries_left > 0
                        and should_reactive_compact(exc)
                    ):
                        reactive_retries_left -= 1
                        logger.warning(
                            "Prompt %s (%s): reactive context lifecycle compact after model failure: %s. Retry %s/%s",
                            idx,
                            short_desc or "prompt",
                            exc,
                            MAX_REACTIVE_COMPACT_RETRIES - reactive_retries_left,
                            MAX_REACTIVE_COMPACT_RETRIES,
                        )
                        compact_ok = await auto_compact_manager.compact(
                            session=underlying_session,
                            compaction_session=session,
                            current_tokens=pre_send_estimate,
                            profile_name=compact_profile or short_desc,
                            stage_memory=stage_memory,
                            artifact_context=artifact_context,
                            force_aggressive=True,
                        )
                        if not compact_ok:
                            logger.warning(
                                "Prompt %s (%s): reactive context lifecycle compact made no effective change; aborting unchanged retry.",
                                idx,
                                short_desc or "prompt",
                            )
                            raise
                        if await _session_has_context_lifecycle_state(underlying_session):
                            current_input = _build_contextual_input(
                                stage_hint=stage_hint,
                                scoped_stage_rules=scoped_rule_assembly.text,
                                stage_contract=_build_stage_contract(scoped_query_ids),
                                stage_memory="",
                                artifact_context="",
                                current_task=text,
                            )
                        else:
                            current_input = input_with_hint
                        current_max_turns = max_turns
                        continue
                    raise
        except Exception as exc:
            descriptor = short_desc or "prompt"
            if classify_model_failure(str(exc)) == "CONTEXT_TOO_LARGE":
                raise RuntimeError(
                    str(
                        _context_too_large_envelope(
                            prompt_index=idx,
                            descriptor=descriptor,
                            detail=str(exc),
                        )
                    )
                ) from exc
            if isinstance(exc, RuntimeError):
                raise
            if (
                is_deepseek_model(getattr(model_config, "accounting_model_name", "") or "")
                and "reasoning_content" in str(exc)
            ):
                raise RuntimeError(
                    f"Prompt {idx} ({descriptor}) failed: "
                    "DeepSeek thinking replay is missing assistant reasoning_content "
                    "for a tool-call continuation."
                ) from exc
            raise RuntimeError(
                f"Prompt {idx} ({descriptor}) failed: {exc}"
            ) from exc
        finally:
            agent.tools = original_tools
            agent.instructions = original_instructions
            if wandb_metrics_hook is not None:
                wandb_metrics_hook.set_current_stage(None)

        # Log cost summary
        get_tokens_context_and_dollar_info(
            result.context_wrapper.usage,
            model_config.accounting_model_name,
            last_entry_only=False,
            log=True,
        )

        # Log final output (truncated)
        logger.info(truncate_model_final_output(result.final_output))

        stage_summary = tool_bundle.runtime.finish_stage(result.final_output)
        await underlying_session.add_items(
            [{"role": "assistant", "content": render_stage_summary(stage_summary)}]
        )
        post_stage_artifact_context = artifact_ledger.refs_for_prompt(
            query_ids=scoped_query_ids,
            stage_name=stage_summary.profile_name or profile_key,
        )
        post_stage_memory = (
            generate_stage_memory(artifact_refs=post_stage_artifact_context)
            if callable(generate_stage_memory)
            else "[Stage Memory v3]\n(unavailable for this runtime)"
        )

        # Stage-end maintenance keeps context bounded without forcing LLM compaction each round.
        if auto_compact_manager is not None:
            token_usage = result.context_wrapper.usage.total_tokens if result.context_wrapper.usage else 0
            last_token_usage = token_usage
            maintenance_policy = get_policy_for_stage(short_desc or stage_summary.profile_name)
            maintain_after_stage = getattr(auto_compact_manager, "maintain_after_stage", None)
            if (
                callable(maintain_after_stage)
                and (
                    maintenance_policy is None
                    or maintenance_policy.stage_end_maintenance
                )
            ):
                maintenance = await maintain_after_stage(
                    session=underlying_session,
                    compaction_session=session,
                    profile_name=stage_summary.profile_name,
                    stage_name=short_desc or stage_summary.profile_name,
                    query_ids=scoped_query_ids,
                    stage_memory=post_stage_memory,
                    artifact_context=post_stage_artifact_context,
                    allow_llm_compaction=(
                        True
                        if maintenance_policy is None
                        else maintenance_policy.stage_end_llm_compact_on_orange
                    ),
                    force_llm_compaction=(
                        False
                        if maintenance_policy is None
                        else maintenance_policy.stage_end_force_llm_compact
                    ),
                )
                last_token_usage = maintenance.post_budget.token_estimate
                if maintenance.should_fail:
                    detail = (
                        f"stage-end maintenance left tokens={maintenance.post_budget.token_estimate} "
                        f"(level={maintenance.post_budget.token_level}) and "
                        f"{maintenance.post_budget.body_bytes} body bytes "
                        f"(level={maintenance.post_budget.body_level}); "
                        f"contributors={_format_budget_contributors(maintenance.post_budget)}; "
                        f"failure={maintenance.failure_detail or '-'}"
                    )
                    raise RuntimeError(
                        str(
                            _context_too_large_envelope(
                                prompt_index=idx,
                                descriptor=short_desc or "prompt",
                                detail=detail,
                            )
                        )
                    )
            elif auto_compact_manager.should_compact(token_usage, stage_summary.profile_name):
                logger.info(f"Auto-compact triggered at {token_usage} tokens")
                await auto_compact_manager.compact(
                    session=underlying_session,
                    compaction_session=session,
                    current_tokens=token_usage,
                    profile_name=stage_summary.profile_name,
                )

        return stage_summary

    # Manually traced conversation
    with trace(
        f"Generated TPC-H-Agent {args.conv_name} Conversation",
        metadata={
            "query": args.conv_name,
            "model": args.model,
            "tools": str([type(t).__name__ for t in tools]),
        },
    ):
        conv_args = dict(
            conversation_json_path=conversations_dir / f"{args.conv_name}.json",
            callback=handle_prompt,
            auto_finish=args.auto_finish,
            replay_cache=args.replay_cache,
            auto_u=args.auto_u,
            replay=args.replay,
            notify=args.notify,
            model=model,
            workspace_root=workspace_path,
        )
        if args.conv_mode == "scripted":
            conv = ScriptedConversation(**conv_args)
        elif args.conv_mode == "optimization":
            assert query_validator is not None, (
                "query_validator must be provided for optimization conversation"
            )
            from tpch_monetdb.conversations.optimization_conversation_tpch_monetdb import (
                TpchMonetdbOptimizationConversation,
            )
            conv = TpchMonetdbOptimizationConversation(
                query_ids=query_list,
                bespoke_storage=True,
                run_tool=run_tool,
                verify_sf_list=active_verify_sf_list,
                benchmark_sf=max_scale_factor,
                git_snapshotter=snapshotter,
                revert_on_regression=True,
                session=underlying_session,
                wandb_run_hook=wandb_metrics_hook,
                wandb_init_result=wandb_result,
                conv_name=args.conv_name,
                artifacts_dir=args.artifacts_dir,
                start_snapshot_hash=args.start_snapshot or "",
                benchmark=args.benchmark,
                benchmark_mode=getattr(args, "benchmark_mode", "system-parity"),
                storage_mode=getattr(args, "storage_mode", "persistent"),
                target_cpu=getattr(args, "target_cpu", None),
                hardware_counter_preflight=getattr(
                    args,
                    "hardware_counter_preflight",
                    None,
                ),
                large_sf=getattr(args, "large_sf", None),
                **conv_args,
            )
        else:
            raise ValueError(f"Unknown conversation mode: {args.conv_mode}")

        await conv.run()

        if args.conv_mode == "scripted":
            _snapshot_final_workspace_state(snapshotter, args.conv_name)
            final_hash = snapshotter.current_hash
            if not final_hash:
                raise RuntimeError(
                    str(
                        ErrorEnvelope(
                            code="HANDOFF_FAILED",
                            category="handoff",
                            stage="scripted_finalize",
                            message="No final snapshot hash available for scripted run summary.",
                            recoverable=False,
                            recommended_next_action="Inspect snapshot creation and rerun scripted conversation.",
                        )
                    )
                )

            # Storage plan phase summary
            if "storageplan" in args.conv_name.lower():
                try:
                    storage_plan_path = workspace_path / "storage_plan.txt"
                    persist_successful_storage_plan_run(
                        benchmark=args.benchmark,
                        conv_name=args.conv_name,
                        query_list=query_list,
                        final_snapshot_hash=final_hash,
                        storage_plan_path=storage_plan_path,
                        conversation_json_path=conversations_dir / f"{args.conv_name}.json",
                        session_db_path=session_db_path,
                        artifacts_dir=Path(args.artifacts_dir),
                        wandb_result=wandb_result,
                    )
                except Exception as exc:
                    raise RuntimeError(
                        str(
                            ErrorEnvelope(
                                code="HANDOFF_FAILED",
                                category="handoff",
                                stage="storage_plan_finalize",
                                message=(
                                    "Storage plan run completed but handoff summary write failed: "
                                    f"{exc}"
                                ),
                                recoverable=False,
                                relevant_files=("tpch_monetdb/utils/storage_plan_summary.py",),
                                recommended_next_action="Inspect summary persistence and rerun storage plan handoff.",
                            )
                        )
                    ) from exc
                _log_final_wandb_summary(wandb_metrics_hook)
                return

            validation_mode = getattr(args, "validation_mode", None)
            if validation_mode is None:
                raise RuntimeError(
                    str(
                        ErrorEnvelope(
                            code="HANDOFF_FAILED",
                            category="handoff",
                            stage="scripted_finalize",
                            message="Scripted run missing validation_mode. Expected 'strict' or 'traversal'.",
                            recoverable=False,
                            recommended_next_action="Pass validation_mode through the scripted runtime configuration.",
                        )
                    )
                )
            _run_final_correctness_gate(
                query_validator=query_validator,
                run_tool=run_tool,
                active_verify_sf_list=active_verify_sf_list,
                max_scale_factor=max_scale_factor,
                query_list=query_list,
                only_from_cache=args.only_from_cache,
            )
            _run_base_impl_promotion_gate(
                workspace_path=workspace_path,
            )
            try:
                stage_summaries = (
                    getattr(conv, "completed_stage_summaries", None)
                    if isinstance(conv, ScriptedConversation)
                    else None
                )
                persist_successful_scripted_run(
                    benchmark=args.benchmark,
                    conv_name=args.conv_name,
                    query_list=query_list,
                    is_bespoke_storage=True,
                    final_snapshot_hash=final_hash,
                    conversation_json_path=conversations_dir / f"{args.conv_name}.json",
                    session_db_path=session_db_path,
                    artifacts_dir=Path(args.artifacts_dir),
                    validation_mode=validation_mode,
                    workspace_path=workspace_path,
                    stage_summaries=stage_summaries,
                    wandb_result=wandb_result,
                )
            except Exception as exc:
                raise RuntimeError(
                    str(
                        ErrorEnvelope(
                            code="HANDOFF_FAILED",
                            category="handoff",
                            stage="scripted_finalize",
                            message=(
                                "Scripted run completed but handoff summary write failed: "
                                f"{exc}"
                            ),
                            recoverable=False,
                            relevant_files=("tpch_monetdb/utils/scripted_summary.py",),
                            recommended_next_action="Inspect summary persistence and rerun scripted handoff.",
                        )
                    )
                ) from exc

    logger.debug(f"Model cache total saved: ${model.total_saved:0.6f}")

    if not args.disable_wandb:
        _log_final_wandb_summary(wandb_metrics_hook)


def run_conv_wrapper(args: argparse.Namespace) -> None:
    if args.continue_run:
        ask_yes_no(
            "Are you really sure you want to continue the current snapshot? "
            "Does not start from fresh and continues from current state of output folder. "
            "This is DANGEROUS as it might include unwanted files already present in the output folder!"
        )

    log_path = Path(args.artifacts_dir) / "logs"
    log_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    setup_logging(logging.DEBUG, log_path / f"{timestamp}_{args.conv_name}.log")
    
    if not hasattr(args, "base_data_dir"):
        args.base_data_dir = getattr(args, "base_parquet_dir", args.artifacts_dir)

    if args.disable_tracing and bool(
        getattr(args, "disable_wandb_when_tracing_disabled", False)
    ):
        logger.info(
            "disable_wandb_when_tracing_disabled is enabled; forcing disable_wandb=True"
        )
        args.disable_wandb = True

    wandb_run: Any | None = None
    _wandb_init_result: Any | None = None
    wandb_init_attempts = int(getattr(args, "wandb_init_max_attempts", 3))
    wandb_init_timeout_s = float(
        getattr(args, "wandb_init_timeout_s", DEFAULT_WANDB_INIT_TIMEOUT_S)
    )
    wandb_upload_timeout_s = float(
        getattr(args, "wandb_upload_timeout_s", DEFAULT_WANDB_UPLOAD_TIMEOUT_S)
    )
    wandb_finish_timeout_s = float(
        getattr(args, "wandb_finish_timeout_s", DEFAULT_WANDB_FINISH_TIMEOUT_S)
    )
    wandb_finish_retries = int(
        getattr(args, "wandb_finish_retries", DEFAULT_WANDB_FINISH_RETRIES)
    )

    load_dotenv()
    if args.disable_tracing:
        set_tracing_disabled(True)
    if not args.disable_wandb:
        entity = os.getenv("WANDB_ENTITY", "learneddb")
        project = os.getenv("WANDB_PROJECT", "bespoke-olap-agents")

        if not args.disable_tracing:
            configure_weave_cache_dirs()
            import weave

            weave.init(
                f"{entity}/{project}",
                settings={"log_level": "INFO", "print_call_link": False},
            )

        tags = [args.benchmark]
        if args.is_bespoke_storage:
            tags.append("bespoke-storage")
        tags.append("generated_tpch")
        _wandb_init_result = init_wandb_run_with_retry(
            wandb_module=wandb,
            args=args,
            entity=entity,
            project=project,
            tags=tags,
            max_attempts=wandb_init_attempts,
            init_timeout_s=wandb_init_timeout_s,
        )
        wandb_run = _wandb_init_result.run

    workspace_path = resolve_runtime_workspace_path(Path(__file__).resolve().parent)
    pending_error: BaseException | None = None
    try:
        asyncio.run(main(args, wandb_result=_wandb_init_result))
    except BaseException as exc:
        pending_error = exc

    finalize_error: Exception | None = None
    teardown_error: Exception | None = None
    if not args.disable_wandb and wandb_run is not None:
        try:
            _upload_workspace_code_to_wandb(
                wandb_run,
                workspace_path,
                timeout_s=wandb_upload_timeout_s,
            )
            finish_wandb_with_guard(
                wandb_module=wandb,
                timeout_s=wandb_finish_timeout_s,
                retries=wandb_finish_retries,
            )
        except Exception as exc:
            finalize_error = exc
        teardown = getattr(wandb, "teardown", None)
        if callable(teardown):
            exit_code = 1 if (pending_error is not None or finalize_error is not None) else 0
            try:
                run_callable_with_timeout(
                    lambda: teardown(exit_code=exit_code),
                    timeout_s=wandb_finish_timeout_s,
                    operation_name="wandb.teardown",
                )
            except Exception as exc:
                teardown_error = exc

    if pending_error is not None:
        _log_pending_exception(pending_error, finalize_error, teardown_error)
        raise pending_error

    if finalize_error is not None:
        if teardown_error is not None:
            logger.error(
                "W&B teardown failed after finalize failure",
                exc_info=teardown_error,
            )
        raise finalize_error
    if teardown_error is not None:
        logger.warning(
            "W&B teardown failed after run completion",
            exc_info=teardown_error,
        )
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    manual = subparsers.add_parser(
        "manual",
        help="Run a conversation using explicit mode/query args.",
    )
    add_common_args(
        manual,
        include_model=True,
        include_reasoning_effort=True,
        include_replay=True,
        include_disable_tracing=True,
        include_disable_wandb=True,
        include_disable_wandb_when_tracing_disabled=True,
        include_wandb_init_max_attempts=True,
        include_wandb_init_timeout_s=True,
        include_wandb_upload_timeout_s=True,
        include_wandb_finish_timeout_s=True,
        include_wandb_finish_retries=True,
        include_conv_name=True,
        include_query_list=True,
        include_continue_run=True,
        include_artifacts_dir=True,
        include_no_preload=True,
        include_notify=True,
        include_start_snapshot=True,
        include_disable_repo_sync=True,
        include_replay_cache=True,
        include_auto_u=True,
        include_keep_csv=True,
        include_disable_valtool=True,
        include_disable_artifacts_context=True,
        include_artifacts_context_mode=True,
        include_benchmark=True,
        include_auto_finish=True,
        include_storage_plan_snapshot=True,
        include_conv_mode=True,
        include_is_bespoke_storage=True,
        include_run_tool_offer_trace_option=True,
        include_base_data_dir=True,
        include_only_from_cache=True,
        include_enable_auto_compact=True,
        include_baseline_backend=False,
        include_baseline_query_file_dir=False,
        include_benchmark_mode=True,
        include_storage_mode=True,
        include_target_cpu=True,
        include_hardware_counter_backend=True,
        include_hardware_counter_runner_cmd=True,
        include_host_kernel=True,
        include_perf_event_paranoid=True,
        include_large_sf=True,
        include_stream_llm=True,
    )
    args = parser.parse_args()
    args.write_query_and_args_files = True

    if args.command == "manual":
        run_conv_wrapper(args)
    else:
        raise Exception(f"Unknown {args.command}")
