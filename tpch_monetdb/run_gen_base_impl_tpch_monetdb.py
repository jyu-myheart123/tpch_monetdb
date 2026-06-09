"""Scripted Base Implementation 入口.

这是 base implementation 生成入口，支持：
- benchmark: tpch
- validation_mode: strict (默认) 或 traversal
- 自动 compact 默认开启

用法:
    PYTHONPATH=. python -m tpch_monetdb.run_gen_base_impl_tpch_monetdb \
      --conv basef1-2v1 \
      --benchmark tpch \
      --model litellm/anthropic/kimi-k2.5 \
      --base_data_dir /tmp/tpch_monetdb_data \
      --validation_mode strict \
      --auto_u \
      --auto_finish
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

from tpch_monetdb.bootstrap_env import bootstrap_runtime_env

bootstrap_runtime_env()

from tpch_monetdb.conversations.conversation import (
    VALIDATE_OFF,
    VALIDATE_ON,
    VALIDATE_OUTPUT_STDOUT_OFF,
)
from tpch_monetdb.conversations.scripted_prompts_gen import render_scripted_prompt_asset
from tpch_monetdb.main_tpch_monetdb import run_conv_wrapper
from tpch_monetdb.config import (
    DEFAULT_TPCH_MONETDB_ARTIFACTS_DIR,
    get_default_benchmark_scale_factor,
)
from tpch_monetdb.runtime_stage_policy import DEFAULT_STAGE_TURN_BUDGET
from tpch_monetdb.utils.cli_config import add_common_args, build_run_config
from tpch_monetdb.utils.gen_common import parse_query_ids
from tpch_monetdb.utils.outer_loop_state import render_workflow_priority_order
from tpch_monetdb.utils.query_units import (
    build_active_unit_metadata,
    build_family_prompt_context,
    build_query_units_for_requested_queries,
    write_manifest_for_conversation,
)
from tpch_monetdb.utils.query_codegen_hints import (
    build_query_codegen_hint_text,
    get_query_generated_code_checks,
)

logger = __import__("logging").getLogger(__name__)


def main(args: argparse.Namespace) -> None:
    """主入口函数."""
    # 提取参数
    validation_mode = args.validation_mode
    short_name = args.conv
    benchmark = args.benchmark
    
    # 根据 validation_mode 调整 conversation 名称，避免 strict/traversal 冲突
    if validation_mode != "strict":
        short_name = f"{short_name}_{validation_mode}"

    # 提取 queries from short name
    prefix = "basef"
    if not short_name.startswith(prefix) and not short_name.startswith(f"basef_{validation_mode}"):
        # 处理带 mode 后缀的名称
        base_short_name = short_name.replace(f"_{validation_mode}", "")
    else:
        base_short_name = short_name
        


    if "v" not in base_short_name:
        raise ValueError(
            f"Cannot parse query ids from short name {short_name}. "
            "Expected format like 'basef1-9v1'."
        )
    query_ids = parse_query_ids(base_short_name, prefix, benchmark=benchmark)
    if query_ids is None:
        raise ValueError(
            f"Could not parse query ids from short name {short_name}. "
            "Expected format like 'basef1-9v1'."
        )

    max_scale_factor = get_default_benchmark_scale_factor(benchmark)
    verify_sf_list = [max_scale_factor]
    
    artifacts_dir = getattr(args, "artifacts_dir", DEFAULT_TPCH_MONETDB_ARTIFACTS_DIR)
    if artifacts_dir is None:
        artifacts_dir = DEFAULT_TPCH_MONETDB_ARTIFACTS_DIR

    base_data_dir = Path(
        getattr(
            args,
            "base_data_dir",
            f"{artifacts_dir}/{benchmark}_data",
        )
    )
    
    logger.info(
        "Scripted startup: benchmark=%s, validation_mode=%s, verify_sf_list=%s",
        benchmark,
        validation_mode,
        verify_sf_list,
    )

    storage_plan_snapshot = getattr(args, "storage_plan_snapshot", None)
    is_bespoke_storage = True

    config = build_run_config(
        benchmark=benchmark,
        conv_name=short_name,
        conv_mode="scripted",
        query_list=",".join(map(str, query_ids)),
        notify=args.notify,
        disable_repo_sync=args.disable_repo_sync,
        max_scale_factor=max_scale_factor,
        replay_cache=args.replay_cache,
        storage_plan_snapshot=storage_plan_snapshot,
        keep_csv=True,
        disable_tracing=args.disable_tracing,
        disable_wandb=args.disable_wandb,
        auto_u=args.auto_u,
        auto_finish=args.auto_finish,
        is_bespoke_storage=is_bespoke_storage,
        replay=args.replay,
        model=args.model,
        reasoning_effort=getattr(args, "reasoning_effort", None),
        base_data_dir=str(base_data_dir),
        artifacts_dir=artifacts_dir,
        disable_wandb_when_tracing_disabled=getattr(
            args,
            "disable_wandb_when_tracing_disabled",
            False,
        ),
        wandb_init_max_attempts=getattr(args, "wandb_init_max_attempts", 3),
        wandb_init_timeout_s=getattr(args, "wandb_init_timeout_s", 30.0),
        wandb_upload_timeout_s=getattr(args, "wandb_upload_timeout_s", 120.0),
        wandb_finish_timeout_s=getattr(args, "wandb_finish_timeout_s", 30.0),
        wandb_finish_retries=getattr(args, "wandb_finish_retries", 1),
        enable_auto_compact=True,  # Scripted 路径默认开启 auto_compact
        only_from_llm_cache=getattr(args, "only_from_llm_cache", False),
        only_from_cache=getattr(args, "only_from_cache", False),
        target_cpu=getattr(args, "target_cpu", None),
        hardware_counter_backend=getattr(args, "hardware_counter_backend", None),
        hardware_counter_runner_cmd=getattr(args, "hardware_counter_runner_cmd", None),
        host_kernel=getattr(args, "host_kernel", None),
        perf_event_paranoid=getattr(args, "perf_event_paranoid", None),
        large_sf=getattr(args, "large_sf", None),
        stream_llm=getattr(args, "stream_llm", False),
    )
    config.generate_design_evidence = False
    config.validation_mode = validation_mode

    sample_query_args_dict: Dict[str, str] = {}

    # create conversation
    create_conversation(
        short_name,
        query_ids,
        verify_sf_list=verify_sf_list,
        max_scale_factor=max_scale_factor,
        artifacts_dir=Path(artifacts_dir),
        conversation_dir=Path(artifacts_dir) / "conversations",
        benchmark=benchmark,
        sample_query_args_dict=sample_query_args_dict,
        base_data_dir=base_data_dir,
        validation_mode=validation_mode,
        storage_plan_snapshot=storage_plan_snapshot,
    )

    # run conversation
    run_conv_wrapper(config)
    return None


def append_prompt_step(
    prompt_list: list[object],
    text: str,
    max_turns: Optional[int] = None,
    descriptor: Optional[str] = None,
    tool_profile: Optional[str] = None,
    rule_area: Optional[str] = None,
    required_nonempty_files: Optional[list[str]] = None,
    required_updated_files: Optional[list[str]] = None,
    stop_conditions: Optional[list[str]] = None,
    expected_query_id: Optional[str] = None,
    generated_code_checks: Optional[list[str]] = None,
    required_control_artifacts: Optional[list[str]] = None,
    control_artifacts_injected: Optional[list[str]] = None,
    active_unit_id: Optional[str] = None,
    active_unit_kind: Optional[str] = None,
    active_unit_files: Optional[list[str]] = None,
    active_unit_query_ids: Optional[list[str]] = None,
) -> None:
    """Append one structured prompt step to the scripted base workflow."""
    step: dict[str, object] = {"text": text}
    if max_turns is not None:
        step["max_turns"] = max_turns
    if descriptor is not None:
        step["descriptor"] = descriptor
    if tool_profile is not None:
        step["tool_profile"] = tool_profile
    if rule_area is not None:
        step["rule_area"] = rule_area
    if required_nonempty_files:
        step["required_nonempty_files"] = required_nonempty_files
    if required_updated_files:
        step["required_updated_files"] = required_updated_files
    if stop_conditions:
        step["stop_conditions"] = stop_conditions
    if expected_query_id is not None:
        step["expected_query_id"] = expected_query_id
    if generated_code_checks:
        step["generated_code_checks"] = generated_code_checks
    if required_control_artifacts:
        step["required_control_artifacts"] = required_control_artifacts
    if active_unit_id is not None:
        step["active_unit_id"] = active_unit_id
    if active_unit_kind is not None:
        step["active_unit_kind"] = active_unit_kind
    if active_unit_files:
        step["active_unit_files"] = active_unit_files
    if active_unit_query_ids:
        step["active_unit_query_ids"] = active_unit_query_ids
    resolved_injected = control_artifacts_injected
    if resolved_injected is None and required_control_artifacts:
        resolved_injected = list(required_control_artifacts)
    if resolved_injected:
        step["control_artifacts_injected"] = resolved_injected
    prompt_list.append(step)
    return None


def build_query_output_protocol() -> str:
    """Return the query output protocol prompt fragment from prompt assets."""
    return _render_base_impl_prompt("query_output_protocol.txt", {})


def _render_base_impl_prompt(
    asset_name: str,
    variables: dict[str, object],
) -> str:
    """Render one scripted base-implementation prompt asset."""
    return render_scripted_prompt_asset(
        "base_impl",
        asset_name,
        variables=variables,
    )


# ---------------------------------------------------------------------------
# Stage builder helpers
# ---------------------------------------------------------------------------

_BASE_PERFORMANCE_PROBE_QUERY_BATCH_SIZE = 5


def _format_query_scope_label(query_ids: list[str]) -> str:
    """Return a compact human-readable label for a benchmark query scope."""
    if not query_ids:
        return "the requested TPC-H query scope"
    if len(query_ids) == 1:
        return f"Q{query_ids[0]}"
    return f"Q{query_ids[0]}-Q{query_ids[-1]}"


def _append_base_performance_probe(
    prompt_list: list[object],
    *,
    descriptor: str,
    budget: dict[str, int],
    max_scale_factor: int,
    query_ids: list[str],
) -> None:
    """Append an observation-only base benchmark stage for a query scope."""
    normalized_query_ids = [str(query_id) for query_id in query_ids]
    append_prompt_step(
        prompt_list,
        _render_base_impl_prompt(
            "benchmark.txt",
            {
                "max_scale_factor": max_scale_factor,
                "query_scope_label": _format_query_scope_label(normalized_query_ids),
                "query_ids_json": json.dumps(normalized_query_ids),
            },
        ),
        max_turns=budget["benchmark"],
        descriptor=descriptor,
        tool_profile="benchmark",
        rule_area="runtime",
        active_unit_id=f"benchmark:{'-'.join(normalized_query_ids)}",
        active_unit_kind="benchmark_scope",
        active_unit_files=[
            f"query_q{query_id}.cpp" for query_id in normalized_query_ids
        ],
        active_unit_query_ids=normalized_query_ids,
        required_control_artifacts=[
            "TODO.md",
            "storage_plan.txt",
            "storage_plan_contract.json",
        ],
    )


def _append_quick_todo_sync(
    prompt_list: list[object],
    descriptor: str,
    budget: dict[str, int],
) -> None:
    """Append a fast TODO-sync stage after a query/family implementation."""
    append_prompt_step(
        prompt_list,
        _render_base_impl_prompt("todo_sync_quick.txt", {}),
        max_turns=budget.get("todo_sync_quick", 32),
        descriptor=descriptor,
        tool_profile="todo_sync",
        rule_area="runtime",
        required_updated_files=["TODO.md"],
        stop_conditions=["write_required"],
        required_control_artifacts=["TODO.md"],
    )


def _append_single_query_correctness(
    prompt_list: list[object],
    qid: str,
    budget: dict[str, int],
    sf_verify_str: str,
    unit_metadata_by_query: dict,
) -> None:
    """Append a correctness stage for a single query."""
    append_prompt_step(
        prompt_list,
        _render_base_impl_prompt(
            "single_query_correctness.txt",
            {
                "qid": qid,
                "sf_verify_str": sf_verify_str,
            },
        ),
        max_turns=budget["correctness_single_query"],
        descriptor=f"correctness_query_{qid}",
        tool_profile="correctness_queries_writeonly",
        rule_area="runtime",
        required_nonempty_files=[f"query_q{qid}.cpp"],
        stop_conditions=["validation_passed"],
        expected_query_id=qid,
        generated_code_checks=[*get_query_generated_code_checks(qid)],
        required_control_artifacts=["TODO.md", "storage_plan.txt", "storage_plan_contract.json"],
        **unit_metadata_by_query[qid],
    )


def _append_independent_query_implementation(
    prompt_list: list[object],
    qid: str,
    budget: dict[str, int],
    sf_verify_str: str,
    sample_query_args_dict: dict,
    query_output_protocol: str,
    unit_metadata_by_query: dict,
) -> None:
    """Append an implementation stage for an independent (non-family) query."""
    sample_args_str = ""
    if sample_query_args_dict and qid in sample_query_args_dict:
        sample_args_str = (
            f" Example instantiation:\n{sample_query_args_dict[qid]}\n"
        )
    codegen_hint_text = build_query_codegen_hint_text(qid)
    if codegen_hint_text:
        codegen_hint_text = f"\n{codegen_hint_text}\n"

    append_prompt_step(
        prompt_list,
        _render_base_impl_prompt(
            "independent_query_implementation.txt",
            {
                "qid": qid,
                "query_output_protocol": query_output_protocol,
                "codegen_hint_text": codegen_hint_text,
                "sample_args_str": sample_args_str,
            },
        ),
        max_turns=budget["implement_single_query"],
        descriptor=f"implement_query_{qid}",
        tool_profile="implement_queries_writeonly",
        rule_area="runtime",
        required_nonempty_files=[f"query_q{qid}.cpp"],
        expected_query_id=qid,
        generated_code_checks=[*get_query_generated_code_checks(qid)],
        required_control_artifacts=["TODO.md", "storage_plan.txt", "storage_plan_contract.json"],
        **unit_metadata_by_query[qid],
    )


def _append_family_kernel_implementation(
    prompt_list: list[object],
    family_ctx: dict,
    budget: dict[str, int],
    query_output_protocol: str,
    sample_query_args_dict: dict,
    unit_metadata_by_query: dict,
) -> None:
    """Append a family kernel implementation stage."""
    family_name = family_ctx["family_name"]
    first_qid = family_ctx["first_query_id"]
    all_qids_str = ", ".join(family_ctx["family_query_ids"])

    codegen_hint = build_query_codegen_hint_text(first_qid)
    codegen_hint_text = f"\n{codegen_hint}\n" if codegen_hint else ""

    sample_args = ""
    if sample_query_args_dict and first_qid in sample_query_args_dict:
        sample_args = (
            f"\nExample instantiation for Q{first_qid}:\n"
            f"{sample_query_args_dict[first_qid]}\n"
        )

    append_prompt_step(
        prompt_list,
        _render_base_impl_prompt(
            "family_kernel_implementation.txt",
            {
                "family_name": family_name,
                "all_qids_str": all_qids_str,
                "kernel_header": family_ctx["kernel_header"],
                "kernel_source": family_ctx["kernel_source"],
                "first_qid": first_qid,
                "query_output_protocol": query_output_protocol,
                "sample_args": sample_args,
                "codegen_hint_text": codegen_hint_text,
            },
        ),
        max_turns=budget.get("implement_family_kernel", 256),
        descriptor=f"implement_family_kernel_{family_name}",
        tool_profile="correctness_foundation",
        rule_area="runtime",
        required_nonempty_files=[
            family_ctx["kernel_header"],
            family_ctx["kernel_source"],
            f"query_q{first_qid}.cpp",
        ],
        expected_query_id=first_qid,
        generated_code_checks=[
            *get_query_generated_code_checks(first_qid),
            "query_family_boundary",
        ],
        required_control_artifacts=["TODO.md", "storage_plan.txt", "storage_plan_contract.json"],
        **unit_metadata_by_query[first_qid],
    )


def _append_family_kernel_correctness(
    prompt_list: list[object],
    family_ctx: dict,
    budget: dict[str, int],
    sf_verify_str: str,
    unit_metadata_by_query: dict,
) -> None:
    """Append a correctness stage that validates the family kernel."""
    first_qid = family_ctx["first_query_id"]
    family_name = family_ctx["family_name"]
    all_qids_str = ", ".join(family_ctx["family_query_ids"])

    append_prompt_step(
        prompt_list,
        _render_base_impl_prompt(
            "family_kernel_correctness.txt",
            {
                "family_name": family_name,
                "first_qid": first_qid,
                "sf_verify_str": sf_verify_str,
                "all_qids_str": all_qids_str,
            },
        ),
        max_turns=budget.get("correctness_family_kernel", 384),
        descriptor=f"correctness_family_kernel_{family_name}",
        tool_profile="correctness_foundation",
        rule_area="runtime",
        required_nonempty_files=[
            family_ctx["kernel_source"],
            f"query_q{first_qid}.cpp",
        ],
        stop_conditions=["validation_passed"],
        expected_query_id=first_qid,
        generated_code_checks=[
            *get_query_generated_code_checks(first_qid),
            "query_family_boundary",
        ],
        required_control_artifacts=["TODO.md", "storage_plan.txt", "storage_plan_contract.json"],
        **unit_metadata_by_query[first_qid],
    )


def _append_family_entrypoint_implementation(
    prompt_list: list[object],
    qid: str,
    family_ctx: dict,
    budget: dict[str, int],
    query_output_protocol: str,
    unit_metadata_by_query: dict,
) -> None:
    """Append a thin entrypoint implementation stage for a family member query."""
    family_name = family_ctx["family_name"]

    append_prompt_step(
        prompt_list,
        _render_base_impl_prompt(
            "family_entrypoint_implementation.txt",
            {
                "qid": qid,
                "family_name": family_name,
                "kernel_source": family_ctx["kernel_source"],
                "query_output_protocol": query_output_protocol,
            },
        ),
        max_turns=budget.get("implement_family_entrypoint", 48),
        descriptor=f"implement_entrypoint_q{qid}",
        tool_profile="implement_queries_writeonly",
        rule_area="runtime",
        required_nonempty_files=[f"query_q{qid}.cpp"],
        expected_query_id=qid,
        generated_code_checks=[*get_query_generated_code_checks(qid)],
        required_control_artifacts=["TODO.md"],
        **unit_metadata_by_query[qid],
    )


def create_conversation(
    short_name,
    query_ids,
    verify_sf_list: List[int],
    max_scale_factor: int,
    artifacts_dir: Path,
    benchmark: str,
    conversation_dir: Path,
    sample_query_args_dict: Optional[Dict[str, str]] = None,
    base_data_dir: Path | None = None,
    validation_mode: str = "strict",
    storage_plan_snapshot: str | None = None,
) -> None:
    """Create the structured base-generation workflow for the TPC-H MonetDB agent."""
    conversation_dir.mkdir(parents=True, exist_ok=True)
    write_manifest_for_conversation(
        conversation_dir,
        benchmark=benchmark,
        conversation_name=short_name,
        query_ids=query_ids,
        storage_plan_snapshot=storage_plan_snapshot,
    )
    prompt_list: list[object] = []
    stage_turn_budget = DEFAULT_STAGE_TURN_BUDGET

    # assemble sf verify string
    if len(verify_sf_list) == 1:
        sf_verify_str = str(verify_sf_list[0])
    elif len(verify_sf_list) == 2:
        sf_verify_str = f"{verify_sf_list[0]} and {verify_sf_list[1]}"
    else:
        sf_verify_str = (
            ", ".join(map(str, verify_sf_list[:-1])) + f", and {verify_sf_list[-1]}"
        )

    # paths
    data_root = (
        artifacts_dir / f"{benchmark}_data"
        if base_data_dir is None
        else base_data_dir
    )
    data_path = data_root / "sf<SCALE_FACTOR>"
    queries_path = "queries.txt"

    loader_path = "`loader_impl.hpp`/`loader_impl.cpp`"
    builder_path = "`builder_impl.hpp`/`builder_impl.cpp`"
    query_impl_path = (
        "`query_impl.hpp`/`query_impl.cpp` (dispatcher ABI) plus companion query "
        "source files `query_q*.cpp` / `query_q*.hpp` and shared helpers "
        "`query_shared_*.cpp` / `query_shared_*.hpp`"
    )

    with_storage_plan = storage_plan_snapshot is not None
    storage_hint = _render_base_impl_prompt(
        "storage_hint_with_plan.txt" if with_storage_plan else "storage_hint_minimal_soa.txt",
        {},
    )
    args_path = "args_parser.hpp"
    query_output_protocol = build_query_output_protocol()
    unit_query_ids = list(dict.fromkeys(query_ids))
    unit_metadata_by_query = {
        qid: build_active_unit_metadata(unit_query_ids, query_id=qid)
        for qid in unit_query_ids
    }

    # Stage 1: TODO Plan
    prompt_list.append(VALIDATE_OFF)
    todo_plan_asset = render_scripted_prompt_asset(
        "storage_plan",
        "todo_plan.txt",
        variables={},
    )
    append_prompt_step(
        prompt_list,
        _render_base_impl_prompt(
            "todo_plan_stage.txt",
            {
                "queries_path": queries_path,
                "query_count": len(query_ids),
                "query_word": "query" if len(query_ids) == 1 else "queries",
                "data_path": data_path,
                "loader_path": loader_path,
                "builder_path": builder_path,
                "query_impl_path": query_impl_path,
                "storage_hint": storage_hint,
                "args_path": args_path,
                "todo_plan_asset": todo_plan_asset,
            },
        ),
        max_turns=stage_turn_budget["todo_plan"],
        descriptor="todo_plan",
        tool_profile="todo_plan",
        rule_area="runtime",
        required_nonempty_files=["TODO.md"],
        required_updated_files=["TODO.md"],
        required_control_artifacts=["storage_plan.txt", "storage_plan_contract.json", "design_evidence.md", "data_law_contract.json", "workload_objective.json"],
    )

    finish_skeleton_asset = render_scripted_prompt_asset(
        "base_impl",
        "finish_skeleton.txt",
        variables={"qid": query_ids[0] if query_ids else "1", "unit_kind": "query"},
    )
    append_prompt_step(
        prompt_list,
        _render_base_impl_prompt(
            "finish_skeleton_stage.txt",
            {"finish_skeleton_asset": finish_skeleton_asset},
        ),
        max_turns=stage_turn_budget["finish_skeleton"],
        descriptor="finish_skeleton",
        tool_profile="finish_skeleton",
        rule_area="runtime",
        stop_conditions=["write_required"],
        required_control_artifacts=["TODO.md", "storage_plan.txt", "storage_plan_contract.json", "design_evidence.md", "data_law_contract.json", "workload_objective.json"],
    )

    append_prompt_step(
        prompt_list,
        _render_base_impl_prompt(
            "compile_fix.txt",
            {"sf_verify_str": sf_verify_str},
        ),
        max_turns=stage_turn_budget["compile_fix"],
        descriptor="compile_fix",
        tool_profile="compile_fix",
        rule_area="runtime",
        required_control_artifacts=["TODO.md", "storage_plan.txt", "storage_plan_contract.json"],
    )

    append_prompt_step(
        prompt_list,
        _render_base_impl_prompt("todo_sync.txt", {}),
        max_turns=stage_turn_budget["todo_sync"],
        descriptor="todo_sync",
        tool_profile="todo_sync",
        rule_area="runtime",
        required_nonempty_files=["TODO.md"],
        required_updated_files=["TODO.md"],
        stop_conditions=["write_required"],
        required_control_artifacts=["TODO.md"],
    )

    append_prompt_step(
        prompt_list,
        _render_base_impl_prompt("add_timings.txt", {}),
        max_turns=stage_turn_budget["add_timings"],
        descriptor="add_timings",
        tool_profile="add_timings",
        rule_area="runtime",
        required_control_artifacts=["TODO.md", "storage_plan.txt", "storage_plan_contract.json"],
    )

    # validation_mode 控制 VALIDATE_ON 的放置
    if validation_mode == "strict":
        prompt_list.append(VALIDATE_ON)
    prompt_list.append(VALIDATE_OUTPUT_STDOUT_OFF)

    # Stage 6+: Implement requested query families as vertical slices.
    requested_units = build_query_units_for_requested_queries(query_ids)
    completed_query_ids: list[str] = []
    next_performance_probe_at = _BASE_PERFORMANCE_PROBE_QUERY_BATCH_SIZE

    for unit in requested_units:
        if unit.unit_kind == "family":
            family_ctx = build_family_prompt_context(unit)
            family_name = family_ctx["family_name"]
            first_qid = family_ctx["first_query_id"]

            _append_family_kernel_implementation(
                prompt_list, family_ctx, stage_turn_budget,
                query_output_protocol, sample_query_args_dict,
                unit_metadata_by_query,
            )
            _append_family_kernel_correctness(
                prompt_list, family_ctx, stage_turn_budget,
                sf_verify_str, unit_metadata_by_query,
            )
            _append_quick_todo_sync(
                prompt_list, f"todo_sync_family_{family_name}", stage_turn_budget,
            )

            for qid in family_ctx["remaining_query_ids"]:
                _append_family_entrypoint_implementation(
                    prompt_list, qid, family_ctx, stage_turn_budget,
                    query_output_protocol, unit_metadata_by_query,
                )
                _append_single_query_correctness(
                    prompt_list, qid, stage_turn_budget,
                    sf_verify_str, unit_metadata_by_query,
                )
                _append_quick_todo_sync(
                    prompt_list, f"todo_sync_q{qid}", stage_turn_budget,
                )

            completed_query_ids.extend(family_ctx["family_query_ids"])
            while (
                len(completed_query_ids) >= next_performance_probe_at
                and next_performance_probe_at < len(unit_query_ids)
            ):
                probe_start = (
                    next_performance_probe_at - _BASE_PERFORMANCE_PROBE_QUERY_BATCH_SIZE
                )
                probe_query_ids = unit_query_ids[probe_start:next_performance_probe_at]
                probe_label = (
                    _format_query_scope_label(probe_query_ids).lower().replace("-", "_")
                )
                _append_base_performance_probe(
                    prompt_list,
                    descriptor=f"base_perf_probe_{probe_label}",
                    budget=stage_turn_budget,
                    max_scale_factor=max_scale_factor,
                    query_ids=probe_query_ids,
                )
                _append_quick_todo_sync(
                    prompt_list,
                    f"todo_sync_perf_{probe_label}",
                    stage_turn_budget,
                )
                next_performance_probe_at += _BASE_PERFORMANCE_PROBE_QUERY_BATCH_SIZE

            continue

        qid = unit.query_ids[0]
        _append_independent_query_implementation(
            prompt_list, qid, stage_turn_budget,
            sf_verify_str, sample_query_args_dict,
            query_output_protocol, unit_metadata_by_query,
        )
        _append_single_query_correctness(
            prompt_list, qid, stage_turn_budget,
            sf_verify_str, unit_metadata_by_query,
        )
        _append_quick_todo_sync(
            prompt_list, f"todo_sync_q{qid}", stage_turn_budget,
        )
        completed_query_ids.extend(unit.query_ids)
        while (
            len(completed_query_ids) >= next_performance_probe_at
            and next_performance_probe_at < len(unit_query_ids)
        ):
            probe_start = (
                next_performance_probe_at - _BASE_PERFORMANCE_PROBE_QUERY_BATCH_SIZE
            )
            probe_query_ids = unit_query_ids[probe_start:next_performance_probe_at]
            probe_label = (
                _format_query_scope_label(probe_query_ids).lower().replace("-", "_")
            )
            _append_base_performance_probe(
                prompt_list,
                descriptor=f"base_perf_probe_{probe_label}",
                budget=stage_turn_budget,
                max_scale_factor=max_scale_factor,
                query_ids=probe_query_ids,
            )
            _append_quick_todo_sync(
                prompt_list,
                f"todo_sync_perf_{probe_label}",
                stage_turn_budget,
            )
            next_performance_probe_at += _BASE_PERFORMANCE_PROBE_QUERY_BATCH_SIZE

    # Check all queries correctness
    append_prompt_step(
        prompt_list,
        _render_base_impl_prompt(
            "all_queries_correctness.txt",
            {"sf_verify_str": sf_verify_str},
        ),
        max_turns=stage_turn_budget["all_queries_correctness"],
        descriptor="all_queries_correctness",
        tool_profile="correctness",
        rule_area="runtime",
        stop_conditions=["validation_passed"],
        required_control_artifacts=["TODO.md", "storage_plan.txt", "storage_plan_contract.json"],
    )

    _append_quick_todo_sync(prompt_list, "todo_sync_final", stage_turn_budget)

    _append_base_performance_probe(
        prompt_list,
        descriptor="benchmark",
        budget=stage_turn_budget,
        max_scale_factor=max_scale_factor,
        query_ids=unit_query_ids,
    )
    _append_quick_todo_sync(prompt_list, "todo_sync_benchmark", stage_turn_budget)

    append_prompt_step(
        prompt_list,
        _render_base_impl_prompt(
            "optimize_build.txt",
            {
                "max_scale_factor": max_scale_factor,
                "sf_verify_str": sf_verify_str,
            },
        ),
        max_turns=stage_turn_budget["optimize_build"],
        descriptor="optimize_build",
        tool_profile="optimize_build",
        rule_area="runtime",
        required_control_artifacts=["TODO.md", "storage_plan.txt", "storage_plan_contract.json"],
    )

    target_path = conversation_dir / f"{benchmark}_{short_name}.json"

    with open(target_path, "w") as f:
        json.dump(prompt_list, f, indent=2)
    return None


def build_parser(*, add_help: bool = True) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="TPC-H MonetDB Scripted Base Implementation entry point",
        add_help=add_help,
    )
    parser.add_argument(
        "--conv",
        type=str,
        required=True,
        help="Short name for the conversation (e.g., basef1-9v1)",
    )
    parser.add_argument(
        "--validation_mode",
        type=str,
        choices=["strict", "traversal"],
        default="strict",
        help="Validation mode: strict (default, with correctness validation) or traversal (without validation)",
    )
    parser.add_argument(
        "--base_data_dir",
        type=str,
        default=None,
        help="Base directory for TPC-H MonetDB data files",
    )

    add_common_args(
        parser,
        include_model=True,
        include_reasoning_effort=True,
        include_notify=True,
        include_disable_repo_sync=True,
        include_replay_cache=True,
        include_benchmark=True,
        include_disable_wandb=True,
        include_disable_tracing=True,
        include_disable_wandb_when_tracing_disabled=True,
        include_wandb_init_max_attempts=True,
        include_wandb_init_timeout_s=True,
        include_wandb_upload_timeout_s=True,
        include_wandb_finish_timeout_s=True,
        include_wandb_finish_retries=True,
        include_auto_u=True,
        include_auto_finish=True,
        include_replay=True,
        include_only_from_llm_cache=True,
        include_only_from_cache=True,
        include_artifacts_dir=True,
        include_storage_plan_snapshot=True,
        include_is_bespoke_storage=True,
        include_enable_auto_compact=True,
        include_target_cpu=True,
        include_hardware_counter_backend=True,
        include_hardware_counter_runner_cmd=True,
        include_host_kernel=True,
        include_perf_event_paranoid=True,
        include_large_sf=True,
        include_stream_llm=True,
    )
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    main(args)
