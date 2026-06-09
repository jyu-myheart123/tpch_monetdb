import argparse
import json
from pathlib import Path

from tpch_monetdb.bootstrap_env import bootstrap_runtime_env

bootstrap_runtime_env()

from tpch_monetdb.conversations.scripted_prompts_gen import render_scripted_prompt_asset
from tpch_monetdb.main_tpch_monetdb import run_conv_wrapper
from tpch_monetdb.dataset.dataset_tables_dict import get_benchmark_schema
from tpch_monetdb.utils.cli_config import add_common_args, build_run_config
from tpch_monetdb.utils.gen_common import parse_query_ids
from tpch_monetdb.config import (
    DEFAULT_TPCH_MONETDB_ARTIFACTS_DIR,
    get_default_benchmark_scale_factor,
    resolve_workflow_scale_factors,
)


def _build_prev_feedback_section(prev_run_report: str | None) -> str:
    """Render previous-round feedback only when a report is available."""
    if not prev_run_report:
        return ""
    return render_scripted_prompt_asset(
        "storage_plan",
        "prev_feedback_section.txt",
        variables={"prev_run_report": prev_run_report},
    )


def _render_default_scale_factors(max_scale_factor: int) -> str:
    scale_factors = resolve_workflow_scale_factors(max_scale_factor)
    return ", ".join(f"sf{sf}" for sf in scale_factors)


def _build_data_source_description(
    *,
    benchmark: str,
    base_data_dir: Path,
    supported_sfs: str,
) -> str:
    """Return the benchmark-specific storage-plan data source description."""
    normalized = benchmark.strip().lower()
    if normalized == "tpch":
        return (
            "TPC-H `.tbl` table directories at "
            f"{base_data_dir}/sf{{N}} or a tiny fixture root containing "
            f"the 8 table files ({supported_sfs})."
        )
    raise ValueError(f"Unsupported benchmark for storage planning: {benchmark}")


def _build_planning_mode_instructions(storage_plan_mode: str) -> str:
    """Render storage-planning requirements for the current outer-loop round."""
    if storage_plan_mode == "initial_candidates":
        return (
            "- Mode: initial_candidates.\n"
            "- Explore at least 3 materially different candidate layouts before selecting one.\n"
            "- Keep at least one conservative candidate and at least one hybrid/aggressive candidate.\n"
            "- Keep the full exploration in `storage_plan_candidates.json` or a compact planning summary, then select one base candidate.\n"
            "- Refine the selected base candidate against MonetDB/TPC-H output semantics, query types, `design_evidence.md`, and `workload_objective.json` before implementation.\n"
            "- Output a v2 committed `storage_plan_contract.json` with `selected_base_candidate_id`, `committed_layout`, `refinement_decisions`, `critical_query_access_paths`, `selected_layout_obligations`, query-family cost, and evidence refs.\n"
            "- Do not put full `candidate_layouts` or rejected designs in `storage_plan_contract.json`."
        )
    if storage_plan_mode == "repair_alignment":
        return (
            "- Mode: repair_alignment.\n"
            "- Do not invent a new storage architecture unless the previous selected layout is impossible.\n"
            "- Repair v2 `storage_plan_contract.json`, selected-layout obligations, critical query access paths, data-law refs, and alignment evidence so the implementation contract is machine-checkable.\n"
            "- Keep any rejected repair alternatives outside the committed contract, preferably in `storage_plan_candidates.json` only when needed for audit.\n"
            "- Do not reintroduce `candidate_layouts` or `candidates` into `storage_plan_contract.json`.\n"
            "- Do not rewrite broad candidate layouts; repair `committed_layout`, `refinement_decisions`, obligations, evidence refs, and TODO/committed-layout consistency only."
        )
    if storage_plan_mode == "delta_plan":
        return (
            "- Mode: delta_plan.\n"
            "- Use the previous-round feedback as the primary evidence and propose targeted layout deltas for the failing query families.\n"
            "- Do not restart from three broad designs; preserve the current selected layout unless a measured failure proves it is the root cause.\n"
            "- Put optional delta alternatives in `storage_plan_candidates.json` or a compact planning summary; commit only the selected delta into `committed_layout`, `refinement_decisions`, and `critical_query_access_paths`.\n"
            "- Do not add `candidate_layouts` or `candidates` to `storage_plan_contract.json`.\n"
            "- Output current committed layout, failing objective route, targeted deltas, changed/unchanged obligations, changed/unchanged critical access paths, and why the layout is not being broadly changed.\n"
            "- If the previous summary success=false and measurement is unavailable, output only instrumentation/evidence repair, vectorization evidence, or measurement-contract repair deltas."
        )
    raise ValueError(f"Unsupported storage_plan_mode: {storage_plan_mode}")


def _build_storage_plan_prompt(
    *,
    benchmark: str,
    schema: str,
    base_data_dir: Path,
    max_scale_factor: int,
    prev_run_report: str | None,
    storage_plan_mode: str,
) -> str:
    """Build the evidence-grounded storage-plan prompt for the benchmark."""
    supported_sfs = _render_default_scale_factors(max_scale_factor)
    return render_scripted_prompt_asset(
        "storage_plan",
        "storage_plan.txt",
        variables={
            "benchmark": benchmark,
            "schema": schema,
            "data_source_description": _build_data_source_description(
                benchmark=benchmark,
                base_data_dir=base_data_dir,
                supported_sfs=supported_sfs,
            ),
            "supported_sfs": supported_sfs,
            "prev_feedback_section": _build_prev_feedback_section(prev_run_report),
            "planning_mode_instructions": _build_planning_mode_instructions(
                storage_plan_mode
            ),
        },
    )


def main(args: argparse.Namespace) -> None:
    short_name = args.conv
    benchmark = args.benchmark

    prefix = "storageplan"
    assert short_name.startswith(prefix), (
        f"Expected conv name starting with '{prefix}', got '{short_name}'"
    )
    if "v" in short_name:
        query_ids = parse_query_ids(short_name, prefix, benchmark=benchmark)
        assert query_ids is not None, f"Failed to parse query ids from {short_name}"

    max_scale_factor = get_default_benchmark_scale_factor(benchmark)
    artifacts_dir = getattr(args, "artifacts_dir", DEFAULT_TPCH_MONETDB_ARTIFACTS_DIR)
    if artifacts_dir is None:
        artifacts_dir = DEFAULT_TPCH_MONETDB_ARTIFACTS_DIR

    base_data_dir = Path(
        getattr(args, "base_data_dir", None)
        or f"{artifacts_dir}/{benchmark}_data"
    )

    # Load optional bottleneck report from previous outer-loop round
    prev_run_report: str | None = None
    prev_run_report_path = getattr(args, "prev_run_report", None)
    if prev_run_report_path:
        report_path = Path(prev_run_report_path)
        if report_path.exists():
            prev_run_report = report_path.read_text()
        else:
            import logging
            logging.getLogger(__name__).warning(
                "prev_run_report path does not exist: %s", prev_run_report_path
            )

    config = build_run_config(
        benchmark=benchmark,
        conv_name=short_name,
        query_list=",".join(map(str, query_ids)),
        notify=args.notify,
        conv_mode="scripted",
        start_snapshot=getattr(args, "start_snapshot", None),
        disable_repo_sync=args.disable_repo_sync,
        max_scale_factor=max_scale_factor,
        replay_cache=args.replay_cache,
        auto_u=args.auto_u,
        auto_finish=args.auto_finish,
        artifacts_dir=artifacts_dir,
        base_data_dir=str(base_data_dir),
        model=args.model,
        reasoning_effort=getattr(args, "reasoning_effort", None),
        disable_wandb=args.disable_wandb,
        disable_tracing=args.disable_tracing,
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
        target_cpu=getattr(args, "target_cpu", None),
        hardware_counter_backend=getattr(args, "hardware_counter_backend", None),
        hardware_counter_runner_cmd=getattr(args, "hardware_counter_runner_cmd", None),
        host_kernel=getattr(args, "host_kernel", None),
        perf_event_paranoid=getattr(args, "perf_event_paranoid", None),
        large_sf=getattr(args, "large_sf", None),
        stream_llm=getattr(args, "stream_llm", False),
    )
    config.generate_design_evidence = False

    create_conversation(
        benchmark,
        short_name,
        conversation_dir=Path(config.artifacts_dir) / "conversations",
        base_data_dir=base_data_dir,
        max_scale_factor=max_scale_factor,
        query_ids=query_ids,
        prev_run_report=prev_run_report,
        storage_plan_mode=getattr(args, "storage_plan_mode", "initial_candidates"),
    )

    run_conv_wrapper(config)
    return None


def create_conversation(
    benchmark: str,
    short_name: str,
    conversation_dir: Path,
    base_data_dir: Path,
    max_scale_factor: int,
    query_ids: list[str],
    prev_run_report: str | None = None,
    storage_plan_mode: str = "initial_candidates",
) -> None:
    """Build the storage-plan conversation JSON.

    Args:
        prev_run_report: Optional bottleneck report from the previous outer-loop
            round, injected at the end of the prompt for round > 1.
    """
    prompt_list: list[object] = []

    schema = get_benchmark_schema(benchmark)
    storage_plan_prompt = _build_storage_plan_prompt(
        benchmark=benchmark,
        schema=schema,
        base_data_dir=base_data_dir,
        max_scale_factor=max_scale_factor,
        prev_run_report=prev_run_report,
        storage_plan_mode=storage_plan_mode,
    )

    prompt_list.append({
        "text": storage_plan_prompt,
        "max_turns": 96,
        "descriptor": "storage_plan",
        "tool_profile": "storage_plan",
        "rule_area": "runtime",
        "required_nonempty_files": ["storage_plan.txt", "storage_plan_contract.json"],
        "required_updated_files": ["storage_plan.txt", "storage_plan_contract.json"],
        "required_control_artifacts": [
            "workload_objective.json",
            "data_law_contract.json",
            "design_evidence.md",
        ],
        "control_artifacts_injected": [
            "workload_objective.json",
            "data_law_contract.json",
            "design_evidence.md",
        ],
        "advisory_postconditions": ["storage_plan_contract_complete"],
        "stop_conditions": ["write_required"],
    })

    conversation_dir.mkdir(parents=True, exist_ok=True)
    target_path = conversation_dir / f"{benchmark}_{short_name}.json"

    with open(target_path, "w") as f:
        json.dump(prompt_list, f, indent=2)
    return None


def build_parser(*, add_help: bool = True) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=add_help)
    parser.add_argument(
        "--conv",
        type=str,
        required=True,
        help="Short name for the conversation",
    )
    parser.add_argument(
        "--base_data_dir",
        type=str,
        default=None,
        help="Base directory for TPC-H MonetDB data files",
    )
    parser.add_argument(
        "--prev_run_report",
        type=str,
        default=None,
        help="Path to bottleneck report from the previous outer-loop round (round > 1)",
    )
    parser.add_argument(
        "--storage_plan_mode",
        choices=["initial_candidates", "repair_alignment", "delta_plan"],
        default="initial_candidates",
        help="Storage-plan generation mode for the current outer-loop round.",
    )

    add_common_args(
        parser,
        include_model=True,
        include_reasoning_effort=True,
        include_notify=True,
        include_disable_repo_sync=True,
        include_replay_cache=True,
        include_start_snapshot=True,
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
        include_artifacts_dir=True,
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
