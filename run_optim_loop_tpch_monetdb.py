"""TPC-H MonetDB Optimization Loop 入口脚本.

便捷入口，组装 conv_mode="optimization" 参数后委托给 tpch_monetdb/main_tpch_monetdb.py。
支持自动发现 scripted run 的 snapshot。

运行示例:
  python run_optim_loop_tpch_monetdb.py --conv runoptim1-9v1 --auto_u --auto_finish
  
  # 或使用显式 snapshot
  python run_optim_loop_tpch_monetdb.py --conv runoptim1-9v1 --start_snapshot <hash> --auto_u --auto_finish
"""

import argparse
from pathlib import Path

from tpch_monetdb.bootstrap_env import bootstrap_runtime_env

bootstrap_runtime_env()

from tpch_monetdb.config import (
    DEFAULT_TPCH_MONETDB_ARTIFACTS_DIR,
    get_default_benchmark_scale_factor,
)
from tpch_monetdb.main_tpch_monetdb import run_conv_wrapper
from tpch_monetdb.utils.cli_config import add_common_args, build_run_config
from tpch_monetdb.utils.gen_common import parse_query_ids
from tpch_monetdb.utils.scripted_summary import auto_discover_start_snapshot


def main(args: argparse.Namespace) -> None:
    """组装运行配置并启动 optimization conversation."""
    bespoke_storage = True
    short_name = args.conv
    benchmark = getattr(args, "benchmark", "tpch")

    prefix = "runoptim"
    assert short_name.startswith(prefix), (
        f"Conversation name must start with '{prefix}', got '{short_name}'"
    )

    query_ids = parse_query_ids(short_name, prefix, benchmark=benchmark)
    assert query_ids is not None, (
        f"Could not parse query ids from short name {short_name}"
    )

    max_scale_factor = get_default_benchmark_scale_factor(benchmark)
    artifacts_dir = Path(
        getattr(args, "artifacts_dir", DEFAULT_TPCH_MONETDB_ARTIFACTS_DIR)
        or DEFAULT_TPCH_MONETDB_ARTIFACTS_DIR
    )

    start_snapshot = getattr(args, "start_snapshot", None)
    
    if start_snapshot is None:
        try:
            start_snapshot = auto_discover_start_snapshot(
                conv_name=None,
                query_list=query_ids,
                benchmark=benchmark,
                artifacts_dir=artifacts_dir,
                explicit_snapshot=None,
                is_bespoke_storage=bespoke_storage,
            )
        except ValueError as e:
            raise ValueError(
                f"{e}\n\n"
                f"To start optimization, you need either:\n"
                f"1. A successful strict scripted run for queries {query_ids}\n"
                f"2. Or provide --start_snapshot <hash> explicitly"
            )

    config = build_run_config(
        benchmark=benchmark,
        conv_name=short_name,
        conv_mode="optimization",
        query_list=",".join(map(str, query_ids)),
        notify=args.notify,
        disable_repo_sync=args.disable_repo_sync,
        max_scale_factor=max_scale_factor,
        replay_cache=args.replay_cache,
        start_snapshot=start_snapshot,
        storage_plan_snapshot=None,
        keep_csv=True,
        disable_tracing=args.disable_tracing,
        disable_wandb=args.disable_wandb,
        auto_u=args.auto_u,
        auto_finish=args.auto_finish,
        is_bespoke_storage=bespoke_storage,
        run_tool_offer_trace_option=True,
        only_from_llm_cache=args.only_from_llm_cache,
        only_from_cache=args.only_from_cache,
        model=args.model,
        reasoning_effort=getattr(args, "reasoning_effort", None),
        enable_auto_compact=args.enable_auto_compact,
        compaction_model_map=getattr(args, "compaction_model_map", None),
        artifacts_dir=str(artifacts_dir),
        base_data_dir=getattr(args, "base_data_dir", None),
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
        baseline_backend=None,
        baseline_query_file_dir=None,
        benchmark_mode=getattr(args, "benchmark_mode", "system-parity"),
        storage_mode=getattr(args, "storage_mode", "persistent"),
        target_cpu=getattr(args, "target_cpu", None),
        hardware_counter_backend=getattr(args, "hardware_counter_backend", None),
        hardware_counter_runner_cmd=getattr(args, "hardware_counter_runner_cmd", None),
        host_kernel=getattr(args, "host_kernel", None),
        perf_event_paranoid=getattr(args, "perf_event_paranoid", None),
        large_sf=getattr(args, "large_sf", None),
        baseline_max_age_seconds=getattr(args, "baseline_max_age_seconds", None),
        stream_llm=getattr(args, "stream_llm", False),
    )

    run_conv_wrapper(config)


def build_parser(*, add_help: bool = True) -> argparse.ArgumentParser:
    """构建命令行参数解析器."""
    parser = argparse.ArgumentParser(
        description="TPC-H MonetDB Optimization Loop entry point",
        add_help=add_help,
    )
    parser.add_argument(
        "--conv",
        type=str,
        required=True,
        help="Short name for the conversation (e.g. runoptim1-9v1)",
    )
    parser.add_argument(
        "--bespoke_storage",
        action="store_true",
        default=True,
        help="Deprecated compatibility flag; TPC-H MonetDB optimization is always storage-enabled.",
    )
    parser.add_argument(
        "--start_snapshot",
        type=str,
        default=None,
        help="Git commit hash to start optimization from (optional, will auto-discover from scripted runs)",
    )

    add_common_args(
        parser,
        include_notify=True,
        include_disable_repo_sync=True,
        include_replay_cache=True,
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
        include_only_from_llm_cache=True,
        include_base_data_dir=True,
        include_only_from_cache=True,
        include_model=True,
        include_reasoning_effort=True,
        include_benchmark=True,
        include_enable_auto_compact=True,
        include_artifacts_dir=True,
        include_benchmark_mode=True,
        include_storage_mode=True,
        include_target_cpu=True,
        include_hardware_counter_backend=True,
        include_hardware_counter_runner_cmd=True,
        include_host_kernel=True,
        include_perf_event_paranoid=True,
        include_large_sf=True,
        include_baseline_max_age_seconds=True,
        include_stream_llm=True,
    )
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    main(args)
