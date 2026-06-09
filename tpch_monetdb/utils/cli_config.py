from __future__ import annotations

import argparse

DEFAULT_MODEL = "gpt-5.2-codex"
DEFAULT_ARTIFACTS_DIR = "/mnt/labstore/bespoke_olap/"
DEFAULT_PARQUET_DIR = "/mnt/labstore/bespoke_olap/"
DEFAULT_WANDB_INIT_MAX_ATTEMPTS = 3
DEFAULT_WANDB_INIT_TIMEOUT_S = 30.0
DEFAULT_WANDB_UPLOAD_TIMEOUT_S = 120.0
DEFAULT_WANDB_FINISH_TIMEOUT_S = 30.0
DEFAULT_WANDB_FINISH_RETRIES = 1


def build_run_config(
    *,
    benchmark: str,
    conv_name: str,
    query_list: str,
    notify: bool,
    conv_mode: str,
    start_snapshot: str | None = None,
    storage_plan_snapshot: str | None = None,
    max_scale_factor: int | None = None,
    continue_run: bool = False,
    replay: bool = False,
    disable_tracing: bool = False,
    disable_wandb: bool = False,
    disable_wandb_when_tracing_disabled: bool = False,
    wandb_init_max_attempts: int = DEFAULT_WANDB_INIT_MAX_ATTEMPTS,
    wandb_init_timeout_s: float = DEFAULT_WANDB_INIT_TIMEOUT_S,
    wandb_upload_timeout_s: float = DEFAULT_WANDB_UPLOAD_TIMEOUT_S,
    wandb_finish_timeout_s: float = DEFAULT_WANDB_FINISH_TIMEOUT_S,
    wandb_finish_retries: int = DEFAULT_WANDB_FINISH_RETRIES,
    model: str = DEFAULT_MODEL,
    reasoning_effort: str | None = None,
    base_parquet_dir: str = DEFAULT_PARQUET_DIR,
    base_data_dir: str | None = None,
    artifacts_dir: str = DEFAULT_ARTIFACTS_DIR,
    no_preload: bool = False,
    disable_repo_sync: bool = False,
    replay_cache: bool = False,
    keep_csv: bool = False,
    disable_valtool: bool = False,
    disable_artifacts_context: bool = False,
    artifacts_context_mode: str = "refs",
    auto_u: bool = False,
    auto_finish: bool = False,
    is_bespoke_storage: bool = True,
    run_tool_offer_trace_option: bool = False,
    only_from_llm_cache: bool = False,
    only_from_cache: bool = False,
    enable_auto_compact: bool = False,
    compaction_model_map: dict[str, str] | None = None,
    baseline_backend: str | None = None,
    baseline_query_file_dir: str | None = None,
    benchmark_mode: str = "system-parity",
    storage_mode: str = "persistent",
    target_cpu: str | None = None,
    hardware_counter_backend: str | None = None,
    hardware_counter_runner_cmd: str | None = None,
    host_kernel: str | None = None,
    perf_event_paranoid: str | None = None,
    large_sf: int | None = None,
    baseline_max_age_seconds: int | None = None,
    stream_llm: bool = False,
) -> argparse.Namespace:
    """Build the normalized runtime config shared by TPC-H MonetDB entrypoints."""
    assert not conv_name.startswith(f"{benchmark}_"), (
        f"conv_name '{conv_name}' should not be prefixed with benchmark name '{benchmark}_'. "
        "We will add it automatically."
    )
    (
        resolved_baseline_backend,
        resolved_baseline_query_file_dir,
    ) = resolve_baseline_config_fields(
        benchmark=benchmark,
        baseline_backend=baseline_backend,
        baseline_query_file_dir=baseline_query_file_dir,
    )
    prefixed_conv_name = f"{benchmark}_{conv_name}"
    return argparse.Namespace(
        benchmark=benchmark,
        conv_name=prefixed_conv_name,
        query_list=query_list,
        continue_run=continue_run,
        replay=replay,
        disable_tracing=disable_tracing,
        disable_wandb=disable_wandb,
        disable_wandb_when_tracing_disabled=disable_wandb_when_tracing_disabled,
        wandb_init_max_attempts=wandb_init_max_attempts,
        wandb_init_timeout_s=wandb_init_timeout_s,
        wandb_upload_timeout_s=wandb_upload_timeout_s,
        wandb_finish_timeout_s=wandb_finish_timeout_s,
        wandb_finish_retries=wandb_finish_retries,
        model=model,
        reasoning_effort=reasoning_effort,
        artifacts_dir=artifacts_dir,
        no_preload=no_preload,
        notify=notify,
        start_snapshot=start_snapshot,
        storage_plan_snapshot=storage_plan_snapshot,
        max_scale_factor=max_scale_factor,
        disable_repo_sync=disable_repo_sync,
        replay_cache=replay_cache,
        keep_csv=keep_csv,
        disable_valtool=disable_valtool,
        disable_artifacts_context=disable_artifacts_context,
        artifacts_context_mode=artifacts_context_mode,
        auto_u=auto_u,
        auto_finish=auto_finish,
        is_bespoke_storage=True,
        conv_mode=conv_mode,
        run_tool_offer_trace_option=run_tool_offer_trace_option,
        only_from_llm_cache=only_from_llm_cache,
        only_from_cache=only_from_cache,
        base_parquet_dir=base_parquet_dir,
        base_data_dir=base_parquet_dir if base_data_dir is None else base_data_dir,
        enable_auto_compact=enable_auto_compact,
        compaction_model_map=compaction_model_map or {},
        baseline_backend=resolved_baseline_backend,
        baseline_query_file_dir=resolved_baseline_query_file_dir,
        benchmark_mode=benchmark_mode,
        storage_mode=storage_mode,
        target_cpu=target_cpu,
        hardware_counter_backend=hardware_counter_backend,
        hardware_counter_runner_cmd=hardware_counter_runner_cmd,
        host_kernel=host_kernel,
        perf_event_paranoid=perf_event_paranoid,
        large_sf=large_sf,
        baseline_max_age_seconds=baseline_max_age_seconds,
        stream_llm=stream_llm,
    )


def resolve_baseline_config_fields(
    *,
    benchmark: str,
    baseline_backend: str | None = None,
    baseline_query_file_dir: str | None = None,
) -> tuple[str | None, str | None]:
    """Resolve generic baseline config for the selected benchmark."""
    normalized_benchmark = benchmark.strip().lower()
    if normalized_benchmark == "tpch":
        backend = baseline_backend or "monetdb"
        if backend != "monetdb":
            raise ValueError(
                "TPC-H replacement path only supports baseline_backend='monetdb'. "
                f"Got {backend!r}."
            )
        if baseline_query_file_dir is not None:
            raise ValueError(
                "TPC-H replacement path does not support baseline_query_file_dir "
                "after legacy query-file baseline removal."
            )
        return backend, None
    raise ValueError(f"Unknown benchmark: {benchmark}")


def add_common_args(
    parser: argparse.ArgumentParser,
    *,
    include_model: bool = False,
    include_reasoning_effort: bool = False,
    include_benchmark: bool = False,
    include_replay: bool = False,
    include_disable_tracing: bool = False,
    include_disable_wandb: bool = False,
    include_disable_wandb_when_tracing_disabled: bool = False,
    include_wandb_init_max_attempts: bool = False,
    include_wandb_init_timeout_s: bool = False,
    include_wandb_upload_timeout_s: bool = False,
    include_wandb_finish_timeout_s: bool = False,
    include_wandb_finish_retries: bool = False,
    include_conv_name: bool = False,
    include_query_list: bool = False,
    include_continue_run: bool = False,
    include_artifacts_dir: bool = False,
    include_no_preload: bool = False,
    include_notify: bool = False,
    include_start_snapshot: bool = False,
    include_storage_plan_snapshot: bool = False,
    start_snapshot_required: bool = False,
    include_disable_repo_sync: bool = False,
    include_replay_cache: bool = False,
    include_auto_u: bool = False,
    include_auto_finish: bool = False,
    include_keep_csv: bool = False,
    include_disable_valtool: bool = False,
    include_disable_artifacts_context: bool = False,
    include_artifacts_context_mode: bool = False,
    include_conv_mode: bool = False,
    include_run_tool_offer_trace_option: bool = False,
    include_is_bespoke_storage: bool = False,
    include_only_from_llm_cache: bool = False,
    include_base_parquet_dir: bool = False,
    include_base_data_dir: bool = False,
    include_only_from_cache: bool = False,
    include_enable_auto_compact: bool = False,
    include_baseline_backend: bool = False,
    include_baseline_query_file_dir: bool = False,
    include_benchmark_mode: bool = False,
    include_storage_mode: bool = False,
    include_target_cpu: bool = False,
    include_hardware_counter_backend: bool = False,
    include_hardware_counter_runner_cmd: bool = False,
    include_host_kernel: bool = False,
    include_perf_event_paranoid: bool = False,
    include_large_sf: bool = False,
    include_baseline_max_age_seconds: bool = False,
    include_stream_llm: bool = False,
) -> None:
    """Attach common CLI arguments used by TPC-H MonetDB runtime entrypoints."""
    if include_model:
        parser.add_argument(
            "--model",
            default=DEFAULT_MODEL,
            help="Model ID to use for the agent.",
        )
    if include_reasoning_effort:
        parser.add_argument(
            "--reasoning_effort",
            choices=["none", "minimal", "low", "medium", "high", "xhigh"],
            default=None,
            help="Reasoning effort to request from models that support it.",
        )
    if include_benchmark:
        parser.add_argument(
            "--benchmark",
            choices=["tpch"],
            default="tpch",
            help="Benchmark to use for the agent.",
        )
    if include_replay:
        parser.add_argument(
            "--replay",
            action="store_true",
            default=False,
            help="Replay previous conversation if set.",
        )
    if include_disable_tracing:
        parser.add_argument(
            "--disable_tracing",
            action="store_true",
            default=False,
            help="Disable tracing if set.",
        )
    if include_disable_wandb:
        parser.add_argument(
            "--disable_wandb",
            action="store_true",
            default=False,
            help="Disable wandb if set.",
        )
    if include_disable_wandb_when_tracing_disabled:
        parser.add_argument(
            "--disable_wandb_when_tracing_disabled",
            action="store_true",
            default=False,
            help="When tracing is disabled, also disable W&B init/log/finish.",
        )
    if include_wandb_init_max_attempts:
        parser.add_argument(
            "--wandb_init_max_attempts",
            type=int,
            default=DEFAULT_WANDB_INIT_MAX_ATTEMPTS,
            help="Maximum retry attempts for W&B init.",
        )
    if include_wandb_init_timeout_s:
        parser.add_argument(
            "--wandb_init_timeout_s",
            type=float,
            default=DEFAULT_WANDB_INIT_TIMEOUT_S,
            help="Timeout in seconds for each W&B init attempt. <=0 disables timeout.",
        )
    if include_wandb_upload_timeout_s:
        parser.add_argument(
            "--wandb_upload_timeout_s",
            type=float,
            default=DEFAULT_WANDB_UPLOAD_TIMEOUT_S,
            help="Timeout in seconds for W&B workspace code upload. <=0 disables timeout.",
        )
    if include_wandb_finish_timeout_s:
        parser.add_argument(
            "--wandb_finish_timeout_s",
            type=float,
            default=DEFAULT_WANDB_FINISH_TIMEOUT_S,
            help="Timeout in seconds for each W&B finish attempt. <=0 disables timeout.",
        )
    if include_wandb_finish_retries:
        parser.add_argument(
            "--wandb_finish_retries",
            type=int,
            default=DEFAULT_WANDB_FINISH_RETRIES,
            help="Retry count for W&B finish after the initial attempt.",
        )
    if include_conv_name:
        parser.add_argument(
            "--conv_name",
            help="Name of conversation.",
            required=True,
        )
    if include_query_list:
        parser.add_argument(
            "--query_list",
            help="Comma-separated list of queries.",
            required=True,
        )
    if include_continue_run:
        parser.add_argument(
            "--continue_run",
            action="store_true",
            default=False,
            help="Continue with the current snapshot in the working-dir. Does not start empty.",
        )
    if include_artifacts_dir:
        parser.add_argument(
            "--artifacts_dir",
            type=str,
            default=DEFAULT_ARTIFACTS_DIR,
            help="Directory to store artifacts like logs.",
        )
    if include_no_preload:
        parser.add_argument(
            "--no_preload",
            action="store_true",
            default=False,
            help="Skip validate tool preloading",
        )
    if include_notify:
        parser.add_argument(
            "--notify",
            action="store_true",
            default=False,
            help="Notify when conversation requires action",
        )
    if include_start_snapshot:
        parser.add_argument(
            "--start_snapshot",
            type=str,
            default=None,
            required=start_snapshot_required,
            help="Path to snapshot to start from (if not continuing current snapshot).",
        )
    if include_base_parquet_dir:
        parser.add_argument(
            "--base_parquet_dir",
            type=str,
            default=DEFAULT_PARQUET_DIR,
            help="Base parquet directory.",
        )
    if include_base_data_dir:
        parser.add_argument(
            "--base_data_dir",
            type=str,
            default=DEFAULT_PARQUET_DIR,
            help="Base data directory.",
        )
    if include_storage_plan_snapshot:
        parser.add_argument(
            "--storage_plan_snapshot",
            type=str,
            default=None,
            help="Path to snapshot to load storage plan from (incompatible with --continue_run).",
        )
    if include_disable_repo_sync:
        parser.add_argument(
            "--disable_repo_sync",
            action="store_true",
            default=False,
            help="Disable syncing snapshots with the cache repo.",
        )
    if include_replay_cache:
        parser.add_argument(
            "--replay_cache",
            action="store_true",
            default=False,
            help="Auto press 'u' until first non-cached LLM call",
        )
    if include_auto_u:
        parser.add_argument(
            "--auto_u",
            action="store_true",
            default=False,
            help="Auto press 'u' for all prompts (skip user interaction, and auto-approve all prompts). This is dangerous and might lead to large bills / unwanted changes / ... Huge caution advised.",
        )
    if include_auto_finish:
        parser.add_argument(
            "--auto_finish",
            action="store_true",
            default=False,
            help="Automatically finish if no more prompt is found in conversation / i.e. Str-D in last iteration",
        )
    if include_keep_csv:
        parser.add_argument(
            "--keep_csv",
            action="store_true",
            default=False,
            help="Keep csv if set.",
        )
    if include_disable_valtool:
        parser.add_argument(
            "--disable_valtool",
            action="store_true",
            default=False,
            help="Disable validate tool if set",
        )
    if include_disable_artifacts_context:
        parser.add_argument(
            "--disable_artifacts_context",
            action="store_true",
            default=False,
            help="Do not include workspace artifacts in cache hashing.",
        )
    if include_artifacts_context_mode:
        parser.add_argument(
            "--artifacts_context_mode",
            choices=("refs", "full", "off"),
            default="refs",
            help="Control generated artifact references included in prompts/cache hashing.",
        )
    if include_conv_mode:
        parser.add_argument(
            "--conv_mode",
            type=str,
            default="scripted",
            help="Conversation mode to use for the agent. E.g. 'scripted', 'optimization', ...",
        )
    if include_run_tool_offer_trace_option:
        parser.add_argument(
            "--run_tool_offer_trace_option",
            action="store_true",
            default=False,
            help="Whether to include trace options in the run tool (and consequently offer the option to enable tracing in the conversation). This is needed for collecting execution traces for training data generation.",
        )
    if include_is_bespoke_storage:
        parser.add_argument(
            "--is_bespoke_storage",
            action="store_true",
            default=True,
            help="Deprecated compatibility flag; TPC-H MonetDB runs are always storage-enabled.",
        )
    if include_only_from_llm_cache:
        parser.add_argument(
            "--only_from_llm_cache",
            action="store_true",
            default=False,
            help="Only answer from LLM cache and do not call the LLM. Will raise an error if a cache miss occurs.",
        )
    if include_only_from_cache:
        parser.add_argument(
            "--only_from_cache",
            action="store_true",
            default=False,
            help="Only answer from cache (including both LLM cache and run tool cache) and do not call the LLM or run tool. Will raise an error if a cache miss occurs.",
        )
    if include_enable_auto_compact:
        parser.add_argument(
            "--enable_auto_compact",
            action="store_true",
            default=False,
            help="Enable automatic context compaction when token usage exceeds threshold.",
        )
    if include_baseline_backend:
        parser.add_argument(
            "--baseline_backend",
            choices=["monetdb"],
            default=None,
            help="Baseline backend. The replacement path only supports monetdb.",
        )
    if include_baseline_query_file_dir:
        parser.add_argument(
            "--baseline_query_file_dir",
            type=str,
            default=None,
            help="Unsupported legacy query-file baseline directory.",
        )
    if include_benchmark_mode:
        parser.add_argument(
            "--benchmark_mode",
            choices=["query-latency", "system-parity"],
            default="system-parity",
            help="Phase9 benchmark mode label used for aggregation guardrails.",
        )
    if include_storage_mode:
        parser.add_argument(
            "--storage_mode",
            choices=["tmpfs", "persistent"],
            default="persistent",
            help="Storage mode label used for aggregation guardrails.",
        )
    if include_target_cpu:
        parser.add_argument(
            "--target_cpu",
            type=str,
            default=None,
            help="Named target CPU for PMU/vectorization acceptance.",
        )
    if include_hardware_counter_backend:
        parser.add_argument(
            "--hardware_counter_backend",
            type=str,
            default=None,
            help="Explicit hardware-counter backend; no automatic fallback is allowed.",
        )
    if include_hardware_counter_runner_cmd:
        parser.add_argument(
            "--hardware_counter_runner_cmd",
            type=str,
            default=None,
            help="Explicit local perf command for PMU collection.",
        )
    if include_host_kernel:
        parser.add_argument(
            "--host_kernel",
            type=str,
            default=None,
            help="Host kernel string recorded in PMU provenance.",
        )
    if include_perf_event_paranoid:
        parser.add_argument(
            "--perf_event_paranoid",
            type=str,
            default=None,
            help="Host perf_event_paranoid value recorded in PMU provenance.",
        )
    if include_large_sf:
        parser.add_argument(
            "--large_sf",
            type=int,
            default=None,
            help="Large scale factor used for PMU/hotspot acceptance.",
        )
    if include_baseline_max_age_seconds:
        parser.add_argument(
            "--baseline_max_age_seconds",
            type=int,
            default=None,
            help="Maximum accepted age for baseline measurements.",
        )
    if include_stream_llm:
        parser.add_argument(
            "--stream_llm",
            action="store_true",
            default=False,
            help="Stream LLM responses when the configured model supports streaming.",
        )
    return None
