"""TPC-H MonetDB Outer Loop Orchestrator.

One-command entry point for the full TPC-H MonetDB autonomous pipeline:
  storage plan -> base impl -> optimization -> round decision -> repeat / stop
"""

import argparse
import logging
import re
import subprocess
import sys
from collections.abc import Callable, Iterator
from pathlib import Path

from tpch_monetdb.config import DEFAULT_TPCH_MONETDB_ARTIFACTS_DIR, get_outer_loop_defaults as _phase10_outer_defaults
from tpch_monetdb.runtime_workspace import (
    _prepare_runtime_workspace,
    build_runtime_snapshotter,
    resolve_runtime_workspace_path,
)
from tpch_monetdb.utils.cli_config import (
    DEFAULT_WANDB_FINISH_RETRIES,
    DEFAULT_WANDB_FINISH_TIMEOUT_S,
    DEFAULT_WANDB_INIT_MAX_ATTEMPTS,
    DEFAULT_WANDB_INIT_TIMEOUT_S,
    DEFAULT_WANDB_UPLOAD_TIMEOUT_S,
    add_common_args,
)
from tpch_monetdb.utils.gen_common import parse_query_ids
from tpch_monetdb.utils.outer_loop_state import (
    PhaseInfo,
    RoundRecord,
    compute_aggregate_runtime_ms,
    compute_round_decision,
    determine_resume_phase,
    get_outer_loop_dir,
    load_latest_round_record,
    load_round_record,
    write_round_record,
    build_conv_names,
    render_workflow_priority_order,
)
from tpch_monetdb.utils.scripted_summary import (
    find_latest_successful_run as find_latest_scripted_run,
)
from tpch_monetdb.utils.storage_plan_summary import find_latest_successful_storage_plan_run
from tpch_monetdb.utils.optimization_summary import find_latest_optimization_run, find_latest_successful_optimization_run
from tpch_monetdb.tools.tpch.pool import FastTestPool
from tpch_monetdb.tools.tpch.runtime_hygiene import cleanup_reload_dir
from tpch_monetdb.utils.outer_loop_supervisor import classify_optimization_result
from tpch_monetdb.utils.large_data_objectives import (
    classify_objective_failure_route,
    collect_large_data_failures,
)
from tpch_monetdb.utils.summary_gates import is_measurable_success, is_successful_large_data_run

logger = logging.getLogger(__name__)
_ERROR_ENVELOPE_RE = re.compile(r"\[ERROR:([A-Z_]+)\]")


def _scripted_handoff_complete(
    storage_summary,
    base_impl_summary,
) -> bool:
    """Return whether the scripted handoff contains the required contract fields."""
    required_base_fields = (
        "storage_plan_sha256",
        "todo_sha256",
        "implementation_manifest_sha256",
        "control_artifact_hashes",
        "todo_reconciliation",
    )
    for field_name in required_base_fields:
        value = getattr(base_impl_summary, field_name, None)
        if value in (None, "", {}, [], ()):
            return False
    storage_plan_sha256 = getattr(storage_summary, "storage_plan_sha256", None)
    base_storage_plan_sha256 = getattr(base_impl_summary, "storage_plan_sha256", None)
    if storage_plan_sha256 in (None, ""):
        return False
    if base_storage_plan_sha256 != storage_plan_sha256:
        return False
    return True

def _run_subprocess(cmd: list[str]) -> subprocess.CompletedProcess:
    logger.info("Running subprocess: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error("Subprocess failed with code %d", result.returncode)
        if result.stdout:
            logger.error("STDOUT:\n%s", result.stdout)
        if result.stderr:
            logger.error("STDERR:\n%s", result.stderr)
    else:
        if result.stdout:
            logger.info("STDOUT:\n%s", result.stdout[-2000:])
    return result


def _cleanup_runtime_for_conv(artifacts_dir: Path, conv_name: str) -> None:
    workspace_path = artifacts_dir / "workspaces" / conv_name
    FastTestPool.terminate_matching(
        lambda key: str(workspace_path) in key or key.startswith("./db "),
        suppress_errors=True,
    )
    if workspace_path.exists():
        cleanup_reload_dir(workspace_path)


def _parse_error_envelope_codes(text: str) -> set[str]:
    """Extract all [ERROR:<CODE>] codes from subprocess output."""
    return set(_iter_error_envelope_codes(text))


def _parse_error_envelope_code(text: str) -> str | None:
    """Extract last [ERROR:<CODE>] from subprocess output (last is most definitive)."""
    matches = list(_iter_error_envelope_codes(text))
    return matches[-1] if matches else None


def _iter_error_envelope_codes(text: str) -> Iterator[str]:
    for line in text.splitlines():
        if "filtered.input=" in line:
            continue
        for match in _ERROR_ENVELOPE_RE.findall(line):
            yield match
    return None


def _reset_phase_session_state(
    artifacts_dir: Path,
    runtime_conv_name: str | None,
) -> None:
    if not runtime_conv_name:
        return None
    session_dir = artifacts_dir / "cache" / "session"
    for suffix in ("", "-wal", "-shm"):
        session_path = session_dir / f"{runtime_conv_name}.sqlite{suffix}"
        if session_path.exists():
            session_path.unlink()
    return None


def _is_optimization_correctness_gate_failure(
    result: subprocess.CompletedProcess,
) -> bool:
    output = (result.stdout or "") + "\n" + (result.stderr or "")
    return "OPTIMIZATION_PRECHECK_FAILED" in _parse_error_envelope_codes(output)


def _is_final_correctness_gate_failure(
    result: subprocess.CompletedProcess,
) -> bool:
    output = (result.stdout or "") + "\n" + (result.stderr or "")
    return "FINAL_CORRECTNESS_GATE_FAILED" in _parse_error_envelope_codes(output)


def _should_continue_after_optimization_gate_failure(
    round_index: int,
    max_rounds: int,
) -> bool:
    return round_index < max_rounds


def _resolve_phase_failure_code(
    result: subprocess.CompletedProcess,
    default_code: str,
) -> str:
    """Resolve phase failure code from subprocess output, falling back to default."""
    output = (result.stdout or "") + "\n" + (result.stderr or "")
    parsed = _parse_error_envelope_code(output)
    if parsed is not None:
        return parsed
    return default_code


def _run_phase_with_retries(
    record: RoundRecord,
    phase: PhaseInfo,
    artifacts_dir: Path,
    retry_budget: int,
    phase_log_name: str,
    failure_reason: str,
    failure_code: str,
    cmd_factory: Callable[[], list[str]],
    runtime_conv_name: str | None = None,
    should_retry: Callable[[subprocess.CompletedProcess], bool] | None = None,
) -> subprocess.CompletedProcess:
    """Run a phase command and keep retrying until the configured budget is exhausted."""
    retry_allowed = should_retry or (lambda _result: True)
    result = _run_subprocess(cmd_factory())
    while (
        result.returncode != 0
        and phase.retry_count < retry_budget
        and retry_allowed(result)
    ):
        resolved_failure_code = _resolve_phase_failure_code(result, failure_code)
        phase.status = "failed"
        phase.failure_code = resolved_failure_code
        phase.failure_detail = failure_reason
        record.outcome = "failed"
        record.action = "failed"
        record.action_reason = failure_reason
        record.action_reason_code = resolved_failure_code
        write_round_record(record, artifacts_dir)

        phase.retry_count += 1
        _reset_phase_session_state(artifacts_dir, runtime_conv_name)
        logger.info("Retrying %s phase (retry %d/%d)", phase_log_name, phase.retry_count, retry_budget)
        phase.status = "running"
        write_round_record(record, artifacts_dir)
        result = _run_subprocess(cmd_factory())

    if result.returncode == 0:
        phase.status = "success"
        phase.failure_code = None
        phase.failure_detail = None
        record.outcome = "pending"
        record.action = "pending"
        record.action_reason = ""
        record.action_reason_code = ""
        return result

    resolved_failure_code = _resolve_phase_failure_code(result, "PHASE_RETRY_EXHAUSTED")
    phase.status = "failed"
    phase.failure_code = resolved_failure_code
    phase.failure_detail = failure_reason
    record.outcome = "failed"
    record.action = "failed"
    record.action_reason = failure_reason
    record.action_reason_code = resolved_failure_code
    write_round_record(record, artifacts_dir)
    return result


def _append_wandb_guard_args(cmd: list[str], args: argparse.Namespace) -> None:
    """Append non-default W&B guardrail CLI flags to child commands."""
    if getattr(args, "disable_wandb_when_tracing_disabled", False):
        cmd.append("--disable_wandb_when_tracing_disabled")
    init_attempts = int(
        getattr(args, "wandb_init_max_attempts", DEFAULT_WANDB_INIT_MAX_ATTEMPTS)
    )
    if init_attempts != DEFAULT_WANDB_INIT_MAX_ATTEMPTS:
        cmd += ["--wandb_init_max_attempts", str(init_attempts)]
    init_timeout = float(
        getattr(args, "wandb_init_timeout_s", DEFAULT_WANDB_INIT_TIMEOUT_S)
    )
    if init_timeout != DEFAULT_WANDB_INIT_TIMEOUT_S:
        cmd += ["--wandb_init_timeout_s", str(init_timeout)]
    upload_timeout = float(
        getattr(args, "wandb_upload_timeout_s", DEFAULT_WANDB_UPLOAD_TIMEOUT_S)
    )
    if upload_timeout != DEFAULT_WANDB_UPLOAD_TIMEOUT_S:
        cmd += ["--wandb_upload_timeout_s", str(upload_timeout)]
    finish_timeout = float(
        getattr(args, "wandb_finish_timeout_s", DEFAULT_WANDB_FINISH_TIMEOUT_S)
    )
    if finish_timeout != DEFAULT_WANDB_FINISH_TIMEOUT_S:
        cmd += ["--wandb_finish_timeout_s", str(finish_timeout)]
    finish_retries = int(
        getattr(args, "wandb_finish_retries", DEFAULT_WANDB_FINISH_RETRIES)
    )
    if finish_retries != DEFAULT_WANDB_FINISH_RETRIES:
        cmd += ["--wandb_finish_retries", str(finish_retries)]
    return None


def _append_measurement_args(cmd: list[str], args: argparse.Namespace) -> None:
    """Append explicit measurement-runtime parameters to child commands."""
    for flag_name in (
        "target_cpu",
        "hardware_counter_backend",
        "hardware_counter_runner_cmd",
        "host_kernel",
        "perf_event_paranoid",
        "large_sf",
        "baseline_max_age_seconds",
    ):
        value = getattr(args, flag_name, None)
        if value in (None, ""):
            continue
        cmd += [f"--{flag_name}", str(value)]
    return None


def _append_stream_llm_arg(cmd: list[str], args: argparse.Namespace) -> None:
    if getattr(args, "stream_llm", False):
        cmd.append("--stream_llm")
    return None


def _build_storage_plan_cmd(
    args: argparse.Namespace,
    round_index: int,
    prev_bottleneck_report_path: str | None = None,
    start_snapshot: str | None = None,
    storage_plan_mode: str = "initial_candidates",
) -> list[str]:
    conv_name = build_conv_names(args.conv, round_index)[0]
    cmd = [
        sys.executable, "-m", "tpch_monetdb.run_gen_storage_plan_tpch_monetdb",
        "--conv", conv_name,
        "--benchmark", args.benchmark,
        "--artifacts_dir", args.artifacts_dir,
        "--storage_plan_mode", storage_plan_mode,
    ]
    if args.base_data_dir:
        cmd += ["--base_data_dir", args.base_data_dir]
    if args.notify:
        cmd.append("--notify")
    if args.disable_repo_sync:
        cmd.append("--disable_repo_sync")
    if args.replay_cache:
        cmd.append("--replay_cache")
    if args.auto_u:
        cmd.append("--auto_u")
    if args.auto_finish:
        cmd.append("--auto_finish")
    if args.disable_wandb:
        cmd.append("--disable_wandb")
    if args.disable_tracing:
        cmd.append("--disable_tracing")
    if args.model:
        cmd += ["--model", args.model]
    if getattr(args, "reasoning_effort", None):
        cmd += ["--reasoning_effort", args.reasoning_effort]
    if prev_bottleneck_report_path:
        cmd += ["--prev_run_report", prev_bottleneck_report_path]
    if start_snapshot:
        cmd += ["--start_snapshot", start_snapshot]
    _append_stream_llm_arg(cmd, args)
    _append_wandb_guard_args(cmd, args)
    _append_measurement_args(cmd, args)
    return cmd


def _resolve_storage_plan_mode(round_index: int, prev_summary) -> str:
    """Choose whether storage planning should explore, repair, or apply deltas."""
    if round_index <= 1 or prev_summary is None:
        return "initial_candidates"
    route = getattr(prev_summary, "objective_failure_route", None)
    detail = str(getattr(prev_summary, "objective_failure_detail", "") or "")
    if route == "storage_plan" and any(
        marker in detail
        for marker in (
            "STORAGE_PLAN",
            "CONTROL_ARTIFACT",
            "DATA_LAW",
        )
    ):
        return "repair_alignment"
    return "delta_plan"


def _build_prev_storage_plan_feedback(prev_summary) -> str:
    from tpch_monetdb.utils.optimization_summary import (
        build_validation_kernel_report,
        render_bottleneck_report,
        render_validation_kernel_report,
    )

    final_validation_metrics = dict(
        getattr(prev_summary, "final_validation_metrics", {}) or {}
    )
    if final_validation_metrics:
        validation_report = build_validation_kernel_report(
            final_validation_metrics,
            list(getattr(prev_summary, "query_list", []) or []),
            getattr(prev_summary, "conv_name", "") or "",
        )
        report = render_validation_kernel_report(validation_report)
        if not is_measurable_success(prev_summary):
            report += (
                "\nSummary gate: optimization summary is not measurable; use the "
                "validator table only as final-validation evidence, not as a "
                "storage-layout success proof."
            )
        return report
    if prev_summary.final_correctness and is_measurable_success(prev_summary):
        return render_bottleneck_report(prev_summary)
    return (
        f"## Previous round ({prev_summary.conv_name}) correctness gate\n\n"
        "Correctness gate: previous round summary is not measurable; do not use it as layout evidence. "
        "Fix correctness and instrumentation/protocol issues before making structural storage-layout changes."
    )


def _build_base_impl_cmd(
    args: argparse.Namespace,
    round_index: int,
    storage_plan_snapshot: str,
    _is_bespoke_storage: bool,
) -> list[str]:
    conv_name = build_conv_names(args.conv, round_index)[1]
    cmd = [
        sys.executable, "-m", "tpch_monetdb.run_gen_base_impl_tpch_monetdb",
        "--conv", conv_name,
        "--benchmark", args.benchmark,
        "--artifacts_dir", args.artifacts_dir,
        "--validation_mode", args.validation_mode,
        "--storage_plan_snapshot", storage_plan_snapshot,
    ]
    cmd.append("--is_bespoke_storage")
    if args.base_data_dir:
        cmd += ["--base_data_dir", args.base_data_dir]
    if args.model:
        cmd += ["--model", args.model]
    if getattr(args, "reasoning_effort", None):
        cmd += ["--reasoning_effort", args.reasoning_effort]
    if args.notify:
        cmd.append("--notify")
    if args.disable_repo_sync:
        cmd.append("--disable_repo_sync")
    if args.replay_cache:
        cmd.append("--replay_cache")
    if args.disable_wandb:
        cmd.append("--disable_wandb")
    if args.disable_tracing:
        cmd.append("--disable_tracing")
    if args.auto_u:
        cmd.append("--auto_u")
    if args.auto_finish:
        cmd.append("--auto_finish")
    if getattr(args, "replay", False):
        cmd.append("--replay")
    if args.only_from_llm_cache:
        cmd.append("--only_from_llm_cache")
    if args.only_from_cache:
        cmd.append("--only_from_cache")
    if args.enable_auto_compact:
        cmd.append("--enable_auto_compact")
    _append_stream_llm_arg(cmd, args)
    _append_wandb_guard_args(cmd, args)
    _append_measurement_args(cmd, args)
    return cmd


def _build_optimization_cmd(
    args: argparse.Namespace,
    round_index: int,
    start_snapshot: str,
    bespoke_storage: bool,
) -> list[str]:
    _ = bespoke_storage
    benchmark = getattr(args, "benchmark", "tpch")
    conv_name = build_conv_names(args.conv, round_index)[2]
    cmd = [
        sys.executable, "run_optim_loop_tpch_monetdb.py",
        "--conv", conv_name,
        "--benchmark", benchmark,
        "--start_snapshot", start_snapshot,
        "--artifacts_dir", args.artifacts_dir,
    ]
    if args.model:
        cmd += ["--model", args.model]
    if getattr(args, "reasoning_effort", None):
        cmd += ["--reasoning_effort", args.reasoning_effort]
    if args.notify:
        cmd.append("--notify")
    if args.disable_repo_sync:
        cmd.append("--disable_repo_sync")
    if args.replay_cache:
        cmd.append("--replay_cache")
    if args.disable_wandb:
        cmd.append("--disable_wandb")
    if args.disable_tracing:
        cmd.append("--disable_tracing")
    if args.auto_u:
        cmd.append("--auto_u")
    if args.auto_finish:
        cmd.append("--auto_finish")
    if args.only_from_llm_cache:
        cmd.append("--only_from_llm_cache")
    if args.only_from_cache:
        cmd.append("--only_from_cache")
    if args.enable_auto_compact:
        cmd.append("--enable_auto_compact")
    if args.base_data_dir:
        cmd += ["--base_data_dir", args.base_data_dir]
    cmd += [
        "--benchmark_mode",
        getattr(args, "benchmark_mode", "system-parity"),
        "--storage_mode",
        getattr(args, "storage_mode", "persistent"),
    ]
    _append_stream_llm_arg(cmd, args)
    _append_wandb_guard_args(cmd, args)
    _append_measurement_args(cmd, args)
    return cmd


def _resolve_optimization_run_conv_name(
    opt_conv_name: str,
    _bespoke_storage: bool,
) -> str:
    return opt_conv_name


def _resolve_runtime_conv_name(
    benchmark: str,
    short_conv_name: str,
) -> str:
    return f"{benchmark}_{short_conv_name}"


def _resolve_scripted_run_conv_name(
    benchmark: str,
    short_conv_name: str,
    validation_mode: str,
) -> str:
    runtime_short_name = short_conv_name
    if validation_mode != "strict":
        runtime_short_name = f"{runtime_short_name}_{validation_mode}"
    return _resolve_runtime_conv_name(benchmark, runtime_short_name)


def _update_best_round(
    record: RoundRecord,
    opt_summary,
    round_index: int,
) -> None:
    """Update best-round tracking using aggregate runtime, independent of outcome labels."""
    current_round_is_better = (
        opt_summary.success
        and opt_summary.final_correctness
        and is_successful_large_data_run(opt_summary)
        and (
            record.best_round_index == 0
            or record.best_optimization_summary_path is None
            or record.best_aggregate_runtime_ms <= 0
            or record.aggregate_runtime_ms < record.best_aggregate_runtime_ms
        )
    )
    if current_round_is_better:
        record.best_round_index = round_index
        record.best_optimization_summary_path = record.optimization.summary_path
        record.best_aggregate_runtime_ms = record.aggregate_runtime_ms
        # Persist best optimization final snapshot hash for recovery
        record.best_final_snapshot_hash = getattr(opt_summary, "final_snapshot_hash", None)


def _write_performance_comparison_report(
    record: RoundRecord,
    opt_summary,
    artifacts_dir: Path,
) -> Path:
    """Write the measurable performance comparison report for the outer loop."""
    from tpch_monetdb.utils.optimization_summary import render_bottleneck_report

    if not is_successful_large_data_run(opt_summary):
        raise ValueError("performance_comparison.md requires a successful large-data summary")
    target_path = (
        get_outer_loop_dir(artifacts_dir, record.outer_loop_name)
        / "performance_comparison.md"
    )
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(
        render_bottleneck_report(opt_summary) + "\n",
        encoding="utf-8",
    )
    return target_path


def _should_treat_latest_record_as_terminal(
    record: RoundRecord,
    retry_budget: int,
) -> bool:
    if record.action in ("converged", "max_rounds"):
        return True
    # continue_with_best means we had a regression but want to keep exploring
    if record.action == "continue_with_best":
        return False
    if record.action != "failed":
        return False
    phases = (record.storage_plan, record.base_impl, record.optimization)
    for phase in phases:
        if phase is None:
            continue
        # Interrupted runs can leave stale failed action markers while a phase
        # remains pending/running. Treat those records as resumable.
        if phase.status in ("pending", "running"):
            return False
        if phase.status == "failed" and phase.retry_count < retry_budget:
            return False
    return True


def _restore_runtime_snapshot(args: argparse.Namespace, snapshot_hash: str) -> None:
    tpch_monetdb_root = Path(__file__).resolve().parent / "tpch_monetdb"
    workspace_path = resolve_runtime_workspace_path(tpch_monetdb_root)
    snapshotter = build_runtime_snapshotter(
        tpch_monetdb_root,
        disable_repo_sync=args.disable_repo_sync,
        keep_csv=getattr(args, "keep_csv", False),
    )
    _prepare_runtime_workspace(
        snapshotter,
        workspace_path,
        continue_run=False,
        reset_git_history=False,
    )
    if not snapshotter.has_snapshot(snapshot_hash):
        raise RuntimeError(f"Best optimization snapshot not found: {snapshot_hash}")
    logger.info("Restoring best optimization snapshot %s into %s", snapshot_hash, workspace_path)
    snapshotter.restore(snapshot_hash)
    return None


def _restore_terminal_best_snapshot(
    args: argparse.Namespace,
    record: RoundRecord,
) -> None:
    if record.best_final_snapshot_hash is None:
        raise RuntimeError(
            f"Terminal action {record.action!r} requires best_final_snapshot_hash, but none was recorded."
        )
    _restore_runtime_snapshot(args, record.best_final_snapshot_hash)
    return None


def main(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    benchmark = args.benchmark
    outer_loop_name = args.conv
    query_ids = parse_query_ids(outer_loop_name, "outer", benchmark=benchmark)
    if query_ids is None:
        raise ValueError(f"Failed to parse query ids from {outer_loop_name}")

    artifacts_dir = Path(args.artifacts_dir)
    max_rounds = args.max_rounds
    convergence_threshold = args.convergence_threshold
    stagnant_rounds = args.stagnant_rounds
    retry_budget = args.retry_budget

    storage_enabled = True

    latest_record = load_latest_round_record(outer_loop_name, artifacts_dir)

    if latest_record is not None and _should_treat_latest_record_as_terminal(
        latest_record,
        retry_budget,
    ):
        logger.info("Outer loop already terminal: %s", latest_record.action)
        _restore_terminal_best_snapshot(args, latest_record)
        _print_terminal_result(latest_record)
        return None

    if latest_record is not None:
        record = latest_record
        resume_phase, record = determine_resume_phase(record, retry_budget)
        if resume_phase == "failed":
            logger.error("Retry budget exhausted. Outer loop failed.")
            record.action = "failed"
            record.action_reason = "retry budget exhausted"
            record.action_reason_code = "PHASE_RETRY_EXHAUSTED"
            write_round_record(record, artifacts_dir)
            sys.exit(1)
        starting_round_index = record.round_index
        if resume_phase == "next_round":
            starting_round_index += 1
            resume_phase = "storage_plan"
        logger.info("Resuming outer loop %s from round %d phase %s", outer_loop_name, starting_round_index, resume_phase)
    else:
        starting_round_index = 1
        sp_conv, bi_conv, opt_conv = build_conv_names(outer_loop_name, 1)
        record = RoundRecord(
            outer_loop_name=outer_loop_name,
            round_index=1,
            query_list=query_ids,
            storage_plan=PhaseInfo(conv_name=sp_conv, status="pending"),
            base_impl=PhaseInfo(conv_name=bi_conv, status="pending"),
            optimization=PhaseInfo(conv_name=opt_conv, status="pending"),
        )
        resume_phase = "storage_plan"
        logger.info("Starting new outer loop %s", outer_loop_name)

    prev_optimization_summary = None
    stagnant_count = 0
    if record.round_index > 1:
        prev_opt = find_latest_successful_optimization_run(
            conv_name=_resolve_optimization_run_conv_name(
                build_conv_names(outer_loop_name, record.round_index - 1)[2],
                storage_enabled,
            ),
            query_list=query_ids,
            benchmark=benchmark,
            artifacts_dir=artifacts_dir,
        )
        if prev_opt is not None:
            prev_optimization_summary = prev_opt
        # Restore stagnant count from prior rounds if resuming
        for ri in range(1, record.round_index):
            r = load_round_record(outer_loop_name, ri, artifacts_dir)
            if r is not None and r.outcome == "stagnant":
                stagnant_count += 1
            else:
                stagnant_count = 0

    for round_index in range(starting_round_index, max_rounds + 1):
        storage_plan_start_snapshot: str | None = None
        if round_index != record.round_index:
            previous_record = record
            if previous_record.action in ("continue_with_best", "continue_from_best"):
                storage_plan_start_snapshot = previous_record.best_final_snapshot_hash
                if storage_plan_start_snapshot is None:
                    raise RuntimeError(
                        f"{previous_record.action} requires best_final_snapshot_hash for the next storage-plan run."
                    )
            record = RoundRecord(
                outer_loop_name=outer_loop_name,
                round_index=round_index,
                query_list=query_ids,
                storage_plan=PhaseInfo(conv_name=build_conv_names(outer_loop_name, round_index)[0], status="pending"),
                base_impl=PhaseInfo(conv_name=build_conv_names(outer_loop_name, round_index)[1], status="pending"),
                optimization=PhaseInfo(conv_name=build_conv_names(outer_loop_name, round_index)[2], status="pending"),
                best_round_index=previous_record.best_round_index,
                best_optimization_summary_path=previous_record.best_optimization_summary_path,
                best_aggregate_runtime_ms=previous_record.best_aggregate_runtime_ms,
                best_final_snapshot_hash=previous_record.best_final_snapshot_hash,
            )
            resume_phase = "storage_plan"

        sp_conv, bi_conv, opt_conv = build_conv_names(outer_loop_name, round_index)
        sp_run_conv = _resolve_runtime_conv_name(benchmark, sp_conv)
        bi_run_conv = _resolve_scripted_run_conv_name(
            benchmark,
            bi_conv,
            args.validation_mode,
        )
        opt_run_conv = _resolve_runtime_conv_name(
            benchmark,
            _resolve_optimization_run_conv_name(opt_conv, storage_enabled),
        )

        # ---- Storage Plan Phase ----
        if resume_phase == "storage_plan" or record.storage_plan is None or record.storage_plan.status != "success":
            if record.storage_plan is None:
                record.storage_plan = PhaseInfo(conv_name=sp_conv)
            record.storage_plan.conv_name = sp_conv
            record.storage_plan.status = "running"
            write_round_record(record, artifacts_dir)

            # Round > 1: generate bottleneck report from previous best optimization summary
            prev_report_path: str | None = None
            if round_index > 1 and prev_optimization_summary is not None:
                _report_text = _build_prev_storage_plan_feedback(prev_optimization_summary)
                _outer_dir = get_outer_loop_dir(artifacts_dir, args.conv)
                _outer_dir.mkdir(parents=True, exist_ok=True)
                _report_file = _outer_dir / f"round_{round_index:03d}_prev_feedback.md"
                _report_file.write_text(_report_text)
                prev_report_path = str(_report_file)
                logger.info("Written bottleneck report for round %d to %s", round_index, prev_report_path)
            storage_plan_mode = _resolve_storage_plan_mode(
                round_index,
                prev_optimization_summary,
            )

            result = _run_phase_with_retries(
                record=record,
                phase=record.storage_plan,
                artifacts_dir=artifacts_dir,
                retry_budget=retry_budget,
                phase_log_name="storage plan",
                failure_reason="storage plan phase failed",
                failure_code="PHASE_RETRY_EXHAUSTED",
                cmd_factory=lambda: _build_storage_plan_cmd(
                    args,
                    round_index,
                    prev_report_path,
                    storage_plan_start_snapshot,
                    storage_plan_mode,
                ),
                runtime_conv_name=sp_run_conv,
            )
            if result.returncode != 0:
                sys.exit(1)

            sp_summary = find_latest_successful_storage_plan_run(
                conv_name=sp_run_conv,
                query_list=query_ids,
                benchmark=benchmark,
                artifacts_dir=artifacts_dir,
            )
            if sp_summary is None:
                logger.error("Storage plan succeeded but summary not found")
                record.storage_plan.status = "failed"
                record.storage_plan.failure_code = "PHASE_SUMMARY_MISSING"
                record.storage_plan.failure_detail = "storage plan summary missing"
                record.outcome = "failed"
                record.action = "failed"
                record.action_reason = "storage plan summary missing"
                record.action_reason_code = "PHASE_SUMMARY_MISSING"
                write_round_record(record, artifacts_dir)
                sys.exit(1)
            record.storage_plan.summary_path = str(
                artifacts_dir / "storage_plan_runs" / sp_summary.conv_name / "latest.json"
            )
            storage_plan_snapshot = sp_summary.final_snapshot_hash
            logger.info("Storage plan completed. Snapshot: %s", storage_plan_snapshot)
        else:
            sp_summary = find_latest_successful_storage_plan_run(
                conv_name=sp_run_conv,
                query_list=query_ids,
                benchmark=benchmark,
                artifacts_dir=artifacts_dir,
            )
            if sp_summary is None:
                logger.error("Missing storage plan summary for round %d", round_index)
                sys.exit(1)
            storage_plan_snapshot = sp_summary.final_snapshot_hash
            record.storage_plan = PhaseInfo(conv_name=sp_conv, status="success", summary_path=str(
                artifacts_dir / "storage_plan_runs" / sp_summary.conv_name / "latest.json"
            ))

        # ---- Base Impl Phase ----
        if resume_phase == "base_impl" or record.base_impl is None or record.base_impl.status != "success":
            if record.base_impl is None:
                record.base_impl = PhaseInfo(conv_name=bi_conv)
            record.base_impl.conv_name = bi_conv
            record.base_impl.status = "running"
            write_round_record(record, artifacts_dir)
            result = _run_phase_with_retries(
                record=record,
                phase=record.base_impl,
                artifacts_dir=artifacts_dir,
                retry_budget=retry_budget,
                phase_log_name="base impl",
                failure_reason="base impl phase failed",
                failure_code="PHASE_RETRY_EXHAUSTED",
                cmd_factory=lambda: _build_base_impl_cmd(
                    args, round_index, storage_plan_snapshot, storage_enabled
                ),
                runtime_conv_name=bi_run_conv,
                should_retry=lambda res: not _is_final_correctness_gate_failure(res),
            )
            if result.returncode != 0:
                if _is_final_correctness_gate_failure(result):
                    logger.error(
                        "Base impl final correctness gate rejected the scripted output; "
                        "advancing to the next round."
                    )
                    record.base_impl.status = "failed"
                    record.base_impl.failure_code = "FINAL_CORRECTNESS_GATE_FAILED"
                    record.base_impl.failure_detail = "final correctness gate failed"
                    record.base_impl.summary_path = None
                    record.base_impl.retry_count = max(
                        record.base_impl.retry_count,
                        retry_budget,
                    )
                    record.outcome = "failed"
                    if _should_continue_after_optimization_gate_failure(round_index, max_rounds):
                        record.action = "continue"
                        record.action_reason = (
                            "final correctness gate failed; advancing to next round"
                        )
                        record.action_reason_code = "FINAL_CORRECTNESS_GATE_FAILED"
                        write_round_record(record, artifacts_dir)
                        logger.warning(
                            "Advancing to round %d after base final correctness gate failure.",
                            round_index + 1,
                        )
                        prev_optimization_summary = None
                        resume_phase = "storage_plan"
                        continue
                    record.action = "failed"
                    record.action_reason = (
                        "final correctness gate failed and max rounds reached"
                    )
                    record.action_reason_code = "FINAL_CORRECTNESS_GATE_FAILED"
                    write_round_record(record, artifacts_dir)
                sys.exit(1)

            bi_summary = find_latest_scripted_run(
                conv_name=bi_run_conv,
                query_list=query_ids,
                benchmark=benchmark,
                artifacts_dir=artifacts_dir,
                validation_mode=args.validation_mode,
                is_bespoke_storage=storage_enabled,
            )
            if bi_summary is None:
                logger.error("Base impl succeeded but summary not found")
                record.base_impl.status = "failed"
                record.base_impl.failure_code = "PHASE_SUMMARY_MISSING"
                record.base_impl.failure_detail = "base impl summary missing"
                record.outcome = "failed"
                record.action = "failed"
                record.action_reason = "base impl summary missing"
                record.action_reason_code = "PHASE_SUMMARY_MISSING"
                write_round_record(record, artifacts_dir)
                sys.exit(1)
            if not _scripted_handoff_complete(sp_summary, bi_summary):
                logger.error("Base impl summary missing required control-artifact lineage fields")
                record.base_impl.status = "failed"
                record.base_impl.failure_code = "CONTROL_ARTIFACT_HANDOFF_INCOMPLETE"
                record.base_impl.failure_detail = "base impl summary missing required control-artifact lineage"
                record.outcome = "failed"
                record.action = "failed"
                record.action_reason = "base impl summary missing required control-artifact lineage"
                record.action_reason_code = "CONTROL_ARTIFACT_HANDOFF_INCOMPLETE"
                write_round_record(record, artifacts_dir)
                sys.exit(1)
            record.base_impl.summary_path = str(
                artifacts_dir / "scripted_runs" / bi_summary.conv_name / "latest.json"
            )
            base_impl_snapshot = bi_summary.final_snapshot_hash
            logger.info("Base impl completed. Snapshot: %s", base_impl_snapshot)
        else:
            bi_summary = find_latest_scripted_run(
                conv_name=bi_run_conv,
                query_list=query_ids,
                benchmark=benchmark,
                artifacts_dir=artifacts_dir,
                validation_mode=args.validation_mode,
                is_bespoke_storage=storage_enabled,
            )
            if bi_summary is None:
                logger.error("Missing base impl summary for round %d", round_index)
                sys.exit(1)
            if not _scripted_handoff_complete(sp_summary, bi_summary):
                logger.error("Resumed base impl summary missing required control-artifact lineage fields")
                sys.exit(1)
            base_impl_snapshot = bi_summary.final_snapshot_hash
            record.base_impl = PhaseInfo(conv_name=bi_conv, status="success", summary_path=str(
                artifacts_dir / "scripted_runs" / bi_summary.conv_name / "latest.json"
            ))

        # ---- Optimization Phase ----
        if resume_phase == "optimization" or record.optimization is None or record.optimization.status != "success":
            if record.optimization is None:
                record.optimization = PhaseInfo(conv_name=opt_run_conv)
            record.optimization.conv_name = opt_run_conv
            record.optimization.status = "running"
            write_round_record(record, artifacts_dir)

            # Supervisor-driven retry loop: each retry is classified before being consumed.
            _opt_cmd = _build_optimization_cmd(
                args, round_index, base_impl_snapshot, storage_enabled
            )
            while True:
                result = subprocess.run(_opt_cmd, capture_output=True, text=True)
                opt_summary = find_latest_optimization_run(
                    conv_name=opt_run_conv,
                    query_list=query_ids,
                    benchmark=benchmark,
                    artifacts_dir=artifacts_dir,
                )
                decision = classify_optimization_result(
                    result,
                    summary_found=opt_summary is not None,
                    retry_count=record.optimization.retry_count,
                    retry_budget=retry_budget,
                )
                if decision.should_cleanup_runtime:
                    _cleanup_runtime_for_conv(artifacts_dir, opt_run_conv)
                if not decision.should_retry:
                    break
                record.optimization.retry_count += 1
                _reset_phase_session_state(artifacts_dir, opt_run_conv)
                write_round_record(record, artifacts_dir)

            if _is_optimization_correctness_gate_failure(result):
                logger.error(
                    "Optimization pre-check detected base correctness regression; "
                    "round output is invalid for optimization."
                )
                record.optimization.status = "failed"
                record.optimization.failure_code = "OPTIMIZATION_PRECHECK_FAILED"
                record.optimization.failure_detail = "optimization precheck correctness failed"
                record.optimization.summary_path = None
                record.optimization.retry_count = max(
                    record.optimization.retry_count, retry_budget
                )
                record.outcome = "failed"
                if _should_continue_after_optimization_gate_failure(round_index, max_rounds):
                    record.action = "continue"
                    record.action_reason = (
                        "optimization precheck correctness failed; advancing to next round"
                    )
                    record.action_reason_code = "OPTIMIZATION_PRECHECK_FAILED"
                    write_round_record(record, artifacts_dir)
                    logger.warning(
                        "Advancing to round %d after optimization pre-check correctness failure.",
                        round_index + 1,
                    )
                    prev_optimization_summary = None
                    resume_phase = "storage_plan"
                    continue
                record.action = "failed"
                record.action_reason = (
                    "optimization precheck correctness failed and max rounds reached"
                )
                record.action_reason_code = "OPTIMIZATION_PRECHECK_FAILED"
                write_round_record(record, artifacts_dir)
                sys.exit(1)

            if decision.failure_code is not None:
                _failure_code = decision.failure_code
                _failure_detail = decision.failure_detail
                if opt_summary is not None and not opt_summary.success:
                    _failure_code = opt_summary.failure_code or _failure_code
                    _failure_detail = opt_summary.failure_detail or _failure_detail
                record.optimization.status = "failed"
                record.optimization.failure_code = _failure_code
                record.optimization.failure_detail = _failure_detail
                record.outcome = "failed"
                record.action = "failed"
                record.action_reason = _failure_detail
                record.action_reason_code = _failure_code
                write_round_record(record, artifacts_dir)
                sys.exit(1)

            if opt_summary is None:
                logger.error(
                    "Optimization summary missing.",
                )
                record.optimization.status = "failed"
                record.optimization.failure_code = "PHASE_SUMMARY_MISSING"
                record.optimization.failure_detail = "optimization summary missing"
                record.outcome = "failed"
                record.action = "failed"
                record.action_reason = "optimization summary missing"
                record.action_reason_code = "PHASE_SUMMARY_MISSING"
                write_round_record(record, artifacts_dir)
                sys.exit(1)
            if not opt_summary.success:
                objective_failures = list(collect_large_data_failures(opt_summary))
                if objective_failures and round_index < max_rounds:
                    logger.warning(
                        "Optimization summary failed on objective gates; routing next round: %s",
                        ", ".join(objective_failures),
                    )
                    record.optimization.status = "failed"
                    record.optimization.failure_code = (
                        opt_summary.failure_code or objective_failures[0]
                    )
                    record.optimization.failure_detail = (
                        opt_summary.failure_detail
                        or "objective failures: " + ", ".join(objective_failures)
                    )
                    record.optimization.summary_path = str(
                        artifacts_dir / "optimization_runs" / opt_summary.conv_name / "latest.json"
                    )
                    record.objective_failures = objective_failures
                    record.failure_route = getattr(
                        opt_summary,
                        "objective_failure_route",
                        None,
                    )
                    record.outcome = "objective_failed"
                    record.action = "continue"
                    record.action_reason = (
                        "optimization failed with routable objective failures; "
                        f"next round route={record.failure_route or 'optimization'}"
                    )
                    record.action_reason_code = (
                        record.optimization.failure_code
                        or "OBJECTIVE_FAILED_ROUTABLE"
                    )
                    write_round_record(record, artifacts_dir)
                    prev_optimization_summary = opt_summary
                    resume_phase = "storage_plan"
                    continue
                else:
                    logger.error(
                        "Optimization failed with summary: %s",
                        opt_summary.failure_detail,
                    )
                    record.optimization.status = "failed"
                    record.optimization.failure_code = (
                        opt_summary.failure_code or "OPTIMIZATION_FAILED_WITH_SUMMARY"
                    )
                    record.optimization.failure_detail = (
                        opt_summary.failure_detail or "optimization failed with summary"
                    )
                    record.optimization.summary_path = str(
                        artifacts_dir / "optimization_runs" / opt_summary.conv_name / "latest.json"
                    )
                    record.outcome = "failed"
                    record.action = "failed"
                    record.action_reason = record.optimization.failure_detail
                    record.action_reason_code = record.optimization.failure_code
                    write_round_record(record, artifacts_dir)
                    sys.exit(1)
            record.optimization.summary_path = str(
                artifacts_dir / "optimization_runs" / opt_summary.conv_name / "latest.json"
            )
            if not is_measurable_success(opt_summary):
                logger.error("Optimization summary is not measurable.")
                record.optimization.status = "failed"
                record.optimization.failure_code = "SUMMARY_NOT_MEASURABLE"
                record.optimization.failure_detail = "optimization summary is not measurable"
                record.outcome = "failed"
                record.action = "failed"
                record.action_reason = record.optimization.failure_detail
                record.action_reason_code = record.optimization.failure_code
                write_round_record(record, artifacts_dir)
                sys.exit(1)
            record.aggregate_runtime_ms = compute_aggregate_runtime_ms(opt_summary.final_runtime_ms_by_query)
            logger.info("Optimization completed. Aggregate runtime: %.3f ms", record.aggregate_runtime_ms)
        else:
            opt_summary = find_latest_optimization_run(
                conv_name=opt_run_conv,
                query_list=query_ids,
                benchmark=benchmark,
                artifacts_dir=artifacts_dir,
            )
            if opt_summary is None or not opt_summary.success:
                logger.error("Missing optimization summary for round %d", round_index)
                sys.exit(1)
            record.optimization = PhaseInfo(conv_name=opt_run_conv, status="success", summary_path=str(
                artifacts_dir / "optimization_runs" / opt_summary.conv_name / "latest.json"
            ))
            record.aggregate_runtime_ms = compute_aggregate_runtime_ms(opt_summary.final_runtime_ms_by_query)

        # ---- Round Decision ----
        outcome, action, stagnant_count = compute_round_decision(
            prev_summary=prev_optimization_summary,
            curr_summary=opt_summary,
            convergence_threshold=convergence_threshold,
            stagnant_rounds=stagnant_rounds,
            regression_tolerance=args.regression_tolerance,
            max_rounds=max_rounds,
            current_round_index=round_index,
            stagnant_count=stagnant_count,
            has_best_snapshot=bool(record.best_final_snapshot_hash),
        )
        record.outcome = outcome
        record.action = action
        record.action_reason = f"round {round_index} outcome={outcome} stagnant={stagnant_count}"
        record.objective_failures = list(collect_large_data_failures(opt_summary))
        record.failure_route = (
            classify_objective_failure_route(tuple(record.objective_failures))
            if record.objective_failures
            else None
        )

        # Track best round (also updates best_final_snapshot_hash on record)
        _update_best_round(record, opt_summary, round_index)
        if is_successful_large_data_run(opt_summary):
            _write_performance_comparison_report(record, opt_summary, artifacts_dir)

        write_round_record(record, artifacts_dir)
        logger.info("Round %d decision: %s (%s)", round_index, action, outcome)

        if action in ("converged", "max_rounds"):
            _restore_terminal_best_snapshot(args, record)
            _print_terminal_result(record)
            return None

        if action == "failed":
            sys.exit(1)

        if action == "continue" and record.objective_failures and record.best_final_snapshot_hash:
            record.action = "continue_from_best"
            record.action_reason = (
                "objective failures remain; next round starts from best snapshot via "
                f"{record.failure_route or 'optimization'} route"
            )
            write_round_record(record, artifacts_dir)
            prev_optimization_summary = opt_summary
            resume_phase = "storage_plan"
            continue

        if action == "continue_with_best":
            # Regression occurred but we have a best snapshot — keep exploring
            # using best round as the reference for the next round decision
            logger.info(
                "Round %d regressed but best_final_snapshot_hash=%s — continuing exploration",
                round_index,
                record.best_final_snapshot_hash,
            )
            # Use the best summary (not this regressed one) as prev for next round
            _best_path = record.best_optimization_summary_path
            if _best_path:
                import json as _json
                _best_data = _json.loads(Path(_best_path).read_text())
                _best_summary_data = _best_data.get("summary") or _best_data
                from tpch_monetdb.utils.optimization_summary import OptimizationRunSummary as _ORS
                prev_optimization_summary = _ORS.from_dict(_best_summary_data)
            else:
                prev_optimization_summary = opt_summary
            resume_phase = "storage_plan"
            continue

        prev_optimization_summary = opt_summary
        resume_phase = "storage_plan"

    # If we exit the loop without a terminal action, mark as max_rounds
    record.action = "max_rounds"
    record.action_reason = "reached max rounds"
    write_round_record(record, artifacts_dir)
    _print_terminal_result(record)
    return None


def _print_terminal_result(record: RoundRecord) -> None:
    priority_line = render_workflow_priority_order()
    print("\n" + "=" * 60)
    print("OUTER LOOP TERMINAL RESULT")
    print("=" * 60)
    print(f"Outer loop:   {record.outer_loop_name}")
    print(f"Rounds:       {record.round_index}")
    print(f"Decision:     {record.action}")
    print(f"Reason:       {record.action_reason}")
    print(f"Priority:     {priority_line}")
    print(f"Best round:   {record.best_round_index}")
    print(f"Best summary: {record.best_optimization_summary_path}")
    print(f"Best snapshot: {record.best_final_snapshot_hash or 'n/a'}")
    aggregate_ms = (
        record.best_aggregate_runtime_ms
        if record.best_aggregate_runtime_ms > 0
        else record.aggregate_runtime_ms
    )
    print(f"Aggregate ms: {aggregate_ms:.3f}")
    print("=" * 60)


def build_parser(*, add_help: bool = True) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="TPC-H MonetDB Outer Loop Orchestrator",
        add_help=add_help,
    )
    parser.add_argument(
        "--conv",
        type=str,
        required=True,
        help="Short name for the outer loop (e.g., outer1-9v1)",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default="tpch",
        help="Benchmark name",
    )
    _p10 = _phase10_outer_defaults()
    parser.add_argument(
        "--max_rounds",
        type=int,
        default=_p10["max_rounds"],
        help="Maximum number of outer-loop rounds",
    )
    parser.add_argument(
        "--convergence_threshold",
        type=float,
        default=_p10["convergence_threshold"],
        help="Workload improvement threshold below which a round is considered stagnant",
    )
    parser.add_argument(
        "--stagnant_rounds",
        type=int,
        default=_p10["stagnant_rounds"],
        help="Consecutive stagnant rounds required to declare convergence",
    )
    parser.add_argument(
        "--retry_budget",
        type=int,
        default=_p10["retry_budget"],
        help="Number of retries allowed per phase",
    )
    parser.add_argument(
        "--regression_tolerance",
        type=float,
        default=_p10["regression_tolerance"],
        help="Fractional runtime regression tolerance",
    )
    parser.add_argument(
        "--bespoke_storage",
        action="store_true",
        default=True,
        help="Deprecated compatibility flag; TPC-H MonetDB outer loop is always storage-enabled.",
    )
    parser.add_argument(
        "--validation_mode",
        type=str,
        choices=["strict", "traversal"],
        default="strict",
        help="Validation mode for base impl",
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
        include_only_from_llm_cache=True,
        include_only_from_cache=True,
        include_enable_auto_compact=True,
        include_benchmark_mode=True,
        include_storage_mode=True,
        include_target_cpu=True,
        include_hardware_counter_backend=True,
        include_hardware_counter_runner_cmd=True,
        include_host_kernel=True,
        include_perf_event_paranoid=True,
        include_large_sf=True,
        include_stream_llm=True,
        include_baseline_max_age_seconds=True,
    )
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    main(args)
