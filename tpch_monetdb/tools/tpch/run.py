import hashlib
import json
import logging
import os
import shlex
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from agents.run_context import RunContextWrapper
from agents.tool import FunctionTool
from pydantic import BaseModel, Field

from tpch_monetdb.llm_cache.git_snapshotter import GitSnapshotter
from tpch_monetdb.misc.tpch.compiler import build_id as read_build_id
from tpch_monetdb.misc.tpch.compiler import build_vectorization_flag_bundle
from tpch_monetdb.misc.tpch.compiler_cached import CachedCompiler
from tpch_monetdb.misc.tpch.fasttest_proc import (
    FasttestProc,
    RunnerBrokenPipePersistentError,
    RunnerInfraFailureError,
    RunnerTransportError,
)
from tpch_monetdb.tools.function_tool_args import load_function_tool_args
from tpch_monetdb.tools.tpch.hardware_counters import (
    DEFAULT_PERF_HOTSPOT_EVENT,
    DEFAULT_PERF_HOTSPOT_FREQUENCY,
    DEFAULT_PERF_HOTSPOT_REPETITIONS,
    HardwareCounterPreflight,
    HardwareCounterSummary,
    PerfHotspotSummary,
    build_hardware_counter_invocation,
    build_perf_record_invocation,
    build_perf_script_invocation,
    parse_perf_script_hotspots,
    parse_perf_stat_csv,
    validate_hardware_counter_summary,
)
from tpch_monetdb.tools.tpch.process_tree import collect_process_tree_pids
from tpch_monetdb.utils.wandb_stats_logging import WandbRunHook

from .pool import FastTestPool
from .runtime_hygiene import (
    classify_infra_failure,
    cleanup_reload_dir,
    inspect_runtime_health,
)
from .utils import make_compiler

logger = logging.getLogger(__name__)

QUERY_OUTPUT_MODE_FULL_CSV = "full_csv"
QUERY_OUTPUT_MODE_NO_OUTPUT = "no_output"
QUERY_OUTPUT_MODE_HASH_ONLY = "hash_only"
TRACE_OUTPUT_FILENAME = "tracing_output.log"
VALID_QUERY_OUTPUT_MODES = frozenset(
    {
        QUERY_OUTPUT_MODE_FULL_CSV,
        QUERY_OUTPUT_MODE_NO_OUTPUT,
        QUERY_OUTPUT_MODE_HASH_ONLY,
    }
)
_BASE_PERFORMANCE_COMPARISON_ARG = "__base_performance_comparison"


def _normalize_query_output_mode(output_mode: str) -> str:
    """Validate and normalize one query output mode for runner execution."""
    normalized = output_mode.strip().lower().replace("-", "_")
    if normalized not in VALID_QUERY_OUTPUT_MODES:
        raise ValueError(
            "Invalid query output mode "
            f"{output_mode!r}. Expected one of: "
            f"{', '.join(sorted(VALID_QUERY_OUTPUT_MODES))}"
        )
    return normalized


def _format_run_envelope(
    *,
    stdout: str,
    stderr: str,
    response: str,
) -> str:
    """Render run output as a structured envelope for downstream artifactizing."""
    payload = {
        "stdout": stdout.rstrip(),
        "stderr": stderr.rstrip(),
        "response": response,
        "stdout_bytes": len(stdout.encode("utf-8")),
        "stderr_bytes": len(stderr.encode("utf-8")),
        "response_bytes": len(response.encode("utf-8")),
    }
    return (
        f"stdout: {stdout.rstrip()}\n"
        f"stderr: {stderr.rstrip()}\n"
        "[Run Tool Envelope]\n"
        + json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


def _approx_timeout_for_validation(
    scale_factor: float,
    num_queries: int,
    repetitions: int,
    num_random_query_instantiations: int,
) -> int:
    """Local fallback for approx_timeout_for_validation."""
    # Simple heuristic: base timeout of 30s + scale factor adjustment
    base_timeout = 30
    sf_multiplier = max(1.0, scale_factor / 10.0)
    query_multiplier = num_queries * num_random_query_instantiations * repetitions
    return int(base_timeout * sf_multiplier * max(1, query_multiplier / 10))


def _approx_timeout_for_hardware_counter_capture(
    scale_factor: float,
    num_queries: int,
) -> int:
    """Bound PMU sampling time separately from full correctness validation."""
    base_timeout = 60
    sf_multiplier = max(1.0, scale_factor / 100.0)
    query_multiplier = max(1.0, num_queries / 2.0)
    estimated_timeout = int(base_timeout * sf_multiplier * query_multiplier)
    return min(300, max(base_timeout, estimated_timeout))


def _approx_timeout_for_perf_hotspot_capture(
    scale_factor: float,
    num_queries: int,
    hotspot_repetitions: int,
) -> int:
    """Bound perf record/script time separately from correctness validation."""
    base_timeout = 90
    sf_multiplier = max(1.0, scale_factor / 100.0)
    query_multiplier = max(1.0, num_queries / 2.0)
    repetition_multiplier = max(1.0, hotspot_repetitions / 3.0)
    estimated_timeout = int(
        base_timeout * sf_multiplier * query_multiplier * repetition_multiplier
    )
    return min(600, max(base_timeout, estimated_timeout))


def _assemble_validation_error(
    scale_factor: float,
    query_ids_executed: list[str],
    exception: bool = True,
    query_id: str | None = None,
) -> dict[str, Any]:
    return {
        "validation/scale_factor": scale_factor,
        "validation/correct": False,
        "validation/error": exception,
        "validation/query_ids_executed": query_ids_executed,
        "validation/num_queries": len(query_ids_executed),
        "validation/num_successful_queries": 0,
        "validation/failed_query_id": query_id,
    }


@dataclass
class RunWorkerResult:
    msg: str
    metrics: Optional[Dict] = None
    resp: Optional[str] = None
    out: Optional[str] = None
    err: Optional[str] = None


def _normalize_runtime_scale_factor(scale_factor: float) -> int:
    """Return an integer scale factor accepted by runtime data layouts."""
    if scale_factor >= 1:
        if int(scale_factor) != scale_factor:
            raise ValueError(f"Scale factor has to be integer >= 1, got {scale_factor!r}")
        return int(scale_factor)
    raise ValueError(f"Scale factor must be >= 1, got {scale_factor!r}")


def _tpch_root_contains_tbl_files(data_root: Path) -> bool:
    """Return True when data_root directly contains tiny TPC-H .tbl fixtures."""
    required_tables = {
        "customer",
        "lineitem",
        "nation",
        "orders",
        "part",
        "partsupp",
        "region",
        "supplier",
    }
    return all((data_root / f"{table}.tbl").is_file() for table in required_tables)


def resolve_runtime_data_path(
    *,
    dataset_name: str,
    base_data_dir: str,
    scale_factor: float,
) -> str:
    """Resolve the data path passed to the generated C++ runtime."""
    sf = _normalize_runtime_scale_factor(scale_factor)
    base_dir = Path(base_data_dir)
    normalized_dataset = dataset_name.strip().lower()
    if normalized_dataset == "tpch":
        sf_dir = base_dir / f"sf{sf}"
        if sf_dir.exists() or not _tpch_root_contains_tbl_files(base_dir):
            return sf_dir.as_posix()
        return base_dir.as_posix()
    raise ValueError(f"Unsupported dataset for runtime execution: {dataset_name}")


class RunTool:
    """Runs the database and executes a query by id"""

    parse_out_and_validate_output: bool = True

    def __init__(
        self,
        cwd: Path,
        dataset_name: str,
        base_data_dir: str,  # must contain per scale-factors subdirs: e.g. base_data_dir/sf1/, base_data_dir/sf10/..., each containing the corresponding data files for the scale factor
        query_validator: Optional[Any] = None,
        wandb_metrics_hook: Optional[WandbRunHook] = None,
        compile_cache_dir: Optional[Path] = None,
        git_snapshotter: Optional[GitSnapshotter] = None,
        parse_out_and_validate_output: bool = True,
        api_path: Optional[Path] = None,
        only_from_cache: bool = False,
        target_cpu: Optional[str] = None,
        emit_vectorization_reports: bool = False,
    ) -> None:
        self.cwd = cwd
        self.dataset_name = dataset_name
        self.base_data_dir = base_data_dir
        self.compile_cache_dir = compile_cache_dir
        self.git_snapshotter = git_snapshotter
        self.api_path = api_path
        self.compiler: CachedCompiler = self._build_compiler()
        self.query_validator: Optional[Any] = query_validator
        self.wandb_metrics_hook = wandb_metrics_hook
        self.parse_out_and_validate_output = parse_out_and_validate_output
        self.only_from_cache = only_from_cache
        self.target_cpu = target_cpu
        self.emit_vectorization_reports = emit_vectorization_reports
        return None

    def _build_compiler(self) -> CachedCompiler:
        return make_compiler(
            self.cwd,
            compile_cache_dir=self.compile_cache_dir,
            git_snapshotter=self.git_snapshotter,
            api_path=self.api_path,
        )

    def _build_execution_compiler(
        self,
        required_query_ids: Optional[List[str]],
    ) -> CachedCompiler:
        """Build a compiler that validates only the query entrypoints being run."""
        return make_compiler(
            self.cwd,
            compile_cache_dir=self.compile_cache_dir,
            git_snapshotter=self.git_snapshotter,
            api_path=self.api_path,
            validate_requested_query_modules=True,
            required_query_ids=required_query_ids,
        )

    def _build_compile_flags(
        self,
        *,
        optimize: bool,
        trace_mode: bool,
        perf_profile: bool = False,
    ) -> list[str]:
        """Build compile flags for one run, including optional vectorization diagnostics."""
        if optimize:
            cxx_flags: list[str] = ["-O3", "-flto"]
            if self.emit_vectorization_reports or self.target_cpu not in (None, ""):
                vectorization_bundle = build_vectorization_flag_bundle(
                    build_dir=self.cwd / "build",
                    target_cpu=self.target_cpu,
                )
                cxx_flags.extend(
                    [str(flag) for flag in vectorization_bundle["flags"]]
                )
        else:
            cxx_flags = ["-O2"]
        if trace_mode:
            cxx_flags.append("-DTRACE")
        if perf_profile:
            cxx_flags.extend(["-g", "-fno-omit-frame-pointer"])
        return cxx_flags

    def _compile_for_execution(
        self,
        *,
        optimize: bool,
        trace_mode: bool,
        force_compile: bool,
        current_git_snapshot: Optional[str],
        required_query_ids: Optional[List[str]] = None,
        perf_profile: bool = False,
    ) -> tuple[Optional[str], bool, str]:
        """Compile the database with the flags required for the current execution mode."""
        cxx_flags = self._build_compile_flags(
            optimize=optimize,
            trace_mode=trace_mode,
            perf_profile=perf_profile,
        )
        self.compiler = self._build_execution_compiler(required_query_ids)
        self.compiler.set_extra_cxxflags(cxx_flags)
        return self.compiler.build_cached(
            skip_cache=force_compile,
            current_git_snapshot=current_git_snapshot,
            only_from_cache=self.only_from_cache,
        )

    def run_hardware_counter_capture(
        self,
        *,
        scale_factor: float,
        optimize: bool,
        hardware_counter_preflight: HardwareCounterPreflight,
        stdin_args_data: List[str],
        query_id: Optional[List[str]] = None,
        force_compile: bool = False,
        current_git_snapshot: Optional[str] = None,
    ) -> HardwareCounterSummary:
        """Execute one run under the configured PMU backend and parse real counter output."""
        if not stdin_args_data:
            raise RuntimeError(
                "run_hardware_counter_capture requires explicit stdin_args_data"
            )
        if scale_factor >= 1:
            assert int(scale_factor) == scale_factor, (
                "Scale factor has to be integer >= 1"
            )
            scale_factor = int(scale_factor)
        err, _compile_used_cache, _compile_key_hash = self._compile_for_execution(
            optimize=optimize,
            trace_mode=False,
            force_compile=force_compile,
            current_git_snapshot=current_git_snapshot,
            required_query_ids=query_id,
        )
        if err is not None:
            raise RuntimeError(err)
        self._ensure_runtime_health()
        data_path = resolve_runtime_data_path(
            dataset_name=self.dataset_name,
            base_data_dir=self.base_data_dir,
            scale_factor=scale_factor,
        )
        executable_cmd = ["./db", data_path]
        command = build_hardware_counter_invocation(
            preflight=hardware_counter_preflight,
            executable_cmd=executable_cmd,
        )
        timeout = _approx_timeout_for_hardware_counter_capture(
            scale_factor=scale_factor,
            num_queries=len(stdin_args_data),
        )
        batch_payload = "".join(f"{arg}\n" for arg in stdin_args_data)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=self.cwd,
            prefix="query_batch_",
            suffix=".txt",
            delete=False,
        ) as batch_file:
            batch_file.write(batch_payload)
            batch_file_path = batch_file.name
        command_text = " ".join(shlex.quote(part) for part in command)
        runner = self._single_use_runner_factory(
            command_text,
            extra_env={"TPCH_MONETDB_QUERY_BATCH_FILE": batch_file_path},
        )
        try:
            resp, out, err = runner.run_batch(stdin_args_data, timeout=timeout)
            drain_out, drain_err = runner.terminate_and_drain(suppress_errors=True)
        finally:
            Path(batch_file_path).unlink(missing_ok=True)
        combined_text = "\n".join(
            part for part in (err, out, resp, drain_err, drain_out) if part
        )
        if "exit_code: 0 signal: 0" not in resp:
            raise RuntimeError(
                "hardware counter execution failed: "
                f"response={resp}, stdout={out}{drain_out}, stderr={err}{drain_err}"
            )
        summary = parse_perf_stat_csv(
            combined_text,
            backend=hardware_counter_preflight.backend,
            provenance={
                "query_ids": [] if query_id is None else list(query_id),
                "scale_factor": scale_factor,
                "command": list(command),
                "runner_cmd": hardware_counter_preflight.runner_cmd,
                "host_kernel": hardware_counter_preflight.host_kernel,
                "perf_event_paranoid": hardware_counter_preflight.perf_event_paranoid,
                "large_sf": hardware_counter_preflight.large_sf,
                "target_cpu": hardware_counter_preflight.target_cpu,
            },
        )
        validate_hardware_counter_summary(
            summary,
            required_events=hardware_counter_preflight.required_events,
        )
        return summary

    def run_perf_hotspot_capture(
        self,
        *,
        scale_factor: float,
        optimize: bool,
        hardware_counter_preflight: HardwareCounterPreflight,
        stdin_args_data: List[str],
        query_id: Optional[List[str]] = None,
        force_compile: bool = False,
        current_git_snapshot: Optional[str] = None,
        hotspot_repetitions: int = DEFAULT_PERF_HOTSPOT_REPETITIONS,
    ) -> PerfHotspotSummary:
        """Execute one run under perf record/script and parse call-stack hotspots."""
        if not stdin_args_data:
            raise RuntimeError(
                "run_perf_hotspot_capture requires explicit stdin_args_data"
            )
        if hotspot_repetitions < 1:
            raise RuntimeError("run_perf_hotspot_capture requires repetitions >= 1")
        if scale_factor >= 1:
            assert int(scale_factor) == scale_factor, (
                "Scale factor has to be integer >= 1"
            )
            scale_factor = int(scale_factor)
        err, _compile_used_cache, _compile_key_hash = self._compile_for_execution(
            optimize=optimize,
            trace_mode=False,
            force_compile=force_compile,
            current_git_snapshot=current_git_snapshot,
            required_query_ids=query_id,
            perf_profile=True,
        )
        if err is not None:
            raise RuntimeError(err)
        self._ensure_runtime_health()
        data_path = resolve_runtime_data_path(
            dataset_name=self.dataset_name,
            base_data_dir=self.base_data_dir,
            scale_factor=scale_factor,
        )
        executable_cmd = ["./db", data_path]
        capture_args = [
            arg
            for _repeat_index in range(hotspot_repetitions)
            for arg in stdin_args_data
        ]
        timeout = _approx_timeout_for_perf_hotspot_capture(
            scale_factor=scale_factor,
            num_queries=len(stdin_args_data),
            hotspot_repetitions=hotspot_repetitions,
        )
        batch_payload = "".join(f"{arg}\n" for arg in capture_args)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=self.cwd,
            prefix="query_batch_",
            suffix=".txt",
            delete=False,
        ) as batch_file:
            batch_file.write(batch_payload)
            batch_file_path = batch_file.name
        artifact_dir = self.cwd / "build" / "perf_hotspots"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        query_label = "_".join([] if query_id is None else query_id) or "unknown"
        capture_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        perf_data_path = str(
            artifact_dir / f"q{query_label}_{capture_stamp}.perf.data"
        )
        perf_script_path = str(
            artifact_dir / f"q{query_label}_{capture_stamp}.perf.script.txt"
        )
        source_line_decode = True
        try:
            Path(perf_data_path).unlink(missing_ok=True)
            runner = self._single_use_runner_factory(
                " ".join(shlex.quote(part) for part in executable_cmd),
                extra_env={"TPCH_MONETDB_QUERY_BATCH_FILE": batch_file_path},
            )
            runner_pid = runner.start_for_external_control()
            warmup_resp, warmup_out, warmup_err = runner.run_batch(
                stdin_args_data,
                timeout=timeout,
            )
            if "exit_code: 0 signal: 0" not in warmup_resp:
                raise RuntimeError(
                    "perf hotspot warmup failed before attach: "
                    f"response={warmup_resp}, stdout={warmup_out}, stderr={warmup_err}"
                )
            attached_pids = collect_process_tree_pids(runner_pid)
            record_command = build_perf_record_invocation(
                preflight=hardware_counter_preflight,
                attach_pids=attached_pids,
                output_path=perf_data_path,
            )
            script_command = build_perf_script_invocation(
                preflight=hardware_counter_preflight,
                input_path=perf_data_path,
            )
            record_proc = subprocess.Popen(
                record_command,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=self.cwd,
            )
            try:
                resp, out, err = runner.run_batch(capture_args, timeout=timeout)
            finally:
                record_proc.terminate()
            try:
                record_stdout, record_stderr = record_proc.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                record_proc.kill()
                record_stdout, record_stderr = record_proc.communicate(timeout=10)
            drain_out, drain_err = runner.terminate_and_drain(suppress_errors=True)
            if "exit_code: 0 signal: 0" not in resp:
                raise RuntimeError(
                    "perf hotspot record failed: "
                    f"response={resp}, stdout={out}{drain_out}, stderr={err}{drain_err}"
                )
            if record_proc.returncode not in (0, -15, -2):
                raise RuntimeError(
                    "perf hotspot record failed: "
                    f"returncode={record_proc.returncode}, "
                    f"stdout={record_stdout}, stderr={record_stderr}"
                )
            script_proc = subprocess.run(
                script_command,
                text=True,
                capture_output=True,
                cwd=self.cwd,
                timeout=timeout,
            )
            if script_proc.returncode != 0:
                source_line_decode = False
                script_command = build_perf_script_invocation(
                    preflight=hardware_counter_preflight,
                    input_path=perf_data_path,
                    include_source_lines=False,
                )
                script_proc = subprocess.run(
                    script_command,
                    text=True,
                    capture_output=True,
                    cwd=self.cwd,
                    timeout=timeout,
                )
        finally:
            Path(batch_file_path).unlink(missing_ok=True)
        if script_proc.returncode != 0:
            raise RuntimeError(
                "perf hotspot script failed: "
                f"returncode={script_proc.returncode}, "
                f"stdout={script_proc.stdout}, stderr={script_proc.stderr}"
            )
        script_text = "\n".join(
            part for part in (script_proc.stdout, script_proc.stderr) if part
        )
        Path(perf_script_path).write_text(script_text, encoding="utf-8")
        return parse_perf_script_hotspots(
            script_text,
            backend=hardware_counter_preflight.backend,
            provenance={
                "query_ids": [] if query_id is None else list(query_id),
                "scale_factor": scale_factor,
                "capture_scope": "query_loop_only",
                "warmup_completed": True,
                "record_started_after_warmup": True,
                "record_command": list(record_command),
                "script_command": list(script_command),
                "perf_data_path": perf_data_path,
                "perf_script_path": perf_script_path,
                "attached_pid": runner_pid,
                "attached_pids": attached_pids,
                "attached_descendant_pids": attached_pids[1:],
                "record_returncode": record_proc.returncode,
                "runner_cmd": hardware_counter_preflight.runner_cmd,
                "host_kernel": hardware_counter_preflight.host_kernel,
                "perf_event_paranoid": hardware_counter_preflight.perf_event_paranoid,
                "large_sf": hardware_counter_preflight.large_sf,
                "target_cpu": hardware_counter_preflight.target_cpu,
                "hotspot_event": DEFAULT_PERF_HOTSPOT_EVENT,
                "hotspot_frequency": DEFAULT_PERF_HOTSPOT_FREQUENCY,
                "hotspot_repetitions": hotspot_repetitions,
                "warmup_query_repetitions": len(stdin_args_data),
                "measured_query_repetitions": len(capture_args),
                "measured_batch_size": len(capture_args),
                "source_line_decode": source_line_decode,
            },
            perf_data_path=perf_data_path,
            perf_script_path=perf_script_path,
        )

    def run(
        self,
        scale_factor: float,
        optimize: bool,
        query_id: Optional[List[str]] = None,
        trace_mode: bool = False,  # set trace flag
        external_call: bool = False,  # only for logging purposes
        force_fresh_validation: bool = False,
        output_mode: str = QUERY_OUTPUT_MODE_FULL_CSV,
        enable_performance_comparison: bool = False,
    ) -> Tuple[str, Optional[Dict]]:
        try:
            run_result = self.run_worker(
                scale_factor=scale_factor,
                optimize=optimize,
                query_id=query_id,
                trace_mode=trace_mode,
                external_call=external_call,
                force_fresh_validation=force_fresh_validation,
                output_mode=output_mode,
                enable_performance_comparison=enable_performance_comparison,
            )
        except FileNotFoundError as exc:
            db_path = self.cwd / "db"
            if not db_path.exists():
                # run with force compile to make sure ./db file exists (and not skipped because of caching)
                run_result = self.run_worker(
                    scale_factor=scale_factor,
                    optimize=optimize,
                    query_id=query_id,
                    trace_mode=trace_mode,
                    force_compile=True,
                    external_call=external_call,
                    force_fresh_validation=force_fresh_validation,
                    output_mode=output_mode,
                    enable_performance_comparison=enable_performance_comparison,
                )
            else:
                raise

        return run_result.msg, run_result.metrics

    def _cleanup_stale_result_files(self) -> None:
        stale_paths = sorted(self.cwd.glob("result*.csv"))
        for stale_path in stale_paths:
            stale_path.unlink(missing_ok=True)
        return None

    def reset_runtime_state(self, *, clean_reload: bool = False) -> None:
        cwd_text = str(self.cwd)
        FastTestPool.terminate_matching(
            lambda key: key.startswith("./db ") or cwd_text in key,
            suppress_errors=True,
        )
        if clean_reload:
            cleanup_reload_dir(self.cwd)
        return None

    def _ensure_runtime_health(self) -> None:
        report = inspect_runtime_health(self.cwd)
        if report.healthy:
            return None
        self.reset_runtime_state(clean_reload=True)
        after = inspect_runtime_health(self.cwd)
        if not after.healthy:
            raise RuntimeError(
                f"[ERROR:INFRA_BLOCKED] runtime health failed after cleanup: {after}"
            )
        return None

    def _runtime_artifact_paths(self) -> dict[str, Path]:
        workdir = getattr(self.compiler, "workdir", self.cwd)
        app_name = getattr(self.compiler, "app_name", "db")
        build_dir_path = getattr(self.compiler, "build_dir_path", self.cwd / "build")
        libs = getattr(self.compiler, "libs", {"query": None})
        paths = {"app": workdir / app_name}
        paths.update(
            {
                f"lib{lib}.so": build_dir_path / f"lib{lib}.so"
                for lib in sorted(libs.keys())
            }
        )
        return paths

    def _artifact_identity(self, path: Path) -> str:
        if not path.exists():
            return "missing"
        artifact_build_id = read_build_id(path)
        if artifact_build_id is not None:
            return f"build-id:{artifact_build_id}"
        stat = path.stat()
        return f"stat:{stat.st_mtime_ns}:{stat.st_size}"

    def _runtime_identity(self, compile_key_hash: str) -> str:
        payload = {
            "compile_key_hash": compile_key_hash,
            "artifacts": {
                name: self._artifact_identity(path)
                for name, path in self._runtime_artifact_paths().items()
            },
        }
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()[:32]

    def _query_scope_key(self, query_id: Optional[List[str]]) -> str:
        if query_id is None:
            return "all"
        cleaned_query_ids = [str(item).strip() for item in query_id]
        if not cleaned_query_ids:
            return "empty"
        return ",".join(cleaned_query_ids)

    def _pool_key(
        self,
        *,
        cmd: str,
        compile_key_hash: str,
        query_id: Optional[List[str]],
        optimize: bool,
        trace_mode: bool,
        output_mode: str,
    ) -> str:
        return (
            f"{cmd} | cwd={self.cwd} | runtime={self._runtime_identity(compile_key_hash)} "
            f"| scope={self._query_scope_key(query_id)} | optimize={int(optimize)} "
            f"| trace={int(trace_mode)} | output={output_mode}"
        )

    def _terminate_stale_runners_for_command(self, *, cmd: str, pool_key: str) -> None:
        pool_prefix = f"{cmd} | cwd={self.cwd} | "
        FastTestPool.terminate_matching(
            lambda key: key.startswith(pool_prefix) and key != pool_key,
            suppress_errors=True,
        )
        return None

    def _trace_output_path(self) -> Path:
        """Return the bounded trace file path owned by the Python runner."""
        return self.cwd / TRACE_OUTPUT_FILENAME

    def _prepare_trace_output_file(self, *, trace_mode: bool) -> None:
        """Clear stale raw trace evidence before a trace-mode execution."""
        if not trace_mode:
            return None
        self._trace_output_path().unlink(missing_ok=True)
        return None

    def _runner_env(self, *, output_mode: str, trace_mode: bool) -> dict[str, str]:
        """Build the child-process environment for output and trace controls."""
        env = {"TPCH_MONETDB_QUERY_OUTPUT_MODE": output_mode}
        if trace_mode:
            env["TPCH_MONETDB_TRACE_OUTPUT_PATH"] = str(self._trace_output_path())
            env["TPCH_MONETDB_TRACE_APPEND"] = "0"
        return env

    def _runner_factory(
        self,
        cmd: str,
        *,
        output_mode: str,
        trace_mode: bool,
    ) -> FasttestProc:
        """Build a persistent runner with the requested output and trace modes."""
        return FasttestProc(
            cmd,
            echo_output=True,
            cwd=self.cwd,
            extra_env=self._runner_env(
                output_mode=output_mode,
                trace_mode=trace_mode,
            ),
        )

    def _single_use_runner_factory(
        self,
        cmd: str,
        *,
        extra_env: dict[str, str] | None = None,
    ) -> FasttestProc:
        return FasttestProc(
            cmd,
            echo_output=False,
            cwd=self.cwd,
            extra_env=extra_env,
        )

    def _trim_failure_detail(self, text: str) -> str:
        detail = text.strip()
        if len(detail) <= 4000:
            return detail
        return detail[-4000:]

    def _raise_for_infra_failure(
        self,
        *,
        pool_key: str,
        resp: str,
        out: str,
        err: str,
    ) -> None:
        text = "\n".join(part for part in (resp, out, err) if part)
        failure_code = classify_infra_failure(text)
        if failure_code is None:
            return None
        try:
            FastTestPool.terminate(pool_key)
        except Exception as exc:
            logger.warning(f"Failed to terminate unhealthy runner: {exc}")
        self._cleanup_stale_result_files()
        raise RunnerInfraFailureError(failure_code, self._trim_failure_detail(text))

    def run_raw_worker(
        self,
        scale_factor: float,
        optimize: bool,
        query_id: Optional[List[str]] = None,
        trace_mode: bool = False,
        force_compile: bool = False,
        external_call: bool = False,
        stdin_args_data: Optional[List[str]] = None,
        current_git_snapshot: Optional[str] = None,
        output_mode: str = QUERY_OUTPUT_MODE_FULL_CSV,
        execution_timeout_s: Optional[int] = None,
    ) -> RunWorkerResult:
        """Execute one or more explicit query args without validator/cache mediation."""
        if stdin_args_data is None:
            raise RuntimeError(
                "run_raw_worker requires stdin_args_data to bypass validation cache"
            )
        return self.run_worker(
            scale_factor=scale_factor,
            optimize=optimize,
            query_id=query_id,
            trace_mode=trace_mode,
            force_compile=force_compile,
            external_call=external_call,
            stdin_args_data=stdin_args_data,
            current_git_snapshot=current_git_snapshot,
            output_mode=output_mode,
            execution_timeout_s=execution_timeout_s,
        )

    def run_worker(
        self,
        scale_factor: float,
        optimize: bool,
        query_id: Optional[List[str]] = None,
        trace_mode: bool = False,  # set trace flag
        force_compile: bool = False,
        external_call: bool = False,
        force_fresh_validation: bool = False,
        stdin_args_data: Optional[List[str]] = None,
        current_git_snapshot: Optional[
            str
        ] = None,  # for external instrumentation: e.g. from benchmarking script (will not use git snapshotter)
        output_mode: str = QUERY_OUTPUT_MODE_FULL_CSV,
        enable_performance_comparison: bool = False,
        execution_timeout_s: Optional[int] = None,
    ) -> RunWorkerResult:
        """Compile the current implementation, execute requested queries, and validate output."""
        normalized_output_mode = _normalize_query_output_mode(output_mode)
        if scale_factor >= 1:
            # it has to be an int
            assert int(scale_factor) == scale_factor, (
                "Scale factor has to be integer >= 1"
            )
            scale_factor = int(scale_factor)

        # check that scalefactor is prepared / available in validator
        if (
            self.query_validator is not None
            and scale_factor not in self.query_validator.sf_list
            and stdin_args_data
            is None  # if manual stdin args are provided, we skip the check and just execute (e.g. for testing purposes
        ):
            metrics = _assemble_validation_error(
                scale_factor=scale_factor,
                query_ids_executed=query_id if query_id is not None else [],
            )
            metrics["type"] = "validate"
            metrics["validation/fasttest_optimize"] = optimize
            metrics["validation/trace_mode"] = trace_mode
            metrics["validation/compile_error"] = True
            metrics["validation/external_call"] = external_call
            if self.wandb_metrics_hook is not None:
                self.wandb_metrics_hook.log_metrics_callback(
                    metrics, log_and_increment=True
                )
            return RunWorkerResult(
                msg=f"Scale factor {scale_factor} not available in query validator (not prepared). Available scale factors: {self.query_validator.sf_list}",
                metrics=metrics,
            )

        if (
            self.query_validator is not None
            and stdin_args_data is None
            and normalized_output_mode != QUERY_OUTPUT_MODE_FULL_CSV
        ):
            raise RuntimeError(
                "Correctness validation requires output_mode='full_csv' because "
                "the validator consumes result<RUN_NR>.csv files."
            )

        if stdin_args_data is not None:
            logger.warning(
                "Launching with manual stdin args data. Query-Validator will not be invoked!"
            )
        err, compile_used_cache, compile_key_hash = self._compile_for_execution(
            optimize=optimize,
            trace_mode=trace_mode,
            force_compile=force_compile,
            current_git_snapshot=current_git_snapshot,
            required_query_ids=query_id,
        )
        if err is not None:
            if self.wandb_metrics_hook is not None:
                metrics = _assemble_validation_error(
                    scale_factor=scale_factor,
                    query_ids_executed=query_id if query_id is not None else [],
                )
                metrics["type"] = "validate"
                metrics["validation/fasttest_optimize"] = optimize
                metrics["validation/trace_mode"] = trace_mode
                metrics["validation/compile_error"] = True
                metrics["validation/external_call"] = external_call
                self.wandb_metrics_hook.log_metrics_callback(
                    metrics, log_and_increment=True
                )
            return RunWorkerResult(msg=err, err=err)

        self._ensure_runtime_health()

        data_path = resolve_runtime_data_path(
            dataset_name=self.dataset_name,
            base_data_dir=self.base_data_dir,
            scale_factor=scale_factor,
        )
        cmd = f"./db {data_path}"
        pool_key = self._pool_key(
            cmd=cmd,
            compile_key_hash=compile_key_hash,
            query_id=query_id,
            optimize=optimize,
            trace_mode=trace_mode,
            output_mode=normalized_output_mode,
        )
        performance_output_mode = (
            QUERY_OUTPUT_MODE_NO_OUTPUT
            if (
                self.query_validator is not None
                and stdin_args_data is None
                and normalized_output_mode == QUERY_OUTPUT_MODE_FULL_CSV
            )
            else normalized_output_mode
        )
        performance_pool_key = self._pool_key(
            cmd=cmd,
            compile_key_hash=compile_key_hash,
            query_id=query_id,
            optimize=optimize,
            trace_mode=trace_mode,
            output_mode=performance_output_mode,
        )
        logger.info(
            f"Run with: {query_id=} {scale_factor=} {self.dataset_name=} "
            f"{trace_mode=} {optimize=} validation_output_mode={normalized_output_mode} "
            f"performance_output_mode={performance_output_mode} "
            f"{self.base_data_dir=}"
        )
        self._terminate_stale_runners_for_command(cmd=cmd, pool_key=pool_key)
        self._cleanup_stale_result_files()
        self._prepare_trace_output_file(trace_mode=trace_mode)

        def _exec_callback_for_output_mode(
            args_list: List[str],
            timeout_s: int,
            *,
            callback_output_mode: str,
            callback_pool_key: str,
        ) -> Tuple[str, str, str]:
            if trace_mode:
                self._prepare_trace_output_file(trace_mode=True)
                FastTestPool.terminate(callback_pool_key)
            try:
                runner = FastTestPool.get(
                    callback_pool_key,
                    lambda: self._runner_factory(
                        cmd,
                        output_mode=callback_output_mode,
                        trace_mode=trace_mode,
                    ),
                )
                resp, out, err = runner.run_batch(args_list, timeout=timeout_s)
                logger.info(f"resp={resp.rstrip()}")
                self._raise_for_infra_failure(
                    pool_key=callback_pool_key,
                    resp=resp,
                    out=out,
                    err=err,
                )
                return resp, out, err
            except RunnerTransportError as original_exc:
                logger.warning(
                    "Runner transport failed; attempting one fresh-runner replay"
                )
                # Destroy dead runner and create a fresh one
                FastTestPool.terminate(callback_pool_key)
                fresh_runner = FastTestPool.get(
                    callback_pool_key,
                    lambda: self._runner_factory(
                        cmd,
                        output_mode=callback_output_mode,
                        trace_mode=trace_mode,
                    ),
                )
                try:
                    resp, out, err = fresh_runner.run_batch(
                        args_list,
                        timeout=timeout_s,
                    )
                    logger.info(f"resp={resp.rstrip()}")
                    self._raise_for_infra_failure(
                        pool_key=callback_pool_key,
                        resp=resp,
                        out=out,
                        err=err,
                    )
                    return resp, out, err
                except RunnerTransportError as exc:
                    raise RunnerBrokenPipePersistentError(
                        "[ERROR:RUNNER_BROKEN_PIPE] "
                        "Runner transport failed after one fresh-runner replay. "
                        f"Original error: {original_exc}; replay error: {exc}"
                ) from exc
            finally:
                if trace_mode:
                    FastTestPool.terminate(callback_pool_key)

        def _exec_callback_with_transport_recovery(
            args_list: List[str],
            timeout_s: int,
        ) -> Tuple[str, str, str]:
            return _exec_callback_for_output_mode(
                args_list,
                timeout_s,
                callback_output_mode=normalized_output_mode,
                callback_pool_key=pool_key,
            )

        def _performance_exec_callback_with_transport_recovery(
            args_list: List[str],
            timeout_s: int,
        ) -> Tuple[str, str, str]:
            return _exec_callback_for_output_mode(
                args_list,
                timeout_s,
                callback_output_mode=performance_output_mode,
                callback_pool_key=performance_pool_key,
            )

        # callback executing the query
        exec_callback = _exec_callback_with_transport_recovery

        # validate output correctness
        # in case query-validator is not provided or manual-stdin args are provided, just execute without validation
        if self.query_validator and stdin_args_data is None:
            try:
                msg, success, metrics, exec_used_cache = (
                    self.query_validator.exec_and_validate(
                        exec_callback_fn=exec_callback,
                        scale_factor=scale_factor,
                        query_id=query_id,
                        other_config={
                            "optimize": optimize,
                            "validation_output_mode": normalized_output_mode,
                            "performance_output_mode": performance_output_mode,
                            "enable_performance_comparison": enable_performance_comparison,
                        },
                        skip_validate=not self.parse_out_and_validate_output,
                        compile_key_hash=compile_key_hash,
                        trace_mode=trace_mode,
                        only_from_cache=self.only_from_cache,
                        skip_validate_cache=force_fresh_validation,
                        performance_exec_callback_fn=(
                            _performance_exec_callback_with_transport_recovery
                            if (
                                enable_performance_comparison
                                and performance_output_mode != normalized_output_mode
                            )
                            else None
                        ),
                    )
                )
            except RunnerInfraFailureError as exc:
                metrics = _assemble_validation_error(
                    scale_factor=scale_factor,
                    query_ids_executed=query_id if query_id is not None else [],
                )
                metrics["type"] = "validate"
                metrics["validation/failure_code"] = exc.failure_code
                metrics["validation/failure_detail"] = exc.detail
                metrics["validation/fasttest_optimize"] = optimize
                metrics["validation/trace_mode"] = trace_mode
                metrics["validation/external_call"] = external_call
                return RunWorkerResult(msg=str(exc), metrics=metrics, err=str(exc))
            except RunnerBrokenPipePersistentError as exc:
                metrics = _assemble_validation_error(
                    scale_factor=scale_factor,
                    query_ids_executed=query_id if query_id is not None else [],
                )
                metrics["type"] = "validate"
                metrics["validation/failure_code"] = "RUNNER_BROKEN_PIPE"
                metrics["validation/failure_detail"] = str(exc)
                metrics["validation/fasttest_optimize"] = optimize
                metrics["validation/trace_mode"] = trace_mode
                metrics["validation/external_call"] = external_call
                return RunWorkerResult(msg=str(exc), metrics=metrics, err=str(exc))

            # this assertion does unfortunately not work: it is valid that args for validate change, but compile is the same. E.g. different scale factors.
            # assert compile_used_cache == exec_used_cache, (
            #     "Inconsistent cache usage between compile and execute. This should always be chained! If this happens, potentially a change in the wrapper code/... happened. Please delete both cache entries (compile & exec), check your changes and re-run."
            # )
            if exec_used_cache and not compile_used_cache:
                logger.warning(
                    "Validation cache hit while compile was rebuilt (compile cache miss). "
                    "Continuing because compile_key_hash is unchanged."
                )
            resp = None
            out = None
            err = None
        else:
            logger.warning(
                "No query validator provided, just executing the query without validation!"
            )

            if stdin_args_data is None:
                default_query_id = "1"
                if query_id is not None and len(query_id) > 0:
                    default_query_id = query_id[0]
                stdin_args_data = [default_query_id]

            timeout = (
                execution_timeout_s
                if execution_timeout_s is not None
                else _approx_timeout_for_validation(
                    scale_factor=scale_factor,
                    num_queries=len(stdin_args_data),
                    repetitions=1,
                    num_random_query_instantiations=1,
                )
            )

            try:
                resp, out, err = exec_callback(stdin_args_data, timeout_s=timeout)
            except RunnerInfraFailureError as exc:
                metrics = _assemble_validation_error(
                    scale_factor=scale_factor,
                    query_ids_executed=query_id if query_id is not None else [],
                )
                metrics["type"] = "validate"
                metrics["validation/failure_code"] = exc.failure_code
                metrics["validation/failure_detail"] = exc.detail
                metrics["validation/fasttest_optimize"] = optimize
                metrics["validation/trace_mode"] = trace_mode
                metrics["validation/external_call"] = external_call
                return RunWorkerResult(msg=str(exc), metrics=metrics, err=str(exc))
            except RunnerBrokenPipePersistentError as exc:
                metrics = _assemble_validation_error(
                    scale_factor=scale_factor,
                    query_ids_executed=query_id if query_id is not None else [],
                )
                metrics["type"] = "validate"
                metrics["validation/failure_code"] = "RUNNER_BROKEN_PIPE"
                metrics["validation/failure_detail"] = str(exc)
                metrics["validation/fasttest_optimize"] = optimize
                metrics["validation/trace_mode"] = trace_mode
                metrics["validation/external_call"] = external_call
                return RunWorkerResult(msg=str(exc), metrics=metrics, err=str(exc))
            msg = _format_run_envelope(stdout=out, stderr=err, response=resp)
            metrics = None

        if self.wandb_metrics_hook is not None and metrics is not None:
            metrics["type"] = "validate"
            metrics["validation/fasttest_optimize"] = optimize
            metrics["validation/trace_mode"] = trace_mode
            metrics["validation/external_call"] = external_call
            self.wandb_metrics_hook.log_metrics_callback(
                metrics, log_and_increment=True
            )

        return RunWorkerResult(msg=msg, metrics=metrics, resp=resp, out=out, err=err)

    def __call__(
        self,
        scale_factor: float,
        optimize: bool,
        query_id: Optional[List[str]] = None,
        trace_mode: bool = False,  # sets trace flag for the run
        enable_performance_comparison: bool = False,
    ) -> str:
        return self.run(
            scale_factor=scale_factor,
            optimize=optimize,
            query_id=query_id,
            trace_mode=trace_mode,
            enable_performance_comparison=enable_performance_comparison,
        )[0]


class RunArgs(BaseModel):
    scale_factor: int = Field(..., ge=1, description="Scale factor (>= 1)")
    optimize: bool = Field(..., description="Enable compiler optimization")
    query_id: List[str] | None = Field(
        None,
        description="List of Query-IDs to execute. None means all queries.",
    )


class IMDBRunArgs(BaseModel):
    scale_factor: float = Field(..., gt=0, description="Scale factor (> 0)")
    optimize: bool = Field(..., description="Enable compiler optimization")
    query_id: List[str] | None = Field(
        None,
        description="List of Query-IDs to execute. None means all queries.",
    )


trace_flag_description = "Whether to set TRACE flag for the run (setting cxx flag -DTRACE, e.g. enables collecting execution statistics for code optimization if implemented in the codebase)"


class RunArgsTrace(RunArgs):
    trace_mode: bool = Field(
        False,
        description=trace_flag_description,
    )


class IMDBRunArgsTrace(IMDBRunArgs):
    trace_mode: bool = Field(
        False,
        description=trace_flag_description,
    )


def make_run_tool(
    cwd: Path,
    dataset_name: str,
    base_data_dir: str,  # must contain per scale-factors subdirs: e.g. base_data_dir/sf1/, base_data_dir/sf10/..., each containing the corresponding data files for the scale factor
    query_validator: Optional[Any] = None,
    wandb_metrics_hook: Optional[WandbRunHook] = None,
    compile_cache_dir: Optional[Path] = None,
    git_snapshotter: Any = None,
    run_tool_offer_trace_option: bool = False,
    only_from_cache: bool = False,
    target_cpu: Optional[str] = None,
    emit_vectorization_reports: bool = False,
) -> Tuple[FunctionTool, RunTool]:
    impl = RunTool(
        cwd,
        query_validator=query_validator,
        wandb_metrics_hook=wandb_metrics_hook,
        compile_cache_dir=compile_cache_dir,
        git_snapshotter=git_snapshotter,
        dataset_name=dataset_name,
        base_data_dir=base_data_dir,
        only_from_cache=only_from_cache,
        target_cpu=target_cpu,
        emit_vectorization_reports=emit_vectorization_reports,
    )

    def get_args_model():
        if dataset_name == "imdb":
            return IMDBRunArgsTrace if run_tool_offer_trace_option else IMDBRunArgs
        else:
            return RunArgsTrace if run_tool_offer_trace_option else RunArgs

    args_model = get_args_model()

    async def on_invoke(ctx: RunContextWrapper[Any], args_json: str) -> str:
        try:
            parsed = load_function_tool_args(args_json)
            enable_performance_comparison = (
                parsed.pop(_BASE_PERFORMANCE_COMPARISON_ARG, False) is True
            )
            args = args_model.model_validate(parsed)
        except json.JSONDecodeError as e:
            return f"Error: Invalid JSON format. {str(e)}."
        except Exception as e:
            return f"Error running query: {str(e)}"

        if run_tool_offer_trace_option:
            return impl(
                scale_factor=args.scale_factor,
                optimize=args.optimize,
                query_id=args.query_id,
                trace_mode=args.trace_mode,  # type: ignore
                enable_performance_comparison=enable_performance_comparison,
            )
        else:
            return impl(
                scale_factor=args.scale_factor,
                optimize=args.optimize,
                query_id=args.query_id,
                enable_performance_comparison=enable_performance_comparison,
            )

    return FunctionTool(
        name="run",
        description="Runs the database and executes a query by query-id",
        params_json_schema=args_model.model_json_schema(),
        on_invoke_tool=on_invoke,
    ), impl
