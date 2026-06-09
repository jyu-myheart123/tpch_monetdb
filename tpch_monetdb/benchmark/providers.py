"""Runtime Provider 和 Reference Baseline Provider 实现.

根据 phase3 D1 设计，提供两个核心 provider：
1. RuntimeProvider: 测量被测引擎的运行时
2. BaselineProvider: 提供参考基线的运行时
"""

import json
import logging
import hashlib
import statistics
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from tpch_monetdb.benchmark.manifest import (
    QueryInstantiation,
    ReferenceManifest,
    RuntimeMeasurement,
)
from tpch_monetdb.benchmark.runtime_accounting import (
    KERNEL_RUNTIME_METRIC_KIND,
    MEASURED_RUNS,
    OPTIMIZATION_RUNTIME_METRIC_KIND,
    QUERY_RUNTIME_METRIC_KIND,
    WARMUP_RUNS,
    QuerySamples,
    build_runtime_timeout_policy,
    is_lazy_build_suspected,
    parse_ingest_timing_from_text,
    parse_query_timing,
    raise_for_runtime_execution_failure,
)
from tpch_monetdb.dataset.gen_tpch.tpch_queries import get_contract as get_tpch_contract
from tpch_monetdb.oracle.monetdb_oracle import MonetDBOracle
from tpch_monetdb.utils.pipeline_evidence import MeasurementKind, MeasurementShapeStatus

logger = logging.getLogger(__name__)
DEFAULT_RUNTIME_WORKERS = 1
DEFAULT_BENCHMARK_MODE = "system-parity"
DEFAULT_STORAGE_MODE = "persistent"
DEFAULT_BESPOKE_ENGINE = "generated_tpch"
DEFAULT_MONETDB_ENGINE = "monetdb"
DEFAULT_MONETDB_DOCKER_IMAGE = "tpch-monetdb:local"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _build_runtime_measurement(
    instantiation_id: str,
    runtime_ms: float,
    num_runs: int,
    all_runtimes_ms: List[float],
    *,
    benchmark_mode: str,
    storage_mode: str,
    workers: int,
    engine: str,
    provenance: Optional[Dict[str, Any]] = None,
    measurement_kind: str = MeasurementKind.EXACT_INSTANTIATION.value,
    query_id: Optional[str] = None,
    args_string: Optional[str] = None,
    scale_factor: Optional[int] = None,
    row_count: Optional[int] = None,
    output_row_count: Optional[int] = None,
    query_file_sha256: Optional[str] = None,
    measurement_shape_status: str = MeasurementShapeStatus.UNKNOWN.value,
) -> RuntimeMeasurement:
    """Build one runtime measurement with benchmark provenance metadata."""
    return RuntimeMeasurement(
        instantiation_id=instantiation_id,
        runtime_ms=runtime_ms,
        num_runs=num_runs,
        all_runtimes_ms=all_runtimes_ms,
        timestamp=_utc_now_iso(),
        benchmark_mode=benchmark_mode,
        storage_mode=storage_mode,
        workers=workers,
        engine=engine,
        measurement_kind=measurement_kind,
        query_id=query_id,
        args_string=args_string,
        scale_factor=scale_factor,
        row_count=row_count,
        output_row_count=output_row_count,
        query_file_sha256=query_file_sha256,
        measurement_shape_status=measurement_shape_status,
        provenance={} if provenance is None else dict(provenance),
    )


def _file_sha256(path: Path) -> str:
    """Return the SHA256 digest for a baseline input artifact."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class RuntimeProvider(ABC):
    """运行时测量 Provider 抽象基类.
    
    负责从特定来源（引擎或基线）测量 runtime。
    """
    
    @abstractmethod
    def measure(
        self,
        instantiation: QueryInstantiation,
        exec_callback: Optional[Callable[[List[str], int], Tuple[str, str, str]]] = None,
    ) -> RuntimeMeasurement:
        """测量单个查询实例的 runtime.
        
        Args:
            instantiation: 查询实例
            exec_callback: 执行回调（用于被测引擎）
        
        Returns:
            RuntimeMeasurement 包含测量结果
        """
        pass
    
    @abstractmethod
    def measure_batch(
        self,
        instantiations: List[QueryInstantiation],
        exec_callback: Optional[Callable[[List[str], int], Tuple[str, str, str]]] = None,
    ) -> Dict[str, RuntimeMeasurement]:
        """批量测量多个查询实例的 runtime.
        
        Args:
            instantiations: 查询实例列表
            exec_callback: 执行回调（用于被测引擎）
        
        Returns:
            映射: instantiation_id -> RuntimeMeasurement
        """
        pass


@dataclass(frozen=True)
class DockerMonetDBLifecycleConfig:
    """Docker Compose paths and command settings for the MonetDB service."""

    repo_root: Path
    image_tag: str = DEFAULT_MONETDB_DOCKER_IMAGE
    docker_binary: str = "docker"
    dockerfile: Path = Path("docker/tpch-monetdb/Dockerfile")
    compose_file: Path = Path("docker/tpch-monetdb/docker-compose.yml")
    service_name: str = "tpch-monetdb"
    init_service_name: str = "tpch-monetdb-init"
    timeout_s: int = 900


@dataclass(frozen=True)
class DockerMonetDBCommandResult:
    """Structured result for one Docker lifecycle command."""

    stage: str
    command: List[str]
    returncode: int
    stdout: str
    stderr: str
    report: Optional[Dict[str, Any]] = None

    @property
    def ok(self) -> bool:
        """Return whether the Docker command completed successfully."""
        return self.returncode == 0


class DockerMonetDBLifecycle:
    """Build, start, and initialize the Docker Compose MonetDB service."""

    def __init__(
        self,
        config: DockerMonetDBLifecycleConfig,
        run_command: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    ) -> None:
        self.config = config
        self.run_command = run_command
        return None

    def preflight(self) -> DockerMonetDBCommandResult:
        """Check Docker Compose availability and required files."""
        dockerfile = self._dockerfile_path()
        if not dockerfile.is_file():
            return DockerMonetDBCommandResult(
                stage="preflight",
                command=[],
                returncode=1,
                stdout="",
                stderr=f"Dockerfile not found: {dockerfile}",
                report={"status": "failed", "reason": "dockerfile_missing", "dockerfile": str(dockerfile)},
            )
        compose_file = self._compose_file_path()
        if not compose_file.is_file():
            return DockerMonetDBCommandResult(
                stage="preflight",
                command=[],
                returncode=1,
                stdout="",
                stderr=f"Docker Compose file not found: {compose_file}",
                report={
                    "status": "failed",
                    "reason": "compose_file_missing",
                    "compose_file": str(compose_file),
                },
            )
        command = [self.config.docker_binary, "compose", "version", "--short"]
        result = self._run("preflight", command, timeout_s=30)
        if result.ok:
            report = {
                "status": "ok",
                "docker_compose_version": result.stdout.strip(),
                "dockerfile": str(dockerfile),
                "compose_file": str(compose_file),
            }
            return DockerMonetDBCommandResult(
                stage=result.stage,
                command=result.command,
                returncode=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
                report=report,
            )
        return DockerMonetDBCommandResult(
            stage=result.stage,
            command=result.command,
            returncode=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
            report={"status": "failed", "reason": "docker_unavailable"},
        )

    def build_image(self) -> DockerMonetDBCommandResult:
        """Build the Compose MonetDB service image."""
        command = [
            self.config.docker_binary,
            "compose",
            "-f",
            str(self._compose_file_path()),
            "build",
            self.config.service_name,
        ]
        result = self._run("build", command, timeout_s=self.config.timeout_s)
        report = {
            "status": "ok" if result.ok else "failed",
            "image_tag": self.config.image_tag,
            "service": self.config.service_name,
        }
        return DockerMonetDBCommandResult(
            stage=result.stage,
            command=result.command,
            returncode=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
            report=report,
        )

    def start_service(self) -> DockerMonetDBCommandResult:
        """Start the Compose MonetDB service.

        The current Docker path is compose-first and does not run a repository
        startup or validation script.
        """
        command = [
            self.config.docker_binary,
            "compose",
            "-f",
            str(self._compose_file_path()),
            "up",
            "-d",
            self.config.service_name,
        ]
        result = self._run("up", command, timeout_s=self.config.timeout_s)
        report = {
            "status": "ok" if result.ok else "failed",
            "service": self.config.service_name,
        }
        return DockerMonetDBCommandResult(
            stage=result.stage,
            command=result.command,
            returncode=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
            report=report,
        )

    def init_tpch(self) -> DockerMonetDBCommandResult:
        """Run the one-shot Compose service that imports the bundled TPC-H fixture."""
        command = [
            self.config.docker_binary,
            "compose",
            "-f",
            str(self._compose_file_path()),
            "--profile",
            "init",
            "run",
            "--rm",
            self.config.init_service_name,
        ]
        result = self._run("init", command, timeout_s=self.config.timeout_s)
        report = _extract_last_json_object(result.stdout)
        if report is None and result.stderr:
            report = _extract_last_json_object(result.stderr)
        if report is None:
            report = {
                "status": "ok" if result.ok else "failed",
                "service": self.config.init_service_name,
            }
        else:
            report = {"status": "ok" if result.ok else "failed", "tpch_prepare": report}
        return DockerMonetDBCommandResult(
            stage=result.stage,
            command=result.command,
            returncode=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
            report=report,
        )

    def run_compose_workflow(self) -> Dict[str, Any]:
        """Run preflight, compose build, service startup, and TPC-H import."""
        preflight = self.preflight()
        if not preflight.ok:
            return _workflow_report("failed", preflight, None, None, None)

        build = self.build_image()
        if not build.ok:
            return _workflow_report("failed", preflight, build, None, None)

        up = self.start_service()
        if not up.ok:
            return _workflow_report("failed", preflight, build, up, None)

        init = self.init_tpch()
        status = "ok" if init.ok and init.report and init.report.get("status") == "ok" else "failed"
        return _workflow_report(status, preflight, build, up, init)

    def _dockerfile_path(self) -> Path:
        """Resolve Dockerfile relative to repo root when needed."""
        dockerfile = self.config.dockerfile
        if dockerfile.is_absolute():
            return dockerfile
        return self.config.repo_root / dockerfile

    def _compose_file_path(self) -> Path:
        """Resolve Docker Compose file relative to repo root when needed."""
        compose_file = self.config.compose_file
        if compose_file.is_absolute():
            return compose_file
        return self.config.repo_root / compose_file

    def _run(
        self,
        stage: str,
        command: List[str],
        *,
        timeout_s: int,
    ) -> DockerMonetDBCommandResult:
        """Run one Docker command and convert subprocess failures to structured results."""
        try:
            completed = self.run_command(
                command,
                cwd=self.config.repo_root,
                text=True,
                capture_output=True,
                timeout=timeout_s,
            )
            return DockerMonetDBCommandResult(
                stage=stage,
                command=command,
                returncode=int(completed.returncode),
                stdout=completed.stdout or "",
                stderr=completed.stderr or "",
            )
        except FileNotFoundError as exc:
            return DockerMonetDBCommandResult(
                stage=stage,
                command=command,
                returncode=127,
                stdout="",
                stderr=str(exc),
                report={"status": "failed", "reason": "docker_binary_missing"},
            )
        except subprocess.TimeoutExpired as exc:
            return DockerMonetDBCommandResult(
                stage=stage,
                command=command,
                returncode=124,
                stdout=exc.stdout or "",
                stderr=exc.stderr or "",
                report={"status": "failed", "reason": "timeout"},
            )


def _workflow_report(
    status: str,
    preflight: DockerMonetDBCommandResult,
    build: Optional[DockerMonetDBCommandResult],
    up: Optional[DockerMonetDBCommandResult],
    init: Optional[DockerMonetDBCommandResult],
) -> Dict[str, Any]:
    """Build a compact Docker lifecycle workflow report."""
    return {
        "status": status,
        "preflight": _command_result_summary(preflight),
        "build": None if build is None else _command_result_summary(build),
        "up": None if up is None else _command_result_summary(up),
        "init": None if init is None else _command_result_summary(init),
    }


def _command_result_summary(result: DockerMonetDBCommandResult) -> Dict[str, Any]:
    """Return the user-facing part of a Docker command result."""
    return {
        "stage": result.stage,
        "ok": result.ok,
        "returncode": result.returncode,
        "command": result.command,
        "report": result.report,
        "stderr_tail": result.stderr[-1200:],
    }


def _extract_last_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Extract the last JSON object from mixed Docker logs."""
    decoder = json.JSONDecoder()
    objects: List[Dict[str, Any]] = []
    index = 0
    while index < len(text):
        brace_index = text.find("{", index)
        if brace_index == -1:
            break
        try:
            value, end_index = decoder.raw_decode(text[brace_index:])
        except json.JSONDecodeError:
            index = brace_index + 1
            continue
        if isinstance(value, dict):
            objects.append(value)
        index = brace_index + end_index
    if not objects:
        return None
    return objects[-1]


class GeneratedTpchRuntimeProvider(RuntimeProvider):
    """Generated TPC-H 引擎的 Runtime Provider.
    
    从引擎的 stdout/stderr 解析 timing 输出。
    """
    
    def __init__(
        self,
        benchmark_mode: str = DEFAULT_BENCHMARK_MODE,
        storage_mode: str = DEFAULT_STORAGE_MODE,
        workers: int = DEFAULT_RUNTIME_WORKERS,
        engine: str = DEFAULT_BESPOKE_ENGINE,
    ) -> None:
        self.benchmark_mode = benchmark_mode
        self.storage_mode = storage_mode
        self.workers = workers
        self.engine = engine
        return None

    def measure(
        self,
        instantiation: QueryInstantiation,
        exec_callback: Optional[Callable[[List[str], int], Tuple[str, str, str]]] = None,
        primary_metric_kind: str = QUERY_RUNTIME_METRIC_KIND,
    ) -> RuntimeMeasurement:
        """测量单个查询实例（1 warmup + MEASURED_RUNS measured runs，取 median）."""
        if exec_callback is None:
            raise ValueError("GeneratedTpchRuntimeProvider requires exec_callback")

        args_list = [instantiation.args_string]
        timeout_policy = build_runtime_timeout_policy(
            instantiation.scale_factor,
            num_queries=1,
        )

        measured_ms: List[float] = []
        query_measured_ms: List[float] = []
        kernel_measured_ms: List[float] = []
        primary_metric_kinds: List[str] = []
        warmup_ingest_ms: List[float] = []
        warmup_load_ms: List[float] = []
        warmup_build_ms: List[float] = []

        total_runs = WARMUP_RUNS + MEASURED_RUNS
        for run_idx in range(total_runs):
            timeout_s = (
                timeout_policy.cold_start_timeout_s
                if run_idx < WARMUP_RUNS
                else timeout_policy.warm_query_timeout_s
            )
            response, stdout, stderr = exec_callback(args_list, timeout_s)
            raise_for_runtime_execution_failure(response, stdout, stderr)
            if run_idx < WARMUP_RUNS:
                ingest_ms, load_ms, build_ms = parse_ingest_timing_from_text(
                    stdout + "\n" + stderr
                )
                if ingest_ms is not None:
                    warmup_ingest_ms.append(ingest_ms)
                if load_ms is not None:
                    warmup_load_ms.append(load_ms)
                if build_ms is not None:
                    warmup_build_ms.append(build_ms)
                continue
            result = parse_query_timing(
                stdout,
                stderr,
                instantiation.query_id,
                primary_metric_kind=primary_metric_kind,
            )
            measured_ms.append(result.primary_runtime_ms)
            primary_metric_kinds.append(result.primary_metric_kind)
            if result.query_runtime_ms is not None:
                query_measured_ms.append(result.query_runtime_ms)
            if result.kernel_runtime_ms is not None:
                kernel_measured_ms.append(result.kernel_runtime_ms)

        samples = QuerySamples(
            measured_runs_ms=(
                query_measured_ms
                if primary_metric_kind == QUERY_RUNTIME_METRIC_KIND
                else measured_ms
            ),
            kernel_runs_ms=kernel_measured_ms,
        )
        median_ms = statistics.median(measured_ms)
        if primary_metric_kinds and len(set(primary_metric_kinds)) > 1:
            raise ValueError(
                f"Generated TPC-H {instantiation.query_id}: mixed runtime metric kinds "
                f"across measured runs: {primary_metric_kinds}"
            )
        lazy_suspected = (
            is_lazy_build_suspected(samples)
            if primary_metric_kind == QUERY_RUNTIME_METRIC_KIND
            else False
        )
        if lazy_suspected:
            logger.warning(
                "Generated TPC-H %s: lazy-build suspected "
                "(first=%.3fms, median=%.3fms)",
                instantiation.query_id,
                samples.first_query_ms,
                samples.median_query_ms,
            )
        provenance: Dict[str, Any] = {
            "query_id": instantiation.query_id,
            "instantiation_id": instantiation.instantiation_id,
            "args_string": instantiation.args_string,
            "scale_factor": instantiation.scale_factor,
            "runtime_metric_kind": primary_metric_kind,
            "kernel_runtime_metric_kind": KERNEL_RUNTIME_METRIC_KIND,
            "query_runs_ms": query_measured_ms,
            "kernel_runs_ms": kernel_measured_ms,
            "primary_runs_ms": measured_ms,
            "optimization_runtime_metric_kind": OPTIMIZATION_RUNTIME_METRIC_KIND,
            "measurement_shape_status": MeasurementShapeStatus.UNKNOWN.value,
            "timeout_policy": timeout_policy.to_provenance(),
            "warmup_ingest_ms": warmup_ingest_ms,
            "warmup_load_ms": warmup_load_ms,
            "warmup_build_ms": warmup_build_ms,
        }
        measurement = _build_runtime_measurement(
            instantiation_id=instantiation.instantiation_id,
            runtime_ms=median_ms,
            num_runs=len(measured_ms),
            all_runtimes_ms=measured_ms,
            benchmark_mode=self.benchmark_mode,
            storage_mode=self.storage_mode,
            workers=self.workers,
            engine=self.engine,
            query_id=instantiation.query_id,
            args_string=instantiation.args_string,
            scale_factor=instantiation.scale_factor,
            measurement_shape_status=MeasurementShapeStatus.UNKNOWN.value,
            provenance=provenance,
        )
        measurement._query_samples = samples
        measurement._lazy_build_suspected = lazy_suspected
        return measurement

    def measure_batch(
        self,
        instantiations: List[QueryInstantiation],
        exec_callback: Optional[Callable[[List[str], int], Tuple[str, str, str]]] = None,
    ) -> Dict[str, RuntimeMeasurement]:
        """批量测量：每个 instantiation 单独执行 warmup + measured runs。

        禁止混用同一次 batch stdout 中不同 query 的 timing index。
        """
        if exec_callback is None:
            raise ValueError("GeneratedTpchRuntimeProvider requires exec_callback")

        results: Dict[str, RuntimeMeasurement] = {}
        for inst in instantiations:
            results[inst.instantiation_id] = self.measure(inst, exec_callback)
        return results
    
    def _estimate_timeout(self, instantiation: QueryInstantiation) -> int:
        """Backward-compatible warm-query timeout estimate."""
        policy = build_runtime_timeout_policy(instantiation.scale_factor, num_queries=1)
        return policy.warm_query_timeout_s
    
    def _parse_timing(
        self,
        stdout: str,
        stderr: str,
        query_id: str,
        index: Optional[int] = None,
    ) -> float:
        """向后兼容包装器；只接受 official Query ms。"""
        result = parse_query_timing(stdout, stderr, query_id, index=index)
        return result.primary_runtime_ms


class MonetDBBaselineProvider(RuntimeProvider):
    """TPC-H MonetDB baseline provider using exact manifest SQL."""

    def __init__(
        self,
        oracle: Optional[MonetDBOracle] = None,
        num_runs: int = 3,
        benchmark_mode: str = DEFAULT_BENCHMARK_MODE,
        storage_mode: str = DEFAULT_STORAGE_MODE,
        workers: int = DEFAULT_RUNTIME_WORKERS,
        engine: str = DEFAULT_MONETDB_ENGINE,
    ) -> None:
        self.oracle = oracle or MonetDBOracle()
        self.num_runs = num_runs
        self.benchmark_mode = benchmark_mode
        self.storage_mode = storage_mode
        self.workers = workers
        self.engine = engine
        return None

    def measure(
        self,
        instantiation: QueryInstantiation,
        exec_callback: Optional[Callable] = None,
    ) -> RuntimeMeasurement:
        """Measure one TPC-H query instantiation against MonetDB."""
        del exec_callback
        contract = get_tpch_contract(instantiation.query_id)
        result, median_runtime_ms = self.oracle.execute_sql_benchmark(
            instantiation.sql,
            query_id=contract.query_id,
            query_type="tpch",
            params=instantiation.params_json,
            sorted_by=contract.sorted_by,
            num_runs=self.num_runs,
        )
        runtimes_ms = []
        if result.raw_response is not None:
            runtimes_ms = list(result.raw_response.get("runtimes_ms", []))
        return _build_runtime_measurement(
            instantiation_id=instantiation.instantiation_id,
            runtime_ms=median_runtime_ms,
            num_runs=self.num_runs,
            all_runtimes_ms=runtimes_ms,
            benchmark_mode=self.benchmark_mode,
            storage_mode=self.storage_mode,
            workers=self.workers,
            engine=self.engine,
            query_id=contract.query_id,
            args_string=instantiation.args_string,
            scale_factor=instantiation.scale_factor,
            output_row_count=result.row_count,
            measurement_shape_status=MeasurementShapeStatus.KNOWN.value,
            provenance={
                "baseline_backend": "monetdb-native-mapi",
                "source_protocol": result.source_protocol,
                "scale_factor": instantiation.scale_factor,
                "query_id": contract.query_id,
                "instantiation_id": instantiation.instantiation_id,
                "args_string": instantiation.args_string,
                "sql_hash": instantiation.sql_hash,
                "row_count": result.row_count,
                "columns": result.columns,
                "measurement_shape_status": MeasurementShapeStatus.KNOWN.value,
                "num_runs": self.num_runs,
            },
        )

    def measure_batch(
        self,
        instantiations: List[QueryInstantiation],
        exec_callback: Optional[Callable] = None,
    ) -> Dict[str, RuntimeMeasurement]:
        """Measure multiple TPC-H instantiations against MonetDB."""
        del exec_callback
        results: Dict[str, RuntimeMeasurement] = {}
        for instantiation in instantiations:
            results[instantiation.instantiation_id] = self.measure(instantiation)
        return results


class BespokeRuntimeProvider(GeneratedTpchRuntimeProvider):
    """BespokeRuntimeProvider 的兼容别名.
    
    保留一个 phase 的过渡期，之后将移除。
    """
    
    def __init__(self) -> None:
        import warnings
        warnings.warn(
            "BespokeRuntimeProvider is deprecated; use GeneratedTpchRuntimeProvider",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__()
        return None
