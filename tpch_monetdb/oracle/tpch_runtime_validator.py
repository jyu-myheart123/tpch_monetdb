"""TPC-H generated runtime validation against MonetDB oracle results."""

from __future__ import annotations

import hashlib
import json
import statistics
from pathlib import Path
from typing import Any, Callable

from tpch_monetdb.benchmark.runtime_accounting import (
    KERNEL_RUNTIME_METRIC_KIND,
    MEASURED_RUNS,
    RUNTIME_SCHEMA_VERSION,
    WARMUP_RUNS,
    build_runtime_timeout_policy,
    parse_ingest_timing_from_text,
    parse_query_timing,
    raise_for_runtime_execution_failure,
)
from tpch_monetdb.dataset.gen_tpch.gen_tpch_query import instantiate_tpch_query
from tpch_monetdb.dataset.gen_tpch.tpch_queries import get_contract, list_all_contracts
from tpch_monetdb.oracle.monetdb_oracle import MonetDBOracle
from tpch_monetdb.oracle.tpch_validator import TpchValidationReport, TpchValidator
from tpch_monetdb.oracle.validate_cache import CacheMissError

ExecCallback = Callable[[list[str], int], tuple[str, str, str]]


class TpchRuntimeValidator:
    """Validate generated runtime CSV output against exact MonetDB TPC-H SQL."""

    def __init__(
        self,
        workspace_path: Path,
        oracle: MonetDBOracle | None = None,
        sf_list: list[int | float] | None = None,
        allowed_query_ids: list[str] | None = None,
        seed: int | None = None,
        timeout_s: int = 120,
        cache_dir: Path | None = None,
    ) -> None:
        """Initialize validator state and keep the RunTool-compatible attributes."""
        self.workspace_path = Path(workspace_path)
        self.oracle = oracle or MonetDBOracle()
        self.sf_list = [1] if sf_list is None else list(sf_list)
        self.allowed_query_ids = self._normalize_query_ids(allowed_query_ids)
        self.seed = seed
        self.timeout_s = timeout_s
        self.cache_dir = None if cache_dir is None else Path(cache_dir)
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.tpch_validator = TpchValidator()
        return None

    def exec_and_validate(
        self,
        *,
        exec_callback_fn: ExecCallback,
        scale_factor: int | float,
        query_id: list[str] | None = None,
        other_config: dict[str, Any] | None = None,
        skip_validate: bool = False,
        compile_key_hash: str | None = None,
        trace_mode: bool = False,
        only_from_cache: bool = False,
        skip_validate_cache: bool = False,
        performance_exec_callback_fn: ExecCallback | None = None,
    ) -> tuple[str, bool, dict[str, Any], bool]:
        """Run generated TPC-H queries and compare result CSVs with MonetDB."""
        del trace_mode

        resolved_other_config = {} if other_config is None else dict(other_config)
        performance_comparison_enabled = self._performance_comparison_enabled(
            resolved_other_config
        )
        normalized_query_ids = self._resolve_query_ids(query_id)
        instantiations = [
            instantiate_tpch_query(
                query_id=query_id_value,
                scale_factor=int(scale_factor),
                seed=self.seed,
            )
            for query_id_value in normalized_query_ids
        ]
        if not skip_validate and not performance_comparison_enabled:
            cached_result = self._load_cached_validation(
                instantiations=instantiations,
                scale_factor=scale_factor,
                compile_key_hash=compile_key_hash,
                only_from_cache=only_from_cache,
                skip_validate_cache=skip_validate_cache,
            )
            if cached_result is not None:
                return cached_result

        args_list = [
            str(instantiation["args_string"]) for instantiation in instantiations
        ]
        response, stdout, stderr = exec_callback_fn(args_list, self.timeout_s)
        if "exit_code: 0" not in response:
            metrics = self._build_failure_metrics(
                scale_factor=scale_factor,
                query_ids=normalized_query_ids,
                reports=[],
                failure="generated runtime returned non-zero exit",
            )
            if performance_comparison_enabled:
                metrics["validation/performance_comparison_enabled"] = True
                metrics["validation/performance_comparison_skipped"] = "runtime_failed"
            metrics["validation/stdout"] = stdout
            metrics["validation/stderr"] = stderr
            metrics["validation/response"] = response
            return "TPC-H generated runtime failed before validation", False, metrics, False

        if skip_validate:
            metrics = {
                "validation/correct": True,
                "validation/skipped": True,
                "validation/query_ids_executed": normalized_query_ids,
                "validation/scale_factor": scale_factor,
            }
            return "TPC-H generated runtime executed with validation skipped", True, metrics, False

        reports: list[TpchValidationReport] = []
        for index, instantiation in enumerate(instantiations, start=1):
            contract = get_contract(str(instantiation["query_id"]))
            expected = self.oracle.execute_sql(
                str(instantiation["sql"]),
                query_id=contract.query_id,
                query_type="tpch",
                params=dict(instantiation["params_json"]),
                sorted_by=contract.sorted_by,
            )
            runtime_csv = self.workspace_path / f"result{index}.csv"
            actual = self.tpch_validator.parse_runtime_csv(runtime_csv, contract.query_id)
            reports.append(
                self.tpch_validator.compare_results(
                    expected=expected,
                    actual=actual,
                    query_id=contract.query_id,
                )
            )

        success = all(report.overall_pass for report in reports)
        metrics = self._build_success_metrics(
            scale_factor=scale_factor,
            query_ids=normalized_query_ids,
            reports=reports,
            stdout=stdout,
            stderr=stderr,
            response=response,
        )
        summary = self._format_summary(reports, success)
        if performance_comparison_enabled and success:
            comparison_metrics, comparison_summary = self._collect_performance_comparison(
                performance_exec_callback_fn=performance_exec_callback_fn,
                scale_factor=scale_factor,
                instantiations=instantiations,
            )
            metrics.update(comparison_metrics)
            if comparison_summary:
                summary = f"{summary}\n{comparison_summary}"
        elif performance_comparison_enabled:
            metrics["validation/performance_comparison_enabled"] = True
            metrics["validation/performance_comparison_skipped"] = "validation_failed"
        self._store_successful_cache_entries(
            instantiations=instantiations,
            scale_factor=scale_factor,
            compile_key_hash=compile_key_hash,
            reports=reports,
        )
        return summary, success, metrics, False

    def _performance_comparison_enabled(self, other_config: dict[str, Any]) -> bool:
        """Return whether this validation request should emit base benchmark evidence."""
        return other_config.get("enable_performance_comparison") is True

    def _collect_performance_comparison(
        self,
        *,
        performance_exec_callback_fn: ExecCallback | None,
        scale_factor: int | float,
        instantiations: list[dict[str, Any]],
    ) -> tuple[dict[str, Any], str]:
        """Measure generated no-output runtime and MonetDB baseline speedup evidence."""
        metrics: dict[str, Any] = {
            "validation/performance_comparison_enabled": True,
            "validation/performance_metric_kind": KERNEL_RUNTIME_METRIC_KIND,
            "validation/performance_warmup_runs": WARMUP_RUNS,
            "validation/performance_measured_runs": MEASURED_RUNS,
        }
        if performance_exec_callback_fn is None:
            error = "no performance execution callback was provided"
            metrics["validation/performance_comparison_error"] = error
            return metrics, f"Base benchmark performance comparison unavailable: {error}"

        try:
            args_list = [
                str(instantiation["args_string"]) for instantiation in instantiations
            ]
            generated_samples_by_query: dict[str, list[float]] = {
                str(instantiation["query_id"]): [] for instantiation in instantiations
            }
            timeout_policy = build_runtime_timeout_policy(
                scale_factor,
                num_queries=len(instantiations),
            )
            warmup_ingest_ms: list[float] = []
            warmup_load_ms: list[float] = []
            warmup_build_ms: list[float] = []
            response = ""
            stdout = ""
            stderr = ""
            total_runs = WARMUP_RUNS + MEASURED_RUNS
            for run_index in range(total_runs):
                timeout_s = (
                    timeout_policy.cold_start_timeout_s
                    if run_index < WARMUP_RUNS
                    else timeout_policy.warm_query_timeout_s
                )
                response, stdout, stderr = performance_exec_callback_fn(
                    args_list,
                    timeout_s,
                )
                raise_for_runtime_execution_failure(response, stdout, stderr)
                if "exit_code: 0" not in response:
                    raise RuntimeError("generated no-output performance run failed")
                if run_index < WARMUP_RUNS:
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
                for index, instantiation in enumerate(instantiations):
                    query_id = str(instantiation["query_id"])
                    timing = parse_query_timing(
                        stdout,
                        stderr,
                        query_id,
                        index=index,
                        primary_metric_kind=KERNEL_RUNTIME_METRIC_KIND,
                    )
                    generated_samples_by_query[query_id].append(
                        timing.primary_runtime_ms
                    )

            generated_ms_by_query = {
                query_id: statistics.median(samples)
                for query_id, samples in generated_samples_by_query.items()
            }
            baseline_ms_by_query: dict[str, float] = {}
            speedup_by_query: dict[str, float] = {}
            lines = [
                (
                    "Base benchmark performance comparison "
                    f"(scale_factor={scale_factor}; generated=no-output "
                    f"{KERNEL_RUNTIME_METRIC_KIND} median vs MonetDB median):"
                )
            ]
            for instantiation in instantiations:
                contract = get_contract(str(instantiation["query_id"]))
                _, baseline_ms = self.oracle.execute_sql_benchmark(
                    str(instantiation["sql"]),
                    query_id=contract.query_id,
                    query_type="tpch",
                    params=dict(instantiation["params_json"]),
                    sorted_by=contract.sorted_by,
                    num_runs=MEASURED_RUNS,
                )
                generated_ms = generated_ms_by_query[contract.query_id]
                baseline_ms_by_query[contract.query_id] = baseline_ms
                speedup = self._safe_speedup(baseline_ms, generated_ms)
                speedup_by_query[contract.query_id] = speedup
                lines.append(
                    f"{contract.query_id}: generated={generated_ms:.3f} ms; "
                    f"MonetDB={baseline_ms:.3f} ms; "
                    f"speedup={self._format_speedup(speedup)}"
                )

            total_generated_ms = sum(generated_ms_by_query.values())
            total_baseline_ms = sum(baseline_ms_by_query.values())
            total_speedup = self._safe_speedup(total_baseline_ms, total_generated_ms)
            lines.append(
                f"TOTAL: generated={total_generated_ms:.3f} ms; "
                f"MonetDB={total_baseline_ms:.3f} ms; "
                f"speedup={self._format_speedup(total_speedup)}"
            )
            metrics.update(
                {
                    "validation/generated_kernel_runtime_ms_by_query": generated_ms_by_query,
                    "validation/generated_kernel_runtime_samples_ms_by_query": generated_samples_by_query,
                    "validation/monetdb_baseline_runtime_ms_by_query": baseline_ms_by_query,
                    "validation/speedup_vs_monetdb_by_query": speedup_by_query,
                    "validation/generated_kernel_runtime_total_ms": total_generated_ms,
                    "validation/monetdb_baseline_runtime_total_ms": total_baseline_ms,
                    "validation/speedup_vs_monetdb_total": total_speedup,
                    "validation/performance_response": response,
                    "validation/performance_stdout": stdout,
                    "validation/performance_stderr": stderr,
                    "validation/performance_timeout_policy": timeout_policy.to_provenance(),
                    "validation/performance_warmup_ingest_ms": warmup_ingest_ms,
                    "validation/performance_warmup_load_ms": warmup_load_ms,
                    "validation/performance_warmup_build_ms": warmup_build_ms,
                }
            )
            return metrics, "\n".join(lines)
        except Exception as exc:
            error = str(exc)
            metrics["validation/performance_comparison_error"] = error
            return metrics, f"Base benchmark performance comparison unavailable: {error}"

    def _safe_speedup(self, baseline_ms: float, generated_ms: float) -> float:
        """Compute baseline/generated speedup while handling zero generated time."""
        if generated_ms == 0:
            return float("inf")
        return baseline_ms / generated_ms

    def _format_speedup(self, speedup: float) -> str:
        """Format one speedup value for human-readable run-tool output."""
        if speedup == float("inf"):
            return "inf"
        return f"{speedup:.3f}x"

    def _resolve_query_ids(self, query_ids: list[str] | None) -> list[str]:
        """Return normalized Q-prefixed query ids for one validation request."""
        normalized = self._normalize_query_ids(query_ids)
        if normalized:
            return normalized
        if self.allowed_query_ids:
            return list(self.allowed_query_ids)
        return list(list_all_contracts())

    def _normalize_query_ids(self, query_ids: list[str] | None) -> list[str]:
        """Normalize query ids through the canonical TPC-H contract registry."""
        if query_ids is None:
            return []
        normalized: list[str] = []
        for query_id in query_ids:
            normalized.append(get_contract(query_id).query_id)
        return list(dict.fromkeys(normalized))

    def _build_success_metrics(
        self,
        *,
        scale_factor: int | float,
        query_ids: list[str],
        reports: list[TpchValidationReport],
        stdout: str,
        stderr: str,
        response: str,
    ) -> dict[str, Any]:
        """Build RunTool-compatible validation metrics for successful execution."""
        success = all(report.overall_pass for report in reports)
        metrics: dict[str, Any] = {
            "validation/correct": success,
            "validation/query_ids_executed": query_ids,
            "validation/scale_factor": scale_factor,
            "validation/report_count": len(reports),
            "validation/failed_query_ids": [
                report.query_id for report in reports if not report.overall_pass
            ],
            "validation/tpch_reports": [report.to_dict() for report in reports],
            "validation/stdout": stdout,
            "validation/stderr": stderr,
            "validation/response": response,
            "validation/baseline_engine": "monetdb",
            "validation/runtime_engine": "generated_runtime",
        }
        return metrics

    def _build_failure_metrics(
        self,
        *,
        scale_factor: int | float,
        query_ids: list[str],
        reports: list[TpchValidationReport],
        failure: str,
    ) -> dict[str, Any]:
        """Build RunTool-compatible validation metrics for execution failures."""
        return {
            "validation/correct": False,
            "validation/query_ids_executed": query_ids,
            "validation/scale_factor": scale_factor,
            "validation/report_count": len(reports),
            "validation/tpch_reports": [report.to_dict() for report in reports],
            "validation/failure": failure,
            "validation/baseline_engine": "monetdb",
            "validation/runtime_engine": "generated_runtime",
        }

    def _format_summary(
        self,
        reports: list[TpchValidationReport],
        success: bool,
    ) -> str:
        """Return a compact validation summary string."""
        status = "PASS" if success else "FAIL"
        report_summaries = "; ".join(report.get_summary() for report in reports)
        return f"TPC-H generated runtime validation {status}: {report_summaries}"

    def _cache_key(
        self,
        *,
        compile_key_hash: str,
        instantiation: dict[str, Any],
        scale_factor: int | float,
    ) -> str:
        """Build a TPC-H validation cache key without legacy table semantics."""
        payload = {
            "benchmark": "tpch",
            "compile_key_hash": compile_key_hash,
            "query_id": instantiation["query_id"],
            "scale_factor": scale_factor,
            "args_string": instantiation["args_string"],
            "sql_hash": instantiation["sql_hash"],
            "params_json": instantiation["params_json"],
            "runtime_schema_version": RUNTIME_SCHEMA_VERSION,
        }
        encoded = json.dumps(payload, sort_keys=True, default=str).encode()
        return hashlib.sha256(encoded).hexdigest()[:32]

    def _cache_file(
        self,
        *,
        compile_key_hash: str,
        instantiation: dict[str, Any],
        scale_factor: int | float,
    ) -> Path:
        """Return the cache file path for one TPC-H validation entry."""
        if self.cache_dir is None:
            raise RuntimeError("TPC-H validation cache_dir is not configured")
        return self.cache_dir / (
            self._cache_key(
                compile_key_hash=compile_key_hash,
                instantiation=instantiation,
                scale_factor=scale_factor,
            )
            + ".json"
        )

    def _load_cached_validation(
        self,
        *,
        instantiations: list[dict[str, Any]],
        scale_factor: int | float,
        compile_key_hash: str | None,
        only_from_cache: bool,
        skip_validate_cache: bool,
    ) -> tuple[str, bool, dict[str, Any], bool] | None:
        """Load a full TPC-H validation response from cache when all entries exist."""
        if self.cache_dir is None or skip_validate_cache:
            return None
        if not compile_key_hash:
            if only_from_cache:
                raise CacheMissError(str(instantiations[0]["query_id"]), scale_factor)
            return None

        entries: list[dict[str, Any]] = []
        missing_query_id = ""
        for instantiation in instantiations:
            cache_file = self._cache_file(
                compile_key_hash=compile_key_hash,
                instantiation=instantiation,
                scale_factor=scale_factor,
            )
            if not cache_file.exists():
                missing_query_id = str(instantiation["query_id"])
                break
            entries.append(json.loads(cache_file.read_text(encoding="utf-8")))
        if len(entries) != len(instantiations):
            if only_from_cache:
                raise CacheMissError(missing_query_id, scale_factor)
            return None

        success = all(bool(entry["success"]) for entry in entries)
        query_ids = [str(entry["query_id"]) for entry in entries]
        reports = [entry["report"] for entry in entries]
        metrics = {
            "validation/correct": success,
            "validation/query_ids_executed": query_ids,
            "validation/scale_factor": scale_factor,
            "validation/report_count": len(reports),
            "validation/failed_query_ids": [
                str(entry["query_id"]) for entry in entries if not entry["success"]
            ],
            "validation/tpch_reports": reports,
            "validation/baseline_engine": "monetdb",
            "validation/runtime_engine": "generated_runtime",
            "validation/used_cache": True,
        }
        msg = "TPC-H generated runtime validation loaded from cache"
        return msg, success, metrics, True

    def _store_successful_cache_entries(
        self,
        *,
        instantiations: list[dict[str, Any]],
        scale_factor: int | float,
        compile_key_hash: str | None,
        reports: list[TpchValidationReport],
    ) -> None:
        """Store successful per-query validation reports in the TPC-H cache."""
        if self.cache_dir is None or not compile_key_hash:
            return None
        for instantiation, report in zip(instantiations, reports):
            if not report.overall_pass:
                continue
            cache_file = self._cache_file(
                compile_key_hash=compile_key_hash,
                instantiation=instantiation,
                scale_factor=scale_factor,
            )
            payload = {
                "query_id": report.query_id,
                "success": report.overall_pass,
                "msg": report.get_summary(),
                "report": report.to_dict(),
            }
            cache_file.write_text(
                json.dumps(payload, indent=2, default=str),
                encoding="utf-8",
            )
        return None
