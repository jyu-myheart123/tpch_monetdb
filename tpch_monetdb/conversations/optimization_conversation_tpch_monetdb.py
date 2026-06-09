"""TPC-H MonetDB Optimization Conversation - two-stage optimization loop.

Stages:
1. trace_expert (per-query): Parse trace output, classify bottleneck,
   apply one targeted local change with light correctness gate.
2. global_human_reference (multi-attempt): Read hotspot summary, TODO,
   storage_plan, and try transaction-scoped global convergence changes.

Architecture:
- Uses Runtime Provider / Baseline Provider architecture (phase3 D1)
- Consumes Reference Instantiation Manifest for consistent query instances
- Measures speedup against the active benchmark baseline (3 runs median)

Default TPC-H path:
- Uses Dockerized MonetDB baseline and TPC-H manifest instantiations
- Rejects legacy optimization branches at construction time
"""

import logging
import math
import json
import re
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from agents.extensions.memory import AdvancedSQLiteSession
from tpch_monetdb.config import get_optim_stage_max_turns, get_stage_turn_budget

from tpch_monetdb.conversations.conversation import (
    COMPACTION_MARKER,
    VALIDATE_ON,
    VALIDATE_OUTPUT_STDOUT_OFF,
    VALIDATE_OUTPUT_STDOUT_ON,
    AbstractConversation,
)
from tpch_monetdb.conversations.agent_text_registry import render_agent_text_asset
from tpch_monetdb.benchmark.manifest import (
    BaselineRoutingPolicy,
    ReferenceManifest,
    check_agent_diff_boundary,
)
from tpch_monetdb.benchmark.runtime_accounting import (
    KERNEL_RUNTIME_METRIC_KIND,
    QUERY_RUNTIME_METRIC_KIND,
    build_runtime_timeout_policy,
    check_ingest_completeness,
    derive_bespoke_ingest_metrics,
)
from tpch_monetdb.benchmark.providers import (
    GeneratedTpchRuntimeProvider,
    MonetDBBaselineProvider,
)
from tpch_monetdb.conversations.tpch_monetdb_prompts_gen import (
    load_expert_knowledge,
    tpch_monetdb_optim_prompt_add_timings,
    tpch_monetdb_optim_prompt_add_timings_per_query,
    tpch_monetdb_optim_prompt_constraints,
    tpch_monetdb_optim_prompt_global_diagnosis,
    tpch_monetdb_optim_prompt_global_human_reference,
    tpch_monetdb_optim_prompt_hypothesis_execution,
    tpch_monetdb_optim_prompt_pinning,
    tpch_monetdb_optim_prompt_pretext,
    tpch_monetdb_optim_prompt_pretext_optim,
    tpch_monetdb_optim_prompt_trace_expert,
)
from tpch_monetdb.llm_cache.git_snapshotter import GitSnapshotter
from tpch_monetdb.conversations.optimization_validation import (
    aggregate_scope_runtime_seconds,
    build_unit_validation_plan,
    required_validation_scale_factors,
    run_required_correctness_checks,
    should_rollback_unit_regression,
)
from tpch_monetdb.conversations.optimization_instrumentation import (
    TraceEvidenceSummary,
    build_instrumentation_prompt_metadata,
    check_trace_mode_smoke,
    check_trace_evidence_and_feedback,
)
from tpch_monetdb.conversations.hotspot_summary_markdown import (
    format_pmu_perf_status_markdown_lines,
)
from tpch_monetdb.tools.error_envelope import ErrorEnvelope
from tpch_monetdb.tools.tpch.run import (
    QUERY_OUTPUT_MODE_FULL_CSV,
    QUERY_OUTPUT_MODE_NO_OUTPUT,
    RunTool,
)
from tpch_monetdb.tools.tpch.runtime_hygiene import classify_infra_failure
from tpch_monetdb.tools.stage_tool_policy import CORE_IMPLEMENTATION_FILES, StageRunSummary
from tpch_monetdb.utils.wandb_stats_logging import WandbRunHook
from tpch_monetdb.utils.outer_loop_supervisor import classify_model_failure
from tpch_monetdb.tools.tpch.trace_analysis import (
    TraceHotspotSummary,
    merge_trace_summaries,
    summarize_trace_file,
)
from tpch_monetdb.utils.optimization_summary import (
    persist_optimization_run,
)
from tpch_monetdb.utils.duration_format import format_duration_ms, safe_speedup
from tpch_monetdb.utils.large_data_objectives import (
    WORKLOAD_OBJECTIVE_FILE,
    build_hot_loop_mapping,
    build_objective_failure_report,
    classify_objective_failure_route,
    load_json_contract,
)
from tpch_monetdb.utils.pipeline_evidence import (
    EvidenceStatus,
    MeasurementKind,
    MeasurementShapeStatus,
    QueryMeasurementRecord,
    build_pipeline_evidence_ledger,
)
from tpch_monetdb.utils.control_artifacts import (
    build_storage_plan_alignment,
    build_todo_reconciliation,
    collect_control_artifact_hashes,
)
from tpch_monetdb.utils.query_units import QueryUnit, build_query_unit_lookup

logger = logging.getLogger(__name__)

PINNING_PROMPT_MAX_TURNS = 600


def _summarize_perf_symbols_from_trace_text(trace_summary: str) -> str | None:
    perf_lines: list[str] = []
    collecting = False
    for line in trace_summary.splitlines():
        if line == "Perf top symbols:":
            collecting = True
            continue
        if collecting and not line.startswith("- "):
            break
        if collecting:
            perf_lines.append(line[2:])
    if not perf_lines:
        return None
    return "; ".join(perf_lines[:5])


def _format_perf_pairs_for_markdown(values: Any, limit: int = 5) -> str | None:
    pairs = [
        (str(item[0]), int(item[1]))
        for item in values
        if isinstance(item, (list, tuple)) and len(item) >= 2
    ]
    if not pairs:
        return None
    return "; ".join(f"{name}={count}" for name, count in pairs[:limit])


def _format_perf_hotspot_markdown_lines(summary: dict[str, Any]) -> list[str]:
    if not summary.get("perf_hotspots_available"):
        return []
    lines = [f"- Perf samples: {int(summary.get('perf_sample_count', 0) or 0)}"]
    for label, key in (
        ("Perf top symbols", "perf_top_symbols"),
        ("Perf top source lines", "perf_top_source_lines"),
        ("Perf top call-stack frames", "perf_top_frames"),
    ):
        rendered = _format_perf_pairs_for_markdown(summary.get(key, []))
        if rendered is not None:
            lines.append(f"- {label}: {rendered}")
    return lines


def _format_hardware_counter_evidence_for_prompt(
    summary: dict[str, Any],
) -> str:
    """Render compact PMU and perf evidence without raw perf script excerpts."""
    lines: list[str] = []
    identity = _format_named_summary_values(summary, ("backend", "target_cpu"))
    if identity is not None:
        lines.append(identity)
    counters = _format_named_summary_values(
        summary.get("counters", {}),
        ("cycles", "instructions", "cache-misses", "LLC-load-misses", "dTLB-load-misses"),
    )
    if counters is not None:
        lines.append(f"counters: {counters}")
    derived = _format_named_summary_values(
        summary.get("derived_metrics", {}),
        ("ipc", "cache_miss_rate", "llc_mpki", "dtlb_mpki", "branch_miss_rate"),
    )
    if derived is not None:
        lines.append(f"derived_metrics: {derived}")
    lines.extend(_format_perf_hotspot_markdown_lines(summary))
    if summary.get("hardware_counter_error"):
        lines.append(
            "hardware_counter_error: "
            + _truncate_evidence_value(str(summary["hardware_counter_error"]))
        )
    if summary.get("perf_hotspot_error"):
        lines.append(
            "perf_hotspot_error: "
            + _truncate_evidence_value(str(summary["perf_hotspot_error"]))
        )
    return "\n".join(lines)


def _format_named_summary_values(
    summary: dict[str, Any],
    keys: tuple[str, ...],
) -> str | None:
    pairs = [
        f"{key}={_truncate_evidence_value(str(summary[key]))}"
        for key in keys
        if key in summary and summary[key] not in (None, "", [], {})
    ]
    if not pairs:
        return None
    return "; ".join(pairs)


def _truncate_evidence_value(value: str, limit: int = 240) -> str:
    if len(value) <= limit:
        return value
    return value[: limit - 3] + "..."


def _merge_scope_issue_class(
    summaries: list[TraceHotspotSummary],
) -> str:
    issue_classes = [
        summary.issue_class
        for summary in summaries
        if summary.issue_class
    ]
    if not issue_classes:
        return "evidence_insufficient"
    unique_issue_classes = list(dict.fromkeys(issue_classes))
    if len(unique_issue_classes) == 1:
        return unique_issue_classes[0]
    return "mixed"


def _format_scope_trace_summary(
    summaries: list[TraceHotspotSummary],
) -> str:
    if not summaries:
        return ""
    rendered_blocks = [
        f"[Query {summary.query_id}]\n{summary.summary_text}"
        for summary in summaries
    ]
    return "\n\n".join(rendered_blocks)


def _format_scope_hardware_counter_evidence(
    summaries: list[TraceHotspotSummary],
) -> str:
    rendered_blocks: list[str] = []
    for summary in summaries:
        evidence = _format_hardware_counter_evidence_for_prompt(
            summary.hardware_counter_summary
        ).strip()
        if not evidence:
            continue
        rendered_blocks.append(f"Query {summary.query_id}:\n{evidence}")
    return "\n\n".join(rendered_blocks)


def _safe_branch_label(text: str) -> str:
    """Return a branch-safe label for unit-scoped optimization sessions."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_") or "unit"


@dataclass
class StageConfig:
    """Configuration for one optimization stage."""
    name: str
    get_prompt: Callable[[float], str]  # rt_before_ms -> prompt
    get_descriptor: Callable[[], Optional[str]] = lambda: None
    max_turns: Optional[int] = None


@dataclass
class StageResult:
    """Result of one optimization stage."""
    name: str
    rt_before_s: float
    rt_after_s: float
    speedup_vs_baseline: float
    written_files: tuple[str, ...] = ()
    runtime_by_query: dict[str, float] | None = None
    failed_scale_factor: Optional[int] = None
    failed: bool = False
    failure_message: Optional[str] = None

    @property
    def improved(self) -> bool:
        if self.failed:
            return False
        return self.rt_after_s < self.rt_before_s

    @property
    def improvement_factor(self) -> float:
        if self.failed:
            return 0.0
        if self.rt_after_s > 0:
            return self.rt_before_s / self.rt_after_s
        return float("inf")


@dataclass
class QueryOptimizationRecord:
    query_id: str
    unit_id: str | None
    unit_query_ids: tuple[str, ...]
    issue_class: str
    trace_summary: str
    sampled_instantiations: tuple[str, ...]
    stage_name: str
    rt_before_s: float
    rt_after_s: float
    written_files: tuple[str, ...]
    failed: bool = False
    failure_code: str | None = None
    failure_detail: str | None = None


@dataclass(frozen=True)
class QueryOutputSplitMeasurement:
    """Full-CSV versus no-output runtime evidence for one query."""
    query_id: str
    full_csv_s: float
    no_output_s: float
    materialization_s: float
    materialization_ratio: float

    def to_dict(self) -> dict[str, float | str]:
        """Serialize output split evidence for logs and summaries."""
        return {
            "query_id": self.query_id,
            "full_csv_ms": self.full_csv_s * 1000.0,
            "no_output_ms": self.no_output_s * 1000.0,
            "materialization_ms": self.materialization_s * 1000.0,
            "materialization_ratio": self.materialization_ratio,
        }


@dataclass(frozen=True)
class OptimizationValidationPolicy:
    light_scale_factors: tuple[int, ...]
    full_scale_factors: tuple[int, ...]
    heavyweight_scale_factors: tuple[int, ...]


@dataclass(frozen=True)
class GlobalHumanReferenceAttempt:
    """One transaction-scoped global human-reference candidate."""
    attempt_index: int
    written_files: tuple[str, ...]
    accepted: bool
    rejection_code: str | None = None
    rejection_detail: str | None = None
    regressed_queries: tuple[str, ...] = ()
    objective_failures: tuple[str, ...] = ()
    control_artifacts_read: tuple[str, ...] = ()


@dataclass(frozen=True)
class GlobalHumanReferenceResult:
    """Compatibility result for the autonomous global optimization pass."""
    runtime_by_query: dict[str, float]
    written_files: tuple[str, ...]
    accepted: bool
    attempts: tuple[GlobalHumanReferenceAttempt, ...] = ()
    regressed_queries: tuple[str, ...] = ()
    failure_detail: str | None = None
    hypotheses: tuple["GlobalOptimizationHypothesis", ...] = ()
    candidates: tuple["GlobalOptimizationCandidate", ...] = ()
    winner: "GlobalOptimizationCandidate | None" = None


@dataclass(frozen=True)
class GlobalOptimizationHypothesis:
    """One evidence-backed optimization hypothesis output by the diagnosis stage."""
    id: str
    summary: str
    evidence: tuple[str, ...]
    affected_queries: tuple[str, ...]
    suspected_runtime_path: tuple[str, ...] = ()
    expected_mechanism: str = ""
    expected_impact: dict[str, float] | None = None
    correctness_risk: str = "medium"
    implementation_scope: tuple[str, ...] = ()
    verification_plan: tuple[str, ...] = ()
    evidence_gap: bool = False


@dataclass(frozen=True)
class GlobalOptimizationCandidate:
    """One attempted hypothesis with gate results."""
    hypothesis: GlobalOptimizationHypothesis
    snapshot_hash: str
    accepted: bool
    runtime_by_query: dict[str, float]
    written_files: tuple[str, ...] = ()
    rejection_codes: tuple[str, ...] = ()
    rejection_detail: str | None = None
    objective_failures: tuple[str, ...] = ()
    measurement_gaps: tuple[str, ...] = ()
    causality_evidence: tuple[str, ...] = ()
    speedup_by_query: dict[str, float] | None = None
    partial: bool = False
    accepted_units: tuple["GlobalPatchUnitResult", ...] = ()
    rejected_units: tuple["GlobalPatchUnitResult", ...] = ()


@dataclass(frozen=True)
class GlobalPatchUnit:
    """One atomic or query-scoped patch unit inside a global candidate."""
    unit_id: str
    files: tuple[str, ...]
    affected_queries: tuple[str, ...]
    atomic: bool = False


@dataclass(frozen=True)
class GlobalPatchUnitResult:
    """Decision evidence for one global patch unit."""
    unit: GlobalPatchUnit
    accepted: bool
    rejection_codes: tuple[str, ...] = ()
    regressed_queries: tuple[str, ...] = ()


def global_hypothesis_to_dict(
    hypothesis: GlobalOptimizationHypothesis,
) -> dict[str, Any]:
    """Serialize one global optimization hypothesis for run summaries."""
    return {
        "id": hypothesis.id,
        "summary": hypothesis.summary,
        "evidence": list(hypothesis.evidence),
        "affected_queries": list(hypothesis.affected_queries),
        "suspected_runtime_path": list(hypothesis.suspected_runtime_path),
        "expected_mechanism": hypothesis.expected_mechanism,
        "expected_impact": hypothesis.expected_impact or {},
        "correctness_risk": hypothesis.correctness_risk,
        "implementation_scope": list(hypothesis.implementation_scope),
        "verification_plan": list(hypothesis.verification_plan),
        "evidence_gap": hypothesis.evidence_gap,
    }


def global_candidate_to_dict(
    candidate: GlobalOptimizationCandidate,
) -> dict[str, Any]:
    """Serialize one global optimization candidate for replay/audit summaries."""
    return {
        "hypothesis": global_hypothesis_to_dict(candidate.hypothesis),
        "snapshot_hash": candidate.snapshot_hash,
        "accepted": candidate.accepted,
        "runtime_by_query": dict(candidate.runtime_by_query),
        "written_files": list(candidate.written_files),
        "rejection_codes": list(candidate.rejection_codes),
        "rejection_detail": candidate.rejection_detail,
        "objective_failures": list(candidate.objective_failures),
        "measurement_gaps": list(candidate.measurement_gaps),
        "causality_evidence": list(candidate.causality_evidence),
        "speedup_by_query": dict(candidate.speedup_by_query or {}),
        "partial": candidate.partial,
        "accepted_units": [
            global_patch_unit_result_to_dict(unit)
            for unit in candidate.accepted_units
        ],
        "rejected_units": [
            global_patch_unit_result_to_dict(unit)
            for unit in candidate.rejected_units
        ],
    }


def global_patch_unit_result_to_dict(result: GlobalPatchUnitResult) -> dict[str, Any]:
    """Serialize one patch-unit decision for audit summaries."""
    return {
        "unit": {
            "unit_id": result.unit.unit_id,
            "files": list(result.unit.files),
            "affected_queries": list(result.unit.affected_queries),
            "atomic": result.unit.atomic,
        },
        "accepted": result.accepted,
        "rejection_codes": list(result.rejection_codes),
        "regressed_queries": list(result.regressed_queries),
    }


GLOBAL_HUMAN_REFERENCE_MAX_ATTEMPTS = 3
GLOBAL_OPTIMIZATION_MAX_HYPOTHESES = 5
GLOBAL_HUMAN_REFERENCE_REQUIRED_CONTROL_ARTIFACTS: tuple[str, ...] = (
    "optimization_hotspot_summary.md",
    "TODO.md",
    "storage_plan.txt",
    WORKLOAD_OBJECTIVE_FILE,
    "data_law_contract.json",
    "storage_plan_contract.json",
)
GLOBAL_ATTEMPT_IGNORED_OBJECTIVE_FAILURES: frozenset[str] = frozenset({
    "MEASUREMENT_REPETITION_MISSING",
    "MEASUREMENT_UNSTABLE",
})


def select_global_winner(
    candidates: tuple[GlobalOptimizationCandidate, ...],
    baseline_runtime_ms_by_query: dict[str, float],
) -> GlobalOptimizationCandidate | None:
    """Select the best accepted candidate from the pool.

    Ranking order:
    1. Correctness: only accepted candidates are eligible (hard gate)
    2. Fewest measurement gaps
    3. Fewest objective failures
    4. Best geometric-mean speedup on critical/affected queries
    5. Fewest rejection codes
    6. Aggregate runtime (secundary tiebreaker)
    """
    accepted = [c for c in candidates if c.accepted]
    if not accepted:
        return None

    def _candidate_priority_query_ids(
        candidate: GlobalOptimizationCandidate,
    ) -> tuple[str, ...]:
        query_ids = tuple(
            qid for qid in candidate.hypothesis.affected_queries
            if qid in candidate.runtime_by_query
        )
        if query_ids:
            return query_ids
        return tuple(candidate.runtime_by_query)

    def _geom_speedup(candidate: GlobalOptimizationCandidate) -> float:
        ratios: list[float] = []
        baseline = baseline_runtime_ms_by_query or {}
        for qid in _candidate_priority_query_ids(candidate):
            rt_s = candidate.runtime_by_query.get(qid)
            base_ms = baseline.get(qid, float("inf"))
            if rt_s is not None and rt_s > 0 and base_ms > 0 and base_ms != float("inf"):
                ratios.append(base_ms / (rt_s * 1000.0))
        if not ratios:
            return 0.0
        prod = 1.0
        for r in ratios:
            prod *= r
        return prod ** (1.0 / len(ratios))

    def _key(candidate: GlobalOptimizationCandidate) -> tuple:
        return (
            -len(candidate.measurement_gaps),
            -len(candidate.objective_failures),
            _geom_speedup(candidate),
            -len(candidate.rejection_codes),
            -(sum((candidate.runtime_by_query or {}).values()) or 0.0),
        )

    accepted_sorted = sorted(accepted, key=_key, reverse=True)
    return accepted_sorted[0]


def _query_id_from_patch_file(path: str) -> str | None:
    """Extract a query id from query-scoped generated source paths."""
    match = re.search(r"(?:^|/)query_q(?P<qid>\d+)\.(?:cpp|hpp)$", path)
    if match is None:
        return None
    return match.group("qid")


def _is_shared_patch_file(path: str) -> bool:
    """Return whether a file change should be treated as atomic shared state."""
    return _query_id_from_patch_file(path) is None


def _normalize_string_tuple(value: Any) -> tuple[str, ...]:
    """Normalize a JSON value into a tuple of non-empty strings."""
    if isinstance(value, str):
        return (value,) if value else ()
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(str(item) for item in value if str(item))


def parse_global_optimization_hypotheses(
    text: str | None,
) -> tuple[GlobalOptimizationHypothesis, ...]:
    """Parse diagnosis-stage JSON lines into structured hypotheses."""
    if not text:
        return ()
    hypotheses: list[GlobalOptimizationHypothesis] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line.startswith("{") or not line.endswith("}"):
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        hypothesis_id = str(payload.get("id") or f"h_{len(hypotheses) + 1:03d}")
        summary = str(payload.get("summary") or "").strip()
        if not summary:
            continue
        expected_impact = payload.get("expected_impact")
        if not isinstance(expected_impact, dict):
            expected_impact = None
        hypotheses.append(
            GlobalOptimizationHypothesis(
                id=hypothesis_id,
                summary=summary,
                evidence=_normalize_string_tuple(payload.get("evidence")),
                affected_queries=_normalize_string_tuple(
                    payload.get("affected_queries")
                ),
                suspected_runtime_path=_normalize_string_tuple(
                    payload.get("suspected_runtime_path")
                ),
                expected_mechanism=str(payload.get("expected_mechanism") or ""),
                expected_impact=expected_impact,
                correctness_risk=str(payload.get("correctness_risk") or "medium"),
                implementation_scope=_normalize_string_tuple(
                    payload.get("implementation_scope")
                ),
                verification_plan=_normalize_string_tuple(
                    payload.get("verification_plan")
                ),
                evidence_gap=payload.get("evidence_gap") is True,
            )
        )
        if len(hypotheses) >= GLOBAL_OPTIMIZATION_MAX_HYPOTHESES:
            break
    return tuple(hypotheses)


@dataclass(frozen=True)
class OptimizationFailureState:
    failure_code: str
    failure_detail: str
    final_correctness: bool = False


@dataclass(frozen=True)
class RawTraceExecutionIssue:
    failure_code: str
    failure_detail: str


class TpchMonetdbOptimizationConversation(AbstractConversation):
    """TPC-H MonetDB-specific 3-stage optimization conversation.
    
    架构说明:
    - 使用 ReferenceManifest 管理查询实例化
    - GeneratedTpchRuntimeProvider: 测量被测引擎 runtime
    - MonetDBBaselineProvider: 测量 TPC-H 参考基线 runtime (3次中位数)
    - Speedup 计算基于同一 instantiation_id 的对比
    """

    def __init__(
        self,
        query_ids: List[str],
        run_tool: RunTool,
        verify_sf_list: List[int],
        benchmark_sf: int,
        git_snapshotter: GitSnapshotter,
        session: AdvancedSQLiteSession,
        wandb_run_hook: Optional[WandbRunHook],
        bespoke_storage: bool = True,
        revert_on_regression: bool = True,
        manifest_path: Optional[Path] = None,
        conv_name: str = "",
        artifacts_dir: str = "",
        start_snapshot_hash: str = "",
        benchmark: str = "tpch",
        baseline_backend: Optional[str] = None,
        baseline_query_file_dir: Optional[Path] = None,
        benchmark_mode: str = "system-parity",
        storage_mode: str = "persistent",
        base_data_dir: Optional[Path] = None,
        wandb_init_result: Optional[Any] = None,
        hardware_counter_preflight: Optional[Any] = None,
        target_cpu: Optional[str] = None,
        large_sf: Optional[int] = None,
        **kwargs,
    ) -> None:
        """初始化 phase9 optimization conversation 与 runtime/baseline provider.

        显式接收 optimization 专属 measurement 参数，避免将它们透传给父类构造器。
        """
        super().__init__(
            allowed_choices=("u",),
            **kwargs,
        )

        self.query_ids = query_ids
        self.bespoke_storage = True
        self.run_tool = run_tool
        self.verify_sf_list = verify_sf_list
        self.benchmark_sf = benchmark_sf
        self.required_validation_sf_list = required_validation_scale_factors(
            verify_sf_list, benchmark_sf
        )
        self.git_snapshotter = git_snapshotter
        self.revert_on_regression = revert_on_regression
        self.session = session
        self.wandb_run_hook = wandb_run_hook
        self.wandb_init_result = wandb_init_result
        self.conv_name = conv_name
        self.artifacts_dir = artifacts_dir
        self.start_snapshot_hash = start_snapshot_hash
        self.benchmark = benchmark.strip().lower()
        if self.benchmark != "tpch":
            raise ValueError(
                "Optimization path only supports benchmark='tpch' after "
                "legacy baseline removal. "
                f"Got {benchmark!r}."
            )
        if baseline_backend not in (None, "monetdb"):
            raise ValueError(
                "TPC-H optimization only supports baseline_backend='monetdb'. "
                f"Got {baseline_backend!r}."
            )
        if baseline_query_file_dir is not None:
            raise ValueError(
                "TPC-H optimization does not support baseline_query_file_dir "
                "after legacy query-file baseline removal."
            )
        self.baseline_backend = "monetdb"
        self.baseline_query_file_dir = None
        self.benchmark_mode = benchmark_mode
        self.storage_mode = storage_mode
        self.baseline_run_started_at = (
            datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        )
        self.baseline_max_age_seconds = kwargs.get("baseline_max_age_seconds")
        self.base_data_dir = Path(base_data_dir) if base_data_dir is not None else None
        self.hardware_counter_preflight = hardware_counter_preflight
        self.target_cpu = target_cpu
        self.large_sf = large_sf
        self.hardware_counter_summary_by_query: dict[str, Any] = {}
        self.compiler_vectorization_summary: dict[str, Any] = {}
        self.measurement_repetition: dict[str, Any] = {}
        self.measurement_records: list[dict[str, Any]] = []
        self.hotspot_analysis_degraded: bool = False
        self.hotspot_analysis_failure_reason: str | None = None
        self.completed_stage_summaries: list[StageRunSummary] = []

        assert not self.replay, (
            "Replay mode is not supported for TpchMonetdbOptimizationConversation."
        )

        self.query_rt_log: Dict[str, float] = {}
        self.best_rt_log: Dict[str, float] = {}
        self.regression_tolerance: float = kwargs.get("regression_tolerance", 0.05)
        self.global_regression_records: list[dict[str, Any]] = []

        # 初始化 Provider 架构
        self.manifest = ReferenceManifest(manifest_path)
        self.impl_provider = GeneratedTpchRuntimeProvider(
            benchmark_mode=benchmark_mode,
            storage_mode=storage_mode,
        )
        self.baseline_provider = MonetDBBaselineProvider(
            benchmark_mode=benchmark_mode,
            storage_mode=storage_mode,
        )
        
        # 生成或加载 manifest
        self._initialize_manifest()
        return None

    def _objective_ids_for_prompt(self) -> list[str]:
        """返回当前 benchmark 对应的优化目标标识。"""
        return ["tpch-docker-monetdb-objective-v1"]

    def _data_law_ids_for_prompt(self) -> list[str]:
        """返回当前 benchmark 对应的数据法则标识。"""
        return [
            "LAW_TPCH_TABLE_CARDINALITY",
            "LAW_TPCH_JOIN_GRAPH",
            "LAW_TPCH_OUTPUT_ORDERING",
            "LAW_TPCH_NUMERIC_TOLERANCE",
            "LAW_TPCH_RUNTIME_BOUNDARY",
        ]

    def _baseline_display_name(self) -> str:
        """返回当前 benchmark 的用户可见 baseline 名称。"""
        return "MonetDB"

    def _baseline_engine_name(self) -> str:
        """返回当前 baseline provider 的机器可读 engine 名称。"""
        provider = getattr(self, "baseline_provider", None)
        engine = getattr(provider, "engine", None)
        if isinstance(engine, str) and engine.strip():
            return engine.strip().lower()
        return "monetdb"

    def _initialize_manifest(self) -> None:
        """初始化 Reference Manifest.

        如果 manifest 文件存在则加载，否则生成新的 manifest。
        """
        if self.manifest.manifest_path.exists():
            logger.info(f"Loading existing manifest from {self.manifest.manifest_path}")
            self.manifest.load()
            pruned_count = 0
            added_count = self.manifest.ensure_tpch_instantiations(
                query_ids=self.query_ids,
                scale_factor=self.benchmark_sf,
                seed=self.benchmark_sf,
                num_instantiations=3,
            )
            if pruned_count > 0 or added_count > 0:
                self.manifest.save()
                logger.info(
                    "Reconciled existing manifest: pruned=%d, backfilled=%d",
                    pruned_count,
                    added_count,
                )
        else:
            logger.info("Generating new reference manifest")
            # 使用相同的 seed 保证可重复性
            self.manifest = ReferenceManifest.generate_from_tpch(
                query_ids=self.query_ids,
                scale_factor=self.benchmark_sf,
                seed=self.benchmark_sf,
                manifest_path=self.manifest.manifest_path,
                num_instantiations=3,
            )
            self.manifest.save()
            logger.info(f"Generated and saved manifest with {len(self.manifest._instantiations)} instantiations")
        return None

    def _persist_failure_summary(
        self,
        *,
        failure_code: str,
        failure_detail: str,
        final_correctness: bool = False,
    ) -> None:
        conv_name = getattr(self, "conv_name", "")
        artifacts_dir = getattr(self, "artifacts_dir", "")
        if not conv_name or not artifacts_dir:
            return None
        start_snapshot_hash = getattr(self, "start_snapshot_hash", "")
        final_snapshot_hash = self.git_snapshotter.current_hash or start_snapshot_hash
        if not final_snapshot_hash:
            final_snapshot_hash = "unknown"
        persist_optimization_run(
            benchmark=getattr(self, "benchmark", "tpch"),
            conv_name=conv_name,
            query_list=self.query_ids,
            is_bespoke_storage=self.bespoke_storage,
            start_snapshot_hash=start_snapshot_hash,
            final_snapshot_hash=final_snapshot_hash,
            artifacts_dir=Path(artifacts_dir),
            success=False,
            final_correctness=final_correctness,
            failure_code=failure_code,
            failure_detail=failure_detail,
        )
        return None

    def _classify_failure_text(self, text: str) -> str | None:
        model_code = classify_model_failure(text)
        if model_code is not None:
            return model_code
        return classify_infra_failure(text)

    def _build_raw_trace_text(self, run_result: Any) -> str:
        return "\n".join(
            item
            for item in (
                run_result.msg,
                run_result.resp or "",
                run_result.out or "",
                run_result.err or "",
            )
            if item
        )

    def _trace_run_has_clean_exit_and_success_markers(self, raw_text: str) -> bool:
        exit_matches = list(re.finditer(r"exit_code:\s*(-?\d+)\s+signal:\s*(\d+)", raw_text))
        if not exit_matches:
            return False
        if not all(
            int(match.group(1)) == 0 and int(match.group(2)) == 0
            for match in exit_matches
        ):
            return False
        success_markers = ("query done", "Query ms:")
        return all(marker in raw_text for marker in success_markers)

    def _classify_raw_trace_execution(
        self,
        *,
        query_id: str,
        run_result: Any,
    ) -> RawTraceExecutionIssue | None:
        """Classify raw trace execution without relying on validator/cache status."""
        raw_text = self._build_raw_trace_text(run_result)
        if (
            "child terminates" in raw_text
            and self._trace_run_has_clean_exit_and_success_markers(raw_text)
        ):
            return None
        known_code = classify_infra_failure(raw_text, run_result.metrics)
        if known_code is not None:
            return RawTraceExecutionIssue(
                failure_code=known_code,
                failure_detail=(
                    f"Query {query_id}: raw trace execution failed with "
                    f"{known_code}. {raw_text[:1200]}"
                ),
            )
        exit_match = re.search(r"exit_code:\s*(-?\d+)\s+signal:\s*(\d+)", raw_text)
        if exit_match is not None:
            exit_code = int(exit_match.group(1))
            signal_code = int(exit_match.group(2))
            if signal_code != 0 or exit_code != 0:
                return RawTraceExecutionIssue(
                    failure_code="TRACE_RAW_EXECUTION_FAILED",
                    failure_detail=(
                        f"Query {query_id}: raw trace execution returned "
                        f"exit_code={exit_code}, signal={signal_code}. "
                        f"{raw_text[:1200]}"
                    ),
                )
        metrics = run_result.metrics or {}
        if metrics.get("validation/compile_error", False):
            return RawTraceExecutionIssue(
                failure_code="TRACE_RAW_EXECUTION_FAILED",
                failure_detail=(
                    f"Query {query_id}: raw trace compile/execution failed. "
                    f"{raw_text[:1200]}"
                ),
            )
        return None

    def _collect_baselines_at_checkpoint(self) -> None:
        """外环 checkpoint：收集所有缺失的 baseline 并写入 manifest.

        Phase9 policy: baseline 测量只允许在外环固定 checkpoint 执行，
        禁止在 prompt 内环（每次 LLM turn）自动触发。合法 checkpoint 为：
          - 初始校准（__init__ 后首次调用）
          - stage 结束
          - round 结束
          - final summary

        prompt 内环只读 manifest cache；若 cache 中存在 baseline 则直接复用，
        不调用 baseline_provider.measure()。
        """
        if not hasattr(self, "manifest") or self.manifest is None:
            logger.debug("Outer-loop checkpoint skipped: manifest not initialised")
            return None
        logger.info(
            "Outer-loop checkpoint: collecting baselines for all manifest instantiations"
        )
        collected = 0
        for inst_id, inst in self.manifest._instantiations.items():
            lookup = self._lookup_baseline_runtime(inst_id)
            if lookup.status == "compatible":
                logger.debug(f"Baseline already cached for {inst_id}, skipping collection")
                continue
            if lookup.status == "stale":
                self.manifest.remove_runtime(inst_id)
                logger.info(
                    "Removed stale baseline for %s: %s",
                    inst_id,
                    lookup.reason,
                )
            measurement = self.baseline_provider.measure(inst)
            self.manifest.record_runtime(measurement)
            collected += 1
            logger.info(
                "Collected baseline for %s: %.3fms",
                inst_id,
                measurement.runtime_ms,
            )
        if collected > 0:
            self.manifest.save()
        logger.info(f"Outer-loop checkpoint done: {collected} new baselines collected")
        return None

    def _lookup_baseline_runtime(self, instantiation_id: str) -> Any:
        """Look up a baseline only when exact-instantiation metadata matches."""
        instantiation = self.manifest.get_instantiation(instantiation_id)
        return self.manifest.lookup_runtime(
            instantiation_id,
            benchmark_mode=self.baseline_provider.benchmark_mode,
            storage_mode=self.baseline_provider.storage_mode,
            workers=self.baseline_provider.workers,
            engine=self.baseline_provider.engine,
            measurement_kind=MeasurementKind.EXACT_INSTANTIATION.value,
            query_id=None if instantiation is None else instantiation.query_id,
            args_string=None if instantiation is None else instantiation.args_string,
            scale_factor=None if instantiation is None else instantiation.scale_factor,
            measurement_shape_status=MeasurementShapeStatus.KNOWN.value,
            baseline_run_started_at=getattr(self, "baseline_run_started_at", None),
            max_age_seconds=getattr(self, "baseline_max_age_seconds", None),
            required_provenance_keys=self._required_baseline_provenance_keys(),
        )

    def _required_baseline_provenance_keys(self) -> tuple[str, ...]:
        """Return required provenance keys for the active baseline backend."""
        engine = getattr(self.baseline_provider, "engine", None)
        if engine == "monetdb":
            return ("baseline_backend", "source_protocol")
        backend = getattr(self.baseline_provider, "backend", None)
        if backend in (None, ""):
            return ()
        return ("baseline_backend",)

    def _refresh_query_baselines_for_stage(
        self,
        changed_files: set[str],
    ) -> None:
        """Refresh query baselines only when the stage diff requires it."""
        if not hasattr(self, "manifest") or self.manifest is None:
            return None
        if not hasattr(self, "baseline_provider") or self.baseline_provider is None:
            return None
        policy = BaselineRoutingPolicy.from_changed_files(sorted(changed_files))
        if policy.should_skip("query_baseline"):
            logger.info(
                "Stage-end checkpoint: skipping query baseline refresh "
                f"for change_type={policy.change_type.value}"
            )
            return None

        refreshed = 0
        for inst_id, inst in self.manifest._instantiations.items():
            lookup = self._lookup_baseline_runtime(inst_id)
            if lookup.status == "stale":
                self.manifest.remove_runtime(inst_id)
                logger.info(
                    "Removed stale baseline for %s: %s",
                    inst_id,
                    lookup.reason,
                )
            measurement = self.baseline_provider.measure(inst)
            self.manifest.record_runtime(measurement)
            refreshed += 1
        if refreshed > 0:
            self.manifest.save()
        logger.info(
            "Stage-end checkpoint: refreshed query baselines "
            f"for {refreshed} instantiations (change_type={policy.change_type.value})"
        )
        return None

    def _collect_scale_factors(self) -> set[int]:
        """从 manifest 的所有 instantiation 中收集 distinct scale_factor。"""
        if not hasattr(self, "manifest") or self.manifest is None:
            return set()
        return {inst.scale_factor for inst in self.manifest._instantiations.values()}

    def _refresh_ingest_baseline_for_stage(
        self,
        changed_files: set[str],
    ) -> None:
        """Skip legacy ingest-baseline refresh on the TPC-H runtime path."""
        _ = changed_files
        return None

    def _log_ingest_comparison_if_complete(
        self,
        stage_name: str,
        validation_metrics: Dict[str, Any],
    ) -> None:
        """Emit formal ingest telemetry only when the comparison payload is complete."""
        if self.wandb_run_hook is None:
            return None
        ingest_auxiliary = getattr(self, "_ingest_auxiliary", {}).get(self.benchmark_sf)
        if ingest_auxiliary is None:
            logger.warning(
                "Skipping ingest telemetry for %s: %s ingest baseline missing for sf=%d",
                stage_name,
                self._baseline_display_name(),
                self.benchmark_sf,
            )
            return None
        bespoke_ingest_ms = validation_metrics.get("validation/generated_tpch_ingest_ms")
        derived_bespoke, derive_missing = derive_bespoke_ingest_metrics(
            self.benchmark_sf,
            bespoke_ingest_ms,
        )
        if derived_bespoke is None:
            logger.warning(
                "Skipping ingest telemetry for %s: %s",
                stage_name,
                ", ".join(derive_missing),
            )
            return None
        is_complete, missing = check_ingest_completeness(
            baseline_ingest_ms=ingest_auxiliary.runtime_measurement.runtime_ms,
            baseline_ingest_rows_per_sec=ingest_auxiliary.rows_per_sec,
            baseline_ingest_metrics_per_sec=ingest_auxiliary.metrics_per_sec,
            baseline_workers=ingest_auxiliary.workers,
            bespoke_ingest_ms=bespoke_ingest_ms,
            bespoke_ingest_rows_per_sec=derived_bespoke.rows_per_sec,
            bespoke_ingest_metrics_per_sec=derived_bespoke.metrics_per_sec,
        )
        if not is_complete:
            logger.warning(
                "Skipping ingest telemetry for %s: incomplete ingest comparison (%s)",
                stage_name,
                ", ".join(missing),
            )
            return None
        baseline_ingest_ms = ingest_auxiliary.runtime_measurement.runtime_ms
        logger.info(
            "[ingest secondary] stage=%s  bespoke=%.0f ms (%.0f rows/s)"
            "  %s=%.0f ms (%.0f rows/s)",
            stage_name,
            bespoke_ingest_ms,
            derived_bespoke.rows_per_sec,
            self._baseline_display_name(),
            baseline_ingest_ms,
            ingest_auxiliary.rows_per_sec,
        )
        self.wandb_run_hook.log_ingest_comparison(
            stage_name=stage_name,
            bespoke_ingest_ms=bespoke_ingest_ms,
            bespoke_load_ms=validation_metrics.get("validation/generated_tpch_load_ms"),
            bespoke_build_ms=validation_metrics.get("validation/generated_tpch_build_ms"),
            bespoke_rows_per_sec=derived_bespoke.rows_per_sec,
            bespoke_metrics_per_sec=derived_bespoke.metrics_per_sec,
            baseline_ingest_ms=baseline_ingest_ms,
            baseline_rows_per_sec=ingest_auxiliary.rows_per_sec,
            baseline_metrics_per_sec=ingest_auxiliary.metrics_per_sec,
            baseline_workers=ingest_auxiliary.workers,
            baseline_engine=self._baseline_engine_name(),
            baseline_label=self._baseline_display_name(),
        )
        log_ingest_summary = getattr(self.wandb_run_hook, "log_ingest_summary", None)
        if callable(log_ingest_summary):
            log_ingest_summary(
                stage_name=stage_name,
                bespoke_ingest_ms=bespoke_ingest_ms,
                baseline_ingest_ms=baseline_ingest_ms,
                baseline_engine=self._baseline_engine_name(),
                baseline_label=self._baseline_display_name(),
            )
        return None

    def _measure_with_manifest(
        self,
        query_id: str,
        exec_callback: Optional[Callable] = None,
        allow_baseline_collection: bool = False,
        scale_factor: int | None = None,
        output_mode: str = QUERY_OUTPUT_MODE_NO_OUTPUT,
    ) -> tuple[float, float, float, bool]:
        """使用 Manifest 全部实例测量，返回 (impl_rt_s, baseline_rt_s, speedup, lazy_build_suspected).

        Phase9 outer-loop policy: baseline 默认只读 manifest cache。
        若 cache miss 且 allow_baseline_collection=False，则 raise；
        仅外环 checkpoint 可传 allow_baseline_collection=True。
        lazy_build_suspected=True 时，本轮结果不得进入正式 speedup 结论。
        """
        instantiations = self.manifest.get_instantiations_for_query(
            query_id=query_id,
            scale_factor=self.benchmark_sf if scale_factor is None else scale_factor,
        )
        if not instantiations:
            raise ValueError(f"No instantiation found for query {query_id}")

        impl_rts: list[float] = []
        baseline_rts: list[float] = []
        any_lazy_suspected = False

        for inst in instantiations:
            lookup = self._lookup_baseline_runtime(inst.instantiation_id)
            baseline_measurement = lookup.measurement
            if lookup.status == "missing":
                if not allow_baseline_collection:
                    raise RuntimeError(
                        f"Baseline not cached for {inst.instantiation_id}. "
                        "Call _collect_baselines_at_checkpoint() at an outer-loop checkpoint first."
                    )
                logger.info(
                    "Outer-loop: collecting baseline for %s",
                    inst.instantiation_id,
                )
                baseline_measurement = self.baseline_provider.measure(inst)
                self.manifest.record_runtime(baseline_measurement)
            elif lookup.status == "stale":
                if not allow_baseline_collection:
                    raise RuntimeError(
                        f"Baseline cache is stale for {inst.instantiation_id}. "
                        f"Reason: {lookup.reason}. "
                        "Call _collect_baselines_at_checkpoint() at an outer-loop checkpoint first."
                    )
                self.manifest.remove_runtime(inst.instantiation_id)
                logger.info(
                    "Outer-loop: refreshing stale baseline for %s: %s",
                    inst.instantiation_id,
                    lookup.reason,
                )
                baseline_measurement = self.baseline_provider.measure(inst)
                self.manifest.record_runtime(baseline_measurement)
            else:
                logger.debug(f"Using cached baseline for {inst.instantiation_id}")
            if baseline_measurement is None:
                raise RuntimeError(
                    f"Failed to resolve baseline for {inst.instantiation_id}"
                )
            baseline_rts.append(baseline_measurement.runtime_ms)
            self._record_measurement_evidence(
                measurement=baseline_measurement,
                engine=self.baseline_provider.engine,
            )

            if exec_callback:
                logger.info(f"Measuring Generated TPC-H runtime for {inst.instantiation_id}")
                primary_metric_kind = (
                    KERNEL_RUNTIME_METRIC_KIND
                    if output_mode == QUERY_OUTPUT_MODE_NO_OUTPUT
                    else QUERY_RUNTIME_METRIC_KIND
                )
                impl_measurement = self.impl_provider.measure(
                    inst,
                    exec_callback,
                    primary_metric_kind=primary_metric_kind,
                )
                impl_measurement.provenance = {
                    **dict(getattr(impl_measurement, "provenance", {}) or {}),
                    "output_mode": output_mode,
                }
                impl_rts.append(impl_measurement.runtime_ms)
                self._record_measurement_evidence(
                    measurement=impl_measurement,
                    engine=self.impl_provider.engine,
                )
                if getattr(impl_measurement, "_lazy_build_suspected", False):
                    any_lazy_suspected = True

        self.manifest.save()

        baseline_median_ms = sorted(baseline_rts)[len(baseline_rts) // 2]
        impl_median_ms = sorted(impl_rts)[len(impl_rts) // 2] if impl_rts else float("inf")

        impl_rt_s = impl_median_ms / 1000.0
        baseline_rt_s = baseline_median_ms / 1000.0
        speedup = safe_speedup(baseline_median_ms, impl_median_ms) or 0.0

        if any_lazy_suspected:
            logger.warning(
                "Query %s: lazy-build suspected – this run is NOT eligible for official "
                "phase9 speedup aggregation (impl=%s baseline=%s speedup=%.2fx)",
                query_id,
                format_duration_ms(impl_median_ms),
                format_duration_ms(baseline_median_ms),
                speedup,
            )
        else:
            metric_label = (
                "no_csv/kernel_ms"
                if output_mode == QUERY_OUTPUT_MODE_NO_OUTPUT
                else "full_csv/query_e2e_ms"
            )
            logger.info(
                "Speedup for %s (%s): impl=%s baseline=%s speedup=%.2fx "
                "(median of %d instances)",
                query_id,
                metric_label,
                format_duration_ms(impl_median_ms),
                format_duration_ms(baseline_median_ms),
                speedup,
                len(instantiations),
            )
        return impl_rt_s, baseline_rt_s, speedup, any_lazy_suspected

    def _record_measurement_evidence(self, *, measurement: Any, engine: str) -> None:
        """Store one normalized measurement record for summary provenance checks."""
        query_id = getattr(measurement, "query_id", None)
        if query_id is None:
            query_id = (getattr(measurement, "provenance", {}) or {}).get("query_id")
        if query_id is None:
            return None
        record = QueryMeasurementRecord(
            query_id=str(query_id),
            engine=engine,
            measurement_kind=getattr(measurement, "measurement_kind", None) or "",
            runtime_ms=float(getattr(measurement, "runtime_ms")),
            instantiation_id=getattr(measurement, "instantiation_id", None),
            args_string=getattr(measurement, "args_string", None),
            scale_factor=getattr(measurement, "scale_factor", None),
            row_count=getattr(measurement, "row_count", None),
            output_row_count=getattr(measurement, "output_row_count", None),
            query_file_sha256=getattr(measurement, "query_file_sha256", None),
            measurement_shape_status=getattr(
                measurement,
                "measurement_shape_status",
                "unknown",
            ),
            provenance=dict(getattr(measurement, "provenance", {}) or {}),
        )
        existing_keys = {
            (
                item.get("query_id"),
                item.get("engine"),
                item.get("instantiation_id"),
                item.get("measurement_kind"),
                (item.get("provenance") or {}).get("output_mode", ""),
            )
            for item in self.measurement_records
        }
        key = (
            record.query_id,
            record.engine,
            record.instantiation_id,
            str(record.measurement_kind),
            record.provenance.get("output_mode", ""),
        )
        if key not in existing_keys:
            self.measurement_records.append(record.to_dict())
        return None

    def _get_baseline_runtime_ms_by_query(self) -> dict[str, float]:
        """Return baseline median runtime in milliseconds for each query."""
        if not hasattr(self, "manifest") or self.manifest is None:
            return {}
        baseline_ms_by_query: dict[str, float] = {}
        for query_id in self.query_ids:
            instantiations = self.manifest.get_instantiations_for_query(
                query_id=query_id,
                scale_factor=self.benchmark_sf,
            )
            runtimes_ms: list[float] = []
            for inst in instantiations:
                lookup = self._lookup_baseline_runtime(inst.instantiation_id)
                measurement = lookup.measurement
                if measurement is None:
                    continue
                if lookup.status != "compatible":
                    logger.warning(
                        "Skipping non-compatible baseline cache entry for %s (%s)",
                        inst.instantiation_id,
                        lookup.status,
                    )
                    continue
                runtimes_ms.append(measurement.runtime_ms)
            if runtimes_ms:
                baseline_ms_by_query[query_id] = sorted(runtimes_ms)[len(runtimes_ms) // 2]
        return baseline_ms_by_query

    def _measure_all_queries(
        self,
        *,
        output_mode: str = QUERY_OUTPUT_MODE_NO_OUTPUT,
    ) -> dict[str, float]:
        """测量全部 query 的当前 impl runtime，返回 {query_id: rt_s}."""
        results: dict[str, float] = {}
        for qid in self.query_ids:
            try:
                rt_s, _, _, _ = self._measure_with_manifest(
                    query_id=qid,
                    exec_callback=self._make_exec_callback(
                        qid,
                        output_mode=output_mode,
                    ),
                    output_mode=output_mode,
                )
                results[qid] = rt_s
            except Exception as exc:
                logger.error(f"Failed to measure query {qid}: {exc}")
                results[qid] = float("inf")
        return results

    def _measure_scope_runtime(
        self,
        query_ids: list[str],
        *,
        scale_factor: int | None = None,
        output_mode: str = QUERY_OUTPUT_MODE_NO_OUTPUT,
    ) -> tuple[dict[str, float], dict[str, float], float, bool]:
        """Measure one query unit or query batch and return aggregate runtime."""
        runtime_by_query: dict[str, float] = {}
        baseline_by_query: dict[str, float] = {}
        lazy_suspected = False
        for query_id in query_ids:
            rt_s, baseline_rt_s, _, query_lazy = self._measure_with_manifest(
                query_id=query_id,
                exec_callback=self._make_exec_callback(
                    query_id,
                    scale_factor=scale_factor,
                    output_mode=output_mode,
                ),
                scale_factor=scale_factor,
                output_mode=output_mode,
            )
            runtime_by_query[query_id] = rt_s
            baseline_by_query[query_id] = baseline_rt_s
            lazy_suspected = lazy_suspected or query_lazy
        aggregate_runtime_s = aggregate_scope_runtime_seconds(runtime_by_query)
        return runtime_by_query, baseline_by_query, aggregate_runtime_s, lazy_suspected

    def _measure_query_output_split(
        self,
        query_id: str,
    ) -> QueryOutputSplitMeasurement:
        """Measure full CSV and no-output runtimes for one query."""
        full_csv_s, _, _, _ = self._measure_with_manifest(
            query_id=query_id,
            exec_callback=self._make_exec_callback(
                query_id,
                output_mode=QUERY_OUTPUT_MODE_FULL_CSV,
            ),
            output_mode=QUERY_OUTPUT_MODE_FULL_CSV,
        )
        no_output_s, _, _, _ = self._measure_with_manifest(
            query_id=query_id,
            exec_callback=self._make_exec_callback(
                query_id,
                output_mode=QUERY_OUTPUT_MODE_NO_OUTPUT,
            ),
            output_mode=QUERY_OUTPUT_MODE_NO_OUTPUT,
        )
        materialization_s = max(0.0, full_csv_s - no_output_s)
        ratio = materialization_s / full_csv_s if full_csv_s > 0 else 0.0
        return QueryOutputSplitMeasurement(
            query_id=query_id,
            full_csv_s=full_csv_s,
            no_output_s=no_output_s,
            materialization_s=materialization_s,
            materialization_ratio=ratio,
        )

    def _collect_query_output_split_measurements(
        self,
        query_ids: list[str],
    ) -> dict[str, QueryOutputSplitMeasurement]:
        """Collect output materialization evidence without invoking correctness validation."""
        split_by_query: dict[str, QueryOutputSplitMeasurement] = {}
        for query_id in query_ids:
            split_by_query[query_id] = self._measure_query_output_split(query_id)
        return split_by_query

    def _collect_repeated_scope_measurements(
        self,
        query_ids: list[str],
        *,
        scale_factor: int,
        repetitions: int,
    ) -> dict[str, Any]:
        """Collect repeated large-SF measurements for hotspot acceptance."""
        from tpch_monetdb.utils.summary_gates import MEASUREMENT_AGGREGATION_GEOMEAN

        aggregate_runtime_ms_samples: list[float] = []
        per_query_runtime_ms_samples: dict[str, list[float]] = {
            query_id: [] for query_id in query_ids
        }
        lazy_build_detected = False
        for _ in range(repetitions):
            runtime_by_query, _baseline_by_query, aggregate_runtime_s, lazy_suspected = (
                self._measure_scope_runtime(query_ids, scale_factor=scale_factor)
            )
            aggregate_runtime_ms_samples.append(aggregate_runtime_s * 1000.0)
            lazy_build_detected = lazy_build_detected or lazy_suspected
            for query_id, runtime_s in runtime_by_query.items():
                per_query_runtime_ms_samples[query_id].append(runtime_s * 1000.0)
        aggregate_median = statistics.median(aggregate_runtime_ms_samples)
        return {
            "scale_factor": scale_factor,
            "query_ids": list(query_ids),
            "repetitions": repetitions,
            "sample_count": len(aggregate_runtime_ms_samples),
            "aggregate_runtime_ms_samples": aggregate_runtime_ms_samples,
            "aggregate_runtime_ms_median": aggregate_median,
            "aggregate_runtime_ms_min": min(aggregate_runtime_ms_samples),
            "aggregate_runtime_ms_max": max(aggregate_runtime_ms_samples),
            "per_query_runtime_ms_samples": per_query_runtime_ms_samples,
            "aggregation_method": MEASUREMENT_AGGREGATION_GEOMEAN,
            "lazy_build_detected": lazy_build_detected,
            "source_command": "GeneratedTpchRuntimeProvider.measure via manifest instantiations",
        }

    def _make_exec_callback(
        self,
        query_id: str,
        trace_mode: bool = False,
        scale_factor: int | None = None,
        output_mode: str = QUERY_OUTPUT_MODE_FULL_CSV,
    ) -> Callable[[List[str], int], tuple[str, str, str]]:
        """为 manifest/provider 路径创建执行回调。"""
        def exec_callback(args_list: List[str], timeout_s: int) -> tuple[str, str, str]:
            run_result = self.run_tool.run_worker(
                scale_factor=self.benchmark_sf if scale_factor is None else scale_factor,
                optimize=True,
                query_id=[query_id],
                trace_mode=trace_mode,
                external_call=True,
                stdin_args_data=args_list,
                output_mode=output_mode,
                execution_timeout_s=timeout_s,
            )
            resp = run_result.resp or ""
            out = run_result.out or ""
            err = run_result.err or ""
            metrics = run_result.metrics or {}
            failure_code = metrics.get("validation/failure_code")
            if isinstance(failure_code, str) and failure_code:
                failure_detail = str(
                    metrics.get("validation/failure_detail")
                    or run_result.err
                    or run_result.msg
                    or "generated runtime execution failed"
                )
                raise RuntimeError(f"[ERROR:{failure_code}] {failure_detail}")
            return resp, out, err

        return exec_callback

    def _build_query_stage(
        self,
        query_id: str,
        mandatory_constraints: str,
        trace_summary: str,
        *,
        hardware_counter_evidence: str = "",
        active_unit_id: str | None = None,
        active_unit_query_ids: tuple[str, ...] = (),
    ) -> StageConfig:
        """Build the trace-expert prompt stage for a representative query/unit."""
        sf = self.benchmark_sf
        expert_knowledge = load_expert_knowledge()
        descriptor = (
            f"TPC-H MonetDB Trace+Expert Optim ({query_id})"
            if active_unit_id is None
            else (
                "TPC-H MonetDB Trace+Expert Optim "
                f"({active_unit_id}; representative q{query_id}; "
                f"queries {','.join(active_unit_query_ids)})"
            )
        )
        resolved_hardware_counter_evidence = hardware_counter_evidence
        if not resolved_hardware_counter_evidence and hasattr(
            self, "hardware_counter_summary_by_query"
        ):
            summary_map = getattr(self, "hardware_counter_summary_by_query") or {}
            summary = summary_map.get(query_id)
            if summary:
                resolved_hardware_counter_evidence = _format_hardware_counter_evidence_for_prompt(
                    summary
                )
        return StageConfig(
            name="trace_expert",
            get_descriptor=lambda: descriptor,
            get_prompt=lambda rt: tpch_monetdb_optim_prompt_trace_expert(
                query_id=query_id,
                constraints_str=mandatory_constraints,
                expert_knowledge=expert_knowledge,
                trace_summary=trace_summary,
                current_rt_ms=rt,
                target_rt_ms=rt / 2,
                sf=sf,
                storage_is_bespoke=True,
                hardware_counter_evidence=resolved_hardware_counter_evidence,
            ),
            max_turns=get_optim_stage_max_turns("trace_expert"),
        )

    async def _run_stage(
        self,
        query_id: str,
        stage: StageConfig,
        pretext_optim: str,
        rt_before_s: float,
        correctness_scale_factors: tuple[int, ...] | None = None,
        scope_query_ids: tuple[str, ...] | None = None,
        prompt_metadata: Optional[dict[str, Any]] = None,
        measurement_output_mode: str = QUERY_OUTPUT_MODE_NO_OUTPUT,
    ) -> StageResult:
        """Execute one optimization stage."""
        current_snapshot = self.git_snapshotter.current_hash
        assert current_snapshot is not None, "Current git snapshot is None."
        retained_written_files: tuple[str, ...] = ()
        resolved_scope_query_ids = (
            (query_id,) if scope_query_ids is None else tuple(scope_query_ids)
        )
        _validation_sf = (
            correctness_scale_factors
            if correctness_scale_factors is not None
            else self.required_validation_sf_list
        )

        # Run the LLM optimization loop
        stage_summary = await self._exec(
            pretext_optim + "\n" + stage.get_prompt(rt_before_s * 1000),
            stage.get_descriptor(),
            max_turns=stage.max_turns,
            tool_profile="optimization_general",
            prompt_metadata=prompt_metadata,
        )
        if stage_summary is not None:
            retained_written_files = stage_summary.written_files

        metrics: dict[str, Any] = {}
        baseline_rt_s = 0.0
        failed_scale_factor: Optional[int] = None
        rt_after_s = rt_before_s
        speedup = 0.0
        lazy_suspected = False
        stage_failed = False
        failure_message: Optional[str] = None
        runtime_by_query: dict[str, float] | None = None
        try:
            validation_summary = run_required_correctness_checks(
                self.run_tool,
                _validation_sf,
                list(resolved_scope_query_ids),
                trace_mode=False,
                optimize=True,
                external_call=True,
            )
            metrics = validation_summary.metrics or {}
            failed_scale_factor = validation_summary.failed_scale_factor

            assert validation_summary.success, (
                f"Implementation not correct after stage '{stage.name}' for query "
                f"{query_id}: {validation_summary.message}"
            )
            if len(resolved_scope_query_ids) == 1:
                rt_after_s, baseline_rt_s, speedup, lazy_suspected = self._measure_with_manifest(
                    query_id=query_id,
                    exec_callback=self._make_exec_callback(
                        query_id,
                        output_mode=measurement_output_mode,
                    ),
                    output_mode=measurement_output_mode,
                )
                runtime_by_query = {query_id: rt_after_s}
            else:
                runtime_by_query, baseline_by_query, rt_after_s, lazy_suspected = (
                    self._measure_scope_runtime(
                        list(resolved_scope_query_ids),
                        output_mode=measurement_output_mode,
                    )
                )
                baseline_rt_s = aggregate_scope_runtime_seconds(baseline_by_query)
                speedup = safe_speedup(baseline_rt_s * 1000.0, rt_after_s * 1000.0) or 0.0

            if (
                not lazy_suspected
                and runtime_by_query is not None
            ):
                for scope_query_id, scope_runtime_s in runtime_by_query.items():
                    self.query_rt_log[scope_query_id] = scope_runtime_s
                    if (
                        scope_query_id not in self.best_rt_log
                        or scope_runtime_s < self.best_rt_log[scope_query_id]
                    ):
                        self.best_rt_log[scope_query_id] = scope_runtime_s
            elif lazy_suspected:
                logger.warning(
                    "Stage '%s' scope %s: lazy-build suspected, skipping best_rt_log update.",
                    stage.name,
                    resolved_scope_query_ids,
                )
        except Exception as exc:
            stage_failed = True
            failure_message = (
                f"Stage '{stage.name}' failed for query {query_id}: {exc}"
            )
            rt_after_s = float("inf")
            speedup = 0.0
            logger.error(
                failure_message,
                exc_info=exc,
            )
            if self.revert_on_regression:
                try:
                    self.git_snapshotter.restore(current_snapshot)
                    retained_written_files = ()
                except Exception as restore_exc:
                    failure_message = (
                        f"{failure_message}; rollback restore failed: {restore_exc}"
                    )
                    logger.error(
                        failure_message,
                        exc_info=restore_exc,
                    )

        if not stage_failed and should_rollback_unit_regression(
            rt_before_s=rt_before_s,
            rt_after_s=rt_after_s,
            revert_on_regression=self.revert_on_regression,
        ):
            logger.warning(
                f"Reverting changes from stage '{stage.name}' for query {query_id}"
            )
            try:
                self.git_snapshotter.restore(current_snapshot)
                retained_written_files = ()
                rollback_summary = run_required_correctness_checks(
                    self.run_tool,
                    _validation_sf,
                    list(resolved_scope_query_ids),
                    trace_mode=False,
                    optimize=True,
                    external_call=True,
                )
                metrics = rollback_summary.metrics or {}

                if not rollback_summary.success:
                    logger.warning(
                        f"Reverted stage '{stage.name}' for query {query_id} "
                        f"but the reverted version is not correct. Re-running LLM fix."
                    )
                    rollback_stage_summary = await self._exec(
                        render_agent_text_asset(
                            "optimization.repair.rollback_correctness",
                            {
                                "query_id": query_id,
                                "query_ids": ", ".join(self.query_ids),
                            },
                        ),
                        stage.get_descriptor(),
                        max_turns=stage.max_turns,
                        tool_profile="optimization_general",
                    )
                    if rollback_stage_summary is not None:
                        retained_written_files = rollback_stage_summary.written_files
                    rollback_summary = run_required_correctness_checks(
                        self.run_tool,
                        self.required_validation_sf_list,
                        list(resolved_scope_query_ids),
                        trace_mode=False,
                        optimize=True,
                        external_call=True,
                    )
                    metrics = rollback_summary.metrics or {}
                    if not rollback_summary.success:
                        raise RuntimeError(rollback_summary.message)

                if len(resolved_scope_query_ids) == 1:
                    rt_after_s, baseline_rt_s, speedup, lazy_suspected = self._measure_with_manifest(
                        query_id=query_id,
                        exec_callback=self._make_exec_callback(
                            query_id,
                            output_mode=measurement_output_mode,
                        ),
                        output_mode=measurement_output_mode,
                    )
                    runtime_by_query = {query_id: rt_after_s}
                else:
                    runtime_by_query, baseline_by_query, rt_after_s, lazy_suspected = (
                        self._measure_scope_runtime(
                            list(resolved_scope_query_ids),
                            output_mode=measurement_output_mode,
                        )
                    )
                    baseline_rt_s = aggregate_scope_runtime_seconds(baseline_by_query)
                    speedup = safe_speedup(baseline_rt_s * 1000.0, rt_after_s * 1000.0) or 0.0
                if (
                    not lazy_suspected
                    and runtime_by_query is not None
                ):
                    for scope_query_id, scope_runtime_s in runtime_by_query.items():
                        self.query_rt_log[scope_query_id] = scope_runtime_s
                        if (
                            scope_query_id not in self.best_rt_log
                            or scope_runtime_s < self.best_rt_log[scope_query_id]
                        ):
                            self.best_rt_log[scope_query_id] = scope_runtime_s
            except Exception as exc:
                stage_failed = True
                failure_message = (
                    f"Rollback failed after stage '{stage.name}' for query {query_id}: {exc}"
                )
                rt_after_s = float("inf")
                speedup = 0.0
                logger.error(
                    failure_message,
                    exc_info=exc,
                )

        result = StageResult(
            name=stage.name,
            rt_before_s=rt_before_s,
            rt_after_s=rt_after_s,
            speedup_vs_baseline=speedup,
            written_files=retained_written_files,
            runtime_by_query=runtime_by_query,
            failed_scale_factor=failed_scale_factor,
            failed=stage_failed,
            failure_message=failure_message,
        )

        if self.wandb_run_hook is not None:
            final_snapshot = self.git_snapshotter.current_hash or ""
            validation_passed = not result.failed and bool(
                metrics.get("validation/correct", False)
            )
            self.wandb_run_hook.log_optimization_stage(
                query_id=query_id,
                stage_name=stage.name,
                rt_before_ms=rt_before_s * 1000,
                rt_after_ms=rt_after_s * 1000,
                snapshot_before=current_snapshot or "",
                snapshot_after=final_snapshot,
                validation_passed=validation_passed,
            )
            measurement_ok = (
                not result.failed
                and rt_after_s != float("inf")
                and baseline_rt_s > 0
                and not lazy_suspected
                and measurement_output_mode == QUERY_OUTPUT_MODE_NO_OUTPUT
            )
            if measurement_ok:
                impl_rt_ms = rt_after_s * 1000
                baseline_rt_ms = baseline_rt_s * 1000
                self.wandb_run_hook.log_optimization_speedup_vs_baseline(
                    query_id=query_id,
                    stage_name=stage.name,
                    no_csv_kernel_runtime_ms=impl_rt_ms,
                    baseline_runtime_ms=baseline_rt_ms,
                    baseline_engine=self._baseline_engine_name(),
                    baseline_label=self._baseline_display_name(),
                )

        if result.failed:
            logger.info(
                f"Query {query_id} | Stage '{stage.name}' failed: {result.failure_message}"
            )
            return result

        if result.improved:
            logger.info(
                f"Query {query_id} | Stage '{stage.name}': "
                f"{rt_before_s:.3f}s -> {rt_after_s:.3f}s "
                f"(improved x{result.improvement_factor:.2f}), "
                f"speedup vs {self._baseline_display_name()}: {speedup:.2f}x"
            )
        else:
            logger.info(
                f"Query {query_id} | Stage '{stage.name}': "
                f"{rt_before_s:.3f}s -> {rt_after_s:.3f}s "
                f"(no improvement), "
                f"speedup vs {self._baseline_display_name()}: {speedup:.2f}x"
            )

        return result

    async def run(self) -> Optional[List[str]]:
        """Run per-query trace_expert optimization followed by global candidates."""
        self.used = []

        queries_path = "queries.txt"
        queries_file = self.run_tool.cwd / queries_path
        assert queries_file.exists() and queries_file.stat().st_size > 0, (
            f"{queries_file} is missing or empty. "
            f"Ensure write_query_and_args_file was called before optimization."
        )
        pretext = tpch_monetdb_optim_prompt_pretext(
            queries_path=queries_path, num_queries=len(self.query_ids)
        )
        pretext_optim = tpch_monetdb_optim_prompt_pretext_optim(
            bespoke_storage=self.bespoke_storage,
        )
        mandatory_constraints = tpch_monetdb_optim_prompt_constraints(
            allow_storage_changes=self.bespoke_storage
        )
        pinning_prompt = tpch_monetdb_optim_prompt_pinning(core_id=3)

        # Ensure initial correctness with bounded repair: base_impl may pass its
        # own checks but still have edge-case bugs (e.g. SAMPLE BY window offset)
        # that only surface at specific scale factors.  A single LLM repair is
        # far cheaper than restarting the entire outer loop.
        precheck_ok = await self._check_correctness(self.query_ids, trace_mode=False)
        if not precheck_ok:
            await self._exec(
                render_agent_text_asset("optimization.repair.precheck_correctness"),
                "Fix Precheck Correctness",
                max_turns=get_stage_turn_budget("add_timings"),
                tool_profile="optimization_instrumentation",
                prompt_metadata=build_instrumentation_prompt_metadata(self.query_ids),
            )
            if not await self._check_correctness(self.query_ids, trace_mode=False):
                raise RuntimeError(
                    str(
                        ErrorEnvelope(
                            code="OPTIMIZATION_PRECHECK_FAILED",
                            category="correctness",
                            stage="optimization_initial",
                            message="Initial implementation is not correct after repair attempt. Fix before optimization.",
                            recoverable=False,
                            recommended_next_action="Run base-implementation phase again to fix correctness before optimization.",
                        )
                    )
                )

        # Perform pinning
        await self._exec(
            pretext + "\n" + pinning_prompt,
            "Pinning",
            max_turns=PINNING_PROMPT_MAX_TURNS,
            tool_profile="optimization_general",
        )

        # Enable validation
        await self._exec(VALIDATE_ON, None)
        await self._exec(VALIDATE_OUTPUT_STDOUT_OFF, None)

        # Ensure TRACE validation writes PROFILE/COUNT to file before the first
        # instrumentation batch asks the model to run with trace_mode enabled.
        await self._exec(
            render_agent_text_asset(
                "optimization.instrumentation.trace_to_file"
            ),
            "Trace->File",
            max_turns=get_stage_turn_budget("add_timings"),
            tool_profile="optimization_instrumentation",
            prompt_metadata=build_instrumentation_prompt_metadata(self.query_ids),
        )

        # Initial validation runs
        initial_validation = run_required_correctness_checks(
            self.run_tool,
            self.required_validation_sf_list,
            self.query_ids,
            optimize=True,
            external_call=True,
        )
        if not initial_validation.success:
            raise RuntimeError(
                str(
                    ErrorEnvelope(
                        code="OPTIMIZATION_PRECHECK_FAILED",
                        category="correctness",
                        stage="optimization_initial_validation",
                        message=(
                            f"Initial validation failed: {initial_validation.message}. "
                            f"failure_code={initial_validation.failure_code}, "
                            f"failed_scale_factor={initial_validation.failed_scale_factor}"
                        ),
                        recoverable=False,
                        recommended_next_action="Run base-implementation phase again to fix correctness before optimization.",
                    )
                )
            )

        # Outer-loop checkpoint: initial calibration — collect all baselines once.
        self._collect_baselines_at_checkpoint()

        try:
            # Add timing instrumentation with smoke-only correctness per batch
            from tpch_monetdb.conversations.optimization_instrumentation import (
                InstrumentationPolicy,
                build_instrumentation_policy,
                check_instrumentation_smoke,
            )
            instr_policy = build_instrumentation_policy(self.required_validation_sf_list)
            add_timings_prompt = tpch_monetdb_optim_prompt_add_timings()
            for i in range(0, len(self.query_ids), instr_policy.batch_size):
                qids = self.query_ids[i : min(i + instr_policy.batch_size, len(self.query_ids))]
                qids_str = ", ".join(qids)
                prompt = tpch_monetdb_optim_prompt_add_timings_per_query(
                    qids_str=qids_str,
                    refer_to_prev_queries=i > 0,
                    scale_factor=self.benchmark_sf,
                )
                full_prompt = add_timings_prompt + "\n" + prompt if i == 0 else prompt
                await self._exec(
                    full_prompt,
                    f"Add Timings for Queries {qids_str}",
                    max_turns=get_stage_turn_budget("add_timings"),
                    tool_profile="optimization_instrumentation",
                    prompt_metadata=build_instrumentation_prompt_metadata(qids),
                )
                await check_instrumentation_smoke(
                    qids=qids,
                    policy=instr_policy,
                    max_turns=get_stage_turn_budget("add_timings"),
                    check_correctness_fn=self._check_correctness_with_scale_factors,
                    exec_fn=self._exec,
                )

            self._delete_result_csvs(self.run_tool.cwd)

            await check_instrumentation_smoke(
                qids=self.query_ids,
                policy=instr_policy,
                max_turns=get_stage_turn_budget("add_timings"),
                check_correctness_fn=self._check_correctness_with_scale_factors,
                exec_fn=self._exec,
            )

            trace_evidence_summary = await check_trace_evidence_and_feedback(
                qids=list(self.query_ids),
                policy=instr_policy,
                summarize_trace_fn=self._summarize_trace_evidence_for_queries,
                exec_fn=self._exec,
                max_turns=get_stage_turn_budget("add_timings"),
            )
            if trace_evidence_summary.degraded:
                self.hotspot_analysis_degraded = True
                self.hotspot_analysis_failure_reason = trace_evidence_summary.message
                logger.warning(
                    "Continuing optimization with degraded hotspot evidence: %s",
                    trace_evidence_summary.message,
                )

            await check_trace_mode_smoke(
                qids=list(self.query_ids),
                policy=instr_policy,
                max_turns=get_stage_turn_budget("add_timings"),
                check_correctness_fn=self._check_correctness_with_scale_factors,
                exec_fn=self._exec,
            )
        except Exception as exc:
            detail = str(exc)
            failure_code = self._classify_failure_text(detail) or "INSTRUMENTATION_FAILED"
            self._persist_failure_summary(
                failure_code=failure_code,
                failure_detail=detail,
                final_correctness=False,
            )
            raise

        # Two-stage optimization: local trace_expert per query, then global candidates
        validation_policy = self._build_validation_policy()
        measurement_cache: dict[tuple[str, int, str, str], float] = {}
        records: list[QueryOptimizationRecord] = []
        unit_by_query = build_query_unit_lookup(self.query_ids)
        processed_units: set[str] = set()

        # Per-query session isolation: fork the session before per-query
        # optimization so each query's LLM turns are isolated and context does
        # not accumulate across all 15 queries. Without this, the accumulated
        # session items can produce HTTP bodies that exceed DeepSeek's edge-proxy
        # limit, causing TCP RST (Connection reset by peer).
        #
        # create_branch_from_turn() mutates session._current_branch_id as a
        # side effect, so every iteration must re-anchor to the same source
        # branch that get_conversation_turns() was queried from.
        source_branch = self.session._current_branch_id
        branch_turns = await self.session.get_conversation_turns()
        if not branch_turns:
            detail = (
                "No user turns available for per-query branching — "
                "compaction may have removed all user messages"
            )
            self._persist_failure_summary(
                failure_code="NO_BRANCH_TURNS",
                failure_detail=detail,
                final_correctness=False,
            )
            raise RuntimeError(detail)
        branch_turn = branch_turns[-1]["turn"]
        per_unit_branch: dict[str, str] = {}
        for query_id in self.query_ids:
            unit = unit_by_query[query_id]
            if unit.unit_id in per_unit_branch:
                continue
            await self.session.switch_to_branch(source_branch)
            branch_name = f"unit_{_safe_branch_label(unit.unit_id)}_{branch_turn}"
            per_unit_branch[unit.unit_id] = await self.session.create_branch_from_turn(
                branch_turn, branch_name=branch_name
            )
            logger.info(
                "Created session branch '%s' for unit %s at turn %s",
                branch_name, unit.unit_id, branch_turn,
            )

        for query_id in self.query_ids:
            active_unit = unit_by_query[query_id]
            if active_unit.unit_id in processed_units:
                continue
            processed_units.add(active_unit.unit_id)
            try:
                # Switch to isolated per-query session branch
                await self.session.switch_to_branch(per_unit_branch[active_unit.unit_id])
                logger.debug(
                    "Switched to branch '%s' for unit %s",
                    per_unit_branch[active_unit.unit_id],
                    active_unit.unit_id,
                )

                self.run_tool.reset_runtime_state(clean_reload=True)
                snapshot_hash = self.git_snapshotter.current_hash
                if snapshot_hash is None:
                    raise RuntimeError("Current git snapshot is None.")
                scope_query_ids = list(active_unit.query_ids)
                before_runtime_by_query: dict[str, float] = {}
                try:
                    for scope_query_id in scope_query_ids:
                        cache_key = (
                            scope_query_id,
                            self.benchmark_sf,
                            snapshot_hash,
                            QUERY_OUTPUT_MODE_NO_OUTPUT,
                        )
                        if cache_key in measurement_cache:
                            before_runtime_by_query[scope_query_id] = measurement_cache[cache_key]
                            continue
                        scope_rt_s, _, _, _ = self._measure_with_manifest(
                            query_id=scope_query_id,
                            exec_callback=self._make_exec_callback(
                                scope_query_id,
                                output_mode=QUERY_OUTPUT_MODE_NO_OUTPUT,
                            ),
                            output_mode=QUERY_OUTPUT_MODE_NO_OUTPUT,
                        )
                        measurement_cache[cache_key] = scope_rt_s
                        before_runtime_by_query[scope_query_id] = scope_rt_s
                except Exception as exc:
                    for scope_query_id in scope_query_ids:
                        records.append(QueryOptimizationRecord(
                            query_id=scope_query_id,
                            unit_id=active_unit.unit_id,
                            unit_query_ids=tuple(scope_query_ids),
                            issue_class="measurement_failed",
                            trace_summary="",
                            sampled_instantiations=(),
                            stage_name="trace_expert",
                            rt_before_s=float("inf"),
                            rt_after_s=float("inf"),
                            written_files=(),
                            failed=True,
                            failure_code="MEASUREMENT_FAILED",
                            failure_detail=str(exc),
                        ))
                    logger.error(
                        "Pre-stage measurement failed for unit %s: %s",
                        active_unit.unit_id,
                        exc,
                    )
                    continue

                representative_query_id = max(
                    before_runtime_by_query,
                    key=lambda qid: before_runtime_by_query[qid],
                )
                impl_rt_s = aggregate_scope_runtime_seconds(before_runtime_by_query)
                try:
                    (
                        trace_summaries_by_query,
                        scope_trace_summary,
                        scope_issue_class,
                        scope_hardware_counter_evidence,
                    ) = self._collect_scope_hotspot_analysis(scope_query_ids)
                except Exception as exc:
                    self.hotspot_analysis_degraded = True
                    self.hotspot_analysis_failure_reason = str(exc)
                    logger.warning(
                        "Trace sampling degraded for unit %s; continuing in best-effort mode: %s",
                        active_unit.unit_id,
                        exc,
                    )
                    trace_summaries_by_query = {
                        scope_query_id: TraceHotspotSummary(
                            query_id=scope_query_id,
                            issue_class="evidence_insufficient",
                            evidence_sufficient=False,
                            top_profiles=(),
                            counters={},
                            summary_text=(
                                f"Query {scope_query_id}: hotspot analysis degraded; "
                                f"continuing without strong trace evidence. Reason: {exc}"
                            ),
                            sampled_instantiations=(),
                            sampled_count=0,
                            omitted_count=0,
                            vectorization_candidate=False,
                            hardware_counter_summary={},
                            compiler_vectorization_summary={},
                            change_scope="query",
                        )
                        for scope_query_id in scope_query_ids
                    }
                    ordered_summaries = [
                        trace_summaries_by_query[scope_query_id]
                        for scope_query_id in scope_query_ids
                    ]
                    scope_trace_summary = _format_scope_trace_summary(
                        ordered_summaries
                    )
                    scope_issue_class = _merge_scope_issue_class(
                        ordered_summaries
                    )
                    scope_hardware_counter_evidence = ""
                stage = self._build_query_stage(
                    query_id=representative_query_id,
                    mandatory_constraints=mandatory_constraints,
                    trace_summary=scope_trace_summary,
                    hardware_counter_evidence=scope_hardware_counter_evidence,
                    active_unit_id=active_unit.unit_id,
                    active_unit_query_ids=tuple(scope_query_ids),
                )
                stage_result = await self._run_stage(
                    query_id=representative_query_id,
                    stage=stage,
                    pretext_optim=pretext_optim,
                    rt_before_s=impl_rt_s,
                    correctness_scale_factors=validation_policy.light_scale_factors,
                    scope_query_ids=tuple(scope_query_ids),
                    measurement_output_mode=QUERY_OUTPUT_MODE_NO_OUTPUT,
                    prompt_metadata={
                        "active_query_ids": scope_query_ids,
                        "active_unit_id": active_unit.unit_id,
                        "active_unit_kind": active_unit.unit_kind,
                        "active_unit_files": list(
                            dict.fromkeys(
                                (
                                    *active_unit.entrypoint_files,
                                    *active_unit.kernel_files,
                                    *active_unit.shared_helper_files,
                                )
                            )
                        ),
                        "active_unit_query_ids": scope_query_ids,
                        "objective_ids": self._objective_ids_for_prompt(),
                        "data_law_ids": self._data_law_ids_for_prompt(),
                        "patch_scope_verdict": scope_issue_class,
                    },
                )
                written_files = (
                    () if stage_result is None else stage_result.written_files
                )
                local_gate = self._run_stage_correctness_gate(
                    query_id=representative_query_id,
                    written_files=written_files,
                    policy=validation_policy,
                    scope_query_ids=tuple(scope_query_ids),
                )
                if not local_gate.success:
                    raise RuntimeError(local_gate.message)
                for scope_query_id in scope_query_ids:
                    scope_summary = trace_summaries_by_query[scope_query_id]
                    self._collect_compiler_vectorization_summary(
                        scope_query_id,
                        force_refresh=True,
                        trace_summary_text=scope_summary.summary_text,
                    )
                after_runtime_by_query = (
                    before_runtime_by_query
                    if stage_result is None or not stage_result.runtime_by_query
                    else stage_result.runtime_by_query
                )
                for scope_query_id in scope_query_ids:
                    scope_summary = trace_summaries_by_query[scope_query_id]
                    records.append(QueryOptimizationRecord(
                        query_id=scope_query_id,
                        unit_id=active_unit.unit_id,
                        unit_query_ids=tuple(scope_query_ids),
                        issue_class=scope_summary.issue_class,
                        trace_summary=scope_summary.summary_text,
                        sampled_instantiations=scope_summary.sampled_instantiations,
                        stage_name=stage.name,
                        rt_before_s=before_runtime_by_query.get(scope_query_id, float("inf")),
                        rt_after_s=after_runtime_by_query.get(scope_query_id, float("inf")),
                        written_files=written_files,
                        failed=False if stage_result is None else stage_result.failed,
                        failure_detail=(
                            None if stage_result is None
                            else stage_result.failure_message
                        ),
                    ))
            finally:
                self._delete_result_csvs(self.run_tool.cwd)
                self.run_tool.reset_runtime_state(clean_reload=True)
                # Return to main branch after per-query optimization
                await self.session.switch_to_branch("main")

        try:
            self.output_split_by_query = self._collect_query_output_split_measurements(
                list(self.query_ids)
            )
        except Exception as exc:
            self.output_split_by_query = {}
            self.hotspot_analysis_degraded = True
            self.hotspot_analysis_failure_reason = (
                f"Output split measurement failed: {exc}"
            )
            logger.warning(
                "Continuing without full_csv/no_output split evidence: %s",
                exc,
            )
        hotspot_summary_path = self._persist_hotspot_summary(records)
        final_runtime_ms_by_query: dict[str, float] = {}
        best_runtime_ms_by_query: dict[str, float] = {}
        baseline_runtime_ms_by_query: dict[str, float] = {}
        global_result = GlobalHumanReferenceResult(
            runtime_by_query={},
            written_files=(),
            accepted=True,
        )
        failure_state: OptimizationFailureState | None = None

        try:
            local_phase_validation = run_required_correctness_checks(
                self.run_tool,
                validation_policy.full_scale_factors,
                self.query_ids,
                trace_mode=False,
                optimize=True,
                external_call=True,
            )
        except Exception as exc:
            local_phase_validation = None
            stage_end_metrics = {
                "validation/correct": False,
                "validation/error": True,
                "validation/failure_code": "LOCAL_PHASE_VALIDATION_FAILED",
                "validation/failure_detail": str(exc),
            }
            stage_end_msg = f"Local phase validation failed: {exc}"
            failure_state = OptimizationFailureState(
                failure_code="LOCAL_PHASE_VALIDATION_FAILED",
                failure_detail=str(exc),
                final_correctness=False,
            )
        else:
            stage_end_metrics = local_phase_validation.metrics or {
                "validation/correct": local_phase_validation.success,
                "validation/message": local_phase_validation.message,
            }
            stage_end_msg = local_phase_validation.message
            if not local_phase_validation.success:
                failure_state = OptimizationFailureState(
                    failure_code=local_phase_validation.failure_code
                    or "LOCAL_PHASE_VALIDATION_FAILED",
                    failure_detail=local_phase_validation.failure_detail
                    or local_phase_validation.message,
                    final_correctness=False,
                )
        if failure_state is None:
            self.query_rt_log = self._measure_all_queries()
            try:
                global_result = await self._run_global_human_reference(
                    mandatory_constraints=mandatory_constraints,
                    hotspot_summary_path=hotspot_summary_path,
                    before_rt_log=dict(self.query_rt_log),
                )
            except Exception as exc:
                stage_end_msg = f"Global human-reference failed: {exc}"
                stage_end_metrics = {
                    **stage_end_metrics,
                    "validation/correct": False,
                    "validation/error": True,
                    "validation/failure_code": "GLOBAL_HUMAN_REFERENCE_FAILED",
                    "validation/failure_detail": str(exc),
                }
                failure_state = OptimizationFailureState(
                    failure_code="GLOBAL_HUMAN_REFERENCE_FAILED",
                    failure_detail=str(exc),
                    final_correctness=False,
                )
            else:
                self.query_rt_log = global_result.runtime_by_query
                for query_id, runtime_s in global_result.runtime_by_query.items():
                    if (
                        math.isfinite(runtime_s)
                        and runtime_s > 0
                        and (
                            query_id not in self.best_rt_log
                            or runtime_s < self.best_rt_log[query_id]
                        )
                    ):
                        self.best_rt_log[query_id] = runtime_s
                if any(not attempt.accepted for attempt in global_result.attempts):
                    self._record_global_regression(global_result)
                if not global_result.accepted:
                    logger.info(
                        "Global human-reference made no accepted candidate changes; "
                        "continuing with local optimization result."
                    )
                elif not global_result.written_files:
                    logger.info(
                        "Global human-reference made no accepted candidate changes."
                    )

        logger.info(f"Final validation metrics: {stage_end_msg}")

        # P6: 切回 main session branch，确保 finish_and_save 在主干上执行
        await self.session.switch_to_branch("main")
        logger.info("Switched back to main session branch after all query optimizations.")

        # Outer-loop checkpoint: final summary — baselines already cached, no-op if complete.
        self._collect_baselines_at_checkpoint()
        self._log_ingest_comparison_if_complete(
            stage_name="final_summary",
            validation_metrics=stage_end_metrics,
        )

        # 收集 optimization summary 数据；W&B 仅作为可选 sink，不应决定本地 summary 是否落盘
        final_correctness = (
            failure_state.final_correctness
            if failure_state is not None
            else stage_end_metrics.get("validation/correct", False)
        )
        final_snapshot = self.git_snapshotter.current_hash or ""
        final_measurement_failed = False
        forced_objective_failures: tuple[str, ...] = ()
        if failure_state is None:
            try:
                self._reject_invalid_final_paths()
            except Exception as exc:
                stage_end_msg = f"Final path gate failed before measurement: {exc}"
                stage_end_metrics = {
                    **stage_end_metrics,
                    "validation/error": True,
                    "optimization/final_path_gate_failed": True,
                    "validation/failure_code": "FORBIDDEN_INSTRUMENTED_FINAL_PATH",
                    "validation/failure_detail": str(exc),
                }
                forced_objective_failures = ("FORBIDDEN_INSTRUMENTED_FINAL_PATH",)
                final_measurement_failed = True
                failure_state = OptimizationFailureState(
                    failure_code="FORBIDDEN_INSTRUMENTED_FINAL_PATH",
                    failure_detail=str(exc),
                    final_correctness=final_correctness,
                )
            for query_id in self.query_ids:
                if failure_state is not None:
                    break
                try:
                    final_rt_s, baseline_rt_s, _, final_lazy = self._measure_with_manifest(
                        query_id=query_id,
                        exec_callback=self._make_exec_callback(
                            query_id,
                            output_mode=QUERY_OUTPUT_MODE_NO_OUTPUT,
                        ),
                        output_mode=QUERY_OUTPUT_MODE_NO_OUTPUT,
                    )
                    if final_lazy:
                        logger.warning(
                            "Final measurement for %s: lazy-build suspected, "
                            "not recording as official result.",
                            query_id,
                        )
                        final_rt_s = float("inf")
                except Exception as exc:
                    stage_end_msg = f"Final measurement failed for {query_id}: {exc}"
                    logger.error(
                        stage_end_msg,
                        exc_info=exc,
                    )
                    stage_end_metrics = {
                        **stage_end_metrics,
                        "validation/correct": False,
                        "validation/error": True,
                        "optimization/final_measurement_failed": True,
                    }
                    final_correctness = False
                    final_measurement_failed = True
                    failure_state = OptimizationFailureState(
                        failure_code="FINAL_MEASUREMENT_FAILED",
                        failure_detail=stage_end_msg,
                        final_correctness=False,
                    )
                    break
                final_rt_ms = final_rt_s * 1000
                baseline_rt_ms = baseline_rt_s * 1000
                final_runtime_ms_by_query[query_id] = final_rt_ms
                if baseline_rt_s > 0:
                    baseline_runtime_ms_by_query[query_id] = baseline_rt_ms

                best_rt_s = self.best_rt_log.get(query_id, final_rt_s)
                best_speedup = safe_speedup(baseline_rt_s * 1000.0, best_rt_s * 1000.0) or 0.0
                best_runtime_ms_by_query[query_id] = best_rt_s * 1000

                if self.wandb_run_hook is not None:
                    self.wandb_run_hook.log_optimization_final_summary(
                        query_id=query_id,
                        baseline_runtime_ms=baseline_rt_ms,
                        final_no_csv_kernel_runtime_ms=final_rt_ms,
                        best_no_csv_kernel_speedup_vs_baseline=best_speedup,
                        final_correctness=final_correctness,
                        final_snapshot=final_snapshot,
                        baseline_engine=self._baseline_engine_name(),
                        baseline_label=self._baseline_display_name(),
                    )
            if failure_state is None:
                resolved_large_sf = getattr(self, "large_sf", None)
                large_sf = resolved_large_sf or self.benchmark_sf
                repetitions = 3 if resolved_large_sf is not None else 1
                try:
                    self.measurement_repetition = self._collect_repeated_scope_measurements(
                        list(self.query_ids),
                        scale_factor=large_sf,
                        repetitions=repetitions,
                    )
                    if repetitions > 1 and self.measurement_repetition["lazy_build_detected"]:
                        raise RuntimeError(
                            "Large-SF repeated measurements detected lazy-build behavior"
                        )
                except Exception as exc:
                    stage_end_msg = f"Repeated measurement failed: {exc}"
                    logger.error(stage_end_msg, exc_info=exc)
                    stage_end_metrics = {
                        **stage_end_metrics,
                        "validation/correct": False,
                        "validation/error": True,
                        "optimization/final_measurement_failed": True,
                    }
                    final_correctness = False
                    final_measurement_failed = True
                    failure_state = OptimizationFailureState(
                        failure_code="FINAL_MEASUREMENT_FAILED",
                        failure_detail=stage_end_msg,
                        final_correctness=False,
                    )
        else:
            logger.error(
                "Skipping final measurement after optimization failure: %s",
                failure_state.failure_detail,
            )

        # Emit the official final-validator kernel report to log.
        if not final_measurement_failed:
            from tpch_monetdb.utils.optimization_summary import (
                build_validation_kernel_report,
                render_validation_kernel_report,
            )

            validation_report = build_validation_kernel_report(
                stage_end_metrics,
                self.query_ids,
                self.conv_name or "",
            )
            logger.info(
                "\n"
                + render_validation_kernel_report(validation_report)
            )

        # Write optimization summary (always, even on failure)
        if self.conv_name and self.artifacts_dir:
            from pathlib import Path as _Path
            from tpch_monetdb.utils.optimization_summary import persist_optimization_run as _persist
            _stage_records = [
                {
                    "query_id": rec.query_id,
                    "unit_id": rec.unit_id,
                    "unit_query_ids": list(rec.unit_query_ids),
                    "issue_class": rec.issue_class,
                    "stage_name": rec.stage_name,
                    "rt_before_s": rec.rt_before_s,
                    "rt_after_s": rec.rt_after_s,
                    "failed": rec.failed,
                    "failure_code": rec.failure_code,
                    "failure_detail": rec.failure_detail,
                }
                for rec in records
            ]
            records_by_unit = {
                rec.unit_id: [item for item in records if item.unit_id == rec.unit_id]
                for rec in records
                if rec.unit_id is not None
            }
            optimization_units = [
                {
                    "unit_id": unit_id,
                    "query_ids": list(records_for_unit[0].unit_query_ids),
                    "issue_class": records_for_unit[0].issue_class,
                }
                for unit_id, records_for_unit in records_by_unit.items()
            ]
            unit_scores: dict[str, float] = {}
            for unit_id, records_for_unit in records_by_unit.items():
                before_map = {
                    item.query_id: item.rt_before_s
                    for item in records_for_unit
                    if math.isfinite(item.rt_before_s) and item.rt_before_s > 0
                }
                after_map = {
                    item.query_id: item.rt_after_s
                    for item in records_for_unit
                    if math.isfinite(item.rt_after_s) and item.rt_after_s > 0
                }
                if not before_map or not after_map:
                    continue
                unit_scores[unit_id] = safe_speedup(
                    aggregate_scope_runtime_seconds(before_map) * 1000.0,
                    aggregate_scope_runtime_seconds(after_map) * 1000.0,
                ) or 0.0
            control_artifact_hashes = collect_control_artifact_hashes(self.run_tool.cwd)
            todo_reconciliation = build_todo_reconciliation(
                self.run_tool.cwd / "TODO.md"
            )
            storage_plan_alignment = build_storage_plan_alignment(
                self.run_tool.cwd / "storage_plan.txt"
            )
            workload_objective = load_json_contract(
                self.run_tool.cwd,
                WORKLOAD_OBJECTIVE_FILE,
            )
            self._refresh_final_vectorization_summaries(records)
            objective_preview = type("_ObjectivePreview", (), {})()
            objective_preview.workload_objective = workload_objective
            objective_preview.final_runtime_ms_by_query = final_runtime_ms_by_query
            objective_preview.baseline_runtime_ms_by_query = baseline_runtime_ms_by_query
            objective_preview.storage_plan_alignment = storage_plan_alignment
            objective_preview.control_artifact_hashes = control_artifact_hashes
            objective_preview.measurement_repetition = getattr(self, "measurement_repetition", {})
            objective_preview.hardware_counter_summary = getattr(self, "hardware_counter_summary_by_query", {})
            objective_preview.compiler_vectorization_summary = getattr(self, "compiler_vectorization_summary", {})
            ledger = build_pipeline_evidence_ledger(
                workspace_path=self.run_tool.cwd,
                workload_objective=workload_objective,
                final_runtime_ms_by_query=final_runtime_ms_by_query,
                baseline_runtime_ms_by_query=baseline_runtime_ms_by_query,
                compiler_vectorization_summary=getattr(
                    self, "compiler_vectorization_summary", {}
                ),
                hardware_counter_summary=getattr(
                    self, "hardware_counter_summary_by_query", {}
                ),
                todo_reconciliation=todo_reconciliation,
                measurement_records=tuple(
                    QueryMeasurementRecord(**record)
                    for record in getattr(self, "measurement_records", [])
                ),
            )
            objective_report = build_objective_failure_report(objective_preview)
            objective_failures = tuple(
                dict.fromkeys((
                    *forced_objective_failures,
                    *objective_report.failures,
                    *ledger.failures,
                ))
            )
            objective_failure_route = (
                ledger.failure_route
                or (
                    classify_objective_failure_route(objective_failures)
                    if objective_failures else None
                )
            )
            _success = (
                failure_state is None
                and not final_measurement_failed
                and bool(final_runtime_ms_by_query)
                and not objective_failures
            )
            failure_code = None
            failure_detail = None
            if not _success:
                if failure_state is not None:
                    failure_code = failure_state.failure_code
                    failure_detail = failure_state.failure_detail
                elif objective_failures:
                    failure_code = objective_failures[0]
                    failure_detail = (
                        "Objective failures: "
                        + ", ".join(objective_failures)
                    )
                else:
                    failure_code = stage_end_metrics.get(
                        "validation/failure_code", "OPTIMIZATION_FAILED"
                    )
                    failure_detail = stage_end_metrics.get(
                        "validation/failure_detail", stage_end_msg
                    )
            _persist(
                benchmark=getattr(self, "benchmark", "tpch"),
                conv_name=self.conv_name,
                query_list=self.query_ids,
                is_bespoke_storage=self.bespoke_storage,
                start_snapshot_hash=self.start_snapshot_hash,
                final_snapshot_hash=self.git_snapshotter.current_hash or "",
                best_runtime_ms_by_query=best_runtime_ms_by_query if best_runtime_ms_by_query else None,
                final_runtime_ms_by_query=final_runtime_ms_by_query if final_runtime_ms_by_query else None,
                final_correctness=final_correctness,
                artifacts_dir=_Path(self.artifacts_dir),
                baseline_runtime_ms_by_query=baseline_runtime_ms_by_query if baseline_runtime_ms_by_query else None,
                success=_success,
                failure_code=failure_code,
                failure_detail=failure_detail,
                hotspot_summary_path=hotspot_summary_path,
                stage_records=_stage_records,
                optimization_units=optimization_units,
                unit_scores=unit_scores,
                control_artifact_hashes=control_artifact_hashes,
                storage_plan_alignment=storage_plan_alignment,
                todo_reconciliation=todo_reconciliation,
                change_scope="family" if any(len(rec.unit_query_ids) > 1 for rec in records) else "query",
                stage_history=_stage_records,
                measurement_repetition=getattr(self, "measurement_repetition", {}),
                hardware_counter_summary=getattr(self, "hardware_counter_summary_by_query", {}),
                compiler_vectorization_summary=getattr(self, "compiler_vectorization_summary", {}),
                workload_objective=workload_objective,
                objective_failures=list(objective_failures),
                objective_failure_route=objective_failure_route,
                measurement_records=getattr(self, "measurement_records", []),
                final_validation_metrics=stage_end_metrics,
                pipeline_evidence_ledger=ledger.to_dict(),
                target_cpu=getattr(self, "target_cpu", None),
                hotspot_analysis_degraded=getattr(self, "hotspot_analysis_degraded", False),
                hotspot_analysis_failure_reason=getattr(
                    self, "hotspot_analysis_failure_reason", None
                ),
                global_regression_records=self.global_regression_records if self.global_regression_records else None,
                global_optimization_candidates=[
                    global_candidate_to_dict(candidate)
                    for candidate in global_result.candidates
                ] or None,
                global_optimization_winner=(
                    global_candidate_to_dict(global_result.winner)
                    if global_result.winner is not None
                    else None
                ),
                completed_stage_summaries=[
                    {
                        "profile_name": summary.profile_name,
                        "prompt_index": summary.prompt_index,
                        "prompt_descriptor": summary.prompt_descriptor,
                        "written_files": list(summary.written_files),
                        "validation_passed": summary.validation_passed,
                        "control_artifacts_read": list(summary.control_artifacts_read),
                        "active_unit_id": summary.active_unit_id,
                        "active_unit_kind": summary.active_unit_kind,
                        "active_unit_query_ids": list(summary.active_unit_query_ids),
                        "objective_ids": list(summary.objective_ids),
                        "data_law_ids": list(summary.data_law_ids),
                        "patch_scope_verdict": summary.patch_scope_verdict,
                    }
                    for summary in getattr(self, "completed_stage_summaries", [])
                ],
            )

        # Finish
        used = await self.ask_to_finish_and_save()
        return used

    def _reject_invalid_final_paths(self) -> None:
        """Reject instrumented replacement entrypoints before final measurement."""
        workload_objective = load_json_contract(self.run_tool.cwd, WORKLOAD_OBJECTIVE_FILE)
        ledger = build_pipeline_evidence_ledger(
            workspace_path=self.run_tool.cwd,
            workload_objective=workload_objective,
            compiler_vectorization_summary=getattr(
                self, "compiler_vectorization_summary", {}
            ),
            hardware_counter_summary=getattr(
                self, "hardware_counter_summary_by_query", {}
            ),
        )
        invalid_query_ids = [
            evidence.query_id
            for evidence in ledger.query_evidence.values()
            if evidence.final_path_status is EvidenceStatus.FAIL
        ]
        if invalid_query_ids:
            raise RuntimeError(
                "FORBIDDEN_INSTRUMENTED_FINAL_PATH for queries: "
                + ", ".join(invalid_query_ids)
            )
        return None

    async def _exec(
        self,
        prompt: str,
        prompt_descriptor: Optional[str],
        max_turns: Optional[int] = None,
        tool_profile: Optional[str] = None,
        rule_area: Optional[str] = "provider",
        prompt_metadata: Optional[dict[str, Any]] = None,
    ) -> Optional[StageRunSummary]:
        """Execute a prompt."""
        control_prompts = {
            COMPACTION_MARKER,
            VALIDATE_ON,
            VALIDATE_OUTPUT_STDOUT_OFF,
            VALIDATE_OUTPUT_STDOUT_ON,
        }
        if (
            prompt not in control_prompts
            and prompt_descriptor not in (None, "compaction")
            and max_turns is None
        ):
            raise RuntimeError(
                f"Optimization prompt '{prompt_descriptor}' is missing explicit max_turns. "
                "Do not fall back to the default conversation turn limit."
            )
        runtime_metadata = {} if prompt_metadata is None else dict(prompt_metadata)
        if tool_profile is not None:
            runtime_metadata["tool_profile"] = tool_profile
        if rule_area is not None:
            runtime_metadata["rule_area"] = rule_area
        user_choice, executed_prompt, last_outcome = await self.process_prompt(
            prompt,
            prompt_descriptor,
            max_turns,
            prompt_metadata=runtime_metadata or None,
        )
        if user_choice in ["u", "r"]:
            if last_outcome is None:
                return None
            if not isinstance(last_outcome, StageRunSummary):
                raise RuntimeError(
                    f"Prompt '{prompt_descriptor or 'prompt'}' returned unexpected callback type: "
                    f"{type(last_outcome).__name__}"
                )
            violations = check_agent_diff_boundary(last_outcome.written_files)
            if violations:
                raise RuntimeError(
                    "Agent diff touched protected baseline-owned/host-owned paths: "
                    f"{', '.join(violations)}"
                )
            query_local_violations = self._query_local_scope_violations(
                tool_profile,
                last_outcome.written_files,
            )
            if query_local_violations:
                raise RuntimeError(
                    "Query-local optimization touched shared runtime layout files: "
                    f"{', '.join(query_local_violations)}. "
                    "Use optimization_infra_layout for explicit global layout changes."
                )
            if not hasattr(self, "completed_stage_summaries"):
                self.completed_stage_summaries = []
            self.completed_stage_summaries.append(last_outcome)
            return last_outcome
        raise Exception(f"Unexpected user choice: {user_choice}")

    def _query_local_scope_violations(
        self,
        tool_profile: str | None,
        written_files: tuple[str, ...],
    ) -> tuple[str, ...]:
        """Return shared layout files written from a query-local optimization stage."""
        if tool_profile != "optimization_general":
            return ()
        core_files = set(CORE_IMPLEMENTATION_FILES)
        violations = [
            path
            for path in written_files
            if Path(path).name in core_files
        ]
        return tuple(sorted(violations))

    async def _check_correctness(self, qids: List[str], trace_mode: bool) -> bool:
        """Check if implementation is correct."""
        validation_summary = run_required_correctness_checks(
            self.run_tool,
            self.required_validation_sf_list,
            qids,
            trace_mode=trace_mode,
            optimize=True,
            external_call=True,
        )
        if not validation_summary.success:
            logger.error(
                f"Validation failed for qids={qids}, trace_mode={trace_mode}: "
                f"{validation_summary.message}"
            )
            return False
        return True

    def _delete_result_csvs(self, workspace_path: Path) -> None:
        """Delete result CSV files."""
        csv_files = list(workspace_path.rglob("result*.csv"))
        logger.info(f"Deleting {len(csv_files)} result CSV files")
        for csv_file in csv_files:
            csv_file.unlink(missing_ok=True)

    async def _check_correctness_with_scale_factors(
        self,
        qids: list[str],
        trace_mode: bool,
        scale_factors: tuple[int, ...],
    ) -> bool:
        validation_summary = run_required_correctness_checks(
            self.run_tool,
            scale_factors,
            qids,
            trace_mode=trace_mode,
            optimize=True,
            external_call=True,
        )
        if not validation_summary.success:
            logger.error(
                "Validation failed for qids=%s, trace_mode=%s, scale_factors=%s: %s",
                qids, trace_mode, scale_factors, validation_summary.message,
            )
            return False
        return True

    def _select_trace_instantiations(
        self,
        query_id: str,
        max_samples: int = 3,
    ) -> tuple[Any, ...]:
        instantiations = self.manifest.get_instantiations_for_query(
            query_id=query_id,
            scale_factor=self.benchmark_sf,
        )
        if not instantiations:
            raise RuntimeError(f"No trace instantiation found for query {query_id}")
        unique_by_args: dict[str, Any] = {}
        for inst in instantiations:
            unique_by_args.setdefault(inst.args_string, inst)
        unique = tuple(unique_by_args.values())
        if len(unique) <= max_samples:
            return unique
        middle_index = len(unique) // 2
        selected = [unique[0], unique[middle_index], unique[-1]]
        deduped: dict[str, Any] = {inst.args_string: inst for inst in selected}
        return tuple(deduped.values())

    def _sample_trace_for_query(self, query_id: str) -> TraceHotspotSummary:
        selected = self._select_trace_instantiations(query_id)
        hardware_counter_summary = self._collect_hardware_counter_summary(
            query_id,
            args_string=selected[0].args_string,
        )
        compiler_vectorization_summary = self._collect_compiler_vectorization_summary(
            query_id
        )
        all_instantiations = self.manifest.get_instantiations_for_query(
            query_id=query_id,
            scale_factor=self.benchmark_sf,
        )
        omitted_count = max(
            0, len({inst.args_string for inst in all_instantiations}) - len(selected)
        )
        trace_path = self.run_tool.cwd / "tracing_output.log"
        summaries: list[TraceHotspotSummary] = []
        for inst in selected:
            trace_path.unlink(missing_ok=True)
            timeout_policy = build_runtime_timeout_policy(self.benchmark_sf, num_queries=1)
            run_result = self.run_tool.run_raw_worker(
                scale_factor=self.benchmark_sf,
                optimize=True,
                query_id=[query_id],
                trace_mode=True,
                external_call=True,
                stdin_args_data=[inst.args_string],
                output_mode=QUERY_OUTPUT_MODE_NO_OUTPUT,
                execution_timeout_s=timeout_policy.trace_timeout_s,
            )
            issue = self._classify_raw_trace_execution(
                query_id=query_id,
                run_result=run_result,
            )
            if issue is not None:
                raise RuntimeError(issue.failure_detail)
            summaries.append(
                summarize_trace_file(
                    query_id=query_id,
                    trace_path=trace_path,
                    instantiation_id=inst.instantiation_id,
                    args_string=inst.args_string,
                    hardware_counter_summary=hardware_counter_summary,
                    compiler_vectorization_summary=compiler_vectorization_summary,
                )
            )
        return merge_trace_summaries(
            query_id=query_id,
            summaries=summaries,
            omitted_count=omitted_count,
        )

    def _collect_scope_hotspot_analysis(
        self,
        scope_query_ids: list[str],
    ) -> tuple[dict[str, TraceHotspotSummary], str, str, str]:
        """Collect trace/PMU/vectorization evidence for every query in one unit."""
        summaries_by_query: dict[str, TraceHotspotSummary] = {}
        for scope_query_id in scope_query_ids:
            summaries_by_query[scope_query_id] = self._sample_trace_for_query(
                scope_query_id
            )
        ordered_summaries = [
            summaries_by_query[scope_query_id]
            for scope_query_id in scope_query_ids
            if scope_query_id in summaries_by_query
        ]
        scope_trace_summary = _format_scope_trace_summary(ordered_summaries)
        scope_issue_class = _merge_scope_issue_class(ordered_summaries)
        hardware_counter_evidence = _format_scope_hardware_counter_evidence(
            ordered_summaries
        )
        return (
            summaries_by_query,
            scope_trace_summary,
            scope_issue_class,
            hardware_counter_evidence,
        )

    def _summarize_trace_evidence_for_queries(
        self,
        qids: list[str],
    ) -> TraceEvidenceSummary:
        """Collect one raw trace sample per query and summarize PROFILE coverage."""
        insufficient: list[str] = []
        messages: list[str] = []
        raw_execution_ok = True
        trace_file_present = True
        failure_code: str | None = None
        profile_count_by_query: dict[str, int] = {}
        trace_path = self.run_tool.cwd / "tracing_output.log"
        for qid in qids:
            selected = self._select_trace_instantiations(qid, max_samples=1)
            inst = selected[0]
            trace_path.unlink(missing_ok=True)
            try:
                timeout_policy = build_runtime_timeout_policy(
                    self.benchmark_sf,
                    num_queries=1,
                )
                run_result = self.run_tool.run_raw_worker(
                    scale_factor=self.benchmark_sf,
                    optimize=True,
                    query_id=[qid],
                    trace_mode=True,
                    external_call=True,
                    stdin_args_data=[inst.args_string],
                    output_mode=QUERY_OUTPUT_MODE_NO_OUTPUT,
                    execution_timeout_s=timeout_policy.trace_timeout_s,
                )
            except Exception as exc:
                raw_execution_ok = False
                text = str(exc)
                failure_code = self._classify_failure_text(text) or "TRACE_RAW_EXECUTION_FAILED"
                insufficient.append(qid)
                messages.append(f"Query {qid}: raw trace execution raised: {text}")
                continue
            issue = self._classify_raw_trace_execution(
                query_id=qid,
                run_result=run_result,
            )
            if issue is not None:
                raw_execution_ok = False
                failure_code = issue.failure_code
                insufficient.append(qid)
                messages.append(issue.failure_detail)
                continue
            summary = summarize_trace_file(
                query_id=qid,
                trace_path=trace_path,
            )
            profile_count_by_query[qid] = len(summary.top_profiles)
            if not trace_path.exists():
                trace_file_present = False
            if not summary.evidence_sufficient:
                insufficient.append(qid)
                messages.append(summary.summary_text)
        if insufficient:
            return TraceEvidenceSummary(
                qids=tuple(qids),
                sufficient=False,
                message="\n\n".join(messages),
                insufficient_qids=tuple(insufficient),
                failure_code=failure_code,
                raw_execution_ok=raw_execution_ok,
                trace_file_present=trace_file_present,
                profile_count_by_query=profile_count_by_query,
            )
        return TraceEvidenceSummary(
            qids=tuple(qids),
            sufficient=True,
            message="Trace evidence sufficient for all requested queries.",
            insufficient_qids=(),
            failure_code=None,
            raw_execution_ok=True,
            trace_file_present=True,
            profile_count_by_query=profile_count_by_query,
        )

    def _collect_hardware_counter_summary(
        self,
        query_id: str,
        args_string: str | None = None,
    ) -> dict[str, Any]:
        """Collect real hardware-counter evidence when configured, otherwise return provenance."""
        summary_by_query = getattr(self, "hardware_counter_summary_by_query", {})
        cached_summary = summary_by_query.get(query_id)
        cached_args_string = None
        if cached_summary:
            cached_args_string = (
                cached_summary.get("provenance", {}) or {}
            ).get("args_string")
        if (
            cached_summary
            and cached_summary.get("hardware_counters_available") is True
            and (args_string is None or cached_args_string == args_string)
        ):
            return cached_summary
        preflight = getattr(self, "hardware_counter_preflight", None)
        if preflight is None:
            return {} if cached_summary is None else cached_summary
        summary = {
            "backend": preflight.backend,
            "target_cpu": preflight.target_cpu,
            "hardware_counters_available": False,
            "required_events": list(preflight.required_events),
            "counters": {},
            "derived_metrics": {},
            "perf_hotspots_available": False,
            "perf_top_symbols": [],
            "perf_top_frames": [],
            "perf_top_source_lines": [],
            "perf_sample_count": 0,
            "perf_raw_script_excerpt": [],
            "provenance": {
                "host_kernel": preflight.host_kernel,
                "perf_event_paranoid": preflight.perf_event_paranoid,
                "large_sf": preflight.large_sf,
            },
        }
        if args_string is not None:
            try:
                capture = self.run_tool.run_hardware_counter_capture(
                    scale_factor=getattr(self, "large_sf", None) or self.benchmark_sf,
                    optimize=True,
                    hardware_counter_preflight=preflight,
                    stdin_args_data=[args_string],
                    query_id=[query_id],
                )
            except Exception as exc:
                summary = {
                    **summary,
                    "hardware_counter_error": str(exc),
                    "provenance": {
                        **dict(summary["provenance"]),
                        "args_string": args_string,
                    },
                }
            else:
                summary = {
                    "backend": capture.backend,
                    "target_cpu": preflight.target_cpu,
                    "hardware_counters_available": True,
                    "required_events": list(preflight.required_events),
                    "counters": dict(capture.counters),
                    "derived_metrics": dict(capture.derived_metrics),
                    **self._capture_perf_hotspot_summary(
                        preflight=preflight,
                        query_id=query_id,
                        args_string=args_string,
                    ),
                    "provenance": {
                        **dict(capture.provenance),
                        "args_string": args_string,
                        "host_kernel": preflight.host_kernel,
                        "perf_event_paranoid": preflight.perf_event_paranoid,
                        "large_sf": preflight.large_sf,
                    },
                }
        summary_by_query[query_id] = summary
        self.hardware_counter_summary_by_query = summary_by_query
        return summary

    def _capture_perf_hotspot_summary(
        self,
        *,
        preflight: Any,
        query_id: str,
        args_string: str,
    ) -> dict[str, Any]:
        """Collect perf record/script hotspot evidence without discarding counters."""
        capture_fn = getattr(self.run_tool, "run_perf_hotspot_capture", None)
        if capture_fn is None:
            return {
                "perf_hotspots_available": False,
                "perf_top_symbols": [],
                "perf_top_frames": [],
                "perf_top_source_lines": [],
                "perf_sample_count": 0,
                "perf_raw_script_excerpt": [],
            }
        try:
            capture = capture_fn(
                scale_factor=getattr(self, "large_sf", None) or self.benchmark_sf,
                optimize=True,
                hardware_counter_preflight=preflight,
                stdin_args_data=[args_string],
                query_id=[query_id],
            )
        except Exception as exc:
            return {
                "perf_hotspots_available": False,
                "perf_top_symbols": [],
                "perf_top_frames": [],
                "perf_top_source_lines": [],
                "perf_sample_count": 0,
                "perf_raw_script_excerpt": [],
                "perf_hotspot_error": str(exc),
            }
        capture_provenance = dict(capture.provenance)
        if capture_provenance.get("capture_scope") != "query_loop_only":
            return {
                "perf_hotspots_available": False,
                "perf_top_symbols": list(capture.top_symbols),
                "perf_top_frames": list(capture.top_frames),
                "perf_top_source_lines": list(capture.top_source_lines),
                "perf_sample_count": capture.sample_count,
                "perf_raw_script_excerpt": list(capture.raw_script_excerpt),
                "perf_hotspot_error": "perf hotspot capture scope is not query_loop_only",
                "perf_hotspot_provenance": {
                    **capture_provenance,
                    "args_string": args_string,
                },
                "perf_data_path": capture.perf_data_path,
                "perf_script_path": capture.perf_script_path,
            }
        return {
            "perf_hotspots_available": bool(capture.top_symbols),
            "perf_top_symbols": list(capture.top_symbols),
            "perf_top_frames": list(capture.top_frames),
            "perf_top_source_lines": list(capture.top_source_lines),
            "perf_sample_count": capture.sample_count,
            "perf_raw_script_excerpt": list(capture.raw_script_excerpt),
            "perf_hotspot_provenance": {
                **capture_provenance,
                "args_string": args_string,
            },
            "perf_data_path": capture.perf_data_path,
            "perf_script_path": capture.perf_script_path,
        }

    def _collect_compiler_vectorization_summary(
        self,
        query_id: str,
        *,
        force_refresh: bool = False,
        trace_summary_text: str | None = None,
    ) -> dict[str, Any]:
        """Collect compiler vectorization report metadata for one query."""
        compiler_summary_by_query = getattr(self, "compiler_vectorization_summary", {})
        if not force_refresh and query_id in compiler_summary_by_query:
            return compiler_summary_by_query[query_id]
        from tpch_monetdb.misc.tpch.compiler import parse_vectorization_reports

        summary = parse_vectorization_reports(
            optimized_report_path=self.run_tool.cwd / "build" / "vectorization.optimized.txt",
            missed_report_path=self.run_tool.cwd / "build" / "vectorization.missed.txt",
            target_cpu=getattr(self, "target_cpu", None),
        )
        if trace_summary_text:
            summary["hot_loop_mapping"] = build_hot_loop_mapping(
                query_id=query_id,
                compiler_summary=summary,
                trace_summary_text=trace_summary_text,
            )
        compiler_summary_by_query[query_id] = summary
        self.compiler_vectorization_summary = compiler_summary_by_query
        return summary

    def _refresh_final_vectorization_summaries(
        self,
        records: list[QueryOptimizationRecord],
    ) -> None:
        """Refresh final compiler vectorization evidence from the latest build reports."""
        trace_by_query = {
            record.query_id: record.trace_summary
            for record in records
            if record.trace_summary
        }
        for query_id in self.query_ids:
            self._collect_compiler_vectorization_summary(
                query_id,
                force_refresh=True,
                trace_summary_text=trace_by_query.get(query_id, ""),
            )
        return None

    def _build_validation_policy(self) -> OptimizationValidationPolicy:
        full = tuple(self.required_validation_sf_list)
        if not full:
            raise RuntimeError("required_validation_sf_list must not be empty")
        light = (min(full),)
        heavy = tuple(sf for sf in full if sf >= 1000)
        return OptimizationValidationPolicy(
            light_scale_factors=light,
            full_scale_factors=full,
            heavyweight_scale_factors=heavy,
        )

    def _touches_shared_runtime_scope(self, written_files: tuple[str, ...]) -> bool:
        shared_markers = (
            "builder_impl", "loader_impl", "query_shared", "query_family_",
            "storage_plan", "args_parser",
        )
        return any(
            any(marker in path for marker in shared_markers)
            for path in written_files
        ) or any(
            path.endswith((".hpp", ".h")) and "query_" not in path
            for path in written_files
        )

    def _run_stage_correctness_gate(
        self,
        query_id: str,
        written_files: tuple[str, ...],
        policy: OptimizationValidationPolicy,
        scope_query_ids: tuple[str, ...] | None = None,
    ) -> Any:
        from tpch_monetdb.conversations.optimization_validation import CorrectnessCheckSummary
        plan = build_unit_validation_plan(
            query_id=query_id,
            scope_query_ids=scope_query_ids,
            written_files=written_files,
            all_query_ids=self.query_ids,
            light_scale_factors=policy.light_scale_factors,
            full_scale_factors=policy.full_scale_factors,
        )
        return run_required_correctness_checks(
            self.run_tool,
            plan.scale_factors,
            list(plan.scope_query_ids),
            trace_mode=False,
            optimize=True,
            external_call=True,
        )

    def _build_global_objective_preview(
        self,
        runtime_by_query: dict[str, float],
    ) -> Any:
        """Build the objective-preview payload available during global attempts."""
        control_artifact_hashes = collect_control_artifact_hashes(self.run_tool.cwd)
        storage_plan_alignment = build_storage_plan_alignment(
            self.run_tool.cwd / "storage_plan.txt"
        )
        workload_objective = load_json_contract(
            self.run_tool.cwd,
            WORKLOAD_OBJECTIVE_FILE,
        )
        preview = type("_GlobalObjectivePreview", (), {})()
        preview.workload_objective = workload_objective
        preview.final_runtime_ms_by_query = {
            query_id: runtime_s * 1000.0
            for query_id, runtime_s in runtime_by_query.items()
            if math.isfinite(runtime_s) and runtime_s > 0
        }
        preview.baseline_runtime_ms_by_query = self._get_baseline_runtime_ms_by_query()
        preview.storage_plan_alignment = storage_plan_alignment
        preview.control_artifact_hashes = control_artifact_hashes
        preview.measurement_repetition = getattr(self, "measurement_repetition", {})
        preview.hardware_counter_summary = getattr(
            self, "hardware_counter_summary_by_query", {}
        )
        preview.compiler_vectorization_summary = getattr(
            self, "compiler_vectorization_summary", {}
        )
        return preview

    def _collect_global_attempt_objective_failures(
        self,
        runtime_by_query: dict[str, float],
    ) -> tuple[str, ...]:
        """Return objective failures that are actionable before final measurement."""
        report = build_objective_failure_report(
            self._build_global_objective_preview(runtime_by_query)
        )
        return tuple(
            failure
            for failure in report.failures
            if failure not in GLOBAL_ATTEMPT_IGNORED_OBJECTIVE_FAILURES
        )

    def _candidate_runtime_ms_by_query(
        self,
        runtime_by_query: dict[str, float],
    ) -> dict[str, float]:
        """Convert finite candidate runtimes from seconds to milliseconds."""
        return {
            query_id: runtime_s * 1000.0
            for query_id, runtime_s in runtime_by_query.items()
            if math.isfinite(runtime_s) and runtime_s > 0
        }

    def _candidate_measurement_records(
        self,
        query_ids: tuple[str, ...],
    ) -> tuple[QueryMeasurementRecord, ...]:
        """Return current official measurement records for the candidate query scope."""
        query_id_set = set(query_ids)
        records: list[QueryMeasurementRecord] = []
        for record in getattr(self, "measurement_records", []):
            query_id = str(record.get("query_id") or "")
            if query_id not in query_id_set:
                continue
            records.append(QueryMeasurementRecord(**record))
        return tuple(records)

    def _candidate_evidence_query_ids(
        self,
        hypothesis: GlobalOptimizationHypothesis,
    ) -> tuple[str, ...]:
        """Choose the query ids that must satisfy candidate evidence gates."""
        affected = tuple(
            query_id for query_id in hypothesis.affected_queries
            if query_id in self.query_ids
        )
        return affected or tuple(self.query_ids)

    def _collect_candidate_structured_gate_failures(
        self,
        *,
        hypothesis: GlobalOptimizationHypothesis,
        before_rt_log: dict[str, float],
        after_rt_log: dict[str, float],
    ) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...], dict[str, float]]:
        """Evaluate structured measurement and causality evidence for one candidate."""
        query_ids = self._candidate_evidence_query_ids(hypothesis)
        workload_objective = load_json_contract(
            self.run_tool.cwd,
            WORKLOAD_OBJECTIVE_FILE,
        )
        scoped_objective = dict(workload_objective)
        scoped_objective["query_ids"] = list(query_ids)
        scoped_objective["critical_query_ids"] = list(query_ids)
        target_map = dict(scoped_objective.get("critical_query_targets", {}) or {})
        for query_id in query_ids:
            target_map.setdefault(query_id, {})
        scoped_objective["critical_query_targets"] = target_map

        after_runtime_ms = self._candidate_runtime_ms_by_query(after_rt_log)
        ledger = build_pipeline_evidence_ledger(
            workspace_path=self.run_tool.cwd,
            workload_objective=scoped_objective,
            final_runtime_ms_by_query=after_runtime_ms,
            baseline_runtime_ms_by_query=self._candidate_runtime_ms_by_query(
                before_rt_log
            ),
            compiler_vectorization_summary=getattr(
                self, "compiler_vectorization_summary", {}
            ),
            hardware_counter_summary=getattr(
                self, "hardware_counter_summary_by_query", {}
            ),
            measurement_records=self._candidate_measurement_records(query_ids),
        )
        causality: list[str] = []
        speedup_by_query: dict[str, float] = {}
        for query_id in query_ids:
            before_s = before_rt_log.get(query_id)
            after_s = after_rt_log.get(query_id)
            if (
                before_s is None
                or after_s is None
                or not math.isfinite(before_s)
                or not math.isfinite(after_s)
                or before_s <= 0
                or after_s <= 0
            ):
                continue
            if after_s < before_s:
                speedup_by_query[query_id] = before_s / after_s
                causality.append(
                    f"{query_id}:runtime {before_s:.6f}s->{after_s:.6f}s"
                )
        measurement_gaps = tuple(
            failure for failure in ledger.failures
            if (
                "RUNTIME" in failure
                or "MEASUREMENT" in failure
                or "PMU" in failure
            )
        )
        rejection_codes = tuple(
            dict.fromkeys((
                *ledger.failures,
                *(
                    ()
                    if causality
                    else ("CAUSALITY_EVIDENCE_MISSING",)
                ),
            ))
        )
        return rejection_codes, measurement_gaps, tuple(causality), speedup_by_query

    def _build_global_patch_units(
        self,
        hypothesis: GlobalOptimizationHypothesis,
        written_files: tuple[str, ...],
    ) -> tuple[GlobalPatchUnit, ...]:
        """Split a global candidate into atomic shared or query-scoped units."""
        shared_files = tuple(path for path in written_files if _is_shared_patch_file(path))
        if shared_files:
            return (
                GlobalPatchUnit(
                    unit_id=f"{hypothesis.id}:atomic_shared",
                    files=tuple(written_files),
                    affected_queries=tuple(hypothesis.affected_queries or self.query_ids),
                    atomic=True,
                ),
            )
        by_query: dict[str, list[str]] = {}
        fallback_queries = tuple(hypothesis.affected_queries or self.query_ids)
        for path in written_files:
            query_id = _query_id_from_patch_file(path)
            if query_id is None:
                query_id = fallback_queries[0] if fallback_queries else "unknown"
            by_query.setdefault(query_id, []).append(path)
        return tuple(
            GlobalPatchUnit(
                unit_id=f"{hypothesis.id}:q{query_id}",
                files=tuple(sorted(paths)),
                affected_queries=(query_id,),
                atomic=False,
            )
            for query_id, paths in sorted(by_query.items())
        )

    async def _try_salvage_global_patch_units(
        self,
        *,
        hypothesis: GlobalOptimizationHypothesis,
        base_snapshot: str,
        candidate_snapshot: str,
        written_files: tuple[str, ...],
        before_rt_log: dict[str, float],
        after_rt_log: dict[str, float],
        regressed_queries: tuple[str, ...],
    ) -> GlobalOptimizationCandidate | None:
        """Apply non-regressed query-scoped units from a rejected global candidate."""
        units = self._build_global_patch_units(hypothesis, written_files)
        if len(units) <= 1 or any(unit.atomic for unit in units):
            return None
        regressed_set = set(regressed_queries)
        accepted_units: list[GlobalPatchUnitResult] = []
        rejected_units: list[GlobalPatchUnitResult] = []
        accepted_files: list[str] = []
        for unit in units:
            unit_regressions = tuple(qid for qid in unit.affected_queries if qid in regressed_set)
            if unit_regressions:
                rejected_units.append(
                    GlobalPatchUnitResult(
                        unit=unit,
                        accepted=False,
                        rejection_codes=("GLOBAL_UNIT_REGRESSION",),
                        regressed_queries=unit_regressions,
                    )
                )
                continue
            accepted_units.append(GlobalPatchUnitResult(unit=unit, accepted=True))
            accepted_files.extend(unit.files)
        if not accepted_units or not accepted_files:
            return None
        self.git_snapshotter.restore(base_snapshot)
        self.git_snapshotter.checkout_paths_from_snapshot(candidate_snapshot, accepted_files)
        validation_summary = run_required_correctness_checks(
            self.run_tool,
            self.required_validation_sf_list,
            self.query_ids,
            trace_mode=False,
            optimize=True,
            external_call=True,
        )
        if not validation_summary.success:
            self.git_snapshotter.restore(base_snapshot)
            return None
        salvaged_rt_log = self._measure_all_queries()
        salvaged_regressed = tuple(
            qid
            for qid, after_s in salvaged_rt_log.items()
            if qid in before_rt_log
            and after_s > before_rt_log[qid] * (1 + self.regression_tolerance)
        )
        if salvaged_regressed:
            self.git_snapshotter.restore(base_snapshot)
            return None
        _parent_hash, salvaged_snapshot = self.git_snapshotter.snapshot(
            f"global_salvaged_{hypothesis.id}"
        )
        if salvaged_snapshot is None:
            salvaged_snapshot = self.git_snapshotter.current_hash or base_snapshot
        _codes, measurement_gaps, causality_evidence, speedup_by_query = (
            self._collect_candidate_structured_gate_failures(
                hypothesis=hypothesis,
                before_rt_log=before_rt_log,
                after_rt_log=salvaged_rt_log,
            )
        )
        return GlobalOptimizationCandidate(
            hypothesis=hypothesis,
            snapshot_hash=salvaged_snapshot,
            accepted=True,
            runtime_by_query=salvaged_rt_log,
            written_files=tuple(sorted(accepted_files)),
            rejection_codes=("PARTIAL_GLOBAL_ACCEPTANCE",),
            rejection_detail="Accepted non-regressed query-scoped patch units only.",
            measurement_gaps=measurement_gaps,
            causality_evidence=causality_evidence,
            speedup_by_query=speedup_by_query,
            partial=True,
            accepted_units=tuple(accepted_units),
            rejected_units=tuple(rejected_units),
        )

    async def _run_global_human_reference(
        self,
        mandatory_constraints: str,
        hotspot_summary_path: Path,
        before_rt_log: dict[str, float],
    ) -> GlobalHumanReferenceResult:
        """Run autonomous diagnosis, candidate competition, and winner selection."""
        base_snapshot = self._anchor_global_control_artifacts(hotspot_summary_path)
        baseline_ms_by_query = self._get_baseline_runtime_ms_by_query()
        before_objective_failures = self._collect_global_attempt_objective_failures(
            before_rt_log
        )

        diagnosis_summary = await self._exec(
            tpch_monetdb_optim_prompt_global_diagnosis(
                constraints_str=mandatory_constraints,
                hotspot_summary_path=hotspot_summary_path.as_posix(),
                sf=self.benchmark_sf,
                trace_evidence="",
                measurement_evidence="",
                objective_evidence=(
                    "\nObjective failures before global optimization: "
                    + json.dumps(
                        list(before_objective_failures),
                        ensure_ascii=False,
                    )
                ),
            ),
            "TPC-H MonetDB Global Diagnosis",
            max_turns=get_optim_stage_max_turns("global_human_reference"),
            tool_profile="optimization_control",
            prompt_metadata={
                "active_query_ids": list(self.query_ids),
                "required_control_artifacts": list(
                    GLOBAL_HUMAN_REFERENCE_REQUIRED_CONTROL_ARTIFACTS
                ),
                "objective_ids": self._objective_ids_for_prompt(),
                "patch_scope_verdict": "global_diagnosis",
            },
        )
        hypotheses = parse_global_optimization_hypotheses(
            None if diagnosis_summary is None else diagnosis_summary.final_output
        )
        if not hypotheses:
            hypotheses = (
                GlobalOptimizationHypothesis(
                    id="h_evidence_gap",
                    summary=(
                        "Diagnosis did not produce parseable hypotheses; record the "
                        "evidence gap instead of applying an ungrounded direction."
                    ),
                    evidence=("diagnosis_final_output",),
                    affected_queries=tuple(self.query_ids),
                    suspected_runtime_path=("measurement",),
                    expected_mechanism="Evidence is insufficient for a code change.",
                    correctness_risk="high",
                    implementation_scope=(),
                    verification_plan=("collect structured evidence",),
                    evidence_gap=True,
                ),
            )

        attempts: list[GlobalHumanReferenceAttempt] = []
        candidates: list[GlobalOptimizationCandidate] = []

        for attempt_index, hypothesis in enumerate(
            hypotheses[:GLOBAL_HUMAN_REFERENCE_MAX_ATTEMPTS],
            start=1,
        ):
            if hypothesis.evidence_gap:
                detail = "Hypothesis is an evidence gap; no code change attempted."
                attempts.append(
                    GlobalHumanReferenceAttempt(
                        attempt_index=attempt_index,
                        written_files=(),
                        accepted=False,
                        rejection_code="CAUSALITY_EVIDENCE_MISSING",
                        rejection_detail=detail,
                    )
                )
                candidates.append(
                    GlobalOptimizationCandidate(
                        hypothesis=hypothesis,
                        snapshot_hash=base_snapshot,
                        accepted=False,
                        runtime_by_query=before_rt_log,
                        rejection_codes=("CAUSALITY_EVIDENCE_MISSING",),
                        rejection_detail=detail,
                        measurement_gaps=("EVIDENCE_GAP",),
                    )
                )
                continue
            if not hypothesis.evidence:
                detail = "Hypothesis does not cite structured evidence."
                attempts.append(
                    GlobalHumanReferenceAttempt(
                        attempt_index=attempt_index,
                        written_files=(),
                        accepted=False,
                        rejection_code="CAUSALITY_EVIDENCE_MISSING",
                        rejection_detail=detail,
                    )
                )
                candidates.append(
                    GlobalOptimizationCandidate(
                        hypothesis=hypothesis,
                        snapshot_hash=base_snapshot,
                        accepted=False,
                        runtime_by_query=before_rt_log,
                        rejection_codes=("CAUSALITY_EVIDENCE_MISSING",),
                        rejection_detail=detail,
                        measurement_gaps=("CAUSALITY_EVIDENCE_MISSING",),
                    )
                )
                continue
            self.git_snapshotter.restore(base_snapshot)
            self._ensure_hotspot_summary_available_after_restore(
                hotspot_summary_path,
                base_snapshot,
            )
            hypothesis_json = json.dumps(
                {
                    "id": hypothesis.id,
                    "summary": hypothesis.summary,
                    "evidence": list(hypothesis.evidence),
                    "affected_queries": list(hypothesis.affected_queries),
                    "suspected_runtime_path": list(hypothesis.suspected_runtime_path),
                    "expected_mechanism": hypothesis.expected_mechanism,
                    "expected_impact": hypothesis.expected_impact or {},
                    "correctness_risk": hypothesis.correctness_risk,
                    "implementation_scope": list(hypothesis.implementation_scope),
                    "verification_plan": list(hypothesis.verification_plan),
                    "evidence_gap": hypothesis.evidence_gap,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
            stage_summary = await self._exec(
                tpch_monetdb_optim_prompt_hypothesis_execution(
                    constraints_str=mandatory_constraints,
                    hypothesis_json=hypothesis_json,
                    sf=self.benchmark_sf,
                    evidence_refs="\n".join(hypothesis.evidence),
                ),
                f"TPC-H MonetDB Global Hypothesis {hypothesis.id}",
                max_turns=get_optim_stage_max_turns("global_human_reference"),
                tool_profile="optimization_infra_layout",
                prompt_metadata={
                    "active_query_ids": list(self.query_ids),
                    "required_control_artifacts": list(
                        GLOBAL_HUMAN_REFERENCE_REQUIRED_CONTROL_ARTIFACTS
                    ),
                    "objective_ids": self._objective_ids_for_prompt(),
                    "patch_scope_verdict": "global_hypothesis",
                },
            )
            written_files = () if stage_summary is None else stage_summary.written_files
            control_artifacts_read = (
                () if stage_summary is None else stage_summary.control_artifacts_read
            )
            if not written_files:
                rejection_detail = (
                    "Hypothesis execution made no source changes."
                    if not hypothesis.evidence_gap
                    else "Hypothesis is an evidence gap and produced no code change."
                )
                attempts.append(
                    GlobalHumanReferenceAttempt(
                        attempt_index=attempt_index,
                        written_files=(),
                        accepted=False,
                        rejection_code="NO_GLOBAL_CHANGE",
                        rejection_detail=rejection_detail,
                        control_artifacts_read=control_artifacts_read,
                    )
                )
                candidates.append(
                    GlobalOptimizationCandidate(
                        hypothesis=hypothesis,
                        snapshot_hash=base_snapshot,
                        accepted=False,
                        runtime_by_query=before_rt_log,
                        rejection_codes=("NO_GLOBAL_CHANGE",),
                        rejection_detail=rejection_detail,
                        measurement_gaps=(
                            ("EVIDENCE_GAP",) if hypothesis.evidence_gap else ()
                        ),
                    )
                )
                continue

            validation_summary = run_required_correctness_checks(
                self.run_tool,
                self.required_validation_sf_list,
                self.query_ids,
                trace_mode=False,
                optimize=True,
                external_call=True,
            )
            if not validation_summary.success:
                detail = validation_summary.failure_detail or validation_summary.message
                _parent_hash, rejected_snapshot = self.git_snapshotter.snapshot(
                    f"global_rejected_{hypothesis.id}_{attempt_index}"
                )
                if rejected_snapshot is None:
                    rejected_snapshot = self.git_snapshotter.current_hash or base_snapshot
                self.git_snapshotter.restore(base_snapshot)
                attempts.append(
                    GlobalHumanReferenceAttempt(
                        attempt_index=attempt_index,
                        written_files=written_files,
                        accepted=False,
                        rejection_code=(
                            validation_summary.failure_code
                            or "GLOBAL_VALIDATION_FAILED"
                        ),
                        rejection_detail=detail,
                        control_artifacts_read=control_artifacts_read,
                    )
                )
                candidates.append(
                    GlobalOptimizationCandidate(
                        hypothesis=hypothesis,
                        snapshot_hash=rejected_snapshot,
                        accepted=False,
                        runtime_by_query=before_rt_log,
                        written_files=written_files,
                        rejection_codes=(
                            validation_summary.failure_code
                            or "GLOBAL_VALIDATION_FAILED",
                        ),
                        rejection_detail=detail,
                    )
                )
                continue

            after_rt_log = self._measure_all_queries()
            regressed = tuple(
                qid
                for qid, after_s in after_rt_log.items()
                if qid in before_rt_log
                and after_s > before_rt_log[qid] * (1 + self.regression_tolerance)
            )
            objective_failures = self._collect_global_attempt_objective_failures(
                after_rt_log
            )
            (
                structured_rejection_codes,
                measurement_gaps,
                causality_evidence,
                speedup_by_query,
            ) = self._collect_candidate_structured_gate_failures(
                hypothesis=hypothesis,
                before_rt_log=before_rt_log,
                after_rt_log=after_rt_log,
            )
            objective_improved = len(objective_failures) < len(before_objective_failures)
            runtime_improved = any(
                qid in before_rt_log
                and math.isfinite(after_s)
                and after_s < before_rt_log[qid]
                for qid, after_s in after_rt_log.items()
            )
            if regressed:
                detail = (
                    "Global hypothesis candidate regressed queries: "
                    + ", ".join(regressed)
                )
                _parent_hash, rejected_snapshot = self.git_snapshotter.snapshot(
                    f"global_rejected_{hypothesis.id}_{attempt_index}"
                )
                if rejected_snapshot is None:
                    rejected_snapshot = self.git_snapshotter.current_hash or base_snapshot
                salvaged_candidate = await self._try_salvage_global_patch_units(
                    hypothesis=hypothesis,
                    base_snapshot=base_snapshot,
                    candidate_snapshot=rejected_snapshot,
                    written_files=written_files,
                    before_rt_log=before_rt_log,
                    after_rt_log=after_rt_log,
                    regressed_queries=regressed,
                )
                if salvaged_candidate is not None:
                    attempts.append(
                        GlobalHumanReferenceAttempt(
                            attempt_index=attempt_index,
                            written_files=salvaged_candidate.written_files,
                            accepted=True,
                            rejection_code="PARTIAL_GLOBAL_ACCEPTANCE",
                            rejection_detail=(
                                "Full candidate regressed, but independent patch units were salvaged."
                            ),
                            regressed_queries=regressed,
                            objective_failures=objective_failures,
                            control_artifacts_read=control_artifacts_read,
                        )
                    )
                    candidates.append(salvaged_candidate)
                    self.git_snapshotter.restore(base_snapshot)
                    continue
                self.git_snapshotter.restore(base_snapshot)
                attempts.append(
                    GlobalHumanReferenceAttempt(
                        attempt_index=attempt_index,
                        written_files=written_files,
                        accepted=False,
                        rejection_code="GLOBAL_REGRESSION",
                        rejection_detail=detail,
                        regressed_queries=regressed,
                        objective_failures=objective_failures,
                        control_artifacts_read=control_artifacts_read,
                    )
                )
                candidates.append(
                    GlobalOptimizationCandidate(
                        hypothesis=hypothesis,
                        snapshot_hash=rejected_snapshot,
                        accepted=False,
                        runtime_by_query=after_rt_log,
                        written_files=written_files,
                        rejection_codes=("GLOBAL_REGRESSION",),
                        rejection_detail=detail,
                        objective_failures=objective_failures,
                    )
                )
                continue
            if structured_rejection_codes:
                detail = (
                    "Global hypothesis candidate failed structured evidence gates: "
                    + ", ".join(structured_rejection_codes)
                )
                _parent_hash, rejected_snapshot = self.git_snapshotter.snapshot(
                    f"global_rejected_{hypothesis.id}_{attempt_index}"
                )
                if rejected_snapshot is None:
                    rejected_snapshot = self.git_snapshotter.current_hash or base_snapshot
                self.git_snapshotter.restore(base_snapshot)
                attempts.append(
                    GlobalHumanReferenceAttempt(
                        attempt_index=attempt_index,
                        written_files=written_files,
                        accepted=False,
                        rejection_code=structured_rejection_codes[0],
                        rejection_detail=detail,
                        objective_failures=objective_failures,
                        control_artifacts_read=control_artifacts_read,
                    )
                )
                candidates.append(
                    GlobalOptimizationCandidate(
                        hypothesis=hypothesis,
                        snapshot_hash=rejected_snapshot,
                        accepted=False,
                        runtime_by_query=after_rt_log,
                        written_files=written_files,
                        rejection_codes=structured_rejection_codes,
                        rejection_detail=detail,
                        objective_failures=objective_failures,
                        measurement_gaps=measurement_gaps,
                        causality_evidence=causality_evidence,
                        speedup_by_query=speedup_by_query,
                    )
                )
                continue
            if objective_failures and not objective_improved and not runtime_improved:
                detail = (
                    "Global hypothesis candidate did not reduce objective "
                    "failures or improve runtime."
                )
                _parent_hash, rejected_snapshot = self.git_snapshotter.snapshot(
                    f"global_rejected_{hypothesis.id}_{attempt_index}"
                )
                if rejected_snapshot is None:
                    rejected_snapshot = self.git_snapshotter.current_hash or base_snapshot
                self.git_snapshotter.restore(base_snapshot)
                attempts.append(
                    GlobalHumanReferenceAttempt(
                        attempt_index=attempt_index,
                        written_files=written_files,
                        accepted=False,
                        rejection_code="OBJECTIVE_NOT_IMPROVED",
                        rejection_detail=detail,
                        objective_failures=objective_failures,
                        control_artifacts_read=control_artifacts_read,
                    )
                )
                candidates.append(
                    GlobalOptimizationCandidate(
                        hypothesis=hypothesis,
                        snapshot_hash=rejected_snapshot,
                        accepted=False,
                        runtime_by_query=after_rt_log,
                        written_files=written_files,
                        rejection_codes=("OBJECTIVE_NOT_IMPROVED",),
                        rejection_detail=detail,
                        objective_failures=objective_failures,
                    )
                )
                continue

            _parent_hash, candidate_snapshot = self.git_snapshotter.snapshot(
                f"global_hypothesis_{hypothesis.id}_{attempt_index}"
            )
            if candidate_snapshot is None:
                candidate_snapshot = self.git_snapshotter.current_hash or base_snapshot
            accepted_attempt = GlobalHumanReferenceAttempt(
                attempt_index=attempt_index,
                written_files=written_files,
                accepted=True,
                objective_failures=objective_failures,
                control_artifacts_read=control_artifacts_read,
            )
            attempts.append(accepted_attempt)
            candidates.append(
                GlobalOptimizationCandidate(
                    hypothesis=hypothesis,
                    snapshot_hash=candidate_snapshot,
                    accepted=True,
                    runtime_by_query=after_rt_log,
                    written_files=written_files,
                    objective_failures=objective_failures,
                    measurement_gaps=measurement_gaps,
                    causality_evidence=causality_evidence,
                    speedup_by_query=speedup_by_query,
                )
            )
            self.git_snapshotter.restore(base_snapshot)

        winner = select_global_winner(tuple(candidates), baseline_ms_by_query)
        if winner is None:
            self.git_snapshotter.restore(base_snapshot)
            return GlobalHumanReferenceResult(
                runtime_by_query=before_rt_log,
                written_files=(),
                accepted=False,
                attempts=tuple(attempts),
                regressed_queries=tuple(
                    dict.fromkeys(
                        qid
                        for attempt in attempts
                        for qid in attempt.regressed_queries
                    )
                ),
                failure_detail=(
                    "Global hypothesis candidates made no accepted source changes."
                ),
                hypotheses=tuple(hypotheses),
                candidates=tuple(candidates),
                winner=None,
            )
        self.git_snapshotter.restore(winner.snapshot_hash)
        return GlobalHumanReferenceResult(
            runtime_by_query=winner.runtime_by_query,
            written_files=winner.written_files,
            accepted=True,
            attempts=tuple(attempts),
            regressed_queries=(),
            failure_detail=None,
            hypotheses=tuple(hypotheses),
            candidates=tuple(candidates),
            winner=winner,
        )

    def _record_global_regression(
        self,
        result: GlobalHumanReferenceResult,
    ) -> None:
        """Append rejected global human-reference attempts to regression records."""
        for attempt in result.attempts:
            if attempt.accepted:
                continue
            self.global_regression_records.append({
                "stage_name": "global_human_reference",
                "attempt_index": attempt.attempt_index,
                "accepted": False,
                "rejection_code": attempt.rejection_code,
                "regressed_queries": list(attempt.regressed_queries),
                "objective_failures": list(attempt.objective_failures),
                "failure_detail": attempt.rejection_detail,
            })
        if not result.attempts:
            self.global_regression_records.append({
                "stage_name": "global_human_reference",
                "accepted": False,
                "regressed_queries": list(result.regressed_queries),
                "failure_detail": result.failure_detail,
            })

    def _persist_hotspot_summary(
        self, records: list[QueryOptimizationRecord]
    ) -> Path:
        """Persist per-query runtime and perf hotspot evidence for global review."""
        hotspot_path = self.run_tool.cwd / "optimization_hotspot_summary.md"
        lines = ["# Optimization Hotspot Summary", ""]
        hardware_summary_by_query = getattr(self, "hardware_counter_summary_by_query", {})
        output_split_by_query = getattr(self, "output_split_by_query", {}) or {}
        quality_failures = self._collect_hotspot_summary_quality_failures(records)
        lines.append("## Summary Quality")
        if quality_failures:
            lines.append("- Status: degraded")
            for failure in quality_failures:
                lines.append(f"- Quality issue: {failure}")
        else:
            lines.append("- Status: complete")
        lines.append("")
        for rec in records:
            hardware_summary = hardware_summary_by_query.get(rec.query_id, {})
            output_split = output_split_by_query.get(rec.query_id)
            lines.append(f"## Query {rec.query_id}")
            lines.append(f"- Issue class: {rec.issue_class}")
            lines.append(f"- Stage: {rec.stage_name}")
            lines.append(
                "- Optimization runtime (no_output): "
                f"{self._format_seconds_for_summary(rec.rt_before_s)} -> "
                f"{self._format_seconds_for_summary(rec.rt_after_s)}"
            )
            if output_split is None:
                lines.append("- Output split: missing")
            else:
                lines.append(
                    "- Output split: "
                    f"full_csv={output_split.full_csv_s * 1000.0:.3f}ms; "
                    f"no_output={output_split.no_output_s * 1000.0:.3f}ms; "
                    f"materialization={output_split.materialization_s * 1000.0:.3f}ms; "
                    f"materialization_ratio={output_split.materialization_ratio:.3f}"
                )
            if rec.failed:
                lines.append(f"- Failed: {rec.failure_code} — {rec.failure_detail}")
            if rec.sampled_instantiations:
                lines.append(
                    f"- Sampled instantiations: {', '.join(rec.sampled_instantiations)}"
                )
            perf_lines = _format_perf_hotspot_markdown_lines(hardware_summary)
            if perf_lines:
                lines.extend(perf_lines)
            else:
                perf_summary = _summarize_perf_symbols_from_trace_text(
                    rec.trace_summary
                )
                if perf_summary is not None:
                    lines.append(f"- Perf top symbols: {perf_summary}")
            lines.extend(format_pmu_perf_status_markdown_lines(hardware_summary))
            lines.append("")
        hotspot_path.write_text("\n".join(lines), encoding="utf-8")
        return hotspot_path

    def _format_seconds_for_summary(self, value: float) -> str:
        """Format runtime evidence without hiding sub-millisecond values as zero."""
        if not math.isfinite(value):
            return "missing"
        return f"{value * 1000.0:.3f}ms"

    def _collect_hotspot_summary_quality_failures(
        self,
        records: list[QueryOptimizationRecord],
    ) -> tuple[str, ...]:
        """Collect non-fatal hotspot summary quality issues for global diagnosis."""
        failures: list[str] = []
        output_split_by_query = getattr(self, "output_split_by_query", {}) or {}
        if not records:
            failures.append("NO_OPTIMIZATION_RECORDS")
        for rec in records:
            if rec.failed:
                continue
            if rec.query_id not in output_split_by_query:
                failures.append(f"Q{rec.query_id}:OUTPUT_SPLIT_MISSING")
            if not math.isfinite(rec.rt_before_s) or rec.rt_before_s <= 0:
                failures.append(f"Q{rec.query_id}:NO_OUTPUT_BEFORE_RUNTIME_MISSING")
            if not math.isfinite(rec.rt_after_s) or rec.rt_after_s <= 0:
                failures.append(f"Q{rec.query_id}:NO_OUTPUT_AFTER_RUNTIME_MISSING")
        return tuple(failures)

    def _anchor_global_control_artifacts(self, hotspot_summary_path: Path) -> str:
        """Snapshot host-owned global control artifacts before global restores."""
        if not hotspot_summary_path.exists():
            raise RuntimeError(
                "[ERROR:CONTROL_ARTIFACT_MISSING] "
                "stage=TPC-H MonetDB Global Diagnosis Missing required control artifact: "
                f"{hotspot_summary_path.name}"
            )
        base_snapshot = self.git_snapshotter.current_hash
        if base_snapshot is None:
            raise RuntimeError("Current git snapshot is None.")

        is_dirty = getattr(self.git_snapshotter, "is_dirty", None)
        if not callable(is_dirty) or not is_dirty():
            return base_snapshot

        _parent_hash, anchored_snapshot = self.git_snapshotter.snapshot(
            "optimization_hotspot_summary_control"
        )
        if anchored_snapshot is not None:
            return anchored_snapshot
        return self.git_snapshotter.current_hash or base_snapshot

    def _ensure_hotspot_summary_available_after_restore(
        self,
        hotspot_summary_path: Path,
        base_snapshot: str,
    ) -> None:
        """Fail fast when a restore loses the global hotspot control artifact."""
        if hotspot_summary_path.exists():
            return None
        raise RuntimeError(
            "[ERROR:CONTROL_ARTIFACT_RESTORED_AWAY] "
            "stage=TPC-H MonetDB Global Hypothesis "
            f"base_snapshot={base_snapshot} lost required control artifact: "
            f"{hotspot_summary_path.name}"
        )
