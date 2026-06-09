"""Outer-loop state model and round ledger management."""

import json
import logging
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from tpch_monetdb.config import DEFAULT_TPCH_MONETDB_ARTIFACTS_DIR
from tpch_monetdb.utils.large_data_objectives import (
    classify_objective_failure_route,
    collect_large_data_failures,
)

logger = logging.getLogger(__name__)

WORKFLOW_PRIORITY_ORDER: tuple[tuple[str, str], ...] = (
    ("P0", "correctness"),
    ("P1", "query runtime / speedup vs MonetDB baseline"),
    ("P2", "build/ingest time guardrail"),
)


@dataclass
class PhaseInfo:
    """Per-phase status within a round."""

    conv_name: str
    status: str = "pending"  # pending | running | success | failed
    summary_path: str | None = None
    retry_count: int = 0
    failure_code: str | None = None
    failure_detail: str | None = None


@dataclass
class RoundRecord:
    """Single round record in the outer loop."""

    outer_loop_name: str
    round_index: int
    query_list: list[str]
    storage_plan: PhaseInfo | None = None
    base_impl: PhaseInfo | None = None
    optimization: PhaseInfo | None = None
    outcome: str = "pending"  # improved | stagnant | regressed | failed | pending
    action: str = "pending"   # continue | converged | failed | max_rounds | continue_with_best | pending
    action_reason: str = ""
    action_reason_code: str = ""
    aggregate_runtime_ms: float = 0.0
    best_round_index: int = 0
    best_optimization_summary_path: str | None = None
    best_aggregate_runtime_ms: float = 0.0
    best_final_snapshot_hash: str | None = None   # final_snapshot_hash from the best optimization run
    objective_failures: list[str] = field(default_factory=list)
    failure_route: str | None = None

    def to_dict(self) -> dict:
        data = asdict(self)
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "RoundRecord":
        if "storage_plan" in data and data["storage_plan"] is not None:
            data["storage_plan"] = PhaseInfo(**data["storage_plan"])
        if "base_impl" in data and data["base_impl"] is not None:
            data["base_impl"] = PhaseInfo(**data["base_impl"])
        if "optimization" in data and data["optimization"] is not None:
            data["optimization"] = PhaseInfo(**data["optimization"])
        return cls(**data)


def get_outer_loop_dir(artifacts_dir: Path, outer_loop_name: str) -> Path:
    return artifacts_dir / "outer_loop_runs" / outer_loop_name


def write_round_record(record: RoundRecord, artifacts_dir: Path) -> Path:
    outer_dir = get_outer_loop_dir(artifacts_dir, record.outer_loop_name)
    outer_dir.mkdir(parents=True, exist_ok=True)

    filename = f"round_{record.round_index:03d}.json"
    file_path = outer_dir / filename
    with open(file_path, "w") as f:
        json.dump(record.to_dict(), f, indent=2)

    latest_path = outer_dir / "latest.json"
    with open(latest_path, "w") as f:
        json.dump({
            "latest_round": record.round_index,
            "latest_file": filename,
            "record": record.to_dict(),
        }, f, indent=2)

    logger.info(f"Written round record to {file_path}")
    return file_path


def load_latest_round_record(outer_loop_name: str, artifacts_dir: Path | None = None) -> RoundRecord | None:
    if artifacts_dir is None:
        artifacts_dir = Path(DEFAULT_TPCH_MONETDB_ARTIFACTS_DIR)
    outer_dir = get_outer_loop_dir(artifacts_dir, outer_loop_name)
    latest_path = outer_dir / "latest.json"
    if not latest_path.exists():
        return None
    with open(latest_path) as f:
        data = json.load(f)
    record_data = data.get("record")
    if not isinstance(record_data, dict):
        return None
    return RoundRecord.from_dict(record_data)


def load_round_record(outer_loop_name: str, round_index: int, artifacts_dir: Path | None = None) -> RoundRecord | None:
    if artifacts_dir is None:
        artifacts_dir = Path(DEFAULT_TPCH_MONETDB_ARTIFACTS_DIR)
    outer_dir = get_outer_loop_dir(artifacts_dir, outer_loop_name)
    file_path = outer_dir / f"round_{round_index:03d}.json"
    if not file_path.exists():
        return None
    with open(file_path) as f:
        data = json.load(f)
    return RoundRecord.from_dict(data)


def build_conv_names(outer_loop_name: str, round_index: int) -> tuple[str, str, str]:
    """Return (storage_plan_conv, base_impl_conv, optimization_conv) for a round."""
    if not outer_loop_name.startswith("outer"):
        raise ValueError(
            f"Outer loop name must start with 'outer', got {outer_loop_name!r}"
        )
    suffix = outer_loop_name[len("outer") :]
    round_suffix = f"_r{round_index:03d}"
    storage_plan_conv = f"storageplan{suffix}{round_suffix}"
    base_impl_conv = f"basef{suffix}{round_suffix}"
    optimization_conv = f"runoptim{suffix}{round_suffix}"
    return storage_plan_conv, base_impl_conv, optimization_conv


def render_workflow_priority_order() -> str:
    labels = [f"{level}={label}" for level, label in WORKFLOW_PRIORITY_ORDER]
    return "Workflow priority: " + " > ".join(labels)


def compute_aggregate_runtime_ms(final_runtime_ms_by_query: dict[str, float]) -> float:
    """Compute geometric mean of final runtimes."""
    values = [final_runtime_ms_by_query[qid] for qid in sorted(final_runtime_ms_by_query)]
    if not values:
        raise ValueError("final_runtime_ms_by_query must not be empty")
    for value in values:
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError("final_runtime_ms_by_query must contain positive finite values")
    product = math.prod(values)
    return product ** (1.0 / len(values))


def compute_round_decision(
    prev_summary: "OptimizationRunSummary | None",
    curr_summary: "OptimizationRunSummary",
    *,
    convergence_threshold: float,
    stagnant_rounds: int,
    regression_tolerance: float,
    max_rounds: int,
    current_round_index: int,
    stagnant_count: int,
    has_best_snapshot: bool = False,
) -> tuple[str, str, int]:
    """Return (outcome, action, new_stagnant_count) for the current round.

    outcome: improved | stagnant | regressed | failed
    action: continue | converged | failed | max_rounds | continue_with_best

    `continue_with_best` is returned instead of `failed` when a regression
    occurs but a best_final_snapshot_hash is already recorded, allowing the
    outer loop to keep exploring without terminating the entire run.
    """
    if not curr_summary.final_correctness:
        return "failed", "failed", 0
    objective_failures = collect_large_data_failures(curr_summary)
    if objective_failures:
        route = classify_objective_failure_route(objective_failures)
        if current_round_index >= max_rounds:
            return "objective_failed", "failed", 0
        if route == "storage_plan":
            return "objective_failed", "continue", 0
        return "objective_failed", "continue", 0
    if not curr_summary.success:
        return "failed", "failed", 0

    curr_rt = compute_aggregate_runtime_ms(curr_summary.final_runtime_ms_by_query)

    if prev_summary is None or not prev_summary.success:
        if current_round_index >= max_rounds:
            return "improved", "max_rounds", 0
        return "improved", "continue", 0

    prev_rt = compute_aggregate_runtime_ms(prev_summary.final_runtime_ms_by_query)
    if prev_rt <= 0:
        if current_round_index >= max_rounds:
            return "improved", "max_rounds", 0
        return "improved", "continue", 0

    improvement = (prev_rt - curr_rt) / prev_rt

    if improvement < -regression_tolerance:
        # Regression: if we already have a best snapshot, continue exploring
        # instead of aborting — the outer loop can restore best at the end.
        if has_best_snapshot and current_round_index < max_rounds:
            return "regressed", "continue_with_best", 0
        return "regressed", "failed", 0

    if improvement < convergence_threshold:
        new_stagnant_count = stagnant_count + 1
        if new_stagnant_count >= stagnant_rounds:
            return "stagnant", "converged", new_stagnant_count
        if current_round_index >= max_rounds:
            return "stagnant", "max_rounds", new_stagnant_count
        return "stagnant", "continue", new_stagnant_count

    if current_round_index >= max_rounds:
        return "improved", "max_rounds", 0

    return "improved", "continue", 0


def determine_resume_phase(record: RoundRecord, retry_budget: int) -> tuple[str, RoundRecord]:
    """Determine which phase to resume from and update retry counts.

    Returns (phase_name, updated_record). phase_name is one of:
    storage_plan, base_impl, optimization, next_round, failed.
    """
    sp = record.storage_plan
    bi = record.base_impl
    opt = record.optimization

    if sp is None:
        sp_conv, _, _ = build_conv_names(record.outer_loop_name, record.round_index)
        record.storage_plan = PhaseInfo(conv_name=sp_conv, status="running")
        return "storage_plan", record
    if sp.status != "success":
        if sp.status == "failed":
            if sp.retry_count >= retry_budget:
                return "failed", record
            sp.retry_count += 1
        sp.status = "running"
        return "storage_plan", record

    if bi is None:
        _, base_conv, _ = build_conv_names(record.outer_loop_name, record.round_index)
        record.base_impl = PhaseInfo(conv_name=base_conv, status="running")
        return "base_impl", record
    if bi.status != "success":
        if bi.status == "failed":
            if bi.retry_count >= retry_budget:
                return "failed", record
            bi.retry_count += 1
        bi.status = "running"
        return "base_impl", record

    if opt is None:
        _, _, opt_conv = build_conv_names(record.outer_loop_name, record.round_index)
        record.optimization = PhaseInfo(conv_name=opt_conv, status="running")
        return "optimization", record
    if opt.status != "success":
        if opt.status == "failed":
            if opt.retry_count >= retry_budget:
                return "failed", record
            opt.retry_count += 1
        opt.status = "running"
        return "optimization", record

    return "next_round", record
