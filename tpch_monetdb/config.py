"""TPC-H validation and workflow configuration."""

from dataclasses import dataclass
from typing import Any, Iterable

DEFAULT_TPCH_MONETDB_ARTIFACTS_DIR = "./tpch_monetdb_artifacts"

# Dockerized TPC-H replacement path defaults.
TPCH_VERIFY_SF_LIST = [1]
TPCH_BENCHMARK_SF_LIST = [1]
TPCH_MAX_SCALE_FACTOR = 1

_SCRIPTED_READINESS_MINIMAL_SF = 1


def resolve_scripted_readiness_scale_factors(
    validation_mode: str,
    verify_sf_list: list[int],
    benchmark_sf: int,
) -> list[int]:
    """返回 scripted 启动前需要覆盖的验证 scale factor 列表."""
    normalized_mode = validation_mode.strip().lower()
    if normalized_mode == "strict":
        return resolve_workflow_scale_factors(
            benchmark_sf=benchmark_sf,
            verify_sf_list=verify_sf_list,
        )
    if normalized_mode == "traversal":
        return [_SCRIPTED_READINESS_MINIMAL_SF]
    raise ValueError(
        "Unknown validation_mode: "
        f"{validation_mode!r}. Use 'strict' or 'traversal'."
    )


def get_tpch_verify_scale_factors() -> list[int]:
    """Return correctness scale factors for the TPC-H replacement path."""
    return list(TPCH_VERIFY_SF_LIST)


def get_tpch_benchmark_scale_factors() -> list[int]:
    """Return benchmark scale factors for the TPC-H replacement path."""
    return list(TPCH_BENCHMARK_SF_LIST)


def resolve_active_verify_scale_factors(
    benchmark_sf: int,
    verify_sf_list: Iterable[int] | None = None,
) -> list[int]:
    """按 benchmark ceiling 裁剪 active verify scale factors."""
    source = TPCH_VERIFY_SF_LIST if verify_sf_list is None else verify_sf_list
    ceiling = int(benchmark_sf)
    result: list[int] = []
    for sf in source:
        value = int(sf)
        if value <= ceiling and value not in result:
            result.append(value)
    return result


def resolve_workflow_scale_factors(
    benchmark_sf: int,
    verify_sf_list: Iterable[int] | None = None,
    benchmark_sf_list: Iterable[int] | None = None,
) -> list[int]:
    """返回当前 workflow 在给定 benchmark ceiling 下应使用的完整 scale set."""
    ceiling = int(benchmark_sf)
    verify_part = resolve_active_verify_scale_factors(ceiling, verify_sf_list)
    benchmark_source = (
        TPCH_BENCHMARK_SF_LIST if benchmark_sf_list is None else benchmark_sf_list
    )
    benchmark_part = [int(sf) for sf in benchmark_source if int(sf) <= ceiling]
    return list(dict.fromkeys([*verify_part, *benchmark_part, ceiling]))


def get_tpch_benchmark_scale_factor() -> int:
    """Return the default benchmark scale factor for the TPC-H path."""
    return TPCH_MAX_SCALE_FACTOR


def get_default_benchmark_scale_factor(benchmark: str) -> int:
    """Return the default benchmark scale factor for the selected benchmark."""
    normalized = benchmark.strip().lower()
    if normalized == "tpch":
        return get_tpch_benchmark_scale_factor()
    raise ValueError(f"Unknown benchmark: {benchmark}")


def get_default_verify_scale_factors(
    benchmark: str,
    purpose: str = "verify",
) -> tuple[list[int], int]:
    """Return scale-factor defaults for the supported benchmark."""
    normalized = benchmark.strip().lower()
    if normalized != "tpch":
        raise ValueError(f"Unknown benchmark: {benchmark}")
    if purpose == "verify":
        return get_tpch_verify_scale_factors(), TPCH_MAX_SCALE_FACTOR
    if purpose == "benchmark":
        return get_tpch_benchmark_scale_factors(), max(TPCH_BENCHMARK_SF_LIST)
    if purpose == "smoke":
        return [1], 1
    raise ValueError(
        f"Unknown purpose: {purpose}. Use 'verify', 'benchmark', or 'smoke'."
    )


# ---------------------------------------------------------------------------
# Phase10 declarative budget configuration
# ---------------------------------------------------------------------------

PHASE10_STAGE_TURN_BUDGETS: dict[str, int] = {
    "todo_plan": 120,
    "finish_skeleton": 384,
    "compile_fix": 512,
    "todo_sync": 64,
    "add_timings": 160,
    "implement_primary_query": 120,
    "correctness_primary_query": 256,
    "implement_single_query": 160,
    "correctness_single_query": 320,
    "implement_family_kernel": 256,
    "implement_family_entrypoint": 48,
    "correctness_family_kernel": 384,
    "todo_sync_quick": 32,
    "refactor_to_family": 48,
    "all_queries_correctness": 320,
    "benchmark": 120,
    "optimize_build": 192,
}

PHASE10_OPTIM_STAGE_MAX_TURNS: dict[str, int] = {
    "trace_expert": 420,
    "global_human_reference": 360,
}

TPCH_MONETDB_RUNTIME_RELOAD_MAX_FILES = 128
TPCH_MONETDB_RUNTIME_RELOAD_MAX_BYTES = 256 * 1024 * 1024


@dataclass(frozen=True)
class Phase10OuterLoopDefaults:
    """Phase10 outer-loop default configuration."""

    max_rounds: int = 6
    stagnant_rounds: int = 3
    convergence_threshold: float = 0.02
    regression_tolerance: float = 0.05
    retry_budget: int = 2


PHASE10_OUTER_LOOP_DEFAULTS = Phase10OuterLoopDefaults()

PHASE10_PROFILE_OBSERVATION_LIMITS: dict[str, tuple[int, int | None]] = {
    "legacy_general": (24, 96),
    "default_general": (24, 96),
    "todo_plan": (48, None),
    "storage_plan": (36, 96),
    "finish_skeleton": (48, 144),
    "compile_fix": (24, 96),
    "todo_sync": (16, None),
    "add_timings": (24, 96),
    "implement_queries": (48, 192),
    "implement_queries_writeonly": (48, 192),
    "correctness_queries_writeonly": (24, 96),
    "correctness_foundation": (24, 96),
    "correctness": (24, 96),
    "benchmark": (18, 72),
    "optimize_build": (24, 96),
    "optimization_instrumentation": (24, 96),
    "optimization_general": (24, 96),
    "optimization_control": (8, 32),
    "optimization_todo_sync": (4, 16),
}
PHASE10_MAX_STALLED_EXECUTIONS = 3
PHASE10_MAX_CONSECUTIVE_FAILURES = 5


def get_stage_turn_budget(stage: str) -> int:
    """Return the phase10 declarative turn budget for a stage name."""
    return PHASE10_STAGE_TURN_BUDGETS.get(stage, 75)


def get_optim_stage_max_turns(stage: str) -> int:
    """Return the phase10 declarative max_turns for an optimization inner stage."""
    return PHASE10_OPTIM_STAGE_MAX_TURNS.get(stage, 450)


def get_profile_observation_limits(profile_name: str) -> tuple[int, int | None]:
    """Return (soft_limit, hard_limit) for a tool profile."""
    if profile_name not in PHASE10_PROFILE_OBSERVATION_LIMITS:
        raise KeyError(f"Unknown phase10 tool profile: {profile_name}")
    return PHASE10_PROFILE_OBSERVATION_LIMITS[profile_name]


def get_max_stalled_executions() -> int:
    return PHASE10_MAX_STALLED_EXECUTIONS


def get_max_consecutive_failures() -> int:
    return PHASE10_MAX_CONSECUTIVE_FAILURES


def get_outer_loop_defaults() -> dict[str, Any]:
    """Return phase10 outer-loop defaults as a plain dict for argparse override."""
    d = PHASE10_OUTER_LOOP_DEFAULTS
    return {
        "max_rounds": d.max_rounds,
        "stagnant_rounds": d.stagnant_rounds,
        "convergence_threshold": d.convergence_threshold,
        "regression_tolerance": d.regression_tolerance,
        "retry_budget": d.retry_budget,
    }
