"""TPC-H MonetDB 专用 Optimization Prompt 生成器.

使用 TPC-H MonetDB 专用约束和配置，而不是 OLAP 通用版本。
"""

import logging
from pathlib import Path
from string import Template

from tpch_monetdb.utils.query_codegen_hints import build_query_codegen_hint_text

logger = logging.getLogger(__name__)

_TPCH_MONETDB_PROMPTS_ROOT_DIR = Path(__file__).parent / "prompts"
_TPCH_MONETDB_SHARED_PROMPTS_DIR = _TPCH_MONETDB_PROMPTS_ROOT_DIR / "shared"
_TPCH_MONETDB_OPTIMIZATION_PROMPTS_DIR = _TPCH_MONETDB_PROMPTS_ROOT_DIR / "optimization"
_TPCH_MONETDB_OPT_BASE_PROMPTS_DIR = _TPCH_MONETDB_OPTIMIZATION_PROMPTS_DIR / "base"
_TPCH_MONETDB_OPT_INSTRUMENTATION_PROMPTS_DIR = (
    _TPCH_MONETDB_OPTIMIZATION_PROMPTS_DIR / "instrumentation"
)
_TPCH_MONETDB_OPT_STAGE_PROMPTS_DIR = _TPCH_MONETDB_OPTIMIZATION_PROMPTS_DIR / "stages"
_TPCH_MONETDB_OPT_SHARED_PROMPTS_DIR = _TPCH_MONETDB_OPTIMIZATION_PROMPTS_DIR / "shared"
_TPCH_MONETDB_CONSTRAINTS_PATH = _TPCH_MONETDB_SHARED_PROMPTS_DIR / "tpch_monetdb_optim_constraints.txt"
_TPCH_MONETDB_EXPERT_KNOWLEDGE_PATH = (
    _TPCH_MONETDB_SHARED_PROMPTS_DIR / "tpch_monetdb_expert_knowledge.txt"
)


def _load_txt(path: Path) -> str:
    """Load text file."""
    return path.read_text(encoding="utf-8")


def _load_storage_text(asset_name: str) -> str:
    """Load a storage-mode prompt fragment from registered optimization assets."""
    return _load_txt(_TPCH_MONETDB_OPT_SHARED_PROMPTS_DIR / asset_name).strip()


def _load_q1_q9_checklist() -> str:
    """Load the Q1/Q9 critical-query optimization checklist."""
    return _load_storage_text("q1_q9_optimization_checklist.txt")


def _render_template(path: Path, variables: dict[str, object]) -> str:
    """Render one prompt asset while keeping natural-language text in asset files."""
    template = Template(_load_txt(path))
    return template.substitute({key: str(value) for key, value in variables.items()})


def tpch_monetdb_optim_prompt_pretext(queries_path: str, num_queries: int) -> str:
    """生成 optimization pretext prompt，加载并填充 pretext txt 资产."""
    query_str = "query" if num_queries == 1 else "queries"
    return _render_template(
        _TPCH_MONETDB_OPT_BASE_PROMPTS_DIR / "tpch_monetdb_optim_pretext.txt",
        {
            "num_queries": num_queries,
            "query_str": query_str,
            "queries_path": queries_path,
        },
    )


def tpch_monetdb_optim_prompt_pretext_optim(bespoke_storage: bool) -> str:
    """生成 optimization problem description prompt，加载 pretext_optim txt 资产."""
    _ = bespoke_storage
    storage_hint = _load_storage_text("storage_layout_allowed.txt")
    return _render_template(
        _TPCH_MONETDB_OPT_BASE_PROMPTS_DIR / "tpch_monetdb_optim_pretext_optim.txt",
        {"storage_layout_hint": f" {storage_hint}"},
    )


def tpch_monetdb_optim_prompt_constraints(allow_storage_changes: bool = True) -> str:
    """加载 TPC-H MonetDB 专用约束文件."""
    return _load_txt(_TPCH_MONETDB_CONSTRAINTS_PATH)


def tpch_monetdb_optim_prompt_pinning(core_id: int) -> str:
    """生成 CPU pinning prompt，加载 pinning txt 资产."""
    return _render_template(
        _TPCH_MONETDB_OPT_BASE_PROMPTS_DIR / "tpch_monetdb_optim_pinning.txt",
        {"core_id": core_id},
    )


def tpch_monetdb_optim_prompt_add_timings() -> str:
    """加载 TPC-H MonetDB 专用 timing instrumentation prompt."""
    return _load_txt(
        _TPCH_MONETDB_OPT_INSTRUMENTATION_PROMPTS_DIR
        / "tpch_monetdb_optim_add_timings_collect_stats.txt"
    )


def tpch_monetdb_optim_prompt_add_timings_per_query(
    qids_str: str, refer_to_prev_queries: bool, scale_factor: float
) -> str:
    """加载 per-query timing 模板并填充变量."""
    refer_to_prev_asset = (
        "align_instrumentation_with_previous_queries.txt"
        if refer_to_prev_queries
        else "no_additional_instrumentation_alignment.txt"
    )
    return _render_template(
        _TPCH_MONETDB_OPT_INSTRUMENTATION_PROMPTS_DIR
        / "tpch_monetdb_optim_add_timings_per_query.txt",
        {
            "qids_str": qids_str,
            "refer_to_prev": _load_storage_text(refer_to_prev_asset),
            "sf": scale_factor,
        },
    )


def load_expert_knowledge() -> str:
    """加载 TPC-H MonetDB 专用 expert knowledge."""
    return _load_txt(_TPCH_MONETDB_EXPERT_KNOWLEDGE_PATH)


def tpch_monetdb_optim_prompt_trace_expert(
    query_id: str,
    constraints_str: str,
    expert_knowledge: str,
    trace_summary: str,
    *,
    query_guidance: str | None = None,
    current_rt_ms: float,
    target_rt_ms: float,
    sf: int,
    storage_is_bespoke: bool,
    hardware_counter_evidence: str = "",
) -> str:
    """Render the active trace-expert optimization prompt."""
    resolved_query_guidance = (
        query_guidance
        if query_guidance is not None
        else build_query_codegen_hint_text(query_id)
    )
    storage_scope = _load_storage_text("storage_change_scope_allowed.txt")
    return _render_template(
        _TPCH_MONETDB_OPT_STAGE_PROMPTS_DIR / "tpch_monetdb_optim_trace_expert.txt",
        {
            "query_id": query_id,
            "constraints": constraints_str,
            "expert_knowledge": expert_knowledge,
            "trace_summary": trace_summary,
            "query_guidance": resolved_query_guidance,
            "target_rt": f"{int(target_rt_ms)}ms",
            "current_rt": f"{int(current_rt_ms)}ms",
            "sf": sf,
            "bespoke_storage_related": storage_scope,
            "hardware_counter_evidence": hardware_counter_evidence,
        },
    )


def tpch_monetdb_optim_prompt_global_human_reference(
    constraints_str: str,
    hotspot_summary_path: str,
    *,
    sf: int,
    storage_is_bespoke: bool,
) -> str:
    """Render the active global convergence prompt."""
    storage_scope = _load_storage_text("global_storage_allowed.txt")
    return _render_template(
        _TPCH_MONETDB_OPT_STAGE_PROMPTS_DIR / "tpch_monetdb_optim_global_human_reference.txt",
        {
            "constraints": constraints_str,
            "hotspot_summary_path": hotspot_summary_path,
            "sf": sf,
            "bespoke_storage_related": storage_scope,
        },
    )


def tpch_monetdb_optim_prompt_global_diagnosis(
    constraints_str: str,
    hotspot_summary_path: str,
    *,
    sf: int,
    trace_evidence: str = "",
    measurement_evidence: str = "",
    objective_evidence: str = "",
) -> str:
    """Render the read-only diagnosis prompt that produces evidence-backed hypotheses.

    This stage must NOT allow code changes. The agent reads evidence bundle,
    builds a system model, and outputs structured hypotheses ranked by evidence.
    """
    return _render_template(
        _TPCH_MONETDB_OPT_STAGE_PROMPTS_DIR / "tpch_monetdb_optim_global_diagnosis.txt",
        {
            "constraints": constraints_str,
            "hotspot_summary_path": hotspot_summary_path,
            "sf": sf,
            "trace_evidence": trace_evidence,
            "measurement_evidence": measurement_evidence,
            "objective_evidence": objective_evidence,
            "q1_q9_checklist": _load_q1_q9_checklist(),
        },
    )


def tpch_monetdb_optim_prompt_hypothesis_execution(
    constraints_str: str,
    hypothesis_json: str,
    *,
    sf: int,
    evidence_refs: str = "",
) -> str:
    """Render the hypothesis-specific implementation prompt.

    Binds one evidence-backed hypothesis as the only execution target.
    Does NOT inject optimization direction, class, or label.
    """
    return _render_template(
        _TPCH_MONETDB_OPT_STAGE_PROMPTS_DIR / "tpch_monetdb_optim_hypothesis_execution.txt",
        {
            "constraints": constraints_str,
            "hypothesis": hypothesis_json,
            "sf": sf,
            "evidence_refs": evidence_refs,
            "q1_q9_checklist": _load_q1_q9_checklist(),
        },
    )
