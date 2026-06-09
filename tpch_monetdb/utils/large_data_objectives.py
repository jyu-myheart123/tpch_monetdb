from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tpch_monetdb.utils.duration_format import safe_speedup
from tpch_monetdb.utils.pipeline_contracts import raise_pipeline_contract_error
from tpch_monetdb.utils.pipeline_evidence import collect_pmu_provenance_failure


WORKLOAD_OBJECTIVE_FILE = "workload_objective.json"
DATA_LAW_CONTRACT_FILE = "data_law_contract.json"
STORAGE_PLAN_CONTRACT_FILE = "storage_plan_contract.json"
STORAGE_PLAN_ALIGNMENT_FILE = "storage_plan_alignment.json"

DEFAULT_CRITICAL_QUERY_IDS: tuple[str, ...] = ("1", "8", "9", "11", "12", "15")
VECTOR_REQUIRED_QUERY_IDS: tuple[str, ...] = ("8", "9", "11", "12", "15")
PMU_REQUIRED_QUERY_IDS: tuple[str, ...] = ("8", "9", "11", "12", "15")
MEASUREMENT_MAX_CV = 0.10
MEASUREMENT_MAX_SPREAD_RATIO = 0.25
REQUIRED_CANDIDATE_COMPARISON_FIELDS: tuple[str, ...] = (
    "query_family_fit",
    "correctness_risk",
    "build_ingest_complexity",
    "vectorization_readiness",
    "memory_locality",
)
REQUIRED_CRITICAL_COST_FIELDS: tuple[tuple[str, ...], ...] = (
    ("predicted_scan_rows", "scan_rows"),
    ("lookup_cost",),
    ("output_rows_cardinality", "output_cardinality", "output_rows"),
    ("cache_locality",),
    ("vectorization_lane_shape", "vectorization_readiness"),
    ("build_memory_cost", "memory_cost", "build_cost"),
    ("risk", "risks"),
)
REQUIRED_OBLIGATION_FIELDS: tuple[str, ...] = (
    "id",
    "file_scope",
    "query_ids",
    "evidence_refs",
)
REQUIRED_CRITICAL_ACCESS_PATH_FIELDS: tuple[tuple[str, ...], ...] = (
    ("semantic_class",),
    ("selected_access_path",),
    ("rejected_access_paths",),
    ("evidence_refs",),
    ("data_law_ids", "data_law_refs", "law_ids"),
    ("access_path_kind", "legality", "legal_basis"),
)
REQUIRED_COMMITTED_LAYOUT_FIELDS: tuple[tuple[str, ...], ...] = (
    ("layout_id", "id", "name"),
    ("summary",),
    ("data_law_ids", "data_law_refs", "law_ids"),
    ("evidence_refs", "evidence"),
    ("selection_rationale", "rationale"),
)
REQUIRED_CRITICAL_ACCESS_PATH_RULES: dict[str, dict[str, object]] = {}
FORBIDDEN_CONTEXT_PREFIX_WORDS: frozenset[str] = frozenset((
    "avoid",
    "avoids",
    "disallow",
    "disallowed",
    "disallows",
    "forbid",
    "forbidden",
    "forbids",
    "never",
    "no",
    "not",
    "prohibit",
    "prohibited",
    "prohibits",
    "reject",
    "rejected",
    "rejects",
    "without",
))
FORBIDDEN_CONTEXT_SUFFIX_WORDS: frozenset[str] = frozenset((
    "disallowed",
    "forbidden",
    "illegal",
    "invalid",
    "prohibited",
    "rejected",
))


@dataclass(frozen=True)
class StoragePlanContractValidation:
    """Validation result for the machine-readable storage-plan contract."""

    status: str
    failures: tuple[str, ...] = ()
    missing_query_ids: tuple[str, ...] = ()
    referenced_data_law_ids: tuple[str, ...] = ()
    selected_candidate_id: str | None = None
    obligation_ids: tuple[str, ...] = ()
    covered_query_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class ObjectiveFailureReport:
    """Large-data objective gate result."""

    failures: tuple[str, ...] = ()
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        """Return True when no objective failure remains."""
        return not self.failures


def _normalize_query_id(query_id: object) -> str:
    """Normalize query IDs so `Q8`, `q8`, and `8` compare identically."""
    value = str(query_id).strip()
    if value.lower().startswith("q"):
        value = value[1:]
    return value


def _read_json_file(path: Path) -> dict[str, Any]:
    """Read a JSON object from disk and fail closed for invalid content."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return data


def load_json_contract(workspace_path: Path, relative_path: str) -> dict[str, Any]:
    """Load one host-owned JSON contract from the workspace."""
    path = workspace_path / relative_path
    if not path.exists():
        return {}
    try:
        return _read_json_file(path)
    except (OSError, json.JSONDecodeError, ValueError):
        return {}


def _has_any_key(payload: dict[str, Any], aliases: tuple[str, ...]) -> bool:
    """Return whether one of the accepted alias fields exists and is non-empty."""
    return any(
        alias in payload and payload.get(alias) not in (None, "", [], {})
        for alias in aliases
    )


def _candidate_id(candidate: dict[str, Any]) -> str:
    """Return the stable ID for one storage candidate."""
    raw_id = (
        candidate.get("id")
        or candidate.get("candidate_id")
        or candidate.get("name")
    )
    if raw_id is None:
        return ""
    return str(raw_id).strip()


def _candidate_comparison(candidate: dict[str, Any]) -> dict[str, Any]:
    """Return comparison fields whether stored directly or under tradeoffs/comparison."""
    comparison = candidate.get("comparison")
    if isinstance(comparison, dict):
        return {**candidate, **comparison}
    tradeoffs = candidate.get("tradeoffs")
    if isinstance(tradeoffs, dict):
        return {**candidate, **tradeoffs}
    return candidate


def _selected_candidate_id(selected_layout: dict[str, Any]) -> str | None:
    """Extract the selected candidate reference from the selected-layout payload."""
    for key in ("candidate_id", "id", "name"):
        value = selected_layout.get(key)
        if value not in (None, ""):
            return str(value).strip()
    return None


def _normalize_query_ids(values: Any) -> tuple[str, ...]:
    """Normalize a query-id list from contract payload fields."""
    if not isinstance(values, list):
        return ()
    return tuple(dict.fromkeys(_normalize_query_id(value) for value in values))


def _is_candidate_aggressive(candidate: dict[str, Any]) -> bool:
    """Return whether a candidate declares an aggressive or hybrid direction."""
    value = " ".join(
        str(candidate.get(key, ""))
        for key in ("layout_kind", "risk_profile", "name", "id", "candidate_id")
    ).lower()
    return any(
        marker in value
        for marker in ("aggressive", "hybrid", "unconventional")
    )


def _is_candidate_conservative(candidate: dict[str, Any]) -> bool:
    """Return whether a candidate declares a conservative direction."""
    value = " ".join(
        str(candidate.get(key, ""))
        for key in ("layout_kind", "risk_profile", "name", "id", "candidate_id")
    ).lower()
    return "conservative" in value


def _collect_obligation_ids(obligations: list[Any]) -> tuple[str, ...]:
    """Collect stable obligation IDs from contract obligation records."""
    ids = [
        str(item.get("id")).strip()
        for item in obligations
        if isinstance(item, dict)
        and item.get("id") is not None
        and str(item.get("id")).strip()
    ]
    return tuple(dict.fromkeys(ids))


def _collect_obligation_query_ids(obligations: list[Any]) -> tuple[str, ...]:
    """Collect query IDs covered by selected-layout obligations."""
    covered: list[str] = []
    for obligation in obligations:
        if not isinstance(obligation, dict):
            continue
        covered.extend(_normalize_query_ids(obligation.get("query_ids")))
    return tuple(dict.fromkeys(covered))


def _missing_required_fields(
    payload: dict[str, Any],
    fields: tuple[str, ...],
) -> tuple[str, ...]:
    """Return required fields that are absent or empty in one payload object."""
    return tuple(field for field in fields if not _has_any_key(payload, (field,)))


def _cost_missing_required_aliases(cost_payload: Any) -> bool:
    """Return True when one critical cost record misses any required cost dimension."""
    if not isinstance(cost_payload, dict):
        return True
    return any(
        not _has_any_key(cost_payload, aliases)
        for aliases in REQUIRED_CRITICAL_COST_FIELDS
    )


def _missing_required_alias_groups(
    payload: Any,
    alias_groups: tuple[tuple[str, ...], ...],
) -> bool:
    """Return True when a payload misses any required alias group."""
    if not isinstance(payload, dict):
        return True
    return any(not _has_any_key(payload, aliases) for aliases in alias_groups)


def _critical_access_path_payloads(
    access_paths: Any,
) -> dict[str, dict[str, Any]]:
    """Normalize critical access-path records keyed by query id."""
    if not isinstance(access_paths, dict):
        return {}
    normalized: dict[str, dict[str, Any]] = {}
    for key, value in access_paths.items():
        if isinstance(value, dict):
            normalized[_normalize_query_id(key)] = value
    return normalized


def _selected_base_candidate_id(payload: dict[str, Any]) -> str | None:
    """Return the selected base candidate id for a committed v2 contract."""
    raw_value = payload.get("selected_base_candidate_id")
    if raw_value not in (None, ""):
        selected_base_candidate_id = str(raw_value).strip()
        if selected_base_candidate_id:
            return selected_base_candidate_id
        return None
    return None


def _normalized_contract_text(value: Any) -> str:
    """Return normalized text for contract rule checks."""
    if isinstance(value, str):
        raw_text = value
    else:
        raw_text = json.dumps(value, default=str, ensure_ascii=False, sort_keys=True)
    normalized = raw_text.lower()
    for separator in ("_", "-", "/", ":", "."):
        normalized = normalized.replace(separator, " ")
    return " ".join(normalized.split())


def _selected_access_path_text(payload: dict[str, Any]) -> str:
    """Return selected-path text while ignoring explicitly rejected evidence."""
    ignored_keys = {
        "data_law_ids",
        "data_law_refs",
        "evidence",
        "evidence_refs",
        "law_ids",
        "rejected_access_paths",
    }
    selected_payload = {
        key: value
        for key, value in payload.items()
        if key not in ignored_keys
    }
    return _normalized_contract_text(selected_payload)


def _has_negated_forbidden_context(text: str, start: int, end: int) -> bool:
    """Return whether a forbidden term occurrence is explicitly negated."""
    prefix_words = text[:start].split()[-4:]
    suffix_words = text[end:].split()[:5]
    suffix_text = " ".join(suffix_words)
    if any(word in FORBIDDEN_CONTEXT_PREFIX_WORDS for word in prefix_words):
        return True
    if any(word in FORBIDDEN_CONTEXT_SUFFIX_WORDS for word in suffix_words):
        return True
    return "not allowed" in suffix_text


def _contains_forbidden_selected_text(text: str, forbidden_terms: tuple[str, ...]) -> bool:
    """Return whether selected-path text contains an unnegated forbidden term."""
    for raw_term in forbidden_terms:
        term = _normalized_contract_text(raw_term)
        if not term:
            continue
        search_from = 0
        while True:
            term_index = text.find(term, search_from)
            if term_index < 0:
                break
            term_end = term_index + len(term)
            if not _has_negated_forbidden_context(text, term_index, term_end):
                return True
            search_from = term_end
    return False


def _required_critical_access_path_failures(
    access_path_payloads: dict[str, dict[str, Any]],
    required_critical: list[str],
) -> tuple[str, ...]:
    """Return failures for critical queries with mandatory access-path rules."""
    failures: list[str] = []
    for qid in required_critical:
        rule = REQUIRED_CRITICAL_ACCESS_PATH_RULES.get(qid)
        payload = access_path_payloads.get(qid)
        if rule is None or payload is None:
            continue
        selected_text = _normalized_contract_text(payload.get("selected_access_path", ""))
        obligation_id = str(rule["obligation_id"])
        payload_text = _selected_access_path_text(payload)
        required_terms = tuple(str(item) for item in rule["required_any_text"])
        forbidden_terms = tuple(str(item) for item in rule.get("forbidden_selected_text", ()))
        if forbidden_terms and _contains_forbidden_selected_text(payload_text, forbidden_terms):
            failures.append(str(rule["forbidden_failure"]))
        if _normalized_contract_text(obligation_id) not in selected_text or not any(
            _normalized_contract_text(term) in payload_text for term in required_terms
        ):
            failures.append(str(rule["failure"]))
    return tuple(dict.fromkeys(failures))


def _query_cost_payload(query_costs: dict[str, Any], query_id: str) -> Any:
    """Return one query-cost payload while accepting Q-prefixed contract keys."""
    for key, value in query_costs.items():
        if _normalize_query_id(key) == query_id:
            return value
    return None


def build_workload_objective_payload(
    *,
    query_ids: list[str],
    benchmark_sf: int,
    large_sf: int | None,
    hardware_counter_backend: str | None,
    target_cpu: str | None,
    benchmark: str = "tpch",
) -> dict[str, Any]:
    """Build the host-owned large-data objective for one benchmark run."""
    normalized_benchmark = benchmark.strip().lower()
    if normalized_benchmark != "tpch":
        raise ValueError(f"Unsupported benchmark for workload objective: {benchmark}")
    normalized_query_ids = tuple(dict.fromkeys(_normalize_query_id(qid) for qid in query_ids))
    critical_query_ids = tuple(
        qid for qid in DEFAULT_CRITICAL_QUERY_IDS if qid in normalized_query_ids
    )
    critical_targets = {
        qid: _critical_query_target(
            qid,
            hardware_counter_backend=hardware_counter_backend,
        )
        for qid in critical_query_ids
    }
    return {
        "version": 1,
        "objective_id": _objective_id_for_benchmark(normalized_benchmark),
        "benchmark": normalized_benchmark,
        "query_ids": list(normalized_query_ids),
        "critical_query_ids": list(critical_query_ids),
        "benchmark_scale_factor": int(benchmark_sf),
        "large_scale_factor": int(large_sf) if large_sf is not None else int(benchmark_sf),
        "measurement_policy": {
            "min_repetitions_when_large_sf": 3,
            "max_cv": MEASUREMENT_MAX_CV,
            "max_spread_ratio": MEASUREMENT_MAX_SPREAD_RATIO,
        },
        "critical_query_targets": critical_targets,
        "hardware_counter_policy": {
            "backend": hardware_counter_backend,
            "target_cpu": target_cpu,
            "fail_closed_when_required": True,
        },
        "required_artifacts": [
            WORKLOAD_OBJECTIVE_FILE,
            DATA_LAW_CONTRACT_FILE,
            STORAGE_PLAN_CONTRACT_FILE,
            STORAGE_PLAN_ALIGNMENT_FILE,
        ],
    }


def _objective_id_for_benchmark(benchmark: str) -> str:
    """Return the objective id used by benchmark-specific control artifacts."""
    normalized = benchmark.strip().lower()
    if normalized == "tpch":
        return "tpch-docker-monetdb-objective-v1"
    raise ValueError(f"Unsupported benchmark for workload objective: {benchmark}")


def _critical_query_target(
    qid: str,
    *,
    hardware_counter_backend: str | None,
) -> dict[str, Any]:
    """Return the declarative target for one critical query."""
    target: dict[str, Any] = {
        "min_speedup_vs_baseline": 1.0,
        "max_runtime_regression_ratio": 0.0,
        "requires_vectorization": qid in VECTOR_REQUIRED_QUERY_IDS,
        "requires_pmu": bool(hardware_counter_backend) and qid in PMU_REQUIRED_QUERY_IDS,
    }
    if qid == "9":
        target["min_speedup_vs_base_impl"] = 1.05
        target["max_runtime_regression_ratio_vs_base_impl"] = 0.0
    return target


def write_workload_objective(
    *,
    workspace_path: Path,
    query_ids: list[str],
    benchmark_sf: int,
    large_sf: int | None,
    hardware_counter_backend: str | None,
    target_cpu: str | None,
    benchmark: str = "tpch",
) -> Path:
    """Write the host-owned workload objective contract."""
    workspace_path.mkdir(parents=True, exist_ok=True)
    payload = build_workload_objective_payload(
        query_ids=query_ids,
        benchmark_sf=benchmark_sf,
        large_sf=large_sf,
        hardware_counter_backend=hardware_counter_backend,
        target_cpu=target_cpu,
        benchmark=benchmark,
    )
    target_path = workspace_path / WORKLOAD_OBJECTIVE_FILE
    target_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return target_path


def build_data_law_contract_payload(
    *,
    query_ids: list[str],
    benchmark_sf: int,
    design_evidence_path: Path,
    benchmark: str = "tpch",
) -> dict[str, Any]:
    """Build host-owned data-law facts from generated design evidence."""
    source_available = design_evidence_path.exists() and design_evidence_path.stat().st_size > 0
    normalized_query_ids = tuple(dict.fromkeys(_normalize_query_id(qid) for qid in query_ids))
    normalized_benchmark = benchmark.strip().lower()
    if normalized_benchmark != "tpch":
        raise ValueError(f"Unsupported benchmark for data-law contract: {benchmark}")
    laws = _build_tpch_data_laws(
        normalized_query_ids=normalized_query_ids,
        source_available=source_available,
    )
    contract_id = "tpch-data-law-contract-v1"
    return {
        "version": 1,
        "contract_id": contract_id,
        "benchmark": normalized_benchmark,
        "benchmark_scale_factor": int(benchmark_sf),
        "source_artifact": "design_evidence.md",
        "source_available": source_available,
        "laws": laws,
    }


def _build_tpch_data_laws(
    *,
    normalized_query_ids: tuple[str, ...],
    source_available: bool,
) -> list[dict[str, Any]]:
    """Return TPC-H data-law records for Dockerized MonetDB replacement runs."""
    confidence = "evidence_observed" if source_available else "blocking_missing_source"
    return [
        {
            "law_id": "LAW_TPCH_TABLE_CARDINALITY",
            "source": "design_evidence.md#TPC-H Table Cardinality",
            "confidence": confidence,
            "applies_to": list(normalized_query_ids),
            "rule": "storage and runtime plans must cite table-level row-count pressure for every hot query path",
        },
        {
            "law_id": "LAW_TPCH_JOIN_GRAPH",
            "source": "design_evidence.md#TPC-H Query Join Graph",
            "confidence": confidence,
            "applies_to": list(normalized_query_ids),
            "rule": "join-order, key-layout, and side-structure claims must cite the query tables and join/filter features",
        },
        {
            "law_id": "LAW_TPCH_OUTPUT_ORDERING",
            "source": "design_evidence.md#TPC-H Output Semantics",
            "confidence": confidence,
            "applies_to": list(normalized_query_ids),
            "rule": "ordered result queries must preserve declared ORDER BY semantics before performance claims count",
        },
        {
            "law_id": "LAW_TPCH_NUMERIC_TOLERANCE",
            "source": "design_evidence.md#TPC-H Output Semantics",
            "confidence": confidence,
            "applies_to": list(normalized_query_ids),
            "rule": "decimal and floating-point comparisons use the TPC-H validator tolerance policy rather than exact string equality",
        },
        {
            "law_id": "LAW_TPCH_RUNTIME_BOUNDARY",
            "source": "design_evidence.md#Docker MonetDB Runtime Boundary",
            "confidence": confidence,
            "applies_to": list(normalized_query_ids),
            "rule": "baseline evidence comes from Dockerized MonetDB native/MAPI execution, not removed HTTP or query-file runner paths",
        },
    ]


def write_data_law_contract(
    *,
    workspace_path: Path,
    query_ids: list[str],
    benchmark_sf: int,
    design_evidence_path: Path,
    benchmark: str = "tpch",
) -> Path:
    """Write the host-owned data-law contract."""
    workspace_path.mkdir(parents=True, exist_ok=True)
    payload = build_data_law_contract_payload(
        query_ids=query_ids,
        benchmark_sf=benchmark_sf,
        design_evidence_path=design_evidence_path,
        benchmark=benchmark,
    )
    target_path = workspace_path / DATA_LAW_CONTRACT_FILE
    target_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return target_path


def _validate_committed_storage_plan_contract_payload(
    payload: dict[str, Any],
    *,
    query_ids: list[str],
    data_law_contract: dict[str, Any] | None = None,
) -> StoragePlanContractValidation:
    """Validate a v2 committed storage-plan contract for implementation stages."""
    failures: list[str] = []
    if payload.get("candidate_layouts") not in (None, [], {}):
        failures.append("STORAGE_PLAN_CONTRACT_COMMITTED_HAS_CANDIDATES")
    if payload.get("candidates") not in (None, [], {}):
        failures.append("STORAGE_PLAN_CONTRACT_COMMITTED_HAS_CANDIDATES")

    selected_candidate_id = _selected_base_candidate_id(payload)
    if selected_candidate_id is None:
        failures.append("STORAGE_PLAN_CONTRACT_SELECTED_BASE_CANDIDATE_MISSING")

    committed_layout = payload.get("committed_layout")
    if _missing_required_alias_groups(committed_layout, REQUIRED_COMMITTED_LAYOUT_FIELDS):
        failures.append("STORAGE_PLAN_CONTRACT_COMMITTED_LAYOUT_FIELDS_MISSING")

    refinement_decisions = payload.get("refinement_decisions")
    if not isinstance(refinement_decisions, list) or not refinement_decisions:
        failures.append("STORAGE_PLAN_CONTRACT_REFINEMENT_DECISIONS_MISSING")

    obligations = payload.get("selected_layout_obligations")
    if not isinstance(obligations, list) or not obligations:
        failures.append("STORAGE_PLAN_CONTRACT_OBLIGATIONS_MISSING")
        obligations = []
    obligation_dicts = [item for item in obligations if isinstance(item, dict)]
    if any(
        _missing_required_fields(obligation, REQUIRED_OBLIGATION_FIELDS)
        for obligation in obligation_dicts
    ) or len(obligation_dicts) != len(obligations):
        failures.append("STORAGE_PLAN_CONTRACT_OBLIGATION_FIELDS_MISSING")

    query_costs = payload.get("query_family_costs") or payload.get("query_costs")
    if not isinstance(query_costs, dict):
        failures.append("STORAGE_PLAN_CONTRACT_QUERY_COSTS_MISSING")
        query_costs = {}
    normalized_cost_keys = {_normalize_query_id(key) for key in query_costs.keys()}
    required_critical = [
        qid for qid in DEFAULT_CRITICAL_QUERY_IDS
        if qid in {_normalize_query_id(item) for item in query_ids}
    ]
    missing_cost_query_ids = tuple(
        qid for qid in required_critical if qid not in normalized_cost_keys
    )
    if missing_cost_query_ids:
        failures.append("STORAGE_PLAN_CONTRACT_CRITICAL_COSTS_MISSING")
    if any(
        _cost_missing_required_aliases(_query_cost_payload(query_costs, qid))
        for qid in required_critical
        if qid in normalized_cost_keys
    ):
        failures.append("STORAGE_PLAN_CONTRACT_CRITICAL_COST_FIELDS_MISSING")

    access_path_payloads = _critical_access_path_payloads(
        payload.get("critical_query_access_paths")
    )
    if not access_path_payloads:
        failures.append("STORAGE_PLAN_CONTRACT_CRITICAL_ACCESS_PATHS_MISSING")
    missing_access_path_query_ids = tuple(
        qid for qid in required_critical if qid not in set(access_path_payloads)
    )
    if missing_access_path_query_ids:
        failures.append("STORAGE_PLAN_CONTRACT_CRITICAL_ACCESS_PATH_QUERY_COVERAGE_MISSING")
    if any(
        _missing_required_alias_groups(
            access_path_payloads.get(qid),
            REQUIRED_CRITICAL_ACCESS_PATH_FIELDS,
        )
        for qid in required_critical
        if qid in set(access_path_payloads)
    ):
        failures.append("STORAGE_PLAN_CONTRACT_CRITICAL_ACCESS_PATH_FIELDS_MISSING")
    if any(
        access_path_payloads[qid].get("not_query_answer_cache") is not True
        for qid in required_critical
        if qid in set(access_path_payloads)
    ):
        failures.append("STORAGE_PLAN_CONTRACT_CRITICAL_ACCESS_PATH_CACHE_BOUNDARY_MISSING")
    failures.extend(
        _required_critical_access_path_failures(
            access_path_payloads,
            required_critical,
        )
    )

    covered_query_ids = tuple(dict.fromkeys(
        (*_collect_obligation_query_ids(obligation_dicts), *access_path_payloads.keys())
    ))
    missing_obligation_query_ids = tuple(
        qid for qid in required_critical if qid not in set(covered_query_ids)
    )
    if missing_obligation_query_ids:
        failures.append("STORAGE_PLAN_CONTRACT_OBLIGATION_QUERY_COVERAGE_MISSING")
    referenced_laws = _extract_data_law_refs(payload)
    if not referenced_laws:
        failures.append("STORAGE_PLAN_CONTRACT_DATA_LAW_REFS_MISSING")
    if data_law_contract:
        known_laws = {
            str(item.get("law_id"))
            for item in data_law_contract.get("laws", [])
            if isinstance(item, dict) and item.get("law_id")
        }
        if known_laws and not set(referenced_laws).issubset(known_laws):
            failures.append("STORAGE_PLAN_CONTRACT_UNKNOWN_DATA_LAW_REFS")

    missing_query_ids = tuple(dict.fromkeys((
        *missing_cost_query_ids,
        *missing_access_path_query_ids,
        *missing_obligation_query_ids,
    )))
    status = "valid" if not failures else "invalid"
    return StoragePlanContractValidation(
        status=status,
        failures=tuple(dict.fromkeys(failures)),
        missing_query_ids=missing_query_ids,
        referenced_data_law_ids=tuple(dict.fromkeys(referenced_laws)),
        selected_candidate_id=selected_candidate_id,
        obligation_ids=_collect_obligation_ids(obligation_dicts),
        covered_query_ids=covered_query_ids,
    )


def validate_storage_plan_contract_payload(
    payload: dict[str, Any],
    *,
    query_ids: list[str],
    data_law_contract: dict[str, Any] | None = None,
) -> StoragePlanContractValidation:
    """Validate the machine-readable storage plan contract."""
    failures: list[str] = []
    contract_version = str(payload.get("version"))
    if contract_version not in ("1", "2"):
        failures.append("STORAGE_PLAN_CONTRACT_VERSION_INVALID")
    if contract_version == "2":
        committed_validation = _validate_committed_storage_plan_contract_payload(
            payload,
            query_ids=query_ids,
            data_law_contract=data_law_contract,
        )
        if not failures:
            return committed_validation
        return StoragePlanContractValidation(
            status="invalid",
            failures=tuple(dict.fromkeys((*failures, *committed_validation.failures))),
            missing_query_ids=committed_validation.missing_query_ids,
            referenced_data_law_ids=committed_validation.referenced_data_law_ids,
            selected_candidate_id=committed_validation.selected_candidate_id,
            obligation_ids=committed_validation.obligation_ids,
            covered_query_ids=committed_validation.covered_query_ids,
        )
    candidates = payload.get("candidate_layouts") or payload.get("candidates")
    if not isinstance(candidates, list) or len(candidates) < 3:
        failures.append("STORAGE_PLAN_CONTRACT_CANDIDATES_INCOMPLETE")
        candidates = []
    candidate_dicts = [item for item in candidates if isinstance(item, dict)]
    if len(candidate_dicts) != len(candidates):
        failures.append("STORAGE_PLAN_CONTRACT_CANDIDATE_FIELDS_MISSING")
    candidate_ids = tuple(_candidate_id(candidate) for candidate in candidate_dicts)
    nonempty_candidate_ids = tuple(candidate_id for candidate_id in candidate_ids if candidate_id)
    if len(nonempty_candidate_ids) != len(candidate_dicts):
        failures.append("STORAGE_PLAN_CONTRACT_CANDIDATE_IDS_MISSING")
    if len(set(nonempty_candidate_ids)) != len(nonempty_candidate_ids):
        failures.append("STORAGE_PLAN_CONTRACT_CANDIDATE_IDS_DUPLICATE")
    if candidate_dicts and (
        not any(_is_candidate_conservative(candidate) for candidate in candidate_dicts)
        or not any(_is_candidate_aggressive(candidate) for candidate in candidate_dicts)
    ):
        failures.append("STORAGE_PLAN_CONTRACT_CANDIDATES_NOT_MATERIAL")
    if any(
        _missing_required_fields(
            _candidate_comparison(candidate),
            REQUIRED_CANDIDATE_COMPARISON_FIELDS,
        )
        for candidate in candidate_dicts
    ):
        failures.append("STORAGE_PLAN_CONTRACT_CANDIDATE_COMPARISON_MISSING")
    if any(
        not _has_any_key(candidate, ("data_law_ids", "data_law_refs", "law_ids"))
        or not _has_any_key(candidate, ("evidence_refs", "evidence"))
        for candidate in candidate_dicts
    ):
        failures.append("STORAGE_PLAN_CONTRACT_CANDIDATE_EVIDENCE_MISSING")
    selected_layout = payload.get("selected_layout")
    if not isinstance(selected_layout, dict) or not selected_layout:
        failures.append("STORAGE_PLAN_CONTRACT_SELECTED_LAYOUT_MISSING")
        selected_layout = {}
    selected_candidate_id = _selected_candidate_id(selected_layout)
    if selected_candidate_id is None:
        failures.append("STORAGE_PLAN_CONTRACT_SELECTED_CANDIDATE_MISSING")
    elif nonempty_candidate_ids and selected_candidate_id not in set(nonempty_candidate_ids):
        failures.append("STORAGE_PLAN_CONTRACT_SELECTED_CANDIDATE_UNKNOWN")
    if not _has_any_key(selected_layout, ("selection_rationale", "rationale")):
        failures.append("STORAGE_PLAN_CONTRACT_SELECTION_RATIONALE_MISSING")
    obligations = payload.get("selected_layout_obligations")
    if obligations is None and isinstance(selected_layout, dict):
        obligations = selected_layout.get("obligations")
    if not isinstance(obligations, list) or not obligations:
        failures.append("STORAGE_PLAN_CONTRACT_OBLIGATIONS_MISSING")
        obligations = []
    obligation_dicts = [item for item in obligations if isinstance(item, dict)]
    if any(
        _missing_required_fields(obligation, REQUIRED_OBLIGATION_FIELDS)
        for obligation in obligation_dicts
    ) or len(obligation_dicts) != len(obligations):
        failures.append("STORAGE_PLAN_CONTRACT_OBLIGATION_FIELDS_MISSING")
    query_costs = payload.get("query_family_costs") or payload.get("query_costs")
    if not isinstance(query_costs, dict):
        failures.append("STORAGE_PLAN_CONTRACT_QUERY_COSTS_MISSING")
        query_costs = {}
    normalized_cost_keys = {_normalize_query_id(key) for key in query_costs.keys()}
    required_critical = [
        qid for qid in DEFAULT_CRITICAL_QUERY_IDS
        if qid in {_normalize_query_id(item) for item in query_ids}
    ]
    missing_query_ids = tuple(qid for qid in required_critical if qid not in normalized_cost_keys)
    if missing_query_ids:
        failures.append("STORAGE_PLAN_CONTRACT_CRITICAL_COSTS_MISSING")
    if any(
        _cost_missing_required_aliases(_query_cost_payload(query_costs, qid))
        for qid in required_critical
        if qid in normalized_cost_keys
    ):
        failures.append("STORAGE_PLAN_CONTRACT_CRITICAL_COST_FIELDS_MISSING")
    covered_query_ids = _collect_obligation_query_ids(obligation_dicts)
    missing_obligation_query_ids = tuple(
        qid for qid in required_critical if qid not in set(covered_query_ids)
    )
    if missing_obligation_query_ids:
        failures.append("STORAGE_PLAN_CONTRACT_OBLIGATION_QUERY_COVERAGE_MISSING")
    referenced_laws = _extract_data_law_refs(payload)
    if not referenced_laws:
        failures.append("STORAGE_PLAN_CONTRACT_DATA_LAW_REFS_MISSING")
    if data_law_contract:
        known_laws = {
            str(item.get("law_id"))
            for item in data_law_contract.get("laws", [])
            if isinstance(item, dict) and item.get("law_id")
        }
        if known_laws and not set(referenced_laws).issubset(known_laws):
            failures.append("STORAGE_PLAN_CONTRACT_UNKNOWN_DATA_LAW_REFS")
    status = "valid" if not failures else "invalid"
    return StoragePlanContractValidation(
        status=status,
        failures=tuple(dict.fromkeys(failures)),
        missing_query_ids=missing_query_ids,
        referenced_data_law_ids=tuple(dict.fromkeys(referenced_laws)),
        selected_candidate_id=selected_candidate_id,
        obligation_ids=_collect_obligation_ids(obligation_dicts),
        covered_query_ids=covered_query_ids,
    )


def validate_storage_plan_contract(
    workspace_path: Path,
    *,
    query_ids: list[str],
) -> StoragePlanContractValidation:
    """Load and validate storage_plan_contract.json from the workspace."""
    contract_path = workspace_path / STORAGE_PLAN_CONTRACT_FILE
    if not contract_path.exists():
        return StoragePlanContractValidation(
            status="missing",
            failures=("STORAGE_PLAN_CONTRACT_MISSING",),
        )
    try:
        payload = _read_json_file(contract_path)
    except (OSError, json.JSONDecodeError, ValueError):
        return StoragePlanContractValidation(
            status="invalid",
            failures=("STORAGE_PLAN_CONTRACT_INVALID_JSON",),
        )
    data_law_contract = load_json_contract(workspace_path, DATA_LAW_CONTRACT_FILE)
    return validate_storage_plan_contract_payload(
        payload,
        query_ids=query_ids,
        data_law_contract=data_law_contract,
    )


def build_storage_plan_alignment(
    workspace_path: Path,
    *,
    query_ids: list[str],
) -> dict[str, Any]:
    """Build evaluated storage-plan alignment; never return a placeholder."""
    if not (workspace_path / "storage_plan.txt").exists():
        return {"status": "missing", "departures": ["storage_plan.txt missing"]}
    validation = validate_storage_plan_contract(workspace_path, query_ids=query_ids)
    requested_critical_query_ids = [
        qid for qid in DEFAULT_CRITICAL_QUERY_IDS
        if qid in {_normalize_query_id(item) for item in query_ids}
    ]
    missing_obligation_query_ids = [
        qid for qid in requested_critical_query_ids
        if qid not in set(validation.covered_query_ids)
    ]
    base_payload = {
        "departures": list(validation.failures),
        "missing_query_ids": list(validation.missing_query_ids),
        "referenced_data_law_ids": list(validation.referenced_data_law_ids),
        "selected_candidate_id": validation.selected_candidate_id,
        "selected_layout_obligation_ids": list(validation.obligation_ids),
        "covered_critical_query_ids": [
            qid for qid in requested_critical_query_ids
            if qid in set(validation.covered_query_ids)
        ],
        "missing_obligation_query_ids": missing_obligation_query_ids,
    }
    if validation.status != "valid":
        return {"status": validation.status, **base_payload}
    manifest_path = workspace_path / "implementation_manifest.json"
    if not manifest_path.exists():
        status = "contract_valid"
    else:
        status = "aligned"
    return {
        "status": status,
        **base_payload,
        "departures": [],
        "missing_query_ids": [],
        "missing_obligation_query_ids": [],
    }


def write_storage_plan_alignment(
    workspace_path: Path,
    *,
    query_ids: list[str],
) -> Path:
    """Write evaluated storage-plan alignment as a tracked host artifact."""
    workspace_path.mkdir(parents=True, exist_ok=True)
    payload = build_storage_plan_alignment(workspace_path, query_ids=query_ids)
    target_path = workspace_path / STORAGE_PLAN_ALIGNMENT_FILE
    target_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return target_path


def collect_large_data_failures(summary: Any) -> tuple[str, ...]:
    """Return blocking large-data objective failures for a run summary."""
    stored_failures = getattr(summary, "objective_failures", None)
    if stored_failures:
        return tuple(str(failure) for failure in stored_failures)
    report = build_objective_failure_report(summary)
    return report.failures


def build_objective_failure_report(summary: Any) -> ObjectiveFailureReport:
    """Evaluate critical-query, contract, vectorization, PMU, and noise gates."""
    failures: list[str] = []
    details: dict[str, Any] = {}
    objective = dict(getattr(summary, "workload_objective", {}) or {})
    if not objective:
        failures.append("WORKLOAD_OBJECTIVE_MISSING")
        return ObjectiveFailureReport(failures=tuple(failures), details=details)

    critical_query_ids = [
        _normalize_query_id(qid)
        for qid in objective.get("critical_query_ids", [])
    ]
    target_map = dict(objective.get("critical_query_targets", {}) or {})
    final_runtime = dict(getattr(summary, "final_runtime_ms_by_query", {}) or {})
    baseline_runtime = dict(getattr(summary, "baseline_runtime_ms_by_query", {}) or {})
    for qid in critical_query_ids:
        target = dict(target_map.get(qid, {}) or {})
        min_speedup = float(target.get("min_speedup_vs_baseline", 1.0))
        final_ms = final_runtime.get(qid)
        baseline_ms = baseline_runtime.get(qid)
        if not _is_positive_finite_number(final_ms) or not _is_positive_finite_number(baseline_ms):
            failures.append("CRITICAL_QUERY_RUNTIME_MISSING")
            details.setdefault("critical_runtime_missing", []).append(qid)
            continue
        speedup = safe_speedup(baseline_ms, final_ms)
        if speedup is None:
            failures.append("CRITICAL_QUERY_RUNTIME_MISSING")
            details.setdefault("critical_runtime_missing", []).append(qid)
            continue
        if speedup < min_speedup:
            failures.append("CRITICAL_QUERY_TARGET_MISSED")
            details.setdefault("critical_query_target_missed", {})[qid] = {
                "speedup": speedup,
                "required": min_speedup,
            }
        min_base_impl_speedup = target.get("min_speedup_vs_base_impl")
        if min_base_impl_speedup is not None and speedup < float(min_base_impl_speedup):
            failures.append("CRITICAL_QUERY_BASE_IMPL_TARGET_MISSED")
            details.setdefault("critical_query_base_impl_target_missed", {})[qid] = {
                "speedup": speedup,
                "required": float(min_base_impl_speedup),
            }

    alignment = dict(getattr(summary, "storage_plan_alignment", {}) or {})
    if alignment.get("status") in (None, "", "missing", "not_evaluated", "invalid"):
        failures.append("STORAGE_PLAN_ALIGNMENT_NOT_EVALUATED")
        details["storage_plan_alignment"] = alignment

    control_hashes = dict(getattr(summary, "control_artifact_hashes", {}) or {})
    for artifact in objective.get("required_artifacts", []):
        if artifact != STORAGE_PLAN_ALIGNMENT_FILE and artifact not in control_hashes:
            failures.append("CONTROL_ARTIFACT_HASH_MISSING")
            details.setdefault("missing_control_hashes", []).append(artifact)

    measurement_repetition = dict(getattr(summary, "measurement_repetition", {}) or {})
    noise_failure = _collect_measurement_noise_failure(measurement_repetition, objective)
    if noise_failure is not None:
        failures.append(noise_failure)

    vector_summary = dict(getattr(summary, "compiler_vectorization_summary", {}) or {})
    for qid in critical_query_ids:
        target = dict(target_map.get(qid, {}) or {})
        if target.get("requires_vectorization") is True:
            query_vector_summary = _summary_for_query(vector_summary, qid)
            if not _has_hot_loop_vectorization(query_vector_summary):
                failures.append("VECTOR_HOT_LOOP_NOT_OPTIMIZED")
                details.setdefault("vectorization_missing", []).append(qid)

    hardware_summary = dict(getattr(summary, "hardware_counter_summary", {}) or {})
    for qid in critical_query_ids:
        target = dict(target_map.get(qid, {}) or {})
        if target.get("requires_pmu") is True:
            query_hardware_summary = _summary_for_query(hardware_summary, qid)
            if query_hardware_summary.get("hardware_counters_available") is not True:
                failures.append("PMU_REQUIRED_BUT_MISSING")
                details.setdefault("pmu_missing", []).append(qid)
                continue
            hotspot_provenance = dict(
                query_hardware_summary.get("perf_hotspot_provenance") or {}
            )
            pmu_failure = collect_pmu_provenance_failure(
                query_id=qid,
                hotspot_provenance=hotspot_provenance,
                details=details,
            )
            if pmu_failure is not None:
                failures.append(pmu_failure)
                continue
            if query_hardware_summary.get("perf_hotspots_available") is not True:
                failures.append("PMU_HOTSPOT_MISSING")
                details.setdefault("pmu_hotspot_missing", []).append(qid)

    return ObjectiveFailureReport(
        failures=tuple(dict.fromkeys(failures)),
        details=details,
    )


def is_large_data_success(summary: Any) -> bool:
    """Return whether the run satisfies the new large-data objective."""
    return not collect_large_data_failures(summary)


def require_large_data_success(summary: Any, *, stage: str | None = None) -> None:
    """Raise a structured error when large-data objective gates fail."""
    failures = collect_large_data_failures(summary)
    if not failures:
        return None
    raise_pipeline_contract_error(
        code=failures[0],
        message="Large-data objective failed: " + ", ".join(failures),
        stage=stage,
    )
    return None


def classify_objective_failure_route(failures: tuple[str, ...]) -> str:
    """Map objective failures to the next pipeline route."""
    route_by_failure = (
        (("STORAGE_PLAN", "CONTROL_ARTIFACT", "DATA_LAW"), "storage_plan"),
        (("VECTOR",), "vectorization"),
        (("PMU", "MEASUREMENT", "RUNTIME"), "instrumentation"),
        (("FORBIDDEN_INSTRUMENTED", "FINAL_PATH"), "instrumentation"),
        (("CRITICAL_QUERY",), "optimization"),
    )
    for prefixes, route in route_by_failure:
        if any(failure.startswith(prefix) for prefix in prefixes for failure in failures):
            return route
    return "optimization"


def build_hot_loop_mapping(
    *,
    query_id: str,
    compiler_summary: dict[str, Any],
    trace_summary_text: str,
) -> dict[str, Any]:
    """Map compiler vectorization evidence to the current query hot path."""
    optimized_sites = (
        compiler_summary.get("workspace_optimized_loop_sites")
        or compiler_summary.get("optimized_loop_sites")
        or []
    )
    if not isinstance(optimized_sites, list) or not optimized_sites:
        return {"status": "missing", "matched_sites": []}
    normalized_query = _normalize_query_id(query_id)
    trace_text = trace_summary_text.lower()
    expected_sources = _expected_hot_loop_sources(normalized_query)
    expected_tokens = _expected_trace_tokens(normalized_query)
    matched_sites: list[dict[str, Any]] = []
    candidate_sites: list[dict[str, Any]] = []
    for site in optimized_sites:
        if not isinstance(site, dict):
            continue
        source_file = str(site.get("file", "")).lower()
        basename = Path(source_file).name
        source_hit = basename in expected_sources or f"query_q{normalized_query}" in source_file
        trace_hit = any(token in trace_text for token in expected_tokens)
        if source_hit:
            candidate_sites.append(site)
        if source_hit and trace_hit:
            matched_sites.append(site)
    status = "matched" if matched_sites else "unmatched"
    return {
        "status": status,
        "matched_sites": matched_sites,
        "candidate_sites": candidate_sites,
        "expected_sources": sorted(expected_sources),
        "expected_trace_tokens": sorted(expected_tokens),
    }


def _expected_hot_loop_sources(query_id: str) -> set[str]:
    """Return generated source files expected to contain the query hot loop."""
    family_sources = {
        "1": {"query_q1.cpp"},
        "2": {"query_q2.cpp"},
        "3": {"query_family_single_groupby.cpp"},
        "4": {"query_family_single_groupby.cpp"},
        "5": {"query_family_single_groupby.cpp"},
        "6": {"query_family_single_groupby.cpp"},
        "7": {"query_family_single_groupby.cpp"},
        "8": {"query_family_cpu_max_groupby.cpp"},
        "9": {"query_family_double_groupby.cpp"},
        "10": {"query_family_cpu_max_groupby.cpp"},
        "11": {"query_family_high_cpu_groupby.cpp"},
        "12": {"query_family_high_cpu_groupby.cpp"},
        "13": {"query_family_double_groupby.cpp"},
        "14": {"query_family_double_groupby.cpp"},
        "15": {"query_q15.cpp"},
    }
    return family_sources.get(query_id, {f"query_q{query_id}.cpp"})


def _expected_trace_tokens(query_id: str) -> set[str]:
    """Return trace tokens that prove the optimized site is on the active query path."""
    tokens = {f"query_q{query_id}"}
    for source in _expected_hot_loop_sources(query_id):
        stem = source.rsplit(".", 1)[0]
        tokens.add(stem)
        tokens.add(stem.replace("query_family_", ""))
    return tokens


def _extract_data_law_refs(payload: dict[str, Any]) -> tuple[str, ...]:
    """Extract all data-law references from common contract field names."""
    refs: list[str] = []
    stack: list[Any] = [payload]
    key_names = {"data_law_ids", "data_law_refs", "referenced_data_law_ids", "law_ids"}
    while stack:
        value = stack.pop()
        if isinstance(value, dict):
            for key, nested in value.items():
                if key in key_names and isinstance(nested, list):
                    refs.extend(str(item) for item in nested if str(item).strip())
                else:
                    stack.append(nested)
        elif isinstance(value, list):
            stack.extend(value)
    return tuple(dict.fromkeys(refs))


def _summary_for_query(summary_by_query: dict[str, Any], query_id: str) -> dict[str, Any]:
    """Return query-specific summary payload from either per-query or flat maps."""
    if query_id in summary_by_query and isinstance(summary_by_query[query_id], dict):
        return dict(summary_by_query[query_id])
    q_prefixed = f"Q{query_id}"
    if q_prefixed in summary_by_query and isinstance(summary_by_query[q_prefixed], dict):
        return dict(summary_by_query[q_prefixed])
    return summary_by_query


def _has_hot_loop_vectorization(query_vector_summary: dict[str, Any]) -> bool:
    """Return True only when vectorization evidence is tied to the hot loop."""
    if query_vector_summary.get("vectorization_applied") is not True:
        return False
    optimized_sites = (
        query_vector_summary.get("workspace_optimized_loop_sites")
        or query_vector_summary.get("optimized_loop_sites")
        or []
    )
    if not optimized_sites:
        return False
    hot_loop_mapping = query_vector_summary.get("hot_loop_mapping") or {}
    if isinstance(hot_loop_mapping, dict):
        return hot_loop_mapping.get("status") == "matched"
    return False


def _collect_measurement_noise_failure(
    measurement_repetition: dict[str, Any],
    objective: dict[str, Any],
) -> str | None:
    """Return measurement noise failure code when large-SF samples are unstable."""
    samples = measurement_repetition.get("aggregate_runtime_ms_samples")
    if not isinstance(samples, list) or len(samples) < 2:
        return "MEASUREMENT_REPETITION_MISSING"
    numeric_samples = [float(sample) for sample in samples if _is_positive_finite_number(sample)]
    if len(numeric_samples) != len(samples):
        return "MEASUREMENT_UNSTABLE"
    mean = sum(numeric_samples) / len(numeric_samples)
    if mean <= 0:
        return "MEASUREMENT_UNSTABLE"
    cv = statistics_stdev(numeric_samples) / mean
    spread_ratio = (max(numeric_samples) - min(numeric_samples)) / mean
    policy = dict(objective.get("measurement_policy", {}) or {})
    max_cv = float(policy.get("max_cv", MEASUREMENT_MAX_CV))
    max_spread = float(policy.get("max_spread_ratio", MEASUREMENT_MAX_SPREAD_RATIO))
    if cv > max_cv or spread_ratio > max_spread:
        return "MEASUREMENT_UNSTABLE"
    return None


def statistics_stdev(values: list[float]) -> float:
    """Compute population standard deviation without importing statistics in hot paths."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return math.sqrt(variance)


def _is_positive_finite_number(value: Any) -> bool:
    """Return True when value is a positive finite int/float."""
    return isinstance(value, (int, float)) and math.isfinite(value) and value > 0
