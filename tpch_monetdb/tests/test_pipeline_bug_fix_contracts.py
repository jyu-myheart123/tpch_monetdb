import json

import pytest

from tpch_monetdb.utils.control_artifacts import (
    build_control_artifact_envelope,
    ensure_required_control_artifacts_acknowledged,
    ensure_required_control_artifacts_present,
    write_control_artifact_audit_copy,
)
from tpch_monetdb.utils.pipeline_contracts import PipelineContractError
from tpch_monetdb.utils.pipeline_invariants import (
    require_mapping_keys,
    require_nonempty_value,
    require_resume_snapshot_fields,
)
from tpch_monetdb.utils.large_data_objectives import (
    build_data_law_contract_payload,
    build_objective_failure_report,
    build_workload_objective_payload,
    is_large_data_success,
    validate_storage_plan_contract_payload,
    write_storage_plan_alignment,
    write_workload_objective,
)


def _valid_storage_plan_contract() -> dict:
    """Build a complete storage-plan contract fixture for validator tests."""
    critical_cost = {
        "predicted_scan_rows": 10,
        "lookup_cost": "host/time index lookup",
        "output_rows_cardinality": 4,
        "cache_locality": "contiguous host blocks",
        "vectorization_lane_shape": "fixed-width metric lanes",
        "build_memory_cost": "one host offset table",
        "risk": "low",
    }
    return {
        "version": 1,
        "candidate_layouts": [
            {
                "id": "conservative-row-major",
                "layout_kind": "conservative",
                "data_law_ids": ["LAW_TPCH_TABLE_CARDINALITY"],
                "evidence_refs": ["design_evidence.md#TPC-H Table Cardinality"],
                "query_family_fit": "baseline fit",
                "correctness_risk": "low",
                "build_ingest_complexity": "low",
                "vectorization_readiness": "medium",
                "memory_locality": "medium",
            },
            {
                "id": "hybrid-host-blocks",
                "layout_kind": "hybrid",
                "data_law_ids": ["LAW_TPCH_JOIN_GRAPH"],
                "evidence_refs": ["design_evidence.md#TPC-H Query Join Graph"],
                "query_family_fit": "critical families",
                "correctness_risk": "medium",
                "build_ingest_complexity": "medium",
                "vectorization_readiness": "high",
                "memory_locality": "high",
            },
            {
                "id": "aggressive-columnar-sidecars",
                "layout_kind": "aggressive",
                "data_law_ids": ["LAW_TPCH_OUTPUT_ORDERING"],
                "evidence_refs": ["design_evidence.md#TPC-H Output Semantics"],
                "query_family_fit": "global aggregates",
                "correctness_risk": "high",
                "build_ingest_complexity": "high",
                "vectorization_readiness": "high",
                "memory_locality": "high",
            },
        ],
        "selected_layout": {
            "candidate_id": "hybrid-host-blocks",
            "data_law_ids": ["LAW_TPCH_JOIN_GRAPH", "LAW_TPCH_TABLE_CARDINALITY"],
            "selection_rationale": "Best critical-query balance from design_evidence.md.",
        },
        "selected_layout_obligations": [
            {
                "id": "obl-critical-host-blocks",
                "file_scope": ["builder_impl.hpp", "builder_impl.cpp", "query_q8.cpp"],
                "query_ids": ["8", "9", "11", "12", "15"],
                "evidence_refs": ["design_evidence.md#Layout Decision Signals"],
            }
        ],
        "query_family_costs": {
            "8": dict(critical_cost),
            "9": dict(critical_cost),
            "11": dict(critical_cost),
            "12": dict(critical_cost),
            "15": dict(critical_cost),
        },
    }


def _valid_committed_storage_plan_contract() -> dict:
    """Build a v2 committed storage-plan contract fixture for validator tests."""
    critical_cost = {
        "predicted_scan_rows": 1,
        "lookup_cost": "join-key dictionary lookup",
        "output_rows_cardinality": 1,
        "cache_locality": "columnar table vectors and join-key maps are contiguous",
        "vectorization_lane_shape": "fixed-width aggregate lanes",
        "build_memory_cost": "no build-time aggregate answers",
        "risk": "low",
    }
    return {
        "version": 2,
        "selected_base_candidate_id": "candidate_B",
        "committed_layout": {
            "layout_id": "committed-tpch-critical-paths",
            "summary": "TPC-H table-vector layout with reusable Q1 scan aggregation and Q9 join/profit aggregation access paths.",
            "data_law_ids": [
                "LAW_TPCH_TABLE_CARDINALITY",
                "LAW_TPCH_JOIN_GRAPH",
                "LAW_TPCH_OUTPUT_ORDERING",
            ],
            "evidence_refs": [
                "design_evidence.md#TPC-H Output Semantics",
                "design_evidence.md#TPC-H Layout Decision Signals",
            ],
            "selection_rationale": "The selected base candidate is refined into reusable scan, join, aggregation, and ordering paths for critical TPC-H queries.",
        },
        "refinement_decisions": [
            {
                "id": "refine-q1-lineitem-scan",
                "query_ids": ["1"],
                "decision": "Promote Q1 lineitem date-filtered scan and grouped aggregation as a committed reusable access path.",
                "evidence_refs": ["design_evidence.md#TPC-H Output Semantics"],
            },
            {
                "id": "refine-q9-profit-join",
                "query_ids": ["9"],
                "decision": "Promote Q9 part/supplier/partsupp/lineitem/orders/nation join and query-time profit aggregation path.",
                "evidence_refs": ["design_evidence.md#TPC-H Layout Decision Signals"],
            },
        ],
        "critical_query_access_paths": {
            "1": {
                "semantic_class": "lineitem_grouped_pricing_summary",
                "selected_access_path": "Q1_LINEITEM_DATE_FILTERED_COLUMN_SCAN",
                "rejected_access_paths": ["per-query materialized answer"],
                "evidence_refs": ["design_evidence.md#TPC-H Output Semantics"],
                "data_law_ids": ["LAW_TPCH_TABLE_CARDINALITY", "LAW_TPCH_OUTPUT_ORDERING"],
                "access_path_kind": "query_time_columnar_scan",
                "legal_basis": "Query scans reusable lineitem vectors, applies date filter, groups by returnflag/linestatus, and preserves ORDER BY.",
                "not_query_answer_cache": True,
            },
            "9": {
                "semantic_class": "profit_by_nation_year",
                "selected_access_path": "Q9_PART_NAME_JOIN_PROFIT_AGGREGATION",
                "rejected_access_paths": ["per-query materialized answer", "build-time aggregate answer"],
                "evidence_refs": ["design_evidence.md#TPC-H Layout Decision Signals"],
                "data_law_ids": ["LAW_TPCH_JOIN_GRAPH", "LAW_TPCH_OUTPUT_ORDERING"],
                "access_path_kind": "query_time_join_aggregation",
                "legal_basis": "Query probes reusable join-key maps and aggregates profit by nation/year at query time.",
                "not_query_answer_cache": True,
            },
        },
        "selected_layout_obligations": [
            {
                "id": "OBLIG_Q1_LINEITEM_SCAN_AGGREGATION",
                "file_scope": ["builder_impl.hpp", "builder_impl.cpp", "query_q1.cpp"],
                "query_ids": ["1"],
                "evidence_refs": ["design_evidence.md#TPC-H Output Semantics"],
            },
            {
                "id": "OBLIG_Q9_JOIN_PROFIT_AGGREGATION",
                "file_scope": [
                    "query_shared_tpch.hpp",
                    "query_q9.cpp",
                ],
                "query_ids": ["9"],
                "evidence_refs": ["design_evidence.md#TPC-H Layout Decision Signals"],
            },
        ],
        "query_family_costs": {
            "1": dict(critical_cost),
            "9": dict(critical_cost),
        },
    }


def test_require_mapping_keys_raises_structured_error() -> None:
    """Missing required keys must raise a structured contract error."""
    with pytest.raises(PipelineContractError, match="MISSING_KEYS"):
        require_mapping_keys(
            {"a": 1},
            required_keys=("a", "b"),
            code="MISSING_KEYS",
            stage="unit_test",
        )
    return None


def test_require_nonempty_value_rejects_empty_fields() -> None:
    """Empty contract fields must fail with the requested code."""
    with pytest.raises(PipelineContractError, match="EMPTY_FIELD"):
        require_nonempty_value(
            "",
            code="EMPTY_FIELD",
            field_name="storage_plan_sha256",
            stage="unit_test",
        )
    return None


def test_require_resume_snapshot_fields_accepts_complete_snapshot() -> None:
    """Complete new-contract resume metadata should pass the invariant gate."""
    require_resume_snapshot_fields(
        {
            "implementation_manifest_sha256": "manifest",
            "storage_plan_sha256": "plan",
            "todo_sha256": "todo",
            "todo_reconciliation": {"status": "completed"},
            "control_artifact_hashes": {"plan": "plan"},
        },
        stage="resume_test",
    )
    return None


def test_require_resume_snapshot_fields_rejects_incomplete_snapshot() -> None:
    """Old snapshots without the new contract fields must be rejected."""
    with pytest.raises(PipelineContractError, match="RESUME_SNAPSHOT_INCOMPLETE"):
        require_resume_snapshot_fields(
            {
                "storage_plan_sha256": "plan",
                "todo_sha256": "todo",
            },
            stage="resume_test",
        )
    return None


def test_control_artifact_envelope_tracks_hashes_and_audit_copy(tmp_path) -> None:
    """Control-artifact audit copy must persist stable hashes and alignment metadata."""
    (tmp_path / "storage_plan.txt").write_text("plan\n", encoding="utf-8")
    (tmp_path / "TODO.md").write_text("- [x] done\n", encoding="utf-8")
    envelope = build_control_artifact_envelope(tmp_path)

    assert envelope.artifact_hashes["storage_plan.txt"]
    assert envelope.artifact_hashes["TODO.md"]

    audit_path = write_control_artifact_audit_copy(tmp_path)
    payload = audit_path.read_text(encoding="utf-8")
    assert "artifact_hashes" in payload
    assert "todo_reconciliation" in payload
    return None


def test_required_control_artifacts_present_rejects_missing_workspace_files(tmp_path) -> None:
    """Missing required control artifacts must fail before stage execution."""
    (tmp_path / "storage_plan.txt").write_text("plan\n", encoding="utf-8")
    with pytest.raises(PipelineContractError, match="CONTROL_ARTIFACT_MISSING"):
        ensure_required_control_artifacts_present(
            tmp_path,
            ("storage_plan.txt", "TODO.md"),
            stage="unit_test",
        )
    return None


def test_required_control_artifacts_acknowledged_rejects_unread_and_uninjected() -> None:
    """Write/run gates must reject required artifacts that were neither read nor injected."""
    with pytest.raises(
        PipelineContractError,
        match="CONTROL_ARTIFACT_NOT_ACKNOWLEDGED",
    ):
        ensure_required_control_artifacts_acknowledged(
            ("TODO.md", "storage_plan.txt"),
            read_artifacts={"storage_plan.txt"},
            injected_artifacts=(),
            action="compile",
            stage="unit_test",
        )
    return None


def test_workload_objective_marks_critical_queries_and_policies() -> None:
    """Host-owned objective must make critical query gates explicit."""
    payload = build_workload_objective_payload(
        query_ids=["1", "8", "9", "12"],
        benchmark_sf=100,
        large_sf=1000,
        hardware_counter_backend="linux_perf_native",
        target_cpu="icelake",
    )
    assert payload["benchmark"] == "tpch"
    assert payload["objective_id"] == "tpch-docker-monetdb-objective-v1"
    assert payload["critical_query_ids"] == ["1", "8", "9", "12"]
    assert payload["critical_query_targets"]["1"]["requires_vectorization"] is False
    assert payload["critical_query_targets"]["8"]["requires_vectorization"] is True
    assert payload["critical_query_targets"]["8"]["requires_pmu"] is True
    assert payload["critical_query_targets"]["9"]["min_speedup_vs_base_impl"] == 1.05
    return None


def test_tpch_data_law_contract_uses_monetdb_tpc_h_laws(tmp_path) -> None:
    """TPC-H data-law contract must not expose QuestDB/TSBS law ids."""
    evidence_path = tmp_path / "design_evidence.md"
    evidence_path.write_text("## TPC-H Table Cardinality\n", encoding="utf-8")

    payload = build_data_law_contract_payload(
        query_ids=["1", "9"],
        benchmark_sf=1,
        design_evidence_path=evidence_path,
    )

    law_ids = {item["law_id"] for item in payload["laws"]}
    assert payload["contract_id"] == "tpch-data-law-contract-v1"
    assert payload["benchmark"] == "tpch"
    assert "LAW_TPCH_TABLE_CARDINALITY" in law_ids
    assert "LAW_TPCH_JOIN_GRAPH" in law_ids
    assert "LAW_TPCH_RUNTIME_BOUNDARY" in law_ids
    assert "LAW_HOST_TAG_STRUCTURE" not in law_ids
    assert "LAW_OUTPUT_CARDINALITY" not in law_ids
    return None


def test_storage_plan_contract_requires_critical_costs_and_data_law_refs() -> None:
    """Storage plan contract must cover critical query costs and data-law refs."""
    validation = validate_storage_plan_contract_payload(
        _valid_storage_plan_contract(),
        query_ids=["8", "9", "11", "12", "15"],
        data_law_contract={
            "laws": [
                {"law_id": "LAW_TPCH_TABLE_CARDINALITY"},
                {"law_id": "LAW_TPCH_JOIN_GRAPH"},
                {"law_id": "LAW_TPCH_OUTPUT_ORDERING"},
            ]
        },
    )
    assert validation.status == "valid"
    assert validation.selected_candidate_id == "hybrid-host-blocks"
    assert validation.obligation_ids == ("obl-critical-host-blocks",)

    invalid = validate_storage_plan_contract_payload(
        {
            "version": 1,
            "candidate_layouts": [{"name": "a"}],
            "selected_layout": {"name": "a"},
            "query_family_costs": {"8": {}},
        },
        query_ids=["8", "9"],
    )
    assert "STORAGE_PLAN_CONTRACT_CRITICAL_COSTS_MISSING" in invalid.failures
    assert "STORAGE_PLAN_CONTRACT_DATA_LAW_REFS_MISSING" in invalid.failures
    return None


def test_storage_plan_contract_rejects_shallow_candidates_and_unknown_selection() -> None:
    """Contract validation must reject non-material candidates and bad selection refs."""
    contract = _valid_storage_plan_contract()
    contract["candidate_layouts"] = [
        {"id": "a", "layout_kind": "conservative"},
        {"id": "b", "layout_kind": "conservative"},
        {"id": "c", "layout_kind": "conservative"},
    ]
    contract["selected_layout"]["candidate_id"] = "missing"

    validation = validate_storage_plan_contract_payload(
        contract,
        query_ids=["8", "9", "11", "12", "15"],
    )

    assert "STORAGE_PLAN_CONTRACT_CANDIDATE_IDS_MISSING" not in validation.failures
    assert "STORAGE_PLAN_CONTRACT_SELECTED_CANDIDATE_UNKNOWN" in validation.failures
    assert "STORAGE_PLAN_CONTRACT_CANDIDATES_NOT_MATERIAL" in validation.failures
    assert "STORAGE_PLAN_CONTRACT_CANDIDATE_COMPARISON_MISSING" in validation.failures
    assert "STORAGE_PLAN_CONTRACT_CANDIDATE_EVIDENCE_MISSING" in validation.failures
    return None


def test_storage_plan_contract_rejects_candidate_records_without_ids() -> None:
    """Candidate records must expose stable IDs for selected-layout references."""
    contract = _valid_storage_plan_contract()
    contract["candidate_layouts"][0].pop("id")
    contract["candidate_layouts"][0].pop("candidate_id", None)
    contract["candidate_layouts"][0].pop("name", None)

    validation = validate_storage_plan_contract_payload(
        contract,
        query_ids=["8", "9", "11", "12", "15"],
    )

    assert "STORAGE_PLAN_CONTRACT_CANDIDATE_IDS_MISSING" in validation.failures
    return None


def test_storage_plan_contract_rejects_missing_cost_dimensions_and_obligations() -> None:
    """Critical costs and selected obligations must be query-covering contracts."""
    contract = _valid_storage_plan_contract()
    contract["query_family_costs"]["8"] = {"predicted_scan_rows": 10}
    contract["selected_layout_obligations"] = [
        {
            "id": "obl-q8-only",
            "file_scope": ["query_q8.cpp"],
            "query_ids": ["8"],
            "evidence_refs": ["design_evidence.md#Layout Decision Signals"],
        }
    ]

    validation = validate_storage_plan_contract_payload(
        contract,
        query_ids=["8", "9", "11", "12", "15"],
    )

    assert "STORAGE_PLAN_CONTRACT_CRITICAL_COST_FIELDS_MISSING" in validation.failures
    assert "STORAGE_PLAN_CONTRACT_OBLIGATION_QUERY_COVERAGE_MISSING" in validation.failures
    return None


def test_storage_plan_contract_accepts_v2_committed_contract_without_candidates() -> None:
    """V2 committed contracts must validate without carrying rejected candidate bodies."""
    validation = validate_storage_plan_contract_payload(
        _valid_committed_storage_plan_contract(),
        query_ids=["1", "9"],
        data_law_contract={
            "laws": [
                {"law_id": "LAW_TPCH_TABLE_CARDINALITY"},
                {"law_id": "LAW_TPCH_JOIN_GRAPH"},
                {"law_id": "LAW_TPCH_OUTPUT_ORDERING"},
            ]
        },
    )

    assert validation.status == "valid"
    assert validation.selected_candidate_id == "candidate_B"
    assert validation.obligation_ids == (
        "OBLIG_Q1_LINEITEM_SCAN_AGGREGATION",
        "OBLIG_Q9_JOIN_PROFIT_AGGREGATION",
    )
    assert set(validation.covered_query_ids) == {"1", "9"}
    assert validation.missing_query_ids == ()
    return None


def test_storage_plan_contract_v2_rejects_candidate_bodies() -> None:
    """Committed v2 contracts must keep full candidate exploration out of the contract."""
    contract = _valid_committed_storage_plan_contract()
    contract["candidate_layouts"] = [{"id": "candidate_A"}]

    validation = validate_storage_plan_contract_payload(
        contract,
        query_ids=["1", "9"],
    )

    assert validation.status == "invalid"
    assert "STORAGE_PLAN_CONTRACT_COMMITTED_HAS_CANDIDATES" in validation.failures
    return None


def test_storage_plan_contract_v2_requires_explicit_selected_base_candidate_id() -> None:
    """V2 contracts must not rely on the old selected_layout candidate pointer."""
    contract = _valid_committed_storage_plan_contract()
    contract.pop("selected_base_candidate_id")
    contract["selected_layout"] = {"candidate_id": "candidate_B"}

    validation = validate_storage_plan_contract_payload(
        contract,
        query_ids=["1", "9"],
    )

    assert "STORAGE_PLAN_CONTRACT_SELECTED_BASE_CANDIDATE_MISSING" in validation.failures
    return None


def test_storage_plan_contract_v2_requires_critical_access_path_coverage() -> None:
    """Every requested critical query must have an explicit committed access path."""
    contract = _valid_committed_storage_plan_contract()
    contract["critical_query_access_paths"].pop("9")

    validation = validate_storage_plan_contract_payload(
        contract,
        query_ids=["1", "9"],
    )

    assert validation.status == "invalid"
    assert "STORAGE_PLAN_CONTRACT_CRITICAL_ACCESS_PATH_QUERY_COVERAGE_MISSING" in validation.failures
    assert "9" in validation.missing_query_ids
    return None


def test_storage_plan_contract_v2_rejects_query_answer_cache_access_path() -> None:
    """Critical access paths must be reusable structures, not per-query answer caches."""
    contract = _valid_committed_storage_plan_contract()
    contract["critical_query_access_paths"]["1"]["not_query_answer_cache"] = False

    validation = validate_storage_plan_contract_payload(
        contract,
        query_ids=["1", "9"],
    )

    assert validation.status == "invalid"
    assert "STORAGE_PLAN_CONTRACT_CRITICAL_ACCESS_PATH_CACHE_BOUNDARY_MISSING" in validation.failures
    return None


def test_storage_plan_contract_v2_uses_generic_tpc_h_access_path_rules() -> None:
    """TPC-H contracts should not enforce legacy TPC-H MonetDB Q1/Q9 access-path names."""
    contract = _valid_committed_storage_plan_contract()
    contract["critical_query_access_paths"]["1"]["selected_access_path"] = "Q1_REUSABLE_LINEITEM_SCAN"
    contract["critical_query_access_paths"]["9"]["access_path_kind"] = "query_time_join_aggregation"

    validation = validate_storage_plan_contract_payload(
        contract,
        query_ids=["1", "9"],
    )

    assert validation.status == "valid"
    assert "STORAGE_PLAN_CONTRACT_Q1_LASTPOINT_SIDECAR_MISSING" not in validation.failures
    assert "STORAGE_PLAN_CONTRACT_Q9_HOST_HOUR_AGGREGATE_PATH_MISSING" not in validation.failures
    return None


def test_storage_plan_contract_v2_keeps_generic_cache_boundary() -> None:
    """TPC-H contracts still reject per-query answer-cache access paths."""
    contract = _valid_committed_storage_plan_contract()
    contract["critical_query_access_paths"]["9"]["not_query_answer_cache"] = False

    validation = validate_storage_plan_contract_payload(
        contract,
        query_ids=["1", "9"],
    )

    assert validation.status == "invalid"
    assert "STORAGE_PLAN_CONTRACT_CRITICAL_ACCESS_PATH_CACHE_BOUNDARY_MISSING" in validation.failures
    return None


def test_storage_plan_contract_v2_requires_committed_layout_and_refinement() -> None:
    """Committed v2 contracts must name the selected layout and explain refinements."""
    contract = _valid_committed_storage_plan_contract()
    contract["committed_layout"].pop("summary")
    contract["refinement_decisions"] = []

    validation = validate_storage_plan_contract_payload(
        contract,
        query_ids=["1", "9"],
    )

    assert "STORAGE_PLAN_CONTRACT_COMMITTED_LAYOUT_FIELDS_MISSING" in validation.failures
    assert "STORAGE_PLAN_CONTRACT_REFINEMENT_DECISIONS_MISSING" in validation.failures
    return None


def test_storage_plan_alignment_is_evaluated_from_contract(tmp_path) -> None:
    """Alignment artifact must not use the old not_evaluated placeholder."""
    write_workload_objective(
        workspace_path=tmp_path,
        query_ids=["8", "9"],
        benchmark_sf=100,
        large_sf=1000,
        hardware_counter_backend=None,
        target_cpu=None,
    )
    (tmp_path / "storage_plan.txt").write_text("layout\n", encoding="utf-8")
    (tmp_path / "data_law_contract.json").write_text(
        (
            '{"laws":[{"law_id":"LAW_TPCH_TABLE_CARDINALITY"},'
            '{"law_id":"LAW_TPCH_JOIN_GRAPH"},'
            '{"law_id":"LAW_TPCH_OUTPUT_ORDERING"}]}\n'
        ),
        encoding="utf-8",
    )
    (tmp_path / "storage_plan_contract.json").write_text(
        json.dumps(_valid_storage_plan_contract(), indent=2) + "\n",
        encoding="utf-8",
    )
    alignment_path = write_storage_plan_alignment(tmp_path, query_ids=["8", "9"])
    payload = json.loads(alignment_path.read_text(encoding="utf-8"))
    assert payload["status"] == "contract_valid"
    assert payload["selected_candidate_id"] == "hybrid-host-blocks"
    assert payload["selected_layout_obligation_ids"] == ["obl-critical-host-blocks"]
    assert payload["missing_obligation_query_ids"] == []
    return None


def test_large_data_gate_rejects_unmapped_vectorization_and_missing_pmu() -> None:
    """Large-data gate must reject unrelated vectorization and PMU gaps."""
    class Summary:
        pass

    summary = Summary()
    summary.workload_objective = {
        "critical_query_ids": ["8"],
        "required_artifacts": ["workload_objective.json"],
        "measurement_policy": {"max_cv": 0.2, "max_spread_ratio": 0.4},
        "critical_query_targets": {
            "8": {
                "min_speedup_vs_baseline": 1.0,
                "requires_vectorization": True,
                "requires_pmu": True,
            }
        },
    }
    summary.final_runtime_ms_by_query = {"8": 10.0}
    summary.baseline_runtime_ms_by_query = {"8": 20.0}
    summary.storage_plan_alignment = {"status": "aligned"}
    summary.control_artifact_hashes = {"workload_objective.json": "hash"}
    summary.measurement_repetition = {
        "aggregate_runtime_ms_samples": [10.0, 10.1, 9.9],
    }
    summary.compiler_vectorization_summary = {
        "8": {
            "vectorization_applied": True,
            "optimized_loop_sites": [{"file": "query_q1.cpp", "line": 10}],
            "hot_loop_mapping": {"status": "unmatched"},
        }
    }
    summary.hardware_counter_summary = {"8": {"hardware_counters_available": False}}

    report = build_objective_failure_report(summary)
    assert is_large_data_success(summary) is False
    assert "VECTOR_HOT_LOOP_NOT_OPTIMIZED" in report.failures
    assert "PMU_REQUIRED_BUT_MISSING" in report.failures
    return None


def test_large_data_gate_rejects_whole_process_pmu_scope() -> None:
    """PMU evidence must be scoped to the measured query loop."""
    summary = type("_Summary", (), {})()
    summary.workload_objective = {
        "critical_query_ids": ["8"],
        "required_artifacts": [],
        "measurement_policy": {"max_cv": 0.2, "max_spread_ratio": 0.4},
        "critical_query_targets": {
            "8": {
                "min_speedup_vs_baseline": 1.0,
                "requires_vectorization": False,
                "requires_pmu": True,
            }
        },
    }
    summary.final_runtime_ms_by_query = {"8": 10.0}
    summary.baseline_runtime_ms_by_query = {"8": 20.0}
    summary.storage_plan_alignment = {"status": "aligned"}
    summary.control_artifact_hashes = {}
    summary.measurement_repetition = {
        "aggregate_runtime_ms_samples": [10.0, 10.1, 9.9],
    }
    summary.compiler_vectorization_summary = {}
    summary.hardware_counter_summary = {
        "8": {
            "hardware_counters_available": True,
            "perf_hotspots_available": True,
            "perf_hotspot_provenance": {"capture_scope": "query_batch_file"},
        }
    }

    report = build_objective_failure_report(summary)

    assert "PMU_CAPTURE_SCOPE_NOT_QUERY_ONLY" in report.failures
    return None
