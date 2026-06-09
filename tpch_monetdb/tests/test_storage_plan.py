"""Tests for phase10 storage-plan open-ended prompt and summary reuse.

验证:
- storage_plan prompt evidence-first
- storage_plan prompt 不再硬编码 Allowed Optimizations 清单
- storage_plan prompt 要求读取 queries.txt
- storage_plan prompt specialized boundary 收口
- StoragePlanRunSummary 有 storage_plan_excerpt 字段
- prev_run_report 条件注入 prompt
- --prev_run_report CLI 参数存在
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest


def _make_prompt(prev_run_report=None) -> str:
    from tpch_monetdb.run_gen_storage_plan_tpch_monetdb import create_conversation
    import json
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        conv_dir = Path(tmpdir)
        create_conversation(
            benchmark="tpch",
            short_name="storageplantest",
            conversation_dir=conv_dir,
            base_data_dir=Path("/data"),
            max_scale_factor=1,
            query_ids=["q1", "q13"],
            prev_run_report=prev_run_report,
        )
        target = conv_dir / "tpch_storageplantest.json"
        data = json.loads(target.read_text())
        return data[0]["text"]


# ---------------------------------------------------------------------------
# Prompt content tests (tasks 9.1 / 9.2)
# ---------------------------------------------------------------------------

def test_storage_plan_prompt_removes_do_not_read_instruction() -> None:
    """prompt 不应包含 'Do not read additional project files'."""
    prompt = _make_prompt()
    assert "Do not read additional project files" not in prompt


def test_storage_plan_prompt_removes_allowed_optimizations_list() -> None:
    """prompt 不应包含 'Allowed Optimizations' 硬编码清单."""
    prompt = _make_prompt()
    assert "Allowed Optimizations" not in prompt


def test_storage_plan_prompt_requires_reading_queries_txt() -> None:
    """prompt 应要求读取 queries.txt."""
    prompt = _make_prompt()
    assert "queries.txt" in prompt


def test_storage_plan_prompt_requires_design_evidence_first() -> None:
    """prompt 应先读取 design_evidence.md."""
    prompt = _make_prompt()
    assert "Read `workload_objective.json` first" in prompt
    assert "Read `data_law_contract.json`" in prompt
    assert "Read `design_evidence.md`" in prompt


def test_storage_plan_prompt_replaces_general_purpose_constraints() -> None:
    """prompt 不再要求 arbitrary SQL generality，也不再全禁 indexes."""
    prompt = _make_prompt().lower()
    assert "single-node" in prompt or "single-threaded" in prompt
    assert "arbitrary sql" not in prompt
    assert "no indexes" not in prompt
    assert "shared specialized physical structures" in prompt
    assert "precomputing aggregate answers" in prompt
    assert "forbidden pre-aggregation" in prompt
    assert "aggregate sidecars or aggregate indexes" in prompt
    assert "host-hour aggregate/index sidecars" not in prompt
    assert "per-query materialized answers" in prompt
    assert "per-query caches" in prompt
    assert "duplicate logical data" in prompt or "duplicated logical rows" in prompt


def test_storage_plan_prompt_requires_machine_readable_contract() -> None:
    """storage plan 阶段必须同时产出可校验 JSON contract."""
    prompt = _make_prompt()
    assert "storage_plan_contract.json" in prompt
    assert "storage_plan_candidates.json" in prompt
    assert "selected_base_candidate_id" in prompt
    assert "committed_layout" in prompt
    assert "refinement_decisions" in prompt
    assert "critical_query_access_paths" in prompt
    assert "key maps" in prompt
    assert "vectorization-friendly loop shape" in prompt
    assert "query_family_costs" in prompt
    assert "selected_layout_obligations" in prompt
    assert "data_law_ids" in prompt
    assert "Do not put full `candidate_layouts` in `storage_plan_contract.json`" in prompt
    return None


def test_storage_plan_prompt_does_not_preset_hostname_major() -> None:
    """prompt 不应预设 hostname -> series_id 为默认布局."""
    prompt = _make_prompt().lower()
    # 不应出现预设 hostname 字典化结构的硬编码提示
    assert "hostname -> series_id dictionary" not in prompt
    assert "per-series latest-point pointer" not in prompt


# ---------------------------------------------------------------------------
# prev_run_report injection tests (task 10.4)
# ---------------------------------------------------------------------------

def test_storage_plan_prompt_injects_prev_run_report_when_provided() -> None:
    """当 prev_run_report 非空时，prompt 应包含该 report 内容."""
    report = "## Previous round\nQ13 is slow: 23ms vs MonetDB 10ms"
    prompt = _make_prompt(prev_run_report=report)
    assert "Q13 is slow" in prompt


def test_storage_plan_prompt_no_prev_section_when_not_provided() -> None:
    """当 prev_run_report 为 None 时，prompt 不应包含 'Previous round feedback'."""
    prompt = _make_prompt(prev_run_report=None)
    assert "Previous round feedback" not in prompt


def test_storage_plan_prompt_requests_different_layout_when_prev_provided() -> None:
    """当提供 prev_run_report 时，prompt 应按 failure route 决定是否结构性重做."""
    report = "slow: q13"
    prompt = _make_prompt(prev_run_report=report)
    assert "failure route and objective failures" in prompt
    assert "do not prefer a convenience minimal patch" in prompt
    assert "Q1" in prompt
    assert "Q8" in prompt
    assert "Q9" in prompt
    assert "Q11" in prompt
    assert "Q12" in prompt
    assert "Q15" in prompt


def test_storage_plan_prompt_requires_quantitative_evidence_and_unverified_assumptions() -> None:
    prompt = _make_prompt()
    assert "runtime ranking positions" in prompt
    assert "`row_count`" in prompt
    assert "table cardinality" in prompt
    assert "Unverified assumption:" in prompt


def test_storage_plan_prompt_requires_query_intent_and_semantic_traps() -> None:
    """storage_plan prompt 应要求引用查询意图和语义陷阱."""
    prompt = _make_prompt()
    assert "Query Intent and Cost Semantics" in prompt
    assert "Semantic traps" in prompt
    assert "do not infer the workload purpose from SQL column names alone" in prompt


def test_storage_plan_prompt_requires_candidate_layout_exploration() -> None:
    prompt = _make_prompt()
    assert "Mode: initial_candidates" in prompt
    assert "Explore at least 3 materially different candidate layouts" in prompt
    assert "Refine the selected base candidate against MonetDB/TPC-H output semantics" in prompt
    assert "storage_plan_candidates.json" in prompt
    assert "Candidate comparison / delta comparison" in prompt
    assert "Rejected candidates / rejected deltas" in prompt
    assert "Do NOT jump directly to one final layout" in prompt
    assert "Do not put full `candidate_layouts`" in prompt


def test_storage_plan_prompt_repair_mode_does_not_restart_three_designs() -> None:
    prompt = _make_prompt(prev_run_report="STORAGE_PLAN_ALIGNMENT_NOT_EVALUATED")
    from tpch_monetdb.run_gen_storage_plan_tpch_monetdb import create_conversation
    import json
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        conv_dir = Path(tmpdir)
        create_conversation(
            benchmark="tpch",
            short_name="storageplanrepair",
            conversation_dir=conv_dir,
            base_data_dir=Path("/data"),
            max_scale_factor=1,
            query_ids=["1"],
            prev_run_report="STORAGE_PLAN_ALIGNMENT_NOT_EVALUATED",
            storage_plan_mode="repair_alignment",
        )
        data = json.loads((conv_dir / "tpch_storageplanrepair.json").read_text())
    assert "Mode: repair_alignment" in data[0]["text"]
    assert "Do not invent a new storage architecture" in data[0]["text"]
    assert "critical query access paths" in data[0]["text"]
    assert "Do not reintroduce `candidate_layouts`" in data[0]["text"]
    assert "Explore at least 3" not in data[0]["text"]
    assert "STORAGE_PLAN_ALIGNMENT_NOT_EVALUATED" in prompt


def test_storage_plan_prompt_delta_mode_disallows_broad_candidates() -> None:
    from tpch_monetdb.run_gen_storage_plan_tpch_monetdb import create_conversation
    import json
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        conv_dir = Path(tmpdir)
        create_conversation(
            benchmark="tpch",
            short_name="storageplandelta",
            conversation_dir=conv_dir,
            base_data_dir=Path("/data"),
            max_scale_factor=1,
            query_ids=["1"],
            prev_run_report="success=false; OFFICIAL_QUERY_RUNTIME_MISSING",
            storage_plan_mode="delta_plan",
        )
        data = json.loads((conv_dir / "tpch_storageplandelta.json").read_text())
    text = data[0]["text"]
    assert "Mode: delta_plan" in text
    assert "Do not restart from three broad designs" in text
    assert "Do not add `candidate_layouts`" in text
    assert "targeted deltas" in text
    assert "critical access paths" in text
    assert "why_not_changing_layout" in text
    assert "measurement-contract repair" in text


def test_storage_plan_prompt_respects_explicit_max_scale_factor_100() -> None:
    prompt = _make_prompt()
    assert "sf1" in prompt
    assert "sf10" not in prompt
    assert "sf100" not in prompt


def test_storage_plan_prompt_handles_correctness_gate_feedback() -> None:
    report = "Correctness gate: previous round final_correctness=false; do not use it as layout evidence."
    prompt = _make_prompt(prev_run_report=report)
    assert report in prompt
    assert "treat it as a gating" in prompt


# ---------------------------------------------------------------------------
# StoragePlanRunSummary tests (task 9.3)
# ---------------------------------------------------------------------------

def test_storage_plan_summary_has_excerpt_field() -> None:
    """StoragePlanRunSummary 应有 storage_plan_excerpt 字段."""
    from tpch_monetdb.utils.storage_plan_summary import StoragePlanRunSummary
    summary = StoragePlanRunSummary(
        benchmark="tpch",
        conv_name="x",
        run_id="x",
        query_list=["q1"],
        final_snapshot_hash="abc",
        storage_plan_path="/tmp/storage_plan.txt",
        completed_at="",
        conversation_json="",
        session_db_path="",
        success=True,
        storage_plan_excerpt="layout description",
    )
    assert summary.storage_plan_excerpt == "layout description"


def test_storage_plan_summary_excerpt_defaults_to_empty() -> None:
    """未提供 excerpt 时应默认为空字符串."""
    from tpch_monetdb.utils.storage_plan_summary import StoragePlanRunSummary
    summary = StoragePlanRunSummary(
        benchmark="tpch",
        conv_name="x",
        run_id="x",
        query_list=["q1"],
        final_snapshot_hash="abc",
        storage_plan_path="/tmp/plan.txt",
        completed_at="",
        conversation_json="",
        session_db_path="",
        success=True,
    )
    assert summary.storage_plan_excerpt == ""


def test_storage_plan_summary_can_store_sha_and_size_metadata() -> None:
    """StoragePlanRunSummary 应能携带 storage_plan 的 hash 与 size 元数据."""
    from tpch_monetdb.utils.storage_plan_summary import StoragePlanRunSummary

    summary = StoragePlanRunSummary(
        benchmark="tpch",
        conv_name="x",
        run_id="x",
        query_list=["q1"],
        final_snapshot_hash="abc",
        storage_plan_path="/tmp/plan.txt",
        completed_at="",
        conversation_json="",
        session_db_path="",
        success=True,
        storage_plan_sha256="deadbeef",
        storage_plan_size_bytes=128,
    )
    assert summary.storage_plan_sha256 == "deadbeef"
    assert summary.storage_plan_size_bytes == 128


# ---------------------------------------------------------------------------
# CLI --prev_run_report argument test
# ---------------------------------------------------------------------------

def test_build_parser_has_prev_run_report_arg() -> None:
    """build_parser 应包含 --prev_run_report 参数."""
    from tpch_monetdb.run_gen_storage_plan_tpch_monetdb import build_parser
    parser = build_parser(add_help=False)
    args = parser.parse_args(["--conv", "storageplantest"])
    assert hasattr(args, "prev_run_report")
    assert args.prev_run_report is None


def test_storage_plan_control_artifacts_reject_missing_todo(tmp_path: Path) -> None:
    from tpch_monetdb.utils.control_artifacts import ensure_required_control_artifacts_present
    from tpch_monetdb.utils.pipeline_contracts import PipelineContractError

    (tmp_path / "storage_plan.txt").write_text("layout\n", encoding="utf-8")
    with pytest.raises(PipelineContractError, match="CONTROL_ARTIFACT_MISSING"):
        ensure_required_control_artifacts_present(
            tmp_path,
            ("storage_plan.txt", "TODO.md"),
            stage="storage_plan_test",
        )


def test_storage_plan_hash_changes_when_plan_becomes_stale(tmp_path: Path) -> None:
    from tpch_monetdb.utils.control_artifacts import build_control_artifact_envelope

    storage_plan = tmp_path / "storage_plan.txt"
    storage_plan.write_text("layout=v1\n", encoding="utf-8")
    first_hash = build_control_artifact_envelope(tmp_path).artifact_hashes["storage_plan.txt"]
    storage_plan.write_text("layout=v2\n", encoding="utf-8")
    second_hash = build_control_artifact_envelope(tmp_path).artifact_hashes["storage_plan.txt"]
    assert first_hash != second_hash
