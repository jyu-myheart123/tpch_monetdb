import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tpch_monetdb.conversations.optimization_conversation_tpch_monetdb import TpchMonetdbOptimizationConversation
from tpch_monetdb.run_gen_base_impl_tpch_monetdb import create_conversation
from tpch_monetdb.utils.query_codegen_hints import (
    QUERY_CODEGEN_HINTS,
    build_query_codegen_hint_text,
    get_query_generated_code_checks,
)


def test_query_codegen_hints_define_expected_high_risk_queries() -> None:
    assert set(QUERY_CODEGEN_HINTS) >= {"6", "9", "12", "13", "14", "15"}


def test_q9_codegen_hint_mentions_dense_or_bounded_path() -> None:
    text = build_query_codegen_hint_text("9")
    assert "bounded" in text
    assert "six-table join" in text
    assert "nation strings" in text
    assert "when" in text.lower()
    assert "should beat a sparse hash path" not in text


def test_q12_codegen_hint_mentions_sort_order_and_local_row_idx() -> None:
    text = build_query_codegen_hint_text("12")
    assert "l_shipmode" in text
    assert "order priority CASE logic" in text
    assert "candidate explosion" in text


def test_q6_codegen_hint_mentions_null_sum_and_typed_lineitem_fields() -> None:
    """Verify Q6 guidance matches current Engine types and SQL SUM NULL semantics."""
    text = build_query_codegen_hint_text("6")
    assert "LineitemRow fields" in text
    assert "already typed numeric fields" in text
    assert "std::stod(args.DISCOUNT)" in text
    assert "SQL SUM returns NULL" in text
    assert "empty revenue CSV cell" in text
    assert "Column values are stored as std::string" not in text
    return None


def test_q15_codegen_hint_mentions_direct_indexed_supplier_revenue_array() -> None:
    text = build_query_codegen_hint_text("15")
    assert "direct-indexed supplier revenue array" in text
    assert "hash map" in text
    assert "when" in text.lower()


def test_q1_q2_codegen_hints_preserve_engine_boundary() -> None:
    q1_text = build_query_codegen_hint_text("1")
    q2_text = build_query_codegen_hint_text("2")
    combined = q1_text + "\n" + q2_text
    assert "Engine-backed lineitem scan" in q1_text
    assert "source directory discovery" in q1_text
    assert "part, partsupp, supplier, nation, and region joins" in q2_text
    assert "minimum supply-cost" in q2_text
    assert "repair the Engine build path first" in q2_text
    assert "`.tbl` parsing" in combined
    assert "loader/builder" in combined
    return None


def test_groupby_codegen_hints_use_conditional_rather_than_absolute_language() -> None:
    q13_text = build_query_codegen_hint_text("13")
    q14_text = build_query_codegen_hint_text("14")
    assert "when" in q13_text.lower() or "if" in q13_text.lower()
    assert "when" in q14_text.lower() or "if" in q14_text.lower()
    assert "should usually stay on a direct-indexed path" not in q13_text
    assert "instead of falling back to a generic sparse map" not in q14_text


def test_query_generated_code_checks_include_antipatterns_for_high_risk_queries() -> None:
    base_checks = ["query_protocol", "final_path_integrity", "usage_double_output"]
    assert get_query_generated_code_checks("9") == [
        *base_checks,
        "query_antipatterns",
    ]
    assert get_query_generated_code_checks("12") == [
        *base_checks,
        "query_antipatterns",
    ]
    assert get_query_generated_code_checks("15") == base_checks
    assert get_query_generated_code_checks("1") == base_checks
    assert get_query_generated_code_checks("2") == base_checks
    return None


def test_base_conversation_emits_hint_backed_guidance_for_q9_and_q12(tmp_path: Path) -> None:
    conversation_dir = tmp_path / "conversations"
    artifacts_dir = tmp_path / "artifacts"
    create_conversation(
        short_name="basef9-12v1",
        query_ids=["9", "12"],
        verify_sf_list=[1],
        max_scale_factor=1,
        artifacts_dir=artifacts_dir,
        benchmark="tpch",
        conversation_dir=conversation_dir,
        sample_query_args_dict={},
        base_data_dir=tmp_path / "real_data_root",
    )
    target_path = conversation_dir / "tpch_basef9-12v1.json"
    data = json.loads(target_path.read_text(encoding="utf-8"))
    stage_items = [item for item in data if isinstance(item, dict)]
    q9_stage = next(
        item for item in stage_items
        if item.get("descriptor") == "implement_query_9"
    )
    q12_stage = next(
        item for item in stage_items
        if item.get("descriptor") == "implement_query_12"
    )
    assert "Additional implementation guidance:" in q9_stage["text"]
    assert "six-table join" in q9_stage["text"]
    assert "bounded `(nation, year)` accumulators" in q9_stage["text"]
    assert "l_shipmode" in q12_stage["text"]
    assert "order priority CASE logic" in q12_stage["text"]


def test_trace_expert_stage_embeds_query_specific_guidance_for_high_risk_queries() -> None:
    conversation = TpchMonetdbOptimizationConversation.__new__(
        TpchMonetdbOptimizationConversation
    )
    conversation.benchmark_sf = 100
    conversation.bespoke_storage = True

    stage = conversation._build_query_stage(
        query_id="9",
        mandatory_constraints="constraints",
        trace_summary="Trace summary",
    )
    expert_prompt = stage.get_prompt(1000.0)

    assert stage.name == "trace_expert"
    assert "Additional implementation guidance:" in expert_prompt
    assert "six-table join" in expert_prompt
    assert "nation strings" in expert_prompt
    return None


def test_trace_expert_stage_embeds_engine_boundary_guidance_for_q1() -> None:
    conversation = TpchMonetdbOptimizationConversation.__new__(
        TpchMonetdbOptimizationConversation
    )
    conversation.benchmark_sf = 100
    conversation.bespoke_storage = True

    stage = conversation._build_query_stage(
        query_id="1",
        mandatory_constraints="constraints",
        trace_summary="Trace summary",
    )
    expert_prompt = stage.get_prompt(1000.0)

    assert "Additional implementation guidance:" in expert_prompt
    assert "Engine-backed lineitem scan" in expert_prompt
    assert "query-time source-file parsing" in expert_prompt
    return None


def test_base_conversation_emits_query_guidance_for_q1_and_q2(tmp_path: Path) -> None:
    conversation_dir = tmp_path / "conversations"
    artifacts_dir = tmp_path / "artifacts"
    create_conversation(
        short_name="basef1-2v1",
        query_ids=["1", "2"],
        verify_sf_list=[1],
        max_scale_factor=1,
        artifacts_dir=artifacts_dir,
        benchmark="tpch",
        conversation_dir=conversation_dir,
        sample_query_args_dict={},
        base_data_dir=tmp_path / "real_data_root",
    )
    target_path = conversation_dir / "tpch_basef1-2v1.json"
    data = json.loads(target_path.read_text(encoding="utf-8"))
    stage_items = [item for item in data if isinstance(item, dict)]
    descriptors = [str(item.get("descriptor")) for item in stage_items]
    assert "implement_q1" not in descriptors
    assert "implement_q2" not in descriptors
    q1_stage = next(
        item for item in stage_items
        if item.get("descriptor") == "implement_query_1"
    )
    q2_stage = next(
        item for item in stage_items
        if item.get("descriptor") == "implement_query_2"
    )
    assert "Engine-backed lineitem scan" in q1_stage["text"]
    assert "source directory discovery" in q1_stage["text"]
    assert "part, partsupp, supplier, nation, and region joins" in q2_stage["text"]
    assert "repair the Engine build path first" in q2_stage["text"]
    assert q1_stage["active_unit_id"] == "query:1"
    assert q2_stage["active_unit_id"] == "query:2"
    return None
