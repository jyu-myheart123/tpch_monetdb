import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tpch_monetdb.conversations.scripted_conversation import PromptStep
from tpch_monetdb.run_gen_base_impl_tpch_monetdb import build_query_output_protocol, create_conversation
from tpch_monetdb.utils.generated_query_checks import run_generated_code_checks


def test_build_query_output_protocol_uses_shared_runtime_owned_sink() -> None:
    text = build_query_output_protocol()
    assert "do NOT open `result*.csv` directly" in text
    assert "get_last_query_result().csv_output" in text
    assert "dispatcher/runtime owns the physical `result<RUN_NR>.csv` write" in text


def test_prompt_step_parses_generated_code_check_metadata() -> None:
    step = PromptStep.from_json_value(
        {
            "text": "prompt",
            "descriptor": "implement_query_3",
            "tool_profile": "implement_queries_writeonly",
            "expected_query_id": "3",
            "generated_code_checks": ["query_protocol"],
            "advisory_postconditions": ["storage_plan_contract_complete"],
        }
    )
    assert step.expected_query_id == "3"
    assert step.generated_code_checks == ("query_protocol",)
    assert step.advisory_postconditions == ("storage_plan_contract_complete",)


def test_generated_query_checks_reject_unknown_check_name(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unknown generated_code_check: query_quality"):
        run_generated_code_checks(
            workspace_root=tmp_path,
            expected_query_id=None,
            checks=("query_quality",),
        )


def test_generated_query_checks_reject_direct_result_file_literal(tmp_path: Path) -> None:
    (tmp_path / "query_q3.hpp").write_text(
        'void execute_q3(Engine& engine, const Q3Args& args);\n',
        encoding="utf-8",
    )
    (tmp_path / "query_q3.cpp").write_text(
        '\n'.join(
            (
                '#include "query_q3.hpp"',
                'void execute_q3(Engine& engine, const Q3Args& args) {',
                '    auto* fp = fopen("result1.csv", "w");',
                '    (void)fp; (void)engine; (void)args;',
                '}',
            )
        ),
        encoding="utf-8",
    )

    violations = run_generated_code_checks(
        workspace_root=tmp_path,
        expected_query_id="3",
        checks=("query_protocol",),
    )

    assert any(v.code == "FORBIDDEN_RESULT_FILE_LITERAL" for v in violations)


def test_generated_query_checks_reject_missing_entrypoint(tmp_path: Path) -> None:
    (tmp_path / "query_q9.hpp").write_text(
        'void execute_q8(Engine& engine, const Q8Args& args);\n',
        encoding="utf-8",
    )
    (tmp_path / "query_q9.cpp").write_text(
        "int helper() { return 1; }\n",
        encoding="utf-8",
    )

    violations = run_generated_code_checks(
        workspace_root=tmp_path,
        expected_query_id="9",
        checks=("query_protocol",),
    )

    codes = {v.code for v in violations}
    assert "MISSING_QUERY_ENTRYPOINT" in codes
    assert "MISSING_QUERY_DECLARATION" in codes


def test_generated_query_checks_accept_shared_runtime_owned_protocol(tmp_path: Path) -> None:
    (tmp_path / "query_q3.hpp").write_text(
        '\n'.join(
            (
                '#pragma once',
                'struct Engine;',
                'struct Q3Args;',
                'void execute_q3(Engine& engine, const Q3Args& args);',
            )
        ),
        encoding="utf-8",
    )
    (tmp_path / "query_q3.cpp").write_text(
        '\n'.join(
            (
                '#include "query_q3.hpp"',
                '#include "query_impl.hpp"',
                'void execute_q3(Engine& engine, const Q3Args& args) {',
                '    auto& result = get_last_query_result();',
                '    if (should_materialize_query_output()) {',
                '        result.csv_output = "timestamp,value\\n";',
                '        result.valid = true;',
                '    }',
                '    (void)engine; (void)args;',
                '}',
            )
        ),
        encoding="utf-8",
    )
    (tmp_path / "query_shared_time.hpp").write_text(
        "#pragma once\n",
        encoding="utf-8",
    )

    violations = run_generated_code_checks(
        workspace_root=tmp_path,
        expected_query_id="3",
        checks=("query_protocol",),
    )

    assert violations == []


def test_generated_query_checks_reject_unguarded_csv_materialization(tmp_path: Path) -> None:
    (tmp_path / "query_q3.hpp").write_text(
        '\n'.join(
            (
                '#pragma once',
                'struct Engine;',
                'struct Q3Args;',
                'void execute_q3(Engine& engine, const Q3Args& args);',
            )
        ),
        encoding="utf-8",
    )
    (tmp_path / "query_q3.cpp").write_text(
        '\n'.join(
            (
                '#include "query_q3.hpp"',
                '#include "query_impl.hpp"',
                'void execute_q3(Engine& engine, const Q3Args& args) {',
                '    auto& result = get_last_query_result();',
                '    result.csv_output = "timestamp,value\\n";',
                '    result.valid = true;',
                '    (void)engine; (void)args;',
                '}',
            )
        ),
        encoding="utf-8",
    )

    violations = run_generated_code_checks(
        workspace_root=tmp_path,
        expected_query_id="3",
        checks=("query_protocol",),
    )

    assert any(v.code == "UNGUARDED_CSV_OUTPUT_MATERIALIZATION" for v in violations)
    return None


def test_generated_query_checks_reject_missing_header(tmp_path: Path) -> None:
    (tmp_path / "query_q4.cpp").write_text(
        '\n'.join(
            (
                "void execute_q4(Engine& engine, const Q4Args& args) {",
                "    (void)engine; (void)args;",
                "}",
            )
        ),
        encoding="utf-8",
    )

    violations = run_generated_code_checks(
        workspace_root=tmp_path,
        expected_query_id="4",
        checks=("query_protocol",),
    )

    assert any(v.code == "MISSING_QUERY_HEADER" for v in violations)


def test_generated_query_checks_reject_family_entrypoint_leak(tmp_path: Path) -> None:
    (tmp_path / "query_family_single_groupby.hpp").write_text(
        "void execute_q3(Engine& engine, const Q3Args& args);\n",
        encoding="utf-8",
    )
    (tmp_path / "query_family_single_groupby.cpp").write_text(
        "void execute_q3(Engine& engine, const Q3Args& args) {}\n",
        encoding="utf-8",
    )

    violations = run_generated_code_checks(
        workspace_root=tmp_path,
        expected_query_id=None,
        checks=("query_family_boundary",),
        active_unit_files=("query_family_single_groupby.hpp", "query_family_single_groupby.cpp"),
    )

    assert any(v.code == "QUERY_UNIT_ENTRYPOINT_MISSING" for v in violations)


def test_generated_query_checks_reject_instrumented_final_path(tmp_path: Path) -> None:
    (tmp_path / "query_q8.hpp").write_text(
        "void execute_q8(Engine& engine, const Q8Args& args);\n",
        encoding="utf-8",
    )
    (tmp_path / "query_q8.cpp").write_text(
        "\n".join(
            (
                "void execute_q8(Engine& engine, const Q8Args& args) {",
                "    execute_cpu_max_groupby_instrumented(engine, args);",
                "}",
            )
        ),
        encoding="utf-8",
    )

    violations = run_generated_code_checks(
        workspace_root=tmp_path,
        expected_query_id="8",
        checks=("final_path_integrity",),
    )

    assert any(v.code == "FORBIDDEN_INSTRUMENTED_FINAL_PATH" for v in violations)


def test_generated_query_checks_reject_family_final_path_fallbacks(tmp_path: Path) -> None:
    (tmp_path / "query_q9.hpp").write_text(
        "void execute_q9(Engine& engine, const Q9Args& args);\n",
        encoding="utf-8",
    )
    (tmp_path / "query_q9.cpp").write_text(
        "void execute_q9(Engine& engine, const Q9Args& args) { run_q9(engine, args); }\n",
        encoding="utf-8",
    )
    (tmp_path / "query_family_double_groupby.cpp").write_text(
        "void run_q9(Engine&, const Q9Args&) { /* placeholder */ }\n",
        encoding="utf-8",
    )

    violations = run_generated_code_checks(
        workspace_root=tmp_path,
        expected_query_id="9",
        checks=("final_path_integrity",),
        active_unit_files=("query_family_double_groupby.cpp",),
    )

    assert any(v.code == "FORBIDDEN_FINAL_PATH_PLACEHOLDER" for v in violations)
    assert any(v.file_path.endswith("query_family_double_groupby.cpp") for v in violations)


def test_generated_query_checks_reject_raw_source_reconstruction(tmp_path: Path) -> None:
    (tmp_path / "query_q3.cpp").write_text(
        "\n".join(
            (
                "void execute_q3(Engine& engine, const Q3Args& args) {",
                "    std::ifstream input(\"cpu.ilp\");",
                "    (void)engine; (void)args;",
                "}",
            )
        ),
        encoding="utf-8",
    )

    violations = run_generated_code_checks(
        workspace_root=tmp_path,
        expected_query_id="3",
        checks=("final_path_integrity",),
    )

    assert any(v.code == "FORBIDDEN_RAW_SOURCE_RECONSTRUCTION" for v in violations)


def test_generated_query_checks_reject_empty_csv_output(tmp_path: Path) -> None:
    (tmp_path / "query_q4.cpp").write_text(
        "\n".join(
            (
                "void execute_q4(Engine& engine, const Q4Args& args) {",
                "    get_last_query_result().csv_output.clear();",
                "    get_last_query_result().valid = false;",
                "    (void)engine; (void)args;",
                "}",
            )
        ),
        encoding="utf-8",
    )

    violations = run_generated_code_checks(
        workspace_root=tmp_path,
        expected_query_id="4",
        checks=("final_path_integrity",),
    )

    assert any(v.code == "FORBIDDEN_EMPTY_CSV_OUTPUT" for v in violations)


def test_generated_query_checks_allow_clear_before_populating_csv_output(
    tmp_path: Path,
) -> None:
    (tmp_path / "query_q4.cpp").write_text(
        "\n".join(
            (
                "void execute_q4(Engine& engine, const Q4Args& args) {",
                "    auto& result = get_last_query_result();",
                "    result.csv_output.clear();",
                "    result.csv_output.append(\"timestamp,value\\n\");",
                "    result.valid = true;",
                "    (void)engine; (void)args;",
                "}",
            )
        ),
        encoding="utf-8",
    )

    violations = run_generated_code_checks(
        workspace_root=tmp_path,
        expected_query_id="4",
        checks=("final_path_integrity",),
    )

    assert not any(v.code == "FORBIDDEN_EMPTY_CSV_OUTPUT" for v in violations)


def test_generated_query_checks_reject_manual_registry_fallback(tmp_path: Path) -> None:
    registry_dir = tmp_path / "build" / "generated"
    registry_dir.mkdir(parents=True)
    (registry_dir / "query_registry_generated.cpp").write_text(
        "void dispatch_unimplemented_query(const QueryRequest& request) { (void)request; }\n",
        encoding="utf-8",
    )
    (tmp_path / "query_q5.cpp").write_text(
        "void execute_q5(Engine& engine, const Q5Args& args) { (void)engine; (void)args; }\n",
        encoding="utf-8",
    )

    violations = run_generated_code_checks(
        workspace_root=tmp_path,
        expected_query_id="5",
        checks=("final_path_integrity",),
    )

    assert any(v.code == "FORBIDDEN_REGISTRY_FALLBACK" for v in violations)


def test_generated_query_checks_reject_usage_double_int_cast(tmp_path: Path) -> None:
    (tmp_path / "query_q11.hpp").write_text(
        "void execute_q11(Engine& engine, const Q11Args& args);\n",
        encoding="utf-8",
    )
    (tmp_path / "query_q11.cpp").write_text(
        "\n".join(
            (
                "void execute_q11(Engine& engine, const Q11Args& args) {",
                "    append_int(static_cast<int64_t>(blk.usage_user[idx]));",
                "}",
            )
        ),
        encoding="utf-8",
    )

    violations = run_generated_code_checks(
        workspace_root=tmp_path,
        expected_query_id="11",
        checks=("usage_double_output",),
    )

    assert any(v.code == "UNSAFE_USAGE_INT_CAST" for v in violations)


def test_generated_query_checks_reject_missing_critical_vector_kernel(tmp_path: Path) -> None:
    """Critical vector loop shape is diagnostic and must not prove optimization."""
    violations = run_generated_code_checks(
        workspace_root=tmp_path,
        expected_query_id="8",
        checks=("critical_vector_loop_shape",),
    )

    assert any(
        v.code == "CRITICAL_VECTOR_HOT_LOOP_OWNER_MISSING"
        and v.severity == "diagnostic"
        for v in violations
    )


def test_generated_query_checks_reject_cross_column_vector_pack(tmp_path: Path) -> None:
    """Suspicious vector packs are diagnostics, not positive vectorization proof."""
    (tmp_path / "query_q9.cpp").write_text(
        (
            "void execute_q9(Engine& engine, const Q9Args& args) {}\n"
            "auto v = _mm256_set_pd(block.usage_user[i], block.usage_system[i], "
            "block.usage_idle[i], block.usage_nice[i]);\n"
        ),
        encoding="utf-8",
    )

    violations = run_generated_code_checks(
        workspace_root=tmp_path,
        expected_query_id="9",
        checks=("critical_vector_loop_shape",),
    )

    assert any(
        v.code == "CRITICAL_VECTOR_CROSS_COLUMN_PACK"
        and v.severity == "diagnostic"
        for v in violations
    )


def test_generated_query_checks_family_member_missing_is_diagnostic(tmp_path: Path) -> None:
    (tmp_path / "query_family_single_groupby.hpp").write_text("#pragma once\n", encoding="utf-8")

    violations = run_generated_code_checks(
        workspace_root=tmp_path,
        expected_query_id=None,
        checks=("query_family_boundary",),
        active_unit_files=("query_family_single_groupby.hpp", "query_family_single_groupby.cpp"),
    )

    member_violations = [v for v in violations if v.code == "QUERY_UNIT_MEMBER_MISSING"]
    assert len(member_violations) == 1
    assert member_violations[0].severity == "diagnostic"


def test_query_impl_template_owns_result_run_nr_write() -> None:
    hpp_text = (ROOT / "tpch_monetdb" / "misc" / "tpch" / "templates" / "query_impl.hpp").read_text()
    cpp_text = (ROOT / "tpch_monetdb" / "misc" / "tpch" / "templates" / "query_impl.cpp").read_text()
    assert "struct QueryResult" in hpp_text
    assert "QueryResult& get_last_query_result();" in hpp_text
    assert "enum class QueryOutputMode" in hpp_text
    assert "bool should_materialize_query_output();" in hpp_text
    assert "has_kernel_ms_override" in hpp_text
    assert "kernel_ms_override" in hpp_text
    assert "row_count" in hpp_text
    assert "output_bytes" in hpp_text
    assert "should_materialize_query_output() && g_last_query_result.valid" in cpp_text
    assert "g_last_query_result.has_kernel_ms_override = false;" in cpp_text
    assert "g_last_query_result.kernel_ms_override = 0.0;" in cpp_text
    assert "g_last_query_result.row_count = 0;" in cpp_text
    assert "g_last_query_result.output_bytes = 0;" in cpp_text
    assert "g_last_query_result.has_kernel_ms_override" in cpp_text
    assert "g_last_query_result.kernel_ms_override" in cpp_text
    assert "TPCH_MONETDB_QUERY_OUTPUT_MODE" in cpp_text
    assert "should_materialize_query_output" in cpp_text
    assert 'std::to_string(run_nr + 1)' in cpp_text
    assert 'std::printf("%s | Execution ms: %.3f\\n", x.id.c_str(), kernel_ms);' in cpp_text
    assert 'std::printf("%s | Query ms: %.3f\\n", x.id.c_str(), query_ms);' in cpp_text
    assert 'std::printf("%zu | Execution ms: %.3f\\n", run_nr + 1, kernel_ms);' not in cpp_text
    assert 'std::printf("%zu | Query ms: %.3f\\n", run_nr + 1, query_ms);' not in cpp_text
    return None


def test_base_conversation_emits_query_protocol_checks(tmp_path: Path) -> None:
    conversation_dir = tmp_path / "conversations"
    artifacts_dir = tmp_path / "artifacts"
    create_conversation(
        short_name="basef1-3v1",
        query_ids=["1", "2", "3"],
        verify_sf_list=[1],
        max_scale_factor=1,
        artifacts_dir=artifacts_dir,
        benchmark="tpch",
        conversation_dir=conversation_dir,
        sample_query_args_dict={},
        base_data_dir=tmp_path / "real_data_root",
    )
    target_path = conversation_dir / "tpch_basef1-3v1.json"
    data = json.loads(target_path.read_text(encoding="utf-8"))
    stage_items = [item for item in data if isinstance(item, dict)]
    q1_stage = next(
        item for item in stage_items
        if item.get("descriptor") == "implement_query_1"
    )
    q2_stage = next(
        item for item in stage_items
        if item.get("descriptor") == "implement_query_2"
    )
    q3_stage = next(
        item for item in stage_items
        if item.get("descriptor") == "implement_query_3"
    )
    assert q1_stage["expected_query_id"] == "1"
    assert q1_stage["generated_code_checks"] == [
        "query_protocol",
        "final_path_integrity",
        "usage_double_output",
    ]
    assert q2_stage["expected_query_id"] == "2"
    assert q2_stage["generated_code_checks"] == [
        "query_protocol",
        "final_path_integrity",
        "usage_double_output",
    ]
    assert q3_stage["expected_query_id"] == "3"
    assert q3_stage["generated_code_checks"] == [
        "query_protocol",
        "final_path_integrity",
        "usage_double_output",
    ]


def test_base_conversation_query_stages_do_not_emit_family_boundary_checks(tmp_path: Path) -> None:
    conversation_dir = tmp_path / "conversations"
    artifacts_dir = tmp_path / "artifacts"
    create_conversation(
        short_name="basef1-5v1",
        query_ids=["1", "2", "3", "4", "5"],
        verify_sf_list=[1],
        max_scale_factor=1,
        artifacts_dir=artifacts_dir,
        benchmark="tpch",
        conversation_dir=conversation_dir,
        sample_query_args_dict={},
        base_data_dir=tmp_path / "real_data_root",
    )
    target_path = conversation_dir / "tpch_basef1-5v1.json"
    data = json.loads(target_path.read_text(encoding="utf-8"))
    stage_items = [item for item in data if isinstance(item, dict)]
    stage_checks = {
        item["descriptor"]: item.get("generated_code_checks", [])
        for item in stage_items
        if item.get("descriptor") in {
            *(f"implement_query_{query_id}" for query_id in range(1, 6)),
            *(f"correctness_query_{query_id}" for query_id in range(1, 6)),
        }
    }
    for descriptor in stage_checks:
        assert stage_checks[descriptor] == [
            "query_protocol",
            "final_path_integrity",
            "usage_double_output",
        ]


def test_all_stages_reference_todo_md(tmp_path: Path) -> None:
    conversation_dir = tmp_path / "conversations"
    artifacts_dir = tmp_path / "artifacts"
    create_conversation(
        short_name="basef1-5v1",
        query_ids=["1", "2", "3", "4", "5"],
        verify_sf_list=[1],
        max_scale_factor=1,
        artifacts_dir=artifacts_dir,
        benchmark="tpch",
        conversation_dir=conversation_dir,
        sample_query_args_dict={},
        base_data_dir=tmp_path / "real_data_root",
    )
    target_path = conversation_dir / "tpch_basef1-5v1.json"
    data = json.loads(target_path.read_text(encoding="utf-8"))
    stages_with_todo_guidance = [
        "compile_fix",
        "add_timings",
        "implement_query_1",
        "correctness_query_1",
        "implement_query_2",
        "correctness_query_2",
        "implement_query_3",
        "correctness_query_3",
        "implement_query_4",
        "correctness_query_4",
        "implement_query_5",
        "correctness_query_5",
        "all_queries_correctness",
        "benchmark",
        "optimize_build",
    ]
    stage_items = [item for item in data if isinstance(item, dict)]
    for descriptor in stages_with_todo_guidance:
        stage = next(
            (item for item in stage_items if item.get("descriptor") == descriptor),
            None,
        )
        assert stage is not None, f"Stage {descriptor} not found"
        assert "TODO.md" in stage["text"], (
            f"Stage {descriptor} prompt should reference TODO.md"
        )
