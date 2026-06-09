import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tpch_monetdb.conversations.optimization_instrumentation import (
    InstrumentationPolicy,
    TraceEvidenceSummary,
    build_instrumentation_prompt_metadata,
    check_instrumentation_smoke,
    check_trace_evidence_and_feedback,
    check_trace_mode_smoke,
)
from tpch_monetdb.conversations import optimization_conversation_tpch_monetdb as optimization_module
from tpch_monetdb.conversations.optimization_conversation_tpch_monetdb import (
    QueryOptimizationRecord,
    QueryOutputSplitMeasurement,
    TpchMonetdbOptimizationConversation,
)
from tpch_monetdb.conversations.optimization_validation import CorrectnessCheckSummary
from tpch_monetdb.conversations.tpch_monetdb_prompts_gen import tpch_monetdb_optim_prompt_trace_expert
from tpch_monetdb.tools.stage_tool_policy import build_tool_profiles
from tpch_monetdb.tools.tpch.hardware_counters import HardwareCounterSummary, PerfHotspotSummary
from tpch_monetdb.tools.tpch_monetdb_agent_tools import RecoverableStagePolicyError, StageToolRuntime


def test_optimization_instrumentation_profile_excludes_shell() -> None:
    profile = build_tool_profiles()["optimization_instrumentation"]
    assert "shell" not in profile.tool_names
    assert "cpu_info" in profile.tool_names
    assert "compile" in profile.tool_names
    assert "run" in profile.tool_names


def test_optimization_final_path_gate_rejects_instrumented_entrypoint(tmp_path: Path) -> None:
    conversation = TpchMonetdbOptimizationConversation.__new__(TpchMonetdbOptimizationConversation)
    conversation.run_tool = SimpleNamespace(cwd=tmp_path)
    conversation.compiler_vectorization_summary = {}
    conversation.hardware_counter_summary_by_query = {}
    (tmp_path / "workload_objective.json").write_text(
        json.dumps(
            {
                "objective_id": "obj",
                "query_ids": ["8"],
                "critical_query_ids": ["8"],
                "critical_query_targets": {
                    "8": {
                        "requires_vectorization": True,
                        "requires_pmu": False,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "query_q8.cpp").write_text(
        "void execute_q8() { execute_cpu_max_groupby_instrumented(); }\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="FORBIDDEN_INSTRUMENTED_FINAL_PATH"):
        conversation._reject_invalid_final_paths()
    return None


def test_instrumentation_runtime_scope_tracks_active_query_batch(tmp_path: Path) -> None:
    runtime = StageToolRuntime(tmp_path)
    (tmp_path / "query_q1.cpp").write_text("seed\n", encoding="utf-8")
    (tmp_path / "query_q9.cpp").write_text("seed\n", encoding="utf-8")
    runtime.activate(
        "optimization_instrumentation",
        0,
        "Add Timings for Queries 1",
        prompt_metadata=build_instrumentation_prompt_metadata(["1"]),
    )
    assert "Active query batch: 1" in runtime.generate_stage_hint()
    assert runtime.edit_file("query_q1.cpp", "seed", "fresh", False).startswith("Updated")
    with pytest.raises(RecoverableStagePolicyError, match="outside the active instrumentation scope"):
        runtime.edit_file("query_q9.cpp", "seed", "fresh", False)


def test_instrumentation_runtime_run_scope_tracks_active_query_batch(tmp_path: Path) -> None:
    runtime = StageToolRuntime(tmp_path)
    runtime.activate(
        "optimization_instrumentation",
        0,
        "Add Timings for Queries 1, 2",
        prompt_metadata=build_instrumentation_prompt_metadata(["1", "2"]),
    )
    runtime.validate_run_request(json.dumps({"scale_factor": 1, "optimize": False, "query_id": ["1", "2"]}))
    with pytest.raises(RecoverableStagePolicyError, match="query_id=\\['1', '2'\\]"):
        runtime.validate_run_request(json.dumps({"scale_factor": 1, "optimize": False, "query_id": None}))
    with pytest.raises(RecoverableStagePolicyError, match="query_id=\\['1', '2'\\]"):
        runtime.validate_run_request(json.dumps({"scale_factor": 1, "optimize": False, "query_id": ["1"]}))
    with pytest.raises(RecoverableStagePolicyError, match="query_id=\\['1', '2'\\]"):
        runtime.validate_run_request(json.dumps({"scale_factor": 1, "optimize": False, "query_id": ["3"]}))


def test_runtime_hint_exposes_active_unit_metadata(tmp_path: Path) -> None:
    runtime = StageToolRuntime(tmp_path)
    runtime.activate(
        "optimization_instrumentation",
        0,
        "Add Timings for Family Unit",
        prompt_metadata={
            "active_query_ids": ["3"],
            "active_unit_id": "family:single_groupby:3",
            "active_unit_kind": "family",
            "active_unit_files": ["query_q3.cpp", "query_family_single_groupby.cpp"],
            "active_unit_query_ids": ["3"],
        },
    )
    hint = runtime.generate_stage_hint()
    assert "Active unit: family:single_groupby:3" in hint
    assert "Active unit kind: family" in hint
    assert "Active unit queries: 3" in hint
    return None


def test_family_instrumentation_run_scope_uses_active_unit_query_ids(tmp_path: Path) -> None:
    runtime = StageToolRuntime(tmp_path)
    runtime.activate(
        "optimization_instrumentation",
        0,
        "Add Timings for Family Unit",
        prompt_metadata=build_instrumentation_prompt_metadata(
            ["3", "4", "5", "6", "7"],
            active_unit_id="family:single_groupby:3-4-5-6-7",
            active_unit_kind="family",
            active_unit_files=["query_family_single_groupby.cpp"],
        ),
    )
    runtime.validate_run_request(
        json.dumps(
            {
                "scale_factor": 1,
                "optimize": False,
                "query_id": ["3", "4", "5", "6", "7"],
            }
        )
    )
    with pytest.raises(
        RecoverableStagePolicyError,
        match="query_id=\\['3', '4', '5', '6', '7'\\]",
    ):
        runtime.validate_run_request(
            json.dumps({"scale_factor": 1, "optimize": False, "query_id": ["3"]})
        )


def test_trace_expert_prompt_includes_hardware_counter_evidence_block() -> None:
    prompt = tpch_monetdb_optim_prompt_trace_expert(
        query_id="3",
        constraints_str="constraints",
        expert_knowledge="knowledge",
        trace_summary="trace-summary",
        current_rt_ms=12.0,
        target_rt_ms=6.0,
        sf=1000,
        storage_is_bespoke=True,
        hardware_counter_evidence="cache_miss_rate=0.12; llc_mpki=30.0",
    )
    assert "Hardware Counter Evidence" in prompt
    assert "cache_miss_rate=0.12" in prompt


def test_trace_expert_stage_compacts_hardware_counter_evidence_for_prompt() -> None:
    conversation = TpchMonetdbOptimizationConversation.__new__(TpchMonetdbOptimizationConversation)
    conversation.benchmark_sf = 1000
    conversation.bespoke_storage = True
    conversation.hardware_counter_summary_by_query = {
        "3": {
            "backend": "linux_perf_native",
            "target_cpu": "icelake",
            "counters": {"cycles": 1000.0, "cache-misses": 50.0},
            "derived_metrics": {"cache_miss_rate": 0.05, "llc_mpki": 12.0},
            "perf_hotspots_available": True,
            "perf_sample_count": 7,
            "perf_top_symbols": [("scan_query_3", 5)],
            "perf_top_source_lines": [("query_q3.cpp:42", 3)],
            "perf_top_frames": [("scan_query_3+0x21", 5)],
            "perf_raw_script_excerpt": [
                "db 123 [000] 1.000: 1 cycles:",
                "        7f000000 scan_query_3+0x21 (/workspace/db)",
            ],
            "perf_hotspot_provenance": {
                "record_command": ["perf", "record", "--output", "/tmp/perf.data"],
            },
        }
    }

    stage = conversation._build_query_stage(
        query_id="3",
        mandatory_constraints="constraints",
        trace_summary="trace-summary",
    )
    prompt = stage.get_prompt(100.0)

    assert "cache_miss_rate=0.05" in prompt
    assert "- Perf top symbols: scan_query_3=5" in prompt
    assert "- Perf top source lines: query_q3.cpp:42=3" in prompt
    assert "perf_raw_script_excerpt" not in prompt
    assert "record_command" not in prompt
    assert "db 123 [000]" not in prompt
    return None


def test_trace_expert_stage_includes_hardware_counter_error_in_prompt() -> None:
    conversation = TpchMonetdbOptimizationConversation.__new__(TpchMonetdbOptimizationConversation)
    conversation.benchmark_sf = 1000
    conversation.bespoke_storage = True
    conversation.hardware_counter_summary_by_query = {
        "3": {
            "backend": "linux_perf_native",
            "target_cpu": "icelake",
            "hardware_counter_error": "Missing required hardware-counter events: cache-misses",
        }
    }

    stage = conversation._build_query_stage(
        query_id="3",
        mandatory_constraints="constraints",
        trace_summary="trace-summary",
    )
    prompt = stage.get_prompt(100.0)

    assert "hardware_counter_error" in prompt
    assert "cache-misses" in prompt
    return None


def test_collect_scope_hotspot_analysis_samples_every_query_and_merges_evidence() -> None:
    conversation = TpchMonetdbOptimizationConversation.__new__(TpchMonetdbOptimizationConversation)
    calls: list[str] = []

    def fake_sample(query_id: str) -> optimization_module.TraceHotspotSummary:
        calls.append(query_id)
        return optimization_module.TraceHotspotSummary(
            query_id=query_id,
            issue_class=(
                "cache/layout bound"
                if query_id == "8"
                else "kernel/compute bound"
            ),
            evidence_sufficient=True,
            top_profiles=((f"query_q{query_id}_scan", 100),),
            counters={"rows_scanned": 1000},
            summary_text=f"Query {query_id}: summary",
            sampled_instantiations=(f"{query_id}:inst",),
            sampled_count=1,
            omitted_count=0,
            vectorization_candidate=False,
            hardware_counter_summary={
                "backend": "linux_perf_native",
                "target_cpu": "native",
                "derived_metrics": {
                    "cache_miss_rate": 0.06 if query_id == "8" else 0.05
                },
            },
            compiler_vectorization_summary={},
            change_scope="query",
        )

    conversation._sample_trace_for_query = fake_sample

    (
        summaries_by_query,
        scope_trace_summary,
        scope_issue_class,
        scope_hardware_counter_evidence,
    ) = conversation._collect_scope_hotspot_analysis(["8", "9"])

    assert calls == ["8", "9"]
    assert tuple(summaries_by_query.keys()) == ("8", "9")
    assert "Query 8: summary" in scope_trace_summary
    assert "Query 9: summary" in scope_trace_summary
    assert "Query 8:" in scope_hardware_counter_evidence
    assert "Query 9:" in scope_hardware_counter_evidence
    assert scope_issue_class == "mixed"
    return None


def test_collect_hardware_counter_summary_uses_preflight_provenance() -> None:
    from tpch_monetdb.tools.tpch.hardware_counters import build_hardware_counter_preflight

    conversation = TpchMonetdbOptimizationConversation.__new__(TpchMonetdbOptimizationConversation)
    conversation.hardware_counter_summary_by_query = {}
    conversation.hardware_counter_preflight = build_hardware_counter_preflight(
        backend="linux_perf_native",
        target_cpu="icelake",
        runner_cmd=None,
        host_kernel="5.14.0",
        perf_event_paranoid="1",
        large_sf=1000,
    )

    summary = conversation._collect_hardware_counter_summary("3")

    assert summary["backend"] == "linux_perf_native"
    assert summary["target_cpu"] == "icelake"
    assert summary["provenance"]["large_sf"] == 1000


def test_collect_hardware_counter_summary_executes_capture_when_args_available() -> None:
    from tpch_monetdb.tools.tpch.hardware_counters import build_hardware_counter_preflight

    conversation = TpchMonetdbOptimizationConversation.__new__(TpchMonetdbOptimizationConversation)
    conversation.hardware_counter_summary_by_query = {}
    conversation.benchmark_sf = 100
    conversation.large_sf = 1000
    calls: list[str] = []
    conversation.run_tool = SimpleNamespace(
        run_hardware_counter_capture=lambda **_kwargs: (
            calls.append(_kwargs["stdin_args_data"][0]),
            HardwareCounterSummary(
                backend="linux_perf_native",
                counters={
                    "cycles": 1000.0,
                    "instructions": 2000.0,
                    "cache-misses": 50.0,
                    "LLC-load-misses": 10.0,
                    "dTLB-load-misses": 5.0,
                },
                derived_metrics={"cache_miss_rate": 0.025},
                provenance={"runner": "perf"},
            ),
        )[1],
        run_perf_hotspot_capture=lambda **_kwargs: PerfHotspotSummary(
            backend="linux_perf_native",
            top_symbols=(("scan_query_3", 4), ("aggregate_bucket", 2)),
            top_frames=(("scan_query_3+0x21", 4),),
            raw_script_excerpt=("db 123 [000] 1.000: 1 cycles:",),
            provenance={
                "record_command": ["perf", "record"],
                "capture_scope": "query_loop_only",
                "warmup_completed": True,
                "record_started_after_warmup": True,
                "measured_query_repetitions": 3,
                "measured_batch_size": 3,
            },
        ),
    )
    conversation.hardware_counter_preflight = build_hardware_counter_preflight(
        backend="linux_perf_native",
        target_cpu="icelake",
        runner_cmd=None,
        host_kernel="5.14.0",
        perf_event_paranoid="1",
        large_sf=1000,
    )

    summary = conversation._collect_hardware_counter_summary(
        "3",
        args_string="3 explicit args",
    )

    assert summary["hardware_counters_available"] is True
    assert summary["counters"]["cache-misses"] == 50.0
    assert summary["derived_metrics"]["cache_miss_rate"] == 0.025
    assert summary["perf_hotspots_available"] is True
    assert summary["perf_top_symbols"][0] == ("scan_query_3", 4)
    assert summary["perf_hotspot_provenance"]["record_command"] == ["perf", "record"]
    assert summary["perf_hotspot_provenance"]["capture_scope"] == "query_loop_only"
    assert summary["perf_hotspot_provenance"]["warmup_completed"] is True
    assert summary["perf_hotspot_provenance"]["record_started_after_warmup"] is True
    assert summary["provenance"]["runner"] == "perf"
    assert summary["provenance"]["args_string"] == "3 explicit args"
    cached = conversation._collect_hardware_counter_summary(
        "3",
        args_string="3 explicit args",
    )
    assert cached is summary
    assert calls == ["3 explicit args"]
    return None


def test_collect_hardware_counter_summary_degrades_when_counter_capture_fails() -> None:
    from tpch_monetdb.tools.tpch.hardware_counters import build_hardware_counter_preflight

    conversation = TpchMonetdbOptimizationConversation.__new__(TpchMonetdbOptimizationConversation)
    conversation.hardware_counter_summary_by_query = {}
    conversation.benchmark_sf = 100
    conversation.large_sf = 1000
    conversation.run_tool = SimpleNamespace(
        run_hardware_counter_capture=lambda **_kwargs: (_ for _ in ()).throw(
            RuntimeError("Missing required hardware-counter events: cache-misses")
        ),
    )
    conversation.hardware_counter_preflight = build_hardware_counter_preflight(
        backend="linux_perf_native",
        target_cpu="icelake",
        runner_cmd=None,
        host_kernel="5.14.0",
        perf_event_paranoid="1",
        large_sf=1000,
    )

    summary = conversation._collect_hardware_counter_summary(
        "8",
        args_string="8 explicit args",
    )

    assert summary["hardware_counters_available"] is False
    assert "cache-misses" in summary["hardware_counter_error"]
    assert summary["provenance"]["args_string"] == "8 explicit args"
    assert summary["perf_hotspots_available"] is False


def test_collect_hardware_counter_summary_keeps_counters_when_perf_hotspot_fails() -> None:
    from tpch_monetdb.tools.tpch.hardware_counters import build_hardware_counter_preflight

    conversation = TpchMonetdbOptimizationConversation.__new__(TpchMonetdbOptimizationConversation)
    conversation.hardware_counter_summary_by_query = {}
    conversation.benchmark_sf = 100
    conversation.large_sf = 1000
    conversation.run_tool = SimpleNamespace(
        run_hardware_counter_capture=lambda **_kwargs: HardwareCounterSummary(
            backend="linux_perf_native",
            counters={
                "cycles": 1000.0,
                "instructions": 2000.0,
                "cache-misses": 50.0,
                "LLC-load-misses": 10.0,
                "dTLB-load-misses": 5.0,
            },
            derived_metrics={"cache_miss_rate": 0.025},
            provenance={"runner": "perf"},
        ),
        run_perf_hotspot_capture=lambda **_kwargs: (_ for _ in ()).throw(
            RuntimeError("perf script failed")
        ),
    )
    conversation.hardware_counter_preflight = build_hardware_counter_preflight(
        backend="linux_perf_native",
        target_cpu="icelake",
        runner_cmd=None,
        host_kernel="5.14.0",
        perf_event_paranoid="1",
        large_sf=1000,
    )

    summary = conversation._collect_hardware_counter_summary(
        "9",
        args_string="9 explicit args",
    )

    assert summary["hardware_counters_available"] is True
    assert summary["counters"]["cache-misses"] == 50.0
    assert summary["perf_hotspots_available"] is False
    assert summary["perf_hotspot_error"] == "perf script failed"


def test_hotspot_summary_persists_perf_symbols_source_lines_and_frames(
    tmp_path: Path,
) -> None:
    conversation = TpchMonetdbOptimizationConversation.__new__(TpchMonetdbOptimizationConversation)
    conversation.run_tool = SimpleNamespace(cwd=tmp_path)
    conversation.hardware_counter_summary_by_query = {
        "3": {
            "perf_hotspots_available": True,
            "perf_sample_count": 7,
            "perf_top_symbols": [("scan_query_3", 5)],
            "perf_top_source_lines": [("query_q3.cpp:42", 3)],
            "perf_top_frames": [("scan_query_3+0x21", 5)],
        }
    }

    path = conversation._persist_hotspot_summary(
        [
            QueryOptimizationRecord(
                query_id="3",
                unit_id="family:single_groupby:3",
                unit_query_ids=("3",),
                issue_class="cache/layout bound",
                trace_summary="",
                sampled_instantiations=("q3_i1",),
                stage_name="trace_expert",
                rt_before_s=0.2,
                rt_after_s=0.1,
                written_files=("query_q3.cpp",),
            )
        ]
    )

    text = path.read_text(encoding="utf-8")
    assert "- Perf samples: 7" in text
    assert "- Perf top symbols: scan_query_3=5" in text
    assert "- Perf top source lines: query_q3.cpp:42=3" in text
    assert "- Perf top call-stack frames: scan_query_3+0x21=5" in text
    return None


def test_hotspot_summary_includes_output_split_materialization_evidence(
    tmp_path: Path,
) -> None:
    """热点摘要应显式区分 full CSV 和 no-output 测量。"""
    conversation = TpchMonetdbOptimizationConversation.__new__(TpchMonetdbOptimizationConversation)
    conversation.run_tool = SimpleNamespace(cwd=tmp_path)
    conversation.hardware_counter_summary_by_query = {}
    conversation.output_split_by_query = {
        "12": QueryOutputSplitMeasurement(
            query_id="12",
            full_csv_s=1.378,
            no_output_s=0.064,
            materialization_s=1.314,
            materialization_ratio=0.9536,
        )
    }

    path = conversation._persist_hotspot_summary(
        [
            QueryOptimizationRecord(
                query_id="12",
                unit_id="family:high_cpu_groupby:12",
                unit_query_ids=("12",),
                issue_class="materialization/output bound",
                trace_summary="",
                sampled_instantiations=("q12_i1",),
                stage_name="trace_expert",
                rt_before_s=0.066,
                rt_after_s=0.064,
                written_files=("query_family_high_cpu_groupby.cpp",),
            )
        ]
    )

    text = path.read_text(encoding="utf-8")
    assert "- Status: complete" in text
    assert "Optimization runtime (no_output): 66.000ms -> 64.000ms" in text
    assert "full_csv=1378.000ms" in text
    assert "no_output=64.000ms" in text
    assert "materialization=1314.000ms" in text
    assert "materialization_ratio=0.954" in text
    return None


def test_global_control_artifact_anchor_snapshots_dirty_hotspot_summary(
    tmp_path: Path,
) -> None:
    """global control artifact 写入后必须先进入 snapshot 再被 restore 使用。"""
    conversation = TpchMonetdbOptimizationConversation.__new__(TpchMonetdbOptimizationConversation)
    hotspot_path = tmp_path / "optimization_hotspot_summary.md"
    hotspot_path.write_text("# Optimization Hotspot Summary\n", encoding="utf-8")

    class FakeSnapshotter:
        current_hash = "base"

        def __init__(self) -> None:
            self.snapshot_calls: list[str] = []
            return None

        def is_dirty(self) -> bool:
            return True

        def snapshot(self, name: str) -> tuple[str, str]:
            self.snapshot_calls.append(name)
            self.current_hash = "anchored"
            return "base", "anchored"

    snapshotter = FakeSnapshotter()
    conversation.git_snapshotter = snapshotter

    anchored = conversation._anchor_global_control_artifacts(hotspot_path)

    assert anchored == "anchored"
    assert snapshotter.snapshot_calls == ["optimization_hotspot_summary_control"]
    return None


def test_global_control_artifact_restore_loss_fails_fast(tmp_path: Path) -> None:
    """global restore 若回滚掉热点摘要，应在 hypothesis 前结构化失败。"""
    conversation = TpchMonetdbOptimizationConversation.__new__(TpchMonetdbOptimizationConversation)
    hotspot_path = tmp_path / "optimization_hotspot_summary.md"

    with pytest.raises(RuntimeError, match="CONTROL_ARTIFACT_RESTORED_AWAY"):
        conversation._ensure_hotspot_summary_available_after_restore(
            hotspot_path,
            "base",
        )
    return None


def test_hotspot_summary_persists_perf_failure_status_and_provenance(
    tmp_path: Path,
) -> None:
    conversation = TpchMonetdbOptimizationConversation.__new__(TpchMonetdbOptimizationConversation)
    conversation.run_tool = SimpleNamespace(cwd=tmp_path)
    conversation.hardware_counter_summary_by_query = {
        "8": {
            "hardware_counters_available": True,
            "perf_hotspots_available": False,
            "perf_sample_count": 0,
            "perf_hotspot_error": "perf script failed",
            "perf_hotspot_provenance": {
                "capture_scope": "query_loop_only",
                "warmup_completed": True,
                "record_started_after_warmup": True,
                "attached_pids": [4242, 4243],
                "measured_query_repetitions": 5,
                "measured_batch_size": 5,
                "source_line_decode": False,
            },
        }
    }

    path = conversation._persist_hotspot_summary(
        [
            QueryOptimizationRecord(
                query_id="8",
                unit_id="family:host_scan:8",
                unit_query_ids=("8",),
                issue_class="evidence_insufficient",
                trace_summary="",
                sampled_instantiations=("q8_i1",),
                stage_name="trace_expert",
                rt_before_s=0.2,
                rt_after_s=0.2,
                written_files=(),
            )
        ]
    )

    text = path.read_text(encoding="utf-8")
    assert "perf_hotspot_error=perf script failed" in text
    assert "capture_scope=query_loop_only" in text
    assert "attached_pids=[4242, 4243]" in text
    return None


def test_add_timings_prompts_are_instrumentation_only() -> None:
    collect_text = (
        ROOT
        / "tpch_monetdb"
        / "conversations"
        / "prompts"
        / "optimization"
        / "instrumentation"
        / "tpch_monetdb_optim_add_timings_collect_stats.txt"
    ).read_text()
    per_query_text = (
        ROOT
        / "tpch_monetdb"
        / "conversations"
        / "prompts"
        / "optimization"
        / "instrumentation"
        / "tpch_monetdb_optim_add_timings_per_query.txt"
    ).read_text()
    trace_to_file_text = (
        ROOT
        / "tpch_monetdb"
        / "conversations"
        / "prompts"
        / "optimization"
        / "instrumentation"
        / "trace_to_file.txt"
    ).read_text()
    assert "Only change tracing / profiling support" in collect_text
    assert "Do NOT change query semantics" in collect_text
    assert "Only instrumentation changes are allowed" in per_query_text
    assert "stop immediately" in per_query_text
    assert "TPCH_MONETDB_TRACE_OUTPUT_PATH" in trace_to_file_text
    assert "overwrite mode" in trace_to_file_text


@pytest.mark.asyncio
async def test_check_instrumentation_smoke_repairs_trace_disabled_failure() -> None:
    observed: list[tuple[tuple[str, ...], bool, tuple[int, ...]]] = []
    exec_calls: list[dict[str, object]] = []
    attempts = {"count": 0}
    policy = InstrumentationPolicy(
        smoke_scale_factors=(1,),
        full_scale_factors=(1, 10),
        repair_attempts=2,
        batch_size=3,
    )

    async def fake_check_correctness(
        qids: list[str],
        trace_mode: bool,
        scale_factors: tuple[int, ...],
    ) -> bool:
        attempts["count"] += 1
        observed.append((tuple(qids), trace_mode, scale_factors))
        return attempts["count"] >= 2

    async def fake_exec(prompt, descriptor, max_turns=None, tool_profile=None, prompt_metadata=None):
        exec_calls.append(
            {
                "prompt": prompt,
                "descriptor": descriptor,
                "max_turns": max_turns,
                "tool_profile": tool_profile,
                "prompt_metadata": prompt_metadata,
            }
        )
        return None

    await check_instrumentation_smoke(
        qids=["1"],
        policy=policy,
        max_turns=160,
        check_correctness_fn=fake_check_correctness,
        exec_fn=fake_exec,
    )

    assert observed == [(("1",), False, (1,)), (("1",), False, (1,))]
    assert len(exec_calls) == 1
    assert exec_calls[0]["descriptor"] == "Fix Instrumentation Smoke"
    assert exec_calls[0]["tool_profile"] == "optimization_instrumentation"
    assert exec_calls[0]["prompt_metadata"] == {"active_query_ids": ["1"]}


@pytest.mark.asyncio
async def test_check_trace_mode_smoke_repairs_trace_enabled_failure() -> None:
    observed: list[tuple[tuple[str, ...], bool, tuple[int, ...]]] = []
    exec_calls: list[dict[str, object]] = []
    attempts = {"count": 0}
    policy = InstrumentationPolicy(
        smoke_scale_factors=(1,),
        full_scale_factors=(1, 10),
        repair_attempts=2,
        batch_size=2,
    )

    async def fake_check_correctness(
        qids: list[str],
        trace_mode: bool,
        scale_factors: tuple[int, ...],
    ) -> bool:
        attempts["count"] += 1
        observed.append((tuple(qids), trace_mode, scale_factors))
        return attempts["count"] >= 2

    async def fake_exec(
        prompt: str,
        descriptor: str,
        max_turns: int | None = None,
        tool_profile: str | None = None,
        prompt_metadata=None,
    ) -> None:
        exec_calls.append(
            {
                "prompt": prompt,
                "descriptor": descriptor,
                "max_turns": max_turns,
                "tool_profile": tool_profile,
                "prompt_metadata": prompt_metadata,
            }
        )
        return None

    await check_trace_mode_smoke(
        qids=["1", "2", "3"],
        policy=policy,
        max_turns=160,
        check_correctness_fn=fake_check_correctness,
        exec_fn=fake_exec,
    )

    assert observed == [(("1", "2"), True, (1,)), (("1", "2"), True, (1,))]
    assert exec_calls[0]["descriptor"] == "Fix Trace-Mode Instrumentation Smoke"
    assert exec_calls[0]["tool_profile"] == "optimization_instrumentation"
    assert exec_calls[0]["prompt_metadata"] == {"active_query_ids": ["1", "2"]}
    return None


@pytest.mark.asyncio
async def test_trace_evidence_runtime_failure_does_not_feedback_loop() -> None:
    policy = InstrumentationPolicy(
        smoke_scale_factors=(1,),
        full_scale_factors=(1,),
        repair_attempts=3,
        batch_size=3,
    )
    exec_calls: list[str] = []

    def fake_summarize(_qids: list[str]) -> TraceEvidenceSummary:
        return TraceEvidenceSummary(
            qids=("1",),
            sufficient=False,
            message="Query 1: raw trace execution failed with RUNNER_SEGFAULT",
            insufficient_qids=("1",),
            failure_code="RUNNER_SEGFAULT",
            raw_execution_ok=False,
            trace_file_present=False,
        )

    async def fake_exec(*_args, **_kwargs) -> None:
        exec_calls.append("called")
        return None

    summary = await check_trace_evidence_and_feedback(
        qids=["1"],
        policy=policy,
        summarize_trace_fn=fake_summarize,
        exec_fn=fake_exec,
        max_turns=160,
    )

    assert summary.degraded is True
    assert summary.sufficient is False
    assert "RUNNER_SEGFAULT" in summary.message
    assert len(exec_calls) == policy.evidence_repair_attempts
    return None


class _StopAfterInstrumentation(Exception):
    pass


@pytest.mark.asyncio
async def test_optimization_run_uses_instrumentation_profile_for_add_timings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    queries_path = tmp_path / "queries.txt"
    queries_path.write_text("SELECT 1;\n", encoding="utf-8")

    conversation = TpchMonetdbOptimizationConversation.__new__(TpchMonetdbOptimizationConversation)
    conversation.query_ids = ["1", "2", "3"]
    conversation.bespoke_storage = True
    conversation.run_tool = SimpleNamespace(cwd=tmp_path)
    conversation.required_validation_sf_list = [1, 10, 100]
    conversation.benchmark_sf = 100
    conversation.best_rt_log = {}
    conversation.query_rt_log = {}
    conversation.revert_on_regression = True
    conversation.regression_tolerance = 0.05

    captured: dict[str, object] = {}

    async def fake_exec(
        _prompt: str,
        descriptor: str | None,
        max_turns: int | None = None,
        tool_profile: str | None = None,
        prompt_metadata=None,
    ) -> None:
        if descriptor == "Add Timings for Queries 1, 2, 3":
            captured["max_turns"] = max_turns
            captured["tool_profile"] = tool_profile
            captured["prompt_metadata"] = prompt_metadata
            raise _StopAfterInstrumentation()
        return None

    monkeypatch.setattr(conversation, "_exec", fake_exec)
    monkeypatch.setattr(conversation, "_collect_baselines_at_checkpoint", lambda: None)
    monkeypatch.setattr(conversation, "_delete_result_csvs", lambda _workspace: None)
    monkeypatch.setattr(
        optimization_module,
        "run_required_correctness_checks",
        lambda *_args, **_kwargs: CorrectnessCheckSummary(
            success=True,
            message="ok",
            metrics={"validation/correct": True},
            failed_scale_factor=None,
        ),
    )

    with pytest.raises(_StopAfterInstrumentation):
        await conversation.run()

    assert captured["max_turns"] == 160
    assert captured["tool_profile"] == "optimization_instrumentation"
    assert captured["prompt_metadata"] == {"active_query_ids": ["1", "2", "3"]}
