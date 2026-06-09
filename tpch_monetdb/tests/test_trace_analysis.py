import pytest
from pathlib import Path
from types import SimpleNamespace

from tpch_monetdb.conversations.optimization_conversation_tpch_monetdb import TpchMonetdbOptimizationConversation
from tpch_monetdb.tools.tpch.hardware_counters import build_hardware_counter_preflight
from tpch_monetdb.tools.tpch.run import RunWorkerResult
from tpch_monetdb.tools.tpch.trace_analysis import (
    TraceSample,
    parse_trace_log,
    summarize_trace_sample,
    summarize_trace_file,
    merge_trace_summaries,
    classify_trace_issue,
)


class TestParseTraceLog:
    def test_parse_profiles_and_counts(self):
        text = "PROFILE query_q1 1000000\nCOUNT rows_scanned 5000\n"
        sample = parse_trace_log(text)
        assert sample.profile_ns_by_name["query_q1"] == (1000000,)
        assert sample.count_by_name["rows_scanned"] == (5000,)

    def test_parse_malformed_lines_skipped(self):
        text = "PROFILE query_q1\nPROFILE query_q1 abc\nINVALID\n"
        sample = parse_trace_log(text)
        assert not sample.profile_ns_by_name
        assert not sample.count_by_name

    def test_multiple_values_stored(self):
        text = "COUNT rows_scanned 100\nCOUNT rows_scanned 200\n"
        sample = parse_trace_log(text)
        assert sample.count_by_name["rows_scanned"] == (100, 200)

    def test_summarize_trace_file_rejects_oversized_trace(
        self,
        tmp_path,
        monkeypatch,
    ) -> None:
        """trace summary 遇到过大原始 trace 时应拒绝全量读取。"""
        trace_path = tmp_path / "tracing_output.log"
        trace_path.write_text("PROFILE query_q1_scan 1\n" * 4, encoding="utf-8")
        monkeypatch.setattr(
            "tpch_monetdb.tools.tpch.trace_analysis.TRACE_SUMMARY_MAX_BYTES",
            8,
        )

        summary = summarize_trace_file(query_id="1", trace_path=trace_path)

        assert summary.issue_class == "evidence_insufficient"
        assert summary.evidence_sufficient is False
        assert "exceeding the safe summary limit" in summary.summary_text
        return None


class TestClassifyTraceIssue:
    def test_insufficient_when_few_profiles(self):
        top = (("query_q1", 1000),)
        counters = {}
        assert classify_trace_issue(top, counters) == "evidence_insufficient"

    def test_output_bound_when_output_in_top(self):
        top = (
            ("query_q1_output", 5000), ("query_q1", 2000), ("query_q1_scan", 1000),
            ("query_q1_filter", 500), ("query_q1_agg", 300),
        )
        counters = {}
        assert classify_trace_issue(top, counters) == "materialization/output bound"

    def test_access_path_bound_when_high_scan_low_output(self):
        top = (
            ("query_q1_scan", 5000), ("query_q1", 2000), ("query_q1_filter", 1000),
            ("query_q1_init", 500), ("query_q1_cleanup", 300),
        )
        counters = {"rows_scanned": 10000, "query_output_rows": 100}
        assert classify_trace_issue(top, counters) == "layout/access-path bound"

    def test_kernel_compute_bound(self):
        top = (
            ("query_q1_aggregate", 5000), ("query_q1_scan", 2000), ("query_q1", 1000),
            ("query_q1_init", 500), ("query_q1_prepare", 300),
        )
        counters = {"rows_scanned": 1000, "query_output_rows": 500}
        assert classify_trace_issue(top, counters) == "kernel/compute bound"

    def test_output_bound_when_high_output_ratio(self):
        top = (
            ("query_q1_scan", 5000), ("query_q1", 2000), ("query_q1_filter", 1000),
            ("query_q1_init", 500), ("query_q1_dispatch", 300),
        )
        counters = {"rows_scanned": 1000, "query_output_rows": 900}
        assert classify_trace_issue(top, counters) == "materialization/output bound"

    def test_cache_layout_bound_when_llc_pressure_high(self):
        top = (
            ("query_q8_scan", 5000), ("query_q8", 2000), ("query_q8_filter", 1000),
            ("query_q8_init", 500), ("query_q8_cleanup", 300),
        )
        counters = {"cache_miss_rate": 0.10, "llc_mpki": 25.0}
        assert classify_trace_issue(top, counters) == "cache/layout bound"

    def test_tlb_allocation_bound_when_dtlb_mpki_high(self):
        top = (
            ("query_q11_scan", 5000), ("query_q11", 2000), ("query_q11_filter", 1000),
            ("query_q11_init", 500), ("query_q11_cleanup", 300),
        )
        counters = {"dtlb_mpki": 6.0}
        assert classify_trace_issue(top, counters) == "tlb/allocation bound"

    def test_branch_filter_bound_when_branch_miss_rate_high(self):
        top = (
            ("query_q15_filter", 5000), ("query_q15", 2000), ("query_q15_scan", 1000),
            ("query_q15_init", 500), ("query_q15_cleanup", 300),
        )
        counters = {"branch_miss_rate": 0.05}
        assert classify_trace_issue(top, counters) == "branch/filter bound"


class TestSummarizeTraceSample:
    def test_evidence_insufficient_for_single_profile(self):
        sample = TraceSample(
            profile_ns_by_name={"query_q1": (1000000,)},
            count_by_name={},
        )
        summary = summarize_trace_sample(query_id="q1", sample=sample)
        assert summary.issue_class == "evidence_insufficient"
        assert not summary.evidence_sufficient

    def test_summary_includes_sampled_instantiation(self):
        sample = TraceSample(
            profile_ns_by_name={
                "query_q1": (1000000,),
                "query_q1_scan": (500000,),
                "query_q1_aggregate": (300000,),
                "query_q1_filter": (100000,),
                "query_q1_output": (50000,),
            },
            count_by_name={"rows_scanned": (5000,), "query_output_rows": (500,)},
        )
        summary = summarize_trace_sample(
            query_id="q1",
            sample=sample,
            instantiation_id="inst_001",
            args_string="q1 --from=2025-01-01 --to=2025-06-01",
        )
        assert summary.sampled_instantiations == ("inst_001",)
        assert summary.sampled_count == 1

    def test_summary_records_vectorization_and_change_scope(self):
        sample = TraceSample(
            profile_ns_by_name={
                "query_q3_scan": (1000000,),
                "query_q3_filter": (500000,),
                "query_q3_bucket": (300000,),
                "query_q3_output": (100000,),
                "query_q3_init": (50000,),
            },
            count_by_name={"rows_scanned": (5000,), "query_output_rows": (50,)},
        )
        summary = summarize_trace_sample(
            query_id="3",
            sample=sample,
            hardware_counter_summary={"derived_metrics": {"cache_miss_rate": 0.08}},
            compiler_vectorization_summary={"missed_loops": 2},
        )
        assert summary.vectorization_candidate is True
        assert summary.change_scope == "family"
        assert "Change scope: family" in summary.summary_text

    def test_summary_text_includes_perf_hotspot_symbols(self) -> None:
        sample = TraceSample(
            profile_ns_by_name={
                "query_q3_scan": (1000000,),
                "query_q3_filter": (500000,),
                "query_q3_bucket": (300000,),
                "query_q3_output": (100000,),
                "query_q3_init": (50000,),
            },
            count_by_name={"rows_scanned": (5000,), "query_output_rows": (50,)},
        )
        summary = summarize_trace_sample(
            query_id="3",
            sample=sample,
            hardware_counter_summary={
                "derived_metrics": {"cache_miss_rate": 0.08},
                "perf_hotspots_available": True,
                "perf_top_symbols": [("scan_query_3", 4), ("aggregate_bucket", 2)],
                "perf_top_frames": [("scan_query_3+0x21", 4)],
                "perf_sample_count": 5,
            },
        )
        assert "Perf hotspots:" in summary.summary_text
        assert "Perf top symbols:" in summary.summary_text
        assert "- scan_query_3: 4 samples (80.0%)" in summary.summary_text
        assert "Perf top call-stack frames:" in summary.summary_text
        return None


class TestMergeTraceSummaries:
    def test_merge_preserves_instantiations(self):
        from tpch_monetdb.tools.tpch.trace_analysis import TraceHotspotSummary
        s1 = TraceHotspotSummary(
            query_id="q1",
            issue_class="kernel/compute bound",
            evidence_sufficient=True,
            top_profiles=(
                ("query_q1_scan", 5000), ("query_q1", 2000), ("query_q1_aggregate", 1000),
                ("query_q1_filter", 500), ("query_q1_output", 300),
            ),
            counters={"rows_scanned": 1000},
            summary_text="",
            sampled_instantiations=("inst_001",),
            sampled_count=1,
        )
        s2 = TraceHotspotSummary(
            query_id="q1",
            issue_class="kernel/compute bound",
            evidence_sufficient=True,
            top_profiles=(
                ("query_q1_scan", 4000), ("query_q1", 1500), ("query_q1_aggregate", 800),
                ("query_q1_filter", 400), ("query_q1_output", 200),
            ),
            counters={"rows_scanned": 800},
            summary_text="",
            sampled_instantiations=("inst_002",),
            sampled_count=1,
        )
        merged = merge_trace_summaries(query_id="q1", summaries=[s1, s2], omitted_count=1)
        assert merged.sampled_count == 2
        assert set(merged.sampled_instantiations) == {"inst_001", "inst_002"}
        assert merged.omitted_count == 1


class TestTraceEvidenceRawExecution:
    def test_trace_evidence_uses_raw_worker_with_manifest_args(
        self,
        tmp_path: Path,
    ) -> None:
        class FakeRunTool:
            def __init__(self) -> None:
                self.cwd = tmp_path
                self.raw_calls: list[dict[str, object]] = []
                self.run_calls = 0

            def run_raw_worker(self, **kwargs) -> RunWorkerResult:
                self.raw_calls.append(kwargs)
                trace_text = (
                    "PROFILE query_q1_scan 100\n"
                    "PROFILE query_q1_aggregate 80\n"
                    "PROFILE query_q1_output 30\n"
                    "PROFILE query_q1_filter 20\n"
                    "PROFILE query_q1_init 10\n"
                    "COUNT rows_scanned 1000\n"
                )
                (self.cwd / "tracing_output.log").write_text(trace_text)
                return RunWorkerResult(msg="raw ok", resp="resp", out="", err="")

            def run_worker(self, **_kwargs) -> RunWorkerResult:
                self.run_calls += 1
                return RunWorkerResult(msg="validator cache hit")

        conversation = TpchMonetdbOptimizationConversation.__new__(TpchMonetdbOptimizationConversation)
        conversation.benchmark_sf = 100
        conversation.run_tool = FakeRunTool()
        conversation.manifest = SimpleNamespace(
            get_instantiations_for_query=lambda query_id, scale_factor: [
                SimpleNamespace(
                    instantiation_id=f"q{query_id}_i1",
                    args_string=f"{query_id} explicit args",
                )
            ]
        )

        summary = conversation._summarize_trace_evidence_for_queries(["1"])

        assert summary.sufficient is True
        assert summary.raw_execution_ok is True
        assert conversation.run_tool.run_calls == 0
        assert conversation.run_tool.raw_calls[0]["stdin_args_data"] == ["1 explicit args"]
        assert conversation.run_tool.raw_calls[0]["trace_mode"] is True
        assert conversation.run_tool.raw_calls[0]["output_mode"] == "no_output"
        return None

    def test_trace_evidence_missing_file_is_structured_failure(
        self,
        tmp_path: Path,
    ) -> None:
        class FakeRunTool:
            def __init__(self) -> None:
                self.cwd = tmp_path

            def run_raw_worker(self, **_kwargs) -> RunWorkerResult:
                return RunWorkerResult(msg="raw ok", resp="resp", out="", err="")

        conversation = TpchMonetdbOptimizationConversation.__new__(TpchMonetdbOptimizationConversation)
        conversation.benchmark_sf = 100
        conversation.run_tool = FakeRunTool()
        conversation.manifest = SimpleNamespace(
            get_instantiations_for_query=lambda query_id, scale_factor: [
                SimpleNamespace(
                    instantiation_id=f"q{query_id}_i1",
                    args_string=f"{query_id} explicit args",
                )
            ]
        )

        summary = conversation._summarize_trace_evidence_for_queries(["1"])

        assert summary.sufficient is False
        assert summary.raw_execution_ok is True
        assert summary.trace_file_present is False
        assert summary.insufficient_qids == ("1",)
        assert "tracing_output.log missing" in summary.message
        return None

    def test_trace_evidence_does_not_collect_hardware_counter_summary(
        self,
        tmp_path: Path,
    ) -> None:
        class FakeRunTool:
            def __init__(self) -> None:
                self.cwd = tmp_path

            def run_raw_worker(self, **_kwargs) -> RunWorkerResult:
                trace_text = (
                    "PROFILE query_q1_scan 100\n"
                    "PROFILE query_q1_aggregate 80\n"
                    "PROFILE query_q1_output 30\n"
                    "PROFILE query_q1_filter 20\n"
                    "PROFILE query_q1_init 10\n"
                    "COUNT rows_scanned 1000\n"
                )
                (self.cwd / "tracing_output.log").write_text(
                    trace_text,
                    encoding="utf-8",
                )
                return RunWorkerResult(msg="raw ok", resp="resp", out="", err="")

        conversation = TpchMonetdbOptimizationConversation.__new__(
            TpchMonetdbOptimizationConversation
        )
        conversation.benchmark_sf = 100
        conversation.run_tool = FakeRunTool()
        conversation.manifest = SimpleNamespace(
            get_instantiations_for_query=lambda query_id, scale_factor: [
                SimpleNamespace(
                    instantiation_id=f"q{query_id}_i1",
                    args_string=f"{query_id} explicit args",
                )
            ]
        )
        conversation._collect_hardware_counter_summary = (
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("trace_evidence must not collect PMU/perf")
            )
        )
        conversation._collect_compiler_vectorization_summary = (
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("trace_evidence must not collect vectorization")
            )
        )

        summary = conversation._summarize_trace_evidence_for_queries(["1"])

        assert summary.sufficient is True
        assert summary.raw_execution_ok is True
        return None

    def test_trace_evidence_segfault_is_not_coverage_feedback(
        self,
        tmp_path: Path,
    ) -> None:
        class FakeRunTool:
            def __init__(self) -> None:
                self.cwd = tmp_path

            def run_raw_worker(self, **_kwargs) -> RunWorkerResult:
                return RunWorkerResult(
                    msg="resp=exit_code: 0 signal: 11\nExpected output file missing",
                    resp="exit_code: 0 signal: 11",
                    out="",
                    err="",
                )

        conversation = TpchMonetdbOptimizationConversation.__new__(TpchMonetdbOptimizationConversation)
        conversation.benchmark_sf = 100
        conversation.run_tool = FakeRunTool()
        conversation.manifest = SimpleNamespace(
            get_instantiations_for_query=lambda query_id, scale_factor: [
                SimpleNamespace(
                    instantiation_id=f"q{query_id}_i1",
                    args_string=f"{query_id} explicit args",
                )
            ]
        )

        summary = conversation._summarize_trace_evidence_for_queries(["1"])

        assert summary.sufficient is False
        assert summary.raw_execution_ok is False
        assert summary.failure_code == "RUNNER_SEGFAULT"
        assert "raw trace execution failed" in summary.message
        return None

    def test_trace_evidence_child_terminates_with_clean_exit_is_not_segfault(
        self,
        tmp_path: Path,
    ) -> None:
        class FakeRunTool:
            def __init__(self) -> None:
                self.cwd = tmp_path

            def run_raw_worker(self, **_kwargs) -> RunWorkerResult:
                trace_text = (
                    "PROFILE scan_ns 100\n"
                    "PROFILE aggregate_ns 80\n"
                    "PROFILE output_ns 60\n"
                    "PROFILE filter_ns 40\n"
                    "PROFILE init_ns 20\n"
                    "COUNT rows_scanned 1000\n"
                    "COUNT query_output_rows 10\n"
                )
                (self.cwd / "tracing_output.log").write_text(trace_text, encoding="utf-8")
                return RunWorkerResult(
                    msg=(
                        "2 | Execution ms: 0.096\n"
                        "2 | Query ms: 0.096\n"
                        "got: run\n"
                        "query done\n"
                        "exit_code: 0 signal: 0\n"
                    ),
                    resp="exit_code: 0 signal: 0",
                    out="query done",
                    err="./build/libquery.so child terminates\n./build/libbuilder.so child terminates\n",
                )

        conversation = TpchMonetdbOptimizationConversation.__new__(TpchMonetdbOptimizationConversation)
        conversation.benchmark_sf = 100
        conversation.run_tool = FakeRunTool()
        conversation.manifest = SimpleNamespace(
            get_instantiations_for_query=lambda query_id, scale_factor: [
                SimpleNamespace(
                    instantiation_id=f"q{query_id}_i1",
                    args_string=f"{query_id} explicit args",
                )
            ]
        )

        summary = conversation._summarize_trace_evidence_for_queries(["1"])

        assert summary.sufficient is True
        assert summary.raw_execution_ok is True
        assert summary.failure_code is None
        return None

    def test_trace_evidence_nonzero_exit_is_raw_execution_failure(
        self,
        tmp_path: Path,
    ) -> None:
        class FakeRunTool:
            def __init__(self) -> None:
                self.cwd = tmp_path

            def run_raw_worker(self, **_kwargs) -> RunWorkerResult:
                return RunWorkerResult(
                    msg="stderr: trace setup failed\nexit_code: 1 signal: 0",
                    resp="exit_code: 1 signal: 0",
                    out="",
                    err="trace setup failed",
                )

        conversation = TpchMonetdbOptimizationConversation.__new__(TpchMonetdbOptimizationConversation)
        conversation.benchmark_sf = 100
        conversation.run_tool = FakeRunTool()
        conversation.manifest = SimpleNamespace(
            get_instantiations_for_query=lambda query_id, scale_factor: [
                SimpleNamespace(
                    instantiation_id=f"q{query_id}_i1",
                    args_string=f"{query_id} explicit args",
                )
            ]
        )

        summary = conversation._summarize_trace_evidence_for_queries(["1"])

        assert summary.sufficient is False
        assert summary.raw_execution_ok is False
        assert summary.failure_code == "TRACE_RAW_EXECUTION_FAILED"
        assert "exit_code=1, signal=0" in summary.message
        assert "tracing_output.log missing" not in summary.message
        return None

    def test_sample_trace_for_query_raises_on_raw_execution_failure(
        self,
        tmp_path: Path,
    ) -> None:
        class FakeRunTool:
            def __init__(self) -> None:
                self.cwd = tmp_path

            def run_raw_worker(self, **_kwargs) -> RunWorkerResult:
                return RunWorkerResult(
                    msg="exit_code: 1 signal: 0",
                    resp="exit_code: 1 signal: 0",
                    out="",
                    err="",
                )

        inst = SimpleNamespace(instantiation_id="q1_i1", args_string="1 args")
        conversation = TpchMonetdbOptimizationConversation.__new__(TpchMonetdbOptimizationConversation)
        conversation.benchmark_sf = 100
        conversation.run_tool = FakeRunTool()
        conversation.manifest = SimpleNamespace(
            get_instantiations_for_query=lambda query_id, scale_factor: [inst]
        )

        with pytest.raises(RuntimeError, match="TRACE_RAW_EXECUTION_FAILED|exit_code=1"):
            conversation._sample_trace_for_query("1")
        return None

    def test_sample_trace_for_query_allows_child_terminates_with_clean_exit(
        self,
        tmp_path: Path,
    ) -> None:
        class FakeRunTool:
            def __init__(self) -> None:
                self.cwd = tmp_path

            def run_raw_worker(self, **_kwargs) -> RunWorkerResult:
                (self.cwd / "tracing_output.log").write_text(
                    (
                        "PROFILE scan_ns 100\n"
                        "PROFILE aggregate_ns 80\n"
                        "PROFILE output_ns 60\n"
                        "PROFILE filter_ns 40\n"
                        "PROFILE init_ns 20\n"
                        "COUNT rows_scanned 1000\n"
                        "COUNT query_output_rows 10\n"
                    ),
                    encoding="utf-8",
                )
                return RunWorkerResult(
                    msg=(
                        "1 | Execution ms: 0.101\n"
                        "1 | Query ms: 0.101\n"
                        "query done\n"
                        "exit_code: 0 signal: 0\n"
                    ),
                    resp="exit_code: 0 signal: 0",
                    out="query done",
                    err="./build/libquery.so child terminates\n",
                )

        inst = SimpleNamespace(instantiation_id="q1_i1", args_string="1 args")
        conversation = TpchMonetdbOptimizationConversation.__new__(TpchMonetdbOptimizationConversation)
        conversation.benchmark_sf = 100
        conversation.run_tool = FakeRunTool()
        conversation.manifest = SimpleNamespace(
            get_instantiations_for_query=lambda query_id, scale_factor: [inst]
        )

        summary = conversation._sample_trace_for_query("1")

        assert summary.query_id == "1"
        assert summary.evidence_sufficient is True
        assert "Top profiles:" in summary.summary_text
        return None


def test_linux_perf_native_allows_current_executor() -> None:
    preflight = build_hardware_counter_preflight(
        backend="linux_perf_native",
        target_cpu="icelake",
        runner_cmd=None,
        host_kernel="5.14",
        perf_event_paranoid="2",
        large_sf=1000,
    )

    assert preflight.backend == "linux_perf_native"
    assert preflight.target_cpu == "icelake"
    return None


def test_unsupported_hardware_counter_backend_fails_preflight() -> None:
    with pytest.raises(Exception, match="HARDWARE_COUNTER_BACKEND_MISSING"):
        build_hardware_counter_preflight(
            backend="perf_auto",
            target_cpu="icelake",
            runner_cmd=None,
            host_kernel="5.14",
            perf_event_paranoid="1",
            large_sf=1000,
        )
    return None


def test_repeated_scope_measurements_record_large_sf_samples() -> None:
    conversation = TpchMonetdbOptimizationConversation.__new__(TpchMonetdbOptimizationConversation)
    samples = [
        ({"3": 0.11, "4": 0.12}, {"3": 0.05, "4": 0.05}, 0.114, False),
        ({"3": 0.10, "4": 0.11}, {"3": 0.05, "4": 0.05}, 0.105, False),
        ({"3": 0.09, "4": 0.10}, {"3": 0.05, "4": 0.05}, 0.095, False),
    ]

    def fake_measure_scope_runtime(query_ids, scale_factor=None):
        return samples.pop(0)

    conversation._measure_scope_runtime = fake_measure_scope_runtime
    result = conversation._collect_repeated_scope_measurements(
        ["3", "4"],
        scale_factor=1000,
        repetitions=3,
    )

    assert result["scale_factor"] == 1000
    assert result["repetitions"] == 3
    assert len(result["aggregate_runtime_ms_samples"]) == 3
    assert result["lazy_build_detected"] is False
