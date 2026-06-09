import pytest
from pathlib import Path

from tpch_monetdb.tools.tpch.hardware_counters import (
    ALLOWED_HARDWARE_COUNTER_BACKENDS,
    build_hardware_counter_preflight,
    build_hardware_counter_invocation,
    build_perf_record_invocation,
    build_perf_script_invocation,
    require_supported_hardware_counter_backend,
)
from tpch_monetdb.misc.tpch.compiler import (
    build_vectorization_flag_bundle,
    parse_vectorization_reports,
)
from tpch_monetdb.utils.pipeline_contracts import PipelineContractError

ROOT = Path(__file__).resolve().parents[1]


def test_supported_hardware_counter_backends_are_explicit() -> None:
    assert ALLOWED_HARDWARE_COUNTER_BACKENDS == ("linux_perf_native",)
    return None


def test_require_supported_hardware_counter_backend_rejects_unknown_backend() -> None:
    with pytest.raises(PipelineContractError, match="HARDWARE_COUNTER_BACKEND_MISSING"):
        require_supported_hardware_counter_backend("perf_auto")
    return None


def test_hardware_counter_preflight_requires_target_cpu_and_runner_contract() -> None:
    with pytest.raises(PipelineContractError, match="HARDWARE_COUNTER_PREFLIGHT_FAILED"):
        build_hardware_counter_preflight(
            backend="linux_perf_native",
            target_cpu=None,
            runner_cmd="perf-wrapper",
            host_kernel="5.14",
            perf_event_paranoid="1",
            large_sf=1000,
        )
    return None


def test_hardware_counter_preflight_rejects_unsupported_backend() -> None:
    with pytest.raises(PipelineContractError, match="HARDWARE_COUNTER_PREFLIGHT_FAILED"):
        build_hardware_counter_preflight(
            backend="linux_perf_native",
            target_cpu="",
            runner_cmd=None,
            host_kernel="5.14",
            perf_event_paranoid="1",
            large_sf=1000,
        )
    with pytest.raises(PipelineContractError, match="HARDWARE_COUNTER_BACKEND_MISSING"):
        build_hardware_counter_preflight(
            backend="perf_auto",
            target_cpu="skylake",
            runner_cmd=None,
            host_kernel="5.14",
            perf_event_paranoid="1",
            large_sf=1000,
        )
    return None


def test_hardware_counter_preflight_records_native_runtime_provenance() -> None:
    preflight = build_hardware_counter_preflight(
        backend="linux_perf_native",
        target_cpu="icelake",
        runner_cmd="/usr/bin/perf",
        host_kernel="5.14.0",
        perf_event_paranoid="1",
        large_sf=1000,
    )
    assert preflight.target_cpu == "icelake"
    assert preflight.runner_cmd == "/usr/bin/perf"
    assert preflight.host_kernel == "5.14.0"
    assert preflight.perf_event_paranoid == "1"
    assert preflight.large_sf == 1000
    return None


def test_linux_perf_native_allows_default_local_perf() -> None:
    preflight = build_hardware_counter_preflight(
        backend="linux_perf_native",
        target_cpu="native",
        runner_cmd=None,
        host_kernel="3.10.0",
        perf_event_paranoid="2",
        large_sf=1000,
    )
    assert preflight.runner_cmd is None
    return None


def test_build_hardware_counter_invocation_uses_native_perf_backend() -> None:
    preflight = build_hardware_counter_preflight(
        backend="linux_perf_native",
        target_cpu="icelake",
        runner_cmd=None,
        host_kernel="5.14.0",
        perf_event_paranoid="1",
        large_sf=1000,
    )
    command = build_hardware_counter_invocation(
        preflight=preflight,
        executable_cmd=["./db", "data.ilp"],
    )
    assert command[:5] == [
        "perf",
        "stat",
        "-x,",
        "-e",
        "cycles,instructions,cache-misses,LLC-load-misses,dTLB-load-misses",
    ]
    assert command[-2:] == ["./db", "data.ilp"]
    return None


def test_linux_perf_native_allows_explicit_perf_binary() -> None:
    preflight = build_hardware_counter_preflight(
        backend="linux_perf_native",
        target_cpu="native",
        runner_cmd="/opt/perf",
        host_kernel="3.10.0",
        perf_event_paranoid="2",
        large_sf=1000,
    )
    command = build_hardware_counter_invocation(
        preflight=preflight,
        executable_cmd=["./db", "data.ilp"],
    )
    assert command[:2] == ["/opt/perf", "stat"]
    return None


def test_build_perf_hotspot_invocations_use_explicit_perf_binary() -> None:
    preflight = build_hardware_counter_preflight(
        backend="linux_perf_native",
        target_cpu="native",
        runner_cmd="/usr/local/bin/tpch-perf",
        host_kernel="3.10.0",
        perf_event_paranoid="2",
        large_sf=1000,
    )
    record_command = build_perf_record_invocation(
        preflight=preflight,
        executable_cmd=["./db", "data.ilp"],
        output_path="/tmp/perf.data",
    )
    script_command = build_perf_script_invocation(
        preflight=preflight,
        input_path="/tmp/perf.data",
    )
    assert record_command[:11] == [
        "/usr/local/bin/tpch-perf",
        "record",
        "-F",
        "99",
        "-g",
        "--call-graph",
        "fp",
        "-e",
        "cycles",
        "--output",
        "/tmp/perf.data",
    ]
    assert record_command[-3:] == ["--", "./db", "data.ilp"]
    attach_command = build_perf_record_invocation(
        preflight=preflight,
        attach_pid=4242,
        output_path="/tmp/perf.data",
    )
    assert attach_command[-2:] == ["-p", "4242"]
    process_tree_command = build_perf_record_invocation(
        preflight=preflight,
        attach_pids=[4242, 4343],
        output_path="/tmp/perf.data",
    )
    assert process_tree_command[-2:] == ["-p", "4242,4343"]
    assert script_command[:4] == [
        "/usr/local/bin/tpch-perf",
        "script",
        "-i",
        "/tmp/perf.data",
    ]
    assert "comm,pid,tid,cpu,time,event,ip,sym,dso,srcline" in script_command
    return None


def test_validation_command_contract_mentions_final_measurement_commands() -> None:
    contract_path = (
        ROOT.parent / "specs" / "001-pipeline-bug-fix" / "contracts" / "validation-command-contract.md"
    )
    if not contract_path.exists():
        pytest.skip("validation command contract is not present in this checkout")
    contract_text = contract_path.read_text(encoding="utf-8")
    assert "run_outer_loop_tpch_monetdb.py" in contract_text
    assert "--target_cpu" in contract_text
    assert "--hardware_counter_backend" in contract_text
    assert "performance_comparison.md" in contract_text
    return None


def test_compiler_vectorization_helpers_emit_flags_and_parse_reports(tmp_path: Path) -> None:
    bundle = build_vectorization_flag_bundle(build_dir=tmp_path, target_cpu="icelake")
    assert "-march=icelake" in bundle["flags"]
    optimized_path = bundle["optimized_report_path"]
    missed_path = bundle["missed_report_path"]
    optimized_path.write_text("query_q1.cpp:42:7: loop vectorized\n", encoding="utf-8")
    missed_path.write_text("query_q1.cpp:99:3: missed: data dependence\n", encoding="utf-8")

    summary = parse_vectorization_reports(
        optimized_report_path=optimized_path,
        missed_report_path=missed_path,
        target_cpu="icelake",
    )
    assert summary["target_cpu"] == "icelake"
    assert summary["optimized_loops"] == 1
    assert summary["missed_loops"] == 1
    assert summary["report_available"] is True
    assert summary["vectorization_applied"] is True
    assert summary["optimized_loop_sites"] == [
        {
            "file": "query_q1.cpp",
            "line": 42,
            "column": 7,
            "message": "loop vectorized",
            "raw": "query_q1.cpp:42:7: loop vectorized",
            "source_category": "workspace",
        }
    ]
    assert summary["workspace_optimized_loop_sites"] == summary["optimized_loop_sites"]
    return None
