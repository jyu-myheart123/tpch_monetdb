import pytest

from tpch_monetdb.tools.tpch.hardware_counters import (
    DEFAULT_PERF_HOTSPOT_FREQUENCY,
    HardwareCounterSummary,
    build_hardware_counter_invocation,
    build_hardware_counter_preflight,
    build_perf_record_invocation,
    build_perf_script_invocation,
    derive_hardware_counter_metrics,
    extract_perf_script_source_line,
    extract_perf_script_symbol,
    parse_perf_script_hotspots,
    parse_perf_stat_csv,
    validate_hardware_counter_summary,
)
from tpch_monetdb.utils.pipeline_contracts import PipelineContractError


def test_optional_parse_perf_stat_csv_extracts_counters_and_provenance() -> None:
    """perf stat CSV parser should ignore unsupported rows and preserve provenance."""
    summary = parse_perf_stat_csv(
        "\n".join(
            (
                "1000,,cycles",
                "2500,,instructions",
                "125,,cache-misses",
                "20,,LLC-load-misses",
                "10,,dTLB-load-misses",
                "25,,branch-misses",
                "<not counted>,,page-faults",
                "bad-value,,context-switches",
                "too-short",
            )
        ),
        backend="linux_perf_native",
        provenance={"host_kernel": "6.8.0", "query_id": "3"},
    )

    assert summary.backend == "linux_perf_native"
    assert summary.counters == {
        "cycles": 1000.0,
        "instructions": 2500.0,
        "cache-misses": 125.0,
        "LLC-load-misses": 20.0,
        "dTLB-load-misses": 10.0,
        "branch-misses": 25.0,
    }
    assert summary.derived_metrics["ipc"] == 2.5
    assert summary.derived_metrics["cache_miss_rate"] == 0.05
    assert summary.derived_metrics["llc_mpki"] == 8.0
    assert summary.derived_metrics["dtlb_mpki"] == 4.0
    assert summary.derived_metrics["branch_miss_rate"] == 0.01
    assert summary.provenance["query_id"] == "3"
    return None


def test_optional_derive_metrics_handles_missing_denominators() -> None:
    """Derived metrics should omit ratios whose denominator is unavailable."""
    assert derive_hardware_counter_metrics({"cycles": 0.0}) == {}
    assert derive_hardware_counter_metrics({"instructions": 0.0}) == {}

    metrics = derive_hardware_counter_metrics(
        {
            "cycles": 400.0,
            "instructions": 1000.0,
            "cache-misses": 5.0,
        }
    )

    assert metrics["ipc"] == 2.5
    assert metrics["cache_miss_rate"] == 0.005
    assert metrics["llc_mpki"] == 0.0
    assert metrics["dtlb_mpki"] == 0.0
    assert metrics["branch_miss_rate"] == 0.0
    return None


def test_optional_validate_counter_summary_fails_closed() -> None:
    """Required perf events should be checked with the project contract error."""
    summary = HardwareCounterSummary(
        backend="linux_perf_native",
        counters={"cycles": 1.0, "instructions": 2.0},
    )

    with pytest.raises(PipelineContractError, match="HARDWARE_COUNTER_EVENTS_MISSING"):
        validate_hardware_counter_summary(
            summary,
            required_events=("cycles", "instructions", "cache-misses"),
        )

    validate_hardware_counter_summary(
        summary,
        required_events=("cycles", "instructions"),
    )
    return None


def test_optional_perf_invocation_builders_remain_available() -> None:
    """Optional parser work must not break perf command construction helpers."""
    preflight = build_hardware_counter_preflight(
        backend="linux_perf_native",
        target_cpu="native",
        runner_cmd="/usr/local/bin/tpch-perf",
        host_kernel="6.8.0",
        perf_event_paranoid="1",
        large_sf=1,
    )

    stat_command = build_hardware_counter_invocation(
        preflight=preflight,
        executable_cmd=["./db", "input.args"],
    )
    record_command = build_perf_record_invocation(
        preflight=preflight,
        executable_cmd=["./db", "input.args"],
        output_path="/tmp/q3.perf.data",
    )
    script_command = build_perf_script_invocation(
        preflight=preflight,
        input_path="/tmp/q3.perf.data",
    )

    assert stat_command[:2] == ["/usr/local/bin/tpch-perf", "stat"]
    assert stat_command[-2:] == ["./db", "input.args"]
    assert record_command[:4] == [
        "/usr/local/bin/tpch-perf",
        "record",
        "-F",
        str(DEFAULT_PERF_HOTSPOT_FREQUENCY),
    ]
    assert record_command[-3:] == ["--", "./db", "input.args"]
    assert script_command[:4] == [
        "/usr/local/bin/tpch-perf",
        "script",
        "-i",
        "/tmp/q3.perf.data",
    ]
    return None


def test_optional_extract_perf_script_symbols_and_source_lines() -> None:
    """perf script parser should normalize symbols and source locations."""
    assert (
        extract_perf_script_symbol(
            "        7f000000 scan_query_3+0x21 (/workspace/db)"
        )
        == "scan_query_3"
    )
    assert (
        extract_perf_script_symbol(
            "db 123 123 000 1.000 cycles: 7f000010 aggregate_bucket /workspace/db"
        )
        == "aggregate_bucket"
    )
    assert extract_perf_script_symbol("        0 [unknown] ([unknown])") is None
    assert extract_perf_script_symbol("        7f000020 /workspace/db") is None
    assert (
        extract_perf_script_source_line(
            "db 123 cycles: 7f000000 scan_query_3 /workspace/db query_q3.cpp:42"
        )
        == "query_q3.cpp:42"
    )
    assert extract_perf_script_source_line("no source location here") is None
    return None


def test_optional_parse_perf_script_hotspots_groups_call_stacks() -> None:
    """perf script samples should produce leaf symbols, frame counts, and excerpts."""
    summary = parse_perf_script_hotspots(
        "\n".join(
            (
                "db 123 [000] 1.000: 1 cycles:",
                "        7f000000 scan_query_3+0x21 (/workspace/db) query_q3.cpp:42",
                "        7f000010 aggregate_bucket+0x9 (/workspace/db) query_q3.cpp:77",
                "",
                "db 123 [000] 1.001: 1 cycles:",
                "        7f000000 scan_query_3+0x21 (/workspace/db) query_q3.cpp:42",
                "        0 [unknown] ([unknown])",
            )
        ),
        backend="linux_perf_native",
        provenance={"query_ids": ["3"]},
        perf_data_path="/tmp/q3.perf.data",
        perf_script_path="/tmp/q3.perf.script.txt",
    )

    assert summary.backend == "linux_perf_native"
    assert summary.top_symbols[0] == ("scan_query_3", 2)
    assert ("query_q3.cpp:42", 2) in summary.top_source_lines
    assert any("aggregate_bucket" in frame for frame, _count in summary.top_frames)
    assert "[unknown]" not in {symbol for symbol, _count in summary.top_symbols}
    assert summary.sample_count == 2
    assert summary.raw_script_excerpt[0].startswith("db 123")
    assert summary.provenance["query_ids"] == ["3"]
    assert summary.perf_data_path == "/tmp/q3.perf.data"
    assert summary.perf_script_path == "/tmp/q3.perf.script.txt"
    return None


def test_optional_parse_perf_script_hotspots_handles_flat_script_lines() -> None:
    """Single-line perf script records should still be counted as samples."""
    summary = parse_perf_script_hotspots(
        "\n".join(
            (
                "db 123 123 000 1.000 cycles: 7f000000 probe_orders /workspace/db query_q5.cpp:12",
                "db 123 123 000 1.001 cycles: 7f000010 probe_orders /workspace/db query_q5.cpp:12",
                "db 123 123 000 1.002 cycles: 7f000020 aggregate_revenue /workspace/db query_q5.cpp:91",
            )
        ),
        backend="linux_perf_native",
        max_symbols=1,
    )

    assert summary.top_symbols == (("probe_orders", 2),)
    assert summary.top_source_lines[0] == ("query_q5.cpp:12", 2)
    assert summary.sample_count == 3
    return None
