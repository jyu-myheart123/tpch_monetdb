from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from agents.run_context import RunContextWrapper

from tpch_monetdb.tools.cpu_info import CpuInfoTool, make_cpu_info_tool
from tpch_monetdb.tools.stage_tool_policy import ToolProfile
from tpch_monetdb.tools.tpch_monetdb_agent_tools import (
    _TOOL_GREP_MAX_BYTES,
    StageToolRuntime,
)


def _runtime(root: Path, read_globs: tuple[str, ...] = ("*", "**", "**/*")) -> StageToolRuntime:
    """Build a minimal StageToolRuntime with a deterministic readable profile."""
    runtime = StageToolRuntime(root)
    runtime.profiles = {
        "assignment": ToolProfile(
            name="assignment",
            tool_names=("list_files", "grep_repo", "read_file"),
            read_globs=read_globs,
        )
    }
    runtime._active_profile_name = "assignment"
    return runtime


def test_list_files_lists_root_and_marks_directories(tmp_path: Path) -> None:
    """list_files should list readable root entries in stable order."""
    (tmp_path / "src").mkdir()
    (tmp_path / "query_q1.cpp").write_text("int main() {}\n", encoding="utf-8")
    runtime = _runtime(tmp_path)

    result = runtime.list_directory("/", None, 20)

    assert "query_q1.cpp" in result.splitlines()
    assert "src/" in result.splitlines()
    return None


def test_list_files_supports_glob_and_limit(tmp_path: Path) -> None:
    """list_files should apply glob filters and stop at the requested limit."""
    (tmp_path / "a.cpp").write_text("a\n", encoding="utf-8")
    (tmp_path / "b.hpp").write_text("b\n", encoding="utf-8")
    (tmp_path / "c.cpp").write_text("c\n", encoding="utf-8")
    runtime = _runtime(tmp_path)

    result = runtime.list_directory("/", "*.cpp", 1)

    lines = result.splitlines()
    assert len(lines) == 1
    assert lines[0].endswith(".cpp")
    return None


def test_list_files_rejects_workspace_escape(tmp_path: Path) -> None:
    """list_files should reject paths outside the workspace."""
    runtime = _runtime(tmp_path)

    with pytest.raises(Exception, match="PATH_OUTSIDE_WORKSPACE"):
        runtime.list_directory("..", None, 20)
    return None


def test_grep_repo_finds_matches_with_line_numbers(tmp_path: Path) -> None:
    """grep_repo should return relative path, line number, and line text."""
    (tmp_path / "query_q1.cpp").write_text("alpha\nneedle here\n", encoding="utf-8")
    runtime = _runtime(tmp_path)

    result = runtime.grep_repo("needle", "query_q1.cpp", None, 10)

    assert result == "query_q1.cpp:2:needle here"
    return None


def test_grep_repo_supports_glob_and_limit(tmp_path: Path) -> None:
    """grep_repo should filter by glob and respect the limit."""
    (tmp_path / "a.cpp").write_text("needle one\nneedle two\n", encoding="utf-8")
    (tmp_path / "b.hpp").write_text("needle hidden\n", encoding="utf-8")
    runtime = _runtime(tmp_path)

    result = runtime.grep_repo("needle", "/", "*.cpp", 1)

    assert result.splitlines() == ["a.cpp:1:needle one"]
    return None


def test_grep_repo_returns_no_matches_for_empty_result(tmp_path: Path) -> None:
    """grep_repo should return a stable empty-result marker."""
    (tmp_path / "query_q1.cpp").write_text("alpha\n", encoding="utf-8")
    runtime = _runtime(tmp_path)

    assert runtime.grep_repo("needle", "/", None, 10) == "(no matches)"
    return None


def test_grep_repo_skips_large_and_non_utf8_files(tmp_path: Path) -> None:
    """grep_repo should skip oversized and non-UTF-8 files without crashing."""
    (tmp_path / "large.log").write_text("x" * (_TOOL_GREP_MAX_BYTES + 1), encoding="utf-8")
    (tmp_path / "binary.bin").write_bytes(b"\xff\xfe\x00needle")
    runtime = _runtime(tmp_path)

    result = runtime.grep_repo("needle", "/", None, 10)

    assert "(no matches)" in result
    assert "grep_repo skipped 1 large file" in result
    return None


def test_cpuinfo_flags_parse_x86_and_arm(tmp_path: Path) -> None:
    """cpu_info should parse ISA flags from x86 flags and ARM Features lines."""
    tool = CpuInfoTool(tmp_path, tmp_path / "cache")

    assert tool._parse_cpuinfo_flags("flags\t: fpu sse4_2 avx avx2\n") == [
        "fpu",
        "sse4_2",
        "avx",
        "avx2",
    ]
    assert tool._parse_cpuinfo_flags("Features\t: fp asimd neon\n") == [
        "fp",
        "asimd",
        "neon",
    ]
    return None


def test_lscpu_summary_extracts_cache_and_numa(tmp_path: Path) -> None:
    """cpu_info should extract stable lscpu keys used by the optimizer."""
    tool = CpuInfoTool(tmp_path, tmp_path / "cache")
    summary = tool._parse_lscpu_summary(
        "\n".join(
            (
                "Architecture: x86_64",
                "Model name: Example CPU",
                "L1d cache: 512 KiB",
                "L2 cache: 8 MiB",
                "L3 cache: 32 MiB",
                "NUMA node(s): 2",
            )
        )
    )

    assert summary["Architecture"] == "x86_64"
    assert summary["Model name"] == "Example CPU"
    assert summary["L3 cache"] == "32 MiB"
    assert summary["NUMA node(s)"] == "2"
    return None


def test_cpu_info_build_response_uses_hardware_evidence(tmp_path: Path) -> None:
    """cpu_info should build a stable JSON payload with vectorization and cache evidence."""
    tool = CpuInfoTool(tmp_path, tmp_path / "cache")
    probes = {
        "uname": {"command": "uname -m", "stdout": "x86_64\n", "stderr": "", "exit_code": 0},
        "lscpu": {
            "command": "lscpu",
            "stdout": "Architecture: x86_64\nModel name: CPU-A\nL3 cache: 32 MiB\n",
            "stderr": "",
            "exit_code": 0,
        },
        "cpuinfo": {
            "command": "cat /proc/cpuinfo",
            "stdout": "flags\t: sse4_2 avx avx2\n",
            "stderr": "",
            "exit_code": 0,
        },
    }

    payload = tool._build_response(probes)

    assert payload["arch"] == "x86_64"
    assert payload["model_name"] == "CPU-A"
    assert payload["target_cpu_hint"] == "native"
    assert payload["vectorization_flags"] == ["avx2", "avx", "sse4_2"]
    assert payload["cache_summary"]["L3"] == "32 MiB"
    return None


def test_cpu_info_omits_native_hint_without_hardware_probe(tmp_path: Path) -> None:
    """cpu_info should not recommend native target CPU without lscpu or cpuinfo evidence."""
    tool = CpuInfoTool(tmp_path, tmp_path / "cache")
    probes = {
        "uname": {"command": "uname -m", "stdout": "x86_64\n", "stderr": "", "exit_code": 0},
        "lscpu": {"command": "lscpu", "stdout": "", "stderr": "missing", "exit_code": 127},
        "cpuinfo": {"command": "cat /proc/cpuinfo", "stdout": "", "stderr": "missing", "exit_code": 127},
    }

    payload = tool._build_response(probes)

    assert payload["target_cpu_hint"] is None
    assert payload["vectorization_recommendation"] == "vectorization_support_unclear"
    return None


def test_cpu_info_truncates_long_raw_output(tmp_path: Path) -> None:
    """cpu_info should include a visible truncation marker for oversized raw output."""
    tool = CpuInfoTool(tmp_path, tmp_path / "cache", max_output_tokens=40)

    truncated = tool._truncate("a" * 300)

    assert "truncated" in truncated
    assert len(truncated) < 300
    return None


@pytest.mark.asyncio
async def test_cpu_info_tool_reports_invalid_json(tmp_path: Path) -> None:
    """The cpu_info FunctionTool wrapper should report invalid JSON arguments."""
    tool = make_cpu_info_tool(tmp_path, tmp_path / "cache")

    result = await tool.on_invoke_tool(RunContextWrapper(context=None), "{bad json")

    assert "Invalid JSON" in result
    return None
