from __future__ import annotations

import re
import shlex
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from tpch_monetdb.utils.pipeline_contracts import raise_pipeline_contract_error

ALLOWED_HARDWARE_COUNTER_BACKENDS: tuple[str, ...] = ("linux_perf_native",)
REQUIRED_HARDWARE_COUNTER_EVENTS: tuple[str, ...] = (
    "cycles",
    "instructions",
    "cache-misses",
    "LLC-load-misses",
    "dTLB-load-misses",
)
DEFAULT_PERF_HOTSPOT_EVENT = "cycles"
DEFAULT_PERF_HOTSPOT_FREQUENCY = 99
DEFAULT_PERF_HOTSPOT_REPETITIONS = 5
_SOURCE_LINE_RE = re.compile(
    r"(?P<path>(?:[A-Za-z0-9_./+-]+/)?[A-Za-z0-9_.+-]+"
    r"\.(?:c|cc|cpp|cxx|h|hh|hpp|hxx)):(?P<line>[0-9]+)"
)


@dataclass(frozen=True)
class HardwareCounterSummary:
    """Parsed hardware-counter values plus derived metrics."""

    backend: str
    counters: dict[str, float] = field(default_factory=dict)
    derived_metrics: dict[str, float] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PerfHotspotSummary:
    """Parsed perf script call-stack evidence for optimization guidance."""

    backend: str
    top_symbols: tuple[tuple[str, int], ...] = field(default_factory=tuple)
    top_frames: tuple[tuple[str, int], ...] = field(default_factory=tuple)
    top_source_lines: tuple[tuple[str, int], ...] = field(default_factory=tuple)
    sample_count: int = 0
    raw_script_excerpt: tuple[str, ...] = field(default_factory=tuple)
    provenance: dict[str, Any] = field(default_factory=dict)
    perf_data_path: str | None = None
    perf_script_path: str | None = None


@dataclass(frozen=True)
class HardwareCounterPreflight:
    """Resolved hardware-counter execution contract for one run."""

    backend: str
    target_cpu: str
    runner_cmd: str | None
    host_kernel: str | None = None
    perf_event_paranoid: str | None = None
    large_sf: int | None = None
    required_events: tuple[str, ...] = REQUIRED_HARDWARE_COUNTER_EVENTS


def _perf_prefix(preflight: HardwareCounterPreflight) -> list[str]:
    if preflight.runner_cmd in (None, ""):
        return ["perf"]
    return shlex.split(preflight.runner_cmd)


def build_hardware_counter_invocation(
    *,
    preflight: HardwareCounterPreflight,
    executable_cmd: list[str],
) -> list[str]:
    """Build the backend-specific command used to collect hardware counters."""
    event_spec = ",".join(preflight.required_events)
    if preflight.backend == "linux_perf_native":
        return [
            *_perf_prefix(preflight),
            "stat",
            "-x,",
            "-e",
            event_spec,
            "--",
            *executable_cmd,
        ]
    raise_pipeline_contract_error(
        code="HARDWARE_COUNTER_BACKEND_MISSING",
        message=f"Unsupported hardware-counter backend: {preflight.backend}",
        stage="hardware_counter_backend",
    )
    return []


def build_perf_record_invocation(
    *,
    preflight: HardwareCounterPreflight,
    output_path: str,
    executable_cmd: list[str] | None = None,
    attach_pid: int | None = None,
    attach_pids: list[int] | tuple[int, ...] | None = None,
    event_name: str = DEFAULT_PERF_HOTSPOT_EVENT,
    frequency: int = DEFAULT_PERF_HOTSPOT_FREQUENCY,
) -> list[str]:
    """Build the perf record command used to collect call-stack hotspots."""
    if preflight.backend != "linux_perf_native":
        raise_pipeline_contract_error(
            code="PERF_HOTSPOT_BACKEND_UNSUPPORTED",
            message=(
                "perf hotspot capture currently requires "
                "linux_perf_native backend"
            ),
            stage="perf_hotspot_backend",
        )
    resolved_attach_pids: tuple[int, ...] | None = None
    if attach_pid is not None:
        resolved_attach_pids = (attach_pid,)
    if attach_pids is not None:
        if resolved_attach_pids is not None:
            raise_pipeline_contract_error(
                code="PERF_HOTSPOT_TARGET_AMBIGUOUS",
                message="perf record accepts attach_pid or attach_pids, not both",
                stage="perf_hotspot_backend",
            )
        resolved_attach_pids = tuple(attach_pids)
    if (executable_cmd is None) == (resolved_attach_pids is None):
        raise_pipeline_contract_error(
            code="PERF_HOTSPOT_TARGET_AMBIGUOUS",
            message="perf record requires exactly one of executable_cmd or attach pids",
            stage="perf_hotspot_backend",
        )
    base_command = [
        *_perf_prefix(preflight),
        "record",
        "-F",
        str(frequency),
        "-g",
        "--call-graph",
        "fp",
        "-e",
        event_name,
        "--output",
        output_path,
    ]
    if resolved_attach_pids is not None:
        pid_values = [str(pid) for pid in resolved_attach_pids if pid > 0]
        if not pid_values:
            raise_pipeline_contract_error(
                code="PERF_HOTSPOT_TARGET_AMBIGUOUS",
                message="perf record requires at least one positive attach pid",
                stage="perf_hotspot_backend",
            )
        return [*base_command, "-p", ",".join(pid_values)]
    return [
        *base_command,
        "--",
        *(executable_cmd or []),
    ]


def build_perf_script_invocation(
    *,
    preflight: HardwareCounterPreflight,
    input_path: str,
    include_source_lines: bool = True,
) -> list[str]:
    """Build the perf script command used to decode recorded call stacks."""
    if preflight.backend != "linux_perf_native":
        raise_pipeline_contract_error(
            code="PERF_HOTSPOT_BACKEND_UNSUPPORTED",
            message=(
                "perf hotspot decoding currently requires "
                "linux_perf_native backend"
            ),
            stage="perf_hotspot_backend",
        )
    command = [
        *_perf_prefix(preflight),
        "script",
        "-i",
        input_path,
    ]
    if include_source_lines:
        command.extend(
            [
                "-F",
                "comm,pid,tid,cpu,time,event,ip,sym,dso,srcline",
            ]
        )
    return command


def require_supported_hardware_counter_backend(backend: str) -> None:
    """Reject unsupported hardware-counter backends."""
    if backend not in ALLOWED_HARDWARE_COUNTER_BACKENDS:
        raise_pipeline_contract_error(
            code="HARDWARE_COUNTER_BACKEND_MISSING",
            message=(
                f"Unsupported hardware-counter backend: {backend}. "
                f"Expected one of {', '.join(ALLOWED_HARDWARE_COUNTER_BACKENDS)}"
            ),
            stage="hardware_counter_backend",
        )
    return None


def build_hardware_counter_preflight(
    *,
    backend: str,
    target_cpu: str | None,
    runner_cmd: str | None,
    host_kernel: str | None,
    perf_event_paranoid: str | None,
    large_sf: int | None,
) -> HardwareCounterPreflight:
    """Resolve and validate the explicit hardware-counter execution contract."""
    require_supported_hardware_counter_backend(backend)
    if target_cpu in (None, ""):
        raise_pipeline_contract_error(
            code="HARDWARE_COUNTER_PREFLIGHT_FAILED",
            message="target_cpu is required when hardware_counter_backend is enabled",
            stage="hardware_counter_preflight",
        )
    return HardwareCounterPreflight(
        backend=backend,
        target_cpu=str(target_cpu),
        runner_cmd=runner_cmd,
        host_kernel=host_kernel,
        perf_event_paranoid=perf_event_paranoid,
        large_sf=large_sf,
    )


def parse_perf_stat_csv(
    text: str,
    *,
    backend: str,
    provenance: dict[str, Any] | None = None,
) -> HardwareCounterSummary:
    """Parse perf stat CSV output and compute derived metrics."""
    raise NotImplementedError("TODO(optional): parse perf stat CSV counters and derived metrics")


def parse_perf_script_hotspots(
    text: str,
    *,
    backend: str,
    provenance: dict[str, Any] | None = None,
    max_symbols: int = 12,
    max_frames: int = 12,
    max_excerpt_lines: int = 80,
    perf_data_path: str | None = None,
    perf_script_path: str | None = None,
) -> PerfHotspotSummary:
    """Parse perf script call stacks into compact symbol and frame hot spots."""
    raise NotImplementedError("TODO(optional): parse perf script hotspots")


def _split_perf_script_samples(text: str) -> tuple[tuple[str, ...], ...]:
    """Group perf script output into samples while preserving call-stack frames."""
    raise NotImplementedError("TODO(optional): split perf script samples")


def _first_perf_symbol(frames: tuple[str, ...]) -> str | None:
    for frame in frames:
        symbol = extract_perf_script_symbol(frame)
        if symbol is not None:
            return symbol
    return None


def _first_perf_source_line(frames: tuple[str, ...]) -> str | None:
    for frame in frames:
        source_line = extract_perf_script_source_line(frame)
        if source_line is not None:
            return source_line
    return None


def extract_perf_script_symbol(line: str) -> str | None:
    """Extract one symbol name from a perf script stack frame."""
    raise NotImplementedError("TODO(optional): extract symbol names from perf script frames")


def extract_perf_script_source_line(line: str) -> str | None:
    """Extract one source file and line from a perf script frame when present."""
    raise NotImplementedError("TODO(optional): extract source file and line from perf script frames")


def _looks_like_perf_address(value: str) -> bool:
    normalized = value.removeprefix("0x")
    if not normalized:
        return False
    if normalized != "0" and len(normalized) < 4:
        return False
    return all(char in "0123456789abcdefABCDEF" for char in normalized)


def _normalize_perf_symbol(value: str) -> str | None:
    """Normalize one perf token into a symbol name or reject non-symbol tokens."""
    symbol = value.strip()
    if not symbol or symbol.startswith("("):
        return None
    if symbol.endswith(":"):
        return None
    if symbol.startswith("[") and symbol.endswith("]"):
        return None
    if not any(char.isalpha() or char == "_" for char in symbol):
        return None
    if _looks_like_perf_address(symbol):
        return None
    if _SOURCE_LINE_RE.search(symbol):
        return None
    if symbol.startswith("/"):
        return None
    symbol = symbol.split("+0x", 1)[0]
    symbol = symbol.split("@plt", 1)[0]
    if symbol in {"0", "[unknown]", "unknown", "__unknown__"}:
        return None
    return symbol


def derive_hardware_counter_metrics(counters: dict[str, float]) -> dict[str, float]:
    """Derive stable metrics from parsed hardware-counter values."""
    raise NotImplementedError("TODO(optional): derive IPC and miss-rate metrics from counters")


def validate_hardware_counter_summary(
    summary: HardwareCounterSummary,
    *,
    required_events: tuple[str, ...],
) -> None:
    """Fail closed when any required hardware-counter event is missing."""
    raise NotImplementedError("TODO(optional): validate required hardware-counter events")
