from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RuntimeHealthThresholds:
    reload_max_files: int = 128
    reload_max_bytes: int = 256 * 1024 * 1024


@dataclass(frozen=True)
class RuntimeHealthReport:
    workspace_path: str
    reload_path: str
    reload_files: int
    reload_bytes: int
    healthy: bool
    reason_code: str | None = None
    detail: str | None = None


INFRA_FAILURE_PATTERNS: tuple[tuple[str, str], ...] = (
    ("signal: 11", "RUNNER_SEGFAULT"),
    ("Segmentation fault", "RUNNER_SEGFAULT"),
    ("fork: Cannot allocate memory", "FORK_ENOMEM"),
    ("[ERROR:INFRA_BLOCKED] fork failed", "FORK_ENOMEM"),
    ("[ERROR:RUNNER_BROKEN_PIPE]", "RUNNER_BROKEN_PIPE"),
    ("BrokenPipeError", "RUNNER_BROKEN_PIPE"),
    ("Terminated after", "RUNNER_TIMEOUT"),
    ("Expected output file missing", "RESULT_CSV_MISSING"),
    ("No timing output found", "TIMING_MISSING"),
)

INFRA_FAILURE_CODE_PRIORITY: tuple[str, ...] = (
    "RUNNER_SEGFAULT",
    "RUNNER_BROKEN_PIPE",
    "RUNNER_TIMEOUT",
    "FORK_ENOMEM",
    "RESULT_CSV_MISSING",
    "TIMING_MISSING",
)


def inspect_runtime_health(
    workspace_path: Path,
    thresholds: RuntimeHealthThresholds | None = None,
) -> RuntimeHealthReport:
    resolved_thresholds = thresholds or RuntimeHealthThresholds()
    reload_path = workspace_path / "build" / ".reload"
    reload_files = _count_files(reload_path)
    reload_bytes = _dir_size_bytes(reload_path)
    reason_code: str | None = None
    detail: str | None = None
    if reload_files > resolved_thresholds.reload_max_files:
        reason_code = "RELOAD_FILE_LIMIT"
        detail = f"reload_files={reload_files}"
    elif reload_bytes > resolved_thresholds.reload_max_bytes:
        reason_code = "RELOAD_BYTE_LIMIT"
        detail = f"reload_bytes={reload_bytes}"
    healthy = reason_code is None
    return RuntimeHealthReport(
        workspace_path=str(workspace_path),
        reload_path=str(reload_path),
        reload_files=reload_files,
        reload_bytes=reload_bytes,
        healthy=healthy,
        reason_code=reason_code,
        detail=detail,
    )


def cleanup_reload_dir(workspace_path: Path) -> RuntimeHealthReport:
    reload_path = workspace_path / "build" / ".reload"
    if reload_path.exists():
        shutil.rmtree(reload_path)
    reload_path.mkdir(parents=True, exist_ok=True)
    return inspect_runtime_health(workspace_path)


_EXIT_STATUS_RE = re.compile(r"exit_code:\s*(-?\d+)\s+signal:\s*(\d+)")
_PROCESS_EXIT_RE = re.compile(r"process exited with code\s+(-?\d+)")


def _extract_exit_statuses(text: str) -> list[tuple[int, int]]:
    statuses: list[tuple[int, int]] = []
    for match in _EXIT_STATUS_RE.finditer(text):
        statuses.append((int(match.group(1)), int(match.group(2))))
    return statuses


def _classify_child_terminates(text: str) -> str | None:
    if "child terminates" not in text:
        return None
    if "signal: 11" in text or "Segmentation fault" in text:
        return "RUNNER_SEGFAULT"
    statuses = _extract_exit_statuses(text)
    if statuses and any(exit_code != 0 or signal_code != 0 for exit_code, signal_code in statuses):
        return "RUNNER_SEGFAULT"
    if "Expected output file missing" in text or "No timing output found" in text:
        return "RUNNER_SEGFAULT"
    return None


def _classify_process_exit(text: str) -> str | None:
    statuses = [int(match.group(1)) for match in _PROCESS_EXIT_RE.finditer(text)]
    if statuses and any(status != 0 for status in statuses):
        return "RUNNER_SEGFAULT"
    return None


def classify_infra_failure(text: str, metrics: dict[str, Any] | None = None) -> str | None:
    detected: list[str] = []
    for marker, code in INFRA_FAILURE_PATTERNS:
        if marker in text:
            detected.append(code)
    child_terminates_code = _classify_child_terminates(text)
    if child_terminates_code is not None:
        detected.append(child_terminates_code)
    process_exit_code = _classify_process_exit(text)
    if process_exit_code is not None:
        detected.append(process_exit_code)
    if metrics is not None:
        failure_code = metrics.get("validation/failure_code")
        if isinstance(failure_code, str) and failure_code:
            detected.append(failure_code)
    for code in INFRA_FAILURE_CODE_PRIORITY:
        if code in detected:
            return code
    return None


def _count_files(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for item in path.rglob("*") if item.is_file())


def _dir_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
