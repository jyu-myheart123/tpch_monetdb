from __future__ import annotations

import fnmatch
import re
from collections import Counter
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Optional

from tpch_monetdb.config import get_profile_observation_limits

CORE_IMPLEMENTATION_FILES: tuple[str, ...] = (
    "loader_impl.hpp",
    "loader_impl.cpp",
    "builder_impl.hpp",
    "builder_impl.cpp",
    "query_impl.hpp",
    "query_impl.cpp",
)

COMPANION_QUERY_GLOBS: tuple[str, ...] = (
    "query_*.cpp",
    "query_*.hpp",
)

QUERY_EDIT_FILES: tuple[str, ...] = CORE_IMPLEMENTATION_FILES + COMPANION_QUERY_GLOBS
QUERY_FOCUSED_EDIT_GLOBS: tuple[str, ...] = (
    "query_q*.cpp",
    "query_q*.hpp",
    "query_family_*.cpp",
    "query_family_*.hpp",
    "query_shared_*.cpp",
    "query_shared_*.hpp",
)
QUERY_CREATE_GLOBS: tuple[str, ...] = (
    "query_q*.cpp",
    "query_q*.hpp",
    "query_family_*.cpp",
    "query_family_*.hpp",
    "query_shared_*.cpp",
    "query_shared_*.hpp",
)
FOUNDATION_CORRECTNESS_EDIT_GLOBS: tuple[str, ...] = (
    *CORE_IMPLEMENTATION_FILES,
    *QUERY_FOCUSED_EDIT_GLOBS,
)

BUILD_OPTIMIZATION_FILES: tuple[str, ...] = QUERY_EDIT_FILES
OPTIMIZATION_QUERY_LOCAL_EDIT_GLOBS: tuple[str, ...] = QUERY_FOCUSED_EDIT_GLOBS
OPTIMIZATION_INFRA_LAYOUT_EDIT_GLOBS: tuple[str, ...] = QUERY_EDIT_FILES
INSTRUMENTATION_EDIT_GLOBS: tuple[str, ...] = (
    "loader_impl.hpp",
    "loader_impl.cpp",
    "builder_impl.hpp",
    "builder_impl.cpp",
    "query_impl.hpp",
    "query_impl.cpp",
    *QUERY_FOCUSED_EDIT_GLOBS,
)


HOST_OWNED_WRITE_GLOBS: tuple[str, ...] = (
    "workload_objective.json",
    "data_law_contract.json",
    "implementation_manifest.json",
    "*.implementation_manifest.json",
    "host_sealed_manifest.json",
    "host-sealed-manifest.json",
    "sealed_manifest.json",
    "benchmark_baseline*",
    "baseline_*.json",
    "query_registry_generated.cpp",
    "query_registry_generated.hpp",
    "generated/query_registry_generated.cpp",
    "generated/query_registry_generated.hpp",
    "generated/query_q*.cpp",
    "generated/query_q*.hpp",
    "generated/query_family_*.cpp",
    "generated/query_family_*.hpp",
    "generated/query_shared_*.cpp",
    "generated/query_shared_*.hpp",
    "build/generated/query_registry_generated.cpp",
    "build/generated/query_registry_generated.hpp",
)


TODO_CHECKBOX_RE = re.compile(r"^- \[(?P<marker>[ xX~>])\] (?P<content>.+)$", re.MULTILINE)
TODO_MARKER_TO_STATUS = {
    " ": "pending",
    "~": "in_progress",
    ">": "in_progress",
    "x": "completed",
    "X": "completed",
}
TODO_STATUS_ORDER = {
    "pending": 0,
    "in_progress": 1,
    "completed": 2,
}


class AgentRole(StrEnum):
    planner = "planner"
    implementer = "implementer"
    validator = "validator"
    optimizer = "optimizer"


@dataclass(frozen=True)
class TodoItem:
    content: str
    status: str
    active_form: str


@dataclass(frozen=True)
class TodoState:
    items: tuple[TodoItem, ...] = ()
    source_text: str = ""

    @classmethod
    def from_text(cls, source_text: str) -> "TodoState":
        items: list[TodoItem] = []
        for match in TODO_CHECKBOX_RE.finditer(source_text):
            status = TODO_MARKER_TO_STATUS[match.group("marker")]
            content = match.group("content").strip()
            items.append(
                TodoItem(
                    content=content,
                    status=status,
                    active_form=_normalize_todo_content(content),
                )
            )
        return cls(items=tuple(items), source_text=source_text)

    @classmethod
    def from_file(cls, todo_path: Path) -> Optional["TodoState"]:
        if not todo_path.exists():
            return None
        source_text = todo_path.read_text(encoding="utf-8")
        return cls.from_text(source_text)

    @property
    def completed_count(self) -> int:
        value = sum(1 for item in self.items if item.status == "completed")
        return value

    @property
    def pending_count(self) -> int:
        value = sum(1 for item in self.items if item.status == "pending")
        return value

    @property
    def in_progress_count(self) -> int:
        value = sum(1 for item in self.items if item.status == "in_progress")
        return value

    def progressed_count_from(self, previous: "TodoState") -> int:
        previous_map = previous.item_map()
        count = 0
        for item in self.items:
            previous_item = previous_map.get(item.active_form)
            if previous_item is None:
                continue
            if _todo_status_rank(item.status) > _todo_status_rank(previous_item.status):
                count += 1
        return count

    def is_valid_successor(self, previous: "TodoState") -> bool:
        previous_map = previous.item_map()
        for item in self.items:
            previous_item = previous_map.get(item.active_form)
            if previous_item is None:
                continue
            if _todo_status_rank(item.status) < _todo_status_rank(previous_item.status):
                return False
        return True

    def item_map(self) -> dict[str, TodoItem]:
        value = {item.active_form: item for item in self.items}
        return value


@dataclass
class StageState:
    profile_name: str
    prompt_index: int
    prompt_descriptor: str | None
    active_query_ids: tuple[str, ...] = ()
    active_unit_id: str | None = None
    active_unit_kind: str | None = None
    active_unit_files: tuple[str, ...] = ()
    active_unit_query_ids: tuple[str, ...] = ()
    objective_ids: tuple[str, ...] = ()
    data_law_ids: tuple[str, ...] = ()
    patch_scope_verdict: str | None = None
    tool_counts: Counter[str] = field(default_factory=Counter)
    consecutive_observation_count: int = 0
    written_files: set[str] = field(default_factory=set)
    last_compile_summary: str | None = None
    last_run_summary: str | None = None
    last_validation_summary: str | None = None
    last_compile_succeeded: bool | None = None
    last_run_succeeded: bool | None = None
    validation_passed: bool | None = None
    last_failure_kind: str | None = None
    consecutive_compile_failures: int = 0
    single_file_rebuild_target: str | None = None
    single_file_rebuild_reason: str | None = None
    last_compile_signature: str | None = None
    last_run_signature: str | None = None
    repeated_compile_count: int = 0
    repeated_run_count: int = 0
    compile_write_revision: int = -1
    run_write_revision: int = -1
    write_revision: int = 0
    require_write_after_failure: bool = False
    todo_before: TodoState | None = None
    todo_current: TodoState | None = None
    todo_after: TodoState | None = None
    soft_observation_limit_exceeded: bool = False
    control_artifacts_read: set[str] = field(default_factory=set)
    control_artifacts_injected: tuple[str, ...] = ()
    required_control_artifacts: tuple[str, ...] = ()


@dataclass(frozen=True)
class StageRunSummary:
    profile_name: str
    prompt_index: int
    prompt_descriptor: str | None
    final_output: str | None
    tool_counts: dict[str, int]
    written_files: tuple[str, ...]
    last_compile_summary: str | None
    last_run_summary: str | None
    todo_before: TodoState | None
    todo_after: TodoState | None
    last_validation_summary: str | None = None
    compile_succeeded: bool | None = None
    run_succeeded: bool | None = None
    validation_passed: bool | None = None
    last_failure_kind: str | None = None
    control_artifacts_read: tuple[str, ...] = ()
    active_unit_id: str | None = None
    active_unit_kind: str | None = None
    active_unit_files: tuple[str, ...] = ()
    active_unit_query_ids: tuple[str, ...] = ()
    objective_ids: tuple[str, ...] = ()
    data_law_ids: tuple[str, ...] = ()
    patch_scope_verdict: str | None = None
    control_artifacts_injected: tuple[str, ...] = ()
    required_control_artifacts: tuple[str, ...] = ()
    compile_write_revision: int = -1
    run_write_revision: int = -1
    write_revision: int = 0

    @property
    def has_writes(self) -> bool:
        return bool(self.written_files)

    @property
    def todo_progressed(self) -> bool:
        if self.todo_before is None or self.todo_after is None:
            return False
        if not self.todo_after.is_valid_successor(self.todo_before):
            return False
        return self.todo_after.progressed_count_from(self.todo_before) > 0


@dataclass(frozen=True)
class ToolProfile:
    name: str
    tool_names: tuple[str, ...]
    read_globs: tuple[str, ...]
    edit_globs: tuple[str, ...] = ()
    create_globs: tuple[str, ...] = ()
    write_globs: tuple[str, ...] = ()
    allow_write_create: bool = False
    allow_write_overwrite: bool = False
    max_consecutive_observations: int = 8
    hard_consecutive_observations: int | None = None

    def allows_read(self, relative_path: str) -> bool:
        return _matches_any(relative_path, self.read_globs)

    def allows_edit(self, relative_path: str) -> bool:
        return (
            _matches_any(relative_path, self.edit_globs)
            and not _is_host_owned_write_path(relative_path)
        )

    def allows_create(self, relative_path: str) -> bool:
        return (
            _matches_any(relative_path, self.create_globs)
            and not _is_host_owned_write_path(relative_path)
        )

    def allows_write(self, relative_path: str) -> bool:
        return (
            _matches_any(relative_path, self.write_globs)
            and not _is_host_owned_write_path(relative_path)
        )


@dataclass(frozen=True)
class DelegationRequest:
    role: AgentRole
    prompt: str
    context_summary: str


@dataclass(frozen=True)
class DelegationResult:
    role: AgentRole
    summary: str
    files_touched: tuple[str, ...] = ()


@dataclass(frozen=True)
class WorkContextSummary:
    prompt_descriptor: str | None
    profile_name: str
    files_in_scope: tuple[str, ...]


def build_tool_profiles() -> dict[str, ToolProfile]:
    """Return the single source of truth for TPC-H MonetDB stage tool permissions."""
    def _obs(profile_name: str) -> dict[str, int | None]:
        soft_limit, hard_limit = get_profile_observation_limits(profile_name)
        return {
            "max_consecutive_observations": soft_limit,
            "hard_consecutive_observations": hard_limit,
        }

    shared_query_profiles = (
        "read_file",
        "read_artifact",
        "list_files",
        "grep_repo",
        "edit_file",
        "compile",
        "run",
    )
    shared_write_only_profiles = (
        "read_file",
        "read_artifact",
        "list_files",
        "grep_repo",
        "edit_file",
    )
    profiles = {
        "legacy_general": ToolProfile(
            name="legacy_general",
            tool_names=(
                "read_file",
                "read_artifact",
                "list_files",
                "grep_repo",
                "edit_file",
                "write_file",
                "shell",
                "compile",
                "run",
            ),
            read_globs=("*",),
            edit_globs=("*",),
            write_globs=("*",),
            allow_write_create=True,
            allow_write_overwrite=True,
            **_obs("legacy_general"),
        ),
        "default_general": ToolProfile(
            name="default_general",
            tool_names=shared_query_profiles,
            read_globs=("*",),
            edit_globs=QUERY_EDIT_FILES,
            **_obs("default_general"),
        ),
        "todo_plan": ToolProfile(
            name="todo_plan",
            tool_names=(
                "read_file",
                "read_artifact",
                "list_files",
                "grep_repo",
                "write_file",
            ),
            read_globs=("*",),
            write_globs=("TODO.md",),
            allow_write_create=True,
            allow_write_overwrite=True,
            **_obs("todo_plan"),
        ),
        "storage_plan": ToolProfile(
            name="storage_plan",
            tool_names=(
                "read_file",
                "read_artifact",
                "list_files",
                "grep_repo",
                "write_file",
            ),
            read_globs=("*",),
            write_globs=("storage_plan.txt", "storage_plan_contract.json"),
            allow_write_create=True,
            allow_write_overwrite=True,
            **_obs("storage_plan"),
        ),
        "finish_skeleton": ToolProfile(
            name="finish_skeleton",
            tool_names=(
                "read_file",
                "read_artifact",
                "list_files",
                "grep_repo",
                "edit_file",
                "apply_patch",
            ),
            read_globs=("*",),
            edit_globs=QUERY_EDIT_FILES,
            create_globs=QUERY_CREATE_GLOBS,
            **_obs("finish_skeleton"),
        ),
        "compile_fix": ToolProfile(
            name="compile_fix",
            tool_names=shared_query_profiles + ("apply_patch",),
            read_globs=("*",),
            edit_globs=QUERY_EDIT_FILES,
            create_globs=QUERY_CREATE_GLOBS,
            **_obs("compile_fix"),
        ),
        "todo_sync": ToolProfile(
            name="todo_sync",
            tool_names=(
                "read_file",
                "read_artifact",
                "write_file",
            ),
            read_globs=("TODO.md",),
            write_globs=("TODO.md",),
            allow_write_create=True,
            allow_write_overwrite=True,
            **_obs("todo_sync"),
        ),
        "add_timings": ToolProfile(
            name="add_timings",
            tool_names=shared_query_profiles + ("apply_patch",),
            read_globs=("*",),
            edit_globs=QUERY_EDIT_FILES,
            **_obs("add_timings"),
        ),
        "implement_queries": ToolProfile(
            name="implement_queries",
            tool_names=shared_query_profiles + ("apply_patch",),
            read_globs=("*",),
            edit_globs=QUERY_EDIT_FILES,
            **_obs("implement_queries"),
        ),
        "implement_queries_writeonly": ToolProfile(
            name="implement_queries_writeonly",
            tool_names=shared_write_only_profiles + ("write_file", "apply_patch"),
            read_globs=("*",),
            edit_globs=QUERY_FOCUSED_EDIT_GLOBS,
            write_globs=QUERY_FOCUSED_EDIT_GLOBS,
            allow_write_create=True,
            allow_write_overwrite=True,
            create_globs=QUERY_CREATE_GLOBS,
            **_obs("implement_queries_writeonly"),
        ),
        "correctness_queries_writeonly": ToolProfile(
            name="correctness_queries_writeonly",
            tool_names=shared_query_profiles + ("write_file", "apply_patch"),
            read_globs=("*",),
            edit_globs=QUERY_FOCUSED_EDIT_GLOBS,
            write_globs=QUERY_FOCUSED_EDIT_GLOBS,
            allow_write_overwrite=True,
            create_globs=QUERY_CREATE_GLOBS,
            **_obs("correctness_queries_writeonly"),
        ),
        "correctness_foundation": ToolProfile(
            name="correctness_foundation",
            tool_names=shared_query_profiles + ("write_file", "apply_patch"),
            read_globs=("*",),
            edit_globs=FOUNDATION_CORRECTNESS_EDIT_GLOBS,
            write_globs=FOUNDATION_CORRECTNESS_EDIT_GLOBS,
            allow_write_overwrite=True,
            create_globs=QUERY_CREATE_GLOBS,
            **_obs("correctness_foundation"),
        ),
        "correctness": ToolProfile(
            name="correctness",
            tool_names=shared_query_profiles + ("apply_patch",),
            read_globs=("*",),
            edit_globs=QUERY_EDIT_FILES,
            create_globs=QUERY_CREATE_GLOBS,
            **_obs("correctness"),
        ),
        "benchmark": ToolProfile(
            name="benchmark",
            tool_names=shared_query_profiles + ("apply_patch", "cpu_info"),
            read_globs=("*",),
            edit_globs=QUERY_EDIT_FILES,
            create_globs=QUERY_CREATE_GLOBS,
            **_obs("benchmark"),
        ),
        "optimize_build": ToolProfile(
            name="optimize_build",
            tool_names=shared_query_profiles + ("apply_patch",),
            read_globs=("*",),
            edit_globs=BUILD_OPTIMIZATION_FILES,
            create_globs=QUERY_CREATE_GLOBS,
            **_obs("optimize_build"),
        ),
        "optimization_general": ToolProfile(
            name="optimization_general",
            tool_names=(
                "read_file",
                "read_artifact",
                "list_files",
                "grep_repo",
                "cpu_info",
                "edit_file",
                "apply_patch",
                "compile",
                "run",
                "shell",
            ),
            read_globs=("*",),
            edit_globs=OPTIMIZATION_QUERY_LOCAL_EDIT_GLOBS,
            create_globs=QUERY_CREATE_GLOBS,
            **_obs("optimization_general"),
        ),
        "optimization_infra_layout": ToolProfile(
            name="optimization_infra_layout",
            tool_names=(
                "read_file",
                "read_artifact",
                "list_files",
                "grep_repo",
                "cpu_info",
                "edit_file",
                "apply_patch",
                "compile",
                "run",
                "shell",
            ),
            read_globs=("*",),
            edit_globs=OPTIMIZATION_INFRA_LAYOUT_EDIT_GLOBS,
            create_globs=QUERY_CREATE_GLOBS,
            **_obs("optimization_general"),
        ),
        "optimization_instrumentation": ToolProfile(
            name="optimization_instrumentation",
            tool_names=(
                "read_file",
                "read_artifact",
                "list_files",
                "grep_repo",
                "cpu_info",
                "edit_file",
                "apply_patch",
                "compile",
                "run",
            ),
            read_globs=("*",),
            edit_globs=INSTRUMENTATION_EDIT_GLOBS,
            create_globs=QUERY_CREATE_GLOBS,
            **_obs("optimization_instrumentation"),
        ),
        "optimization_control": ToolProfile(
            name="optimization_control",
            tool_names=(
                "read_file",
                "read_artifact",
                "list_files",
                "grep_repo",
            ),
            read_globs=(
                "TODO.md",
                "storage_plan.txt",
                "optimization_hotspot_summary.md",
                "workload_objective.json",
                "data_law_contract.json",
                "storage_plan_contract.json",
                "tracing_output.log",
                "query_*.cpp",
                "query_*.hpp",
                "builder_impl.*",
                "loader_impl.*",
            ),
            edit_globs=(),
            create_globs=(),
            **_obs("optimization_control"),
        ),
        "optimization_todo_sync": ToolProfile(
            name="optimization_todo_sync",
            tool_names=(
                "read_file",
                "read_artifact",
                "write_file",
            ),
            read_globs=(
                "TODO.md",
                "storage_plan.txt",
                "optimization_hotspot_summary.md",
            ),
            write_globs=("TODO.md",),
            allow_write_overwrite=True,
            **_obs("optimization_todo_sync"),
        ),
    }
    return profiles


def summarize_tool_output(output: str | None, max_lines: int = 12) -> str | None:
    if output is None:
        return None
    lines = output.splitlines()
    if len(lines) <= max_lines:
        return output
    head = lines[: max_lines // 2]
    tail = lines[-(max_lines // 2) :]
    summarized = "\n".join(head + [f"... [{len(lines) - len(head) - len(tail)} lines truncated] ..."] + tail)
    return summarized


VALIDATION_TEXT_MARKERS: tuple[str, ...] = (
    "validation fail",
    "validation failed",
    "validation pass",
    "validation passed",
    "all queries passed validation",
    "validation result",
    "validator:",
    "oracle execution failed",
)


def looks_like_validation_text(text: str | None) -> bool:
    if text is None:
        return False
    normalized = text.lower()
    return any(marker in normalized for marker in VALIDATION_TEXT_MARKERS)


def extract_validation_summary(output: str | None, max_lines: int = 8) -> str | None:
    if output is None:
        return None
    matched_lines: list[str] = []
    for line in output.splitlines():
        normalized_line = line.strip()
        if not normalized_line:
            continue
        if looks_like_validation_text(normalized_line):
            matched_lines.append(normalized_line)
    if not matched_lines:
        return None
    return summarize_tool_output("\n".join(matched_lines), max_lines=max_lines)


def infer_validation_passed(
    output: str | None,
    success: bool,
    validation_summary: str | None,
) -> bool | None:
    if validation_summary is not None:
        normalized = validation_summary.lower()
        if "validation fail" in normalized or "oracle execution failed" in normalized:
            return False
        if "validation pass" in normalized or "all queries passed validation" in normalized:
            return True
    if output is None:
        return None
    normalized_output = output.lower()
    if "all queries passed validation" in normalized_output:
        return True
    if "validation fail" in normalized_output or "oracle execution failed" in normalized_output:
        return False
    if not success:
        return False
    return None


def _matches_any(relative_path: str, patterns: tuple[str, ...]) -> bool:
    if not patterns:
        return False
    normalized = Path(relative_path).as_posix()
    return any(fnmatch.fnmatch(normalized, pattern) for pattern in patterns)


def _is_host_owned_write_path(relative_path: str) -> bool:
    """Return whether a path is host-owned and must not be edited by agents."""
    normalized = Path(relative_path).as_posix().lstrip("./")
    return any(fnmatch.fnmatch(normalized, pattern) for pattern in HOST_OWNED_WRITE_GLOBS)


def _normalize_todo_content(content: str) -> str:
    value = " ".join(content.strip().lower().split())
    return value


def _todo_status_rank(status: str) -> int:
    value = TODO_STATUS_ORDER[status]
    return value
