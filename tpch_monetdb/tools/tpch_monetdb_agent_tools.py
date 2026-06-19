from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from agents.run_context import RunContextWrapper
from agents.tool import FunctionTool
from pydantic import BaseModel, Field

from tpch_monetdb.config import get_max_stalled_executions
from tpch_monetdb.conversations.agent_text_registry import render_agent_text_asset
from tpch_monetdb.llm_cache.artifact_ledger import ArtifactLedger, build_preview
from tpch_monetdb.llm_cache.stage_memory import render_stage_memory
from tpch_monetdb.tools.function_tool_args import load_function_tool_args
from tpch_monetdb.tools.error_envelope import ErrorEnvelope
from tpch_monetdb.tools.cpu_info import make_cpu_info_tool
from tpch_monetdb.tools.litellm_shell import make_litellm_shell_tool
from tpch_monetdb.tools.stage_tool_policy import (
    extract_validation_summary,
    infer_validation_passed,
    StageRunSummary,
    StageState,
    TodoState,
    ToolProfile,
    build_tool_profiles,
    summarize_tool_output,
)
from tpch_monetdb.tools.tool_parallelism import (
    AsyncRWLock,
    is_exclusive_tool,
    is_read_only_tool,
)
from tpch_monetdb.utils.control_artifacts import (
    TRACKED_CONTROL_ARTIFACTS,
    ensure_required_control_artifacts_acknowledged,
    ensure_required_control_artifacts_present,
)

_PRIMARY_QUERY_DESCRIPTOR_RE = re.compile(
    r"^correctness_(?:q(?P<short>\d+)|query_(?P<long>\d+))$"
)
_CORRUPTED_QUERY_FILE_PATTERNS = (
    re.compile(r"redefinition of"),
    re.compile(r"extraneous closing brace"),
    re.compile(r"expected unqualified-id"),
)
_PRIMARY_QUERY_VALIDATION_PROFILES = frozenset({
    "correctness_queries_writeonly",
    "correctness_foundation",
})
_RUN_EVIDENCE_CHAR_LIMIT = 24_000
_RUN_INLINE_CHAR_LIMIT = 8_000
_COMPILE_INLINE_CHAR_LIMIT = 12_000
_READ_INLINE_CHAR_LIMIT = 16_000
_GREP_INLINE_CHAR_LIMIT = 12_000
_TOOL_FULL_READ_MAX_BYTES = 1_000_000
_TOOL_GREP_MAX_BYTES = 2_000_000
_ARTIFACT_READ_DEFAULT_LIMIT = 200
_ARTIFACT_READ_MAX_LIMIT = 1000
_ARTIFACT_READ_MAX_OUTPUT_BYTES = 16_000
_BASE_PERFORMANCE_COMPARISON_ARG = "__base_performance_comparison"


@dataclass(frozen=True)
class ToolBundle:
    all_tools: list[Any]
    tools_by_profile: dict[str, list[Any]]
    runtime: "StageToolRuntime"


def _render_stage_hint_scope(label: str, values: tuple[str, ...]) -> str:
    """Render one stage-hint scope line from the registered text asset."""
    return render_agent_text_asset(
        "runtime.stage_hint.scope",
        {"label": label, "values": ", ".join(values)},
    )


def _render_stage_stop_condition(condition: str) -> str:
    """Render one stage-hint stop-condition line from the registered text asset."""
    return render_agent_text_asset(
        "runtime.stage_hint.stop_condition",
        {"condition": condition},
    )


def _render_runtime_policy(asset_suffix: str, **variables: object) -> str:
    """Render one registered runtime policy message or next-action asset."""
    return render_agent_text_asset(f"runtime.{asset_suffix}", variables).rstrip()


def _render_runtime_text(asset_suffix: str, **variables: object) -> str:
    """Render one registered runtime text asset outside the policy namespace."""
    return render_agent_text_asset(f"runtime.{asset_suffix}", variables).rstrip()


class ReadFileArgs(BaseModel):
    file_path: str = Field(..., description="Path relative to workspace root")
    offset: int | None = Field(None, ge=1, description="1-based start line")
    limit: int | None = Field(None, ge=1, description="Maximum number of lines")


class ReadArtifactArgs(BaseModel):
    artifact_ref: str = Field(
        ...,
        description="Stable artifact_ref from Evidence Digest or Artifact Refs",
    )
    offset: int | None = Field(None, ge=1, description="1-based start line")
    limit: int | None = Field(
        None,
        ge=1,
        le=_ARTIFACT_READ_MAX_LIMIT,
        description="Maximum number of lines",
    )


class ListFilesArgs(BaseModel):
    path: str | None = Field(None, description="Directory relative to workspace root")
    pattern: str | None = Field(None, description="Optional glob pattern")
    limit: int = Field(200, ge=1, le=1000, description="Maximum number of paths")


class GrepRepoArgs(BaseModel):
    pattern: str = Field(..., description="Python regular expression")
    path: str | None = Field(None, description="Directory relative to workspace root")
    glob: str | None = Field(None, description="Optional glob pattern")
    limit: int = Field(50, ge=1, le=500, description="Maximum number of matches")


class EditFileArgs(BaseModel):
    file_path: str = Field(..., description="Path relative to workspace root")
    old_string: str = Field(..., description="Exact text to replace")
    new_string: str = Field(..., description="Replacement text")
    replace_all: bool = Field(False, description="Replace all exact matches")


class WriteFileArgs(BaseModel):
    file_path: str = Field(..., description="Path relative to workspace root")
    content: str = Field(..., description="Full file content")


class StagePolicyError(RuntimeError):
    """Base class for stage policy violations."""
    pass


class RecoverableStagePolicyError(StagePolicyError):
    """Recoverable stage policy violation - returned to model as tool output."""
    pass


class FatalStagePolicyError(StagePolicyError):
    """Fatal stage policy violation - aborts the agent run."""
    pass


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


class StageToolRuntime:
    def __init__(
        self,
        workspace_root: Path,
        *,
        extra_read_roots: tuple[Path, ...] = (),
        artifact_ledger: ArtifactLedger | None = None,
    ) -> None:
        self.workspace_root = workspace_root.resolve()
        self.artifact_ledger = artifact_ledger
        ledger_roots = (artifact_ledger.root_dir,) if artifact_ledger is not None else ()
        self.extra_read_roots = tuple(
            path.resolve() for path in (*extra_read_roots, *ledger_roots)
        )
        self.profiles = build_tool_profiles()
        self._active_profile_name = "default_general"
        self._state = StageState(
            profile_name=self._active_profile_name,
            prompt_index=-1,
            prompt_descriptor=None,
        )
        # phase10: 对 _state 的任何写入都走这把同步锁，避免并发 tool 调用下
        # observation count / write-after-failure / tool_counts 出现 race。
        self._state_lock = threading.Lock()
        # phase10: 对工具执行语义加读写锁，读工具 shared、写/执行工具 exclusive，
        # 保证 mixed batch 里 write 与 read 不会真正并发。
        self._rw_lock = AsyncRWLock()

    def tool_guard(self, tool_name: str):
        """返回 tool-level 锁的 async guard.

        读工具（read_file/list_files/grep_repo/只读 shell）获取 shared lock；
        写 / 执行工具获取 exclusive lock。未知工具按 exclusive 处理以保守起见。
        """
        if is_read_only_tool(tool_name) and not is_exclusive_tool(tool_name):
            return self._rw_lock.shared()
        return self._rw_lock.exclusive()

    def activate(
        self,
        profile_name: Optional[str],
        prompt_index: int,
        prompt_descriptor: Optional[str],
        prompt_metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """Activate a stage profile and attach any prompt-scoped runtime metadata."""
        resolved_profile = profile_name or "default_general"
        if resolved_profile not in self.profiles:
            raise RuntimeError(f"Unknown tool_profile: {resolved_profile}")
        todo_state = self._load_todo_state()
        active_query_ids: tuple[str, ...] = ()
        active_unit_id: str | None = None
        active_unit_kind: str | None = None
        active_unit_files: tuple[str, ...] = ()
        active_unit_query_ids: tuple[str, ...] = ()
        objective_ids: tuple[str, ...] = ()
        data_law_ids: tuple[str, ...] = ()
        patch_scope_verdict: str | None = None
        control_artifacts_injected: tuple[str, ...] = ()
        required_control_artifacts: tuple[str, ...] = ()
        if prompt_metadata is not None:
            raw_query_ids = prompt_metadata.get("active_query_ids", ())
            if isinstance(raw_query_ids, (list, tuple)):
                active_query_ids = tuple(str(item) for item in raw_query_ids)
            raw_injected = prompt_metadata.get("control_artifacts_injected", ())
            if isinstance(raw_injected, (list, tuple)):
                control_artifacts_injected = tuple(str(item) for item in raw_injected)
            raw_required = prompt_metadata.get("required_control_artifacts", ())
            if isinstance(raw_required, (list, tuple)):
                required_control_artifacts = tuple(str(item) for item in raw_required)
            raw_unit_id = prompt_metadata.get("active_unit_id")
            if isinstance(raw_unit_id, str) and raw_unit_id:
                active_unit_id = raw_unit_id
            raw_unit_kind = prompt_metadata.get("active_unit_kind")
            if isinstance(raw_unit_kind, str) and raw_unit_kind:
                active_unit_kind = raw_unit_kind
            raw_unit_files = prompt_metadata.get("active_unit_files", ())
            if isinstance(raw_unit_files, (list, tuple)):
                active_unit_files = tuple(str(item) for item in raw_unit_files)
            raw_unit_query_ids = prompt_metadata.get("active_unit_query_ids", ())
            if isinstance(raw_unit_query_ids, (list, tuple)):
                active_unit_query_ids = tuple(str(item) for item in raw_unit_query_ids)
            raw_objective_ids = prompt_metadata.get("objective_ids", ())
            if isinstance(raw_objective_ids, (list, tuple)):
                objective_ids = tuple(str(item) for item in raw_objective_ids)
            raw_data_law_ids = prompt_metadata.get("data_law_ids", ())
            if isinstance(raw_data_law_ids, (list, tuple)):
                data_law_ids = tuple(str(item) for item in raw_data_law_ids)
            raw_patch_scope_verdict = prompt_metadata.get("patch_scope_verdict")
            if isinstance(raw_patch_scope_verdict, str) and raw_patch_scope_verdict:
                patch_scope_verdict = raw_patch_scope_verdict
        with self._state_lock:
            self._active_profile_name = resolved_profile
            self._state = StageState(
                profile_name=resolved_profile,
                prompt_index=prompt_index,
                prompt_descriptor=prompt_descriptor,
                active_query_ids=active_query_ids,
                active_unit_id=active_unit_id,
                active_unit_kind=active_unit_kind,
                active_unit_files=active_unit_files,
                active_unit_query_ids=active_unit_query_ids,
                objective_ids=objective_ids,
                data_law_ids=data_law_ids,
                patch_scope_verdict=patch_scope_verdict,
                todo_before=todo_state,
                todo_current=todo_state,
                control_artifacts_injected=control_artifacts_injected,
                required_control_artifacts=required_control_artifacts,
            )
        self._ensure_required_control_artifacts_present()
        return None

    def finish_stage(self, final_output: str | None) -> StageRunSummary:
        """Build the immutable summary used by conversation postcondition checks."""
        self._state.todo_after = self._state.todo_current
        if self._state.todo_after is None:
            self._state.todo_after = self._load_todo_state()
        summary = StageRunSummary(
            profile_name=self._state.profile_name,
            prompt_index=self._state.prompt_index,
            prompt_descriptor=self._state.prompt_descriptor,
            active_unit_id=self._state.active_unit_id,
            active_unit_kind=self._state.active_unit_kind,
            active_unit_files=self._state.active_unit_files,
            active_unit_query_ids=self._state.active_unit_query_ids,
            objective_ids=self._state.objective_ids,
            data_law_ids=self._state.data_law_ids,
            patch_scope_verdict=self._state.patch_scope_verdict,
            final_output=final_output,
            tool_counts=dict(self._state.tool_counts),
            written_files=tuple(sorted(self._state.written_files)),
            last_compile_summary=self._state.last_compile_summary,
            last_run_summary=self._state.last_run_summary,
            todo_before=self._state.todo_before,
            todo_after=self._state.todo_after,
            last_validation_summary=self._state.last_validation_summary,
            compile_succeeded=self._state.last_compile_succeeded,
            run_succeeded=self._state.last_run_succeeded,
            validation_passed=self._state.validation_passed,
            last_failure_kind=self._state.last_failure_kind,
            control_artifacts_read=tuple(
                sorted(self._state.control_artifacts_read)
            ),
            control_artifacts_injected=self._state.control_artifacts_injected,
            required_control_artifacts=self._state.required_control_artifacts,
            compile_write_revision=self._state.compile_write_revision,
            run_write_revision=self._state.run_write_revision,
            write_revision=self._state.write_revision,
        )
        return summary

    def get_active_profile_name(self) -> str:
        return self._active_profile_name

    def run_args_with_stage_metadata(self, args_json: str) -> str:
        """Attach internal run-tool metadata required by the active stage."""
        if self._active_profile_name != "benchmark":
            return args_json
        args = load_function_tool_args(args_json)
        args["optimize"] = True
        args[_BASE_PERFORMANCE_COMPARISON_ARG] = True
        return json.dumps(args)

    def generate_stage_hint(self) -> str:
        """Generate a stage hint with available tools and editable scope.
        
        Returns:
            Stage hint text to prepend to model input
        """
        profile = self.profiles[self._active_profile_name]

        lines = [
            render_agent_text_asset(
                "runtime.stage_hint.header",
                {
                    "profile_name": profile.name,
                    "tool_names": ", ".join(profile.tool_names),
                },
            )
        ]
        
        if profile.edit_globs:
            lines.append(_render_stage_hint_scope(
                _render_runtime_text("stage_hint_label_editable_files"),
                profile.edit_globs,
            ))
        if profile.create_globs:
            lines.append(_render_stage_hint_scope(
                _render_runtime_text("stage_hint_label_creatable_files"),
                profile.create_globs,
            ))
        if profile.write_globs:
            lines.append(_render_stage_hint_scope(
                _render_runtime_text("stage_hint_label_writable_files"),
                profile.write_globs,
            ))
        
        # Keep stage hint limited to tool truth, file scope, and exit condition.
        if profile.name == "todo_plan":
            lines.append(_render_stage_stop_condition(
                _render_runtime_text("stage_hint_condition_todo_plan")
            ))
        if profile.name == "storage_plan":
            lines.append(_render_stage_stop_condition(
                _render_runtime_text("stage_hint_condition_storage_plan")
            ))
        if profile.name == "finish_skeleton":
            lines.append(render_agent_text_asset("runtime.stage_hint.compile_run_unavailable"))
            lines.append(_render_stage_stop_condition(
                _render_runtime_text("stage_hint_condition_finish_skeleton")
            ))
        if profile.name == "implement_queries_writeonly":
            lines.append(render_agent_text_asset("runtime.stage_hint.compile_run_unavailable"))
            lines.append(_render_stage_stop_condition(
                _render_runtime_text("stage_hint_condition_implement_queries_writeonly")
            ))
        if profile.name == "correctness_queries_writeonly":
            lines.append(_render_stage_stop_condition(
                _render_runtime_text("stage_hint_condition_correctness_queries_writeonly")
            ))
        if profile.name == "correctness_foundation":
            lines.append(_render_stage_stop_condition(
                _render_runtime_text("stage_hint_condition_correctness_foundation")
            ))
        if self._state.active_query_ids:
            lines.append(render_agent_text_asset(
                "runtime.stage_hint.active_query_batch",
                {"query_ids": ", ".join(self._state.active_query_ids)},
            ))
        if self._state.active_unit_id is not None:
            lines.append(render_agent_text_asset(
                "runtime.stage_hint.active_unit",
                {"unit_id": self._state.active_unit_id},
            ))
        if self._state.active_unit_kind is not None:
            lines.append(render_agent_text_asset(
                "runtime.stage_hint.active_unit_kind",
                {"unit_kind": self._state.active_unit_kind},
            ))
        if self._state.active_unit_query_ids:
            lines.append(render_agent_text_asset(
                "runtime.stage_hint.active_unit_queries",
                {"query_ids": ", ".join(self._state.active_unit_query_ids)},
            ))
        if self._state.single_file_rebuild_target is not None:
            lines.append(render_agent_text_asset(
                "runtime.stage_hint.single_file_rebuild",
                {"target": self._state.single_file_rebuild_target},
            ))
        
        # Add write capabilities info
        write_tools = [t for t in profile.tool_names if t in ("edit_file", "write_file", "apply_patch")]
        if write_tools:
            lines.append(render_agent_text_asset(
                "runtime.stage_hint.write_tools",
                {"tool_names": ", ".join(write_tools)},
            ))
        
        return "\n".join(lines)

    def generate_stage_memory(self, artifact_refs: str | None = None) -> str:
        """Generate a stable stage-memory block for the prompt builder."""
        with self._state_lock:
            return render_stage_memory(self._state, artifact_refs=artifact_refs)

    def prepare_tool_evidence(
        self,
        *,
        tool_name: str,
        output: str,
        success: bool | None,
        inline_limit: int,
        kind: str,
    ) -> str:
        """Return inline output or an artifact digest for large tool evidence."""
        stripped = output.strip()
        if len(stripped) <= inline_limit or self.artifact_ledger is None:
            return stripped
        summary = summarize_tool_output(stripped) or ""
        with self._state_lock:
            metadata = {
                "stage_name": self._state.profile_name,
                "prompt_index": self._state.prompt_index,
                "prompt_descriptor": self._state.prompt_descriptor,
                "tool_name": tool_name,
                "query_ids": self._state.active_unit_query_ids or self._state.active_query_ids,
                "success": success,
                "summary": summary,
                "tags": (tool_name, self._state.profile_name),
            }
        artifact = self.artifact_ledger.record_text(
            kind=kind,
            text=stripped,
            metadata=metadata,
        )
        preview, omitted = build_preview(stripped, limit=inline_limit)
        return self.artifact_ledger.render_digest(
            artifact,
            preview=preview,
            omitted_chars=omitted,
        )

    def get_tools_for_profile(self, profile_name: Optional[str], tool_map: dict[str, Any]) -> list[Any]:
        resolved_profile = profile_name or "default_general"
        if resolved_profile not in self.profiles:
            raise RuntimeError(f"Unknown tool_profile: {resolved_profile}")
        profile = self.profiles[resolved_profile]
        tools = [tool_map[name] for name in profile.tool_names]
        return tools

    def require_tool(self, tool_name: str) -> ToolProfile:
        profile = self.profiles[self._active_profile_name]
        if tool_name not in profile.tool_names:
            raise self._recoverable_error(
                code="TOOL_NOT_ALLOWED",
                category="stage_policy",
                message=_render_runtime_policy(
                    "policy_tool_not_allowed_message",
                    tool_name=tool_name,
                    profile_name=profile.name,
                ),
                allowed_next_actions=profile.tool_names,
                relevant_files=self._profile_scope_files(),
                recommended_next_action=_render_runtime_policy(
                    "policy_tool_not_allowed_next_action"
                ),
            )
        return profile

    def resolve_readable_path(self, relative_path: str) -> Path:
        profile = self.require_tool("read_file")
        return self._resolve_path(relative_path, profile, mode="read")

    def _is_under_extra_read_root(self, target: Path) -> bool:
        return any(
            _is_relative_to(target, root)
            for root in self.extra_read_roots
        )

    def _display_path(self, target: Path) -> str:
        try:
            return target.relative_to(self.workspace_root).as_posix()
        except ValueError:
            return target.as_posix()

    def list_directory(self, relative_path: Optional[str], pattern: Optional[str], limit: int) -> str:
        """List readable workspace files for the active tool profile."""
        # 第1步：获取权限检查配置
        profile = self.require_tool("read_file")  # list_files 使用与 read_file 相同的权限模型
        
        # 第2步：确定要列表的目录
        # 如果用户没指定路径，默认列表工作区根目录
        if relative_path is None:
            target_dir = self.workspace_root
        else:
            # 解析相对路径为绝对路径
            target_dir = self._resolve_path(relative_path, profile, mode="read")
        
        # 确保目标是一个目录
        if not target_dir.is_dir():
            display = self._display_path(target_dir)
            return f"{display} is not a directory"
        
        # 第3步：收集候选文件
        # 如果有 glob pattern，使用它来过滤；否则列表所有内容
        candidates = []
        if pattern is not None:
            # 使用 glob pattern 搜索
            # pattern 支持 * ? ** 等标准 glob 语法
            try:
                matches = list(target_dir.glob(pattern))
            except (ValueError, OSError) as e:
                return f"Invalid glob pattern: {pattern} ({e})"
            candidates = sorted(matches)
        else:
            # 没有 pattern，列表目录中的直接子项
            try:
                items = sorted(target_dir.iterdir())
            except (PermissionError, OSError) as e:
                return f"Cannot read directory: {self._display_path(target_dir)} ({e})"
            candidates = items
        
        # 第4步：过滤不可读的路径（权限检查）
        readable = []
        for candidate in candidates:
            # 检查路径是否在可读范围内（workspace_root 或 extra_read_roots）
            if _is_relative_to(candidate, self.workspace_root):
                readable.append(candidate)
            elif self._is_under_extra_read_root(candidate):
                readable.append(candidate)
            # 否则跳过不在授权范围内的路径
        
        # 第5步：应用数量限制
        limited = readable[:limit]
        
        # 第6步：格式化输出
        # 目录后面追加 /，文件直接显示
        # 输出顺序稳定（已排序）
        lines = []
        for item in limited:
            display = self._display_path(item)
            if item.is_dir():
                lines.append(display + "/")
            else:
                lines.append(display)
        
        # 只返回限制数量的结果，不附加额外的摘要行
        return "\n".join(lines) if lines else "(empty)"

    def grep_repo(self, pattern: str, relative_path: Optional[str], glob: Optional[str], limit: int) -> str:
        """Search readable files while surfacing skipped oversized evidence files."""
        # 第1步：编译正则表达式
        try:
            regex = re.compile(pattern)
        except re.error as e:
            return f"Invalid regex pattern: {pattern} ({e})"
        
        # 第2步：确定搜索根目录
        profile = self.require_tool("read_file")
        if relative_path is None:
            search_root = self.workspace_root
        else:
            search_root = self._resolve_path(relative_path, profile, mode="read")
        
        if search_root.is_file():
            candidates = [search_root]
        elif not search_root.is_dir():
            display = self._display_path(search_root)
            return f"{display} is not a directory"
        elif glob is not None:
            # 使用 glob pattern 过滤
            try:
                candidates = list(search_root.glob(glob))
            except (ValueError, OSError) as e:
                return f"Invalid glob pattern: {glob} ({e})"
        else:
            # 递归遍历所有文件
            candidates = []
            for item in search_root.rglob("*"):
                candidates.append(item)
        
        # 第4步：过滤出可读的普通文件
        readable_files = []
        for candidate in candidates:
            # 跳过目录
            if not candidate.is_file():
                continue
            
            # 权限检查
            if not (_is_relative_to(candidate, self.workspace_root) or 
                    self._is_under_extra_read_root(candidate)):
                continue
            
            readable_files.append(candidate)
        
        # 第5步：搜索匹配
        matches = []
        skipped_large_files = []
        
        for file_path in readable_files:
            # 检查文件大小
            try:
                size = file_path.stat().st_size
            except (OSError, PermissionError):
                continue
            
            if size > _TOOL_GREP_MAX_BYTES:
                # 文件太大，记录为跳过，不读取
                skipped_large_files.append((self._display_path(file_path), size))
                continue
            
            # 尝试读取文件
            try:
                content = file_path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError, PermissionError):
                # 跳过无法解码或无法读取的文件
                continue
            
            # 在文件中搜索匹配
            # 逐行进行正则搜索
            lines = content.splitlines()
            for line_no, line in enumerate(lines, start=1):
                if regex.search(line):
                    # 找到匹配！格式化为 "path:line_no:line"
                    display_path = self._display_path(file_path)
                    match_text = f"{display_path}:{line_no}:{line}"
                    matches.append(match_text)
                    
                    # 检查是否达到限制
                    if len(matches) >= limit:
                        break
            
            if len(matches) >= limit:
                break
        
        # 第6步：组装结果
        result_lines = []
        
        if matches:
            result_lines.extend(matches[:limit])
            if len(matches) > limit:
                remaining = len(matches) - limit
                result_lines.append(f"... and {remaining} more matches (increase limit to see more)")
        
        # 如果没有任何匹配结果，添加稳定占位符
        if not matches:
            result_lines.append("(no matches)")

        # 如果有被跳过的大文件，添加提示
        if skipped_large_files:
            result_lines.append(f"grep_repo skipped {len(skipped_large_files)} large file{'s' if len(skipped_large_files) != 1 else ''}")
            for file_path, size in skipped_large_files[:10]:
                result_lines.append(f"  {file_path} ({size} bytes)")
            if len(skipped_large_files) > 10:
                result_lines.append(f"  ... and {len(skipped_large_files) - 10} more")
        
        return "\n".join(result_lines)

    def read_file(self, relative_path: str, offset: int | None, limit: int | None) -> str:
        """Read a file or bounded large-file slice without loading huge traces."""
        profile = self.require_tool("read_file")
        target = self._resolve_path(relative_path, profile, mode="read")
        display_path = self._display_path(target)
        size_bytes = target.stat().st_size
        if size_bytes > _TOOL_FULL_READ_MAX_BYTES and limit is None:
            return (
                f"{display_path} file_size={size_bytes} bytes exceeds "
                f"read_file full-read limit ({_TOOL_FULL_READ_MAX_BYTES} bytes). "
                "Pass offset and limit to read a bounded slice."
            )
        if size_bytes > _TOOL_FULL_READ_MAX_BYTES:
            result = self._read_large_file_slice(target, offset, limit)
            self.record_control_artifact_read(display_path)
            return result
        lines = target.read_text(encoding="utf-8").splitlines()
        start = 1 if offset is None else offset
        end = len(lines) if limit is None else min(len(lines), start + limit - 1)
        selected = lines[start - 1 : end]
        numbered = "\n".join(f"{idx}: {line}" for idx, line in enumerate(selected, start=start))
        header = f"{display_path} lines {start}-{end} / {len(lines)}"
        self.record_control_artifact_read(display_path)
        return f"{header}\n{numbered}" if numbered else header

    def read_artifact(
        self,
        artifact_ref: str,
        offset: int | None,
        limit: int | None,
    ) -> str:
        """Read a ledger artifact by stable prompt ref without exposing its path."""
        self.require_tool("read_artifact")
        if self.artifact_ledger is None:
            raise RuntimeError("Artifact ledger is not configured")
        artifact = self.artifact_ledger.lookup_ref(artifact_ref)
        target = Path(artifact.path)
        if not target.exists():
            raise FileNotFoundError(f"Artifact payload is missing: {artifact_ref}")
        header = (
            f"artifact_ref={artifact.prompt_ref()} kind={artifact.kind} "
            f"tool={artifact.tool_name or '-'} sha256={artifact.sha256}"
        )
        size_bytes = target.stat().st_size
        bounded_limit = (
            _ARTIFACT_READ_DEFAULT_LIMIT
            if limit is None
            else min(limit, _ARTIFACT_READ_MAX_LIMIT)
        )
        slice_text = self._read_large_file_slice(
            target,
            offset,
            bounded_limit,
            display_name="artifact payload",
            max_output_bytes=_ARTIFACT_READ_MAX_OUTPUT_BYTES,
        )
        return (
            f"{header}\n"
            f"file_size={size_bytes} bytes\n"
            f"bounded_read=true default_limit={_ARTIFACT_READ_DEFAULT_LIMIT} "
            f"max_limit={_ARTIFACT_READ_MAX_LIMIT} "
            f"max_output_bytes={_ARTIFACT_READ_MAX_OUTPUT_BYTES}\n"
            f"{slice_text}"
        )

    def _read_large_file_slice(
        self,
        target: Path,
        offset: int | None,
        limit: int | None,
        display_name: str | None = None,
        max_output_bytes: int | None = None,
    ) -> str:
        """Read a bounded slice from a large file without loading it all."""
        start = max(1, 1 if offset is None else offset)
        bounded_limit = max(1, 200 if limit is None else limit)
        end = start + bounded_limit - 1
        selected: list[str] = []
        emitted_bytes = 0
        truncated_by_bytes = False
        with target.open("r", encoding="utf-8", errors="replace") as handle:
            for line_number, line in enumerate(handle, start=1):
                if line_number < start:
                    continue
                if line_number > end:
                    break
                rendered_line = line.rstrip("\r\n")
                if max_output_bytes is not None:
                    line_bytes = len(rendered_line.encode("utf-8"))
                    if emitted_bytes + line_bytes > max_output_bytes:
                        remaining = max(0, max_output_bytes - emitted_bytes)
                        rendered_line = (
                            rendered_line.encode("utf-8")[:remaining]
                            .decode("utf-8", errors="ignore")
                            + " ... [line truncated by read budget]"
                        )
                        truncated_by_bytes = True
                        selected.append(rendered_line)
                        break
                    emitted_bytes += line_bytes
                selected.append(rendered_line)
        actual_end = start + len(selected) - 1
        display = display_name or self._display_path(target)
        header = (
            f"{display} lines {start}-{actual_end} / unknown "
            f"(streamed large file; file_size={target.stat().st_size} bytes)"
        )
        if max_output_bytes is not None:
            header += f" output_byte_limit={max_output_bytes}"
        numbered = "\n".join(
            f"{idx}: {line}" for idx, line in enumerate(selected, start=start)
        )
        if truncated_by_bytes:
            numbered = (
                f"{numbered}\n"
                "[read output truncated by byte budget; pass a narrower offset/limit]"
            )
        return f"{header}\n{numbered}" if numbered else header

    def edit_file(self, relative_path: str, old_string: str, new_string: str, replace_all: bool) -> str:
        profile = self.require_tool("edit_file")
        self._ensure_required_control_artifacts_acknowledged("edit_file")
        self._enforce_single_file_rebuild_mode("edit_file", relative_path)
        self._enforce_no_write_after_current_validation("edit_file", relative_path)
        target = self._resolve_path(relative_path, profile, mode="edit")
        original = target.read_text(encoding="utf-8")
        count = original.count(old_string)
        if old_string == new_string:
            raise RuntimeError("old_string and new_string must differ")
        if count == 0:
            raise RuntimeError(f"old_string not found in {relative_path}")
        if count > 1 and not replace_all:
            raise RuntimeError(
                f"old_string matched {count} times in {relative_path}. "
                "Use replace_all=true or provide a more specific snippet."
            )
        updated = original.replace(old_string, new_string) if replace_all else original.replace(old_string, new_string, 1)
        target.write_text(updated, encoding="utf-8")
        self._record_write("edit_file", target)
        self._sync_todo_state(target, updated)
        replaced = count if replace_all else 1
        return f"Updated {relative_path} with {replaced} replacement(s)"

    def write_file(self, relative_path: str, content: str) -> str:
        profile = self.require_tool("write_file")
        self._ensure_required_control_artifacts_acknowledged("write_file")
        self._enforce_single_file_rebuild_mode("write_file", relative_path)
        self._enforce_no_write_after_current_validation("write_file", relative_path)
        if not content.strip():
            raise RuntimeError(f"write_file requires non-empty content for {relative_path}")
        target = self._resolve_path(relative_path, profile, mode="write", allow_missing=True)
        exists = target.exists()
        if exists and not profile.allow_write_overwrite:
            raise RuntimeError(f"write_file cannot overwrite {relative_path} in profile {profile.name}")
        if not exists and not profile.allow_write_create:
            raise RuntimeError(f"write_file cannot create {relative_path} in profile {profile.name}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        self._record_write("write_file", target)
        self._sync_todo_state(target, content)
        action = "Updated" if exists else "Created"
        return f"{action} {relative_path}"

    def record_control_artifact_read(self, relative_path: str) -> None:
        if relative_path in TRACKED_CONTROL_ARTIFACTS:
            with self._state_lock:
                self._state.control_artifacts_read.add(relative_path)

    def _ensure_required_control_artifacts_present(self) -> None:
        """Require each declared control artifact to exist before the stage starts."""
        if not self._state.required_control_artifacts:
            return None
        ensure_required_control_artifacts_present(
            self.workspace_root,
            self._state.required_control_artifacts,
            stage=self._state.prompt_descriptor,
        )
        return None

    def _ensure_required_control_artifacts_acknowledged(self, action: str) -> None:
        """Require each declared control artifact to be read or injected before action."""
        if not self._state.required_control_artifacts:
            return None
        ensure_required_control_artifacts_acknowledged(
            self._state.required_control_artifacts,
            read_artifacts=set(self._state.control_artifacts_read),
            injected_artifacts=self._state.control_artifacts_injected,
            action=action,
            stage=self._state.prompt_descriptor,
        )
        return None

    def record_observation(self, tool_name: str) -> None:
        """Record an observation tool call and enforce stage policy.

        Returns None on success, raises RecoverableStagePolicyError for recoverable
        violations (returned to model), or FatalStagePolicyError for fatal violations.
        """
        profile = self.require_tool(tool_name)

        # Fatal: workspace escape or internal state corruption
        # These are caught in _resolve_path and re-raised as fatal

        self._enforce_write_after_failure(tool_name)

        with self._state_lock:
            self._state.tool_counts[tool_name] += 1
            self._state.consecutive_observation_count += 1
            current_count = self._state.consecutive_observation_count

        # Check soft limit (first violation is recoverable)
        soft_limit = profile.max_consecutive_observations
        hard_limit = profile.hard_consecutive_observations

        if hard_limit is not None and current_count > hard_limit:
            raise self._fatal_error(
                code="OBSERVATION_LIMIT_FATAL",
                category="stage_policy",
                message=(
                    f"Observation count ({current_count}) "
                    f"exceeds hard upper bound ({hard_limit}) in profile {profile.name}."
                ),
                relevant_files=self._profile_scope_files(),
            )

        if current_count > soft_limit:
            excess = current_count - soft_limit
            should_warn = excess == 1 or excess % 4 == 0
            if should_warn:
                with self._state_lock:
                    self._state.soft_observation_limit_exceeded = True
                raise self._recoverable_error(
                    code="OBSERVATION_LIMIT",
                    category="stage_policy",
                    message=self._build_observation_limit_message(
                        current_count=current_count,
                        hard_limit=hard_limit,
                    ),
                    relevant_files=self._profile_scope_files(),
                    allowed_next_actions=tuple(self._get_available_progress_tools()),
                    recommended_next_action=_render_runtime_policy(
                        "policy_observation_limit_next_action"
                    ),
                )

        return None
    
    def _get_available_write_tools(self) -> list[str]:
        """Get list of write-capable tools available in current profile."""
        profile = self.profiles[self._active_profile_name]
        if (
            self._state.single_file_rebuild_target is not None
            and "write_file" in profile.tool_names
        ):
            return ["write_file"]
        write_tools = []
        for tool_name in profile.tool_names:
            if tool_name in ("edit_file", "write_file", "apply_patch"):
                write_tools.append(tool_name)
        return write_tools if write_tools else [
            _render_runtime_policy("policy_no_write_tools_available")
        ]

    def _get_available_progress_tools(self) -> list[str]:
        return self._get_available_write_tools()

    def _profile_scope_files(self) -> tuple[str, ...]:
        profile = self.profiles[self._active_profile_name]
        scopes = profile.edit_globs + profile.create_globs + profile.write_globs
        if not scopes:
            return ()
        return tuple(dict.fromkeys(scopes))

    def _build_stage_specific_observation_guidance(self, profile_name: str) -> str:
        asset_suffixes = {
            "todo_plan": "policy_observation_guidance_todo_plan",
            "storage_plan": "policy_observation_guidance_storage_plan",
            "finish_skeleton": "policy_observation_guidance_finish_skeleton",
            "implement_queries_writeonly": "policy_observation_guidance_implement_queries_writeonly",
            "correctness_queries_writeonly": "policy_observation_guidance_correctness_queries_writeonly",
            "correctness_foundation": "policy_observation_guidance_correctness_foundation",
        }
        asset_suffix = asset_suffixes.get(profile_name)
        if asset_suffix is not None:
            return _render_runtime_policy(asset_suffix)
        return ""

    def _build_observation_limit_message(self, current_count: int, hard_limit: int | None) -> str:
        profile = self.profiles[self._active_profile_name]
        progress_tools = self._get_available_progress_tools()
        editable_scope = (
            ", ".join(profile.edit_globs)
            if profile.edit_globs
            else _render_runtime_policy("policy_no_editable_files")
        )
        hard_limit_text = (
            _render_runtime_policy("policy_no_hard_limit")
            if hard_limit is None
            else str(hard_limit)
        )
        stage_guidance = self._build_stage_specific_observation_guidance(profile.name)
        create_scope = ""
        if profile.create_globs:
            create_scope = _render_runtime_policy(
                "policy_creatable_scope_suffix",
                create_scope=", ".join(profile.create_globs),
            )
        return _render_runtime_policy(
            "policy_observation_limit_message",
            current_count=current_count,
            profile_name=profile.name,
            hard_limit_text=hard_limit_text,
            progress_tools=", ".join(progress_tools),
            editable_scope=editable_scope,
            create_scope=create_scope,
            stage_guidance=stage_guidance,
        )

    def _enforce_write_after_failure(self, tool_name: str) -> None:
        """After compile/run failure, only allow true write tools until a write succeeds."""
        if not self._state.require_write_after_failure:
            return None
        if tool_name in ("edit_file", "write_file", "apply_patch"):
            return None
        write_tools = self._get_available_write_tools()
        raise self._recoverable_error(
            code="MUST_WRITE_FIRST",
            category="stage_policy",
            message=_render_runtime_policy(
                "policy_must_write_first_message",
                write_tools=", ".join(write_tools),
            ),
            relevant_files=self._profile_scope_files(),
            allowed_next_actions=tuple(write_tools),
            recommended_next_action=_render_runtime_policy(
                "policy_must_write_first_next_action"
            ),
        )

    def _derive_primary_query_cpp(self) -> str | None:
        descriptor = self._state.prompt_descriptor or ""
        match = _PRIMARY_QUERY_DESCRIPTOR_RE.match(descriptor)
        if match is None:
            return None
        qid = match.group("short") or match.group("long")
        if qid is None:
            return None
        return f"query_q{qid}.cpp"

    def _derive_primary_query_id(self) -> str | None:
        descriptor = self._state.prompt_descriptor or ""
        match = _PRIMARY_QUERY_DESCRIPTOR_RE.match(descriptor)
        if match is None:
            return None
        qid = match.group("short") or match.group("long")
        if qid is None:
            return None
        return qid

    def validate_run_request(self, args_json: str) -> None:
        """Validate run-tool args against stage-local query scope."""
        profile = self.require_tool("run")
        self._ensure_required_control_artifacts_acknowledged("run")
        if profile.name == "optimization_instrumentation":
            allowed_query_ids = list(
                self._state.active_unit_query_ids or self._state.active_query_ids
            )
            if not allowed_query_ids:
                return None
            args = load_function_tool_args(args_json)
            query_ids = args.get("query_id")
            normalized = [] if query_ids is None else [str(item) for item in query_ids]
            if len(normalized) == len(allowed_query_ids) and set(normalized) == set(allowed_query_ids):
                return None
            raise self._recoverable_error(
                code="RUN_QUERY_BATCH_SCOPE_DENIED",
                category="stage_policy",
                message=_render_runtime_policy(
                    "policy_run_query_batch_scope_denied_message",
                    stage_descriptor=self._state.prompt_descriptor,
                    allowed_query_ids=allowed_query_ids,
                    received_query_ids=repr(query_ids),
                ),
                relevant_files=tuple(f"query_q{qid}.cpp" for qid in allowed_query_ids),
                allowed_next_actions=("run",),
                recommended_next_action=_render_runtime_policy(
                    "policy_run_query_batch_scope_denied_next_action",
                    allowed_query_ids=allowed_query_ids,
                ),
            )
        if profile.name not in _PRIMARY_QUERY_VALIDATION_PROFILES:
            return None
        allowed_qid = self._derive_primary_query_id()
        if allowed_qid is None:
            return None
        args = load_function_tool_args(args_json)
        query_ids = args.get("query_id")
        normalized = [] if query_ids is None else [str(item) for item in query_ids]
        if normalized == [allowed_qid]:
            return None
        raise self._recoverable_error(
            code="RUN_QUERY_SCOPE_DENIED",
            category="stage_policy",
            message=_render_runtime_policy(
                "policy_run_query_scope_denied_message",
                stage_descriptor=self._state.prompt_descriptor,
                allowed_qid=allowed_qid,
                received_query_ids=repr(query_ids),
            ),
            relevant_files=(f"query_q{allowed_qid}.cpp",),
            allowed_next_actions=("run",),
            recommended_next_action=_render_runtime_policy(
                "policy_run_query_scope_denied_next_action",
                allowed_qid=allowed_qid,
            ),
        )

    def _compile_failure_requires_rebuild(self, output: str) -> bool:
        if self._active_profile_name not in _PRIMARY_QUERY_VALIDATION_PROFILES:
            return False
        if self._state.consecutive_compile_failures < 3:
            return False
        if self._derive_primary_query_cpp() is None:
            return False
        return any(pattern.search(output) for pattern in _CORRUPTED_QUERY_FILE_PATTERNS)

    def _enforce_single_file_rebuild_mode(
        self,
        tool_name: str,
        relative_path: str,
    ) -> None:
        target = self._state.single_file_rebuild_target
        if target is None:
            return None
        normalized_path = Path(relative_path.strip() or ".").as_posix()
        reason = (
            self._state.single_file_rebuild_reason
            or _render_runtime_policy(
                "policy_single_file_rebuild_default_reason",
                target=target,
            )
        )
        if tool_name == "write_file" and normalized_path == target:
            return None
        code = "REBUILD_TARGET_ONLY" if tool_name == "write_file" else "SINGLE_FILE_REBUILD_MODE"
        raise self._recoverable_error(
            code=code,
            category="stage_policy",
            message=reason,
            relevant_files=(target,),
            allowed_next_actions=("write_file",),
            recommended_next_action=_render_runtime_policy(
                "policy_single_file_rebuild_next_action",
                target=target,
            ),
        )

    def record_execution(self, tool_name: str, output: str, success: bool) -> None:
        """Record compile/run outcomes and activate rebuild mode on structural corruption."""
        self.require_tool(tool_name)
        summary = summarize_tool_output(output)
        with self._state_lock:
            self._state.tool_counts[tool_name] += 1
            self._state.consecutive_observation_count = 0
            if tool_name == "compile":
                self._state.last_compile_summary = summary
                self._state.last_compile_succeeded = success
                if not success:
                    self._state.last_failure_kind = "compile"
                    self._state.consecutive_compile_failures += 1
                elif self._state.last_failure_kind == "compile":
                    self._state.last_failure_kind = None
                if success:
                    self._state.consecutive_compile_failures = 0
                    self._state.single_file_rebuild_target = None
                    self._state.single_file_rebuild_reason = None
            elif tool_name == "run":
                self._state.last_run_summary = summary
                self._state.last_run_succeeded = success
                validation_summary = extract_validation_summary(output)
                current_passed = infer_validation_passed(
                    output=output,
                    success=success,
                    validation_summary=validation_summary,
                )
                preserve_validation_state = (
                    self._should_preserve_validation_state_after_unclassified_run(
                        success=success,
                        current_passed=current_passed,
                    )
                )
                if not preserve_validation_state:
                    self._state.last_validation_summary = validation_summary
                    self._state.validation_passed = current_passed
                    if current_passed is False:
                        self._state.last_failure_kind = "validation"
                    elif not success:
                        self._state.last_failure_kind = "run"
                    elif self._state.last_failure_kind in {"run", "validation"}:
                        self._state.last_failure_kind = None
            validation_failed = (
                tool_name == "run" and self._state.validation_passed is False
            )
            self._state.require_write_after_failure = not success or validation_failed
            if tool_name == "compile" and self._compile_failure_requires_rebuild(output):
                target = self._derive_primary_query_cpp()
                if target is not None:
                    self._state.single_file_rebuild_target = target
                    self._state.single_file_rebuild_reason = _render_runtime_policy(
                        "policy_query_file_structural_corruption_reason",
                        target=target,
                    )
        self._enforce_stalled_execution(tool_name, summary, success)
        return None

    def _should_preserve_validation_state_after_unclassified_run(
        self,
        success: bool,
        current_passed: bool | None,
    ) -> bool:
        if self._active_profile_name not in _PRIMARY_QUERY_VALIDATION_PROFILES:
            return False
        if current_passed is not None:
            return False
        if not success:
            return False
        return self._state.validation_passed is not None

    def validate_apply_patch(self, args_json: str) -> Path:
        """Validate apply_patch request against stage policy before execution.
        
        Args:
            args_json: JSON string with type, path, diff
            
        Returns:
            Resolved target path for a valid update_file request.

        Raises:
            RecoverableStagePolicyError: If request violates stage policy
        """
        args = load_function_tool_args(args_json)
        profile = self.require_tool("apply_patch")
        self._ensure_required_control_artifacts_acknowledged("apply_patch")
        op_type = args.get("type")
        path = args.get("path", "")
        self._enforce_single_file_rebuild_mode("apply_patch", path)
        self._enforce_no_write_after_current_validation("apply_patch", path)

        allowed_ops = ("update_file",)
        if profile.create_globs:
            allowed_ops = ("create_file", "update_file")
        if op_type not in allowed_ops:
            raise self._recoverable_error(
                code="PATCH_OP_DENIED",
                category="stage_policy",
                message=_render_runtime_policy(
                    "policy_patch_op_denied_message",
                    op_type=op_type,
                    allowed_ops=", ".join(allowed_ops),
                ),
                allowed_next_actions=allowed_ops,
                relevant_files=self._profile_scope_files(),
                recommended_next_action=_render_runtime_policy(
                    "policy_patch_op_denied_next_action"
                ),
            )

        if op_type == "create_file":
            target = self._resolve_path(path, profile, mode="create", allow_missing=True)
            if target.exists():
                relative_target = target.relative_to(self.workspace_root).as_posix()
                raise self._recoverable_error(
                    code="PATCH_CREATE_EXISTS",
                    category="stage_policy",
                    message=_render_runtime_policy(
                        "policy_patch_create_exists_message",
                        relative_path=relative_target,
                    ),
                    relevant_files=(relative_target,),
                    allowed_next_actions=("update_file",),
                    recommended_next_action=_render_runtime_policy(
                        "policy_patch_create_exists_next_action"
                    ),
                )
            return target

        target = self._resolve_path(path, profile, mode="edit")
        return target

    def _enforce_no_write_after_current_validation(
        self,
        tool_name: str,
        relative_path: str,
    ) -> None:
        if self._active_profile_name not in _PRIMARY_QUERY_VALIDATION_PROFILES:
            return None
        if self._state.validation_passed is not True:
            return None
        if self._state.run_write_revision != self._state.write_revision:
            return None
        normalized_path = Path(relative_path.strip() or ".").as_posix()
        raise self._recoverable_error(
            code="VALIDATION_ALREADY_PASSED_NO_MORE_WRITES",
            category="stage_policy",
            message=(
                "Validation has already passed for the latest file revision. "
                f"Do not use {tool_name} on {normalized_path}; finish the stage now."
            ),
            relevant_files=(normalized_path,),
            allowed_next_actions=(),
            recommended_next_action="Finish the stage now without more file edits.",
        )

    def _record_write(self, tool_name: str, target: Path) -> None:
        self.require_tool(tool_name)
        with self._state_lock:
            self._state.tool_counts[tool_name] += 1
            self._state.consecutive_observation_count = 0
            self._state.write_revision += 1
            self._state.require_write_after_failure = False
            self._state.validation_passed = None
            self._state.written_files.add(target.relative_to(self.workspace_root).as_posix())
        return None

    def _enforce_stalled_execution(
        self,
        tool_name: str,
        summary: str | None,
        success: bool,
    ) -> None:
        signature = f"{tool_name}|{int(success)}|{summary or ''}"
        repeat_count = self._update_execution_repeat_count(tool_name, signature)
        if repeat_count < get_max_stalled_executions():
            return None
        raise self._recoverable_error(
            code="STALLED_STAGE",
            category="stage_policy",
            message=_render_runtime_policy(
                "policy_stalled_execution_message",
                tool_name=tool_name,
                repeat_count=repeat_count,
                stage_name=self._active_profile_name,
            ),
            relevant_files=self._profile_scope_files(),
            allowed_next_actions=tuple(self._get_available_write_tools()),
            recommended_next_action=_render_runtime_policy(
                "policy_stalled_execution_next_action"
            ),
        )

    def _update_execution_repeat_count(self, tool_name: str, signature: str) -> int:
        signature_attr = f"last_{tool_name}_signature"
        count_attr = f"repeated_{tool_name}_count"
        revision_attr = f"{tool_name}_write_revision"
        last_signature = getattr(self._state, signature_attr)
        last_revision = getattr(self._state, revision_attr)
        if last_signature == signature and last_revision == self._state.write_revision:
            count = getattr(self._state, count_attr) + 1
        else:
            count = 1
        setattr(self._state, signature_attr, signature)
        setattr(self._state, count_attr, count)
        setattr(self._state, revision_attr, self._state.write_revision)
        return count

    def _load_todo_state(self) -> Optional[TodoState]:
        todo_path = self.workspace_root / "TODO.md"
        return TodoState.from_file(todo_path)

    def _sync_todo_state(self, target: Path, content: str) -> None:
        relative = target.relative_to(self.workspace_root).as_posix()
        if relative != "TODO.md":
            return None
        self._state.todo_current = TodoState.from_text(content)
        return None

    def _resolve_path(
        self,
        relative_path: str,
        profile: ToolProfile,
        mode: str,
        allow_missing: bool = False,
    ) -> Path:
        raw = relative_path.strip()
        if raw in ("", ".", "./", "/"):
            target = self.workspace_root
        else:
            candidate = Path(raw)
            if candidate.is_absolute():
                absolute_target = candidate.resolve()
                try:
                    absolute_target.relative_to(self.workspace_root)
                    target = absolute_target
                except ValueError:
                    if mode == "read" and self._is_under_extra_read_root(absolute_target):
                        target = absolute_target
                    else:
                        target = (self.workspace_root / raw.lstrip("/")).resolve()
            else:
                target = (self.workspace_root / candidate).resolve()
        try:
            relative = target.relative_to(self.workspace_root).as_posix()
            target_in_workspace = True
        except ValueError as exc:
            if mode == "read" and self._is_under_extra_read_root(target):
                relative = target.as_posix()
                target_in_workspace = False
            else:
                if mode == "read":
                    raise self._recoverable_error(
                        code="PATH_OUTSIDE_WORKSPACE",
                        category="stage_policy",
                        message=_render_runtime_policy(
                            "policy_path_outside_workspace_message",
                            relative_path=relative_path,
                        ),
                        relevant_files=self._profile_scope_files(),
                        allowed_next_actions=profile.read_globs,
                        recommended_next_action=_render_runtime_policy(
                            "policy_path_outside_workspace_next_action"
                        ),
                    ) from exc
                # Fatal: write/edit workspace escape remains a security violation
                raise self._fatal_error(
                    code="PATH_OUTSIDE_WORKSPACE",
                    category="stage_policy",
                    message=f"Path outside workspace: {relative_path}",
                ) from exc
        if mode == "read" and target_in_workspace and not profile.allows_read(relative):
            raise self._recoverable_error(
                code="READ_DENIED",
                category="stage_policy",
                message=_render_runtime_policy(
                    "policy_read_denied_message",
                    relative_path=relative,
                    profile_name=profile.name,
                ),
                relevant_files=(relative,),
                allowed_next_actions=profile.read_globs,
                recommended_next_action=_render_runtime_policy("policy_read_denied_next_action"),
            )
        if mode == "edit" and not profile.allows_edit(relative):
            raise self._recoverable_error(
                code="EDIT_DENIED",
                category="stage_policy",
                message=_render_runtime_policy(
                    "policy_edit_denied_message",
                    relative_path=relative,
                    profile_name=profile.name,
                ),
                relevant_files=(relative,),
                allowed_next_actions=profile.edit_globs,
                recommended_next_action=_render_runtime_policy("policy_edit_denied_next_action"),
            )
        if mode == "create" and not profile.allows_create(relative):
            raise self._recoverable_error(
                code="CREATE_DENIED",
                category="stage_policy",
                message=_render_runtime_policy(
                    "policy_create_denied_message",
                    relative_path=relative,
                    profile_name=profile.name,
                ),
                relevant_files=(relative,),
                allowed_next_actions=profile.create_globs,
                recommended_next_action=_render_runtime_policy("policy_create_denied_next_action"),
            )
        if mode == "write" and not profile.allows_write(relative):
            raise self._recoverable_error(
                code="WRITE_DENIED",
                category="stage_policy",
                message=_render_runtime_policy(
                    "policy_write_denied_message",
                    relative_path=relative,
                    profile_name=profile.name,
                ),
                relevant_files=(relative,),
                allowed_next_actions=profile.write_globs,
                recommended_next_action=_render_runtime_policy("policy_write_denied_next_action"),
            )
        self._enforce_dynamic_scope(relative=relative, profile=profile, mode=mode)
        if not allow_missing and not target.exists():
            raise self._recoverable_error(
                code="PATH_NOT_FOUND",
                category="stage_policy",
                message=_render_runtime_policy(
                    "policy_path_not_found_message",
                    relative_path=relative,
                ),
                relevant_files=(relative,),
                recommended_next_action=_render_runtime_policy("policy_path_not_found_next_action"),
            )
        return target

    def _enforce_dynamic_scope(self, *, relative: str, profile: ToolProfile, mode: str) -> None:
        """Apply prompt-scoped path narrowing on top of static profile globs."""
        if profile.name != "optimization_instrumentation":
            return None
        if mode not in {"edit", "create", "write"}:
            return None
        allowed_patterns = self._instrumentation_allowed_patterns()
        if any(Path(relative).match(pattern) for pattern in allowed_patterns):
            return None
        raise self._recoverable_error(
            code="INSTRUMENTATION_SCOPE_DENIED",
            category="stage_policy",
            message=_render_runtime_policy(
                "policy_instrumentation_scope_denied_message",
                relative_path=relative,
                allowed_patterns=", ".join(allowed_patterns),
            ),
            relevant_files=(relative,),
            allowed_next_actions=allowed_patterns,
            recommended_next_action=_render_runtime_policy(
                "policy_instrumentation_scope_denied_next_action"
            ),
        )

    def _instrumentation_allowed_patterns(self) -> tuple[str, ...]:
        patterns = [
            "loader_impl.hpp",
            "loader_impl.cpp",
            "builder_impl.hpp",
            "builder_impl.cpp",
            "query_impl.hpp",
            "query_impl.cpp",
            "query_shared_*.cpp",
            "query_shared_*.hpp",
        ]
        for query_id in self._state.active_query_ids:
            patterns.append(f"query_q{query_id}.cpp")
            patterns.append(f"query_q{query_id}.hpp")
        return tuple(patterns)

    def build_execution_error(
        self,
        *,
        code: str,
        category: str,
        message: str,
        recommended_next_action: str,
    ) -> str:
        envelope = ErrorEnvelope(
            code=code,
            category=category,
            stage=self._active_profile_name,
            message=message,
            recoverable=True,
            relevant_files=self._profile_scope_files(),
            allowed_next_actions=tuple(self._get_available_write_tools()),
            recommended_next_action=recommended_next_action,
        )
        return str(envelope)

    def _recoverable_error(
        self,
        *,
        code: str,
        category: str,
        message: str,
        relevant_files: tuple[str, ...] = (),
        allowed_next_actions: tuple[str, ...] = (),
        recommended_next_action: str | None = None,
    ) -> RecoverableStagePolicyError:
        return RecoverableStagePolicyError(
            str(
                ErrorEnvelope(
                    code=code,
                    category=category,
                    stage=self._active_profile_name,
                    message=message,
                    recoverable=True,
                    relevant_files=relevant_files,
                    allowed_next_actions=allowed_next_actions,
                    recommended_next_action=recommended_next_action,
                )
            )
        )

    def _fatal_error(
        self,
        *,
        code: str,
        category: str,
        message: str,
        relevant_files: tuple[str, ...] = (),
        allowed_next_actions: tuple[str, ...] = (),
        recommended_next_action: str | None = None,
    ) -> FatalStagePolicyError:
        return FatalStagePolicyError(
            str(
                ErrorEnvelope(
                    code=code,
                    category=category,
                    stage=self._active_profile_name,
                    message=message,
                    recoverable=False,
                    relevant_files=relevant_files,
                    allowed_next_actions=allowed_next_actions,
                    recommended_next_action=recommended_next_action,
                )
            )
        )


def _looks_like_failure(output: str) -> bool:
    normalized = output.lower()
    markers = (
        "error",
        "failed",
        "not correct",
        "compile_error",
        "traceback",
        "exception",
    )
    return any(marker in normalized for marker in markers)


def _build_function_tool(
    name: str,
    description: str,
    schema: dict[str, Any],
    impl,
) -> FunctionTool:
    return FunctionTool(
        name=name,
        description=description,
        params_json_schema=schema,
        on_invoke_tool=impl,
    )


def _wrap_parse_error(exc: Exception) -> str:
    return f"Error: Invalid JSON format. {str(exc)}."


def _mark_evidence(output: str) -> str:
    stripped = output.strip()
    if not stripped:
        return "[Evidence]"
    if stripped.startswith("[Evidence]"):
        return stripped
    return f"[Evidence]\n{stripped}"


def _truncate_run_evidence(output: str, limit: int = _RUN_EVIDENCE_CHAR_LIMIT) -> str:
    """Return a bounded head/tail fallback when no artifact ledger is available."""
    stripped = output.strip()
    if len(stripped) <= limit:
        return stripped
    keep = max(512, (limit - 64) // 2)
    omitted = max(0, len(stripped) - (2 * keep))
    return (
        f"{stripped[:keep]}\n"
        f"... [{omitted} chars truncated] ...\n"
        f"{stripped[-keep:]}"
    )


def build_tpch_monetdb_agent_tools(
    workspace_path: Path,
    cache_path: Path,
    compile_tool: FunctionTool,
    run_tool: FunctionTool,
    git_snapshotter: Any = None,
    wandb_metrics_hook: Any = None,
    apply_patch_tool: FunctionTool | None = None,
    extra_read_roots: tuple[Path, ...] = (),
    artifact_ledger: ArtifactLedger | None = None,
) -> ToolBundle:
    """Build the TPC-H MonetDB single-agent tool set and its profile-specific views."""
    resolved_artifact_ledger = artifact_ledger or ArtifactLedger(
        cache_path / "context_artifacts"
    )
    runtime = StageToolRuntime(
        workspace_path,
        extra_read_roots=extra_read_roots,
        artifact_ledger=resolved_artifact_ledger,
    )

    async def read_file_invoke(_ctx: RunContextWrapper[Any], args_json: str) -> str:
        try:
            parsed = load_function_tool_args(args_json)
            args = ReadFileArgs.model_validate(parsed)
            async with runtime.tool_guard("read_file"):
                runtime.record_observation("read_file")
                result = runtime.read_file(args.file_path, args.offset, args.limit)
                evidence = runtime.prepare_tool_evidence(
                    tool_name="read_file",
                    output=result,
                    success=True,
                    inline_limit=_READ_INLINE_CHAR_LIMIT,
                    kind="read_file_output",
                )
                return _mark_evidence(evidence)
        except json.JSONDecodeError as exc:
            return _wrap_parse_error(exc)
        except RecoverableStagePolicyError as exc:
            return str(exc)
        except FatalStagePolicyError:
            raise
        except Exception as exc:
            return f"Error reading file: {str(exc)}"

    async def read_artifact_invoke(_ctx: RunContextWrapper[Any], args_json: str) -> str:
        try:
            parsed = load_function_tool_args(args_json)
            args = ReadArtifactArgs.model_validate(parsed)
            async with runtime.tool_guard("read_artifact"):
                runtime.record_observation("read_artifact")
                return _mark_evidence(
                    runtime.read_artifact(args.artifact_ref, args.offset, args.limit)
                )
        except json.JSONDecodeError as exc:
            return _wrap_parse_error(exc)
        except RecoverableStagePolicyError as exc:
            return str(exc)
        except FatalStagePolicyError:
            raise
        except Exception as exc:
            return f"Error reading artifact: {str(exc)}"

    async def list_files_invoke(_ctx: RunContextWrapper[Any], args_json: str) -> str:
        try:
            parsed = load_function_tool_args(args_json)
            args = ListFilesArgs.model_validate(parsed)
            async with runtime.tool_guard("list_files"):
                runtime.record_observation("list_files")
                return _mark_evidence(
                    runtime.list_directory(args.path, args.pattern, args.limit)
                )
        except json.JSONDecodeError as exc:
            return _wrap_parse_error(exc)
        except RecoverableStagePolicyError as exc:
            return str(exc)
        except FatalStagePolicyError:
            raise
        except Exception as exc:
            return f"Error listing files: {str(exc)}"

    async def grep_repo_invoke(_ctx: RunContextWrapper[Any], args_json: str) -> str:
        try:
            parsed = load_function_tool_args(args_json)
            args = GrepRepoArgs.model_validate(parsed)
            async with runtime.tool_guard("grep_repo"):
                runtime.record_observation("grep_repo")
                result = runtime.grep_repo(args.pattern, args.path, args.glob, args.limit)
                evidence = runtime.prepare_tool_evidence(
                    tool_name="grep_repo",
                    output=result,
                    success=True,
                    inline_limit=_GREP_INLINE_CHAR_LIMIT,
                    kind="grep_repo_output",
                )
                return _mark_evidence(evidence)
        except json.JSONDecodeError as exc:
            return _wrap_parse_error(exc)
        except RecoverableStagePolicyError as exc:
            return str(exc)
        except FatalStagePolicyError:
            raise
        except Exception as exc:
            return f"Error grepping repo: {str(exc)}"

    async def edit_file_invoke(_ctx: RunContextWrapper[Any], args_json: str) -> str:
        try:
            parsed = load_function_tool_args(args_json)
            args = EditFileArgs.model_validate(parsed)
            async with runtime.tool_guard("edit_file"):
                return runtime.edit_file(
                    args.file_path,
                    args.old_string,
                    args.new_string,
                    args.replace_all,
                )
        except json.JSONDecodeError as exc:
            return _wrap_parse_error(exc)
        except RecoverableStagePolicyError as exc:
            return str(exc)
        except FatalStagePolicyError:
            raise
        except Exception as exc:
            return f"Error editing file: {str(exc)}"

    async def write_file_invoke(_ctx: RunContextWrapper[Any], args_json: str) -> str:
        try:
            parsed = load_function_tool_args(args_json)
            args = WriteFileArgs.model_validate(parsed)
            async with runtime.tool_guard("write_file"):
                return runtime.write_file(args.file_path, args.content)
        except json.JSONDecodeError as exc:
            return _wrap_parse_error(exc)
        except RecoverableStagePolicyError as exc:
            return str(exc)
        except FatalStagePolicyError:
            raise
        except Exception as exc:
            return f"Error writing file: {str(exc)}"

    async def apply_patch_invoke(ctx: RunContextWrapper[Any], args_json: str) -> str:
        if apply_patch_tool is None:
            return "Error: apply_patch tool is not configured"
        try:
            async with runtime.tool_guard("apply_patch"):
                # Pre-validate before execution
                target = runtime.validate_apply_patch(args_json)
                result = await apply_patch_tool.on_invoke_tool(ctx, args_json)
                if not result.lower().startswith("error"):
                    runtime._record_write("apply_patch", target)
                return result
        except RecoverableStagePolicyError as exc:
            return str(exc)
        except FatalStagePolicyError:
            raise
        except Exception as exc:
            return f"Error applying patch: {str(exc)}"

    read_file_tool = _build_function_tool(
        "read_file",
        "Read a workspace file with optional line slicing",
        ReadFileArgs.model_json_schema(),
        read_file_invoke,
    )
    read_artifact_tool = _build_function_tool(
        "read_artifact",
        "Read a large evidence artifact by stable artifact_ref with optional line slicing",
        ReadArtifactArgs.model_json_schema(),
        read_artifact_invoke,
    )
    list_files_tool = _build_function_tool(
        "list_files",
        "List workspace files with optional glob filtering",
        ListFilesArgs.model_json_schema(),
        list_files_invoke,
    )
    grep_repo_tool = _build_function_tool(
        "grep_repo",
        "Search workspace file contents with a regular expression",
        GrepRepoArgs.model_json_schema(),
        grep_repo_invoke,
    )
    edit_file_tool = _build_function_tool(
        "edit_file",
        "Edit an existing file by exact string replacement",
        EditFileArgs.model_json_schema(),
        edit_file_invoke,
    )
    write_file_tool = _build_function_tool(
        "write_file",
        "Create or fully rewrite a file when the current stage allows it",
        WriteFileArgs.model_json_schema(),
        write_file_invoke,
    )

    shell_tool = make_litellm_shell_tool(
        cwd=workspace_path,
        cache_dir=cache_path / "shell",
        git_snapshotter=git_snapshotter,
        wandb_metrics_hook=wandb_metrics_hook,
        read_only=True,
    )
    cpu_info_tool = make_cpu_info_tool(
        cwd=workspace_path,
        cache_dir=cache_path / "cpu_info",
        git_snapshotter=git_snapshotter,
        wandb_metrics_hook=wandb_metrics_hook,
    )

    async def compile_invoke(ctx: RunContextWrapper[Any], args_json: str) -> str:
        try:
            async with runtime.tool_guard("compile"):
                runtime.require_tool("compile")
                runtime._ensure_required_control_artifacts_acknowledged("compile")
                runtime._enforce_write_after_failure("compile")
                result = await compile_tool.on_invoke_tool(ctx, args_json)
                success = result.strip() == "**Compilation successfull**"
                evidence = runtime.prepare_tool_evidence(
                    tool_name="compile",
                    output=result,
                    success=success,
                    inline_limit=_COMPILE_INLINE_CHAR_LIMIT,
                    kind="compile_output",
                )
                runtime.record_execution("compile", evidence, success=success)
                if success:
                    return _mark_evidence(evidence)
                return (
                    runtime.build_execution_error(
                        code="COMPILE_FAILED",
                        category="compile",
                        message=_render_runtime_policy("execution_compile_failed_message"),
                        recommended_next_action=_render_runtime_policy(
                            "execution_compile_failed_next_action"
                        ),
                    )
                    + "\n\n"
                    + _mark_evidence(evidence)
                )
        except RecoverableStagePolicyError as exc:
            return str(exc)
        except FatalStagePolicyError:
            raise
        except Exception as exc:
            return f"Error running compile: {str(exc)}"

    async def run_invoke(ctx: RunContextWrapper[Any], args_json: str) -> str:
        try:
            async with runtime.tool_guard("run"):
                runtime.require_tool("run")
                runtime.validate_run_request(args_json)
                runtime._enforce_write_after_failure("run")
                raw_result = await run_tool.on_invoke_tool(
                    ctx,
                    runtime.run_args_with_stage_metadata(args_json),
                )
                success = not _looks_like_failure(raw_result)
                result = runtime.prepare_tool_evidence(
                    tool_name="run",
                    output=raw_result,
                    success=success,
                    inline_limit=_RUN_INLINE_CHAR_LIMIT,
                    kind="run_output",
                )
                if result == raw_result.strip():
                    result = _truncate_run_evidence(raw_result)
                runtime.record_execution("run", result, success=success)
                if success:
                    return _mark_evidence(result)
                return (
                    runtime.build_execution_error(
                        code="RUN_FAILED",
                        category="run",
                        message=_render_runtime_policy("execution_run_failed_message"),
                        recommended_next_action=_render_runtime_policy(
                            "execution_run_failed_next_action"
                        ),
                    )
                    + "\n\n"
                    + _mark_evidence(result)
                )
        except RecoverableStagePolicyError as exc:
            return str(exc)
        except FatalStagePolicyError:
            raise
        except Exception as exc:
            return f"Error running query: {str(exc)}"

    async def shell_invoke(ctx: RunContextWrapper[Any], args_json: str) -> str:
        try:
            async with runtime.tool_guard("shell"):
                runtime._enforce_write_after_failure("shell")
                runtime.record_observation("shell")
                raw_result = await shell_tool.on_invoke_tool(ctx, args_json)
                evidence = runtime.prepare_tool_evidence(
                    tool_name="shell",
                    output=raw_result,
                    success=not _looks_like_failure(raw_result),
                    inline_limit=_RUN_INLINE_CHAR_LIMIT,
                    kind="shell_output",
                )
                return _mark_evidence(evidence)
        except RecoverableStagePolicyError as exc:
            return str(exc)
        except FatalStagePolicyError:
            raise
        except Exception as exc:
            return f"Error running shell command: {str(exc)}"

    async def cpu_info_invoke(ctx: RunContextWrapper[Any], args_json: str) -> str:
        try:
            async with runtime.tool_guard("cpu_info"):
                runtime._enforce_write_after_failure("cpu_info")
                runtime.record_observation("cpu_info")
                return _mark_evidence(await cpu_info_tool.on_invoke_tool(ctx, args_json))
        except RecoverableStagePolicyError as exc:
            return str(exc)
        except FatalStagePolicyError:
            raise
        except Exception as exc:
            return f"Error collecting cpu info: {str(exc)}"

    compile_wrapper = _build_function_tool(
        "compile",
        compile_tool.description,
        compile_tool.params_json_schema,
        compile_invoke,
    )
    run_wrapper = _build_function_tool(
        "run",
        run_tool.description,
        run_tool.params_json_schema,
        run_invoke,
    )
    shell_wrapper = _build_function_tool(
        "shell",
        shell_tool.description,
        shell_tool.params_json_schema,
        shell_invoke,
    )
    cpu_info_wrapper = _build_function_tool(
        "cpu_info",
        cpu_info_tool.description,
        cpu_info_tool.params_json_schema,
        cpu_info_invoke,
    )
    apply_patch_wrapper = _build_function_tool(
        "apply_patch",
        "Apply a unified diff patch to update an existing file, or create a focused query module when the current stage allows create_file",
        apply_patch_tool.params_json_schema if apply_patch_tool else {"type": "object", "properties": {}},
        apply_patch_invoke,
    )

    tool_map: dict[str, Any] = {
        "read_file": read_file_tool,
        "read_artifact": read_artifact_tool,
        "list_files": list_files_tool,
        "grep_repo": grep_repo_tool,
        "edit_file": edit_file_tool,
        "write_file": write_file_tool,
        "apply_patch": apply_patch_wrapper,
        "shell": shell_wrapper,
        "cpu_info": cpu_info_wrapper,
        "compile": compile_wrapper,
        "run": run_wrapper,
    }
    tools_by_profile = {
        profile_name: runtime.get_tools_for_profile(profile_name, tool_map)
        for profile_name in runtime.profiles
    }
    all_tools = [
        tool_map[name]
        for name in (
            "read_file",
            "read_artifact",
            "list_files",
            "grep_repo",
            "edit_file",
            "write_file",
            "apply_patch",
            "shell",
            "cpu_info",
            "compile",
            "run",
        )
    ]
    return ToolBundle(
        all_tools=all_tools,
        tools_by_profile=tools_by_profile,
        runtime=runtime,
    )
