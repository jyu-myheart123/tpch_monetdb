import asyncio
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Optional

from agents.run_context import RunContextWrapper
from agents.tool import FunctionTool
from pydantic import BaseModel, Field

from tpch_monetdb.llm_cache import utils
from tpch_monetdb.llm_cache.git_snapshotter import GitSnapshotter
from tpch_monetdb.tools.function_tool_args import load_function_tool_args
from tpch_monetdb.tools.sandbox import SandboxConfig, sandbox_shell_async
from tpch_monetdb.utils.wandb_stats_logging import WandbRunHook

logger = logging.getLogger(__name__)

MAX_TOOL_RESULT_TOKENS = 4000
CHARS_PER_TOKEN = 4
MAX_OUTPUT_CHARS = MAX_TOOL_RESULT_TOKENS * CHARS_PER_TOKEN


def truncate_shell_output(
    command: str,
    stdout: str,
    stderr: str,
    exit_code: Optional[int],
    timed_out: bool,
    max_tokens: int = MAX_TOOL_RESULT_TOKENS,
) -> str:
    """Truncate shell output to fit within token budget."""
    max_chars = max_tokens * CHARS_PER_TOKEN
    if len(stderr) > max_chars // 4:
        keep = max_chars // 8
        stderr_display = (
            f"{stderr[:keep]}\n... [{len(stderr) - 2 * keep} chars truncated] ...\n"
            f"{stderr[-keep:]}"
        )
    else:
        stderr_display = stderr
    fixed_parts = len(command) + 20 + len(stderr_display) + 20 + 50
    stdout_budget = max_chars - fixed_parts
    if stdout_budget < 200 or len(stdout) <= stdout_budget:
        stdout_display = stdout
    else:
        half_budget = stdout_budget // 2 - 30
        if half_budget > 100:
            stdout_display = (
                f"{stdout[:half_budget]}\n"
                f"... [{len(stdout) - 2 * half_budget} chars truncated] ...\n"
                f"{stdout[-half_budget:]}"
            )
        else:
            stdout_display = f"{stdout[:stdout_budget]}\n... [truncated] ..."
    status = "timeout" if timed_out else "exit"
    return (
        f"$ {command}\n"
        f"stdout: {stdout_display}\n"
        f"stderr: {stderr_display}\n"
        f"exit_code: {exit_code}\n"
        f"status: {status}"
    )


class LitellmShellTool:
    def __init__(
        self,
        cwd: Path,
        cache_dir: Path,
        git_snapshotter: Optional[GitSnapshotter] = None,
        wandb_metrics_hook: Optional[WandbRunHook] = None,
        max_output_tokens: int = MAX_TOOL_RESULT_TOKENS,
        read_only: bool = False,
    ) -> None:
        self.cwd = cwd
        self.cache_dir = cache_dir
        self.git_snapshotter = git_snapshotter
        self.wandb_metrics_hook = wandb_metrics_hook
        self.max_output_tokens = max_output_tokens
        self.read_only = read_only
        self._last_command: str | None = None
        self._last_snapshot_hash: str | None = None
        self._consecutive_repeat_count = 0
        self._inspection_snapshot_hash: str | None = None
        self._consecutive_inspection_count = 0
        if not self.cache_dir.exists():
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self.cache_dir.chmod(0o777)

    def _cache_path_for(self, hash_value: str) -> Path:
        return self.cache_dir / f"{hash_value}.pkl"

    def _check_repeated_command(self, command: str) -> str | None:
        snapshot_hash = (
            self.git_snapshotter.current_hash if self.git_snapshotter is not None else None
        )
        normalized = " ".join(command.split())
        if normalized == self._last_command and snapshot_hash == self._last_snapshot_hash:
            self._consecutive_repeat_count += 1
        else:
            self._last_command = normalized
            self._last_snapshot_hash = snapshot_hash
            self._consecutive_repeat_count = 1
        if self._consecutive_repeat_count < 3:
            return None
        return (
            "Error: repeated shell command blocked. "
            f"You have already run `{normalized}` multiple times without any workspace change. "
            "Do not repeat broad inspection commands. "
            "Read a specific file, use edit_file or write_file to make progress, or call compile/run."
        )

    def _is_read_only_inspection(self, command: str) -> bool:
        normalized = " ".join(command.split()).lower()
        prefixes = (
            "cat ",
            "head ",
            "tail ",
            "ls ",
            "find ",
            "rg ",
            "sed -n ",
            "grep ",
            "wc ",
            "pwd",
        )
        if normalized == "pwd":
            return True
        return normalized.startswith(prefixes)

    def _check_inspection_loop(self, command: str) -> str | None:
        snapshot_hash = (
            self.git_snapshotter.current_hash if self.git_snapshotter is not None else None
        )
        if snapshot_hash != self._inspection_snapshot_hash:
            self._inspection_snapshot_hash = snapshot_hash
            self._consecutive_inspection_count = 0
        if not self._is_read_only_inspection(command):
            self._consecutive_inspection_count = 0
            return None
        self._consecutive_inspection_count += 1
        if self._consecutive_inspection_count < 10:
            return None
        return (
            "Error: excessive read-only shell inspection blocked. "
            "You have inspected the workspace many times without any workspace change. "
            "Do not continue reading files. Your next step must be edit_file, write_file, compile, or run."
        )

    async def __call__(self, command: str, timeout_ms: int | None) -> str:
        if "sudo" in command:
            raise RuntimeError("sudo rejected")
        if self.read_only and not self._is_read_only_inspection(command):
            return (
                "Error: shell command blocked in this stage. "
                "Only read-only inspection commands are allowed."
            )
        repeated_command_error = self._check_repeated_command(command)
        if repeated_command_error is not None:
            return repeated_command_error
        inspection_loop_error = self._check_inspection_loop(command)
        if inspection_loop_error is not None:
            return inspection_loop_error
        logger.debug(f"Running shell command: {command}")
        payload = {
            "snapshotter_hash": self.git_snapshotter.current_hash
            if self.git_snapshotter
            else None,
            "command": command,
            "timeout_ms": timeout_ms,
        }
        hash_value = utils.sha256(utils.stable_json(payload))
        path = self._cache_path_for(hash_value)
        if path.exists():
            cached = utils.load_pickle(path, str)
            if cached is not None:
                return cached

        tmp_root = os.environ.get("TMPDIR") or tempfile.gettempdir()
        cfg = SandboxConfig(
            writable_roots=[str(self.cwd)],
            cwd=str(self.cwd),
            tmp_root=tmp_root,
            fail_if_unavailable=True,
            nproc=None,
        )
        proc = await sandbox_shell_async(
            command,
            cfg=cfg,
            env=os.environ.copy(),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        timed_out = False
        try:
            timeout = (timeout_ms or 0) / 1000 or None
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(), timeout=timeout
            )
        except asyncio.TimeoutError:
            proc.kill()
            stdout_bytes, stderr_bytes = await proc.communicate()
            timed_out = True

        stdout = stdout_bytes.decode("utf-8", errors="ignore")
        stderr = stderr_bytes.decode("utf-8", errors="ignore")
        exit_code = getattr(proc, "returncode", None)
        output = truncate_shell_output(
            command=command,
            stdout=stdout,
            stderr=stderr,
            exit_code=exit_code,
            timed_out=timed_out,
            max_tokens=self.max_output_tokens,
        )
        utils.dump_pickle(path, output)

        if self.wandb_metrics_hook is not None:
            self.wandb_metrics_hook.log_metrics_callback(
                {
                    "type": "shell_command",
                    "shell/num_commands": 1,
                    "shell/commands": [command[:20]],
                },
                log_and_increment=True,
            )
        return output


class LitellmShellArgs(BaseModel):
    command: str = Field(..., description="Shell command to execute")
    timeout_ms: int | None = Field(
        None, description="Timeout in milliseconds (optional)"
    )


def make_litellm_shell_tool(
    cwd: Path,
    cache_dir: Path,
    git_snapshotter: Optional[GitSnapshotter] = None,
    wandb_metrics_hook: Optional[WandbRunHook] = None,
    max_output_tokens: int = MAX_TOOL_RESULT_TOKENS,
    read_only: bool = False,
) -> FunctionTool:
    impl = LitellmShellTool(
        cwd=cwd,
        cache_dir=cache_dir,
        git_snapshotter=git_snapshotter,
        wandb_metrics_hook=wandb_metrics_hook,
        max_output_tokens=max_output_tokens,
        read_only=read_only,
    )

    async def on_invoke(ctx: RunContextWrapper[Any], args_json: str) -> str:
        try:
            parsed = load_function_tool_args(args_json)
            args = LitellmShellArgs.model_validate(parsed)
            return await impl(command=args.command, timeout_ms=args.timeout_ms)
        except json.JSONDecodeError as exc:
            return (
                f"Error: Invalid JSON format. {str(exc)}. "
                "Please ensure the arguments are valid JSON."
            )
        except Exception as exc:
            return f"Error running shell command: {str(exc)}"

    return FunctionTool(
        name="shell",
        description="Runs a shell command locally",
        params_json_schema=LitellmShellArgs.model_json_schema(),
        on_invoke_tool=on_invoke,
    )
