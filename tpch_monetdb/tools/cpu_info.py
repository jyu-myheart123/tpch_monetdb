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

from tpch_monetdb.llm_cache.git_snapshotter import GitSnapshotter
from tpch_monetdb.tools.function_tool_args import load_function_tool_args
from tpch_monetdb.tools.litellm_shell import CHARS_PER_TOKEN, MAX_TOOL_RESULT_TOKENS
from tpch_monetdb.tools.sandbox import SandboxConfig, sandbox_shell_async
from tpch_monetdb.utils.wandb_stats_logging import WandbRunHook

logger = logging.getLogger(__name__)


class CpuInfoArgs(BaseModel):
    timeout_ms: int | None = Field(
        2000, description="Timeout in milliseconds for each probe command"
    )


class CpuInfoTool:
    def __init__(
        self,
        cwd: Path,
        cache_dir: Path,
        git_snapshotter: Optional[GitSnapshotter] = None,
        wandb_metrics_hook: Optional[WandbRunHook] = None,
        max_output_tokens: int = MAX_TOOL_RESULT_TOKENS,
    ) -> None:
        """Collect read-only CPU topology and ISA evidence for optimization decisions."""
        self.cwd = cwd
        self.cache_dir = cache_dir
        self.git_snapshotter = git_snapshotter
        self.wandb_metrics_hook = wandb_metrics_hook
        self.max_output_tokens = max_output_tokens
        return None

    async def _run_probe(self, command: str, timeout_ms: int | None) -> dict[str, Any]:
        """Run a single read-only probe command inside the sandbox."""
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
        return {
            "command": command,
            "stdout": stdout,
            "stderr": stderr,
            "exit_code": getattr(proc, "returncode", None),
            "timed_out": timed_out,
        }

    def _truncate(self, text: str) -> str:
        """Clamp oversized raw probe output to the shared tool token budget."""
        raise NotImplementedError("TODO(student): implement stable head/tail truncation")

    def _parse_cpuinfo_flags(self, text: str) -> list[str]:
        """Extract ISA flags from /proc/cpuinfo when present."""
        raise NotImplementedError("TODO(student): parse flags/Features from cpuinfo text")

    def _parse_lscpu_summary(self, text: str) -> dict[str, str]:
        """Extract stable key/value fields from lscpu output."""
        raise NotImplementedError("TODO(student): parse selected lscpu key/value fields")

    def _build_response(self, probes: dict[str, dict[str, Any]]) -> dict[str, Any]:
        """Normalize raw probe outputs into a stable JSON payload for the agent."""
        raise NotImplementedError("TODO(student): build the structured cpu_info JSON payload")

    async def __call__(self, timeout_ms: int | None) -> str:
        """Collect CPU evidence and return it as structured JSON text.

        CPU topology and ISA facts belong to the current execution environment,
        not to the workspace snapshot, so this tool intentionally avoids
        cross-call caching.
        """
        probes = {
            "uname": await self._run_probe("uname -m", timeout_ms),
            "lscpu": await self._run_probe("lscpu", timeout_ms),
            "cpuinfo": await self._run_probe("cat /proc/cpuinfo", timeout_ms),
        }
        response = self._build_response(probes)
        result = json.dumps(response, ensure_ascii=True, indent=2, sort_keys=True)

        if self.wandb_metrics_hook is not None:
            self.wandb_metrics_hook.log_metrics_callback(
                {
                    "type": "cpu_info_command",
                    "cpu_info/num_probes": len(probes),
                },
                log_and_increment=True,
            )
        return result


def make_cpu_info_tool(
    cwd: Path,
    cache_dir: Path,
    git_snapshotter: Optional[GitSnapshotter] = None,
    wandb_metrics_hook: Optional[WandbRunHook] = None,
    max_output_tokens: int = MAX_TOOL_RESULT_TOKENS,
) -> FunctionTool:
    impl = CpuInfoTool(
        cwd=cwd,
        cache_dir=cache_dir,
        git_snapshotter=git_snapshotter,
        wandb_metrics_hook=wandb_metrics_hook,
        max_output_tokens=max_output_tokens,
    )

    async def on_invoke(ctx: RunContextWrapper[Any], args_json: str) -> str:
        del ctx
        try:
            parsed = load_function_tool_args(args_json)
            args = CpuInfoArgs.model_validate(parsed)
            return await impl(timeout_ms=args.timeout_ms)
        except json.JSONDecodeError as exc:
            return (
                f"Error: Invalid JSON format. {str(exc)}. "
                "Please ensure the arguments are valid JSON."
            )
        except Exception as exc:
            logger.exception("cpu_info tool failed")
            return f"Error collecting cpu info: {str(exc)}"

    return FunctionTool(
        name="cpu_info",
        description="Collects read-only CPU architecture, flags, cache, and NUMA information",
        params_json_schema=CpuInfoArgs.model_json_schema(),
        on_invoke_tool=on_invoke,
    )
