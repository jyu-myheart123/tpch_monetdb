import json
import logging
import re
from pathlib import Path
from typing import Any

from agents.editor import ApplyPatchOperation
from agents.run_context import RunContextWrapper
from agents.tool import FunctionTool
from pydantic import BaseModel, Field

from tpch_monetdb.tools.function_tool_args import load_function_tool_args
from tpch_monetdb.tools.workspace_editor import WorkspaceEditor
from tpch_monetdb.utils.wandb_stats_logging import WandbRunHook

logger = logging.getLogger(__name__)


class LitellmApplyPatchArgs(BaseModel):
    type: str = Field(..., description="create_file, update_file, or delete_file")
    path: str = Field(..., description="Path relative to workspace root")
    diff: str | None = Field(None, description="Unified diff for create/update")


class LitellmApplyPatchTool:
    def __init__(self, root: Path, wandb_metrics_hook: WandbRunHook | None) -> None:
        self._editor = WorkspaceEditor(root, wandb_metrics_hook)

    @staticmethod
    def _normalize_diff(diff: str, op_type: str) -> str:
        lines = diff.splitlines()
        cleaned: list[str] = []
        for line in lines:
            if line.startswith("diff --git "):
                continue
            if line.startswith("index "):
                continue
            if line.startswith("--- "):
                continue
            if line.startswith("+++ "):
                continue
            if re.match(r"@@ .* @@$", line):
                cleaned.append("@@")
                continue
            cleaned.append(line)
        if op_type == "create_file":
            cleaned = [line for line in cleaned if line.startswith("+")]
        return "\n".join(cleaned)

    async def __call__(self, op_type: str, path: str, diff: str | None) -> str:
        if diff is not None:
            diff = self._normalize_diff(diff, op_type)
        if op_type == "create_file":
            if diff is None:
                raise RuntimeError("create_file requires a non-empty diff.")
            if not any(line.startswith("+") for line in diff.splitlines()):
                raise RuntimeError(
                    "create_file requires at least one '+' line after diff normalization."
                )
        op = ApplyPatchOperation(type=op_type, path=path, diff=diff)
        if op.type == "create_file":
            result = self._editor.create_file(op)
        elif op.type == "update_file":
            result = self._editor.update_file(op)
        elif op.type == "delete_file":
            result = self._editor.delete_file(op)
        else:
            raise RuntimeError(f"Unknown apply_patch operation type: {op_type}")
        if hasattr(result, "output") and result.output is not None:
            return result.output
        return "ok"


def make_litellm_apply_patch_tool(
    root: Path,
    wandb_metrics_hook: WandbRunHook | None = None,
) -> FunctionTool:
    impl = LitellmApplyPatchTool(root=root, wandb_metrics_hook=wandb_metrics_hook)

    async def on_invoke(ctx: RunContextWrapper[Any], args_json: str) -> str:
        try:
            parsed = load_function_tool_args(args_json)
            args = LitellmApplyPatchArgs.model_validate(parsed)
            return await impl(args.type, args.path, args.diff)
        except json.JSONDecodeError as exc:
            return (
                f"Error: Invalid JSON format. {str(exc)}. "
                "Please ensure the arguments are valid JSON with proper escaping."
            )
        except Exception as exc:
            return f"Error applying patch: {str(exc)}"

    return FunctionTool(
        name="apply_patch",
        description="Applies a unified diff to create/update/delete a file",
        params_json_schema=LitellmApplyPatchArgs.model_json_schema(),
        on_invoke_tool=on_invoke,
    )
