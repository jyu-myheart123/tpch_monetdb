from pathlib import Path
from typing import Any, Optional

from agents.tool import FunctionTool

from tpch_monetdb.llm_cache.artifact_ledger import ArtifactLedger
from tpch_monetdb.tools.litellm_apply_patch import make_litellm_apply_patch_tool
from tpch_monetdb.tools.tpch import make_compile_tool
from tpch_monetdb.tools.tpch_monetdb_agent_tools import ToolBundle, build_tpch_monetdb_agent_tools
from tpch_monetdb.utils.wandb_stats_logging import WandbRunHook


def build_tools(
    use_litellm: bool,
    workspace_path: Path,
    cache_path: Path,
    shell_executor: Any,
    workspace_editor: Any,
    compile_cache_dir: Path,
    run_tool_wrapper: FunctionTool,
    git_snapshotter: Any = None,
    wandb_metrics_hook: Optional[WandbRunHook] = None,
    extra_read_roots: tuple[Path, ...] = (),
    artifact_ledger: ArtifactLedger | None = None,
) -> ToolBundle:
    """Build all runtime tools and share one artifact ledger across wrappers."""
    del use_litellm
    del shell_executor
    del workspace_editor
    compile_tool = make_compile_tool(
        cwd=workspace_path,
        compile_cache_dir=compile_cache_dir,
        git_snapshotter=git_snapshotter,
        wandb_metrics_hook=wandb_metrics_hook,
    )
    apply_patch_tool = make_litellm_apply_patch_tool(
        root=workspace_path,
        wandb_metrics_hook=wandb_metrics_hook,
    )
    bundle = build_tpch_monetdb_agent_tools(
        workspace_path=workspace_path,
        cache_path=cache_path,
        compile_tool=compile_tool,
        run_tool=run_tool_wrapper,
        git_snapshotter=git_snapshotter,
        wandb_metrics_hook=wandb_metrics_hook,
        apply_patch_tool=apply_patch_tool,
        extra_read_roots=extra_read_roots,
        artifact_ledger=artifact_ledger,
    )
    return bundle
