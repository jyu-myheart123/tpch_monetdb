import json
import logging
from pathlib import Path
from typing import Any, Optional

from agents.run_context import RunContextWrapper
from agents.tool import FunctionTool
from pydantic import BaseModel, Field

from tpch_monetdb.llm_cache.git_snapshotter import GitSnapshotter
from tpch_monetdb.tools.function_tool_args import load_function_tool_args
from tpch_monetdb.utils.wandb_stats_logging import WandbRunHook

from .utils import make_compiler

logger = logging.getLogger(__name__)


class CompileTool:
    """Compiles the database"""

    def __init__(
        self,
        cwd: Path,
        compile_cache_dir: Optional[Path] = None,
        git_snapshotter: Optional[GitSnapshotter] = None,
        wandb_metrics_hook: Optional[WandbRunHook] = None,
    ) -> None:
        self.cwd = cwd
        self.compile_cache_dir = compile_cache_dir
        self.git_snapshotter = git_snapshotter
        self.wandb_metrics_hook = wandb_metrics_hook
        self.compiler = self._build_compiler()

    def _build_compiler(self) -> Any:
        return make_compiler(self.cwd, self.compile_cache_dir, self.git_snapshotter)

    def __call__(self, optimize: bool) -> str:
        logger.info("compile call")
        self.compiler = self._build_compiler()

        cxx_flags = []
        if optimize:
            cxx_flags.extend(["-O3", "-flto"])
        self.compiler.set_extra_cxxflags(
            cxx_flags
        )  # if this methodolyg is changed, keep in mind to update the cache hash calculation

        err = self.compiler.build()
        if err is None:
            output = "**Compilation successfull**"
        else:
            output = err

        if self.wandb_metrics_hook is not None:
            self.wandb_metrics_hook.log_metrics_callback(
                {
                    "type": "compile",
                    "compile/error": True if err is not None else False,
                },
                log_and_increment=True,
            )

        return output


class CompileArgs(BaseModel):
    optimize: bool = Field(..., description="Enable compiler optimization")


def make_compile_tool(
    cwd: Path,
    compile_cache_dir: Optional[Path] = None,
    git_snapshotter: Any = None,
    wandb_metrics_hook: Optional[WandbRunHook] = None,
) -> FunctionTool:
    impl = CompileTool(cwd, compile_cache_dir, git_snapshotter, wandb_metrics_hook)

    async def on_invoke(ctx: RunContextWrapper[Any], args_json: str) -> str:
        try:
            parsed = load_function_tool_args(args_json)
            args = CompileArgs.model_validate(parsed)
            return impl(optimize=args.optimize)
        except json.JSONDecodeError as e:
            return f"Error: Invalid JSON format. {str(e)}."
        except Exception as e:
            return f"Error running compile: {str(e)}"

    return FunctionTool(
        name="compile",
        description="Compiles the database",
        params_json_schema=CompileArgs.model_json_schema(),
        on_invoke_tool=on_invoke,
    )
