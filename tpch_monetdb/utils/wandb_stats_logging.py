import logging
from collections import defaultdict
from pathlib import Path
from typing import Optional

from agents import RunContextWrapper, RunHooks, TContext, Tool

import wandb
from tpch_monetdb.llm_cache.git_snapshotter import GitSnapshotter
from tpch_monetdb.utils.cloc_utils import calculate_loc, calculate_loc_breakdown
from tpch_monetdb.utils.token_usage import get_tokens_context_and_dollar_info

logger = logging.getLogger(__name__)


class WandbRunHook(RunHooks):
    """Hooks for tracking agent execution metrics to wandb."""

    def __init__(
        self,
        model,
        git_snapshotter: GitSnapshotter,
        prompt_idx: int = 0,
        disable: bool = False,
        cloc_cache_dir: Path | None = None,
    ):
        self.model = model
        self.git_snapshotter = git_snapshotter
        self.prompt_idx = prompt_idx
        self.disable = disable
        self.current_prompt: Optional[str] = None
        self.current_prompt_descriptor: Optional[str] = None
        self.current_turn_tools = {}
        self.last_turn = 0
        self.total_stats = defaultdict(int)
        self.total_type_counts = defaultdict(int)
        self.apply_patch_stats = defaultdict(int)
        self.cloc_cache_dir = cloc_cache_dir
        if self.cloc_cache_dir is not None:
            self.cloc_cache_dir.mkdir(parents=True, exist_ok=True)
        self.logged_turn = -1
        self.apply_patch_added_ctr = 0
        self.apply_patch_deleted_ctr = 0
        self.pricing_missing_seen = False
        self.known_cost_seen = False
        self._prev_loc: int | None = None
        self._current_stage: str | None = None

    def log_metrics_callback(
        self, metrics: dict, log_and_increment: bool = False
    ) -> None:
        if self.disable:
            return None
        turn = self.last_turn
        assert self.logged_turn + 1 == turn, (
            f"Logged turn {self.logged_turn} is not one behind current turn {turn}"
        )
        self.logged_turn = turn
        metrics["turn"] = turn
        self.total_type_counts[metrics["type"]] += 1
        action_names = [
            "llm_call",
            "apply_patch_tool",
            "handoff",
            "shell_command",
            "validate",
            "compaction",
        ]
        for action in self.total_type_counts.keys():
            if action not in action_names:
                action_names.append(action)
        for action in action_names:
            action_str = action.replace("_", "")
            metrics[f"tool/{action_str}_count"] = self.total_type_counts[action]

        metrics["current_hash"] = self.git_snapshotter.current_hash
        assert self.git_snapshotter.current_hash is not None, (
            "Current hash should not be None"
        )

        # LOC breakdown (cpp/hpp/py/other/total) — kept backward-compat via
        # current_loc = total
        loc_breakdown = calculate_loc_breakdown(
            self.cloc_cache_dir,
            self.git_snapshotter.current_hash,
            self.git_snapshotter.working_dir,
        )
        metrics["current_loc"] = loc_breakdown.get("total", 0)
        for ext_key, count in loc_breakdown.items():
            if ext_key != "total":
                metrics[f"loc/{ext_key}"] = count

        # LOC delta vs previous turn
        prev_loc = getattr(self, "_prev_loc", None)
        if prev_loc is not None:
            metrics["loc/delta"] = metrics["current_loc"] - prev_loc
        self._prev_loc = metrics["current_loc"]

        # Stage name (set externally via set_current_stage)
        if hasattr(self, "_current_stage") and self._current_stage:
            metrics["stage"] = self._current_stage

        wandb.log(metrics, step=turn, commit=log_and_increment)
        assert log_and_increment, "log_and_increment must be True to increment turn"
        if log_and_increment:
            self.last_turn += 1
        return None

    def log_apply_patch_stats(
        self, operation_type: str, added_lines: int, deleted_lines: int
    ) -> None:
        if self.disable:
            return None
        self.apply_patch_stats[operation_type] += 1
        wandb.log(
            {
                f"apply_patch/{operation_type}_count": self.apply_patch_stats[
                    operation_type
                ]
            },
            step=self.last_turn,
            commit=False,
        )
        self.apply_patch_added_ctr += added_lines
        self.apply_patch_deleted_ctr += deleted_lines
        return None

    def set_current_stage(self, stage_name: str | None) -> None:
        """Update the stage name embedded in per-turn metrics."""
        self._current_stage = stage_name
        return None

    def log_query_hotspot_summary(
        self,
        stage_name: str,
        query_rt_ms: dict[str, float],
        baseline_rt_ms: dict[str, float] | None = None,
    ) -> None:
        """Log per-query no-CSV speedup and hotspot indicators to W&B."""
        if self.disable:
            return None
        metrics: dict[str, float] = {}
        for qid, rt_ms in query_rt_ms.items():
            metrics[f"query/{qid}/no_csv_kernel_runtime_ms"] = rt_ms
            if baseline_rt_ms and qid in baseline_rt_ms and baseline_rt_ms[qid] > 0 and rt_ms > 0:
                metrics[f"query/{qid}/no_csv_kernel_speedup_vs_baseline"] = baseline_rt_ms[qid] / rt_ms
        wandb.log(
            {**metrics, "stage": stage_name, "runtime_metric_kind": "kernel_ms"},
            step=self.last_turn,
            commit=False,
        )
        return None

    def log_ingest_summary(
        self,
        stage_name: str,
        bespoke_ingest_ms: float,
        baseline_ingest_ms: float,
        baseline_engine: str = "baseline",
        baseline_label: str = "baseline",
    ) -> None:
        """Log ingest secondary metric summary to W&B."""
        if self.disable:
            return None
        normalized_engine = baseline_engine.strip().lower()
        metrics: dict[str, object] = {
            "ingest/bespoke_ms": bespoke_ingest_ms,
            "ingest/baseline_ms": baseline_ingest_ms,
            "ingest/baseline_engine": normalized_engine,
            "ingest/baseline_label": baseline_label,
        }
        if baseline_ingest_ms > 0 and bespoke_ingest_ms > 0:
            metrics["ingest/speedup_vs_baseline"] = (
                baseline_ingest_ms / bespoke_ingest_ms
            )
        wandb.log({**metrics, "stage": stage_name}, step=self.last_turn, commit=False)
        return None

    async def on_agent_start(self, ctx, agent):
        if self.disable:
            return None
        logger.debug(f"[HOOK] Agent {agent.name} started (turn {self.last_turn})")
        return None

    async def on_llm_end(self, ctx, agent, output):
        if self.disable:
            return None
        assert hasattr(ctx, "usage"), "Context missing usage attribute"
        usage = ctx.usage
        token_stats = get_tokens_context_and_dollar_info(
            usage, self.model, last_entry_only=True, log=False
        )
        assert token_stats["num_llm_request"] == 1, (
            "Expected single LLM request for last entry"
        )
        cost_val = token_stats["cost"]
        cost_str = f"${cost_val:0.6f}" if cost_val is not None else "n/a"
        logger.info(
            f"[HOOK] LLM ended: Turn {self.last_turn} - Input tokens: "
            f"{token_stats['input_tokens']}, Output tokens: "
            f"{token_stats['visible_output_tokens']}, Cost: {cost_str}, "
            f"Context window usage: {token_stats['context_window_usage'] * 100:.1f}%"
        )

        wandb_metrics: dict[str, object] = {
            "type": "llm_call",
            "prompt_idx": self.prompt_idx,
            "agent_name": agent.name,
            "input_tokens": token_stats["input_tokens"],
            "cached_tokens": token_stats["cached_tokens"],
            "visible_output_tokens": token_stats["visible_output_tokens"],
            "billed_output_tokens": token_stats["billed_output_tokens"],
            "reasoning_tokens": token_stats["reasoning_tokens"],
            "context_window_usage": token_stats["context_window_usage"],
            "current_prompt": self.current_prompt,
            "current_prompt_descriptor": self.current_prompt_descriptor,
            "pricing_missing": token_stats["pricing_missing"],
        }
        if token_stats["pricing_missing"]:
            self.pricing_missing_seen = True
        if cost_val is not None:
            self.known_cost_seen = True
            wandb_metrics["cost_usd"] = cost_val
        self.current_prompt = None
        self.current_prompt_descriptor = None
        self.total_stats["input_tokens"] += token_stats["input_tokens"]
        self.total_stats["cached_tokens"] += token_stats["cached_tokens"]
        self.total_stats["visible_output_tokens"] += token_stats["visible_output_tokens"]
        self.total_stats["billed_output_tokens"] += token_stats["billed_output_tokens"]
        self.total_stats["reasoning_tokens"] += token_stats["reasoning_tokens"]
        if cost_val is not None:
            self.total_stats["cost_usd"] += cost_val
        wandb_metrics.update(
            {
                "total/input_tokens": self.total_stats["input_tokens"],
                "total/cached_tokens": self.total_stats["cached_tokens"],
                "total/visible_output_tokens": self.total_stats["visible_output_tokens"],
                "total/billed_output_tokens": self.total_stats["billed_output_tokens"],
                "total/reasoning_tokens": self.total_stats["reasoning_tokens"],
                "total/pricing_missing": int(self.pricing_missing_seen),
            }
        )
        if self.known_cost_seen:
            wandb_metrics["total/cost_usd"] = self.total_stats["cost_usd"]
        self.log_metrics_callback(wandb_metrics, log_and_increment=True)
        return None

    async def on_agent_end(self, ctx, agent, output):
        if self.disable:
            return None
        logger.debug(f"[HOOK] Agent {agent.name} ended (turn {self.last_turn})")
        return None

    async def on_tool_start(
        self,
        context: RunContextWrapper[TContext],
        agent,
        tool: Tool,
    ):
        if self.disable:
            return None
        tool_name = tool.name if hasattr(tool, "name") else str(tool)
        logger.debug(
            f"[HOOK] Agent {agent.name} starting tool: {tool_name} "
            f"(turn {self.last_turn})"
        )
        if tool_name == "apply_patch":
            self.apply_patch_added_ctr = 0
            self.apply_patch_deleted_ctr = 0
        return None

    async def on_tool_end(
        self,
        context: RunContextWrapper[TContext],
        agent,
        tool: Tool,
        result: str,
    ):
        if self.disable:
            return None
        tool_name = tool.name if hasattr(tool, "name") else str(tool)
        self.current_turn_tools[tool_name] = (
            self.current_turn_tools.get(tool_name, 0) + 1
        )
        if tool_name == "apply_patch":
            self.log_metrics_callback(
                {
                    "type": "apply_patch_tool",
                    "apply_patch/added_loc_count": self.apply_patch_added_ctr,
                    "apply_patch/deleted_loc_count": self.apply_patch_deleted_ctr,
                },
                log_and_increment=True,
            )
        return None

    async def on_handoff(self, ctx, from_agent, to_agent):
        if self.disable:
            return None
        logger.info(
            f"[HOOK] Handoff from {from_agent.name} to {to_agent.name} "
            f"(turn {self.last_turn})"
        )
        self.log_metrics_callback(
            {
                "handoff/from": from_agent.name,
                "handoff/to": to_agent.name,
                "type": "handoff",
            },
            log_and_increment=True,
        )
        return None

    def log_optimization_stage(
        self,
        query_id: str,
        stage_name: str,
        rt_before_ms: float,
        rt_after_ms: float,
        snapshot_before: str,
        snapshot_after: str,
        validation_passed: bool = True,
    ) -> None:
        """记录 optimization stage 的指标.
        
        Args:
            query_id: 查询 ID
            stage_name: Stage 名称 (trace/expert_knowledge/human_reference)
            rt_before_ms: 优化前 runtime (ms)
            rt_after_ms: 优化后 runtime (ms)
            snapshot_before: 优化前 snapshot hash
            snapshot_after: 优化后 snapshot hash
            validation_passed: 验证是否通过
        """
        if self.disable:
            return None
        
        # 计算 speedup metrics
        improvement_factor = rt_before_ms / rt_after_ms if rt_after_ms > 0 else float("inf")
        improvement_pct = ((rt_before_ms - rt_after_ms) / rt_before_ms * 100) if rt_before_ms > 0 else 0
        
        metrics = {
            "optimization/query_id": query_id,
            "optimization/stage_name": stage_name,
            "optimization/rt_before_ms": rt_before_ms,
            "optimization/rt_after_ms": rt_after_ms,
            "optimization/improvement_factor": improvement_factor,
            "optimization/improvement_pct": improvement_pct,
            "optimization/snapshot_before": snapshot_before,
            "optimization/snapshot_after": snapshot_after,
            "optimization/validation_passed": validation_passed,
            "engine": "generated_tpch",
            "type": "optimization_stage",
        }
        
        self.log_metrics_callback(metrics, log_and_increment=True)
        
        logger.info(
            f"[OPTIMIZATION] {query_id}/{stage_name} no-CSV kernel runtime: "
            f"{rt_before_ms:.2f}ms -> {rt_after_ms:.2f}ms "
            f"({improvement_factor:.2f}x, {improvement_pct:+.1f}%)"
        )
        return None

    def log_optimization_speedup_vs_baseline(
        self,
        query_id: str,
        stage_name: str,
        no_csv_kernel_runtime_ms: float,
        baseline_runtime_ms: float,
        baseline_engine: str = "baseline",
        baseline_label: str = "baseline",
        lazy_build_suspected: bool = False,
        total_kernel_runtime_ms: Optional[float] = None,
        first_query_ms: Optional[float] = None,
        median_query_ms: Optional[float] = None,
        runtime_boundary: Optional[str] = None,
    ) -> None:
        """记录相对当前 baseline 的 no-CSV kernel speedup（不含 ingest）."""
        if self.disable:
            return None

        speedup = (
            baseline_runtime_ms / no_csv_kernel_runtime_ms
            if no_csv_kernel_runtime_ms > 0
            else float("inf")
        )
        normalized_engine = baseline_engine.strip().lower()

        metrics: dict[str, object] = {
            "optimization/query_id": query_id,
            "optimization/stage_name": stage_name,
            "optimization/no_csv_kernel_speedup_vs_baseline": speedup,
            "optimization/no_csv_kernel_runtime_ms": no_csv_kernel_runtime_ms,
            "optimization/baseline_runtime_ms": baseline_runtime_ms,
            "optimization/baseline_engine": normalized_engine,
            "optimization/baseline_label": baseline_label,
            "optimization/runtime_metric_kind": "kernel_ms",
            "optimization/lazy_build_suspected": lazy_build_suspected,
            "engine": "generated_tpch",
            "type": "optimization_speedup",
        }
        if total_kernel_runtime_ms is not None:
            metrics["optimization/total_kernel_runtime_ms"] = total_kernel_runtime_ms
        if first_query_ms is not None:
            metrics["optimization/first_query_ms"] = first_query_ms
        if median_query_ms is not None:
            metrics["optimization/median_query_ms"] = median_query_ms
        if runtime_boundary is not None:
            metrics["optimization/runtime_boundary"] = runtime_boundary

        self.log_metrics_callback(metrics, log_and_increment=True)
        logger.info(
            "[OPTIMIZATION] %s/%s no-CSV kernel speedup vs %s: %.2fx%s",
            query_id,
            stage_name,
            baseline_label,
            speedup,
            " [lazy-build suspected]" if lazy_build_suspected else "",
        )
        return None

    def log_ingest_comparison(
        self,
        stage_name: str,
        bespoke_ingest_ms: float,
        bespoke_load_ms: Optional[float],
        bespoke_build_ms: Optional[float],
        bespoke_rows_per_sec: Optional[float],
        bespoke_metrics_per_sec: Optional[float],
        baseline_ingest_ms: Optional[float],
        baseline_rows_per_sec: Optional[float],
        baseline_metrics_per_sec: Optional[float],
        baseline_workers: Optional[int],
        baseline_engine: str = "baseline",
        baseline_label: str = "baseline",
    ) -> None:
        """记录 ingest comparison 指标（与 runtime speedup 完全分离，不产生 combined speedup）."""
        if self.disable:
            return None

        normalized_engine = baseline_engine.strip().lower()
        metrics: dict[str, object] = {
            "ingest/stage_name": stage_name,
            "ingest/generated_tpch_ingest_ms": bespoke_ingest_ms,
            "ingest/baseline_engine": normalized_engine,
            "ingest/baseline_label": baseline_label,
            "engine": "generated_tpch",
            "type": "ingest_comparison",
        }
        if bespoke_load_ms is not None:
            metrics["ingest/generated_tpch_load_ms"] = bespoke_load_ms
        if bespoke_build_ms is not None:
            metrics["ingest/generated_tpch_build_ms"] = bespoke_build_ms
        if bespoke_rows_per_sec is not None:
            metrics["ingest/generated_tpch_ingest_rows_per_sec"] = bespoke_rows_per_sec
        if bespoke_metrics_per_sec is not None:
            metrics["ingest/generated_tpch_ingest_metrics_per_sec"] = bespoke_metrics_per_sec
        if baseline_ingest_ms is not None:
            metrics["ingest/baseline_ingest_ms"] = baseline_ingest_ms
            if bespoke_ingest_ms > 0:
                metrics["ingest/ingest_speedup_vs_baseline"] = (
                    baseline_ingest_ms / bespoke_ingest_ms
                )
        if baseline_rows_per_sec is not None:
            metrics["ingest/baseline_ingest_rows_per_sec"] = baseline_rows_per_sec
            if bespoke_rows_per_sec is not None and baseline_rows_per_sec > 0:
                metrics["ingest/ingest_throughput_ratio_vs_baseline"] = (
                    bespoke_rows_per_sec / baseline_rows_per_sec
                )
        if baseline_metrics_per_sec is not None:
            metrics["ingest/baseline_ingest_metrics_per_sec"] = baseline_metrics_per_sec
        if baseline_workers is not None:
            metrics["ingest/workers"] = baseline_workers

        self.log_metrics_callback(metrics, log_and_increment=True)
        logger.info(
            "[INGEST] %s: bespoke=%.3fms %s=%s rows/s(bespoke)=%s",
            stage_name, bespoke_ingest_ms,
            baseline_label,
            f"{baseline_ingest_ms:.3f}ms" if baseline_ingest_ms else "n/a",
            f"{bespoke_rows_per_sec:.0f}" if bespoke_rows_per_sec else "n/a",
        )
        return None

    def log_optimization_final_summary(
        self,
        query_id: str,
        baseline_runtime_ms: float,
        final_no_csv_kernel_runtime_ms: float,
        best_no_csv_kernel_speedup_vs_baseline: float,
        final_correctness: bool,
        final_snapshot: str,
        baseline_engine: str = "baseline",
        baseline_label: str = "baseline",
    ) -> None:
        """记录 optimization final no-CSV kernel summary.
        
        Args:
            query_id: 查询 ID
            baseline_runtime_ms: 当前 baseline runtime (ms)
            final_no_csv_kernel_runtime_ms: 最终 no-CSV kernel runtime (ms)
            best_no_csv_kernel_speedup_vs_baseline: 最佳 no-CSV kernel baseline speedup
            final_correctness: 最终正确性状态
            final_snapshot: 最终 snapshot hash
        """
        if self.disable:
            return None

        normalized_engine = baseline_engine.strip().lower()
        
        total_improvement_factor = (
            baseline_runtime_ms / final_no_csv_kernel_runtime_ms
            if final_no_csv_kernel_runtime_ms > 0
            else float("inf")
        )
        total_improvement_pct = (
            (baseline_runtime_ms - final_no_csv_kernel_runtime_ms)
            / baseline_runtime_ms
            * 100
            if baseline_runtime_ms > 0
            else 0
        )
        
        metrics: dict[str, object] = {
            f"optimization_final/{query_id}/baseline_runtime_ms": baseline_runtime_ms,
            f"optimization_final/{query_id}/final_no_csv_kernel_runtime_ms": final_no_csv_kernel_runtime_ms,
            f"optimization_final/{query_id}/final_no_csv_kernel_speedup_vs_baseline": total_improvement_factor,
            f"optimization_final/{query_id}/final_no_csv_kernel_improvement_pct": total_improvement_pct,
            f"optimization_final/{query_id}/best_no_csv_kernel_speedup_vs_baseline": best_no_csv_kernel_speedup_vs_baseline,
            f"optimization_final/{query_id}/final_correctness": final_correctness,
            f"optimization_final/{query_id}/final_snapshot": final_snapshot,
            f"optimization_final/{query_id}/runtime_metric_kind": "kernel_ms",
            f"optimization_final/{query_id}/baseline_engine": normalized_engine,
            f"optimization_final/{query_id}/baseline_label": baseline_label,
            "optimization/query_id": query_id,
            "engine": "generated_tpch",
            "type": "optimization_final_summary",
        }
        self.log_metrics_callback(metrics, log_and_increment=True)
        
        logger.info(
            f"[OPTIMIZATION FINAL] {query_id} no-CSV kernel: "
            f"{baseline_runtime_ms:.2f}ms {baseline_label} -> "
            f"{final_no_csv_kernel_runtime_ms:.2f}ms Generated TPC-H "
            f"({total_improvement_factor:.2f}x final, "
            f"{best_no_csv_kernel_speedup_vs_baseline:.2f}x best vs {baseline_label})"
        )
        return None
