import argparse
import hashlib
import logging
import traceback
import uuid
from dataclasses import dataclass, field
from typing import Any

from tpch_monetdb.tools.error_envelope import ErrorEnvelope
from tpch_monetdb.utils.wandb_runtime_guard import (
    DEFAULT_WANDB_INIT_TIMEOUT_S,
    run_callable_with_timeout,
)

logger = logging.getLogger(__name__)
WANDB_INIT_MAX_ATTEMPTS = 3
_FAILURE_EXCERPT_BUDGET = 1000


@dataclass
class WandbInitResult:
    """Structured result from init_wandb_run_with_retry."""

    run: Any
    primary_run_id: str
    final_run_id: str
    attempt_count: int
    attempted_run_ids: list[str] = field(default_factory=list)
    used_fallback: bool = False
    first_failure_excerpt: str | None = None


def init_wandb_run_with_retry(
    wandb_module: Any,
    args: argparse.Namespace,
    entity: str,
    project: str,
    tags: list[str],
    max_attempts: int = WANDB_INIT_MAX_ATTEMPTS,
    init_timeout_s: float = DEFAULT_WANDB_INIT_TIMEOUT_S,
) -> WandbInitResult:
    """Initialize W&B run with retry and return structured result."""
    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1")

    primary_run_id = hashlib.md5(args.conv_name.encode("utf-8")).hexdigest()
    run_ids = [primary_run_id] + [uuid.uuid4().hex for _ in range(max_attempts - 1)]

    first_failure_excerpt: str | None = None
    terminal_error_code = "WANDB_INIT_FAILED"

    for idx, run_id in enumerate(run_ids, start=1):
        resume_mode = "allow" if idx == 1 else "never"
        try:
            run = run_callable_with_timeout(
                lambda: wandb_module.init(
                    config=vars(args),
                    entity=entity,
                    project=project,
                    name=f"{args.conv_name}",
                    id=run_id,
                    resume=resume_mode,
                    tags=tags,
                ),
                timeout_s=init_timeout_s,
                operation_name="wandb.init",
            )
            return WandbInitResult(
                run=run,
                primary_run_id=primary_run_id,
                final_run_id=run_id,
                attempt_count=idx,
                attempted_run_ids=run_ids[:idx],
                used_fallback=(idx > 1),
                first_failure_excerpt=first_failure_excerpt,
            )
        except Exception as exc:
            terminal_error_code = (
                "WANDB_INIT_TIMEOUT"
                if isinstance(exc, TimeoutError)
                else "WANDB_INIT_FAILED"
            )
            formatted = (
                f"{type(exc).__name__}: {exc}\n"
                + "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
            )
            if first_failure_excerpt is None:
                excerpt = formatted[:_FAILURE_EXCERPT_BUDGET]
                if len(formatted) > _FAILURE_EXCERPT_BUDGET:
                    excerpt += "\n...[TRUNCATED]"
                first_failure_excerpt = excerpt
            if idx < max_attempts:
                logger.warning(
                    "W&B init failed for run_id=%s (attempt %d/%d); retrying with a new run_id",
                    run_id,
                    idx,
                    max_attempts,
                )
                continue
            raise RuntimeError(
                str(
                    ErrorEnvelope(
                        code=terminal_error_code,
                        category="telemetry",
                        stage="wandb_init",
                        message=(
                            "Failed to initialize W&B after retries.\n"
                            f"conv_name={args.conv_name}\n"
                            f"entity={entity}, project={project}\n"
                            f"init_timeout_s={init_timeout_s}\n"
                            f"attempted_run_ids={run_ids}\n"
                            f"attempt_count={max_attempts}\n"
                            f"first_failure_excerpt:\n{first_failure_excerpt or ''}"
                        ),
                        recoverable=False,
                        relevant_files=(
                            "tpch_monetdb/main_tpch_monetdb.py",
                            "tpch_monetdb/utils/wandb_init.py",
                        ),
                        recommended_next_action=(
                            "Check W&B logs, then rerun with a fresh conv name or disable_wandb."
                        ),
                    )
                )
            ) from exc

    raise RuntimeError("W&B init retry loop reached an unexpected terminal state")
