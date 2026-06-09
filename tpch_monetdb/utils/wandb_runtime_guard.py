from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any, Callable

from tpch_monetdb.tools.error_envelope import ErrorEnvelope

logger = logging.getLogger(__name__)

DEFAULT_WANDB_INIT_TIMEOUT_S = 30.0
DEFAULT_WANDB_UPLOAD_TIMEOUT_S = 120.0
DEFAULT_WANDB_FINISH_TIMEOUT_S = 30.0
DEFAULT_WANDB_FINISH_RETRIES = 1


def run_callable_with_timeout(
    func: Callable[[], Any],
    *,
    timeout_s: float,
    operation_name: str,
) -> Any:
    """Run a callable with a timeout guard using a daemon thread."""
    if timeout_s <= 0:
        return func()

    state: dict[str, Any] = {"value": None, "error": None}

    def _runner() -> None:
        try:
            state["value"] = func()
        except Exception as exc:  # pragma: no cover - exercised via callers
            state["error"] = exc
        return None

    thread = threading.Thread(
        target=_runner,
        name=f"{operation_name}-guard",
        daemon=True,
    )
    thread.start()
    thread.join(timeout=timeout_s)
    if thread.is_alive():
        raise TimeoutError(
            f"{operation_name} timed out after {timeout_s:.2f}s"
        )
    if state["error"] is not None:
        raise state["error"]
    return state["value"]


def upload_workspace_code_with_guard(
    wandb_run: Any,
    workspace_path: Path,
    include_fn: Callable[[str, str], bool],
    *,
    timeout_s: float,
) -> None:
    """Upload runtime workspace code to W&B with timeout and explicit failure."""
    if wandb_run is None or not workspace_path.exists():
        return None

    def _upload() -> None:
        wandb_run.log_code(
            root=str(workspace_path),
            name="workspace_code",
            include_fn=include_fn,
        )
        return None

    try:
        run_callable_with_timeout(
            _upload,
            timeout_s=timeout_s,
            operation_name="wandb.log_code",
        )
    except TimeoutError as exc:
        raise RuntimeError(
            str(
                ErrorEnvelope(
                    code="WANDB_LOG_CODE_TIMEOUT",
                    category="telemetry",
                    stage="wandb_finalize",
                    message=(
                        "W&B workspace code upload timed out. "
                        f"workspace={workspace_path}, timeout_s={timeout_s}"
                    ),
                    recoverable=False,
                    relevant_files=(
                        "tpch_monetdb/main_tpch_monetdb.py",
                        "tpch_monetdb/utils/wandb_runtime_guard.py",
                    ),
                    recommended_next_action=(
                        "Increase --wandb_upload_timeout_s or inspect W&B/network health."
                    ),
                )
            )
        ) from exc
    except Exception as exc:
        raise RuntimeError(
            str(
                ErrorEnvelope(
                    code="WANDB_LOG_CODE_FAILED",
                    category="telemetry",
                    stage="wandb_finalize",
                    message=(
                        "W&B workspace code upload failed. "
                        f"workspace={workspace_path}, error={exc}"
                    ),
                    recoverable=False,
                    relevant_files=(
                        "tpch_monetdb/main_tpch_monetdb.py",
                        "tpch_monetdb/utils/wandb_runtime_guard.py",
                    ),
                    recommended_next_action=(
                        "Inspect W&B credentials and upload filters, then rerun."
                    ),
                )
            )
        ) from exc
    return None


def finish_wandb_with_guard(
    wandb_module: Any,
    *,
    timeout_s: float,
    retries: int,
) -> None:
    """Finish W&B run with bounded retries and explicit failure envelopes."""
    if retries < 0:
        raise ValueError("retries must be >= 0")

    finish = getattr(wandb_module, "finish", None)
    if not callable(finish):
        return None

    total_attempts = retries + 1
    last_error: Exception | None = None

    for attempt in range(1, total_attempts + 1):
        try:
            run_callable_with_timeout(
                finish,
                timeout_s=timeout_s,
                operation_name="wandb.finish",
            )
            return None
        except Exception as exc:
            last_error = exc
            if attempt < total_attempts:
                logger.warning(
                    "W&B finish failed (attempt %d/%d): %s",
                    attempt,
                    total_attempts,
                    exc,
                )

    if isinstance(last_error, TimeoutError):
        code = "WANDB_FINISH_TIMEOUT"
        message = (
            "W&B finish timed out after retries. "
            f"timeout_s={timeout_s}, attempts={total_attempts}"
        )
        action = "Increase --wandb_finish_timeout_s/--wandb_finish_retries or inspect W&B/network health."
    else:
        code = "WANDB_FINISH_FAILED"
        message = (
            "W&B finish failed after retries. "
            f"attempts={total_attempts}, error={last_error}"
        )
        action = "Inspect W&B client logs and credentials, then rerun."

    raise RuntimeError(
        str(
            ErrorEnvelope(
                code=code,
                category="telemetry",
                stage="wandb_finalize",
                message=message,
                recoverable=False,
                relevant_files=(
                    "tpch_monetdb/main_tpch_monetdb.py",
                    "tpch_monetdb/utils/wandb_runtime_guard.py",
                ),
                recommended_next_action=action,
            )
        )
    )
