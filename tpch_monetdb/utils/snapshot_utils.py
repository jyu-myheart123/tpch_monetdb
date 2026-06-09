import logging
from pathlib import Path

from tpch_monetdb.llm_cache.git_snapshotter import GitSnapshotter

logger = logging.getLogger(__name__)


def load_storage_plan_from_snapshot(
    args, snapshotter: GitSnapshotter, workspace_path: Path
) -> str:
    assert not args.continue_run, (
        "storage_plan_snapshot and continue_current_snapshot not compatible"
    )
    assert snapshotter.has_snapshot(args.storage_plan_snapshot), (
        f"Snapshot {args.storage_plan_snapshot} not found in repo."
    )
    logger.info(f"Restoring snapshot {args.storage_plan_snapshot}")
    snapshotter.restore(args.storage_plan_snapshot)
    storage_plan_path = workspace_path / "storage_plan.txt"
    assert storage_plan_path.exists(), (
        f"storage_plan.txt not found in snapshot {args.storage_plan_snapshot}"
    )
    storage_plan = storage_plan_path.read_text()
    return storage_plan

