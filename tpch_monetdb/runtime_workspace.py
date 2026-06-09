import logging
from pathlib import Path

from tpch_monetdb.llm_cache import GitSnapshotter

logger = logging.getLogger(__name__)

_RUNTIME_CACHE_REPO = "git://c01/bespoke_cache.git"
_RUNTIME_EXTRA_GITIGNORE = ("*.o", "*.d", "/db", "/build/", "*.log", "*.tmp")


def resolve_runtime_workspace_path(tpch_monetdb_root: Path) -> Path:
    workspace_path = tpch_monetdb_root / "output"
    return workspace_path


def build_runtime_snapshotter(
    tpch_monetdb_root: Path,
    *,
    disable_repo_sync: bool,
    keep_csv: bool,
) -> GitSnapshotter:
    workspace_path = resolve_runtime_workspace_path(tpch_monetdb_root)
    workspace_path.mkdir(parents=True, exist_ok=True)
    extra_gitignore = list(_RUNTIME_EXTRA_GITIGNORE)
    if not keep_csv:
        extra_gitignore.append("*.csv")
    cache_repo = None if disable_repo_sync else _RUNTIME_CACHE_REPO
    return GitSnapshotter(
        cache_repo=cache_repo,
        working_dir=workspace_path,
        extra_gitignore=extra_gitignore,
    )


def _prepare_runtime_workspace(
    snapshotter: GitSnapshotter,
    workspace_path: Path,
    *,
    continue_run: bool = False,
    reset_git_history: bool = False,
) -> None:
    if continue_run:
        return None
    if reset_git_history:
        logger.warning(
            'Runtime workspace "%s" keeps old git history; recreating repo before the run.',
            workspace_path,
        )
        snapshotter.recreate_repo()
    else:
        was_dirty = snapshotter.is_dirty()
        if was_dirty:
            logger.warning(
                'Runtime workspace "%s" is dirty; resetting generated files before the run.',
                workspace_path,
            )
        snapshotter.clean_worktree(include_ignored=True)
    assert not snapshotter.is_dirty(), (
        f'Please remove all uncommitted changes in "{workspace_path}". '
        f"We expect a clean working directory to ensure reproducibility."
    )
    return None


def _snapshot_final_workspace_state(
    snapshotter: GitSnapshotter,
    conv_name: str,
) -> None:
    if not snapshotter.is_dirty():
        return None
    _, commit_hash = snapshotter.snapshot(f"{conv_name}-finalize")
    if commit_hash is None:
        raise RuntimeError(
            f'Workspace "{snapshotter.working_dir}" is dirty but final snapshot creation produced no commit.'
        )
    return None
