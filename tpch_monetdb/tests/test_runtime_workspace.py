import subprocess
from pathlib import Path

import tpch_monetdb.runtime_workspace
from tpch_monetdb.llm_cache import GitSnapshotter


def _run_git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
    )


def test_git_snapshotter_recreate_repo_clears_visible_history(tmp_path) -> None:
    snapshotter = GitSnapshotter(
        working_dir=tmp_path,
        cache_repo=None,
        extra_gitignore=["*.tmp"],
    )
    snapshotter.create_empty_snapshot("seed")
    tracked = tmp_path / "query_impl.cpp"
    tracked.write_text("v1\n", encoding="utf-8")
    snapshotter.snapshot("seed")
    leftover = tmp_path / "storage_plan.txt"
    leftover.write_text("plan\n", encoding="utf-8")

    snapshotter.recreate_repo()

    assert snapshotter.is_dirty() is False
    assert tracked.exists() is False
    assert leftover.exists() is False
    assert _run_git(tmp_path, "rev-parse", "--verify", "HEAD").returncode != 0
    assert _run_git(tmp_path, "show-ref", "--head").returncode != 0
    assert "*.tmp" in (tmp_path / ".git" / "info" / "exclude").read_text(encoding="utf-8")


def test_prepare_runtime_workspace_recreates_repo_for_fresh_root_run(tmp_path) -> None:
    snapshotter = GitSnapshotter(working_dir=tmp_path, cache_repo=None)
    snapshotter.create_empty_snapshot("seed")
    target = tmp_path / "storage_plan.txt"
    target.write_text("plan\n", encoding="utf-8")

    tpch_monetdb.runtime_workspace._prepare_runtime_workspace(
        snapshotter,
        tmp_path,
        reset_git_history=True,
    )

    assert snapshotter.is_dirty() is False
    assert target.exists() is False
    assert _run_git(tmp_path, "rev-parse", "--verify", "HEAD").returncode != 0


def test_prepare_runtime_workspace_continue_run_keeps_existing_state(tmp_path) -> None:
    snapshotter = GitSnapshotter(working_dir=tmp_path, cache_repo=None)
    snapshotter.create_empty_snapshot("seed")
    target = tmp_path / "storage_plan.txt"
    target.write_text("plan\n", encoding="utf-8")

    tpch_monetdb.runtime_workspace._prepare_runtime_workspace(
        snapshotter,
        tmp_path,
        continue_run=True,
        reset_git_history=True,
    )

    assert target.exists() is True
    assert snapshotter.is_dirty() is True
