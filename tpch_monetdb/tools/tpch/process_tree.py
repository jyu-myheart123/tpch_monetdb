from __future__ import annotations

from pathlib import Path


def collect_process_tree_pids(
    root_pid: int,
    *,
    proc_root: Path = Path("/proc"),
) -> list[int]:
    if root_pid <= 0:
        raise ValueError("root_pid must be positive")
    seen: set[int] = set()
    ordered: list[int] = []
    stack = [root_pid]
    while stack:
        pid = stack.pop()
        if pid in seen:
            continue
        seen.add(pid)
        ordered.append(pid)
        children = _read_proc_children(pid, proc_root=proc_root)
        stack.extend(reversed([child for child in children if child not in seen]))
    return ordered


def _read_proc_children(
    pid: int,
    *,
    proc_root: Path,
) -> list[int]:
    task_dir = proc_root / str(pid) / "task"
    if not task_dir.exists():
        return []
    children: set[int] = set()
    for children_path in sorted(task_dir.glob("*/children")):
        try:
            text = children_path.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        for value in text.split():
            try:
                child_pid = int(value)
            except ValueError:
                continue
            if child_pid > 0:
                children.add(child_pid)
    return sorted(children)
