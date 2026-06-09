from tpch_monetdb.tools.tpch.process_tree import collect_process_tree_pids


def test_collect_process_tree_pids_reads_proc_children(tmp_path) -> None:
    def write_children(pid: int, tid: int, children: str) -> None:
        children_path = tmp_path / str(pid) / "task" / str(tid) / "children"
        children_path.parent.mkdir(parents=True, exist_ok=True)
        children_path.write_text(children, encoding="utf-8")
        return None

    write_children(100, 100, "101 102\n")
    write_children(101, 101, "103\n")
    write_children(102, 102, "\n")
    write_children(103, 103, "\n")

    assert collect_process_tree_pids(100, proc_root=tmp_path) == [100, 101, 103, 102]
    return None
