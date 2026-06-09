from __future__ import annotations

from tpch_monetdb.tools.stage_tool_policy import StageRunSummary


def render_stage_summary(summary: StageRunSummary) -> str:
    descriptor = summary.prompt_descriptor or summary.profile_name
    lines = [
        "[Stage Summary]",
        f"Stage: {summary.profile_name}",
        f"Descriptor: {descriptor}",
        f"Files changed: {', '.join(summary.written_files) if summary.written_files else '(none)'}",
        f"Control artifacts read: {', '.join(summary.control_artifacts_read) if summary.control_artifacts_read else '(none)'}",
        f"Tool counts: {_render_tool_counts(summary)}",
        f"Last compile result: {summary.last_compile_summary or '(none)'}",
        f"Last run result: {summary.last_run_summary or '(none)'}",
        f"Last validate result: {summary.last_validation_summary or '(none)'}",
        f"Compile succeeded: {_render_tri_state(summary.compile_succeeded)}",
        f"Run succeeded: {_render_tri_state(summary.run_succeeded)}",
        f"Validation passed: {_render_tri_state(summary.validation_passed)}",
        f"Last failure kind: {summary.last_failure_kind or '(none)'}",
        f"TODO progress: {_render_todo_progress(summary)}",
        f"Current blocker: {_infer_blocker(summary)}",
    ]
    if summary.final_output:
        lines.append(f"Model final output: {summary.final_output.strip()}")
    return "\n".join(lines)


def _render_tool_counts(summary: StageRunSummary) -> str:
    if not summary.tool_counts:
        return "(none)"
    pairs = [f"{tool}={count}" for tool, count in sorted(summary.tool_counts.items())]
    return ", ".join(pairs)


def _render_todo_progress(summary: StageRunSummary) -> str:
    if summary.todo_after is None:
        return "(TODO.md unavailable)"
    return (
        f"completed={summary.todo_after.completed_count}, "
        f"in_progress={summary.todo_after.in_progress_count}, "
        f"pending={summary.todo_after.pending_count}"
    )


def _render_tri_state(value: bool | None) -> str:
    if value is None:
        return "(unknown)"
    return "yes" if value else "no"


def _infer_blocker(summary: StageRunSummary) -> str:
    if summary.last_failure_kind == "validation" and summary.last_validation_summary:
        return summary.last_validation_summary.strip()
    for candidate in (
        summary.last_validation_summary,
        summary.last_compile_summary,
        summary.last_run_summary,
        summary.final_output,
    ):
        if not candidate:
            continue
        normalized = candidate.lower()
        if any(marker in normalized for marker in ("error", "failed", "exception")):
            return candidate.strip()
    if not summary.written_files and not summary.todo_progressed:
        return "No file changes or TODO progress recorded in this stage."
    return "(none)"
