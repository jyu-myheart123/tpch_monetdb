from __future__ import annotations

import fnmatch
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_RULE_TOKEN_BUDGET = 2000
APPROX_CHARS_PER_TOKEN = 3
_FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n?", re.DOTALL)


@dataclass(frozen=True)
class RuleAssembly:
    text: str
    included_files: tuple[str, ...]
    truncated_files: tuple[str, ...]
    excluded_files: tuple[str, ...]
    char_budget: int

    @property
    def was_truncated(self) -> bool:
        value = bool(self.truncated_files or self.excluded_files)
        return value


@dataclass(frozen=True)
class RuleScope:
    stage_name: str | None
    area_name: str | None
    candidate_paths: tuple[str, ...] = ()


def load_agent_rules(
    rules_dir: Path,
    *,
    scope: RuleScope,
    include_global: bool,
    token_budget: int = DEFAULT_RULE_TOKEN_BUDGET,
) -> RuleAssembly:
    char_budget = token_budget * APPROX_CHARS_PER_TOKEN
    if not rules_dir.is_dir():
        return RuleAssembly(
            text="",
            included_files=(),
            truncated_files=(),
            excluded_files=(),
            char_budget=char_budget,
        )

    selected: list[tuple[int, str, str]] = []
    for rule_file in sorted(rules_dir.glob("*.md")):
        content = rule_file.read_text(encoding="utf-8")
        metadata, body = _split_rule_content(content)
        if _matches_scope(
            metadata=metadata,
            scope=scope,
            include_global=include_global,
        ):
            selected.append(
                (
                    _rule_priority(metadata),
                    rule_file.name,
                    body.strip(),
                )
            )
    selected.sort(key=lambda item: (item[0], item[1]))

    if not selected:
        return RuleAssembly(
            text="",
            included_files=(),
            truncated_files=(),
            excluded_files=(),
            char_budget=char_budget,
        )

    parts: list[str] = []
    included_files: list[str] = []
    truncated_files: list[str] = []
    excluded_files: list[str] = []
    current_length = 0

    for selected_index, (_priority, filename, body) in enumerate(selected):
        if not body:
            continue
        chunk = _render_rule_chunk(filename, body)
        separator = "\n\n" if parts else ""
        next_length = current_length + len(separator) + len(chunk)
        if next_length <= char_budget:
            if separator:
                parts.append(separator)
            parts.append(chunk)
            included_files.append(filename)
            current_length = next_length
            continue

        remaining = char_budget - current_length - len(separator)
        if remaining > 20:
            truncated_chunk = chunk[:remaining].rstrip() + "\n[rule truncated]"
            if separator:
                parts.append(separator)
            parts.append(truncated_chunk)
            included_files.append(filename)
            truncated_files.append(filename)
        else:
            excluded_files.append(filename)
        for _later_priority, later_filename, _later_body in selected[selected_index + 1 :]:
            excluded_files.append(later_filename)
        break

    text = "".join(parts).strip()
    return RuleAssembly(
        text=text,
        included_files=tuple(included_files),
        truncated_files=tuple(dict.fromkeys(truncated_files)),
        excluded_files=tuple(dict.fromkeys(excluded_files)),
        char_budget=char_budget,
    )


def log_rule_assembly(scope_label: str, assembly: RuleAssembly) -> None:
    if not assembly.included_files:
        logger.info("Rule assembly (%s): no matching rules loaded", scope_label)
        return None
    logger.info(
        "Rule assembly (%s): loaded=%s budget_chars=%s",
        scope_label,
        ", ".join(assembly.included_files),
        assembly.char_budget,
    )
    if assembly.was_truncated:
        logger.warning(
            "Rule assembly (%s): truncated=%s excluded=%s",
            scope_label,
            ", ".join(assembly.truncated_files) if assembly.truncated_files else "(none)",
            ", ".join(assembly.excluded_files) if assembly.excluded_files else "(none)",
        )
    return None


def _render_rule_chunk(filename: str, body: str) -> str:
    value = f"[Rule File: {filename}]\n{body}"
    return value


def _split_rule_content(content: str) -> tuple[dict[str, tuple[str, ...]], str]:
    match = _FRONTMATTER_RE.match(content)
    if match is None:
        return {}, content
    metadata = _parse_frontmatter(match.group(1))
    body = content[match.end() :]
    return metadata, body


def _parse_frontmatter(frontmatter: str) -> dict[str, tuple[str, ...]]:
    parsed: dict[str, tuple[str, ...]] = {}
    for raw_line in frontmatter.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, raw_value = line.split(":", 1)
        parsed[key.strip()] = _parse_list_like_value(raw_value.strip())
    return parsed


def _parse_list_like_value(raw_value: str) -> tuple[str, ...]:
    if raw_value in ("", "[]"):
        return ()
    if raw_value.startswith("[") and raw_value.endswith("]"):
        inner = raw_value[1:-1].strip()
        if not inner:
            return ()
        values = tuple(
            item.strip().strip("'\"")
            for item in inner.split(",")
            if item.strip()
        )
        return values
    return (raw_value.strip().strip("'\""),)


def _rule_priority(metadata: dict[str, tuple[str, ...]]) -> int:
    raw_priority = metadata.get("priority", ())
    if not raw_priority:
        return 1000
    try:
        return int(raw_priority[0])
    except ValueError:
        return 1000


def _matches_scope(
    *,
    metadata: dict[str, tuple[str, ...]],
    scope: RuleScope,
    include_global: bool,
) -> bool:
    stages = metadata.get("stages", ())
    if stages:
        if scope.stage_name is None or scope.stage_name not in stages:
            return False
    elif not include_global:
        return False

    areas = metadata.get("areas", ())
    if areas:
        if scope.area_name is None or scope.area_name not in areas:
            return False

    paths = metadata.get("paths", ())
    if paths:
        if not scope.candidate_paths:
            return False
        if not any(
            fnmatch.fnmatch(candidate_path, pattern)
            for candidate_path in scope.candidate_paths
            for pattern in paths
        ):
            return False

    return True
