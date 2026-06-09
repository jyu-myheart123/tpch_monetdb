import logging
import os
import re
from pathlib import Path
from typing import Iterable, List, Optional

from tpch_monetdb.dataset.dataset_tables_dict import get_tables_for_benchmark
from tpch_monetdb.llm_cache.git_snapshotter import GitSnapshotter
from tpch_monetdb.misc.tpch.compiler_cached import CachedCompiler

logger = logging.getLogger(__name__)


def _tpch_monetdb_misc_root() -> Path:
    project_dir = Path(__file__).parents[3]
    return project_dir / "tpch_monetdb" / "misc" / "tpch"


def _tpch_monetdb_layout_paths(
    api_path: Optional[Path] = None,
) -> tuple[Path, Path, Path]:
    root = _tpch_monetdb_misc_root() if api_path is None else api_path
    if root.name in {"templates", "support", "runtime"}:
        root = root.parent
    templates_dir = root / "templates"
    support_dir = root / "support"
    runtime_dir = root / "runtime"
    return templates_dir, support_dir, runtime_dir


def _gen_table_defs(tables: List[str]) -> str:
    return ""


def _gen_table_reads(tables: List[str]) -> str:
    return ""


_DISPATCHER_IMPL_FILES: tuple[str, ...] = (
    "loader_impl.hpp",
    "loader_impl.cpp",
    "builder_impl.hpp",
    "builder_impl.cpp",
    "query_impl.hpp",
    "query_impl.cpp",
)

_SUPPORT_API_FILES: tuple[str, ...] = (
    "loader_api.hpp",
    "builder_api.hpp",
    "query_api.hpp",
)
_GENERATED_QUERY_REGISTRY_HEADER = Path("build/generated/query_registry_generated.hpp")
_GENERATED_QUERY_REGISTRY_SOURCE = Path("build/generated/query_registry_generated.cpp")
_QUERY_IDS: tuple[str, ...] = (
    "1", "2", "3", "4", "5", "6", "7",
    "8", "9", "10", "11", "12", "13", "14", "15",
    "16", "17", "18", "19", "20", "21", "22",
)
_FORBIDDEN_QUERY_MODULE_DISPATCH_PATTERN = re.compile(
    r"\b(dispatch_query|dispatch_unimplemented_query)\s*\("
)
_FORBIDDEN_SHARED_QUERY_ENTRYPOINT_PATTERN = re.compile(r"\bexecute_q\d+\s*\(")
_REQUESTED_QUERY_RE = re.compile(r"^Query\s+Q?(\d+):", re.MULTILINE)


def _discover_companion_query_files(templates_dir: Path) -> list[str]:
    """扫描模板目录下已有的 companion query source / header."""
    companions: list[str] = []
    for pattern in ("query_*.cpp", "query_*.hpp"):
        for path in sorted(templates_dir.glob(pattern)):
            if path.name in _DISPATCHER_IMPL_FILES:
                continue
            companions.append(path.name)
    return companions


def _normalize_query_id_token(query_id: object) -> str:
    """Normalize query ids to the numeric suffix used by query_q*.cpp files."""
    value = str(query_id).strip()
    if value[:1].lower() == "q":
        value = value[1:]
    return value


def _query_module_pair_exists(cwd: Path, query_id: str) -> bool:
    normalized_query_id = _normalize_query_id_token(query_id)
    header = cwd / f"query_q{normalized_query_id}.hpp"
    source = cwd / f"query_q{normalized_query_id}.cpp"
    return header.is_file() and source.is_file()


def _read_requested_query_ids(cwd: Path) -> tuple[str, ...]:
    """Read query ids from the host-generated queries.txt artifact when present."""
    queries_path = cwd / "queries.txt"
    if not queries_path.is_file():
        return ()
    query_text = queries_path.read_text(encoding="utf-8")
    return tuple(dict.fromkeys(_REQUESTED_QUERY_RE.findall(query_text)))


def _normalize_required_query_ids(
    query_ids: Iterable[str] | None,
) -> tuple[str, ...]:
    """Normalize an optional query-id scope for generated registry validation."""
    if query_ids is None:
        return ()
    normalized: list[str] = []
    for query_id in query_ids:
        value = _normalize_query_id_token(query_id)
        if value:
            normalized.append(value)
    return tuple(dict.fromkeys(normalized))


def _validate_requested_query_modules(
    cwd: Path,
    required_query_ids: Iterable[str] | None = None,
) -> None:
    """Fail fast when required query entrypoints are missing."""
    requested_query_ids = _normalize_required_query_ids(required_query_ids)
    if not requested_query_ids:
        requested_query_ids = _read_requested_query_ids(cwd)
    missing = [
        query_id for query_id in requested_query_ids
        if not _query_module_pair_exists(cwd, query_id)
    ]
    if missing:
        raise RuntimeError(
            "Generated query registry missing entrypoints for requested queries: "
            + ", ".join(f"Q{query_id}" for query_id in missing)
        )
    return None


def _validate_query_module_boundaries(cwd: Path) -> None:
    """Reject dispatcher or query-entrypoint ownership inside focused modules."""
    offenders: list[str] = []
    for path in sorted(cwd.glob("query_q*.cpp")):
        content = path.read_text(encoding="utf-8")
        if _FORBIDDEN_QUERY_MODULE_DISPATCH_PATTERN.search(content):
            offenders.append(path.name)
    for path in sorted(cwd.glob("query_shared_*.cpp")):
        content = path.read_text(encoding="utf-8")
        if (
            _FORBIDDEN_QUERY_MODULE_DISPATCH_PATTERN.search(content)
            or _FORBIDDEN_SHARED_QUERY_ENTRYPOINT_PATTERN.search(content)
        ):
            offenders.append(path.name)
    if offenders:
        joined = ", ".join(offenders)
        raise RuntimeError(
            "Focused query modules must not define dispatcher symbols or "
            f"query entrypoints in shared files; move routing/entrypoint logic out of: {joined}"
        )
    return None


def _render_query_registry_header() -> str:
    lines = [
        "#pragma once",
        "",
        '#include "builder_impl.hpp"',
        '#include "args_parser.hpp"',
        "",
        "void dispatch_query(Engine& engine, const QueryRequest& request);",
        "",
    ]
    return "\n".join(lines)


def _render_query_registry_source(cwd: Path) -> str:
    """Generate a registry for implemented query modules without runtime fallbacks.

    Missing query modules are intentionally not given a dispatch branch. If a run asks
    for such a query, the registry fails fast instead of routing through a placeholder.
    """
    implemented_query_ids = tuple(
        query_id for query_id in _QUERY_IDS
        if _query_module_pair_exists(cwd, query_id)
    )
    lines = [
        '#include "query_registry_generated.hpp"',
        "",
        "#include <stdexcept>",
        "#include <string>",
        "",
    ]
    for query_id in implemented_query_ids:
        lines.append(f'#include "query_q{query_id}.hpp"')
    lines.extend(
        [
            "",
            "namespace {",
            "std::string normalize_query_id(const std::string& query_id) {",
            "    if (query_id.size() > 1 && (query_id[0] == 'Q' || query_id[0] == 'q')) {",
            "        return query_id.substr(1);",
            "    }",
            "    return query_id;",
            "}",
            "}",
            "",
            "void dispatch_query(Engine& engine, const QueryRequest& request) {",
            "    const std::string normalized_query_id = normalize_query_id(request.id);",
        ]
    )
    for idx, query_id in enumerate(implemented_query_ids):
        prefix = "if" if idx == 0 else "else if"
        lines.append(f'    {prefix} (normalized_query_id == "{query_id}") {{')
        lines.append(f"        const auto args = parse_q{query_id}(request);")
        lines.append(f"        execute_q{query_id}(engine, args);")
        lines.append("        return;")
        lines.append("    }")
    lines.extend(
        [
            '    throw std::runtime_error("No generated query entrypoint for query " + request.id);',
            "}",
            "",
        ]
    )
    return "\n".join(lines)


def _write_if_changed(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_text(encoding="utf-8") == content:
        return None
    path.write_text(content, encoding="utf-8")
    return None


def _ensure_query_registry_generated(
    cwd: Path,
    *,
    validate_requested_query_modules: bool = False,
    required_query_ids: Iterable[str] | None = None,
) -> tuple[Path, Path]:
    """在 build/generated 下生成 query registry 文件."""
    _validate_query_module_boundaries(cwd)
    if validate_requested_query_modules:
        _validate_requested_query_modules(cwd, required_query_ids)
    header_path = cwd / _GENERATED_QUERY_REGISTRY_HEADER
    source_path = cwd / _GENERATED_QUERY_REGISTRY_SOURCE
    _write_if_changed(header_path, _render_query_registry_header())
    _write_if_changed(source_path, _render_query_registry_source(cwd))
    return header_path, source_path


def copy_template_to(destination_dir: Path, benchmark: str) -> str:
    """将 TPC-H MonetDB 运行时骨架模板复制到工作空间."""
    assert destination_dir.exists()

    templates_dir, support_dir, _ = _tpch_monetdb_layout_paths()
    files = list(_DISPATCHER_IMPL_FILES)
    if benchmark == "tpch":
        files.extend(_discover_companion_query_files(templates_dir))
    support_files = list(_SUPPORT_API_FILES)

    tables = get_tables_for_benchmark(benchmark)

    content = ""
    for filename in files:
        source = templates_dir / filename

        if not source.is_file():
            raise FileNotFoundError(f"Source file not found: {source}")

        file_content = source.read_text()

        if filename == "loader_impl.hpp":
            file_content = replace_cpp_marked_block(
                file_content, "table-defs", _gen_table_defs(tables)
            )
        elif filename == "loader_impl.cpp":
            file_content = replace_cpp_marked_block(
                file_content, "table-reads", _gen_table_reads(tables)
            )

        content += f"// ---- {filename} ----\n"
        content += file_content + "\n\n"

        dest = destination_dir / filename
        logger.info(f"Writing {filename} to {dest}")
        dest.write_text(file_content)

    for filename in support_files:
        source = support_dir / filename

        if not source.is_file():
            raise FileNotFoundError(f"Source file not found: {source}")

        file_content = source.read_text()

        # assemble string containing content of copied files - for versioning / snapshotting
        content += f"// ---- {filename} ----\n"
        content += file_content + "\n\n"

        dest = destination_dir / filename
        logger.info(f"Writing {filename} to {dest}")
        dest.write_text(file_content)

    return content


def replace_cpp_marked_block(text, marker_name, replacement):
    name = re.escape(marker_name)

    pattern = re.compile(
        rf"""(?ms)
        ^[ \t]*//[ \t]*start:[ \t]*{name}[ \t]*\r?\n?
        .*?
        ^[ \t]*//[ \t]*end:[ \t]*{name}[ \t]*(?:\r?\n|$)
        """,
        re.VERBOSE,
    )

    if replacement and not replacement.endswith(("\n", "\r\n")):
        replacement += "\n"

    result, n = pattern.subn(replacement, text, count=1)

    if n != 1:
        raise ValueError(f"expected exactly one replacement, got {n}")

    return result


def relpath(target: Path, base: Path) -> Path:
    return Path(os.path.relpath(target, base))


def _collect_companion_query_sources(cwd: Path) -> list[str]:
    """在 workspace 根下收集 companion query `.cpp` 文件名，供 query library 编译使用."""
    companions: list[str] = []
    for path in sorted(cwd.glob("query_*.cpp")):
        if path.name == "query_impl.cpp":
            continue
        companions.append(path.name)
    return companions


def make_compiler(
    cwd: Path,
    compile_cache_dir: Optional[Path] = None,
    git_snapshotter: Optional[GitSnapshotter] = None,
    api_path: Optional[Path] = None,
    validate_requested_query_modules: bool = False,
    required_query_ids: Iterable[str] | None = None,
) -> CachedCompiler:
    """构造 TPC-H MonetDB 编译器；query library 自动纳入 companion query `.cpp` 文件."""
    generated_header, generated_source = _ensure_query_registry_generated(
        cwd,
        validate_requested_query_modules=validate_requested_query_modules,
        required_query_ids=required_query_ids,
    )
    if api_path is None:
        layout_root = _tpch_monetdb_misc_root()
    else:
        layout_root = api_path if api_path.is_absolute() else cwd / api_path
    _, support_dir, runtime_dir = _tpch_monetdb_layout_paths(layout_root)
    rel_support_dir = relpath(support_dir, cwd.resolve())
    rel_runtime_dir = relpath(runtime_dir, cwd.resolve())
    rel_generated_dir = relpath(generated_header.parent, cwd.resolve())
    rel_generated_source = relpath(generated_source, cwd.resolve())

    query_sources: list = [
        rel_runtime_dir / "query_api.cpp",
        "query_impl.cpp",
        rel_generated_source,
    ]
    for companion in _collect_companion_query_sources(cwd):
        query_sources.append(companion)

    args = dict(
        working_dir=cwd,
        libs={
            "loader": [
                rel_runtime_dir / "loader_api.cpp",
                "loader_impl.cpp",
                rel_runtime_dir / "loader_utils.cpp",
            ],  # for now do not share the loader_impl
            "builder": [rel_runtime_dir / "builder_api.cpp", "builder_impl.cpp"],
            "query": query_sources,
        },
        main_src=rel_runtime_dir / "db.cpp",
        include_dirs=[cwd.resolve(), rel_support_dir, rel_runtime_dir, rel_generated_dir],
        app_extra_srcs=[rel_runtime_dir / "utils/build_id.cpp"],
        build_dir="build",
        link_libs=[],
        pkgconfig_libs=[],  # TODO: TPC-H MonetDB - no Arrow/Parquet dependency in Phase 1
    )
    return CachedCompiler(
        args=args,
        compile_cache_dir=compile_cache_dir,
        git_snapshotter=git_snapshotter,
    )
