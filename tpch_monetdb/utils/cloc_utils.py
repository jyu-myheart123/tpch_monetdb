import json
import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

# File types that are excluded from LOC counts (non-code assets)
_EXCLUDED_FILE_TYPES: frozenset[str] = frozenset({"SUM", "header", "SUM!", "Text", "JSON", "Markdown"})

# Extension-level grouping: cloc file_type -> canonical breakdown key
_EXT_GROUP: dict[str, str] = {
    "C++": "cpp",
    "C/C++ Header": "hpp",
    "Python": "py",
}


def calculate_loc(
    cloc_cache_dir: Path | None, current_hash: str, working_dir: Path
) -> int:
    """Return total LOC (code lines) excluding prose / data file types."""
    return _run_and_cache_loc(cloc_cache_dir, current_hash, working_dir).get("total", 0)


def calculate_loc_breakdown(
    cloc_cache_dir: Path | None,
    current_hash: str,
    working_dir: Path,
) -> dict[str, int]:
    """Return LOC by extension group plus total.

    Returns a dict with keys: "total", "cpp", "hpp", "py", and any other
    canonical keys derived from _EXT_GROUP.  Unknown file types are
    accumulated under "other".
    """
    return _run_and_cache_loc(cloc_cache_dir, current_hash, working_dir)


def _run_and_cache_loc(
    cloc_cache_dir: Path | None,
    current_hash: str,
    working_dir: Path,
) -> dict[str, int]:
    """Run cloc and return a breakdown dict, using a cache keyed by snapshot hash."""
    from tpch_monetdb.llm_cache import utils  # lazy import to avoid circular dep

    if cloc_cache_dir is not None:
        payload = {"snapshot_hash": current_hash, "v": 2}  # v=2 for breakdown format
        hash_value = utils.sha256(utils.stable_json(payload))
        cache_path = _cache_path_for(cloc_cache_dir, hash_value)
        if cache_path.exists():
            output = utils.load_pickle(cache_path, expected=dict)
            if output is not None:
                return output
    else:
        cache_path = None

    result = subprocess.run(
        "cloc . --json", shell=True, cwd=working_dir, capture_output=True, text=True
    )
    if result.returncode != 0:
        logger.error(f"Error running cloc: {result.stderr}")
        return {"total": 0}
    count_stats = result.stdout.strip()
    if not count_stats:
        return {"total": 0}

    breakdown: dict[str, int] = {"total": 0, "cpp": 0, "hpp": 0, "py": 0, "other": 0}
    for file_type, stats in json.loads(count_stats).items():
        if file_type in _EXCLUDED_FILE_TYPES:
            continue
        code_lines = stats.get("code", 0)
        breakdown["total"] += code_lines
        key = _EXT_GROUP.get(file_type, "other")
        breakdown[key] = breakdown.get(key, 0) + code_lines

    if cache_path is not None:
        from tpch_monetdb.llm_cache import utils as _utils
        _utils.dump_pickle(cache_path, breakdown)
    return breakdown


def _cache_path_for(cloc_cache_dir: Path, hash_value: str) -> Path:
    return cloc_cache_dir / f"{hash_value}.pkl"

