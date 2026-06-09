from pathlib import Path
from typing import Any, Callable

from tpch_monetdb.dataset.gen_tpch.gen_tpch_query import gen_query as gen_tpch_query


QueryGenerator = Callable[..., tuple[str, str, dict[str, Any]]]
PlaceholderGenerator = Callable[..., dict[str, Any]]

QUERY_GENERATORS: dict[str, QueryGenerator] = {
    "tpch": gen_tpch_query,
}


def get_query_gen(benchmark: str) -> QueryGenerator:
    """Return the SQL generator for a supported benchmark."""
    if benchmark not in QUERY_GENERATORS:
        raise ValueError(f"Unknown benchmark: {benchmark}")
    return QUERY_GENERATORS[benchmark]


def get_placeholders_fn(benchmark: str, cache_dir: Path | None = None) -> PlaceholderGenerator:
    """Return a placeholder-only generator for a supported benchmark."""
    del cache_dir
    gen_query = get_query_gen(benchmark)

    def gen_placeholder(**kwargs: Any) -> dict[str, Any]:
        """Generate only placeholders while preserving generator kwargs."""
        return gen_query(**kwargs)[2]

    return gen_placeholder
