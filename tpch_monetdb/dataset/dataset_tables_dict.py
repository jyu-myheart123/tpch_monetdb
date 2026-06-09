from tpch_monetdb.dataset.gen_tpch.tpch_queries import TPCH_TABLES, tpc_h_schema

TABLES_LIST = {
    "tpch": list(TPCH_TABLES),
}

DATASET_NAMES = {
    "tpch": "tpch",
}

BENCHMARK_SCHEMAS = {
    "tpch": tpc_h_schema,
}


def get_tables_for_benchmark(benchmark: str) -> list[str]:
    """Return the declared table list for a supported benchmark."""
    if benchmark not in TABLES_LIST:
        raise ValueError(f"Unknown benchmark {benchmark}")
    return list(TABLES_LIST[benchmark])


def get_dataset_name(benchmark: str) -> str:
    """Return the dataset name used by runtime loaders for a benchmark."""
    if benchmark not in DATASET_NAMES:
        raise ValueError(f"Unknown benchmark {benchmark}")
    return DATASET_NAMES[benchmark]


def get_benchmark_schema(benchmark: str) -> str:
    """Return the SQL schema text for a supported benchmark."""
    if benchmark not in BENCHMARK_SCHEMAS:
        raise ValueError(f"Unknown benchmark {benchmark}")
    return BENCHMARK_SCHEMAS[benchmark]
