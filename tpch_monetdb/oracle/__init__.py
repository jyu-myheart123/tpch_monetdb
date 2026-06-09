"""Oracle adapters and result helpers."""

__all__ = [
    "MonetDBConnectionConfig",
    "MonetDBOracle",
    "TpchQueryResult",
    "TpchRuntimeValidator",
    "TpchValidator",
    "compare_tpch_results",
    "compare_results",
    "ComparisonReport",
]


def __getattr__(name: str):
    """Load oracle exports lazily so independent adapters do not inherit old dependencies."""
    if name == "MonetDBConnectionConfig":
        from .monetdb_oracle import MonetDBConnectionConfig

        return MonetDBConnectionConfig
    if name == "MonetDBOracle":
        from .monetdb_oracle import MonetDBOracle

        return MonetDBOracle
    if name == "TpchQueryResult":
        from .result import TpchQueryResult

        return TpchQueryResult
    if name == "TpchRuntimeValidator":
        from .tpch_runtime_validator import TpchRuntimeValidator

        return TpchRuntimeValidator
    if name == "TpchValidator":
        from .tpch_validator import TpchValidator

        return TpchValidator
    if name == "compare_tpch_results":
        from .tpch_validator import compare_tpch_results

        return compare_tpch_results
    if name == "compare_results":
        from .comparator import compare_results

        return compare_results
    if name == "ComparisonReport":
        from .comparator import ComparisonReport

        return ComparisonReport
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
