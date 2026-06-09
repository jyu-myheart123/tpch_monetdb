from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "copy_template_to": ("tpch_monetdb.tools.tpch.utils", "copy_template_to"),
    "make_compile_tool": ("tpch_monetdb.tools.tpch.compile", "make_compile_tool"),
    "make_run_tool": ("tpch_monetdb.tools.tpch.run", "make_run_tool"),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    """Load TPC-H MonetDB tool exports lazily to avoid unnecessary agent dependencies."""
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = _EXPORTS[name]
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value
