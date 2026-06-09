"""Shared duration display and runtime safety helpers."""

from __future__ import annotations

import math


def is_positive_finite_runtime_ms(value: object) -> bool:
    """Return whether a runtime value is finite and strictly positive milliseconds."""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(numeric) and numeric > 0.0


def safe_speedup(baseline_ms: object, implementation_ms: object) -> float | None:
    """Return baseline/implementation speedup or None for invalid runtimes."""
    if not is_positive_finite_runtime_ms(baseline_ms):
        return None
    if not is_positive_finite_runtime_ms(implementation_ms):
        return None
    return float(baseline_ms) / float(implementation_ms)


def format_duration_ms(value: object) -> str:
    """Format milliseconds using units that do not collapse small values to 0.00."""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "invalid"
    if not math.isfinite(numeric):
        return "invalid"
    sign = "-" if numeric < 0 else ""
    abs_ms = abs(numeric)
    if abs_ms == 0:
        return "0ns"
    if abs_ms < 0.001:
        return f"{sign}{abs_ms * 1_000_000:.0f}ns"
    if abs_ms < 1.0:
        return f"{sign}{abs_ms * 1_000:.2f}us"
    if abs_ms < 1000.0:
        return f"{sign}{abs_ms:.3f}ms"
    return f"{sign}{abs_ms / 1000.0:.3f}s"
