"""TPC-H Q1-Q22 parameter sampling and SQL instantiation."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import random
from typing import Any, Callable

from tpch_monetdb.dataset.gen_tpch.tpch_queries import (
    PARAMETER_PATTERN,
    get_contract,
)


REGIONS: tuple[str, ...] = (
    "AFRICA",
    "AMERICA",
    "ASIA",
    "EUROPE",
    "MIDDLE EAST",
)

NATIONS: tuple[str, ...] = (
    "ALGERIA",
    "ARGENTINA",
    "BRAZIL",
    "CANADA",
    "EGYPT",
    "ETHIOPIA",
    "FRANCE",
    "GERMANY",
    "INDIA",
    "INDONESIA",
    "IRAN",
    "IRAQ",
    "JAPAN",
    "JORDAN",
    "KENYA",
    "MOROCCO",
    "MOZAMBIQUE",
    "PERU",
    "CHINA",
    "ROMANIA",
    "SAUDI ARABIA",
    "VIETNAM",
    "RUSSIA",
    "UNITED KINGDOM",
    "UNITED STATES",
)

SEGMENTS: tuple[str, ...] = (
    "AUTOMOBILE",
    "BUILDING",
    "FURNITURE",
    "HOUSEHOLD",
    "MACHINERY",
)

SHIP_MODES: tuple[str, ...] = (
    "AIR",
    "AIR REG",
    "RAIL",
    "SHIP",
    "TRUCK",
    "MAIL",
    "FOB",
)

CONTAINERS: tuple[str, ...] = (
    "SM CASE",
    "SM BOX",
    "SM PACK",
    "SM PKG",
    "MED BAG",
    "MED BOX",
    "MED PACK",
    "MED PKG",
    "LG CASE",
    "LG BOX",
    "LG PACK",
    "LG PKG",
)

COLORS: tuple[str, ...] = (
    "almond",
    "antique",
    "aquamarine",
    "azure",
    "beige",
    "bisque",
    "black",
    "blanched",
    "blue",
    "blush",
    "brown",
    "burlywood",
    "burnished",
    "chartreuse",
    "chiffon",
    "chocolate",
    "coral",
    "cornflower",
    "cornsilk",
    "cream",
    "cyan",
    "dark",
    "deep",
    "dim",
    "dodger",
    "drab",
    "firebrick",
    "floral",
    "forest",
    "frosted",
    "gainsboro",
    "ghost",
    "goldenrod",
    "green",
    "grey",
    "honeydew",
    "hot",
    "indian",
    "ivory",
    "khaki",
    "lace",
    "lavender",
    "lawn",
    "lemon",
    "light",
    "lime",
    "linen",
    "magenta",
    "maroon",
    "medium",
    "metallic",
    "midnight",
    "mint",
    "misty",
    "moccasin",
    "navajo",
    "navy",
    "olive",
    "orange",
    "orchid",
    "pale",
    "papaya",
    "peach",
    "peru",
    "pink",
    "plum",
    "powder",
    "puff",
    "purple",
    "red",
    "rose",
    "rosy",
    "royal",
    "saddle",
    "salmon",
    "sandy",
    "seashell",
    "sienna",
    "sky",
    "slate",
    "smoke",
    "snow",
    "spring",
    "steel",
    "tan",
    "thistle",
    "tomato",
    "turquoise",
    "violet",
    "wheat",
    "white",
    "yellow",
)

TYPE_SYLLABLE1: tuple[str, ...] = ("STANDARD", "SMALL", "MEDIUM", "LARGE", "ECONOMY", "PROMO")
TYPE_SYLLABLE2: tuple[str, ...] = ("ANODIZED", "BURNISHED", "PLATED", "POLISHED", "BRUSHED")
TYPE_SYLLABLE3: tuple[str, ...] = ("TIN", "NICKEL", "BRASS", "STEEL", "COPPER")
WORD1_OPTIONS: tuple[str, ...] = ("special", "pending", "unusual", "express")
WORD2_OPTIONS: tuple[str, ...] = ("packages", "requests", "accounts", "deposits")
COUNTRY_CODES: tuple[str, ...] = ("13", "31", "23", "29", "30", "18", "17")
SCALE_FACTOR = 1.0

Sampler = Callable[[random.Random, float], dict[str, Any]]


def _random_date(rnd: random.Random, start: dt.date, end: dt.date) -> dt.date:
    """Return a deterministic random date between inclusive bounds."""
    delta_days = (end - start).days
    return start + dt.timedelta(days=rnd.randint(0, delta_days))


def _random_month_start(
    rnd: random.Random,
    start_year: int,
    start_month: int,
    end_year: int,
    end_month: int,
) -> dt.date:
    """Return the first day of a random month between inclusive bounds."""
    start_index = start_year * 12 + (start_month - 1)
    end_index = end_year * 12 + (end_month - 1)
    month_index = rnd.randint(start_index, end_index)
    year = month_index // 12
    month = month_index % 12 + 1
    return dt.date(year, month, 1)


def _random_brand(rnd: random.Random) -> str:
    """Return a TPC-H style random brand token."""
    return f"Brand#{rnd.randint(1, 5)}{rnd.randint(1, 5)}"


def _random_type_full(rnd: random.Random) -> str:
    """Return a full TPC-H part type token."""
    return f"{rnd.choice(TYPE_SYLLABLE1)} {rnd.choice(TYPE_SYLLABLE2)} {rnd.choice(TYPE_SYLLABLE3)}"


def _random_type_prefix(rnd: random.Random) -> str:
    """Return a TPC-H part type prefix token."""
    return f"{rnd.choice(TYPE_SYLLABLE1)} {rnd.choice(TYPE_SYLLABLE2)}"


def _format_fraction(value: float) -> str:
    """Format a fraction without unnecessary trailing zeroes."""
    return f"{value:.6f}".rstrip("0").rstrip(".")


def _sample_q1(rnd: random.Random, _scale_factor: float) -> dict[str, Any]:
    """Generate Q1 placeholders."""
    return {"DELTA": rnd.randint(60, 120)}


def _sample_q2(rnd: random.Random, _scale_factor: float) -> dict[str, Any]:
    """Generate Q2 placeholders."""
    return {
        "SIZE": rnd.randint(1, 50),
        "TYPE": rnd.choice(TYPE_SYLLABLE3),
        "REGION": rnd.choice(REGIONS),
    }


def _sample_q3(rnd: random.Random, _scale_factor: float) -> dict[str, Any]:
    """Generate Q3 placeholders."""
    return {
        "SEGMENT": rnd.choice(SEGMENTS),
        "DATE": _random_date(rnd, dt.date(1995, 3, 1), dt.date(1995, 3, 31)).isoformat(),
    }


def _sample_q4(rnd: random.Random, _scale_factor: float) -> dict[str, Any]:
    """Generate Q4 placeholders."""
    return {"DATE": _random_month_start(rnd, 1993, 1, 1997, 10).isoformat()}


def _sample_q5(rnd: random.Random, _scale_factor: float) -> dict[str, Any]:
    """Generate Q5 placeholders."""
    return {"REGION": rnd.choice(REGIONS), "DATE": f"{rnd.randint(1993, 1997)}-01-01"}


def _sample_q6(rnd: random.Random, _scale_factor: float) -> dict[str, Any]:
    """Generate Q6 placeholders."""
    return {
        "DATE": f"{rnd.randint(1993, 1997)}-01-01",
        "DISCOUNT": f"{rnd.randint(2, 9) / 100:.2f}",
        "QUANTITY": rnd.randint(24, 25),
    }


def _sample_q7(rnd: random.Random, _scale_factor: float) -> dict[str, Any]:
    """Generate Q7 placeholders."""
    nation1, nation2 = rnd.sample(NATIONS, 2)
    return {"NATION1": nation1, "NATION2": nation2}


def _sample_q8(rnd: random.Random, _scale_factor: float) -> dict[str, Any]:
    """Generate Q8 placeholders."""
    return {
        "NATION": rnd.choice(NATIONS),
        "REGION": rnd.choice(REGIONS),
        "TYPE": _random_type_full(rnd),
    }


def _sample_q9(rnd: random.Random, _scale_factor: float) -> dict[str, Any]:
    """Generate Q9 placeholders."""
    return {"COLOR": rnd.choice(COLORS)}


def _sample_q10(rnd: random.Random, _scale_factor: float) -> dict[str, Any]:
    """Generate Q10 placeholders."""
    return {"DATE": _random_month_start(rnd, 1993, 2, 1995, 1).isoformat()}


def _sample_q11(rnd: random.Random, scale_factor: float) -> dict[str, Any]:
    """Generate Q11 placeholders."""
    return {
        "NATION": rnd.choice(NATIONS),
        "FRACTION": _format_fraction(0.0001 / scale_factor),
    }


def _sample_q12(rnd: random.Random, _scale_factor: float) -> dict[str, Any]:
    """Generate Q12 placeholders."""
    shipmode1, shipmode2 = rnd.sample(SHIP_MODES, 2)
    return {
        "SHIPMODE1": shipmode1,
        "SHIPMODE2": shipmode2,
        "DATE": f"{rnd.randint(1993, 1997)}-01-01",
    }


def _sample_q13(rnd: random.Random, _scale_factor: float) -> dict[str, Any]:
    """Generate Q13 placeholders."""
    return {"WORD1": rnd.choice(WORD1_OPTIONS), "WORD2": rnd.choice(WORD2_OPTIONS)}


def _sample_q14(rnd: random.Random, _scale_factor: float) -> dict[str, Any]:
    """Generate Q14 placeholders."""
    return {"DATE": _random_month_start(rnd, 1993, 1, 1997, 12).isoformat()}


def _sample_q15(rnd: random.Random, _scale_factor: float) -> dict[str, Any]:
    """Generate Q15 placeholders."""
    return {"DATE": _random_month_start(rnd, 1993, 1, 1997, 12).isoformat()}


def _sample_q16(rnd: random.Random, _scale_factor: float) -> dict[str, Any]:
    """Generate Q16 placeholders."""
    placeholders: dict[str, Any] = {
        "BRAND": _random_brand(rnd),
        "TYPE": _random_type_prefix(rnd),
    }
    placeholders.update(
        {f"SIZE{idx}": size for idx, size in enumerate(rnd.sample(range(1, 51), 8), start=1)}
    )
    return placeholders


def _sample_q17(rnd: random.Random, _scale_factor: float) -> dict[str, Any]:
    """Generate Q17 placeholders."""
    return {"BRAND": _random_brand(rnd), "CONTAINER": rnd.choice(CONTAINERS)}


def _sample_q18(rnd: random.Random, _scale_factor: float) -> dict[str, Any]:
    """Generate Q18 placeholders."""
    return {"QUANTITY": rnd.randint(312, 315)}


def _sample_q19(rnd: random.Random, _scale_factor: float) -> dict[str, Any]:
    """Generate Q19 placeholders."""
    return {
        "QUANTITY1": rnd.randint(1, 10),
        "QUANTITY2": rnd.randint(10, 20),
        "QUANTITY3": rnd.randint(20, 30),
        "BRAND1": _random_brand(rnd),
        "BRAND2": _random_brand(rnd),
        "BRAND3": _random_brand(rnd),
    }


def _sample_q20(rnd: random.Random, _scale_factor: float) -> dict[str, Any]:
    """Generate Q20 placeholders."""
    return {
        "COLOR": rnd.choice(COLORS),
        "DATE": f"{rnd.randint(1993, 1997)}-01-01",
        "NATION": rnd.choice(NATIONS),
    }


def _sample_q21(rnd: random.Random, _scale_factor: float) -> dict[str, Any]:
    """Generate Q21 placeholders."""
    return {"NATION": rnd.choice(NATIONS)}


def _sample_q22(rnd: random.Random, _scale_factor: float) -> dict[str, Any]:
    """Generate Q22 placeholders."""
    return {f"I{idx}": code for idx, code in enumerate(rnd.sample(COUNTRY_CODES, 7), start=1)}


PLACEHOLDER_SAMPLERS: dict[str, Sampler] = {
    "Q1": _sample_q1,
    "Q2": _sample_q2,
    "Q3": _sample_q3,
    "Q4": _sample_q4,
    "Q5": _sample_q5,
    "Q6": _sample_q6,
    "Q7": _sample_q7,
    "Q8": _sample_q8,
    "Q9": _sample_q9,
    "Q10": _sample_q10,
    "Q11": _sample_q11,
    "Q12": _sample_q12,
    "Q13": _sample_q13,
    "Q14": _sample_q14,
    "Q15": _sample_q15,
    "Q16": _sample_q16,
    "Q17": _sample_q17,
    "Q18": _sample_q18,
    "Q19": _sample_q19,
    "Q20": _sample_q20,
    "Q21": _sample_q21,
    "Q22": _sample_q22,
}


def _resolve_scale_factor(kwargs: dict[str, Any]) -> float:
    """Resolve the TPC-H scale factor used by scale-sensitive parameters."""
    scale_factor = float(kwargs.get("scale_factor", SCALE_FACTOR))
    if scale_factor <= 0:
        raise ValueError(f"scale_factor must be positive, got {scale_factor}")
    return scale_factor


def _sample_placeholders(query_name: str, rnd: random.Random, scale_factor: float) -> dict[str, Any]:
    """Sample placeholders for a normalized TPC-H query name."""
    if query_name not in PLACEHOLDER_SAMPLERS:
        raise ValueError(f"No placeholder generator defined for {query_name}")
    return PLACEHOLDER_SAMPLERS[query_name](rnd, scale_factor)


def _instantiate_sql(
    template: str,
    placeholders: dict[str, Any],
    required_parameters: tuple[str, ...],
) -> str:
    """Replace bracket placeholders in a SQL template and reject incomplete SQL."""
    missing = [name for name in required_parameters if name not in placeholders]
    if missing:
        raise ValueError(f"Missing TPC-H placeholders: {', '.join(missing)}")

    query = template
    for key, value in placeholders.items():
        query = query.replace(f"[{key}]", str(value))

    unresolved = sorted(set(PARAMETER_PATTERN.findall(query)))
    if unresolved:
        raise ValueError(f"Unresolved TPC-H placeholders: {', '.join(unresolved)}")
    return query


def gen_query(
    query_name: str = "Q1",
    rnd: random.Random | None = None,
    seed: int = 42,
    **kwargs: Any,
) -> tuple[str, str, dict[str, Any]]:
    """Generate a TPC-H query tuple compatible with the current factory."""
    contract = get_contract(query_name)
    query_random = rnd if rnd is not None else random.Random(seed)
    scale_factor = _resolve_scale_factor(kwargs)
    placeholders = _sample_placeholders(contract.query_id, query_random, scale_factor)
    query = _instantiate_sql(contract.sql_template, placeholders, contract.parameter_names)
    return contract.sql_template, query, placeholders


def _format_tpch_args_string(query_id: str, placeholders: dict[str, Any]) -> str:
    """Return a stable key=value argument string for generated TPC-H runtimes."""
    if not placeholders:
        return query_id
    args_parts = [query_id]
    for key in sorted(placeholders):
        value = placeholders[key]
        formatted_value = json.dumps(value) if isinstance(value, str) else str(value)
        args_parts.append(f"{key}={formatted_value}")
    return " ".join(args_parts)


def instantiate_tpch_query(
    query_id: str,
    scale_factor: int,
    seed: int | None = None,
) -> dict[str, Any]:
    """Generate the canonical manifest payload for a TPC-H query instance.

    The returned shape lets manifest and runtime providers avoid legacy
    host/time-window arguments.
    """
    if seed is None:
        seed = scale_factor

    contract = get_contract(query_id)
    query_random = random.Random(seed)
    scale_factor_float = _resolve_scale_factor({"scale_factor": scale_factor})
    placeholders = _sample_placeholders(contract.query_id, query_random, scale_factor_float)
    sql = _instantiate_sql(contract.sql_template, placeholders, contract.parameter_names)
    sql_hash = hashlib.sha256(sql.encode()).hexdigest()[:16]
    args_string = _format_tpch_args_string(contract.query_id, placeholders)

    return {
        "query_id": contract.query_id,
        "scale_factor": scale_factor,
        "params_json": dict(placeholders),
        "args_string": args_string,
        "sql": sql,
        "sql_hash": sql_hash,
        "instantiation_id": f"{contract.query_id}_SF{scale_factor}_{sql_hash}",
    }


generate_tpch_query = instantiate_tpch_query
