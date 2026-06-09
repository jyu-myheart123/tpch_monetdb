from __future__ import annotations

from dataclasses import dataclass

@dataclass(frozen=True)
class QueryCodegenHint:
    prompt_lines: tuple[str, ...]
    generated_code_checks: tuple[str, ...] = ()
    diagnostics: tuple[str, ...] = ()


QUERY_CODEGEN_HINTS: dict[str, QueryCodegenHint] = {
    "1": QueryCodegenHint(
        prompt_lines=(
            "Q1 is an Engine-backed lineitem scan and aggregation over `l_returnflag` and `l_linestatus`; `execute_q1(...)` should consume typed lineitem columns already built into `Engine`.",
            "TPC-H `.tbl` parsing, source directory discovery, and raw-row reconstruction belong to loader/builder; if Q1 lacks lineitem fields, fix RawData, build, or Engine layout instead of rebuilding source data inside `query_q1.cpp`.",
            "Treat scale runtime as evidence: Q1 should be shaped by the lineitem date filter, aggregation cardinality, and output ordering, not by query-time source-file parsing.",
            "Q1 formula: sum(l_extendedprice * (1-l_discount) * (1+l_tax)). This is NOT the same as Q6 which uses sum(l_extendedprice * l_discount) with the raw discount.",
        ),
    ),
    "2": QueryCodegenHint(
        prompt_lines=(
            "Q2 should use Engine table state for part, partsupp, supplier, nation, and region joins plus reusable key maps where they are part of the general storage layout.",
            "Per-part minimum supply-cost selection belongs in query code, but source `.tbl` parsing and data-path discovery belong to loader/builder; if Engine lacks table keys or columns, repair the Engine build path first.",
            "Treat scale runtime as evidence: Q2 should be shaped by part filters, region filters, join candidate counts, and top-k ordering, not by full source reconstruction at every scale.",
        ),
    ),
    "6": QueryCodegenHint(
        prompt_lines=(
            "Q6 is a lineitem-only scan with shipdate, discount, and quantity predicates.",
            "Apply date and numeric predicates before revenue arithmetic.",
            "Keep the aggregation path scalar and allocation-free; no map is needed for the single-row result.",
            "CRITICAL: Q6 revenue formula is sum(l_extendedprice * l_discount) — the raw discount, NOT (1-l_discount). Do NOT confuse Q6 with Q1 which uses (1-l_discount)*(1+l_tax). Q6 multiplies directly by l_discount, not by (1-l_discount).",
            "LineitemRow fields `l_extendedprice`, `l_discount`, and `l_quantity` are already typed numeric fields in Engine; only `args.DISCOUNT` is a std::string and should be parsed once with `std::stod(args.DISCOUNT)` before predicate checks.",
            "SQL SUM returns NULL when no lineitem rows match Q6 predicates; emit an empty revenue CSV cell for that case, not 0.000000.",
            "CSV output protocol: guard all csv_output population with `if (should_materialize_query_output())`. Performance benchmarks use no_output mode which skips CSV; unguarded writes will fail the generated-code check.",
        ),
        generated_code_checks=("query_antipatterns",),
        diagnostics=("lineitem_predicate_order", "q6_wrong_formula",),
    ),
    "9": QueryCodegenHint(
        prompt_lines=(
            "Q9 is a high-risk six-table join and profit aggregation path.",
            "Use reusable part, supplier, partsupp, lineitem, orders, and nation access paths from Engine instead of reconstructing joins from source rows.",
            "Prefer compact join candidate lists and bounded `(nation, year)` accumulators over broad `std::unordered_map` paths when the key space is bounded.",
            "Filter part name color and supplier/partsupp candidates before expanding lineitem/order joins when that preserves SQL semantics.",
            "Avoid result sorting paths that repeatedly compare raw nation strings; use dictionary ids or compact keys until final output materialization.",
        ),
        generated_code_checks=("query_antipatterns",),
        diagnostics=("q9_join_candidate_explosion",),
    ),
    "12": QueryCodegenHint(
        prompt_lines=(
            "Q12 must preserve final output order by `l_shipmode`.",
            "If you add an orders-lineitem access path, it must preserve enough information to evaluate receipt/commit/ship date predicates and order priority CASE logic correctly.",
            "Under large data sizes, treat join candidate explosion and final sorting/materialization as first-class bottlenecks.",
            "Prefer compact row references or integer ids before string materialization.",
        ),
        generated_code_checks=("query_antipatterns",),
    ),
    "13": QueryCodegenHint(
        prompt_lines=(
            "Q13 is a customer-orders left outer join with a not-like comment predicate and customer distribution aggregation.",
            "Prefer bounded count accumulators over hash-heavy aggregation paths when customer id ranges remain compact.",
            "When the customer id domain is direct-indexable, use direct indexing for per-customer order counts before the distribution aggregation.",
            "Avoid repeated comment string scans after an order has already been classified by the not-like predicate.",
        ),
        generated_code_checks=("query_antipatterns",),
        diagnostics=("q13_outer_join_distribution"),
    ),
    "14": QueryCodegenHint(
        prompt_lines=(
            "Q14 is a part-lineitem join with a promo revenue ratio.",
            "Apply the lineitem shipdate range and part promo predicate before ratio arithmetic when that preserves SQL semantics.",
            "Keep numerator and denominator accumulation in a compact scalar path instead of defaulting to a generic sparse map.",
            "Avoid repeated part type string checks by using dictionary ids or cached prefix classification when available in the general Engine layout.",
        ),
        generated_code_checks=("query_antipatterns",),
        diagnostics=("q14_ratio_path"),
    ),
    "3": QueryCodegenHint(
        prompt_lines=(
            "Q3 is a customer-orders-lineitem join with shipping-priority ordering and revenue aggregation.",
            "CRITICAL: Query code MUST consume Engine column data (l_extendedprice, l_discount, o_orderdate, o_shippriority, c_mktsegment) through the Engine reference. DO NOT open or parse .tbl files, raw data directories, or base_data_dir inside query_q3.cpp.",
            "Prefer join-candidate filtering by order date and ship date before revenue aggregation.",
        ),
    ),
    "4": QueryCodegenHint(
        prompt_lines=(
            "Q4 is an orders-lineitem exists subquery with order priority distribution.",
            "CRITICAL: Query code MUST consume Engine column data through the Engine reference. DO NOT open or parse .tbl files, raw data directories, or base_data_dir inside query_q4.cpp.",
            "Prefer pre-grouping by order priority before counting.",
        ),
    ),
    "5": QueryCodegenHint(
        prompt_lines=(
            "Q5 is a customer-orders-lineitem-supplier-nation-region join with revenue aggregation by nation.",
            "CRITICAL: Query code MUST consume Engine column data through the Engine reference. DO NOT open or parse .tbl files, raw data directories, or base_data_dir inside query_q5.cpp.",
            "Prefer join-key maps from Engine over repeated full-table scans.",
        ),
    ),
    "7": QueryCodegenHint(
        prompt_lines=(
            "Q7 is a supplier-lineitem-orders-customer-nation join with shipping volume by nation-year pairs.",
            "CRITICAL: Query code MUST consume Engine column data through the Engine reference. DO NOT open or parse .tbl files, raw data directories, or base_data_dir inside query_q7.cpp.",
            "Prefer nested nation key lookups over repeated full-table scans.",
        ),
    ),
    "8": QueryCodegenHint(
        prompt_lines=(
            "Q8 is a part-supplier-lineitem-orders-customer-nation-region join with market share by year.",
            "CRITICAL: Query code MUST consume Engine column data through the Engine reference. DO NOT open or parse .tbl files, raw data directories, or base_data_dir inside query_q8.cpp.",
            "Prefer filtering by region and part type before expanding joins.",
        ),
    ),
    "10": QueryCodegenHint(
        prompt_lines=(
            "Q10 is a customer-orders-lineitem-nation join with revenue by customer, ordered by revenue descending.",
            "CRITICAL: Query code MUST consume Engine column data through the Engine reference. DO NOT open or parse .tbl files, raw data directories, or base_data_dir inside query_q10.cpp.",
            "Filter orders by the 3-month orderdate window and lineitem by l_returnflag='R' before aggregation.",
            "Do not implement an orders x lineitem nested join; use an Engine-backed orderkey lookup or contiguous lineitem range/list per orderkey so the SF1 path is linear in orders plus lineitems.",
            "Sort only the final customer aggregate groups by revenue descending.",
        ),
    ),
    "11": QueryCodegenHint(
        prompt_lines=(
            "Q11 is a partsupp-supplier-nation join finding partsupp pairs whose supply cost exceeds a fraction of the global sum.",
            "CRITICAL: Query code MUST consume Engine column data through the Engine reference. DO NOT open or parse .tbl files, raw data directories, or base_data_dir inside query_q11.cpp.",
            "Prefer computing the global threshold in one pass before the per-part selection.",
        ),
    ),
    "15": QueryCodegenHint(
        prompt_lines=(
            "Q15 computes supplier revenue over a date window and returns suppliers tied for maximum revenue.",
            "Prefer a direct-indexed supplier revenue array over a hash map when supplier keys are compact or already dictionary-mapped.",
            "Preserve the TPC-H max-revenue tie semantics and final supplier ordering contract.",
        ),
    ),
    "16": QueryCodegenHint(
        prompt_lines=(
            "Q16 is a partsupp-supplier join with part brand/type/size filters and supplier comment NOT LIKE predicate.",
            "CRITICAL: Query code MUST consume Engine column data through the Engine reference. DO NOT open or parse .tbl files, raw data directories, or base_data_dir inside query_q16.cpp.",
            "Prefer filtering by part attributes before joining with partsupp and supplier.",
        ),
    ),
    "17": QueryCodegenHint(
        prompt_lines=(
            "Q17 is a lineitem-part join with quantity threshold derived from a lineitem sub-aggregate.",
            "CRITICAL: Query code MUST consume Engine column data through the Engine reference. DO NOT open or parse .tbl files, raw data directories, or base_data_dir inside query_q17.cpp.",
            "Prefer computing the per-part average quantity threshold in one pass before the main join.",
        ),
    ),
    "18": QueryCodegenHint(
        prompt_lines=(
            "Q18 is a large-volume customer-orders-lineitem join with a HAVING subquery for large-quantity orders, returning top-100 by total quantity.",
            "CRITICAL — ENGINE DATA ONLY: query_q18.cpp MUST consume all data through `Engine& engine`. You are FORBIDDEN from using std::ifstream, fopen, mmap, std::filesystem, or any file-I/O to open/read .tbl files, raw data directories, or base_data_dir from inside query code. All lineitem, orders, and customer columns arrive through Engine typed columns, key maps, and join structures built by loader/builder at ingestion time. If Engine lacks columns or keys Q18 needs, edit loader_impl, builder_impl, or RawData — NEVER bypass Engine by reading files directly.",
            "Prefer join-candidate filtering by order quantity threshold and order date before expanding to full customer details.",
            "Top-100 ordering should be a bounded heap or partial sort over the final aggregated results.",
        ),
    ),
    "19": QueryCodegenHint(
        prompt_lines=(
            "Q19 is a lineitem-part join with a complex three-branch brand/quantity/size disjunction.",
            "CRITICAL: Query code MUST consume Engine column data through the Engine reference. DO NOT open or parse .tbl files, raw data directories, or base_data_dir inside query_q19.cpp.",
            "Prefer decomposing the three disjunct branches into sequential filters evaluated against part attributes before joining with lineitem.",
        ),
    ),
    "20": QueryCodegenHint(
        prompt_lines=(
            "Q20 is a supplier-partsupp-part-nation join with a lineitem subquery for partsupp availability in a date range.",
            "CRITICAL: Query code MUST consume Engine column data through the Engine reference. DO NOT open or parse .tbl files, raw data directories, or base_data_dir inside query_q20.cpp.",
            "Prefer computing the per-part available quantity from lineitem in one pass before joining with supplier dimensions.",
        ),
    ),
    "21": QueryCodegenHint(
        prompt_lines=(
            "Q21 is a supplier-lineitem-orders-nation join finding suppliers who kept orders waiting, with an EXISTS subquery.",
            "CRITICAL: Query code MUST consume Engine column data through the Engine reference. DO NOT open or parse .tbl files, raw data directories, or base_data_dir inside query_q21.cpp.",
            "Prefer identifying late-lineitem supplier keys in one pass before expanding to supplier and nation details.",
        ),
    ),
    "22": QueryCodegenHint(
        prompt_lines=(
            "Q22 is a customer aggregation with a phone-substring filter and a revenue subquery on orders.",
            "CRITICAL: Query code MUST consume Engine column data through the Engine reference. DO NOT open or parse .tbl files, raw data directories, or base_data_dir inside query_q22.cpp.",
            "Prefer filtering customers by phone pattern and account balance before computing the per-country average and count.",
        ),
    ),
}


def build_query_codegen_hint_text(query_id: str) -> str:
    hint = QUERY_CODEGEN_HINTS.get(str(query_id))
    if hint is None:
        return ""
    return "\n".join(
        ["Additional implementation guidance:"] + [f"- {line}" for line in hint.prompt_lines]
    )


def get_query_generated_code_checks(query_id: str) -> list[str]:
    hint = QUERY_CODEGEN_HINTS.get(str(query_id))
    base_checks = ["query_protocol", "final_path_integrity", "usage_double_output"]
    checks = [*base_checks]
    if hint is not None:
        checks.extend(hint.generated_code_checks)
    return list(dict.fromkeys(checks))
