---
priority: 5
stages: [finish_skeleton, compile_fix, add_timings, implement_queries_writeonly, correctness, benchmark, optimize_build, optimization_general]
areas: [runtime, provider]
---

# C++ Code Style (TPC-H MonetDB Generated Code)

These rules apply exclusively to TPC-H MonetDB generated C++ code (`tpch_monetdb/misc/tpch/templates/` and companion files). They do not introduce repo-wide formatting requirements.

## Google C++ Baseline

- Variables and function parameters: `snake_case`
- Types and structs: `PascalCase`
- Constants and compile-time values: `kCamelCase` or `ALL_CAPS`
- Files: `snake_case.cpp` / `snake_case.hpp` pairs
- Put the opening brace on the same line for `if`, `else`, `for`, `while`, function definitions, and class/struct definitions.
- Always use braces for control-flow blocks, even when the body is a single statement.
- Prefer early `return` / `continue` to reduce nesting instead of long `else` ladders.
- Keep control-flow layout simple and readable; avoid deeply nested condition trees in one function.

## Procedural / Imperative Main Flow

- Keep `build()`, `load()`, and `query()` as thin procedural entry points that delegate to smaller helpers.
- Main execution flow should read top-to-bottom in imperative order: parse/setup -> choose mode -> execute -> emit output.
- Keep side effects, state mutation, and output writes in the outer procedural layer rather than spreading them across many helpers.
- Keep per-query ABI entrypoints in `query_q{qid}.cpp`, keep `query_impl.cpp` as dispatcher/routing glue, put shared family kernels in manifest-owned `query_family_*` files when the active workload unit requires them, and use `query_shared_*` only for pure helpers.
- Functions longer than ~40 lines must be split; each function should do exactly one thing.
- Avoid hidden control flow via macro tricks, template indirection, or helper chains that obscure the runtime path.

## Declarative Rules

- Prefer dispatch tables or lookup arrays over long if/else chains when selecting query kernels.
- Mode selection (e.g., 1-metric vs 5-metric vs 10-metric path) must be decided before the hot loop, not re-evaluated per row.
- Keep business rules in one declarative source of truth; do not duplicate the same branch logic in multiple query helpers.
- Use declarative mappings for query-id -> family / mode / kernel selection when the mapping is static.

## Functional Rules

- For pure transforms, prefer small helper functions with explicit inputs/outputs and no hidden shared mutable state.
- Prefer functional-style helpers for pure filtering / projection / aggregation setup when they stay readable and allocation-free.
- Keep pure data transforms (aggregations, filters, index lookups) in dedicated functions, not inlined inside the dispatcher.
- Do not force functional style into hot loops when an explicit imperative loop is clearer or faster.

## Performance / Hot Loop Rules

- Imperative loops are the default inside hot paths when they are the clearest measured implementation.
- Hoist all loop-invariant work out of tight loops.
- Decide execution strategy (mode, branch) before entering the loop.
- Avoid `std::string`, `std::stoi`, `std::stod`, or heap allocation inside hot loops.
- Use dedicated loop variants per mode rather than one mega-loop with many runtime branches.

## SIMD

- SIMD intrinsics (`<immintrin.h>`) must be guarded with `#if defined(__AVX2__)` (or the relevant ISA define).
- Every SIMD path must have a matching scalar fallback path.

## Comments

- No comments unless the WHY is non-obvious (hidden constraint, subtle invariant, workaround for a known bug).
- Never add comments that just describe what the code does.

## Scope

These rules do NOT apply to:
- Removed legacy baseline tooling unless the task is the explicit cleanup/removal pass
- Python orchestration code in `tpch_monetdb/`
- Any file outside `tpch_monetdb/misc/tpch/templates/` and its companion files
