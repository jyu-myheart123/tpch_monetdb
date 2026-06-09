---
priority: 20
stages: [storage_plan, todo_plan, finish_skeleton, compile_fix, todo_sync, add_timings, implement_queries, implement_queries_writeonly, correctness, benchmark, optimize_build]
areas: [runtime]
---

# Scripted Workflow

- Workflow priority: `P0 correctness` > `P1 no-CSV kernel runtime / speedup vs MonetDB` > `P2 build/import time guardrail`.
- `storage_plan`: write `storage_plan.txt`, not `TODO.md`.
- `finish_skeleton`: build one vertical slice through loader, builder, and query.
- `finish_skeleton`: preserve host-facing entrypoints and defer parser / API
  reconciliation to `compile_fix`; do not widen the task into general
  cleanup.
- `implement_queries_writeonly`: edit only the current query family scope
  (matching existing focused query modules already present in the workspace,
  such as `query_q*.cpp` / `query_q*.hpp`, `query_family_*.cpp` /
  `query_family_*.hpp`, and `query_shared_*.cpp` / `query_shared_*.hpp`).
  Do not treat host-facing API files such as
  `query_api.hpp`, dispatcher files such as `query_impl.cpp`, or generated
  registry files as query-module scope in this stage. If the current family
  needs a new focused module, keep per-query ABI entrypoints in
  `query_q*.cpp` / `query_q*.hpp`, use manifest-owned `query_family_*.cpp` /
  `query_family_*.hpp` for shared family kernels, and use
  `query_shared_*.cpp` / `query_shared_*.hpp` only for pure helpers. Family
  kernels must never own `dispatch_query(...)`,
  `dispatch_unimplemented_query(...)`, or any `execute_q*` entrypoint. Then
  stop. No `compile` or `run`.

- `implement_queries_writeonly` — Family-first implementation order (MANDATORY
  when the stage descriptor starts with `implement_family_kernel_` or
  `implement_entrypoint_`):
  1. Shared helper functions (date parsing, decimal formatting, join-key
     lookup, string predicates, ordering, and reusable aggregation helpers)
     MUST be defined exactly once
     in `query_shared_helpers.cpp` / `query_shared_helpers.hpp`. They MUST NOT
     be duplicated in any `query_q*.cpp` or `query_family_*.cpp` file.
  2. Family kernels (`query_family_*.cpp` / `query_family_*.hpp`) own the
     parameterized aggregation logic. Family kernels MUST NOT define
     `execute_q*`, `dispatch_query`, or `dispatch_unimplemented_query`.
  3. Thin entrypoints (`query_q*.cpp` / `query_q*.hpp`) are ~15 lines each:
     parse args via `parse_q*()` (already in `args_parser.hpp`), call the
     family kernel, return. Entrypoints MUST NOT contain helper functions or
     aggregation logic.
  4. Implementation order: shared helpers first → family kernel second →
     thin entrypoints last.
  5. After each implementation stage, read TODO.md and update completed
     checklist items to `[x]` and in-progress items to `[~]`.
- `compile_fix` and `correctness`: use compile/run evidence, then write before retrying.
- `todo_sync`: update `TODO.md`. Do not report progress only in natural language.
- `optimize_build`: prioritises builder work, but may touch loader or query
  code when ingest evidence or shared data layout is the direct bottleneck
  (for example when a loader-side buffering change unlocks builder cache
  behavior). Query correctness and query runtime remain the primary gates;
  `Build ms` / `Ingest ms = Load ms + Build ms` are the ingest guardrail.
  Follow the original workflow's idea of a final build-tuning pass without
  any fixed `10s` gate, and use Q1/Q8/Q9 as the first regression probe
  before any full rerun. Hardware-counter diagnostics (cache-miss, cycles)
  remain diagnostic only and do not gate base-stage completion.

## Phase10 Guardrails (Active)

- Agent scope: TPC-H MonetDB implementation sources under `tpch_monetdb/misc/tpch/templates/`,
  including the stable dispatcher (`query_impl.cpp` / `query_impl.hpp`) and
  focused query modules already present in the workspace plus any new modules
  created under `query_q*.cpp` / `query_q*.hpp`,
  `query_family_*.cpp` / `query_family_*.hpp`, and
  `query_shared_*.cpp` / `query_shared_*.hpp`, together with
  `builder_impl.*` and `loader_impl.*`. Query modules are legal targets;
  `query_impl.cpp` is no longer the only query implementation file. Keep
  `query_q*` as thin ABI entrypoints when the active unit requires them, let
  `query_family_*` own shared family kernels, use `query_shared*` only for
  pure helpers, and leave dispatcher wiring to structure-oriented stages.
  `query_impl.cpp` must stay thin and must not own `execute_q*` stubs once
  query modules exist.
- **Do not** modify removed legacy baseline-owned files or recreate old
  query-file runner paths.
- For the default TPC-H path, treat removed legacy files as cleanup targets
  only. Do not revive them or add new dependencies on them.
- **Do not** introduce HTTP-to-HTTP alignment experiments or Bespoke HTTP endpoints.
- **Do not** implement multi-worker ingest or multi-worker query baseline.
- Workload coverage: **Q1-Q22** (TPC-H). Do not treat Q1/Q6 readiness or
  Q1/Q9 obligation probes as the complete set.
- Timing output must include `Query ms` (full-CSV correctness diagnostic) and `Execution ms` (no-CSV kernel primary for optimization) for queries; `Load ms`, `Build ms`, `Ingest ms` for ingest.
- All timing prints use `%.3f ms` precision (sub-millisecond preserved).
- Query semantics, CSV column order, sorting behavior, and stdout protocol
  are frozen once correctness stages begin unless a stage explicitly changes
  runtime wiring.
