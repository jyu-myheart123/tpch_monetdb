---
priority: 5
stages: []
---

# Data Contract

- Builder is the single source of truth for hostname→block-position mappings.
  It must store stable, deterministic mappings in Engine during build().
- Query modules must read hostname→position mappings from Engine's stored state.
  Do not reconstruct hostname→position mappings independently in query modules.
- CSV output column order and sorting are frozen once correctness stages begin.
- `query_impl.cpp` must print both `Execution ms` and `Query ms` via `std::printf`
  for every query. `Execution ms` is the no-CSV kernel performance metric;
  `Query ms` is full-CSV correctness/materialization diagnostic only.
