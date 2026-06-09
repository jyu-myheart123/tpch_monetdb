---
priority: 21
stages: [optimization_general]
areas: [runtime, provider]
---

# Optimization Workflow

- Preserve correctness while optimizing. Performance never overrides validation.
- Generated C++ must follow `tpch_monetdb/agent_rules/code-style.md`.
- Compare against the active provider baseline through the runtime/provider flow.
  The default path uses Dockerized MonetDB with TPC-H data.
- Keep each optimization step small enough to revert cleanly.
- Measure speedup with multiple runs and take the median; a single-run
  delta is noise and is not sufficient evidence to claim improvement.
- Target the identified hot region from `tracing_output.log` or the stage
  bottleneck report. Scatter-shot edits across unrelated queries in one
  stage are out of scope.
- Use tracing only to gather or refresh bottleneck evidence; benchmark with
  trace instrumentation off.
- Validate correctness and benchmark after edits before claiming that a
  change improved runtime.
- Record `rt_before / rt_after / reason` in the stage summary so the outer
  loop can reason about the change.
- Query runtime is the primary gate; ingest comparison and hardware
  counters are diagnostic signals that do not override query regressions.
- Avoid regressions on the other queries in the active workload.
- When the active unit is a shared family, optimize the manifest-owned
  `query_family_*` kernel first and treat `query_q*` files as ABI wrappers
  unless the measured bottleneck is truly local to one query.
- If a direction does not improve runtime, revert or remove it before trying
  another direction.
- Revert immediately when query runtime regresses beyond
  `regression_tolerance`; do not wait for a following stage to "fix" the
  regression.
- Split-query modules (for example `query_q12.cpp` or
  `query_shared_groupby.hpp`) are legitimate edit targets; create them only
  when the split directly serves the measured bottleneck, and keep the
  dispatcher `query_impl.cpp` as the ABI surface.
