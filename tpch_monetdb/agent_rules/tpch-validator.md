---
priority: 30
stages: [correctness, optimization_general]
areas: [oracle, provider, runtime]
---

# TPC-H Baseline And Validator

- Canonical baseline SQL comes from the TPC-H Q1-Q22 contracts; table routing
  belongs in runtime/oracle resolution.
- Strict validation covers `verify_sf_list + benchmark_sf`.
- Readiness failures and handoff persistence failures are fatal runtime boundaries.
