---
priority: 10
stages: []
areas: [runtime]
---

# TPC-H MonetDB Runtime

- `run_gen_base_impl_tpch_monetdb.py` is the scripted entrypoint.
- `TODO.md` is the scripted progress source of truth.
- Keep host-facing entrypoints stable unless the stage explicitly changes runtime wiring.
- Host-facing entrypoints remain `RawData* load(std::string)`,
  `Engine* build(RawData*)`, and `void query(Engine*)`.
- `validation_mode`, Dockerized MonetDB readiness, and scripted handoff are hard runtime boundaries for the default TPC-H path.
- CSV timestamps must be emitted as ISO-8601 UTC strings with `Z`, not raw epoch integers.
- Query rows belong in `result<RUN_NR>.csv`; stdout stays limited to
  timing/status lines.
