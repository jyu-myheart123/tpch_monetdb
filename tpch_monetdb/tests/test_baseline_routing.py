"""Phase9 baseline routing and outer-loop checkpoint tests.

Covers:
- 9.6: inner-loop prompt iteration must NOT re-run baseline
- 9.7: diff routing skips the right baseline based on change type
- 9.5: agent diff touching baseline-owned files is rejected
- 9.2: BaselineRoutingPolicy declarative routing table
"""

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tpch_monetdb.benchmark.manifest import (
    BASELINE_OWNED_PATHS,
    BaselineRoutingPolicy,
    ChangeType,
    RuntimeMeasurement,
    ReferenceManifest,
    QueryInstantiation,
    check_agent_diff_boundary,
)


# ---------------------------------------------------------------------------
# 9.2 — BaselineRoutingPolicy declarative routing
# ---------------------------------------------------------------------------


def test_routing_query_only_skips_ingest_baseline() -> None:
    """query_only 变更类型应可跳过 ingest baseline。"""
    policy = BaselineRoutingPolicy.from_changed_files(["query_impl.cpp"])
    assert policy.change_type == ChangeType.QUERY_ONLY
    assert policy.should_skip("ingest_baseline") is True
    assert policy.should_skip("query_baseline") is False


def test_routing_loader_builder_only_skips_query_baseline() -> None:
    """loader/builder-only 变更应可跳过 query baseline。"""
    policy = BaselineRoutingPolicy.from_changed_files(["loader_impl.hpp", "builder_impl.cpp"])
    assert policy.change_type == ChangeType.LOADER_BUILDER_ONLY
    assert policy.should_skip("query_baseline") is True
    assert policy.should_skip("ingest_baseline") is False


def test_routing_mixed_change_skips_nothing() -> None:
    """混合变更（query + loader）不允许跳过任何 baseline。"""
    policy = BaselineRoutingPolicy.from_changed_files(["query_impl.cpp", "loader_impl.cpp"])
    assert policy.change_type == ChangeType.ALL
    assert policy.should_skip("query_baseline") is False
    assert policy.should_skip("ingest_baseline") is False


def test_routing_unknown_paths_treated_as_all() -> None:
    """无法识别的路径应归类为 ALL 不跳过任何 baseline。"""
    policy = BaselineRoutingPolicy.from_changed_files(["some_other_file.cpp"])
    assert policy.change_type == ChangeType.ALL
    assert policy.should_skip("query_baseline") is False
    assert policy.should_skip("ingest_baseline") is False


def test_routing_empty_change_treated_as_all() -> None:
    """空变更列表归类为 ALL。"""
    policy = BaselineRoutingPolicy.from_changed_files([])
    assert policy.change_type == ChangeType.ALL


# ---------------------------------------------------------------------------
# 9.5 — Agent diff boundary: baseline-owned paths are rejected
# ---------------------------------------------------------------------------


def test_check_agent_diff_boundary_rejects_monetdb_oracle() -> None:
    """monetdb_oracle.py 属于 baseline-owned，必须触发 violation。"""
    violations = check_agent_diff_boundary(["tpch_monetdb/oracle/monetdb_oracle.py"])
    assert len(violations) == 1
    assert "monetdb_oracle.py" in violations[0]


def test_check_agent_diff_boundary_allows_template_files() -> None:
    """模板文件（query_impl.cpp 等）不应被拒绝。"""
    safe_paths = [
        "tpch_monetdb/misc/tpch/templates/query_impl.cpp",
        "tpch_monetdb/misc/tpch/templates/builder_impl.hpp",
        "tpch_monetdb/benchmark/providers.py",
    ]
    violations = check_agent_diff_boundary(safe_paths)
    assert violations == []


def test_check_agent_diff_boundary_multi_file_partial_violation() -> None:
    """多文件中只有部分 violation，应只返回 violation 的文件。"""
    changed = [
        "tpch_monetdb/misc/tpch/templates/query_impl.cpp",
        "tpch_monetdb/oracle/monetdb_prepare.py",
        "tpch_monetdb/benchmark/providers.py",
    ]
    violations = check_agent_diff_boundary(changed)
    assert len(violations) == 1
    assert "monetdb_prepare.py" in violations[0]


def test_baseline_owned_paths_covers_key_files() -> None:
    """BASELINE_OWNED_PATHS 必须覆盖所有关键 baseline 文件。"""
    expected = {"monetdb_oracle.py", "monetdb_prepare.py", "tpch_validator.py"}
    assert expected.issubset(BASELINE_OWNED_PATHS)


# ---------------------------------------------------------------------------
# 9.6 — Inner-loop must NOT re-run baseline
# ---------------------------------------------------------------------------


def _make_inst(qid: str, inst_id: str) -> QueryInstantiation:
    return QueryInstantiation(
        query_id=qid,
        scale_factor=1,
        instantiation_id=inst_id,
        params_json={},
        args_string="1",
        sql="SELECT 1",
        sql_hash="abc",
    )


def _make_measurement(inst_id: str) -> RuntimeMeasurement:
    return RuntimeMeasurement(
        instantiation_id=inst_id,
        runtime_ms=100.0,
        num_runs=1,
        all_runtimes_ms=[100.0],
        timestamp="2026-01-01T00:00:00Z",
        benchmark_mode="system-parity",
        storage_mode="persistent",
        workers=1,
        engine="questdb",
        measurement_shape_status="known",
    )


def _make_legacy_validator_instantiation(query_id: str, scale_factor: int) -> QueryInstantiation:
    return QueryInstantiation(
        query_id=query_id,
        scale_factor=scale_factor,
        instantiation_id=f"legacy_q{query_id}_sf{scale_factor}",
        params_json={
            "hostnames": ["host_1"],
            "time_start": "2016-01-01T00:00:00Z",
            "time_end": "2016-01-01T01:00:00Z",
        },
        args_string=(
            f"{query_id} ('host_1') "
            "\"2016-01-01T00:00:00Z\" \"2016-01-01T01:00:00Z\""
        ),
        sql="SELECT 1",
        sql_hash=f"legacy_{query_id}_{scale_factor}",
    )


def _make_out_of_range_instantiation(query_id: str, scale_factor: int) -> QueryInstantiation:
    """构造一个超出 validator 数据窗口的 legacy instantiation."""
    return QueryInstantiation(
        query_id=query_id,
        scale_factor=scale_factor,
        instantiation_id=f"stale_q{query_id}",
        params_json={
            "hostnames": ["host_1"],
            "time_start": "2016-01-21T00:00:00Z",
            "time_end": "2016-01-21T01:00:00Z",
            "threshold": None,
            "limit": None,
        },
        args_string=f'{query_id} (\'host_1\') "2016-01-21T00:00:00Z" "2016-01-21T01:00:00Z"',
        sql="SELECT 1",
        sql_hash="stale",
    )


def _make_baseline_provider(measure_calls: list[str]) -> Any:
    class FakeBaselineProvider:
        benchmark_mode = "system-parity"
        storage_mode = "persistent"
        workers = 1
        engine = "questdb"

        def measure(self, inst: QueryInstantiation) -> RuntimeMeasurement:
            measure_calls.append(inst.instantiation_id)
            return _make_measurement(inst.instantiation_id)

        def validate_configuration(self) -> None:
            return None

    return FakeBaselineProvider()


def _mark_baseline_run_started(conv: Any, value: str = "2025-01-01T00:00:00Z") -> None:
    conv.baseline_run_started_at = value
    conv.baseline_max_age_seconds = None
    return None


def test_measure_with_manifest_raises_on_cache_miss_without_collection_flag(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """若 manifest 中没有 baseline 且 allow_baseline_collection=False，
    _measure_with_manifest 应 raise，而不是调用 baseline_provider.measure()。

    这验证了 9.6：内环不应自动触发 baseline 收集。
    """
    from tpch_monetdb.conversations.optimization_conversation_tpch_monetdb import TpchMonetdbOptimizationConversation

    baseline_called = []

    conv = TpchMonetdbOptimizationConversation.__new__(TpchMonetdbOptimizationConversation)
    conv.query_ids = ["1"]
    conv.benchmark_sf = 1
    conv.manifest = ReferenceManifest(tmp_path / "manifest.json")
    conv.manifest.add_instantiation(_make_inst("1", "inst_1_0"))
    conv.baseline_provider = _make_baseline_provider(baseline_called)
    conv.impl_provider = SimpleNamespace()
    _mark_baseline_run_started(conv)

    with pytest.raises(RuntimeError, match="not cached"):
        conv._measure_with_manifest(
            query_id="1",
            exec_callback=None,
            allow_baseline_collection=False,
        )

    assert baseline_called == [], "baseline_provider.measure must NOT be called in inner-loop mode"


def test_measure_with_manifest_uses_cached_baseline_without_provider_call(
    tmp_path: Path,
) -> None:
    """manifest 中有缓存 baseline 时，_measure_with_manifest 不调用 baseline_provider.measure()。"""
    from tpch_monetdb.conversations.optimization_conversation_tpch_monetdb import TpchMonetdbOptimizationConversation

    baseline_called = []

    conv = TpchMonetdbOptimizationConversation.__new__(TpchMonetdbOptimizationConversation)
    conv.query_ids = ["1"]
    conv.benchmark_sf = 1
    conv.manifest = ReferenceManifest(tmp_path / "manifest.json")
    inst = _make_inst("1", "inst_1_0")
    conv.manifest.add_instantiation(inst)
    conv.manifest.record_runtime(_make_measurement("inst_1_0"))
    conv.baseline_provider = _make_baseline_provider(baseline_called)
    conv.impl_provider = SimpleNamespace()
    _mark_baseline_run_started(conv)

    rt_s, baseline_rt_s, speedup, lazy_suspected = conv._measure_with_manifest(
        query_id="1",
        exec_callback=None,
        allow_baseline_collection=False,
    )

    assert baseline_called == [], "baseline_provider.measure must NOT be called when cache hit"
    assert baseline_rt_s == pytest.approx(0.1)  # 100ms -> 0.1s
    assert lazy_suspected is False


def test_measure_with_manifest_raises_on_stale_cache_without_collection_flag(
    tmp_path: Path,
) -> None:
    """stale baseline 在内环必须报错，不能静默复用或重测。"""
    from tpch_monetdb.conversations.optimization_conversation_tpch_monetdb import TpchMonetdbOptimizationConversation

    baseline_called: list[str] = []
    conv = TpchMonetdbOptimizationConversation.__new__(TpchMonetdbOptimizationConversation)
    conv.query_ids = ["1"]
    conv.benchmark_sf = 1
    conv.manifest = ReferenceManifest(tmp_path / "manifest.json")
    inst = _make_inst("1", "inst_1_0")
    conv.manifest.add_instantiation(inst)
    conv.manifest.record_runtime(
        RuntimeMeasurement(
            instantiation_id="inst_1_0",
            runtime_ms=100.0,
            num_runs=1,
            all_runtimes_ms=[100.0],
            timestamp="2026-01-01T00:00:00",
            benchmark_mode="query-latency",
            storage_mode="tmpfs",
            workers=1,
            engine="questdb",
        )
    )
    conv.baseline_provider = _make_baseline_provider(baseline_called)
    conv.impl_provider = SimpleNamespace()
    _mark_baseline_run_started(conv)

    with pytest.raises(RuntimeError, match="stale"):
        conv._measure_with_manifest(
            query_id="1",
            exec_callback=None,
            allow_baseline_collection=False,
        )

    assert baseline_called == []


def test_collect_baselines_at_checkpoint_fills_missing_entries(
    tmp_path: Path,
) -> None:
    """_collect_baselines_at_checkpoint 只补充缺失的 baseline，已缓存的不重测。"""
    from tpch_monetdb.conversations.optimization_conversation_tpch_monetdb import TpchMonetdbOptimizationConversation

    measure_calls = []

    conv = TpchMonetdbOptimizationConversation.__new__(TpchMonetdbOptimizationConversation)
    conv.manifest = ReferenceManifest(tmp_path / "manifest.json")

    inst_cached = _make_inst("1", "inst_cached")
    inst_missing = _make_inst("1", "inst_missing")
    conv.manifest.add_instantiation(inst_cached)
    conv.manifest.add_instantiation(inst_missing)
    conv.manifest.record_runtime(_make_measurement("inst_cached"))
    conv.baseline_provider = _make_baseline_provider(measure_calls)
    _mark_baseline_run_started(conv)

    conv._collect_baselines_at_checkpoint()

    assert measure_calls == ["inst_missing"], (
        "Only missing instantiation should be measured at checkpoint"
    )
    assert conv.manifest.get_runtime("inst_cached") is not None
    assert conv.manifest.get_runtime("inst_missing") is not None


def test_collect_baselines_at_checkpoint_replaces_stale_entries(
    tmp_path: Path,
) -> None:
    """outer-loop checkpoint 遇到 stale baseline 时应删旧值后重测。"""
    from tpch_monetdb.conversations.optimization_conversation_tpch_monetdb import TpchMonetdbOptimizationConversation

    measure_calls: list[str] = []
    conv = TpchMonetdbOptimizationConversation.__new__(TpchMonetdbOptimizationConversation)
    conv.manifest = ReferenceManifest(tmp_path / "manifest.json")
    inst = _make_inst("1", "inst_stale")
    conv.manifest.add_instantiation(inst)
    conv.manifest.record_runtime(
        RuntimeMeasurement(
            instantiation_id="inst_stale",
            runtime_ms=100.0,
            num_runs=1,
            all_runtimes_ms=[100.0],
            timestamp="2026-01-01T00:00:00",
            benchmark_mode="query-latency",
            storage_mode="tmpfs",
            workers=1,
            engine="questdb",
        )
    )
    conv.baseline_provider = _make_baseline_provider(measure_calls)
    _mark_baseline_run_started(conv)

    conv._collect_baselines_at_checkpoint()

    assert measure_calls == ["inst_stale"]
    stored = conv.manifest.get_runtime("inst_stale")
    assert stored is not None
    assert stored.benchmark_mode == "system-parity"
    assert stored.storage_mode == "persistent"


def test_collect_baselines_at_checkpoint_replaces_pre_run_baseline(
    tmp_path: Path,
) -> None:
    """本轮启动前采集的 baseline 必须重测，不能进入最终统计。"""
    from tpch_monetdb.conversations.optimization_conversation_tpch_monetdb import TpchMonetdbOptimizationConversation

    measure_calls: list[str] = []
    conv = TpchMonetdbOptimizationConversation.__new__(TpchMonetdbOptimizationConversation)
    conv.manifest = ReferenceManifest(tmp_path / "manifest.json")
    inst = _make_inst("1", "inst_old")
    conv.manifest.add_instantiation(inst)
    conv.manifest.record_runtime(_make_measurement("inst_old"))
    conv.baseline_provider = _make_baseline_provider(measure_calls)
    _mark_baseline_run_started(conv, "2026-02-01T00:00:00Z")

    conv._collect_baselines_at_checkpoint()

    assert measure_calls == ["inst_old"]


# ---------------------------------------------------------------------------
# 9.7 — Diff routing skips right baseline
# ---------------------------------------------------------------------------


def test_diff_routing_query_only_change_skips_ingest_not_query() -> None:
    """query-only diff → skip ingest baseline, keep query baseline."""
    changed = ["tpch_monetdb/misc/tpch/templates/query_impl.cpp"]
    policy = BaselineRoutingPolicy.from_changed_files(changed)

    assert policy.should_skip("ingest_baseline") is True
    assert policy.should_skip("query_baseline") is False


def test_diff_routing_loader_change_skips_query_not_ingest() -> None:
    """loader-only diff → skip query baseline, keep ingest baseline."""
    changed = ["tpch_monetdb/misc/tpch/templates/loader_impl.cpp"]
    policy = BaselineRoutingPolicy.from_changed_files(changed)

    assert policy.should_skip("query_baseline") is True
    assert policy.should_skip("ingest_baseline") is False


def test_diff_routing_builder_change_skips_query_not_ingest() -> None:
    """builder-only diff → skip query baseline, keep ingest baseline."""
    changed = ["tpch_monetdb/misc/tpch/templates/builder_impl.hpp"]
    policy = BaselineRoutingPolicy.from_changed_files(changed)

    assert policy.should_skip("query_baseline") is True
    assert policy.should_skip("ingest_baseline") is False


def test_diff_routing_query_plus_builder_skips_nothing() -> None:
    """query + builder diff → ALL → skip nothing."""
    changed = [
        "tpch_monetdb/misc/tpch/templates/query_impl.cpp",
        "tpch_monetdb/misc/tpch/templates/builder_impl.cpp",
    ]
    policy = BaselineRoutingPolicy.from_changed_files(changed)

    assert policy.should_skip("query_baseline") is False
    assert policy.should_skip("ingest_baseline") is False


def test_refresh_query_baselines_for_stage_skips_loader_builder_only_changes(
    tmp_path: Path,
) -> None:
    from tpch_monetdb.conversations.optimization_conversation_tpch_monetdb import TpchMonetdbOptimizationConversation

    measure_calls: list[str] = []
    conv = TpchMonetdbOptimizationConversation.__new__(TpchMonetdbOptimizationConversation)
    conv.manifest = ReferenceManifest(tmp_path / "manifest.json")
    inst = _make_inst("1", "inst_1_0")
    conv.manifest.add_instantiation(inst)
    conv.manifest.record_runtime(_make_measurement("inst_1_0"))
    conv.baseline_provider = _make_baseline_provider(measure_calls)

    conv._refresh_query_baselines_for_stage(
        {"tpch_monetdb/misc/tpch/templates/builder_impl.hpp"}
    )

    assert measure_calls == []


def test_refresh_query_baselines_for_stage_refreshes_query_changes(
    tmp_path: Path,
) -> None:
    from tpch_monetdb.conversations.optimization_conversation_tpch_monetdb import TpchMonetdbOptimizationConversation

    measure_calls: list[str] = []
    conv = TpchMonetdbOptimizationConversation.__new__(TpchMonetdbOptimizationConversation)
    conv.manifest = ReferenceManifest(tmp_path / "manifest.json")
    inst = _make_inst("1", "inst_1_0")
    conv.manifest.add_instantiation(inst)
    conv.manifest.record_runtime(_make_measurement("inst_1_0"))
    conv.baseline_provider = _make_baseline_provider(measure_calls)

    conv._refresh_query_baselines_for_stage(
        {"tpch_monetdb/misc/tpch/templates/query_impl.cpp"}
    )

    assert measure_calls == ["inst_1_0"]


# ---------------------------------------------------------------------------
# Ingest baseline removal
# ---------------------------------------------------------------------------


def test_refresh_ingest_baseline_is_noop_after_tsbs_loader_removal(
    tmp_path: Path,
) -> None:
    """TSBS ingest baseline 删除后，stage-end refresh 不应调用 provider 或写 manifest。"""
    from tpch_monetdb.conversations.optimization_conversation_tpch_monetdb import TpchMonetdbOptimizationConversation

    ingest_calls: list[int] = []

    class FakeProvider:
        benchmark_mode = "system-parity"
        storage_mode = "persistent"
        workers = 1
        engine = "questdb"

        def measure_ingest(self, scale_factor, ilp_file_path):
            ingest_calls.append(scale_factor)
            raise AssertionError("measure_ingest should not be called")

    base_data_dir = tmp_path / "data"
    (base_data_dir / "sf1").mkdir(parents=True, exist_ok=True)
    (base_data_dir / "sf1" / "cpu.ilp").write_text("cpu,host=a load=0.5\n")

    conv = TpchMonetdbOptimizationConversation.__new__(TpchMonetdbOptimizationConversation)
    conv.manifest = ReferenceManifest(tmp_path / "manifest.json")
    conv.manifest.add_instantiation(_make_inst("1", "inst_1_0"))
    conv.baseline_provider = FakeProvider()
    conv.base_data_dir = base_data_dir

    conv._refresh_ingest_baseline_for_stage(
        {"tpch_monetdb/misc/tpch/templates/builder_impl.hpp"}
    )
    assert ingest_calls == []
    assert conv.manifest.get_runtime("ingest_sf1") is None
    assert not hasattr(conv, "_ingest_auxiliary")


def test_manifest_ensure_tpch_instantiations_backfills_partial_manifest(
    tmp_path: Path,
) -> None:
    manifest = ReferenceManifest(tmp_path / "manifest.json")
    manifest.ensure_tpch_instantiations(
        query_ids=["Q1"],
        scale_factor=1,
        seed=1,
        num_instantiations=1,
    )

    added_count = manifest.ensure_tpch_instantiations(
        query_ids=["Q1", "Q6"],
        scale_factor=1,
        seed=1,
        num_instantiations=3,
    )

    assert added_count == 5
    assert len(manifest.get_instantiations_for_query("Q1", scale_factor=1)) == 3
    assert len(manifest.get_instantiations_for_query("Q6", scale_factor=1)) == 3
    assert not hasattr(manifest, "ensure_instantiations_from_validator")


def test_manifest_prunes_out_of_range_instantiation_and_runtime(
    tmp_path: Path,
) -> None:
    """Legacy time-window pruning API should be removed with the TSBS path."""
    _ = tmp_path
    assert not hasattr(ReferenceManifest, "prune_out_of_range_instantiations")


def test_initialize_manifest_backfills_existing_partial_manifest(
    tmp_path: Path,
) -> None:
    from tpch_monetdb.conversations.optimization_conversation_tpch_monetdb import TpchMonetdbOptimizationConversation

    manifest_path = tmp_path / "reference_manifest.json"
    partial_manifest = ReferenceManifest.generate_from_tpch(
        query_ids=["Q1"],
        scale_factor=1,
        seed=1,
        manifest_path=manifest_path,
        num_instantiations=1,
    )
    partial_manifest.save()

    assert not hasattr(ReferenceManifest, "ensure_instantiations_from_validator")

    conv = TpchMonetdbOptimizationConversation.__new__(TpchMonetdbOptimizationConversation)
    conv.query_ids = ["Q1", "Q6"]
    conv.benchmark_sf = 1
    conv.manifest = ReferenceManifest(manifest_path)

    conv._initialize_manifest()

    assert len(conv.manifest.get_instantiations_for_query("Q1", scale_factor=1)) == 3
    assert len(conv.manifest.get_instantiations_for_query("Q6", scale_factor=1)) == 3
    return None


def test_initialize_manifest_backfills_tpch_manifest(
    tmp_path: Path,
) -> None:
    """TPC-H optimization path must not use TPC-H MonetDB validator instantiation."""
    from tpch_monetdb.conversations.optimization_conversation_tpch_monetdb import TpchMonetdbOptimizationConversation

    manifest_path = tmp_path / "reference_manifest.json"
    partial_manifest = ReferenceManifest.generate_from_tpch(
        query_ids=["Q1"],
        scale_factor=1,
        seed=1,
        manifest_path=manifest_path,
        num_instantiations=1,
    )
    partial_manifest.save()

    conv = TpchMonetdbOptimizationConversation.__new__(TpchMonetdbOptimizationConversation)
    conv.benchmark = "tpch"
    conv.query_ids = ["1", "Q6"]
    conv.benchmark_sf = 1
    conv.manifest = ReferenceManifest(manifest_path)

    conv._initialize_manifest()

    q1_instantiations = conv.manifest.get_instantiations_for_query("Q1", scale_factor=1)
    q6_instantiations = conv.manifest.get_instantiations_for_query("Q6", scale_factor=1)
    assert len(q1_instantiations) == 3
    assert len(q6_instantiations) == 3
    for instantiation in [*q1_instantiations, *q6_instantiations]:
        assert instantiation.query_id in {"Q1", "Q6"}
        assert instantiation.args_string.startswith(instantiation.query_id)
        assert "time_start" not in instantiation.params_json
        assert "host" not in instantiation.args_string.lower()
    return None


def test_optimization_conversation_uses_monetdb_provider_for_tpch(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """TPC-H optimization path should initialize a MonetDB baseline provider."""
    from tpch_monetdb.benchmark.providers import MonetDBBaselineProvider
    from tpch_monetdb.conversations.optimization_conversation_tpch_monetdb import TpchMonetdbOptimizationConversation

    monkeypatch.setattr(
        TpchMonetdbOptimizationConversation,
        "_initialize_manifest",
        lambda self: None,
    )

    conv = TpchMonetdbOptimizationConversation(
        benchmark="tpch",
        query_ids=["1"],
        run_tool=object(),
        verify_sf_list=[1],
        benchmark_sf=1,
        git_snapshotter=object(),
        session=object(),
        wandb_run_hook=None,
        manifest_path=tmp_path / "reference_manifest.json",
        conversation_json_path=tmp_path / "conv.json",
        callback=lambda *_args, **_kwargs: None,
        workspace_root=tmp_path,
    )

    assert conv.benchmark == "tpch"
    assert isinstance(conv.baseline_provider, MonetDBBaselineProvider)
    assert conv.baseline_provider.engine == "monetdb"
    assert conv._objective_ids_for_prompt() == ["tpch-docker-monetdb-objective-v1"]
    assert conv._data_law_ids_for_prompt() == [
        "LAW_TPCH_TABLE_CARDINALITY",
        "LAW_TPCH_JOIN_GRAPH",
        "LAW_TPCH_OUTPUT_ORDERING",
        "LAW_TPCH_NUMERIC_TOLERANCE",
        "LAW_TPCH_RUNTIME_BOUNDARY",
    ]
    assert conv._baseline_display_name() == "MonetDB"
    return None


def test_optimization_conversation_defaults_metadata_to_tpch_monetdb() -> None:
    """未完整初始化的 helper 也不能回落到 QuestDB/TPC-H MonetDB metadata。"""
    from tpch_monetdb.conversations.optimization_conversation_tpch_monetdb import TpchMonetdbOptimizationConversation

    conv = TpchMonetdbOptimizationConversation.__new__(TpchMonetdbOptimizationConversation)

    assert conv._objective_ids_for_prompt() == ["tpch-docker-monetdb-objective-v1"]
    assert conv._data_law_ids_for_prompt() == [
        "LAW_TPCH_TABLE_CARDINALITY",
        "LAW_TPCH_JOIN_GRAPH",
        "LAW_TPCH_OUTPUT_ORDERING",
        "LAW_TPCH_NUMERIC_TOLERANCE",
        "LAW_TPCH_RUNTIME_BOUNDARY",
    ]
    assert conv._baseline_display_name() == "MonetDB"
    return None


def test_log_ingest_comparison_skips_after_legacy_ingest_derivation_removal() -> None:
    """QuestDB/TSBS ingest 推导移除后不得发送旧正式 ingest telemetry。"""
    from tpch_monetdb.conversations.optimization_conversation_tpch_monetdb import TpchMonetdbOptimizationConversation

    logged: list[dict[str, Any]] = []

    class FakeHook:
        def log_ingest_comparison(self, **kwargs) -> None:
            logged.append(kwargs)
            return None

    conv = TpchMonetdbOptimizationConversation.__new__(TpchMonetdbOptimizationConversation)
    conv.benchmark_sf = 1
    conv.wandb_run_hook = FakeHook()
    conv._ingest_auxiliary = {
        1: SimpleNamespace(
            runtime_measurement=_make_measurement("ingest_sf1"),
            rows_per_sec=1000.0,
            metrics_per_sec=8000.0,
            row_count=8000,
            metric_count=80000,
            workers=1,
        )
    }

    conv._log_ingest_comparison_if_complete(
        stage_name="stage_end_0",
        validation_metrics={
            "validation/generated_tpch_ingest_ms": 4000.0,
            "validation/generated_tpch_load_ms": 1500.0,
            "validation/generated_tpch_build_ms": 2500.0,
        },
    )

    assert logged == []


def test_log_ingest_comparison_if_complete_skips_incomplete_payload() -> None:
    """full gate 缺字段时不得发正式 ingest telemetry。"""
    from tpch_monetdb.conversations.optimization_conversation_tpch_monetdb import TpchMonetdbOptimizationConversation

    logged: list[dict[str, Any]] = []

    class FakeHook:
        def log_ingest_comparison(self, **kwargs) -> None:
            logged.append(kwargs)
            return None

    conv = TpchMonetdbOptimizationConversation.__new__(TpchMonetdbOptimizationConversation)
    conv.benchmark_sf = 1
    conv.wandb_run_hook = FakeHook()
    conv._ingest_auxiliary = {
        1: SimpleNamespace(
            runtime_measurement=_make_measurement("ingest_sf1"),
            rows_per_sec=None,
            metrics_per_sec=8000.0,
            row_count=8000,
            metric_count=80000,
            workers=1,
        )
    }

    conv._log_ingest_comparison_if_complete(
        stage_name="final_summary",
        validation_metrics={"validation/generated_tpch_ingest_ms": 4000.0},
    )

    assert logged == []
