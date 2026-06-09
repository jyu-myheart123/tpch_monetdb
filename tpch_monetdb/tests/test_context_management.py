from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

from tpch_monetdb.llm_cache.artifact_ledger import ArtifactLedger
from tpch_monetdb.llm_cache.auto_compact import AutoCompactManager
from tpch_monetdb.llm_cache.cached_litellm import CachedLitellmModel
from tpch_monetdb.llm_cache.cached_litellm_compaction import (
    format_compaction_summary_v3,
    validate_compaction_summary_v3,
)
from tpch_monetdb.llm_cache.context_budget import (
    build_provider_request_budget_estimate,
    build_request_budget_estimate,
    estimate_json_bytes,
)
from tpch_monetdb.llm_cache.logger import setup_logging
from tpch_monetdb.llm_cache.micro_compact import micro_compact_tool_results
from tpch_monetdb.llm_cache.prompt_cache_diagnostics import extract_provider_cache_tokens
from tpch_monetdb.llm_cache.stage_memory import STAGE_MEMORY_HEADER, render_stage_memory
from tpch_monetdb.utils.duration_format import format_duration_ms, safe_speedup
from tpch_monetdb.utils.large_data_objectives import build_workload_objective_payload
from tpch_monetdb.tools.tpch_monetdb_agent_tools import StageToolRuntime


def test_artifact_ledger_records_text_and_renders_digest(tmp_path: Path) -> None:
    ledger = ArtifactLedger(tmp_path / "context_artifacts")
    artifact = ledger.record_text(
        kind="run_output",
        text="line 1\n" + "x" * 2000,
        metadata={
            "stage_name": "optimization",
            "prompt_index": 3,
            "tool_name": "run",
            "query_ids": ("9",),
            "success": False,
            "summary": "Q9 runtime regression",
        },
    )

    digest = ledger.render_digest(artifact, preview="line 1", omitted_chars=2000)
    refs = ledger.refs_for_prompt(query_ids=("9",), stage_name="optimization")

    assert Path(artifact.path).exists()
    assert (tmp_path / "context_artifacts" / "ledger.jsonl").exists()
    assert "artifact_ref:" in digest
    assert artifact.prompt_ref() in digest
    assert "Q9 runtime regression" in refs
    assert "artifact_ref=" in refs
    assert "path=" not in refs
    assert "sha256=" not in refs


def test_setup_logging_suppresses_pymonetdb_debug_noise(tmp_path: Path) -> None:
    """setup_logging 默认不应放出 pymonetdb MAPI DEBUG SQL 噪声."""
    setup_logging(logging.DEBUG, tmp_path / "run.log")
    try:
        assert logging.getLogger("pymonetdb").getEffectiveLevel() == logging.WARNING
        assert logging.getLogger("pymonetdb.mapi").getEffectiveLevel() == logging.WARNING
    finally:
        logging.basicConfig(level=logging.WARNING, force=True)


def test_artifact_ledger_loads_existing_records_and_records_files(tmp_path: Path) -> None:
    ledger = ArtifactLedger(tmp_path / "context_artifacts")
    source = tmp_path / "large.log"
    source.write_text("hello artifact\n", encoding="utf-8")
    artifact = ledger.record_file(
        path=source,
        kind="run_output",
        metadata={"tool_name": "run", "query_ids": ("1",), "summary": "q1 log"},
    )

    reloaded = ArtifactLedger(tmp_path / "context_artifacts")

    assert reloaded.lookup(artifact.artifact_id).sha256 == artifact.sha256
    assert reloaded.lookup_ref(artifact.prompt_ref()).sha256 == artifact.sha256
    assert reloaded.top_contributors(limit=1)[0].artifact_id == artifact.artifact_id


def test_artifact_ledger_cleanup_prunes_old_run_artifacts_but_keeps_refs(
    tmp_path: Path,
) -> None:
    ledger = ArtifactLedger(tmp_path / "context_artifacts")
    old_artifact = ledger.record_text(kind="run_output", text="old")
    kept_artifact = ledger.record_text(kind="run_output", text="kept")
    newest_artifact = ledger.record_text(kind="run_output", text="newest")

    pruned = ledger.cleanup_retention(
        keep_artifact_ids=(kept_artifact.artifact_id,),
        max_artifacts=2,
    )

    assert [artifact.artifact_id for artifact in pruned] == [old_artifact.artifact_id]
    assert not Path(old_artifact.path).exists()
    assert Path(kept_artifact.path).exists()
    assert Path(newest_artifact.path).exists()
    reloaded = ArtifactLedger(tmp_path / "context_artifacts")
    assert [artifact.artifact_id for artifact in reloaded.artifacts()] == [
        kept_artifact.artifact_id,
        newest_artifact.artifact_id,
    ]


def test_artifact_ledger_scope_ids_match_relevance_order(tmp_path: Path) -> None:
    ledger = ArtifactLedger(tmp_path / "context_artifacts")
    old_artifact = ledger.record_text(
        kind="run_output",
        text="old",
        metadata={"stage_name": "compile_fix", "query_ids": ("1",)},
    )
    kept_artifact = ledger.record_text(
        kind="run_output",
        text="q9 failure",
        metadata={
            "stage_name": "optimization",
            "query_ids": ("9",),
            "success": False,
        },
    )
    newest_artifact = ledger.record_text(
        kind="run_output",
        text="new",
        metadata={"stage_name": "optimization", "query_ids": ("2",)},
    )

    keep_ids = ledger.artifact_ids_for_scope(
        max_entries=1,
        query_ids=("9",),
        stage_name="optimization",
    )
    pruned = ledger.cleanup_retention(
        keep_artifact_ids=keep_ids,
        max_artifacts=1,
    )

    assert keep_ids == (kept_artifact.artifact_id,)
    assert [artifact.artifact_id for artifact in pruned] == [
        old_artifact.artifact_id,
        newest_artifact.artifact_id,
    ]
    assert Path(kept_artifact.path).exists()


def test_artifact_ledger_record_does_not_prune_visible_ref_before_stage_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TPCH_MONETDB_CONTEXT_ARTIFACT_MAX_COUNT", "1")
    ledger = ArtifactLedger(tmp_path / "context_artifacts")
    visible_artifact = ledger.record_text(kind="run_output", text="visible q9")
    stale_artifact = ledger.record_text(kind="run_output", text="stale q8")

    assert Path(visible_artifact.path).exists()
    assert Path(stale_artifact.path).exists()

    keep_ids = ledger.artifact_ids_for_refs((visible_artifact.prompt_ref(),))
    pruned = ledger.cleanup_default_retention(keep_artifact_ids=keep_ids)

    assert Path(visible_artifact.path).exists()
    assert keep_ids == (visible_artifact.artifact_id,)
    assert [artifact.artifact_id for artifact in pruned] == [stale_artifact.artifact_id]
    assert not Path(stale_artifact.path).exists()


def test_artifact_ledger_refs_prefer_newest_equal_relevance(tmp_path: Path) -> None:
    ledger = ArtifactLedger(tmp_path / "context_artifacts")
    older = ledger.record_text(
        kind="run_output",
        text="older q9 failure",
        metadata={
            "stage_name": "optimization",
            "prompt_index": 1,
            "tool_name": "run",
            "query_ids": ("9",),
            "success": False,
        },
    )
    newer = ledger.record_text(
        kind="run_output",
        text="newer q9 failure",
        metadata={
            "stage_name": "optimization",
            "prompt_index": 7,
            "tool_name": "run",
            "query_ids": ("9",),
            "success": False,
        },
    )

    keep_ids = ledger.artifact_ids_for_scope(
        max_entries=1,
        query_ids=("9",),
        stage_name="optimization",
    )

    assert keep_ids == (newer.artifact_id,)
    assert keep_ids != (older.artifact_id,)


def test_context_budget_uses_json_bytes_and_reports_top_contributor() -> None:
    items = [
        {"role": "user", "content": "small"},
        {"type": "function_call_output", "output": "x" * 20_000},
    ]

    estimate = build_request_budget_estimate(
        items,
        new_input="task",
        token_limit=100_000,
        body_warn_bytes=1_000,
        body_compact_bytes=10_000,
        body_fail_bytes=100_000,
    )

    assert estimate.body_bytes == estimate_json_bytes({"input": items, "new_input": "task"})
    assert estimate.body_level == "orange"
    assert estimate.top_contributors[0].source == "tool_output"
    assert estimate.should_compact is True
    assert estimate.should_fail is False


def test_provider_body_budget_uses_converted_payload_shape() -> None:
    payload = {
        "model": "deepseek-v4-pro",
        "messages": [{"role": "user", "content": "x" * 2000}],
        "tools": [{"type": "function", "function": {"name": "run"}}],
        "thinking": {"type": "enabled"},
    }

    estimate = build_provider_request_budget_estimate(
        payload,
        token_limit=100_000,
        body_warn_bytes=100,
        body_compact_bytes=1000,
        body_fail_bytes=10_000,
    )
    body_payload = CachedLitellmModel._provider_body_payload({
        **payload,
        "api_key": "secret",
        "base_url": "https://example.invalid",
        "extra_headers": {
            "Authorization": "Bearer secret-token",
            "X-Trace": "trace-id",
        },
    })
    header_estimate = build_provider_request_budget_estimate(
        body_payload,
        token_limit=100_000,
        body_warn_bytes=1,
        body_compact_bytes=100_000,
        body_fail_bytes=200_000,
    )

    assert estimate.body_compact is True
    assert "api_key" not in body_payload
    assert "base_url" not in body_payload
    assert body_payload["__http_headers__"]["Authorization"] == "*" * len("Bearer secret-token")
    assert body_payload["__http_headers__"]["X-Trace"] == "trace-id"
    assert any(item.source == "provider.headers" for item in header_estimate.top_contributors)
    assert "provider.messages.user" in estimate.top_contributors[0].source


def test_auto_compact_body_compact_env_prefers_new_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TPCH_MONETDB_BODY_BLOCKING_BYTES", "111")
    monkeypatch.setenv("TPCH_MONETDB_BODY_COMPACT_BYTES", "222")
    manager = AutoCompactManager("anthropic/deepseek-v4-pro")

    assert manager.get_body_compact_threshold() == 222
    assert manager.get_body_blocking_threshold() == 222


def test_stage_memory_renders_stable_fields() -> None:
    state = SimpleNamespace(
        profile_name="optimization",
        prompt_index=7,
        prompt_descriptor="Q9 optimization",
        active_query_ids=("9",),
        active_unit_id="q9_unit",
        active_unit_kind="query_family",
        active_unit_files=("query_family_double_groupby.cpp",),
        active_unit_query_ids=("9",),
        objective_ids=("critical_q9",),
        data_law_ids=("law_host_hour",),
        patch_scope_verdict="query_local",
        last_compile_succeeded=True,
        last_compile_summary="ok",
        last_run_succeeded=False,
        last_run_summary="runtime regression",
        validation_passed=False,
        last_validation_summary="gate failed",
        last_failure_kind="run",
        written_files={"query_family_double_groupby.cpp"},
        control_artifacts_read={"workload_objective.json"},
        control_artifacts_injected=("optimization_hotspot_summary.md",),
        required_control_artifacts=("workload_objective.json",),
    )

    rendered = render_stage_memory(
        state,
        artifact_refs="[Artifact Refs]\n- artifact_ref=abc kind=run_output",
    )

    assert rendered.startswith(STAGE_MEMORY_HEADER)
    assert "schema_version: 3" in rendered
    assert 'profile_name: "optimization"' in rendered
    assert 'query_ids: ["9"]' in rendered
    assert "runtime regression" in rendered
    assert "open_failures:" in rendered
    assert 'artifact_refs: ["abc"]' in rendered
    assert "OBLIG_Q9_JOIN_PROFIT_AGGREGATION" in rendered
    assert "reusable part/supplier/partsupp/lineitem/orders/nation join support" in rendered
    assert "aggregate profit at query time" in rendered
    assert "artifact_" + "manifest:" not in rendered
    assert "snapshot_hash:" in rendered
    assert "inspect the latest run failure" in rendered


def test_stage_tool_runtime_artifactizes_large_tool_evidence(tmp_path: Path) -> None:
    ledger = ArtifactLedger(tmp_path / "cache" / "context_artifacts")
    runtime = StageToolRuntime(tmp_path, artifact_ledger=ledger)
    runtime.activate(
        "default_general",
        2,
        "Q9 validation",
        {"active_query_ids": ("9",)},
    )

    evidence = runtime.prepare_tool_evidence(
        tool_name="run",
        output="validation failed\n" + ("x" * 20_000),
        success=False,
        inline_limit=1_000,
        kind="run_output",
    )

    assert evidence.startswith("validation failed")
    assert "artifact_ref:" in evidence
    assert len(ledger.artifacts()) == 1
    assert Path(ledger.artifacts()[0].path).exists()
    reread = runtime.read_artifact(ledger.artifacts()[0].prompt_ref(), offset=1, limit=1)
    assert "validation failed" in reread


def test_stage_tool_runtime_read_artifact_defaults_to_bounded_slice(
    tmp_path: Path,
) -> None:
    ledger = ArtifactLedger(tmp_path / "cache" / "context_artifacts")
    runtime = StageToolRuntime(tmp_path, artifact_ledger=ledger)
    runtime.activate("default_general", 2, "Q9 validation", {"active_query_ids": ("9",)})
    artifact = ledger.record_text(
        kind="run_output",
        text="\n".join(f"line-{index}" for index in range(500)),
        metadata={"tool_name": "run", "query_ids": ("9",)},
    )

    reread = runtime.read_artifact(artifact.prompt_ref(), offset=None, limit=None)

    assert "bounded_read=true" in reread
    assert "line-0" in reread
    assert "line-199" in reread
    assert "line-250" not in reread


def test_stage_tool_runtime_read_artifact_truncates_oversized_line(
    tmp_path: Path,
) -> None:
    ledger = ArtifactLedger(tmp_path / "cache" / "context_artifacts")
    runtime = StageToolRuntime(tmp_path, artifact_ledger=ledger)
    runtime.activate("default_general", 2, "Q9 validation", {"active_query_ids": ("9",)})
    artifact = ledger.record_text(
        kind="run_output",
        text="x" * 50_000,
        metadata={"tool_name": "run", "query_ids": ("9",)},
    )

    reread = runtime.read_artifact(artifact.prompt_ref(), offset=None, limit=None)

    assert len(reread.encode("utf-8")) < 20_000
    assert "line truncated by read budget" in reread


def test_micro_compact_preserves_artifact_ref_in_digest_outputs() -> None:
    items = []
    for index in range(12):
        call_id = f"call_{index}"
        items.append({"type": "function_call", "call_id": call_id, "name": "run"})
        items.append(
            {
                "type": "function_call_output",
                "call_id": call_id,
                    "output": (
                        f"[Evidence Digest]\nartifact_ref: run_{index}\n"
                        f"sha256: hash_{index}"
                    ),
                }
            )

    compacted = micro_compact_tool_results(items)

    cleared_outputs = [
        item["output"]
        for item in compacted
        if item.get("type") == "function_call_output" and item.get("_compacted")
    ]
    assert cleared_outputs
    assert all("artifact_ref=run_" in output for output in cleared_outputs)
    assert all("sha256=hash_" in output for output in cleared_outputs)
    assert all("path=" not in output for output in cleared_outputs)


def test_deterministic_trim_artifactizes_large_raw_tool_output(tmp_path: Path) -> None:
    ledger = ArtifactLedger(tmp_path / "context_artifacts")
    manager = AutoCompactManager("anthropic/deepseek-v4-pro", artifact_ledger=ledger)
    items = [
        {"type": "function_call", "call_id": "run_1", "name": "run"},
        {"type": "function_call_output", "call_id": "run_1", "output": "x" * 50_000},
    ]

    result = manager.deterministic_trim_items(items, profile_name="optimization")

    assert result.changed_count == 1
    assert result.bytes_after < result.bytes_before
    assert "artifact_ref:" in result.items[1]["output"]
    assert Path(ledger.artifacts()[0].path).exists()


def test_stage_end_maintenance_trims_without_unneeded_llm_compaction(
    tmp_path: Path,
) -> None:
    ledger = ArtifactLedger(tmp_path / "context_artifacts")
    manager = AutoCompactManager("anthropic/deepseek-v4-pro", artifact_ledger=ledger)

    class FakeSession:
        def __init__(self) -> None:
            self.items = [
                {"type": "function_call", "call_id": "run_1", "name": "run"},
                {"type": "function_call_output", "call_id": "run_1", "output": "x" * 50_000},
            ]
            return None

        async def get_items(self) -> list[dict[str, str]]:
            return list(self.items)

        async def clear_session(self) -> None:
            self.items.clear()
            return None

        async def add_items(self, items) -> None:
            self.items.extend(items)
            return None

    class FakeCompactionSession:
        def __init__(self) -> None:
            self.calls = 0
            return None

        async def run_compaction(self, _args) -> None:
            self.calls += 1
            return None

    session = FakeSession()
    compaction_session = FakeCompactionSession()

    result = asyncio.run(
        manager.maintain_after_stage(
            session=session,
            compaction_session=compaction_session,
            profile_name="optimization",
            stage_name="Q9 optimization",
            query_ids=("9",),
        )
    )

    assert result.deterministic_trimmed_items == 1
    assert result.llm_compaction_attempted is False
    assert compaction_session.calls == 0
    assert result.post_budget.body_bytes < result.pre_budget.body_bytes
    assert "artifact_ref:" in session.items[1]["output"]


def test_stage_end_maintenance_skips_stage_memory_compact_on_green_budget(
    tmp_path: Path,
) -> None:
    manager = AutoCompactManager("anthropic/deepseek-v4-pro")

    class FakeSession:
        def __init__(self) -> None:
            self.items = [
                {"role": "user", "content": "[Stage Memory v3]\nactive_scope: []"},
                *[
                    {"role": "assistant", "content": f"old-{index} " + ("x" * 800)}
                    for index in range(80)
                ],
            ]
            return None

        async def get_items(self) -> list[dict[str, str]]:
            return list(self.items)

        async def clear_session(self) -> None:
            self.items.clear()
            return None

        async def add_items(self, items) -> None:
            self.items.extend(items)
            return None

    class FakeCompactionSession:
        async def run_compaction(self, _args) -> None:
            return None

    calls = {"count": 0}
    original = manager.stage_memory_compact_items

    def counted_stage_memory_compact(*args, **kwargs) -> object:
        calls["count"] += 1
        return original(*args, **kwargs)

    manager.stage_memory_compact_items = counted_stage_memory_compact

    result = asyncio.run(
        manager.maintain_after_stage(
            session=FakeSession(),
            compaction_session=FakeCompactionSession(),
            profile_name="optimization",
            stage_name="Q9 optimization",
            query_ids=("9",),
        )
    )

    assert result.pre_budget.should_compact is False
    assert result.post_budget.should_compact is False
    assert calls["count"] == 0


def test_stage_end_maintenance_runs_llm_compaction_on_orange_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TPCH_MONETDB_BODY_COMPACT_BYTES", "100")
    manager = AutoCompactManager("anthropic/deepseek-v4-pro")

    class FakeSession:
        async def get_items(self) -> list[dict[str, str]]:
            return [{"role": "user", "content": "x" * 500}]

    class FakeCompactionSession:
        def __init__(self) -> None:
            self.args: list[dict[str, object]] = []
            return None

        async def run_compaction(self, args) -> None:
            self.args.append(dict(args))
            return None

    compaction_session = FakeCompactionSession()

    result = asyncio.run(
        manager.maintain_after_stage(
            session=FakeSession(),
            compaction_session=compaction_session,
            profile_name="optimization",
            stage_name="Q9 optimization",
            query_ids=("9",),
        )
    )

    assert result.llm_compaction_attempted is True
    assert result.llm_compaction_succeeded is True
    assert compaction_session.args[0]["selection_policy"] == "stage_memory_v3"


def test_stage_memory_compact_replaces_stale_dialogue_with_bounded_state() -> None:
    manager = AutoCompactManager("anthropic/kimi-k2.5")
    items = [
        {"role": "system", "content": "rules"},
        *[
            {"role": "assistant", "content": f"old-{index} " + ("x" * 4000)}
            for index in range(30)
        ],
    ]

    result = manager.stage_memory_compact_items(
        items,
        stage_memory="[Stage Memory v3]\nactive_scope:\n  query_ids: [\"9\"]",
        artifact_context="[Artifact Refs]\n- artifact_ref=run_q9",
        profile_name="optimization",
    )

    rendered = "\n".join(
        str(item.get("content", ""))
        for item in result.items
        if isinstance(item, dict)
    )
    assert result.changed_count == 1
    assert result.post_tokens < result.pre_tokens
    assert "[Compact Boundary]" in rendered
    assert "TPC-H MonetDB Context Lifecycle v3" in rendered
    assert "[Stage Memory v3]" in rendered
    assert "artifact_ref=run_q9" in rendered


def test_stage_memory_compact_skips_first_tail_item_over_token_cap() -> None:
    manager = AutoCompactManager("anthropic/kimi-k2.5")
    items = [
        {"role": "system", "content": "rules"},
        {"role": "assistant", "content": "old-small"},
        {"role": "assistant", "content": "x" * 180_000},
    ]

    result = manager.stage_memory_compact_items(
        items,
        stage_memory="[Stage Memory v3]\nactive_scope:\n  query_ids: [\"9\"]",
        artifact_context="[Artifact Refs]\n- artifact_ref=run_q9",
        profile_name="optimization",
    )
    rendered = "\n".join(
        str(item.get("content", ""))
        for item in result.items
        if isinstance(item, dict)
    )

    assert result.changed_count == 1
    assert "old-small" in rendered
    assert "x" * 1000 not in rendered


def test_stage_memory_compact_preserves_tail_tool_pairs_atomically() -> None:
    manager = AutoCompactManager("anthropic/kimi-k2.5")
    items = [
        {"role": "system", "content": "rules"},
        *[
            {"role": "assistant", "content": f"old-{index} " + ("x" * 4000)}
            for index in range(30)
        ],
        {"type": "function_call", "call_id": "run_1", "name": "run", "arguments": "{}"},
        {
            "type": "function_call_output",
            "call_id": "run_1",
            "output": "Validation failed: row count mismatch",
        },
    ]

    result = manager.stage_memory_compact_items(
        items,
        stage_memory="[Stage Memory v3]\nactive_scope:\n  query_ids: [\"9\"]",
        artifact_context="[Artifact Refs]\n- artifact_ref=run_q9",
        profile_name="optimization",
    )

    call_ids = [
        item.get("call_id")
        for item in result.items
        if isinstance(item, dict) and item.get("type") == "function_call"
    ]
    output_ids = [
        item.get("call_id")
        for item in result.items
        if isinstance(item, dict) and item.get("type") == "function_call_output"
    ]
    assert "run_1" in call_ids
    assert "run_1" in output_ids


def test_context_budget_splits_generated_prompt_contributors() -> None:
    budget = build_request_budget_estimate(
        [],
        new_input=(
            "[Runtime Stage Hint]\nrules\n\n"
            "[Scoped Stage Rules]\n[Rule File: scripted.md]\nkeep scope\n\n"
            "[Current Task]\nfix q9\n\n"
            "[Stage Memory v3]\nopen_failures: []\n\n"
            "[Artifact Refs]\n- artifact_ref=abc"
        ),
        token_limit=100_000,
    )

    sources = {item.source for item in budget.top_contributors}
    assert {
        "runtime_stage_hint",
        "scoped_stage_rules",
        "new_input",
        "stage_memory",
        "artifact_refs",
    }.issubset(sources)


def test_compaction_summary_v3_requires_source_refs() -> None:
    summary = format_compaction_summary_v3(
        "Q9 regression was caused by stale aggregate evidence.",
        source_refs=("item:0", "artifact_a"),
    )

    validate_compaction_summary_v3(summary)
    assert summary.startswith("[Compaction Summary v3]")
    assert "source_refs:" in summary
    assert "q1_q9_obligations:" in summary
    with pytest.raises(ValueError):
        validate_compaction_summary_v3("[Compaction Summary v3]\ndecisions:\n  missing refs")


def test_duration_format_and_safe_speedup_handle_tiny_and_invalid_values() -> None:
    assert format_duration_ms(0.0004) == "400ns"
    assert format_duration_ms(float("nan")) == "invalid"
    assert safe_speedup(10.0, 0.0) is None
    assert safe_speedup(10.0, 2.0) == 5.0


def test_workload_objective_makes_q1_critical_and_q9_stricter() -> None:
    payload = build_workload_objective_payload(
        query_ids=["1", "9", "12"],
        benchmark_sf=1000,
        large_sf=1000,
        hardware_counter_backend=None,
        target_cpu=None,
    )

    assert payload["critical_query_ids"] == ["1", "9", "12"]
    assert payload["critical_query_targets"]["1"]["min_speedup_vs_baseline"] == 1.0
    assert payload["critical_query_targets"]["9"]["min_speedup_vs_base_impl"] == 1.05


def test_prompt_cache_diagnostics_extracts_deepseek_raw_cache_tokens() -> None:
    usage = SimpleNamespace(
        input_tokens=0,
        input_tokens_details=SimpleNamespace(cached_tokens=0),
        prompt_cache_hit_tokens=1234,
        prompt_cache_miss_tokens=66,
        request_usage_entries=[],
    )

    input_tokens, cached_tokens = extract_provider_cache_tokens(usage)

    assert input_tokens == 1300
    assert cached_tokens == 1234
