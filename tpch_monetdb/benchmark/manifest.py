"""Reference Instantiation Manifest 实现.

根据 phase3 D1 设计，manifest 包含查询实例的完整信息：
- query_id
- scale_factor
- instantiation_id
- params_json
- args_string
- sql
- sql_hash
"""

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional, Sequence

logger = logging.getLogger(__name__)
RUNTIME_METADATA_FIELDS = (
    "benchmark_mode",
    "storage_mode",
    "workers",
    "engine",
    "measurement_kind",
    "query_id",
    "args_string",
    "scale_factor",
    "row_count",
    "output_row_count",
    "query_file_sha256",
    "measurement_shape_status",
)

# ---------------------------------------------------------------------------
# Phase9 baseline routing — which baselines to refresh based on change type
# ---------------------------------------------------------------------------

class ChangeType(str, Enum):
    """What category of source files changed in this optimization round."""
    QUERY_ONLY = "query_only"           # only query_impl.* touched
    LOADER_BUILDER_ONLY = "loader_builder_only"  # only loader/builder touched
    ALL = "all"                         # mixed or unknown changes

# Declarative routing table: change_type → set of baseline names that MAY be skipped.
_BASELINE_SKIP_RULES: Dict[str, FrozenSet[str]] = {
    ChangeType.QUERY_ONLY:          frozenset({"ingest_baseline"}),
    ChangeType.LOADER_BUILDER_ONLY: frozenset({"query_baseline"}),
    ChangeType.ALL:                 frozenset(),
}

# Baseline-owned paths that optimization agents must not touch.
BASELINE_OWNED_PATHS: FrozenSet[str] = frozenset({
    "monetdb_oracle.py",
    "monetdb_prepare.py",
    "tpch_validator.py",
})
HOST_OWNED_CONTRACT_PATHS: FrozenSet[str] = frozenset({
    "workload_objective.json",
    "data_law_contract.json",
    "storage_plan_alignment.json",
    "control_artifacts.json",
})


@dataclass(frozen=True)
class BaselineRoutingPolicy:
    """Routing policy: which baselines to skip for a given change type.

    Usage:
        policy = BaselineRoutingPolicy.from_changed_files(changed_paths)
        if policy.should_skip("ingest_baseline"):
            ...
    """
    change_type: ChangeType
    skippable: FrozenSet[str]

    @classmethod
    def from_changed_files(cls, changed_paths: Sequence[str]) -> "BaselineRoutingPolicy":
        """Derive routing policy from the list of changed file paths."""
        change_type = _classify_change(changed_paths)
        return cls(
            change_type=change_type,
            skippable=_BASELINE_SKIP_RULES[change_type],
        )

    def should_skip(self, baseline_name: str) -> bool:
        """Return True if this baseline can be skipped for the current change type."""
        return baseline_name in self.skippable


def _classify_change(changed_paths: Sequence[str]) -> ChangeType:
    """Classify changed paths into a ChangeType for routing decisions."""
    query_patterns = ("query_impl",)
    loader_builder_patterns = ("loader_impl", "builder_impl")

    has_query = any(
        any(p in path for p in query_patterns)
        for path in changed_paths
    )
    has_loader_builder = any(
        any(p in path for p in loader_builder_patterns)
        for path in changed_paths
    )

    if has_query and not has_loader_builder:
        return ChangeType.QUERY_ONLY
    if has_loader_builder and not has_query:
        return ChangeType.LOADER_BUILDER_ONLY
    return ChangeType.ALL


def check_agent_diff_boundary(changed_paths: Sequence[str]) -> List[str]:
    """Return violation paths touching baseline or host-owned contracts.

    If the returned list is non-empty, the agent diff must be rejected.
    """
    protected_paths = BASELINE_OWNED_PATHS | HOST_OWNED_CONTRACT_PATHS
    violations = [
        path for path in changed_paths
        if any(owned in path for owned in protected_paths)
    ]
    return violations


@dataclass(frozen=True)
class QueryInstantiation:
    """单个查询实例的完整描述.
    
    这是 D1 reference instantiation 的核心数据结构。
    """
    query_id: str
    scale_factor: int
    instantiation_id: str
    params_json: Dict[str, Any]
    args_string: str
    sql: str
    sql_hash: str


@dataclass
class RuntimeMeasurement:
    """运行时测量结果.
    
    绑定到特定的 instantiation_id，确保 speedup 计算来自同一实例。
    """
    instantiation_id: str
    runtime_ms: float
    num_runs: int
    all_runtimes_ms: List[float]
    timestamp: str
    benchmark_mode: Optional[str] = None  # "query-latency" or "system-parity"
    storage_mode: Optional[str] = None    # e.g. "tmpfs" or "persistent"
    workers: Optional[int] = None         # ingest/query worker count
    engine: Optional[str] = None          # e.g. "generated_runtime" or "monetdb"
    measurement_kind: Optional[str] = None
    query_id: Optional[str] = None
    args_string: Optional[str] = None
    scale_factor: Optional[int] = None
    row_count: Optional[int] = None
    output_row_count: Optional[int] = None
    query_file_sha256: Optional[str] = None
    measurement_shape_status: str = "unknown"
    provenance: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RuntimeLookup:
    status: str
    measurement: Optional[RuntimeMeasurement]
    reason: Optional[str] = None


class ReferenceManifest:
    """Reference Instantiation Manifest 管理器.
    
    负责：
    1. 生成和保存查询实例化 manifest
    2. 提供实例查询接口
    3. 存储和检索 runtime 测量结果
    """
    
    def __init__(self, manifest_path: Optional[Path] = None):
        """初始化 Manifest.
        
        Args:
            manifest_path: Manifest 文件路径（默认工作目录下的 reference_manifest.json）
        """
        self.manifest_path = manifest_path or Path("reference_manifest.json")
        self._instantiations: Dict[str, QueryInstantiation] = {}
        self._runtimes: Dict[str, RuntimeMeasurement] = {}
        
    def add_instantiation(self, inst: QueryInstantiation) -> None:
        """添加查询实例到 manifest."""
        self._instantiations[inst.instantiation_id] = inst
        logger.debug(f"Added instantiation: {inst.instantiation_id}")
        
    def get_instantiation(self, instantiation_id: str) -> Optional[QueryInstantiation]:
        """获取查询实例."""
        return self._instantiations.get(instantiation_id)
    
    def get_instantiations_for_query(
        self,
        query_id: str,
        scale_factor: Optional[int] = None,
    ) -> List[QueryInstantiation]:
        """获取特定查询的所有实例.

        Args:
            query_id: 查询 ID（如 "1", "2", "Q1", "Q2"）
            scale_factor: 过滤特定 scale factor（可选）
        """
        from tpch_monetdb.dataset.gen_tpch.tpch_queries import get_contract

        try:
            normalized = get_contract(query_id).query_id
        except (ValueError, KeyError):
            normalized = query_id
        results = []
        for inst in self._instantiations.values():
            if inst.query_id in (query_id, normalized):
                if scale_factor is None or inst.scale_factor == scale_factor:
                    results.append(inst)
        return results

    def ensure_tpch_instantiations(
        self,
        query_ids: List[str],
        scale_factor: int,
        seed: int = 42,
        num_instantiations: int = 3,
    ) -> int:
        """Backfill missing TPC-H query instantiations using canonical SQL payloads.

        This is the replacement-path manifest entry point. It stores exact SQL
        and deterministic key=value arguments without legacy host/time-window
        fields, so MonetDB and generated TPC-H runtimes measure the same query.
        """
        from tpch_monetdb.dataset.gen_tpch.gen_tpch_query import instantiate_tpch_query
        from tpch_monetdb.dataset.gen_tpch.tpch_queries import get_contract

        added_count = 0
        for query_id in query_ids:
            normalized_query_id = get_contract(query_id).query_id
            existing_fingerprints = {
                (inst.args_string, inst.sql_hash)
                for inst in self.get_instantiations_for_query(
                    query_id=normalized_query_id,
                    scale_factor=scale_factor,
                )
            }
            for i in range(num_instantiations):
                inst_dict = instantiate_tpch_query(
                    query_id=normalized_query_id,
                    scale_factor=scale_factor,
                    seed=seed + i,
                )
                fingerprint = (inst_dict["args_string"], inst_dict["sql_hash"])
                if fingerprint in existing_fingerprints:
                    continue
                self.add_instantiation(
                    QueryInstantiation(
                        query_id=inst_dict["query_id"],
                        scale_factor=scale_factor,
                        instantiation_id=f"{inst_dict['instantiation_id']}_I{i}",
                        params_json=inst_dict["params_json"],
                        args_string=inst_dict["args_string"],
                        sql=inst_dict["sql"],
                        sql_hash=inst_dict["sql_hash"],
                    )
                )
                existing_fingerprints.add(fingerprint)
                added_count += 1
        return added_count

    def record_runtime(self, measurement: RuntimeMeasurement) -> None:
        """记录 runtime 测量结果并拒绝跨模式覆盖."""
        existing = self._runtimes.get(measurement.instantiation_id)
        merged = self._merge_runtime_measurements(existing, measurement)
        self._runtimes[measurement.instantiation_id] = merged
        logger.debug(
            f"Recorded runtime for {merged.instantiation_id}: "
            f"{merged.runtime_ms:.2f}ms"
        )
        return None
    
    def get_runtime(self, instantiation_id: str) -> Optional[RuntimeMeasurement]:
        """获取 runtime 测量结果."""
        return self._runtimes.get(instantiation_id)

    def lookup_runtime(
        self,
        instantiation_id: str,
        *,
        benchmark_mode: Optional[str],
        storage_mode: Optional[str],
        workers: Optional[int],
        engine: Optional[str],
        measurement_kind: Optional[str] = None,
        query_id: Optional[str] = None,
        args_string: Optional[str] = None,
        scale_factor: Optional[int] = None,
        row_count: Optional[int] = None,
        output_row_count: Optional[int] = None,
        query_file_sha256: Optional[str] = None,
        measurement_shape_status: Optional[str] = None,
        baseline_run_started_at: Optional[str] = None,
        max_age_seconds: Optional[int] = None,
        required_provenance_keys: Sequence[str] = (),
    ) -> RuntimeLookup:
        """Look up a runtime measurement and reject incompatible provenance."""
        measurement = self._runtimes.get(instantiation_id)
        if measurement is None:
            return RuntimeLookup(status="missing", measurement=None, reason="missing")
        if not self._is_runtime_compatible(
            measurement,
            benchmark_mode=benchmark_mode,
            storage_mode=storage_mode,
            workers=workers,
            engine=engine,
            measurement_kind=measurement_kind,
            query_id=query_id,
            args_string=args_string,
            scale_factor=scale_factor,
            row_count=row_count,
            output_row_count=output_row_count,
            query_file_sha256=query_file_sha256,
            measurement_shape_status=measurement_shape_status,
        ):
            return RuntimeLookup(
                status="stale",
                measurement=measurement,
                reason="metadata_mismatch",
            )
        freshness_reason = self._runtime_freshness_reason(
            measurement,
            baseline_run_started_at=baseline_run_started_at,
            max_age_seconds=max_age_seconds,
        )
        if freshness_reason is not None:
            return RuntimeLookup(
                status="stale",
                measurement=measurement,
                reason=freshness_reason,
            )
        missing_keys = [
            key for key in required_provenance_keys
            if key not in (measurement.provenance or {})
        ]
        if missing_keys:
            return RuntimeLookup(
                status="stale",
                measurement=measurement,
                reason=f"missing_provenance:{','.join(missing_keys)}",
            )
        return RuntimeLookup(
            status="compatible",
            measurement=measurement,
            reason="compatible",
        )

    def remove_runtime(self, instantiation_id: str) -> None:
        self._runtimes.pop(instantiation_id, None)
        return None

    def _is_runtime_compatible(
        self,
        measurement: RuntimeMeasurement,
        *,
        benchmark_mode: Optional[str],
        storage_mode: Optional[str],
        workers: Optional[int],
        engine: Optional[str],
        measurement_kind: Optional[str] = None,
        query_id: Optional[str] = None,
        args_string: Optional[str] = None,
        scale_factor: Optional[int] = None,
        row_count: Optional[int] = None,
        output_row_count: Optional[int] = None,
        query_file_sha256: Optional[str] = None,
        measurement_shape_status: Optional[str] = None,
    ) -> bool:
        expected_values = {
            "benchmark_mode": benchmark_mode,
            "storage_mode": storage_mode,
            "workers": workers,
            "engine": engine,
            "measurement_kind": measurement_kind,
            "query_id": query_id,
            "args_string": args_string,
            "scale_factor": scale_factor,
            "row_count": row_count,
            "output_row_count": output_row_count,
            "query_file_sha256": query_file_sha256,
            "measurement_shape_status": measurement_shape_status,
        }
        for field_name, expected_value in expected_values.items():
            existing_value = getattr(measurement, field_name)
            if (
                existing_value is not None
                and expected_value is not None
                and existing_value != expected_value
            ):
                return False
        return True

    def _runtime_freshness_reason(
        self,
        measurement: RuntimeMeasurement,
        *,
        baseline_run_started_at: Optional[str],
        max_age_seconds: Optional[int],
    ) -> Optional[str]:
        """Return a stale reason when a runtime timestamp is outside policy."""
        measured_at = _parse_runtime_timestamp(measurement.timestamp)
        if measured_at is None:
            return "invalid_timestamp"
        if baseline_run_started_at not in (None, ""):
            run_started_at = _parse_runtime_timestamp(baseline_run_started_at)
            if run_started_at is None:
                return "invalid_run_start_timestamp"
            if measured_at < run_started_at:
                return "timestamp_before_run_start"
        if max_age_seconds is not None and max_age_seconds >= 0:
            now = datetime.now(timezone.utc)
            age_seconds = (now - measured_at).total_seconds()
            if age_seconds > max_age_seconds:
                return "timestamp_too_old"
        return None

    def _merge_runtime_measurements(
        self,
        existing: Optional[RuntimeMeasurement],
        incoming: RuntimeMeasurement,
    ) -> RuntimeMeasurement:
        """合并 runtime metadata，并拒绝不同模式/存储/worker/engine 混写.

        规则：
        1. 若旧值不存在，直接接收新值。
        2. 若旧值与新值在 metadata 上都非空且不同，立即报错。
        3. 若旧值缺 metadata 而新值提供了，则用新值补齐。
        4. runtime 数值本身允许被兼容 metadata 的新测量覆盖。
        """
        if existing is None:
            return incoming

        merged_fields: Dict[str, Any] = {}
        for field_name in RUNTIME_METADATA_FIELDS:
            existing_value = getattr(existing, field_name)
            incoming_value = getattr(incoming, field_name)
            if (
                existing_value is not None
                and incoming_value is not None
                and existing_value != incoming_value
            ):
                raise ValueError(
                    "Incompatible runtime metadata for "
                    f"{incoming.instantiation_id}: {field_name} "
                    f"{existing_value!r} != {incoming_value!r}"
                )
            merged_fields[field_name] = (
                incoming_value if incoming_value is not None else existing_value
            )

        return RuntimeMeasurement(
            instantiation_id=incoming.instantiation_id,
            runtime_ms=incoming.runtime_ms,
            num_runs=incoming.num_runs,
            all_runtimes_ms=incoming.all_runtimes_ms,
            timestamp=incoming.timestamp,
            benchmark_mode=merged_fields["benchmark_mode"],
            storage_mode=merged_fields["storage_mode"],
            workers=merged_fields["workers"],
            engine=merged_fields["engine"],
            measurement_kind=merged_fields["measurement_kind"],
            query_id=merged_fields["query_id"],
            args_string=merged_fields["args_string"],
            scale_factor=merged_fields["scale_factor"],
            row_count=merged_fields["row_count"],
            output_row_count=merged_fields["output_row_count"],
            query_file_sha256=merged_fields["query_file_sha256"],
            measurement_shape_status=merged_fields["measurement_shape_status"] or "unknown",
            provenance=dict(incoming.provenance or existing.provenance or {}),
        )
    
    def save(self, path: Optional[Path] = None) -> None:
        """保存 manifest 到文件."""
        save_path = path or self.manifest_path
        
        data = {
            "instantiations": [
                asdict(inst) for inst in self._instantiations.values()
            ],
            "runtimes": [
                asdict(rt) for rt in self._runtimes.values()
            ],
        }
        
        with open(save_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        logger.info(f"Saved manifest to {save_path}")
    
    def load(self, path: Optional[Path] = None) -> None:
        """从文件加载 manifest."""
        load_path = path or self.manifest_path
        
        if not load_path.exists():
            logger.warning(f"Manifest file not found: {load_path}")
            return
        
        with open(load_path, 'r') as f:
            data = json.load(f)
        
        # 加载 instantiations
        for inst_data in data.get("instantiations", []):
            inst = QueryInstantiation(**inst_data)
            self._instantiations[inst.instantiation_id] = inst
        
        # 加载 runtimes
        for rt_data in data.get("runtimes", []):
            rt = RuntimeMeasurement(**rt_data)
            self._runtimes[rt.instantiation_id] = rt
        
        logger.info(
            f"Loaded manifest from {load_path}: "
            f"{len(self._instantiations)} instantiations, "
            f"{len(self._runtimes)} runtime measurements"
        )
    
    @classmethod
    def generate_from_tpch(
        cls,
        query_ids: List[str],
        scale_factor: int,
        seed: int = 42,
        manifest_path: Optional[Path] = None,
        num_instantiations: int = 3,
    ) -> "ReferenceManifest":
        """Generate a manifest from canonical TPC-H query instantiations."""
        manifest = cls(manifest_path=manifest_path)
        manifest.ensure_tpch_instantiations(
            query_ids=query_ids,
            scale_factor=scale_factor,
            seed=seed,
            num_instantiations=num_instantiations,
        )
        return manifest


def _parse_runtime_timestamp(value: Optional[str]) -> Optional[datetime]:
    """Parse manifest timestamps as UTC-aware datetimes."""
    if value in (None, ""):
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)
