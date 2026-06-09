"""TPC-H MonetDB Validate Cache 实现.

提供 validation 结果的缓存和 replay 功能，支持 only_from_cache 语义。
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Optional

from tpch_monetdb.benchmark.runtime_accounting import RUNTIME_SCHEMA_VERSION

logger = logging.getLogger(__name__)

CachedValidationResult = tuple[str, bool, dict[str, Any], bool]


class ValidateCacheEntry:
    """单个验证缓存条目."""
    
    def __init__(
        self,
        cache_key: str,
        success: bool,
        msg: str,
        metrics: dict[str, Any],
        oracle_result_hash: str,
    ) -> None:
        self.cache_key = cache_key
        self.success = success
        self.msg = msg
        self.metrics = metrics
        self.oracle_result_hash = oracle_result_hash
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "cache_key": self.cache_key,
            "success": self.success,
            "msg": self.msg,
            "metrics": self.metrics,
            "oracle_result_hash": self.oracle_result_hash,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ValidateCacheEntry":
        return cls(
            cache_key=data["cache_key"],
            success=data["success"],
            msg=data["msg"],
            metrics=data["metrics"],
            oracle_result_hash=data["oracle_result_hash"],
        )


class TpchValidateCache:
    """TPC-H MonetDB 验证缓存管理器.
    
    Cache key 包含:
    - compile_key_hash (代码身份)
    - query_id (查询 ID)
    - scale_factor (数据规模)
    - query_params_hash (参数实例)
    - dataset_key (数据集/基线身份)
    """
    
    def __init__(self, cache_dir: Path | None = None) -> None:
        """初始化缓存.
        
        Args:
            cache_dir: 缓存目录，默认为 ./tpch_monetdb_artifacts/cache/validate_cache
        """
        if cache_dir is None:
            cache_dir = Path("./tpch_monetdb_artifacts/cache/validate_cache")
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _compute_cache_key(
        self,
        compile_key_hash: str,
        query_id: str,
        scale_factor: float,
        params: dict[str, Any],
        validator_config: dict[str, Any] | None = None,
    ) -> str:
        """计算缓存 key.
        
        Args:
            compile_key_hash: 编译缓存键（代码身份）
            query_id: 查询 ID
            scale_factor: 数据规模因子
            params: 查询参数字典
            
        Returns:
            缓存 key 字符串
        """
        normalized_sf = int(scale_factor)
        if normalized_sf != scale_factor:
            raise ValueError(f"scale_factor must be integral, got {scale_factor!r}")
        config = validator_config or {}
        data_config = config.get("data_config")
        dataset_key = config.get("dataset_key") or config.get("benchmark")
        if dataset_key is None and isinstance(data_config, dict):
            dataset_key = data_config.get("benchmark") or data_config.get("dataset")
        if dataset_key is None:
            dataset_key = "default"
        
        key_components = {
            "compile_key_hash": compile_key_hash,
            "query_id": query_id,
            "scale_factor": normalized_sf,
            "dataset_key": dataset_key,
            "params": params,
            "validator_config": config,
        }
        
        key_str = json.dumps(key_components, sort_keys=True, default=str)
        return hashlib.sha256(key_str.encode()).hexdigest()[:32]
    
    def _get_cache_file_path(self, cache_key: str) -> Path:
        """获取缓存文件路径."""
        return self.cache_dir / f"{cache_key}.json"
    
    def get(
        self,
        compile_key_hash: str,
        query_id: str,
        scale_factor: float,
        params: dict[str, Any],
        validator_config: dict[str, Any] | None = None,
    ) -> Optional[ValidateCacheEntry]:
        """获取缓存条目.
        
        Args:
            compile_key_hash: 编译缓存键
            query_id: 查询 ID
            scale_factor: 数据规模因子
            params: 查询参数
            
        Returns:
            缓存条目，如果不存在则返回 None
        """
        cache_key = self._compute_cache_key(
            compile_key_hash,
            query_id,
            scale_factor,
            params,
            validator_config,
        )
        cache_file = self._get_cache_file_path(cache_key)
        
        if not cache_file.exists():
            return None
        
        try:
            with open(cache_file) as f:
                data = json.load(f)
            return ValidateCacheEntry.from_dict(data)
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to load cache entry {cache_key}: {e}")
            return None
    
    def put(
        self,
        compile_key_hash: str,
        query_id: str,
        scale_factor: float,
        params: dict[str, Any],
        success: bool,
        msg: str,
        metrics: dict[str, Any],
        oracle_result_hash: str,
        validator_config: dict[str, Any] | None = None,
    ) -> None:
        """写入缓存条目.
        
        Args:
            compile_key_hash: 编译缓存键
            query_id: 查询 ID
            scale_factor: 数据规模因子
            params: 查询参数
            success: 验证是否成功
            msg: 验证消息
            metrics: 验证指标
            oracle_result_hash: oracle 结果哈希
        """
        cache_key = self._compute_cache_key(
            compile_key_hash,
            query_id,
            scale_factor,
            params,
            validator_config,
        )
        
        entry = ValidateCacheEntry(
            cache_key=cache_key,
            success=success,
            msg=msg,
            metrics=metrics,
            oracle_result_hash=oracle_result_hash,
        )
        
        cache_file = self._get_cache_file_path(cache_key)
        try:
            with open(cache_file, "w") as f:
                json.dump(entry.to_dict(), f, indent=2)
            logger.debug(f"Cached validation result: {cache_key}")
        except Exception as e:
            logger.warning(f"Failed to write cache entry {cache_key}: {e}")
    
    def get_batch(
        self,
        compile_key_hash: str,
        query_ids: list[str],
        scale_factor: float,
        params_list: list[dict[str, Any]],
        validator_config: dict[str, Any] | None = None,
    ) -> tuple[Optional[list[ValidateCacheEntry]], list[str]]:
        """批量获取缓存条目.
        
        Args:
            compile_key_hash: 编译缓存键
            query_ids: 查询 ID 列表
            scale_factor: 数据规模因子
            params_list: 查询参数列表（与 query_ids 一一对应）
            
        Returns:
            (缓存条目列表, 缺失的 query_ids) 元组
            如果有任何条目缺失，返回 (None, 缺失的 query_ids)
        """
        entries = []
        missing = []
        
        for qid, params in zip(query_ids, params_list):
            entry = self.get(
                compile_key_hash,
                qid,
                scale_factor,
                params,
                validator_config,
            )
            if entry is None:
                missing.append(qid)
            else:
                entries.append(entry)
        
        if missing:
            return None, missing
        
        return entries, []
    
    def clear(self) -> None:
        """清除所有缓存."""
        for cache_file in self.cache_dir.glob("*.json"):
            cache_file.unlink()
        logger.info(f"Cleared validate cache: {self.cache_dir}")


class CacheMissError(Exception):
    """缓存未找到错误（用于 only_from_cache=True 时）."""
    
    def __init__(self, query_id: str, scale_factor: float) -> None:
        self.query_id = query_id
        self.scale_factor = scale_factor
        super().__init__(
            f"Cache miss for query {query_id} at scale_factor {scale_factor}. "
            f"only_from_cache=True, cannot proceed without cache."
        )


def merge_cached_validation_metrics(
    scale_factor: float,
    query_ids: list[str],
    entries: list[ValidateCacheEntry],
) -> dict[str, Any]:
    """Merge per-query cached metrics without losing aggregate query identity."""
    metrics: dict[str, Any] = {
        "validation/scale_factor": int(scale_factor),
        "validation/correct": all(entry.success for entry in entries),
        "validation/error": not all(entry.success for entry in entries),
        "validation/query_ids_executed": query_ids,
        "validation/num_queries": len(query_ids),
        "validation/num_successful_queries": sum(
            1 for entry in entries if entry.success
        ),
        "validation/used_cache": True,
    }
    for entry in entries:
        for key, value in entry.metrics.items():
            if key.startswith("validation/query_"):
                metrics[key] = value
    return metrics


def build_validate_cache_context(
    *,
    validation_mode: str,
    trace_mode: bool,
    other_config: dict[str, Any] | None,
    data_config: dict[str, Any],
    allowed_query_ids: list[str],
    oracle_http_url: str,
    oracle_timeout_s: int,
    output_stdout_stderr: bool,
) -> dict[str, Any]:
    return {
        "validation_mode": validation_mode,
        "trace_mode": trace_mode,
        "other_config": other_config or {},
        "data_config": data_config,
        "allowed_query_ids": allowed_query_ids,
        "oracle_http_url": oracle_http_url,
        "oracle_timeout_s": oracle_timeout_s,
        "output_stdout_stderr": output_stdout_stderr,
        "runtime_schema_version": RUNTIME_SCHEMA_VERSION,
    }


def get_cached_validation_result(
    *,
    cache: TpchValidateCache,
    compile_key_hash: str,
    query_ids: list[str],
    scale_factor: float,
    params_list: list[dict[str, Any]],
    validator_config: dict[str, Any],
    only_from_cache: bool,
    skip_cache: bool = False,
) -> CachedValidationResult | None:
    """Return cached validation tuple, or None when fresh execution is allowed."""
    if skip_cache:
        return None
    if not compile_key_hash:
        if only_from_cache:
            raise CacheMissError(query_ids[0], scale_factor)
        return None

    cache_entries, missing_query_ids = cache.get_batch(
        compile_key_hash=compile_key_hash,
        query_ids=query_ids,
        scale_factor=scale_factor,
        params_list=params_list,
        validator_config=validator_config,
    )
    if cache_entries is None:
        if only_from_cache:
            raise CacheMissError(missing_query_ids[0], scale_factor)
        return None

    all_passed = all(entry.success for entry in cache_entries)
    msg = "\n".join([entry.msg for entry in cache_entries])
    if not all_passed:
        msg = "Validation failed (from cache):\n" + msg
    else:
        msg = "All queries passed validation! (from cache)\n" + msg
    logger.info(
        f"Validation results loaded from cache for {len(cache_entries)} queries"
    )
    return (
        msg,
        all_passed,
        merge_cached_validation_metrics(scale_factor, query_ids, cache_entries),
        True,
    )
