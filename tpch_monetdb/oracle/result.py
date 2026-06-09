"""TPC-H MonetDB 查询结果统一格式定义."""

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional


@dataclass
class TpchQueryResult:
    """TPC-H MonetDB 查询统一结果格式.
    
    用于封装来自 baseline 或 generated runtime 的查询结果，
    提供统一的比较和序列化接口。
    """
    
    # 格式版本（用于后续格式变更时区分新旧结果）
    format_version: str = "1.0"
    
    # 查询标识
    query_id: str = ""                          # "Q1", "Q2", ...
    query_type: str = ""                        # TPC-H query family
    
    # 查询参数和 SQL
    params: dict = field(default_factory=dict)  # 本次查询的具体参数
    sql: str = ""                               # 实际执行的完整 SQL
    
    # 结果数据
    columns: list[str] = field(default_factory=list)      # 返回列名列表
    column_types: list[str] = field(default_factory=list)  # 返回列类型
    rows: list[list] = field(default_factory=list)        # 数据行（二维数组）
    row_count: int = 0                                    # len(rows)
    
    # 比较元数据
    sorted_by: tuple[str, ...] = ()           # 排序键；空 tuple 表示无序
    time_precision: str = "us"                # 时间精度："us" 微秒
    float_tolerance: dict = field(default_factory=lambda: {"atol": 1e-2, "rtol": 1e-2})
    
    # 来源信息
    source: str = ""                          # e.g. "monetdb" | "generated_runtime"
    source_protocol: str = ""                 # "http" | "pgwire"
    
    # 执行信息
    exec_time_ms: Optional[float] = None      # 执行耗时（毫秒）
    created_at: str = ""                      # ISO 格式时间戳
    
    # 调试信息（可选）
    raw_response: Optional[dict] = None       # 原始 HTTP JSON 响应
    
    def __post_init__(self) -> None:
        """初始化后的处理."""
        raise NotImplementedError("TODO(student): initialize created_at and row_count")
    
    def to_dict(self) -> dict[str, Any]:
        """转换为字典格式."""
        raise NotImplementedError("TODO(student): serialize TpchQueryResult to a JSON-friendly dict")
    
    def to_json(self, indent: int = 2) -> str:
        """序列化为 JSON 字符串."""
        raise NotImplementedError("TODO(student): serialize TpchQueryResult to JSON text")
    
    def save_to_file(self, path: str) -> None:
        """保存到 JSON 文件."""
        with open(path, "w") as f:
            f.write(self.to_json())
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TpchQueryResult":
        """从字典创建实例."""
        raise NotImplementedError("TODO(student): restore TpchQueryResult from a dict")
    
    @classmethod
    def from_json(cls, json_str: str) -> "TpchQueryResult":
        """从 JSON 字符串创建实例."""
        raise NotImplementedError("TODO(student): restore TpchQueryResult from JSON text")
    
    @classmethod
    def from_file(cls, path: str) -> "TpchQueryResult":
        """从 JSON 文件加载实例."""
        with open(path, "r") as f:
            return cls.from_json(f.read())
    
    def get_summary(self) -> dict[str, Any]:
        """获取结果摘要（用于日志和调试）."""
        raise NotImplementedError("TODO(student): return a compact result summary")
