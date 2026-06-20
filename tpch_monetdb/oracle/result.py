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
        # 自动设置创建时间戳（ISO 格式，以 Z 结尾）
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        
        # 自动计算行数
        if self.row_count == 0 and self.rows:
            # 如果 row_count 为 0 但有数据行，自动计算
            self.row_count = len(self.rows)
    
    def to_dict(self) -> dict[str, Any]:
        """转换为字典格式."""
        # 使用 dataclass 的 asdict 函数获取所有字段
        result = asdict(self)
        
        # 需要特殊处理 sorted_by：从 tuple 转换为 list
        # 因为 JSON 不直接支持 tuple，需要转为 list
        if isinstance(result.get("sorted_by"), tuple):
            result["sorted_by"] = list(result["sorted_by"])
        
        return result
    
    def to_json(self, indent: int = 2) -> str:
        """序列化为 JSON 字符串."""
        # 先转换为字典，再转为 JSON
        data = self.to_dict()
        return json.dumps(data, indent=indent, ensure_ascii=False)
    
    def save_to_file(self, path: str) -> None:
        """保存到 JSON 文件."""
        with open(path, "w") as f:
            f.write(self.to_json())
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TpchQueryResult":
        """从字典创建实例."""
        # 创建一个数据副本，避免修改原始输入
        data_copy = dict(data)
        
        # 需要特殊处理 sorted_by：从 list 恢复为 tuple
        # JSON 中是 list，但 dataclass 字段定义为 tuple
        if "sorted_by" in data_copy and isinstance(data_copy["sorted_by"], list):
            data_copy["sorted_by"] = tuple(data_copy["sorted_by"])
        
        # 使用这个字典创建实例
        return cls(**data_copy)
    
    @classmethod
    def from_json(cls, json_str: str) -> "TpchQueryResult":
        """从 JSON 字符串创建实例."""
        # 先解析 JSON，再使用 from_dict
        data = json.loads(json_str)
        return cls.from_dict(data)
    
    @classmethod
    def from_file(cls, path: str) -> "TpchQueryResult":
        """从 JSON 文件加载实例."""
        with open(path, "r") as f:
            return cls.from_json(f.read())
    
    def get_summary(self) -> dict[str, Any]:
        """获取结果摘要（用于日志和调试）."""
        # 返回关键信息的摘要，用于快速查看结果概况
        return {
            "query_id": self.query_id,
            "source": self.source,
            "row_count": self.row_count,
            "columns": self.columns,
            "sorted_by": self.sorted_by,
            "exec_time_ms": self.exec_time_ms,
            "created_at": self.created_at,
        }
