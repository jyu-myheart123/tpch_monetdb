"""TPC-H MonetDB 查询结果正确性比较器."""

import logging
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np

from .result import TpchQueryResult

logger = logging.getLogger(__name__)

INTEGER_TYPES = ("LONG", "INT", "INTEGER", "SHORT", "BYTE")
ROW_COUNT_SAMPLE_LIMIT = 3


@dataclass
class CellMismatch:
    """单个单元格不匹配记录."""
    row: int
    column: str
    expected: Any
    actual: Any
    diff_type: str  # "float", "timestamp", "type", "value"


@dataclass
class ComparisonReport:
    """比较报告结构."""
    overall_pass: bool = False
    
    # 列名检查
    column_check_pass: bool = False
    column_check_message: str = ""
    
    # 行数检查
    row_count_check_pass: bool = False
    expected_row_count: int = 0
    actual_row_count: int = 0
    row_count_check_message: str = ""
    row_count_check_samples: dict[str, Any] = field(default_factory=dict)
    
    # 单元格不匹配列表
    cell_mismatches: list[CellMismatch] = field(default_factory=list)
    
    # 比较元数据
    expected_source: str = ""
    actual_source: str = ""
    query_id: str = ""
    
    def to_dict(self) -> dict[str, Any]:
        """转换为字典."""
        return {
            "overall_pass": self.overall_pass,
            "column_check": {
                "pass": self.column_check_pass,
                "message": self.column_check_message,
            },
            "row_count_check": {
                "pass": self.row_count_check_pass,
                "expected": self.expected_row_count,
                "actual": self.actual_row_count,
                "message": self.row_count_check_message,
                "samples": self.row_count_check_samples,
            },
            "cell_mismatches": [
                {
                    "row": m.row,
                    "column": m.column,
                    "expected": m.expected,
                    "actual": m.actual,
                    "diff_type": m.diff_type,
                }
                for m in self.cell_mismatches
            ],
            "expected_source": self.expected_source,
            "actual_source": self.actual_source,
            "query_id": self.query_id,
        }
    
    def get_summary(self) -> str:
        """获取人类可读的摘要."""
        lines = [
            f"Comparison Report for {self.query_id}:",
            f"  Overall: {'PASS' if self.overall_pass else 'FAIL'}",
            f"  Columns: {'PASS' if self.column_check_pass else 'FAIL'} - {self.column_check_message}",
            f"  Row Count: {'PASS' if self.row_count_check_pass else 'FAIL'} "
            f"(expected={self.expected_row_count}, actual={self.actual_row_count})",
        ]
        
        if self.cell_mismatches:
            lines.append(f"  Cell Mismatches: {len(self.cell_mismatches)}")
            for m in self.cell_mismatches[:5]:  # 最多显示前 5 个
                lines.append(
                    f"    Row {m.row}, Col '{m.column}': "
                    f"expected={m.expected}, actual={m.actual} ({m.diff_type})"
                )
            if len(self.cell_mismatches) > 5:
                lines.append(f"    ... and {len(self.cell_mismatches) - 5} more")

        if self.row_count_check_message:
            lines.append(f"  Row Count Detail: {self.row_count_check_message}")
        
        return "\n".join(lines)


def compare_results(
    expected: TpchQueryResult,
    actual: TpchQueryResult,
    atol: float = 1e-2,
    rtol: float = 1e-2,
) -> ComparisonReport:
    """比较两个 TpchQueryResult 实例.
    
    比较规则:
    1. 列名必须完全一致（顺序敏感）
    2. 行数必须一致
    3. 浮点值使用近似比较（atol=1e-2, rtol=1e-2）
    4. 时间戳规范化到相同精度后比较
    5. 对于 sorted_by 为空的查询，使用集合语义比较（忽略行顺序）
    
    Args:
        expected: 预期结果（通常是 MonetDB baseline 结果）
        actual: 实际结果（通常是 Generated TPC-H 引擎结果）
        atol: 绝对容差
        rtol: 相对容差
    
    Returns:
        ComparisonReport 实例
    """
    report = ComparisonReport(
        expected_source=expected.source,
        actual_source=actual.source,
        query_id=expected.query_id or actual.query_id,
    )
    
    # 1. 列名检查
    report.column_check_pass, report.column_check_message = _check_columns(
        expected.columns, actual.columns
    )
    if not report.column_check_pass:
        logger.error(f"Column check failed: {report.column_check_message}")
        return report
    
    # 2. 行数检查
    report.row_count_check_pass = expected.row_count == actual.row_count
    report.expected_row_count = expected.row_count
    report.actual_row_count = actual.row_count
    if not report.row_count_check_pass:
        message, samples = _build_row_count_mismatch_detail(
            expected=expected,
            actual=actual,
        )
        report.row_count_check_message = message
        report.row_count_check_samples = samples
        logger.error(message)
        return report
    
    # 3. 内容比较
    report.cell_mismatches = _compare_rows(
        expected=expected,
        actual=actual,
        atol=atol,
        rtol=rtol,
    )
    
    report.overall_pass = len(report.cell_mismatches) == 0
    
    return report


def _check_columns(expected_cols: list[str], actual_cols: list[str]) -> tuple[bool, str]:
    """检查列名是否一致.
    
    Returns:
        (是否通过, 消息)
    """
    if expected_cols == actual_cols:
        return True, "Columns match"
    
    if set(expected_cols) != set(actual_cols):
        missing = set(expected_cols) - set(actual_cols)
        extra = set(actual_cols) - set(expected_cols)
        msg = f"Column sets differ. Missing: {missing}, Extra: {extra}"
        return False, msg
    
    # 列名集合相同但顺序不同
    return False, f"Column order differs. Expected: {expected_cols}, Actual: {actual_cols}"


def _sample_rows(rows: list[list[Any]]) -> dict[str, list[list[Any]]]:
    """采样首尾行，避免错误摘要过长."""
    if len(rows) <= ROW_COUNT_SAMPLE_LIMIT * 2:
        return {"head": rows, "tail": []}

    return {
        "head": rows[:ROW_COUNT_SAMPLE_LIMIT],
        "tail": rows[-ROW_COUNT_SAMPLE_LIMIT:],
    }


def _row_counter(rows: list[list[Any]]) -> Counter[tuple[Any, ...]]:
    """按完整行构建多重集."""
    return Counter(tuple(row) for row in rows)


def _rows_from_counter(counter: Counter[tuple[Any, ...]]) -> list[list[Any]]:
    """把多重集差异转换成有限行样例."""
    examples: list[list[Any]] = []
    for row, count in counter.items():
        for _ in range(count):
            examples.append(list(row))
            if len(examples) >= ROW_COUNT_SAMPLE_LIMIT:
                return examples

    return examples


def _build_row_count_mismatch_detail(
    expected: TpchQueryResult,
    actual: TpchQueryResult,
) -> tuple[str, dict[str, Any]]:
    """构造行数不一致诊断，给自治修复阶段直接定位缺失 bucket."""
    expected_counter = _row_counter(expected.rows)
    actual_counter = _row_counter(actual.rows)
    missing_expected_rows = _rows_from_counter(expected_counter - actual_counter)
    extra_actual_rows = _rows_from_counter(actual_counter - expected_counter)
    samples: dict[str, Any] = {
        "expected": _sample_rows(expected.rows),
        "actual": _sample_rows(actual.rows),
        "missing_expected_rows_sample": missing_expected_rows,
        "extra_actual_rows_sample": extra_actual_rows,
    }
    message = (
        f"Row count mismatch: expected={expected.row_count}, actual={actual.row_count}; "
        f"missing_expected_rows_sample={missing_expected_rows}; "
        f"extra_actual_rows_sample={extra_actual_rows}; "
        f"expected_head={samples['expected']['head']}; "
        f"expected_tail={samples['expected']['tail']}; "
        f"actual_head={samples['actual']['head']}; "
        f"actual_tail={samples['actual']['tail']}"
    )

    return message, samples


def _compare_rows(
    expected: TpchQueryResult,
    actual: TpchQueryResult,
    atol: float,
    rtol: float,
) -> list[CellMismatch]:
    """比较行数据.
    
    对于无序结果（sorted_by 为空），使用集合语义比较。
    对于有序结果，使用顺序敏感比较。
    """
    mismatches = []
    
    # 确定比较策略
    is_ordered = len(expected.sorted_by) > 0
    
    if is_ordered:
        expected_rows = _canonicalize_ordered_rows(
            rows=expected.rows,
            columns=expected.columns,
            column_types=expected.column_types,
            sorted_by=expected.sorted_by,
        )
        actual_rows = _canonicalize_ordered_rows(
            rows=actual.rows,
            columns=actual.columns,
            column_types=actual.column_types,
            sorted_by=expected.sorted_by,
        )
        # 顺序敏感比较
        mismatches = _compare_rows_ordered(
            expected_rows, actual_rows, expected.columns,
            expected.column_types, actual.column_types, atol, rtol
        )
    else:
        # 集合语义比较（无序）
        mismatches = _compare_rows_unordered(
            expected.rows, actual.rows, expected.columns,
            expected.column_types, actual.column_types, atol, rtol
        )
    
    return mismatches


def _canonicalize_ordered_rows(
    rows: list[list[Any]],
    columns: list[str],
    column_types: list[str],
    sorted_by: tuple[str, ...],
) -> list[list[Any]]:
    """按 contract 的 sorted_by 对有序结果做规范化排序。"""
    if not rows or not sorted_by:
        return rows

    column_index = {name: idx for idx, name in enumerate(columns)}
    sort_indices = [column_index[name] for name in sorted_by if name in column_index]
    if not sort_indices:
        return rows

    def sortable_value(value: Any, col_type: str) -> tuple[int, Any]:
        if value is None:
            return (0, "")

        type_upper = col_type.upper()
        if type_upper in ("DOUBLE", "FLOAT", "REAL"):
            return (1, float(value))
        if type_upper in INTEGER_TYPES:
            return (1, int(float(value)))
        if type_upper in ("TIMESTAMP", "DATE"):
            return (1, _normalize_timestamp(str(value)))
        return (1, str(value))

    def row_key(row: list[Any]) -> tuple[tuple[int, Any], ...]:
        return tuple(
            sortable_value(
                row[idx],
                column_types[idx] if idx < len(column_types) else "UNKNOWN",
            )
            for idx in sort_indices
        )

    value = sorted(rows, key=row_key)
    return value


def _compare_rows_ordered(
    expected_rows: list[list],
    actual_rows: list[list],
    columns: list[str],
    expected_types: list[str],
    actual_types: list[str],
    atol: float,
    rtol: float,
) -> list[CellMismatch]:
    """顺序敏感行比较."""
    mismatches = []
    
    for row_idx, (exp_row, act_row) in enumerate(zip(expected_rows, actual_rows)):
        for col_idx, (exp_val, act_val) in enumerate(zip(exp_row, act_row)):
            col_name = columns[col_idx]
            exp_type = expected_types[col_idx] if col_idx < len(expected_types) else "UNKNOWN"
            
            if not _values_equal(exp_val, act_val, exp_type, atol, rtol):
                diff_type = _get_diff_type(exp_val, act_val, exp_type)
                mismatches.append(CellMismatch(
                    row=row_idx,
                    column=col_name,
                    expected=exp_val,
                    actual=act_val,
                    diff_type=diff_type,
                ))
    
    return mismatches


def _compare_rows_unordered(
    expected_rows: list[list],
    actual_rows: list[list],
    columns: list[str],
    expected_types: list[str],
    actual_types: list[str],
    atol: float,
    rtol: float,
) -> list[CellMismatch]:
    """多重集语义行比较（无序，保留重复重数）.
    
    将每行转换为可哈希的 tuple，然后使用多重集（multiset/counter）比较。
    对于浮点列，使用近似相等。
    
    与简单集合比较的区别：
    - 集合：{A, A, B} == {A, B, B} 会被认为是相等的（错误）
    - 多重集：{A:2, B:1} != {A:1, B:2} 正确反映重复行差异
    """
    from collections import Counter
    
    mismatches = []
    
    # 将行转换为可比较的格式
    def row_to_tuple(row: list, types: list[str]) -> tuple:
        """将行转换为 tuple，处理浮点近似."""
        result = []
        for val, typ in zip(row, types):
            if val is None:
                result.append(None)
            elif typ.upper() in ("DOUBLE", "FLOAT", "REAL"):
                # 浮点值：保留为 float，近似比较时处理
                result.append(float(val))
            elif typ.upper() in INTEGER_TYPES:
                result.append(int(float(val)))
            else:
                result.append(str(val))
        return tuple(result)
    
    # 构建行的多重集（带索引，用于错误报告）
    def build_multiset(rows: list[list], types: list[str]) -> Counter:
        """构建行的多重集."""
        counter: Counter = Counter()
        for idx, row in enumerate(rows):
            key = row_to_tuple(row, types)
            counter[key] += 1
        return counter
    
    # 构建从 key 到行索引列表的映射（用于错误报告）
    def build_key_to_indices(rows: list[list], types: list[str]) -> dict:
        """构建 key 到行索引列表的映射."""
        key_to_indices: dict = {}
        for idx, row in enumerate(rows):
            key = row_to_tuple(row, types)
            if key not in key_to_indices:
                key_to_indices[key] = []
            key_to_indices[key].append(idx)
        return key_to_indices
    
    # 尝试直接匹配（使用多重集计数）
    exp_counter = build_multiset(expected_rows, expected_types)
    act_counter = build_multiset(actual_rows, actual_types)
    
    exp_key_to_indices = build_key_to_indices(expected_rows, expected_types)
    act_key_to_indices = build_key_to_indices(actual_rows, actual_types)
    
    # 收集所有唯一的 key（考虑近似相等）
    all_exp_keys = list(exp_counter.keys())
    all_act_keys = list(act_counter.keys())
    
    # 建立 key 之间的匹配关系（处理浮点近似）
    matched_exp_keys = set()
    matched_act_keys = set()
    key_matches = []  # [(exp_key, act_key, match_count), ...]
    
    for exp_key in all_exp_keys:
        if exp_key in matched_exp_keys:
            continue
        
        # 寻找匹配的 actual key
        for act_key in all_act_keys:
            if act_key in matched_act_keys:
                continue
            
            if _tuples_approx_equal(exp_key, act_key, expected_types, atol, rtol):
                # 找到匹配，记录匹配数量（取较小值）
                match_count = min(exp_counter[exp_key], act_counter[act_key])
                key_matches.append((exp_key, act_key, match_count))
                matched_exp_keys.add(exp_key)
                matched_act_keys.add(act_key)
                break
    
    # 计算每个 key 的不匹配数量
    for exp_key, act_key, match_count in key_matches:
        exp_count = exp_counter[exp_key]
        act_count = act_counter[act_key]
        
        if exp_count > match_count:
            # expected 有多余的行
            excess = exp_count - match_count
            indices = exp_key_to_indices[exp_key][:excess]
            for idx in indices:
                mismatches.append(CellMismatch(
                    row=idx,
                    column="*",
                    expected=expected_rows[idx],
                    actual=None,
                    diff_type="missing_row",
                ))
        
        if act_count > match_count:
            # actual 有多余的行
            excess = act_count - match_count
            indices = act_key_to_indices[act_key][:excess]
            for idx in indices:
                mismatches.append(CellMismatch(
                    row=idx,
                    column="*",
                    expected=None,
                    actual=actual_rows[idx],
                    diff_type="extra_row",
                ))
    
    # 处理未匹配的预期行
    for exp_key in all_exp_keys:
        if exp_key not in matched_exp_keys:
            count = exp_counter[exp_key]
            indices = exp_key_to_indices[exp_key][:count]
            for idx in indices:
                mismatches.append(CellMismatch(
                    row=idx,
                    column="*",
                    expected=expected_rows[idx],
                    actual=None,
                    diff_type="missing_row",
                ))
    
    # 处理未匹配的实际行
    for act_key in all_act_keys:
        if act_key not in matched_act_keys:
            count = act_counter[act_key]
            indices = act_key_to_indices[act_key][:count]
            for idx in indices:
                mismatches.append(CellMismatch(
                    row=idx,
                    column="*",
                    expected=None,
                    actual=actual_rows[idx],
                    diff_type="extra_row",
                ))
    
    return mismatches


def _values_equal(
    expected: Any,
    actual: Any,
    col_type: str,
    atol: float,
    rtol: float,
) -> bool:
    """比较两个值是否相等."""
    # 处理 None
    if expected is None and actual is None:
        return True
    if expected is None or actual is None:
        return False
    
    # 根据类型选择比较方法
    type_upper = col_type.upper()
    
    if type_upper in ("DOUBLE", "FLOAT", "REAL"):
        # 浮点近似比较
        try:
            exp_f = float(expected)
            act_f = float(actual)
            return np.isclose(exp_f, act_f, atol=atol, rtol=rtol)
        except (ValueError, TypeError):
            return str(expected) == str(actual)

    elif type_upper in INTEGER_TYPES:
        try:
            return int(float(expected)) == int(float(actual))
        except (ValueError, TypeError):
            return str(expected) == str(actual)
    
    elif type_upper in ("TIMESTAMP", "DATE"):
        # 时间戳：规范化后比较
        exp_norm = _normalize_timestamp(str(expected))
        act_norm = _normalize_timestamp(str(actual))
        return exp_norm == act_norm
    
    else:
        # 字符串或其他类型：直接比较
        return str(expected) == str(actual)


def _get_diff_type(expected: Any, actual: Any, col_type: str) -> str:
    """确定差异类型."""
    type_upper = col_type.upper()
    
    if type_upper in ("DOUBLE", "FLOAT", "REAL"):
        return "float"
    elif type_upper in INTEGER_TYPES:
        return "integer"
    elif type_upper in ("TIMESTAMP", "DATE"):
        return "timestamp"
    elif type(expected) != type(actual):
        return "type"
    else:
        return "value"


def _normalize_timestamp(ts: str) -> str:
    """规范化时间戳字符串到微秒精度.
    
    处理不同精度的时间戳格式:
    - 2016-01-01T00:00:00Z -> 2016-01-01T00:00:00.000000Z
    - 2016-01-01T00:00:00.000Z -> 2016-01-01T00:00:00.000000Z
    - 2016-01-01T00:00:00.000000Z -> 2016-01-01T00:00:00.000000Z
    - 2016-01-01T00:00:00+00:00 -> 2016-01-01T00:00:00.000000Z
    """
    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    dt = dt.astimezone(timezone.utc)
    return dt.isoformat(timespec="microseconds").replace("+00:00", "Z")


def _tuples_approx_equal(
    tuple1: tuple,
    tuple2: tuple,
    types: list[str],
    atol: float,
    rtol: float,
) -> bool:
    """检查两个 tuple 是否近似相等（用于无序比较）."""
    if len(tuple1) != len(tuple2):
        return False
    
    for val1, val2, typ in zip(tuple1, tuple2, types):
        if not _values_equal(val1, val2, typ, atol, rtol):
            return False
    
    return True
