# TPC-H MonetDB 代理实验 - 代码补全完成指南

## 📋 概述

本指南介绍如何补全和运行三个实验，包括详细的学习说明和操作步骤。

---

## 🎯 三个实验的内容

### 实验1：DeepSeek 模型接入 ✅ 已完成

**目的**：集成 DeepSeek V4 Flash/Pro 模型到系统中

**文件修改**：

- `tpch_monetdb/utils/model_aliases.py` - 5 个函数
- `tpch_monetdb/utils/model_setup.py` - 1 个函数
- `tpch_monetdb/llm_cache/models.py` - 2 个函数 + 模型定价
- `tpch_monetdb/llm_cache/litellm_model_costs.py` - 2 个函数
- `tpch_monetdb/main_tpch_monetdb.py` - 3 个函数
- `tpch_monetdb/llm_cache/litellm_model_cost_overrides.json` - 新增配置

**核心概念**：

```
模型名称规范化流程：
openai/deepseek-v4-flash → normalize → deepseek-v4-flash
                               ↓
                          计费查询 MODEL_REGISTRY
                               ↓
                         深度搜索: flash 型号
                               ↓
                         返回定价信息
```

### 实验2：Agent 基础工具 ✅ 已完成

**目的**：实现三个只读工具，让代理能访问工作区信息

**文件修改**：

- `tpch_monetdb/tools/tpch_monetdb_agent_tools.py` - 2 个函数
- `tpch_monetdb/tools/cpu_info.py` - 4 个函数

**三个工具**：

1. **list_directory** - 列表文件

   ```
   输入：path (可选), pattern (glob), limit
   输出：排序的文件列表（目录名后追加 /）
   ```

2. **grep_repo** - 搜索代码

   ```
   输入：pattern (正则), path, glob, limit
   输出：filename:line_no:line_content
   ```

3. **cpu_info** - CPU 信息
   ```
   输入：timeout_ms
   输出：JSON {architecture, model_name, isa_support, ...}
   ```

### 实验3：TPC-H 结果校验器 ✅ 已完成

**目的**：比较数据库查询结果（baseline vs 优化后）

**文件修改**：

- `tpch_monetdb/oracle/result.py` - 6 个函数
- `tpch_monetdb/oracle/tpch_validator.py` - 18+ 个函数

**核心流程**：

```
CSV 文件 → 解析行和列
    ↓
预期结果 vs 实际结果
    ↓
有序查询？逐行比较 : 多重集语义
    ↓
浮点容差判断
    ↓
生成验证报告
```

---

## 📚 关键知识点详解

### 知识1：模型提供商的映射

**问题**：同一个模型可能通过不同的提供商访问

```
deepseek-v4-flash 可以通过：
- OpenAI 兼容 API (legacy): openai/deepseek-v4-flash
- DeepSeek 原生 API:        deepseek/deepseek-v4-flash
```

**解决方案**：规范化模型名称

```python
def normalize_accounting_model_name(model_name: str) -> str:
    # 把所有前缀剥离，只保留核心模型名称
    # 这样 MODEL_REGISTRY 中只需要一个 "deepseek-v4-flash" entry
    if "deepseek-v4" in model_name:
        if "/" in model_name:
            return model_name.split("/", 1)[1]  # openai/model -> model
        return model_name
    # ... 其他模型的规范化
```

**学习意义**：

- 简化计费管理（一个模型一个价格）
- 支持多 provider 兼容性
- 代码维护更容易

---

### 知识2：工具权限管理

**问题**：代理不应该能访问整个文件系统

```
❌ 危险：agent 可以读 /etc/passwd
✓ 安全：agent 只能读 workspace_root 目录
```

**解决方案**：权限检查

```python
def list_directory(self, relative_path, pattern, limit):
    # 1. 解析相对路径为绝对路径
    target = self._resolve_path(relative_path, profile, mode="read")

    # 2. 权限检查：确保在 workspace_root 内
    if not _is_relative_to(target, self.workspace_root):
        raise PolicyError("Access denied")

    # 3. 列表并返回
    return sorted(target.iterdir())[:limit]
```

**学习意义**：

- 安全性最佳实践
- 沙箱隔离思想
- 工作区边界定义

---

### 知识3：浮点数比较

**问题**：数据库查询的浮点结果可能有精度差异

```
期望：3.14159265358979
实际：3.14159265358979312  <- 浮点精度误差
```

**解决方案**：容差比较

```python
def _values_equal(expected, actual, contract):
    # 获取浮点容差参数
    atol = contract.float_atol    # 绝对容差，如 1e-2
    rtol = contract.float_rtol    # 相对容差，如 1e-2

    # 计算差异
    abs_diff = abs(expected - actual)

    # 绝对容差：|expected - actual| <= atol
    if abs_diff <= atol:
        return True

    # 相对容差：|expected - actual| / |expected| <= rtol
    if expected != 0 and abs_diff / abs(expected) <= rtol:
        return True

    return False
```

**学习意义**：

- 数值计算中的常见问题
- Decimal 模块的使用
- 容差的合理设置

---

### 知识4：有序 vs 无序结果比较

**有序查询**（如 Q1：有 ORDER BY 子句）

```python
def _compare_ordered_rows(expected, actual, contract):
    # 直接逐行比较，顺序很重要
    for i, (exp_row, act_row) in enumerate(zip(...)):
        if exp_row != act_row:
            mismatch()
```

**无序查询**（如 Q8：没有 ORDER BY）

```python
def _compare_unordered_rows(expected, actual, contract):
    # 使用多重集语义：
    # 1. 对每个 expected 行，在 actual 中找匹配行
    # 2. 删除已匹配的 actual 行
    # 3. 处理剩余的未匹配行

    remaining_actual = list(actual.rows)
    for exp_row in expected.rows:
        matched = find_matching(exp_row, remaining_actual)
        if matched:
            remaining_actual.remove(matched)  # 每行只匹配一次！
        else:
            mismatch(row_missing)

    # 剩余的 actual 行是多出来的
    for act_row in remaining_actual:
        mismatch(extra_row)
```

**学习意义**：

- 多重集（multiset）概念
- TPC-H 规范理解
- 算法设计思想

---

## 🚀 运行步骤（详细版）

### 准备阶段

```bash
# 1. 进入工作区
cd /path/to/tpch_monetdb

# 2. 查看目录结构
ls -la
# 应看到：README.md, pyproject.toml, tpch_monetdb/, docker/ 等

# 3. 检查 Python 版本
python --version
# 应该是 Python 3.9+

# 4. 确认依赖已安装
python -c "import pytest; print('pytest OK')"
python -c "from tpch_monetdb.utils.model_aliases import normalize_accounting_model_name; print('imports OK')"
```

### 实验1 运行步骤

```bash
# 第1步：配置 API Key（可选，单元测试不需要真实 Key）
# 如果想完整运行，编辑 .env 文件：
# LITELLM_API_KEY=sk-...
# LITELLM_BASE_URL=https://api.deepseek.com/v1

# 第2步：运行单个测试看详细输出
python -m pytest \
  tpch_monetdb/tests/test_assignment_deepseek_public.py::test_normalize_accounting_model_name \
  -vv -s

# 输出应该包含：
# test_normalize_accounting_model_name PASSED

# 第3步：运行全部 DeepSeek 测试
python -m pytest \
  tpch_monetdb/tests/test_assignment_deepseek_public.py \
  -v

# 预期结果：
# ====== 5 passed in 0.42s ======
```

**故障排查**：

```bash
# 如果看到 NotImplementedError：
#   检查函数是否被正确填充，使用 grep 确认
grep -A 5 "def normalize_accounting_model_name" tpch_monetdb/utils/model_aliases.py

# 如果看到 ImportError：
#   确保所有修改的模块可以导入
python -c "from tpch_monetdb.utils.model_setup import setup_model_config; print('OK')"
```

### 实验2 运行步骤

```bash
# 第1步：检查 CPU info 工具（可能需要 Linux 环境）
python -c "from tpch_monetdb.tools.cpu_info import CpuInfoTool; print('import OK')"

# 第2步：运行 tools 测试
python -m pytest \
  tpch_monetdb/tests/test_assignment_tools_public.py \
  -v

# 预期结果：
# ====== 3 passed in 0.5s ======

# 第3步：手动测试文件列表工具
python << 'EOF'
from pathlib import Path
from tpch_monetdb.tools.tpch_monetdb_agent_tools import StageToolRuntime

runtime = StageToolRuntime(Path.cwd())
# 列表当前目录
result = runtime.list_directory(None, "*.py", 5)
print(result)
EOF
```

### 实验3 运行步骤

```bash
# 第1步：运行验证器测试
python -m pytest \
  tpch_monetdb/tests/test_assignment_validator_public.py \
  -v

# 预期结果：
# ====== 8 passed in 0.6s ======

# 第2步：手动测试 CSV 解析
python << 'EOF'
from pathlib import Path
from tpch_monetdb.oracle.tpch_validator import parse_runtime_csv, TpchQueryResult
import csv

# 创建临时 CSV
csv_file = Path("/tmp/test.csv")
with open(csv_file, "w") as f:
    f.write("id,value\n")
    f.write("1,3.14\n")
    f.write("2,2.71\n")

# 解析
result = parse_runtime_csv(csv_file, "Q1")
print(f"Rows: {result.rows}")
print(f"Columns: {result.columns}")
EOF
```

### 全部测试一起运行

```bash
# 运行所有三个实验的公开测试
python -m pytest \
  tpch_monetdb/tests/test_assignment_deepseek_public.py \
  tpch_monetdb/tests/test_assignment_tools_public.py \
  tpch_monetdb/tests/test_assignment_validator_public.py \
  -v --tb=short

# 预期输出：
# ====== 16+ passed in 2.0s ======
```

---

## 💻 代码修改检查

使用以下命令验证所有修改都已应用：

```bash
# 检查所有函数是否都被实现（不应该有 NotImplementedError）
grep -r "NotImplementedError.*TODO" tpch_monetdb/ || echo "✓ No TODOs found"

# 检查模型定价是否已添加
grep -c "deepseek-v4-flash" tpch_monetdb/llm_cache/models.py
# 应该输出 ≥ 2

# 检查 JSON 配置是否有效
python -c "
import json
with open('tpch_monetdb/llm_cache/litellm_model_cost_overrides.json') as f:
    data = json.load(f)
    print(f'✓ Found {len(data)} models in config')
"
```

---

## 📊 文件修改统计

| 文件                              | 行数     | 函数数 | 状态   |
| --------------------------------- | -------- | ------ | ------ |
| model_aliases.py                  | ~60      | 5      | ✅     |
| model_setup.py                    | ~30      | 1      | ✅     |
| models.py                         | ~30      | 2      | ✅     |
| litellm_model_costs.py            | ~35      | 2      | ✅     |
| main_tpch_monetdb.py              | ~40      | 3      | ✅     |
| litellm_model_cost_overrides.json | ~25      | -      | ✅     |
| cpu_info.py                       | ~150     | 4      | ✅     |
| tpch_monetdb_agent_tools.py       | ~100     | 2      | ✅     |
| result.py                         | ~60      | 6      | ✅     |
| tpch_validator.py                 | ~350     | 18     | ✅     |
| **总计**                          | **~820** | **43** | **✅** |

---

## 🎓 学习路径建议

### 初级（理解基础）

1. 阅读 README.md 的"实验背景"部分
2. 运行实验1，理解模型配置流程
3. 查看 test_assignment_deepseek_public.py，理解测试预期

### 中级（深入学习）

4. 运行实验2，学习文件系统访问权限
5. 运行实验3 CSV 解析部分，理解数据处理
6. 修改测试用例，验证自己的理解

### 高级（实际应用）

7. 在本地运行完整实验（需要真实 DeepSeek Key）
8. 研究 TPC-H 查询规范，理解有序/无序的定义
9. 尝试扩展：支持更多模型、更复杂的比较逻辑

---

## 🔍 常见问题 (FAQ)

### Q1: 为什么要规范化模型名称？

**A**: 避免计费配置重复。如果支持 3 种 provider，原本需要 3 个配置项，规范化后只需 1 个。

### Q2: grep_repo 为什么要跳过大文件？

**A**: 工具有 token 预算限制。读取一个 100MB 的二进制文件会浪费大量 token，所以应该跳过。

### Q3: 无序查询为什么要用贪心匹配？

**A**: 多重集比较（如果有重复行）需要精确配对。贪心从第一行开始匹配，逐个消除已配对行。

### Q4: 为什么要支持 atol 和 rtol 两种容差？

**A**: 不同数值范围需要不同容差：

- 小值（如 0.001）用 atol（绝对）
- 大值（如 1e6）用 rtol（相对）

---

## ✅ 完成清单

在继续下一步之前，确保：

- [ ] 三个实验的所有函数都已实现
- [ ] 没有 `NotImplementedError` 或 `raise NotImplementedError`
- [ ] 所有公开测试都通过了
- [ ] 代码中的注释清晰易懂
- [ ] JSON 配置文件语法正确
- [ ] 没有修改任何测试文件

---

## 📞 获取帮助

如果遇到问题：

1. **查看错误信息**

   ```bash
   # 获取详细错误堆栈
   python -m pytest test_file.py -vv --tb=long
   ```

2. **检查文件修改**

   ```bash
   # 确认修改是否保存
   git diff tpch_monetdb/utils/model_aliases.py
   ```

3. **验证导入**

   ```bash
   # 确保模块可以导入
   python -c "from tpch_monetdb.oracle.result import TpchQueryResult; print('OK')"
   ```

4. **阅读相关文档**
   - README.md - 完整需求
   - 启动指南.md - 环境配置
   - 本文件 - 学习和运行指南

---

**祝实验顺利！** 🎉

任何问题都可以查阅相关的测试文件（test*assignment*\*.py）来理解预期行为。
