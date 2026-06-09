# TPC-H MonetDB Agent 作业

---

## 目录

- [运行环境选择](#运行环境选择)
- [实验1：DeepSeek 模型接入](#实验1deepseek-模型接入)
- [实验2：Agent 基础工具实现](#实验2agent-基础工具实现)
- [实验3：TPC-H 结果校验器实现](#实验3tpc-h-结果校验器实现)
- [统一测试命令](#统一测试命令)
- [最终完整实验](#最终完整实验)
- [选做附加题](#选做附加题)

---

# 运行环境选择

---

推荐使用 Docker 路径完成本作业。Docker 会自动构建项目运行环境、启动 MonetDB、导入 bundled tiny TPC-H fixture，并提供容器内测试入口；本机路径需要你自己安装和启动 MonetDB。

## 一、按操作系统选择

- Windows：必须使用 Docker 或 WSL2。不要直接在 Windows 原生命令行里跑最终 outer-loop。
- macOS：可以本机启动，也可以使用 Docker；推荐 Docker。
- Linux：可以本机启动，也可以使用 Docker；推荐 Docker。

说明：

- 前三个公开单元测试不需要真实 MonetDB，也不需要真实 DeepSeek API。
- 最终完整实验需要可用的 DeepSeek/LiteLLM 配置，并需要可连接的 MonetDB baseline。
- 如果使用 Docker，MonetDB 和测试环境由 `docker/tpch-monetdb/deploy.sh` 自动准备。
- 如果本机启动，需要自己安装 MonetDB server/client，并保证连接参数与项目默认值一致。

## 二、通用配置：`.env`

无论使用 Docker 还是本机启动，都需要配置仓库里的 `tpch_monetdb/.env`。Docker 会把当前仓库挂载到容器内 `/workspace/tpch_monetdb_project`，所以容器读取的也是这份 `.env`。

从仓库根目录执行：

```bash
cp tpch_monetdb/.env_example tpch_monetdb/.env
```

把 `tpch_monetdb/.env` 改成下面的形式：

```bash
LITELLM_API_KEY="your-deepseek-api-key"
LITELLM_BASE_URL="https://api.deepseek.com/v1"

WANDB_ENTITY="your-wandb-user-or-team"
WANDB_PROJECT="tpch-monetdb-assignment"
WANDB_API_KEY="your-wandb-api-key"
```

W&B API key 获取方式：

1. 打开 W&B 官方授权页：`https://wandb.ai/authorize`。
2. 登录或注册 W&B 账号。
3. 页面会显示可复制的 API key；也可以从右上角头像进入 `User Settings`，在 `API Keys` 区域创建新的 key。
4. 创建新 key 后要立刻复制完整 key，并填入 `WANDB_API_KEY`。
5. `WANDB_ENTITY` 填个人用户名或 team slug，`WANDB_PROJECT` 填已有项目名或新项目名。

注意：

- API key 是个人密钥，不要提交到 Git。
- 如果最终运行命令保留 `--disable_wandb`，`WANDB_API_KEY` 可以先留空。
- 如果要启用 W&B 记录，先填好 `WANDB_API_KEY`，再从运行命令中去掉 `--disable_wandb`。
- `LITELLM_BASE_URL="https://api.deepseek.com/v1"` 是 DeepSeek OpenAI-compatible endpoint 的常见写法，可以用于真实运行。
- 课程填空仍要求原生 `litellm/deepseek/deepseek-v4-*` 在没有 `LITELLM_BASE_URL` 时也能工作；也就是说，代码不能强制依赖这个环境变量。
- 如果当前 shell 里有旧的其它 provider 地址，运行前先执行 `unset LITELLM_BASE_URL`，再 `source tpch_monetdb/.env`。

确认模型配置可以解析：

```bash
python - <<'PY'
from tpch_monetdb.bootstrap_env import bootstrap_runtime_env
from tpch_monetdb.utils.model_setup import setup_model_config

bootstrap_runtime_env()
cfg = setup_model_config("litellm/deepseek/deepseek-v4-flash")
print("model_name=", cfg.model_name)
print("accounting_model_name=", cfg.accounting_model_name)
print("base_url=", cfg.base_url)
PY
```

期望看到：

```text
model_name= deepseek/deepseek-v4-flash
accounting_model_name= deepseek-v4-flash
base_url= https://api.deepseek.com/v1
```

说明：

- Docker 路径下，先执行 `docker/tpch-monetdb/deploy.sh root-shell`，再在容器内运行上面的确认命令。
- 本机路径下，先完成 Python 环境安装，再在本机运行上面的确认命令。

## 三、推荐路径：Docker

从仓库根目录执行：

```bash
docker/tpch-monetdb/deploy.sh deploy
docker/tpch-monetdb/deploy.sh workspace
docker/tpch-monetdb/deploy.sh test tpch_monetdb/tests/test_docker_runtime_harness.py -q
docker/tpch-monetdb/deploy.sh root-shell
```

`deploy` 会完成：

1. 构建 Docker 镜像。
2. 启动长驻 MonetDB 服务。
3. 导入 bundled tiny TPC-H fixture。
4. 运行 smoke check。

进入容器后，工作区路径是：

```text
/workspace/tpch_monetdb_project
```

后面的单元测试和最终 outer-loop 命令都可以在这个目录下执行。

## 四、本机路径：Python 环境

本机直接运行时，先准备 Python 依赖：

```bash
uv --version
python --version
rg --version

uv sync
uv pip install pymonetdb==1.9.0
```

## 五、本机路径：MonetDB 环境

本机启动需要安装 MonetDB server/client。示例：

```bash
# macOS
brew install monetdb

# Debian/Ubuntu Linux
sudo apt-get update
sudo apt-get install monetdb5-sql monetdb-client
```

启动本机 MonetDB 时，请保持项目默认连接参数：

```text
host: 127.0.0.1
port: 50000
database: tpch_smoke
username: monetdb
password: monetdb
```

一个常见的本机启动流程如下：

```bash
export MONETDB_DBFARM=/tmp/tpch_monetdb_dbs
monetdbd create "$MONETDB_DBFARM"
monetdbd start "$MONETDB_DBFARM"
monetdb create tpch_smoke
monetdb release tpch_smoke
```

先确认 Python 可以连上本机 MonetDB：

```bash
python - <<'PY'
import pymonetdb

conn = pymonetdb.connect(
    hostname="127.0.0.1",
    port=50000,
    database="tpch_smoke",
    username="monetdb",
    password="monetdb",
    autocommit=True,
)
cur = conn.cursor()
cur.execute("SELECT 1")
print(cur.fetchone())
cur.close()
conn.close()
PY
```

再导入 TPC-H 数据。首次 smoke 可以导入仓库自带 tiny fixture：

```bash
python - <<'PY'
import json
from dataclasses import asdict
from pathlib import Path

import pymonetdb

from tpch_monetdb.oracle.monetdb_prepare import prepare_tpch_database

fixture_dir = Path("docker/tpch-monetdb/fixtures/tiny-tpch")
conn = pymonetdb.connect(
    hostname="127.0.0.1",
    port=50000,
    database="tpch_smoke",
    username="monetdb",
    password="monetdb",
    autocommit=True,
)
report = prepare_tpch_database(conn, fixture_dir)
print(json.dumps(asdict(report), ensure_ascii=False, sort_keys=True))
conn.close()
PY
```

最终完整实验使用真实 `sf1` 数据时，数据目录应是：

```text
$BASE_DATA_DIR/sf1/customer.tbl
$BASE_DATA_DIR/sf1/lineitem.tbl
...
```

如果本机 MonetDB 遇到 `COPY INTO` 文件权限或路径不可见问题，优先切换到 Docker 路径；Docker 已经处理好 MonetDB 服务端可见的导入目录。

---

# 实验1：DeepSeek 模型接入

---

## 一、实验背景

项目通过 LiteLLM 统一接入不同模型。一个新模型接入系统时，不能只写模型名，还需要处理：

- provider 前缀识别。
- accounting model name 归一化。
- API key 和 base url 配置。
- 模型价格和上下文窗口。
- reasoning effort 到 provider 请求体参数的转换。

本实验要求你补全 DeepSeek V4 Flash/Pro 的接入逻辑。

---

## 二、实验目标

1. 理解 provider prefix 和 model alias 的作用。
2. 掌握模型配置对象 `ModelConfig` 的构造方式。
3. 掌握 cached input 和 uncached input 的价格计算。
4. 理解 LiteLLM local model cost override 的注册流程。
5. 理解 DeepSeek thinking/reasoning 参数如何注入 `ModelSettings`。

---

## 三、需要修改的代码

请只修改下面列出的 TODO 位置。

### 1. 模型别名与 provider 判断

文件：`tpch_monetdb/utils/model_aliases.py`

- 第 19 行：补全 `normalize_accounting_model_name()` 中 DeepSeek provider alias 归一化。
- 第 27 行：补全 `get_model_provider()` 中 DeepSeek provider 前缀解析。
- 第 40 行：补全 `is_deepseek_model()`。
- 第 45 行：补全 `is_openai_deepseek_model()`。
- 第 50 行：补全 `is_anthropic_deepseek_model()`。

要求：

- `deepseek-v4-flash` 和 `deepseek-v4-pro` 应识别为 DeepSeek。
- `deepseek/deepseek-v4-*`、`openai/deepseek-v4-*`、`anthropic/deepseek-v4-*` 应归一化为不带 provider 的 accounting model name。
- 非 DeepSeek 模型不能误判。

### 2. LiteLLM 模型配置

文件：`tpch_monetdb/utils/model_setup.py`

- 第 70-71 行：补全 `setup_model_config()` 中 DeepSeek LiteLLM 分支。

要求：

- `litellm/deepseek/deepseek-v4-pro` 使用 LiteLLM native DeepSeek provider，不强制 `LITELLM_BASE_URL`。
- `litellm/openai/deepseek-v4-pro` 作为 legacy 路径允许通过，但需要 warning；没有 base url 时填 `https://api.deepseek.com`。
- `litellm/anthropic/deepseek-v4-pro` 必须抛出 `RuntimeError`。

### 3. 模型价格和计费

文件：`tpch_monetdb/llm_cache/models.py`

- 第 216 行：补全 `request_cost_usd()` 中 DeepSeek cache hit/miss 计费。
- 同文件还需要在 `MODEL_REGISTRY` 中补回 DeepSeek V4 Flash/Pro 的 pricing。

要求：

- `deepseek-v4-flash`：
  - input: `0.14 / 1_000_000`
  - cached input: `0.0028 / 1_000_000`
  - output: `0.28 / 1_000_000`
  - context window: `1_000_000`
- `deepseek-v4-pro`：
  - input: `0.435 / 1_000_000`
  - cached input: `0.003625 / 1_000_000`
  - output: `0.87 / 1_000_000`
  - context window: `1_000_000`
- `cached_tokens > input_tokens` 时不能产生负计费。

### 4. LiteLLM cost override 注册

文件：`tpch_monetdb/llm_cache/litellm_model_costs.py`

- 第 19 行：补全 `load_tpch_monetdb_litellm_model_cost_overrides()`。
- 第 28 行：补全 `register_tpch_monetdb_litellm_model_costs()` 中更新 `litellm.model_cost` 和刷新 lowercase map 的逻辑。

要求：

- 从 `tpch_monetdb/llm_cache/litellm_model_cost_overrides.json` 读取 JSON。
- 校验 JSON 必须是非空 dict。
- 每个 model name 必须是 str，每个 model info 必须是 dict。
- 注册过程必须幂等，重复调用不能重复刷新。

### 5. DeepSeek reasoning 参数注入

文件：`tpch_monetdb/main_tpch_monetdb.py`

- 第 126 行：补全 `_normalize_deepseek_reasoning_effort()`。
- 第 182 行：补全 `_build_model_settings()` 中 DeepSeek thinking enabled 分支。
- 第 184 行：补全 `_build_model_settings()` 中 DeepSeek thinking disabled 分支。

要求：

- `xhigh` 和 `max` 映射为 provider 请求体中的 `max`。
- `minimal`、`low`、`medium`、`high` 映射为 provider 请求体中的 `high`。
- DeepSeek 请求应通过 `extra_body` 注入：
  - `{"thinking": {"type": "enabled"}, "reasoning_effort": "..."}`
- `reasoning_effort=none` 时应注入：
  - `{"thinking": {"type": "disabled"}}`
- 非 DeepSeek 模型不能注入 `thinking`。

---

## 四、运行测试

只运行实验 1：

```bash
python -m pytest tpch_monetdb/tests/test_assignment_deepseek_public.py -q
```

对应测试文件：

- `tpch_monetdb/tests/test_assignment_deepseek_public.py`

---

# 实验2：Agent 基础工具实现

---

## 一、实验背景

Agent 需要通过工具读取工作区文件、搜索代码、获取 CPU 信息。本实验要求你实现三个只读工具的核心逻辑：

- `list_files`
- `grep_repo`
- `cpu_info`

这些工具只通过本地临时目录和固定文本测试，不需要真实 agent 对话。

---

## 二、实验目标

1. 掌握 workspace path 解析和越界保护。
2. 掌握 glob、limit 和稳定输出格式。
3. 掌握正则搜索、非 UTF-8 文件跳过和大文件保护。
4. 掌握 CPU flags、cache、NUMA 信息解析。
5. 掌握稳定 JSON 输出格式。

---

## 三、需要修改的代码

### 1. `list_files`

文件：`tpch_monetdb/tools/tpch_monetdb_agent_tools.py`

- 第 517 行：补全 `StageToolRuntime.list_directory()`。

要求：

- 支持 workspace root、子目录和 `/`。
- 支持 glob pattern。
- 支持 limit。
- 目录输出追加 `/`。
- 输出顺序稳定。
- 只显示当前 profile 允许读取的路径。
- workspace 外路径必须报 recoverable policy error。

### 2. `grep_repo`

文件：`tpch_monetdb/tools/tpch_monetdb_agent_tools.py`

- 第 521 行：补全 `StageToolRuntime.grep_repo()`。

要求：

- 接收 Python 正则表达式。
- 支持 `path` 指定搜索根。
- 支持 `glob` 过滤文件名。
- 支持 `limit` 限制匹配数量。
- 返回格式为 `relative/path:line_no:line`。
- 跳过目录、二进制文件和非 UTF-8 文件。
- 超过 `_TOOL_GREP_MAX_BYTES` 的大文件不读取，并返回 skipped 提示。
- 没有匹配时返回 `(no matches)`。

### 3. `cpu_info`

文件：`tpch_monetdb/tools/cpu_info.py`

- 第 84 行：补全 `_truncate()`。
- 第 88 行：补全 `_parse_cpuinfo_flags()`。
- 第 92 行：补全 `_parse_lscpu_summary()`。
- 第 96 行：补全 `_build_response()`。

要求：

- 从 `/proc/cpuinfo` 的 `flags` 或 `Features` 行提取 ISA flags。
- 从 `lscpu` 输出提取 architecture、model name、cache、NUMA 信息。
- 根据 flags 识别 `avx512f`、`avx2`、`avx`、`sse4_2`、`sse4_1`、`neon`、`asimd`。
- 有真实硬件探测证据时才返回 `target_cpu_hint="native"`。
- 原始输出过长时做 head/tail 截断，并包含 `truncated` 提示。
- 输出稳定 JSON 字段，便于测试读取。

---

## 四、运行测试

只运行实验 2：

```bash
python -m pytest tpch_monetdb/tests/test_assignment_tools_public.py -q
```

对应测试文件：

- `tpch_monetdb/tests/test_assignment_tools_public.py`

---

# 实验3：TPC-H 结果校验器实现

---

## 一、实验背景

数据库系统需要把 generated runtime 输出和 baseline 查询结果进行比较。这个比较层要负责：

- 保存和序列化查询结果。
- 解析 runtime CSV。
- 检查列名、行数和值。
- 根据 TPC-H contract 决定 ordered/unordered 比较。
- 输出结构化 mismatch report。

本实验只测试小数据，不要求连接真实 MonetDB。

---

## 二、实验目标

1. 掌握 dataclass 结果对象的初始化和序列化。
2. 掌握 CSV 结果解析和类型推断。
3. 掌握 ordered 和 unordered 查询结果比较。
4. 掌握浮点容差比较。
5. 掌握结构化错误报告和 summary 输出。

---

## 三、需要修改的代码

### 1. `TpchQueryResult`

文件：`tpch_monetdb/oracle/result.py`

- 第 52 行：补全 `__post_init__()`。
- 第 56 行：补全 `to_dict()`。
- 第 60 行：补全 `to_json()`。
- 第 70 行：补全 `from_dict()`。
- 第 75 行：补全 `from_json()`。
- 第 85 行：补全 `get_summary()`。

要求：

- 自动补 `created_at`。
- `row_count` 为空时根据 `rows` 自动补齐。
- `to_dict()` 中 `sorted_by` 要从 tuple 转为 list。
- `from_dict()` 中 `sorted_by` 要从 list 恢复为 tuple。
- JSON 往返后关键字段保持一致。

### 2. `TpchValidationReport` 和 `TpchValidator`

文件：`tpch_monetdb/oracle/tpch_validator.py`

- 第 50 行：补全 `TpchValidationReport.to_dict()`。
- 第 54 行：补全 `TpchValidationReport.get_summary()`。
- 第 67 行：补全 `TpchValidator.compare_results()`。
- 第 77 行：补全 `TpchValidator.parse_runtime_csv()`。

要求：

- `to_dict()` 输出 JSON-friendly dict。
- `get_summary()` 至少包含 PASS/FAIL、query id、columns、rows、ordered 和 first mismatch。
- facade 方法要正确调用模块级 `compare_tpch_results()` 和 `parse_runtime_csv()`。

### 3. CSV 解析

文件：`tpch_monetdb/oracle/tpch_validator.py`

- 第 96 行：补全 `parse_runtime_csv()`。
- 第 194 行：补全 `_parse_csv_cell()`。
- 第 199 行：补全 `_infer_column_types()`。
- 第 204 行：补全 `_first_non_null()`。
- 第 209 行：补全 `_infer_value_type()`。

要求：

- CSV 第一行作为列名。
- 空 CSV 抛 `ValueError`。
- 空 cell 解析为 `None`。
- 整数、浮点数、字符串要稳定解析。
- column type 至少支持 `INTEGER`、`DOUBLE`、`STRING`、`UNKNOWN`。

### 4. 结果比较和 report 构造

文件：`tpch_monetdb/oracle/tpch_validator.py`

- 第 86 行：补全 `compare_tpch_results()`。
- 第 110 行：补全 `_build_report()`。
- 第 119 行：补全 `_compare_ordered_rows()`。
- 第 128 行：补全 `_compare_unordered_rows()`。
- 第 137 行：补全 `_rows_match_unordered()`。
- 第 147 行：补全 `_find_matching_row()`。
- 第 159 行：补全 `_compare_row_values()`。
- 第 164 行：补全 `_values_equal()`。
- 第 169 行：补全 `_format_summary_row()`。
- 第 174 行：补全 `_format_summary_value()`。
- 第 179 行：补全 `_to_decimal()`。
- 第 184 行：补全 `_diff_type()`。
- 第 189 行：补全 `_normalized_sort_columns()`。

要求：

- 列名和列顺序必须一致。
- 行数必须一致。
- `compare_tpch_results()` 使用 contract 决定 ordered/unordered、`sorted_by`、`float_atol`、`float_rtol`。
- ordered query 逐行比较，顺序不同必须失败。
- unordered query 用多重集语义比较，重复行数量必须正确。
- 浮点值支持 `atol` 和 `rtol`。
- `None` 和 `0` 不相等。
- mismatch 至少包含 `row`、`column`、`expected`、`actual`、`diff_type`、`message`。

---

## 四、运行测试

只运行实验 3：

```bash
python -m pytest tpch_monetdb/tests/test_assignment_validator_public.py -q
```

对应测试文件：

- `tpch_monetdb/tests/test_assignment_validator_public.py`

---

# 统一测试命令

---

## 运行全部公开测试

```bash
python -m pytest \
  tpch_monetdb/tests/test_assignment_deepseek_public.py \
  tpch_monetdb/tests/test_assignment_tools_public.py \
  tpch_monetdb/tests/test_assignment_validator_public.py \
  -q
```

## 公开测试文件清单

- `tpch_monetdb/tests/test_assignment_deepseek_public.py`
- `tpch_monetdb/tests/test_assignment_tools_public.py`
- `tpch_monetdb/tests/test_assignment_validator_public.py`

# 最终完整实验

---

完成三个实验并通过公开单元测试后，还需要完整跑通一次 TPC-H outer-loop 实验。这个步骤不是单元测试，而是端到端验收。

开始前必须先阅读启动指南：

- `tpch_monetdb/启动指南/启动指南.md`
- 第 332 行开始的“正式 Q1-Q22 outer-loop”
- 第 490 行开始的“查看 outer-loop 最新状态”

下面只给出本作业的简化验收命令；如果 Docker、MonetDB、数据目录、`.env` 或 resume 行为有疑问，以启动指南为准。

## 一、完整实验目标

必须满足：

- scale factor 使用 `sf1`。
- 模型使用 `litellm/deepseek/deepseek-v4-flash`。
- outer-loop 只跑一轮，即 `--max_rounds 1`。
- 一轮中完整经过三个阶段：
  - storage plan
  - base implementation
  - optimization
- 必须关闭 perf/PMU：不要运行 `grant-perf` / `pmu-smoke`，不要传 `--target_cpu`、`--hardware_counter_backend`、`--hardware_counter_runner_cmd`、`--host_kernel`、`--perf_event_paranoid`、`--large_sf`。

## 二、数据目录要求

正式验收时，`BASE_DATA_DIR` 应包含 `sf1` 子目录：

```text
$BASE_DATA_DIR/sf1/customer.tbl
$BASE_DATA_DIR/sf1/lineitem.tbl
$BASE_DATA_DIR/sf1/nation.tbl
$BASE_DATA_DIR/sf1/orders.tbl
$BASE_DATA_DIR/sf1/part.tbl
$BASE_DATA_DIR/sf1/partsupp.tbl
$BASE_DATA_DIR/sf1/region.tbl
$BASE_DATA_DIR/sf1/supplier.tbl
```

如果只是首次 smoke，可以参考启动指南使用仓库自带 tiny fixture；最终提交验收按 `sf1` 数据目录执行。

## 三、启动 Docker 工作区

先按 `tpch_monetdb/启动指南/启动指南.md` 完成 Docker 工作区部署，再执行下面命令。

在仓库根目录运行：

```bash
docker/tpch-monetdb/deploy.sh deploy
docker/tpch-monetdb/deploy.sh root-shell
```

下面命令都在容器内 `/workspace/tpch_monetdb_project` 执行。

## 四、运行一轮完整 outer-loop

运行前再次确认已经阅读 `tpch_monetdb/启动指南/启动指南.md`，并且 `.env`、MonetDB baseline 和 `sf1` 数据目录都已经准备好。

先配置环境变量：

```bash
# 清理 shell 里可能残留的旧 provider 地址；source .env 后会使用 .env 中的新配置。
unset LITELLM_BASE_URL
set -a
source tpch_monetdb/.env
set +a

# 必做完整实验关闭 perf/PMU。附加题才允许打开这些开关。
unset TPCH_MONETDB_HARDWARE_COUNTER_BACKEND
unset TPCH_MONETDB_HARDWARE_COUNTER_RUNNER_CMD
unset TPCH_MONETDB_PERF_BINARY
unset TPCH_MONETDB_PERF_EVENT_PARANOID

export TPCH_MONETDB_MODEL="litellm/deepseek/deepseek-v4-flash"
export TPCH_MONETDB_REASONING_EFFORT="xhigh"
export BASE_DATA_DIR="/path/to/tpch_sf1_data"
export ARTIFACTS_DIR="$PWD/tpch_monetdb_artifacts/student_sf1_deepseek_flash_$(date +%Y%m%d_%H%M%S)"
```

其中 `BASE_DATA_DIR` 要替换为真实的 sf1 数据根目录，且该目录下必须有 `sf1/*.tbl`。

然后运行：

下面命令是必做完整实验命令，不能追加任何 perf/PMU 参数；perf/PMU 只属于后面的选做附加题。

```bash
python run_outer_loop_tpch_monetdb.py \
  --conv outer1-22v1 \
  --benchmark tpch \
  --artifacts_dir "$ARTIFACTS_DIR" \
  --base_data_dir "$BASE_DATA_DIR" \
  --validation_mode strict \
  --model "$TPCH_MONETDB_MODEL" \
  --reasoning_effort "$TPCH_MONETDB_REASONING_EFFORT" \
  --max_rounds 1 \
  --auto_u \
  --auto_finish \
  --disable_wandb \
  --disable_tracing
```

## 五、检查完整实验结果

查看 outer-loop 最新状态：

```bash
python - <<'PY'
import json
import os
from pathlib import Path

artifacts = Path(os.environ["ARTIFACTS_DIR"])
latest = artifacts / "outer_loop_runs" / "outer1-22v1" / "latest.json"
data = json.loads(latest.read_text())
record = data["record"]
print("latest_round=", data["latest_round"])
print("action=", record["action"])
print("best_summary=", record["best_optimization_summary_path"])
print("best_snapshot=", record["best_final_snapshot_hash"])
PY
```

检查三个阶段都产出了 summary：

```bash
find "$ARTIFACTS_DIR/storage_plan_runs" -maxdepth 2 -name latest.json -print
find "$ARTIFACTS_DIR/scripted_runs" -maxdepth 2 -name latest.json -print
find "$ARTIFACTS_DIR/optimization_runs" -maxdepth 2 -name latest.json -print
```

完成标准：

1. 三个公开测试文件全部通过。
2. `outer_loop_runs/outer1-22v1/latest.json` 存在。
3. `storage_plan_runs`、`scripted_runs`、`optimization_runs` 下都能找到 `latest.json`。
4. `latest_round` 打印为 `1`。
5. 运行过程没有修改公开测试文件。

# 选做附加题

---

附加题不作为前三个必做实验和最终完整实验的前置条件。主实验已经要求关闭 perf/PMU，所以不做附加题也不影响必做验收。

附加题分两步：

1. 补全 PMU/perf 原始证据解析代码，并通过选做单元测试。
2. 在 Linux perf 环境下打开 PMU 跑一次 sf1 outer-loop，并和必做完整实验的无 PMU artifact 做性能与证据对比。

## 一、PMU/perf 解析代码题

文件：`tpch_monetdb/tools/tpch/hardware_counters.py`

需要补全下面的 `TODO(optional)`：

- 第 236 行：补全 `parse_perf_stat_csv()`。
- 第 246 行：补全 `parse_perf_script_hotspots()`。
- 第 261 行：补全 `_split_perf_script_samples()`。
- 第 282 行：补全 `extract_perf_script_symbol()`。
- 第 287 行：补全 `extract_perf_script_source_line()`。
- 第 325 行：补全 `derive_hardware_counter_metrics()`。
- 第 330 行：补全 `validate_hardware_counter_summary()`。

要求：

- `parse_perf_stat_csv()` 能解析 `perf stat -x,` 的 CSV 输出。
- 跳过空行、无效行、`<not counted>`、`<not supported>` 和无法转成数字的 counter。
- `derive_hardware_counter_metrics()` 至少计算：
  - `ipc = instructions / cycles`
  - `cache_miss_rate = cache-misses / instructions`
  - `llc_mpki = LLC-load-misses * 1000 / instructions`
  - `dtlb_mpki = dTLB-load-misses * 1000 / instructions`
  - `branch_miss_rate = branch-misses / instructions`
- denominator 不存在或为 0 时不要产生无穷大、NaN 或异常。
- `validate_hardware_counter_summary()` 缺少 required event 时抛项目结构化错误。
- `parse_perf_script_hotspots()` 能从 `perf script` 输出中统计 top symbols、top frames、top source lines、sample count 和 raw excerpt。
- `extract_perf_script_symbol()` 要过滤地址、路径、`[unknown]`、source line、纯数字等非 symbol token。
- `extract_perf_script_source_line()` 要提取类似 `query_q3.cpp:42` 的源码位置。

运行选做单元测试：

```bash
PYTHONPATH=. python -m pytest optional_tests/test_assignment_perf_optional_public.py -q
```

对应测试文件：

- `optional_tests/test_assignment_perf_optional_public.py`

## 二、PMU/perf 真实消融实验

只有在选做单元测试通过后，才继续做真实 perf 消融。

环境限制：

- 只建议在 Linux 本机 runner 的 Docker 容器里做。
- macOS、Windows Docker Desktop、WSL2、普通无权限容器不作为最终 PMU 证据。
- `pmu-smoke` 失败时不要继续开启 PMU 参数。

先阅读启动指南中的 PMU/perf 可选验收：

- `tpch_monetdb/启动指南/启动指南.md`
- 第 525 行开始的“PMU / perf 可选验收”

在 Linux 上启动容器并检查 perf：

```bash
docker/tpch-monetdb/deploy.sh deploy
docker/tpch-monetdb/deploy.sh grant-perf
docker/tpch-monetdb/deploy.sh pmu-smoke
docker/tpch-monetdb/deploy.sh root-shell
```

容器内确认：

```bash
/usr/local/bin/tpch-perf --version
cat /proc/sys/kernel/perf_event_paranoid
```

需要至少保留两组 sf1 outer-loop artifact：

1. `baseline`：必做完整实验已经跑出的无 PMU artifact，不加 `--target_cpu`，不加 PMU/perf 参数。
2. `native-pmu`：新建 `ARTIFACTS_DIR`，加 `--target_cpu native` 和 PMU/perf 参数。

有能力的同学可以再加一组 `native-only`：

- `native-only`：只加 `--target_cpu native`，不加 PMU/perf 参数。

`native-only` 用于区分 native 编译目标和 PMU 证据本身的影响。只额外加入：

```bash
--target_cpu native
```

`native-pmu` 额外加入：

```bash
--target_cpu native \
--hardware_counter_backend linux_perf_native \
--hardware_counter_runner_cmd /usr/local/bin/tpch-perf \
--host_kernel "$(uname -r)" \
--perf_event_paranoid "$(cat /proc/sys/kernel/perf_event_paranoid)" \
--large_sf 1
```

## 注意事项

1. 不要修改测试文件。
2. 不要删除函数签名、参数或返回类型。
3. 不要把 TODO 改成硬编码公开测试样例。
4. 前三个公开单元测试不要求调用真实 DeepSeek API。
5. 最终完整实验需要 `tpch_monetdb/.env` 中有可用的 DeepSeek/LiteLLM 配置。
6. 如果某个测试失败，先看失败栈中的文件和行号，再回到上面对应的 TODO 位置修改。
