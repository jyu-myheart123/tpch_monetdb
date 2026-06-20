#!/bin/bash
set -euo pipefail

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
ARTIFACTS_DIR="/workspace/tpch_monetdb_project/tpch_monetdb_artifacts/deepseek_pro_q1_q22_${TIMESTAMP}"
LOGFILE="${ARTIFACTS_DIR}/outer_loop_nohup.log"
mkdir -p "$ARTIFACTS_DIR"

exec > >(tee -a "$LOGFILE") 2>&1

echo "=========================================="
echo "TPC-H MonetDB Outer Loop (Q1-Q22)"
echo "Start:     $(date)"
echo "Artifacts: $ARTIFACTS_DIR"
echo "Model:     litellm/deepseek/deepseek-v4-pro"
echo "Reasoning: xhigh"
echo "Max Round: 1"
echo "W&B:       ENABLED"
echo "Data:      SF1 (/tmp/tpch_monetdb_data)"
echo "Perf:      linux_perf_native (tsdb-perf)"
echo "=========================================="

source /opt/tpch-monetdb/venv/bin/activate

unset all_proxy ALL_PROXY
unset LITELLM_BASE_URL
set -a
source tpch_monetdb/.env
set +a

export TPCH_MONETDB_MODEL="litellm/deepseek/deepseek-v4-pro"
export TPCH_MONETDB_REASONING_EFFORT="xhigh"
export BASE_DATA_DIR="/tmp/tpch_monetdb_data"

python run_outer_loop_tpch_monetdb.py \
  --conv outer1-22v1 \
  --benchmark tpch \
  --artifacts_dir "$ARTIFACTS_DIR" \
  --base_data_dir "$BASE_DATA_DIR" \
  --validation_mode strict \
  --model "$TPCH_MONETDB_MODEL" \
  --reasoning_effort "$TPCH_MONETDB_REASONING_EFFORT" \
  --max_rounds 1 \
  --target_cpu native \
  --hardware_counter_backend linux_perf_native \
  --hardware_counter_runner_cmd /usr/local/bin/tsdb-perf \
  --host_kernel "$(uname -r)" \
  --perf_event_paranoid "$(cat /proc/sys/kernel/perf_event_paranoid)" \
  --large_sf 1 \
  --disable_repo_sync \
  --auto_u \
  --auto_finish

EC=$?
echo "=========================================="
echo "Exit: $EC  |  End: $(date)"
echo "Artifacts: $ARTIFACTS_DIR"
echo "=========================================="
exit $EC
