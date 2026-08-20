#!/usr/bin/env bash
# Strict source-only sensitivity check for fixed source-selection k values.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

OUTPUT_ROOT="${OUTPUT_ROOT:-result_1/k_sensitivity_source_only_zscore_20260813}"
PYTHON_BIN="${PYTHON_BIN:-/home/zhangyu/.conda/envs/unixcoder/bin/python}"
# Four k values run together; two project workers each keeps CPU use bounded.
PROJECT_N_JOBS="${PROJECT_N_JOBS:-2}"
K_VALUES="${K_VALUES:-3,4,5,7}"
THRESHOLDS="0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90"

IFS=',' read -r -a K_ARRAY <<< "$K_VALUES"
pids=()
for k in "${K_ARRAY[@]}"; do
  mkdir -p "${OUTPUT_ROOT}/k_${k}"
  "$PYTHON_BIN" -u experiments/source_only_tuning/CoDA-FTP-source-only-tuning.py \
    --feature-mode precomputed_fused --feature-scaling source_zscore --use-coral --coral-reg 0.001 \
    --source-selection top_k --source-selection-top-k "$k" --source-selection-min-projects 3 \
    --threshold-policy source_only_cv --source-only-xgb-tune \
    --source-only-xgb-candidate-set conservative_v1 --thresholds "$THRESHOLDS" \
    --balance SMOTE --smote-sampling-strategy 0.075 --smote-k-neighbors 3 \
    --random-state 8 --project-n-jobs "$PROJECT_N_JOBS" --xgb-n-jobs 1 \
    --xgb-tree-method hist --torch-num-threads 1 --torch-num-interop-threads 1 \
    --output-dir "${OUTPUT_ROOT}/k_${k}/" \
    > "${OUTPUT_ROOT}/k_${k}.log" 2>&1 &
  pids+=("$!")
done

for pid in "${pids[@]}"; do
  wait "$pid"
done

echo "Strict source-only fixed-k sensitivity complete: ${OUTPUT_ROOT}"
