#!/usr/bin/env bash
# Strict source-only nested selected-source TCA diagnostic for Table 8.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

OUTPUT_ROOT="${OUTPUT_ROOT:-result_1/table8_tca_source_only_zscore_20260813}"
PYTHON_BIN="${PYTHON_BIN:-/home/zhangyu/.conda/envs/unixcoder/bin/python}"
COMMON=(
  --precomputed-data-path result/processed_data_with_vocabulary_per_test.csv
  --flakeflagger-features-path input_data/FlakeFlaggerFeaturesTypes.csv
  --source-selection-top-k 6 --source-selection-min-projects 3
  --tca-dim 128 --tca-reg 1.0
  --smote-sampling-strategy 0.075 --smote-k-neighbors 3
  --xgb-candidate-set conservative_v1 --random-state 8 --xgb-n-jobs 1 --xgb-tree-method hist
)

export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

"$PYTHON_BIN" -u external_baselines/run_tca_xgb_source_only_nested.py \
  "${COMMON[@]}" --source-selection top_k --output-dir "$OUTPUT_ROOT/selected_source_tca"

echo "Strict source-only selected-source TCA diagnostic complete: $OUTPUT_ROOT"
