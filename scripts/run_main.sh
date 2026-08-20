#!/usr/bin/env bash
# Reproduce the revised primary CoDA-FTP protocol.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
PROJECT_N_JOBS="${PROJECT_N_JOBS:-1}"
XGB_N_JOBS="${XGB_N_JOBS:-1}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/coda_ftp_primary}"
THRESHOLDS="0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90"

exec "$PYTHON_BIN" -u src/coda_ftp.py \
  --feature-mode precomputed_fused \
  --precomputed-data-path data/processed_data_with_vocabulary_per_test.csv \
  --flakeflagger-features-path data/FlakeFlaggerFeaturesTypes.csv \
  --feature-scaling source_zscore \
  --use-coral --coral-reg 0.001 \
  --source-selection top_k --source-selection-top-k 6 --source-selection-min-projects 3 \
  --threshold-policy source_only_cv \
  --source-only-xgb-tune --source-only-xgb-candidate-set conservative_v1 \
  --thresholds "$THRESHOLDS" \
  --balance SMOTE --smote-sampling-strategy 0.075 --smote-k-neighbors 3 \
  --random-state 8 --project-n-jobs "$PROJECT_N_JOBS" --xgb-n-jobs "$XGB_N_JOBS" \
  --xgb-tree-method hist --torch-num-threads 1 --torch-num-interop-threads 1 \
  --output-dir "$OUTPUT_DIR/"
