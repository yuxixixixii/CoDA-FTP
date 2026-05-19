#!/usr/bin/env bash
set -euo pipefail

python src/coda_ftp.py \
  --data data/processed_data_with_vocabulary_per_test.csv \
  --feature-list data/FlakeFlaggerFeaturesTypes.csv \
  --output-dir outputs/coda_ftp_main \
  --top-k 6 \
  --min-projects 3 \
  --use-coral \
  --coral-reg 0.001 \
  --smote-ratio 0.075 \
  --smote-k-neighbors 3 \
  --threshold 0.65 \
  --n-estimators 100 \
  --scale-pos-weight 3 \
  --max-depth 3 \
  --min-child-weight 10 \
  --gamma 5 \
  --subsample 0.8 \
  --colsample-bytree 0.8 \
  --reg-lambda 10 \
  --random-state 8 \
  --xgb-n-jobs "${XGB_N_JOBS:-32}" \
  --xgb-tree-method hist
