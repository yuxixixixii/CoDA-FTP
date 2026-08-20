#!/usr/bin/env bash
# Strict source-only rerun of the Table 7 component ablations.
# Each variant selects its operating threshold using source-project pseudo-targets.
# Variants retaining the regularized classifier also choose among the same five
# pre-specified XGBoost configurations used by the primary result.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

OUTPUT_ROOT="${OUTPUT_ROOT:-result_1/table7_source_only_zscore_ablations_20260813}"
# The remote system Python lacks xgboost; use the validated experiment runtime.
PYTHON_BIN="${PYTHON_BIN:-/home/zhangyu/.conda/envs/unixcoder/bin/python}"
# Four variants run concurrently.  Each fold uses source-fitted z-score
# normalization, so keep the aggregate CPU use conservative.
PROJECT_N_JOBS="${PROJECT_N_JOBS:-2}"
THRESHOLDS="0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90"

COMMON=(
  --feature-mode precomputed_fused
  --feature-scaling source_zscore
  --source-selection-top-k 6 --source-selection-min-projects 3
  --threshold-policy source_only_cv --thresholds "$THRESHOLDS"
  --balance SMOTE --smote-sampling-strategy 0.075 --smote-k-neighbors 3
  --random-state 8 --project-n-jobs "$PROJECT_N_JOBS" --xgb-n-jobs 1
  --xgb-tree-method hist --torch-num-threads 1 --torch-num-interop-threads 1
)

run_tuned_variant() {
  local name="$1"; shift
  "$PYTHON_BIN" -u experiments/source_only_tuning/CoDA-FTP-source-only-tuning.py \
    "${COMMON[@]}" --source-only-xgb-tune --source-only-xgb-candidate-set conservative_v1 \
    "$@" --output-dir "${OUTPUT_ROOT}/${name}/"
}

run_unregularized_variant() {
  "$PYTHON_BIN" -u experiments/source_only_tuning/CoDA-FTP-source-only-tuning.py \
    "${COMMON[@]}" --source-only-xgb-tune \
    --source-only-xgb-candidate-set unregularized_v1 \
    --use-coral --source-selection top_k \
    --output-dir "${OUTPUT_ROOT}/wo_regularized_xgb/"
}

run_tuned_variant wo_source_selection --use-coral --source-selection none &
pid_no_selection=$!
run_tuned_variant wo_coral --source-selection top_k &
pid_no_coral=$!
run_tuned_variant wo_source_selection_and_coral --source-selection none &
pid_no_both=$!
run_unregularized_variant &
pid_no_regularization=$!

wait "$pid_no_selection"
wait "$pid_no_coral"
wait "$pid_no_both"
wait "$pid_no_regularization"

echo "Strict deterministic Table 7 ablations complete: ${OUTPUT_ROOT}"
