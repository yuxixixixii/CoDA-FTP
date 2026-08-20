#!/usr/bin/env bash
# Strict source-only nested selected-source DANN diagnostic.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

OUTPUT_ROOT="${OUTPUT_ROOT:-result_1/table8_dann_source_only_20260818}"
PYTHON_BIN="${PYTHON_BIN:-/home/zhangyu/.conda/envs/unixcoder/bin/python}"
CUDA_DEVICE="${CUDA_DEVICE:-1}"
TARGETS="${TARGETS:-}"

export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

ARGS=(
  --precomputed-data-path result/processed_data_with_vocabulary_per_test.csv
  --flakeflagger-features-path input_data/FlakeFlaggerFeaturesTypes.csv
  --source-selection-top-k 6 --source-selection-min-projects 3
  --candidate-set nested_v1 --batch-size 512 --device cuda --random-state 20260818
  --output-dir "$OUTPUT_ROOT"
)
if [[ -n "$TARGETS" ]]; then
  ARGS+=(--targets "$TARGETS")
fi

exec "$PYTHON_BIN" -u external_baselines/run_dann_source_only_nested.py "${ARGS[@]}"
