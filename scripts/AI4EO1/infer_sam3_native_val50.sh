#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
python -m ovrs_segmentation.inference.sam3_multilevel_suite \
  --model-type native --output-dir outputs/sam3_native_val50 \
  --reference-summary outputs/sam3_multilevel_val50/summary.json \
  --data-root "${OVRS_DATA_ROOT:-/home/Datasets/RSOD}" \
  --sam3-input-size 518 --samples-per-dataset 50 --seed 42 --threshold 0.5 "$@"
