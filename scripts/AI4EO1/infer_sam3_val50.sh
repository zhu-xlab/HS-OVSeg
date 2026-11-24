#!/bin/bash
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -r -a gpus <<< "${CUDA_VISIBLE_DEVICES}"
cd "${project_root}"
exec torchrun --master-addr=127.0.0.1 --master-port="${MASTER_PORT:-29500}" \
  --nnodes=1 --nproc-per-node="${#gpus[@]}" \
  -m ovrs_segmentation.inference.sam3_multilevel_suite \
  --data-root "${OVRS_DATA_ROOT:-/home/Datasets/RSOD}" \
  --samples-per-dataset 50 --seed 42 --threshold 0.5 "$@"
