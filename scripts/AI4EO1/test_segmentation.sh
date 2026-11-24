#!/bin/bash
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
checkpoint="${OVRS_TEST_CHECKPOINT:-${project_root}/checkpoints/sam3_multilevel/last.pt}"
data_root="${OVRS_DATA_ROOT:-/home/Datasets/RSOD}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4}"
IFS=',' read -r -a gpus <<< "${CUDA_VISIBLE_DEVICES}"
cd "${project_root}"
exec torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:0 --nnodes=1 --nproc-per-node="${#gpus[@]}" \
  -m ovrs_segmentation.evaluation.segmentation_test \
  --suite "${project_root}/configs/segmentation_test.yaml" \
  --data-root "${data_root}" \
  --test-root "${OVRS_TEST_DATA_ROOT:-${data_root}/segmentation_test}" \
  --checkpoint "${checkpoint}" --sam3-checkpoint "${project_root}/checkpoints/sam3.pt" \
  --output-json "${project_root}/outputs/segmentation_test/$(basename "${checkpoint}" .pt)/metrics.json" "$@"
