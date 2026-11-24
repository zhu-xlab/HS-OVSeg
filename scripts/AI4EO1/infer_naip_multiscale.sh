#!/bin/bash
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
data_root="${OVRS_DATA_ROOT:-/home/Datasets/RSOD}"
cd "${project_root}"
exec python -m ovrs_segmentation.inference.georaster_multiscale \
  --image "${data_root}/large_scene/san_diego_20km_naip2022/0.5m/image.tif" \
  --classes-json "${data_root}/NWPU-RESISC45/classes.json" \
  --checkpoint "${OVRS_TEST_CHECKPOINT:-${project_root}/checkpoints/sam3_multilevel/last.pt}" \
  --output-dir "${OVRS_MULTISCALE_OUTPUT:-${project_root}/outputs/sam3_multilevel_san_diego_nwpu_multiscale_b2c45}" \
  --image-batch-size "${OVRS_IMAGE_BATCH_SIZE:-2}" --prompt-chunk-size "${OVRS_PROMPT_CHUNK_SIZE:-45}" \
  --nproc-per-node "${OVRS_INFERENCE_GPUS:-7}" --auto-gpus "$@"
