#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
data_root="${OVRS_DATA_ROOT:-/home/Datasets/RSOD}"
exec python -m ovrs_segmentation.inference.georaster_multiscale \
  --model-type native \
  --image "${data_root}/large_scene/san_diego_20km_naip2022/0.5m/image.tif" \
  --classes-json "${data_root}/NWPU-RESISC45/classes.json" \
  --output-dir "${OVRS_MULTISCALE_OUTPUT:-outputs/sam3_native_san_diego_nwpu_multiscale}" \
  --image-batch-size 2 --prompt-chunk-size 45 --nproc-per-node 8 --auto-gpus "$@"
