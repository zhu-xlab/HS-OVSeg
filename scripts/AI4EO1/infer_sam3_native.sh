#!/bin/bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
data_root="${OVRS_DATA_ROOT:-/home/Datasets/RSOD}"
dataset_name="OpenEarthMap"
dataset_root="${data_root}/${dataset_name}"
sam3_checkpoint="${project_root}/checkpoints/sam3.pt"
output_dir="${project_root}/outputs/infer_sam3_native/${dataset_name}"

split="val"
level="lulc"
prompt_batch_size=4
sam3_input_size=518
threshold=0.5
precision="bf16"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4}"
mkdir -p "${output_dir}"
cd "${project_root}"
exec python -m ovrs_segmentation.inference.sam3_native --dataset-root "${dataset_root}" --classes-json "${dataset_root}/classes.json" --split "${split}" --level "${level}" --checkpoint "${sam3_checkpoint}" --sam3-input-size "${sam3_input_size}" --prompt-policy native --prompt-batch-size "${prompt_batch_size}" --threshold "${threshold}" --precision "${precision}" --device cuda --output-dir "${output_dir}" --overlay-only --overwrite "$@"
