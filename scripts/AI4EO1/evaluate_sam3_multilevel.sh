#!/bin/bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
data_root="${OVRS_DATA_ROOT:-/home/Datasets/RSOD}"
dataset_name="OpenEarthMap"
dataset_root="${data_root}/${dataset_name}"
checkpoint="${project_root}/checkpoints/sam3_multilevel/last.pt"
sam3_checkpoint="${project_root}/checkpoints/sam3.pt"
output_dir="${project_root}/outputs/evaluate_sam3_multilevel"

split="val"
level="lulc"
sam3_input_size=518
prompt_chunk_size=8
workers=4
precision="bf16"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
IFS=',' read -r -a visible_gpus <<< "${CUDA_VISIBLE_DEVICES}"
mkdir -p "${output_dir}"
cd "${project_root}"
exec torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:0 --nnodes=1 --nproc-per-node="${#visible_gpus[@]}" -m ovrs_segmentation.evaluation.sam3_multilevel --dataset-root "${dataset_root}" --split "${split}" --level "${level}" --checkpoint "${checkpoint}" --sam3-checkpoint "${sam3_checkpoint}" --sam3-input-size "${sam3_input_size}" --prompt-chunk-size "${prompt_chunk_size}" --workers "${workers}" --precision "${precision}" --device cuda --output-json "${output_dir}/${dataset_name}_${split}_${level}.json" "$@"
