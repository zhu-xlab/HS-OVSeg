#!/bin/bash

set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
data_root="${OVRS_DATA_ROOT:?Set OVRS_DATA_ROOT to the prepared dataset root}"
checkpoint_dir="${project_root}/checkpoints/sam3_multilevel"
output_dir="${project_root}/outputs/sam3_multilevel"
log_dir="${project_root}/training-logs/sam3_multilevel"

sam3_input_size=518
lora_rank=8
lr_lora=1e-4
samples_per_epoch=100000
epochs=30
batch_size=4
workers=4
precision=bf16
log_every=50
validation_every=3

scene_roots=(
    "${data_root}/AID"
    "${data_root}/NWPU-RESISC45"
    "${data_root}/PatternNet"
)
object_roots=(
    "${data_root}/DIOR-R"
    "${data_root}/DOTA-v2"
    "${data_root}/FAIR1M-2.0"
    "${data_root}/xView"
)
lulc_roots=(
    "${data_root}/DeepGlobe-LandCover"
    "${data_root}/FLAIR-HUB"
    "${data_root}/Five-Billion-Pixels"
    "${data_root}/FloodNet"
    "${data_root}/GID-15"
    "${data_root}/Global-LoveDA"
    "${data_root}/ISPRS-Potsdam"
    "${data_root}/LandCover.ai"
    "${data_root}/OpenEarthMap"
    "${data_root}/RescueNet"
    "${data_root}/SkyScapes"
    "${data_root}/UAVid"
    "${data_root}/iSAID"
)

dataset_args=()
for root in "${scene_roots[@]}"; do dataset_args+=(--scene-root "${root}"); done
for root in "${object_roots[@]}"; do dataset_args+=(--object-root "${root}"); done
for root in "${lulc_roots[@]}"; do dataset_args+=(--lulc-root "${root}"); done

run_mode="${1:-fresh}"
if (( $# )); then shift; fi
resume_args=()
tee_args=()
case "${run_mode}" in
fresh) ;;
resume)
    resume_args=(--resume "${checkpoint_dir}/last.pt")
    tee_args=(-a)
    ;;
*)
    echo "First argument must be 'fresh' or 'resume', got: ${run_mode}" >&2
    exit 2
    ;;
esac

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
IFS=',' read -r -a visible_gpus <<< "${CUDA_VISIBLE_DEVICES}"
nproc_per_node="${#visible_gpus[@]}"

mkdir -p "${checkpoint_dir}" "${output_dir}" "${log_dir}"
cd "${project_root}"

torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:0 --nnodes=1 --nproc-per-node="${nproc_per_node}" -m ovrs_segmentation.training.sam3_multilevel "${dataset_args[@]}" --sam3-checkpoint "${project_root}/checkpoints/sam3.pt" --config configs/sam3_multilevel.yaml --checkpoint-dir "${checkpoint_dir}" --output-dir "${output_dir}" --sam3-input-size "${sam3_input_size}" --lora-rank "${lora_rank}" --lr-lora "${lr_lora}" --samples-per-epoch "${samples_per_epoch}" --epochs "${epochs}" --batch-size "${batch_size}" --workers "${workers}" --precision "${precision}" --log-every "${log_every}" --validation-every "${validation_every}" "${resume_args[@]}" "$@" 2>&1 | tee "${tee_args[@]}" "${log_dir}/train_out.log"
