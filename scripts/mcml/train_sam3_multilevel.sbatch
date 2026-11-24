#!/bin/bash
#SBATCH -q mcml
#SBATCH --partition=mcml-dgx-a100-40x8
#SBATCH --job-name=sam3-all
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

container_image="/dss/dsshome1/06/ge89dol2/pytorch_container_enhanced.sqsh"
mount_src="/dss/dssmcmlfs01/pn36te/pn36te-dss-0000/ge89dol2"
mount_dst="/home"
host_root="${mount_src}/Codes/OVRS_segmentation"
container_root="${mount_dst}/Codes/OVRS_segmentation"
container_data_root="${mount_dst}/Datasets/RSOD"

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
    "${container_data_root}/AID"
    "${container_data_root}/NWPU-RESISC45"
    "${container_data_root}/PatternNet"
)
object_roots=(
    "${container_data_root}/DIOR-R"
    "${container_data_root}/DOTA-v2"
    "${container_data_root}/FAIR1M-2.0"
    "${container_data_root}/xView"
)
lulc_roots=(
    "${container_data_root}/DeepGlobe-LandCover"
    "${container_data_root}/FLAIR-HUB"
    "${container_data_root}/Five-Billion-Pixels"
    "${container_data_root}/FloodNet"
    "${container_data_root}/GID-15"
    "${container_data_root}/Global-LoveDA"
    "${container_data_root}/ISPRS-Potsdam"
    "${container_data_root}/LandCover.ai"
    "${container_data_root}/OpenEarthMap"
    "${container_data_root}/RescueNet"
    "${container_data_root}/SkyScapes"
    "${container_data_root}/UAVid"
    "${container_data_root}/iSAID"
)

run_name="sam3_multilevel"

dataset_args=()
for root in "${scene_roots[@]}"; do dataset_args+=(--scene-root "${root}"); done
for root in "${object_roots[@]}"; do dataset_args+=(--object-root "${root}"); done
for root in "${lulc_roots[@]}"; do dataset_args+=(--lulc-root "${root}"); done

run_mode="${1:-fresh}"
if (( $# )); then shift; fi
case "${run_mode}" in
fresh|resume) ;;
*)
    echo "First argument must be 'fresh' or 'resume', got: ${run_mode}" >&2
    exit 2
    ;;
esac

nproc_per_node=4
host_log_dir="${host_root}/training-logs/${run_name}"
container_log_dir="${container_root}/training-logs/${run_name}"
host_checkpoint_dir="${host_root}/checkpoints/${run_name}"
container_checkpoint_dir="${container_root}/checkpoints/${run_name}"
host_output_dir="${host_root}/outputs/${run_name}"
container_output_dir="${container_root}/outputs/${run_name}"

training_args=(
    "${dataset_args[@]}"
    --sam3-checkpoint "${container_root}/checkpoints/sam3.pt"
    --config "${container_root}/configs/sam3_multilevel.yaml"
    --sam3-input-size "${sam3_input_size}" --lora-rank "${lora_rank}"
    --lr-lora "${lr_lora}" --samples-per-epoch "${samples_per_epoch}"
    --epochs "${epochs}" --batch-size "${batch_size}" --workers "${workers}"
    --precision "${precision}" --log-every "${log_every}"
    --validation-every "${validation_every}"
)

mkdir -p "${host_log_dir}" "${host_checkpoint_dir}" "${host_output_dir}"
echo "[run] name=${run_name} samples_per_epoch=${samples_per_epoch} global_batch=$((nproc_per_node * batch_size)) epochs=${epochs}"

srun \
    --container-image="${container_image}" \
    --container-mounts="${mount_src}:${mount_dst}" \
    --mpi=pmi2 \
    --ntasks=1 \
    --output="${host_log_dir}/out-%j.log" \
    --error="${host_log_dir}/err-%j.log" \
    bash -c '
set -euo pipefail
project_root=$1; checkpoint_dir=$2; output_dir=$3; log_dir=$4; run_mode=$5; nproc=$6
shift 6
cd "${project_root}"
export OMP_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
resume_args=(); tee_args=()
if [[ "${run_mode}" == resume ]]; then
    resume_args=(--resume "${checkpoint_dir}/last.pt")
    tee_args=(-a)
fi
torchrun --rdzv-backend=c10d --rdzv-endpoint=localhost:0 --nnodes=1 --nproc-per-node="${nproc}" \
    -m ovrs_segmentation.training.sam3_multilevel --checkpoint-dir "${checkpoint_dir}" \
    --output-dir "${output_dir}" "${resume_args[@]}" "$@" 2>&1 | tee "${tee_args[@]}" "${log_dir}/train_out.log"
' bash "${container_root}" "${container_checkpoint_dir}" "${container_output_dir}" \
    "${container_log_dir}" "${run_mode}" "${nproc_per_node}" "${training_args[@]}" "$@"
