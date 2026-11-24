# Training

**English** | [简体中文](README_TRAINING.zh-CN.md) | [Project](../README.md)

Run commands from the repository root in a SAM3-capable environment. Python entrypoint: `ovrs_segmentation.training.sam3_multilevel`. Server scripts are launch wrappers; model and training behavior live in the package.

## Data

Prepared roots provide `dataset.json`, `classes.json`, and split files under `splits/`. Images and annotations follow dataset metadata; LMDB storage is also supported. Masks contain dataset-local integer class IDs, with ignore ID 255. Class text comes from the dataset vocabulary.

The joint launchers list 3 scene datasets (AID, NWPU-RESISC45, PatternNet), 4 object datasets (DIOR-R, DOTA-v2, FAIR1M-2.0, xView), and 13 semantic datasets. Manual semantic test sets remain under a separate `segmentation_test` root and are not training/validation targets.

## Joint training

```bash
export OVRS_DATA_ROOT=/path/to/RSOD
export CUDA_VISIBLE_DEVICES=0,1,2,3
bash scripts/train_sam3_multilevel.sh fresh
bash scripts/train_sam3_multilevel.sh resume
```

| Run | Images/epoch | Artifact name |
|---|---:|---|
| Joint | 100,000; scene:object:semantic = 1:2:4 | `sam3_multilevel` |

Hardware: 4 × NVIDIA A100 (40 GB each). Defaults: batch 4/GPU, 30 epochs, BF16, LoRA rank 8, LoRA learning rate `1e-4`. The raw Python CLI defaults to batch 2/GPU; the launchers explicitly override it to 4.

Training scales images by 0.75–1.5, upsamples if needed to cover the crop, and randomly crops 518 × 518. Images, boxes, and masks share geometry; discrete masks use nearest-neighbor interpolation and ignored padding. Validation uses native-scale 518 tiles, including minimum-size upsampling for small images, and restores the evaluation grid.

## Direct Python

For custom dataset subsets, repeat `--scene-root`, `--object-root`, or `--lulc-root`:

```bash
torchrun --standalone --nproc-per-node=4 \
  -m ovrs_segmentation.training.sam3_multilevel \
  --scene-root /path/to/RSOD/AID \
  --object-root /path/to/RSOD/DIOR-R \
  --lulc-root /path/to/RSOD/OpenEarthMap \
  --sam3-checkpoint checkpoints/sam3.pt \
  --config configs/sam3_multilevel.yaml \
  --sam3-input-size 518 --batch-size 4 --epochs 30 \
  --samples-per-epoch 100000 --validation-every 3 \
  --checkpoint-dir checkpoints/sam3_multilevel --output-dir outputs/sam3_multilevel
```

## Validation, saving, and resume

Validation runs every 3 epochs plus the final epoch. Scene uses full-vocabulary Top-1 accuracy, object mAP50, and semantic mIoU. The best score is their equal-weight mean across available datasets. Configured validation sample caps are in `configs/sam3_multilevel.yaml`; this is separate from full semantic test evaluation.

Default saving keeps `last.pt` each epoch and `best.pt` when the validation mean improves. Checkpoints include LoRA/Router, optimizer, scheduler, epoch, and configuration. Fresh runs overwrite files without an existence prompt. Resume uses `last.pt` and restores state; direct Python uses `--resume PATH`.

Server launchers write `training-logs/RUN/`, `checkpoints/RUN/`, and `outputs/RUN/`; Python metrics are under `outputs/RUN/`. Foundation `sam3.pt` is still required for later loading.
