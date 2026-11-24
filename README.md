# HS-OVSeg

**Open-Vocabulary Semantic Segmentation with Heterogeneous Supervision**

[简体中文](README.zh-CN.md)

HS-OVSeg adapts frozen SAM3 using shared LoRA and scene labels, object boxes, and semantic masks. Calibrated Presence produces `concept_logits`; a learned pixel Router fuses **only Semantic and Query** into `unified_pixel_logits`. Inference takes an image and text prompts, with no supervision-level labels or ground-truth boxes.

## Installation

Use Linux, Python 3.12+, PyTorch 2.7+, and a CUDA GPU compatible with [official SAM3](https://github.com/facebookresearch/sam3#installation). Install CUDA-enabled PyTorch following those instructions, then:

```bash
git clone https://github.com/hw2hwei/HS-OVSeg.git
cd HS-OVSeg
python tools/install_sam3_local.py
python -m pip install -e '.[dev,geo]'
```

Request SAM3 checkpoint access from the [official model repository](https://huggingface.co/facebook/sam3), accept its terms, and download the original **SAM3** `sam3.pt` to `checkpoints/sam3.pt`. Foundation weights are not included. The adapted checkpoint requires these original weights; SAM3.1 is not a drop-in replacement.

## Data from Hugging Face

Published data includes only the 20 training-source datasets and their selected train/validation/test samples; OOD data is excluded. The dataset repository ID is configurable; replace `OWNER/DATASET` with the repository you publish. For private/gated data, use `hf auth login` or set `HF_TOKEN` in your environment. No token belongs in a configuration file.

```bash
export OVRS_HF_DATASET_REPO=WeiHuang95/HS-OVSeg-Data
export OVRS_DATA_ROOT=/path/to/RSOD
python scripts/data/download_huggingface.py \
  --repo-id "$OVRS_HF_DATASET_REPO" --data-root "$OVRS_DATA_ROOT" --mode all
```

Before uploading, export a clean publication tree from your prepared datasets. Run this on the machine holding the data, with a new output directory:

```bash
python scripts/data/export_huggingface.py \
  --data-root /path/to/RSOD --output-root /path/to/HS-OVSeg-Data
```

The exporter materializes only split-referenced files, unpacks required LMDB records, stores samples shared by train/val once, and excludes OOD data, raw archives, backups, unused samples, and LMDB files. Source data is left untouched; image/mask encodings and class IDs are preserved. Upload the **contents** of the export directory as your dataset repository:


```text
AID/                       # and the other training dataset directories
  dataset.json
  classes.json
  splits/{train,val}.txt
  ...                      # image labels / boxes / masks and images
segmentation_test/
  AID-GT90/                # and other semantic test directories
    dataset.json
    classes.json
    splits/test.txt
    ...                    # class-ID masks and images
```

Metadata and annotation paths must be relative to each dataset root. Preserve vocabularies, split files, and annotation files from the export. Masks use dataset-local class IDs and ignore ID 255. Keep corrected semantic test labels out of training and validation. See [data preparation](scripts/data/README.md) for raw-data conversion.

Downloads preserve the directory structure and reuse up-to-date files. Use `--mode train` or `--mode test` to download separately, and `--revision TAG_OR_COMMIT` for reproducibility. The default test scope is `training_source` (20 datasets); use `--test-suite PATH` for a custom subset. After downloading, the command resolves every selected split and decodes its first sample with the actual training/test reader; it does not audit every image or mask.

Alternatively, upload `AID.tar.gz` / `AID.zip`, etc., and `segmentation_test/AID-GT90.tar.gz`, etc., then add `--extract-archives`. Each archive must contain paths relative to the RSOD root, including its dataset directory, e.g. `AID/dataset.json` or `segmentation_test/AID-GT90/dataset.json`. Extraction updates matching local files. Arbitrary raw datasets and Parquet exports require preparation first.

## Training

```bash
export OVRS_DATA_ROOT=/path/to/RSOD
export CUDA_VISIBLE_DEVICES=0,1,2,3
bash scripts/train_sam3_multilevel.sh fresh
# Resume your own training run, including optimizer/scheduler state:
bash scripts/train_sam3_multilevel.sh resume
```

The portable launcher jointly trains on 20 datasets (3 scene, 4 object, 13 semantic). Defaults: 30 epochs, 518 x 518 crops after random scaling 0.75-1.5, batch 4/GPU, 100,000 images/epoch, scene:object:semantic sampling 1:2:4, LoRA rank 8, learning rate `1e-4`, BF16. The original setup uses 4 x A100 40 GB. Extra CLI options may follow `fresh` or `resume` (e.g. `--batch-size 1`).

Scene and object supervision use online pseudo masks from frozen SAM3 with LoRA temporarily disabled; semantic supervision uses ground-truth masks. Validation runs every 3 epochs and at the final epoch. `best.pt` uses the equal-weight mean of per-dataset validation scores (scene Top-1, object mAP50, semantic mIoU), not a semantic test score. Training writes `checkpoints/sam3_multilevel/{best,last}.pt`; logs and outputs stay local. See [training options](scripts/README_TRAINING.md).

## Inference checkpoint and prediction

Included: [`best-inference.pt`](checkpoints/sam3_multilevel/best-inference.pt), **36.7 MiB**, selected at epoch 21 / step 131250. It retains every LoRA/Router tensor from the trained `best.pt` unchanged and omits optimizer/scheduler state and machine paths. It is for inference/evaluation, not `--resume`. [Metadata and SHA-256](checkpoints/sam3_multilevel/best-inference.json) accompany the weights. Always provide your own foundation weight path.

```bash
python -m ovrs_segmentation.inference.sam3_multilevel \
  --input /path/to/image.tif --prompt building --prompt tree \
  --checkpoint checkpoints/sam3_multilevel/best-inference.pt \
  --sam3-checkpoint checkpoints/sam3.pt \
  --device cuda --precision bf16 --no-overlay-only \
  --output-dir outputs/inference
```

Use `--classes-json PATH` instead of repeated `--prompt` options for a class catalog. `--no-overlay-only` saves masks and prediction metadata as well as overlays; `--save-probabilities` also saves probabilities. Native-scale 518-pixel tiles are stitched onto the original grid, with aspect-preserving upsampling for small images. Image inference uses Presence probability times unified pixel probability. The semantic benchmark below uses `unified_pixel_logits` directly. See [inference options](scripts/README_INFERENCE.md).

## Semantic testing

```bash
python -m ovrs_segmentation.evaluation.segmentation_test \
  --suite configs/segmentation_test.yaml --scope training_source \
  --data-root "$OVRS_DATA_ROOT" --test-root "$OVRS_DATA_ROOT/segmentation_test" \
  --checkpoint checkpoints/sam3_multilevel/best-inference.pt \
  --sam3-checkpoint checkpoints/sam3.pt \
  --output-json outputs/test.json
```

The published dataset covers the 20 source-domain entries, all with separate pixel-labeled test roots. The catalog also retains 5 optional OOD entries for independently prepared data; these are excluded from the published data and default download/export. `configs/segmentation_test_manual.yaml` selects the seven corrected scene/object sets (1,000 images). Testing reports mIoU, Dice, pixel accuracy, and per-class metrics on the original label grid, ignoring 255. See [evaluation options](scripts/README_EVALUATION.md).

Core code lives under `ovrs_segmentation/`; `scripts/` contains CLI launchers. Ablation/comparison experiments, paper assets, datasets, training logs, generated outputs, credentials, and SAM3 foundation weights are excluded from Git. Run the published tests with `python -m pytest`.
