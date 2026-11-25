# HS-OVSeg

Open-vocabulary remote-sensing segmentation with heterogeneous supervision.

## 1. Install

Use Python 3.12+ and CUDA-enabled PyTorch 2.7+ compatible with [SAM3](https://github.com/facebookresearch/sam3#installation).

```bash
git clone https://github.com/hw2hwei/HS-OVSeg.git
cd HS-OVSeg
python tools/install_sam3_local.py
python -m pip install -e '.[geo]'
```

Request the original SAM3 weights from the [official repository](https://huggingface.co/facebook/sam3) and save them as `checkpoints/sam3.pt`. The trained inference checkpoint is available at [WeiHuang95/HS-OVSeg](https://huggingface.co/WeiHuang95/HS-OVSeg) (36.7 MiB).

## 2. Zero-shot inference

No additional training or training-data download is needed.

```bash
hf download WeiHuang95/HS-OVSeg best-inference.pt best-inference.json \
  --local-dir checkpoints/sam3_multilevel
```

```bash
python -m ovrs_segmentation.inference.sam3_multilevel \
  --input /path/to/image.tif --prompt building --prompt tree \
  --checkpoint checkpoints/sam3_multilevel/best-inference.pt \
  --sam3-checkpoint checkpoints/sam3.pt \
  --device cuda --precision bf16 --no-overlay-only \
  --output-dir outputs/inference
```

Replace the image and prompts with your own. Results include segmentation masks, overlays, and prediction metadata.

## 3. Download data (training / evaluation)

```bash
export OVRS_DATA_ROOT=/path/to/RSOD
python scripts/data/download_huggingface.py \
  --repo-id WeiHuang95/HS-OVSeg-Data \
  --data-root "$OVRS_DATA_ROOT" --mode all --extract-archives
```

The [dataset](https://huggingface.co/datasets/WeiHuang95/HS-OVSeg-Data) contains the selected train/validation and independent semantic test splits for 20 source datasets; OOD data is excluded. Use `--mode train` or `--mode test` to download separately. The initial upload is complete when `data-manifest.json` appears.

## 4. Train (optional)

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
bash scripts/train_sam3_multilevel.sh fresh
# Resume your own run:
bash scripts/train_sam3_multilevel.sh resume
```

Defaults: 30 epochs, 518-pixel crops, batch 4/GPU, LoRA rank 8, BF16. Append options such as `--batch-size 1` after `fresh` or `resume`. Training saves `best.pt` and `last.pt` under `checkpoints/sam3_multilevel/`; the included inference checkpoint cannot resume training.

## 5. Test

```bash
python -m ovrs_segmentation.evaluation.segmentation_test \
  --suite configs/segmentation_test.yaml --scope training_source \
  --data-root "$OVRS_DATA_ROOT" --test-root "$OVRS_DATA_ROOT/segmentation_test" \
  --checkpoint checkpoints/sam3_multilevel/best-inference.pt \
  --sam3-checkpoint checkpoints/sam3.pt \
  --output-json outputs/test.json
```

The report includes mIoU, Dice, and pixel accuracy. For details: [data preparation](scripts/data/README.md), [training](scripts/README_TRAINING.md), [inference](scripts/README_INFERENCE.md), [evaluation](scripts/README_EVALUATION.md).
