# HS-OVSeg

异质监督下的遥感图像开放词汇语义分割。

[English](README.md)

## 1. 安装

使用 Python 3.12+、CUDA 版 PyTorch 2.7+，环境需符合 [SAM3 官方要求](https://github.com/facebookresearch/sam3#installation)。

```bash
git clone https://github.com/hw2hwei/HS-OVSeg.git
cd HS-OVSeg
python tools/install_sam3_local.py
python -m pip install -e '.[dev,geo]'
```

到[官方模型仓库](https://huggingface.co/facebook/sam3)申请原始 SAM3 权重，保存为 `checkpoints/sam3.pt`。本项目训练后的推理权重已附带：`checkpoints/sam3_multilevel/best-inference.pt`。

## 2. 下载数据

```bash
export OVRS_DATA_ROOT=/path/to/RSOD
python scripts/data/download_huggingface.py \
  --repo-id WeiHuang95/HS-OVSeg-Data \
  --data-root "$OVRS_DATA_ROOT" --mode all --extract-archives
```

[数据仓库](https://huggingface.co/datasets/WeiHuang95/HS-OVSeg-Data)包含 20 个来源数据集所选的训练、验证及独立语义测试部分，不含 OOD。可用 `--mode train` 或 `--mode test` 分别下载。首次上传完成后，仓库中会出现 `data-manifest.json`。

## 3. 训练

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
bash scripts/train_sam3_multilevel.sh fresh
# 恢复自己的训练任务：
bash scripts/train_sam3_multilevel.sh resume
```

默认 30 epochs、518 像素裁剪、每 GPU batch 4、LoRA rank 8、BF16。可在 `fresh` 或 `resume` 后追加参数，如 `--batch-size 1`。训练权重保存至 `checkpoints/sam3_multilevel/{best,last}.pt`；附带的推理权重不能用于断点续训。

## 4. 推理

```bash
python -m ovrs_segmentation.inference.sam3_multilevel \
  --input /path/to/image.tif --prompt building --prompt tree \
  --checkpoint checkpoints/sam3_multilevel/best-inference.pt \
  --sam3-checkpoint checkpoints/sam3.pt \
  --device cuda --precision bf16 --no-overlay-only \
  --output-dir outputs/inference
```

替换图像路径和文本提示即可。结果包含分割掩码、叠加图和预测元数据。

## 5. 测试

```bash
python -m ovrs_segmentation.evaluation.segmentation_test \
  --suite configs/segmentation_test.yaml --scope training_source \
  --data-root "$OVRS_DATA_ROOT" --test-root "$OVRS_DATA_ROOT/segmentation_test" \
  --checkpoint checkpoints/sam3_multilevel/best-inference.pt \
  --sam3-checkpoint checkpoints/sam3.pt \
  --output-json outputs/test.json
```

报告包含 mIoU、Dice 和像素准确率。更多选项：[数据准备](scripts/data/README.zh-CN.md)、[训练](scripts/README_TRAINING.zh-CN.md)、[推理](scripts/README_INFERENCE.zh-CN.md)、[评估](scripts/README_EVALUATION.zh-CN.md)。
