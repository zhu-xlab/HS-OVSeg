# HS-OVSeg

**异质监督下的开放词汇语义分割** · [English](README.md)

HS-OVSeg 使用场景标签、目标框和语义掩码，通过共享 LoRA 适配冻结的 SAM3。校准后的 Presence 输出 `concept_logits`；像素 Router **仅融合 Semantic 与 Query**，输出 `unified_pixel_logits`。推理只需要图像和文本提示。

## 安装与基础权重

使用 Linux、Python 3.12+、PyTorch 2.7+ 和兼容的 CUDA GPU，先按照 [SAM3 官方要求](https://github.com/facebookresearch/sam3#installation) 安装 CUDA 版 PyTorch，然后：

```bash
git clone https://github.com/hw2hwei/HS-OVSeg.git
cd HS-OVSeg
python tools/install_sam3_local.py
python -m pip install -e '.[dev,geo]'
```

请到 [官方模型仓库](https://huggingface.co/facebook/sam3) 申请原始 **SAM3** 权重，接受其使用条款后，将 `sam3.pt` 放到 `checkpoints/sam3.pt`。本仓库不分发基础权重；SAM3.1 不能直接替代本项目所需的 SAM3 权重。

## 数据发布与下载

发布范围仅包含 20 个训练来源数据集实际参与训练、验证和独立语义测试的样本，**不包含 OOD 数据**。先在保存数据的机器上导出到一个尚不存在的新目录：

```bash
python scripts/data/export_huggingface.py \
  --data-root /path/to/RSOD --output-root /path/to/HS-OVSeg-Data
```

导出只保留 split 引用的图像、必要标注、词表和 split 文件；训练/验证共用的样本只保存一次。所需 LMDB 记录会转换为普通文件，不复制 LMDB、原始压缩包、备份、多余样本或原始目录。源数据保持原样，图像/掩码编码和类别 ID 不变。请将导出目录的**内容**上传到 Hugging Face dataset repo。

目录布局为 `AID/` 等训练目录，以及 `segmentation_test/AID-GT90/` 等独立语义测试目录。每个数据集包含 `dataset.json`、`classes.json`、`splits/` 和被引用的图像/标注。元数据及标注路径须相对于各数据集目录；掩码为原有类别 ID，255 为忽略值。测试标签不参与训练或验证。原始数据转换方法见[数据准备](scripts/data/README.zh-CN.md)。

数据仓库为 [WeiHuang95/HS-OVSeg-Data](https://huggingface.co/datasets/WeiHuang95/HS-OVSeg-Data)，仓库 ID 也可通过参数更换。私有或 gated 数据使用 `hf auth login` 或环境变量 `HF_TOKEN` 认证，不将令牌写进配置：

```bash
export OVRS_HF_DATASET_REPO=WeiHuang95/HS-OVSeg-Data
export OVRS_DATA_ROOT=/path/to/RSOD
python scripts/data/download_huggingface.py \
  --repo-id "$OVRS_HF_DATASET_REPO" --data-root "$OVRS_DATA_ROOT" --mode all --extract-archives
```

下载默认排除 OOD，并使用真实训练/测试 reader 检查所选 split、解码各 split 的首个样本；这不是逐样本完整审计。`--mode train` 只下载训练/验证，`--mode test` 只下载测试；`--revision TAG_OR_COMMIT` 固定版本；`--test-suite PATH` 选择测试子集。重复运行会复用已下载且未变更的文件。

发布版本按 `archives/AID/part-00000.tar` 等路径保存整理后的 tar 分片，`data-manifest.json` 记录 SHA-256；下载命令中的 `--extract-archives` 会重建目录。维护者可用 `python scripts/data/publish_huggingface.py --repo-id WeiHuang95/HS-OVSeg-Data --data-root /path/to/RSOD --work-root /path/to/publication --allow-public` 完成导出、默认 4 GiB 分片和上传。公开仓库上传须显式传入 `--allow-public`；重复运行跳过已校验上传的分片。也支持按数据集上传 `.tar`、`.tar.gz`、`.tgz` 或 `.zip` 并添加 `--extract-archives`。例如 `AID.tar.gz` 内须包含 `AID/dataset.json`，`segmentation_test/AID-GT90.tar.gz` 内须包含 `segmentation_test/AID-GT90/dataset.json`；内部路径均相对于 RSOD 根目录。解压会更新对应本地文件。原始数据或 Parquet 格式需先转换。

## 训练

```bash
export OVRS_DATA_ROOT=/path/to/RSOD
export CUDA_VISIBLE_DEVICES=0,1,2,3
bash scripts/train_sam3_multilevel.sh fresh
# 使用自己训练产生的 last.pt 恢复优化器和调度器状态：
bash scripts/train_sam3_multilevel.sh resume
```

默认使用全部 20 个数据集（3 场景、4 目标、13 语义）：30 epochs，随机缩放 0.75–1.5 后裁剪 518×518，每 GPU batch 4，每 epoch 100,000 样本，场景:目标:语义采样 1:2:4，LoRA rank 8，学习率 `1e-4`，BF16；原始训练配置为 4×A100 40 GB。可在 `fresh` / `resume` 后追加参数，如 `--batch-size 1`。

场景与目标监督使用暂时关闭 LoRA 的冻结 SAM3 在线生成伪掩码；语义监督使用真实像素标注。每 3 个 epoch 及最终 epoch 验证一次，`best.pt` 根据各数据集主验证指标的等权均值选择（场景 Top-1、目标 mAP50、语义 mIoU），不是语义测试得分。训练权重写入 `checkpoints/sam3_multilevel/{best,last}.pt`，日志和预测保持本地。更多参数见[训练说明](scripts/README_TRAINING.zh-CN.md)。

## 推理 checkpoint 与预测

已附带 [`best-inference.pt`](checkpoints/sam3_multilevel/best-inference.pt)，36.7 MiB，来自 epoch 21 / step 131250 的 `best.pt`。全部 LoRA/Router 张量保持一致，仅去掉优化器、调度器和机器路径。它用于推理/评估，不能用于 `--resume`；[元数据与 SHA-256](checkpoints/sam3_multilevel/best-inference.json) 同时提供。基础 SAM3 权重仍需官方申请并显式指定。

```bash
python -m ovrs_segmentation.inference.sam3_multilevel \
  --input /path/to/image.tif --prompt building --prompt tree \
  --checkpoint checkpoints/sam3_multilevel/best-inference.pt \
  --sam3-checkpoint checkpoints/sam3.pt \
  --device cuda --precision bf16 --no-overlay-only \
  --output-dir outputs/inference
```

也可用 `--classes-json PATH` 提供词表。`--no-overlay-only` 同时保存掩码和预测元数据，`--save-probabilities` 额外保存概率。推理以原始尺度 518 像素分块并拼接回原网格，小图保持比例上采样。此图像推理入口使用 Presence 概率乘统一像素概率；下面的语义基准直接使用 `unified_pixel_logits`。更多选项见[推理说明](scripts/README_INFERENCE.zh-CN.md)。

## 语义测试

```bash
python -m ovrs_segmentation.evaluation.segmentation_test \
  --suite configs/segmentation_test.yaml --scope training_source \
  --data-root "$OVRS_DATA_ROOT" --test-root "$OVRS_DATA_ROOT/segmentation_test" \
  --checkpoint checkpoints/sam3_multilevel/best-inference.pt \
  --sam3-checkpoint checkpoints/sam3.pt \
  --output-json outputs/test.json
```

发布数据覆盖 20 个训练来源对应的独立像素测试集。测试目录表还保留 5 个可选 OOD 条目，需另行准备，不纳入本次数据发布或默认下载/导出。`configs/segmentation_test_manual.yaml` 选择 7 个人工校正场景/目标测试集，共 1,000 张。测试在原始标注网格报告 mIoU、Dice、像素准确率及各类别指标，忽略 255。更多选项见[评估说明](scripts/README_EVALUATION.zh-CN.md)。

核心代码位于 `ovrs_segmentation/`，`scripts/` 保持命令行包装。Git 排除消融/对比实验、论文资产、数据、训练日志、生成结果、凭据和 SAM3 基础权重。测试命令为 `python -m pytest`。
