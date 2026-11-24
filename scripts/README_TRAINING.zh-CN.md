# 训练

[English](README_TRAINING.md) | **简体中文** | [项目主页](../README.md)

在具备 SAM3 依赖的环境中，从项目根目录运行命令。Python 入口为 `ovrs_segmentation.training.sam3_multilevel`。服务器脚本负责启动，模型与训练实现在包内。

## 数据

整理后的数据根目录提供 `dataset.json`、`classes.json` 和 `splits/` 下的划分文件。影像与标注位置由元数据描述，也支持 LMDB。掩膜为数据集内整数类别 ID，忽略值 255；类别文本来自数据集词表。

联合脚本列出 3 个 scene 集（AID、NWPU-RESISC45、PatternNet）、4 个 object 集（DIOR-R、DOTA-v2、FAIR1M-2.0、xView）和 13 个 semantic 集。人工语义测试集单独放在 `segmentation_test`，不用于训练或验证。

## 联合训练

```bash
export OVRS_DATA_ROOT=/path/to/RSOD
export CUDA_VISIBLE_DEVICES=0,1,2,3
bash scripts/train_sam3_multilevel.sh fresh
bash scripts/train_sam3_multilevel.sh resume
```

| 运行 | 每轮采样量 | 产物名称 |
|---|---:|---|
| 联合 | 100,000；scene:object:semantic = 1:2:4 | `sam3_multilevel` |

硬件为 4 张 NVIDIA A100（每张 40 GB）；默认每 GPU batch 4、30 epoch、BF16、LoRA rank 8、LoRA 学习率 `1e-4`。Python CLI 本身默认每 GPU batch 2，启动脚本显式覆盖为 4。

训练随机缩放 0.75–1.5 倍，尺寸不足时上采样覆盖裁剪窗口，然后随机裁剪 518 × 518。影像、框和掩膜采用相同几何变换；离散掩膜最近邻插值，填充区域忽略。验证采用原尺度 518 分块，小图先满足最小输入尺寸，再恢复评估网格。

## 直接使用 Python

自定义子集可重复指定 `--scene-root`、`--object-root` 或 `--lulc-root`：

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

## 验证、保存与恢复

每 3 轮及最后一轮验证。Scene 使用完整词表 Top-1 accuracy，object 用 mAP50，semantic 用 mIoU；best 分数为可用数据集主指标等权平均。验证样本上限在 `configs/sam3_multilevel.yaml` 配置，与最终全量语义测试分开。

默认每轮保存 `last.pt`，验证均值提升时保存 `best.pt`。包含 LoRA/Router、optimizer、scheduler、epoch 和配置。Fresh 直接覆盖，不检查已有文件；resume 从 `last.pt` 恢复，直接 Python 可用 `--resume PATH`。

服务器脚本产物为 `training-logs/RUN/`、`checkpoints/RUN/` 和 `outputs/RUN/`，Python 指标在 `outputs/RUN/`。后续加载仍需基础 `sam3.pt`。
