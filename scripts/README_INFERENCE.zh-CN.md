# 推理

[English](README_INFERENCE.md) | **简体中文** | [项目主页](../README.zh-CN.md)

## 影像与文本提示

```bash
python -m ovrs_segmentation.inference.sam3_multilevel \
  --input /path/to/image.tif --prompt airport --prompt forest \
  --checkpoint checkpoints/sam3_multilevel/best.pt \
  --sam3-checkpoint checkpoints/sam3.pt \
  --sam3-input-size 518 --precision bf16 --device cuda \
  --output-dir outputs/sam3_inference --overwrite
```

更多类别重复 `--prompt`。可用 `--classes-json PATH` 读取词表，或 `--dataset-root PATH --split val --prompt-policy native` 对数据集推理。`--prompt-batch-size` 控制文本批大小。默认保存叠加图；加 `--no-overlay-only` 输出其他产物，`--save-probabilities` 保存概率数组。

推理输入为图像和文本，不需要 GT 框或标注级别路由。该入口使用 `sigmoid(concept_logits) × sigmoid(unified_pixel_logits)`，再按类别取最大值并应用阈值（默认 0.5），与直接使用 unified logits 的语义基准不同。

普通推理在原始尺度按 518 × 518 无重叠分块，边缘不足时填充，仅拼接有效像素。小图保持宽高比上采样，预测后恢复原尺寸。该入口不会自动启用重叠或多尺度。

## 每数据集 50 张验证可视化

```bash
python -m ovrs_segmentation.inference.sam3_multilevel_suite \
  --suite configs/inference_val.yaml --data-root /path/to/RSOD \
  --checkpoint checkpoints/sam3_multilevel/best.pt \
  --sam3-checkpoint checkpoints/sam3.pt \
  --samples-per-dataset 50 --seed 42 --threshold 0.5 \
  --output-dir outputs/sam3_multilevel_val50
```

清单覆盖 20 个验证集，固定种子采样，查询每个数据集全部类别。对 unified 最大激活应用 0.5 阈值，输出类别 ID 掩膜、带图例叠加图及样本 ID。这是可视化协议，不是最终人工语义测试。

原生 SAM3 使用相同目录结构时，加 `--model-type native`，输出改为 `--output-dir outputs/sam3_native_val50`。Native 使用原始 Semantic logits，不使用适配或学习 Router；可用 `--reference-summary PATH` 引用已完成的适配结果摘要，对齐采样。不同结果使用不同目录，脚本会防止覆盖已有摘要。

## 地理大图

`ovrs_segmentation.inference.georaster` 和 `.georaster_multiscale` 提供独立栅格流程，重叠、batch、栅格和尺度参数见对应模块 `--help`。安装 `.[geo]` 提供 rasterio/pyproj 支持；导出结果需保留 CRS、transform 和网格元数据。

NAIP 流程在服务器保留 0.5、1、2、5 m 各尺度结果，逐类别对齐到 1 m 后，按**跨尺度概率最大值**融合。0.5 m 网格来自所选 0.6 m NAIP 影像重采样。概率导出使用 [0,1] 概率，并按配置施加 Presence 门控；uint8 为概率 × 255。
