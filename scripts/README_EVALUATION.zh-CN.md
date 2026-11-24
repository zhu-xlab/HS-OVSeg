# 评估

[English](README_EVALUATION.md) | **简体中文** | [项目主页](../README.md)

训练验证保留各原始任务指标；最终语义测试则对所有监督级别统一评估像素掩膜。两类报告分开使用。

| 阶段 | Scene | Object | Semantic |
|---|---|---|---|
| 验证 / best 选择 | 完整词表 Top-1 | mAP50 | mIoU |
| 语义测试 | 像素指标 | 像素指标 | 像素指标 |

## 语义测试

```bash
python -m ovrs_segmentation.evaluation.segmentation_test \
  --suite configs/segmentation_test_manual.yaml \
  --data-root /path/to/RSOD --test-root /path/to/RSOD/segmentation_test \
  --checkpoint checkpoints/sam3_multilevel/best.pt \
  --sam3-checkpoint checkpoints/sam3.pt \
  --output-json outputs/sam3_multilevel/semantic_test.json
```

测试最新状态时改用 `last.pt`。单个已整理语义数据集可将 `--suite` 替换为 `--dataset-root /path/to/DATASET`。在相同命令中添加 `--preflight` 可只检查数据而不加载权重；解析器仍要求 checkpoint 参数。

| 清单 | 范围 |
|---|---|
| `configs/segmentation_test_manual.yaml` | AID 90、NWPU-RESISC45 135、PatternNet 114、DIOR-R 200、DOTA-v2 160、FAIR1M-2.0 201、xView 100 |
| `configs/segmentation_test.yaml` | 全部 20 个来源数据集的语义测试目标 |

7 个人工校正集共提供 1,000 张测试影像，位于 `segmentation_test` 下，目录名采用“数据集名-GT样本数”。其他 semantic 数据集使用整理后的 test 划分。数据缺失时保存未完成摘要并退出 2，报告时需明确说明覆盖范围。

模型查询数据集所有可用类别，对 `unified_pixel_logits` 取 argmax；有背景定义时，最大 logit 为负的像素映射为背景。评估保持原始像素尺寸，排除忽略值 255。输出 mIoU、mean Dice、像素精度、有效像素数和逐类指标；定义背景时，平均 IoU/Dice 排除背景。

`--max-samples N` 仅用于限量检查，报告需注明部分评估。默认 `--precision bf16`；`--prompt-chunk-size` 控制类别提示显存开销。

## 原始任务验证

`ovrs_segmentation.evaluation.sam3_multilevel` 和 `.sam3_native` 保留原始任务验证协议，与上述语义测试不同。每次评估需记录预测规则、词表、基础权重和预处理。
