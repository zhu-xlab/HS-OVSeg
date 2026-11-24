# 数据准备

[English](README.md) | **简体中文** | [项目主页](../../README.md)

从项目根目录运行本目录 CLI。可复用代码位于 `ovrs_segmentation/preprocessing/`。服务器上始终显式指定数据和输出根目录；训练数据、人工语义测试集及大图与代码分开存放。

## 整理后的数据约定

每个根目录包含 `dataset.json`、`classes.json` 和 `splits/{train,val,test}.txt`。Scene 使用图像标签，object 使用逐图框标注，semantic 使用类别 ID 掩膜；实际文件路径由元数据描述。LMDB 可替代分散的图像、掩膜和标注文件。

数值类别 ID 属于数据集内部，通过规范化类别文本和 prompts 对齐语义。语义掩膜为整数 ID，忽略值 255；图像与掩膜尺寸必须一致。背景定义以数据集元数据为准，不假定统一背景 ID。

| 用途 | 本目录脚本 |
|---|---|
| Scene 下载 / 整理 | `download_scene_dataset.py`、`prepare_scene_dataset.py` |
| 原生 semantic 数据 | `download_native_dataset.py`、`prepare_native_semantic_dataset.py` |
| 特殊 semantic 来源 | `prepare_five_billion_pixels.py`、`download_flair_hub.py`、`download_global_loveda.py` |
| Object 来源 | `download_dior_r.py`、`download_dota_v2.py`、`download_fair1m.py`、`prepare_xview.py` |
| 人工语义转换 | `convert_manual_coco.py` |
| 词表 | `build_text_prompt_vocabulary.py`、`audit_dataset_text_prompts.py` |
| LMDB 与存储 | `pack_dataset_lmdb.py`、`audit_dataset_storage.py`、`repair_dataset_storage.py` |
| 元数据维护 | `normalize_dataset_names.py`、`export_native_class_backups.py` |
| 校验 | `validate_text_prompt_dataset.py`、`validate_semantic_dataset.py`、`check_dataset_integrity.py` |
| 正射影像 | `download_orthophoto.py` |

使用前查看各脚本 `--help`，不同下载器和转换器的参数不同。存储修复或转换前先执行审计和校验。

## 示例：Scene 数据集

```bash
python scripts/data/download_scene_dataset.py resisc45 --output-dir /path/to/RSOD/NWPU-RESISC45
python scripts/data/prepare_scene_dataset.py resisc45 --dataset-root /path/to/RSOD/NWPU-RESISC45 --workers 8
python scripts/data/validate_text_prompt_dataset.py /path/to/RSOD/NWPU-RESISC45 --split val
```

Scene key 支持 `aid`、`patternnet`、`resisc45`。许可与下载可用性见各来源脚本。

## 独立语义测试数据

人工校正集存放在 `/path/to/RSOD/segmentation_test/` 下：`AID-GT90`、`NWPU-RESISC45-GT135`、`PatternNet-GT114`、`DIOR-R-GT200`、`DOTA-v2-GT160`、`FAIR1M-2.0-GT201` 和 `xView-GT100`。无论原始监督类型如何，转换后统一遵循 semantic 数据约定。这 1,000 张影像仅用于测试，不进入训练或 checkpoint 选择。

大图输入另存于 `/path/to/RSOD/large_scene/`。移动数据后修改清单根目录或 CLI 参数，不要把影像复制到代码仓库。当前语义清单位于 `configs/`，详见[评估指南](../README_EVALUATION.zh-CN.md)。
