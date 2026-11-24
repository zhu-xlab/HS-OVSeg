# Data preparation

**English** | [简体中文](README.zh-CN.md) | [Project](../../README.md)

Run these CLI wrappers from the project root. Reusable preparation code lives in `ovrs_segmentation/preprocessing/`. Always pass explicit dataset/output roots on servers; training data, corrected semantic test data, and large-scene imagery are stored separately from code.

## Prepared dataset contract

Each root contains `dataset.json`, `classes.json`, and `splits/{train,val,test}.txt`. Scene samples use image labels, object samples use per-image box annotations, and semantic samples use class-ID masks. Actual file paths follow metadata. LMDB packing may replace loose image/mask/annotation storage.

Numeric class IDs are dataset-local; normalized class texts/prompts provide semantic alignment. Semantic masks use integer IDs, with 255 ignored. Image and mask dimensions must agree. Background follows dataset metadata rather than an assumed universal ID.

| Purpose | Scripts in this directory |
|---|---|
| Scene download / preparation | `download_scene_dataset.py`, `prepare_scene_dataset.py` |
| Native semantic data | `download_native_dataset.py`, `prepare_native_semantic_dataset.py` |
| Special semantic sources | `prepare_five_billion_pixels.py`, `download_flair_hub.py`, `download_global_loveda.py` |
| Object sources | `download_dior_r.py`, `download_dota_v2.py`, `download_fair1m.py`, `prepare_xview.py` |
| Manual semantic conversion | `convert_manual_coco.py` |
| Vocabulary | `build_text_prompt_vocabulary.py`, `audit_dataset_text_prompts.py` |
| LMDB and storage | `pack_dataset_lmdb.py`, `audit_dataset_storage.py`, `repair_dataset_storage.py` |
| Metadata maintenance | `normalize_dataset_names.py`, `export_native_class_backups.py` |
| Checks | `validate_text_prompt_dataset.py`, `validate_semantic_dataset.py`, `check_dataset_integrity.py` |
| Orthophotos | `download_orthophoto.py` |

Inspect each script's `--help` before using it; downloader and converter arguments differ. Prefer audits and validation before storage repair or conversion.

## Example: scene dataset

```bash
python scripts/data/download_scene_dataset.py resisc45 --output-dir /path/to/RSOD/NWPU-RESISC45
python scripts/data/prepare_scene_dataset.py resisc45 --dataset-root /path/to/RSOD/NWPU-RESISC45 --workers 8
python scripts/data/validate_text_prompt_dataset.py /path/to/RSOD/NWPU-RESISC45 --split val
```

Supported scene keys are `aid`, `patternnet`, and `resisc45`. See source-specific scripts for licenses and download availability.

## Independent semantic test data

Keep manually corrected sets below `/path/to/RSOD/segmentation_test/`: `AID-GT90`, `NWPU-RESISC45-GT135`, `PatternNet-GT114`, `DIOR-R-GT200`, `DOTA-v2-GT160`, `FAIR1M-2.0-GT201`, and `xView-GT100`. Converted outputs follow the semantic dataset contract, regardless of the original supervision type. These 1,000 images are test-only and must not enter training or checkpoint selection.

Large orthophoto inputs belong below a separate `/path/to/RSOD/large_scene/`. Update suite roots/CLI arguments after moving data; do not duplicate imagery into the source repository. Current semantic test suites are under `configs/`. See the [evaluation guide](../README_EVALUATION.md).
