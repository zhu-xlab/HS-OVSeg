# Evaluation

**English** | [简体中文](README_EVALUATION.zh-CN.md) | [Project](../README.md)

Training validation retains each source task's metric. Final semantic testing evaluates pixel masks across every supervision level. Keep these reports separate.

| Stage | Scene | Object | Semantic |
|---|---|---|---|
| Validation / best checkpoint | Full-vocabulary Top-1 | mAP50 | mIoU |
| Semantic test | Pixel metrics | Pixel metrics | Pixel metrics |

## Semantic test

```bash
python -m ovrs_segmentation.evaluation.segmentation_test \
  --suite configs/segmentation_test_manual.yaml \
  --data-root /path/to/RSOD --test-root /path/to/RSOD/segmentation_test \
  --checkpoint checkpoints/sam3_multilevel/best.pt \
  --sam3-checkpoint checkpoints/sam3.pt \
  --output-json outputs/sam3_multilevel/semantic_test.json
```

Use `last.pt` to test the latest state. For one prepared semantic dataset, replace `--suite` with `--dataset-root /path/to/DATASET`. Add `--preflight` to the same command to validate datasets without loading model weights; checkpoint arguments remain required by the parser.

| Suite | Scope |
|---|---|
| `configs/segmentation_test_manual.yaml` | AID 90, NWPU-RESISC45 135, PatternNet 114, DIOR-R 200, DOTA-v2 160, FAIR1M-2.0 201, xView 100 |
| `configs/segmentation_test.yaml` | All 20 source datasets with semantic test targets |

The seven manually corrected sets provide 1,000 test images below `segmentation_test`, with directories named `DATASET-GTCOUNT`. The remaining semantic datasets use their prepared test splits. Missing datasets produce an incomplete summary and exit code 2; report that coverage explicitly.

The model queries all eligible dataset classes. Predictions use `unified_pixel_logits` argmax; where a dataset defines background, negative winning logits map to background. Evaluation preserves original pixel dimensions and excludes ignore ID 255. Reports include mIoU, mean Dice, pixel accuracy, valid pixels, and per-class values; background is excluded from mean IoU/Dice when defined.

Use `--max-samples N` only for limited checks and identify such reports as partial. `--precision bf16` is the default; `--prompt-chunk-size` controls prompt memory use.

## Source-task validation

`ovrs_segmentation.evaluation.sam3_multilevel` and `.sam3_native` retain source-task validation protocols; they are different from the semantic test entry above. Record prediction rules, vocabulary, foundation weights, and preprocessing with each evaluation.
