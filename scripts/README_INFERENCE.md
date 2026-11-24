# Inference

**English** | [简体中文](README_INFERENCE.zh-CN.md) | [Project](../README.md)

## Images and text prompts

```bash
python -m ovrs_segmentation.inference.sam3_multilevel \
  --input /path/to/image.tif --prompt airport --prompt forest \
  --checkpoint checkpoints/sam3_multilevel/best.pt \
  --sam3-checkpoint checkpoints/sam3.pt \
  --sam3-input-size 518 --precision bf16 --device cuda \
  --output-dir outputs/sam3_inference --overwrite
```

Repeat `--prompt` for more classes. Use `--classes-json PATH` for a prepared vocabulary, or `--dataset-root PATH --split val --prompt-policy native` for dataset-driven inference. `--prompt-batch-size` controls text batching. Defaults save overlays; add `--no-overlay-only` for the other output products and `--save-probabilities` for probability arrays.

Inference uses an image and text, without GT boxes or annotation-level routing. Class decisions in this entry use `sigmoid(concept_logits) × sigmoid(unified_pixel_logits)`, followed by a class maximum and threshold (default 0.5). This differs from the raw unified-logit semantic benchmark.

Ordinary inference uses non-overlapping 518 × 518 tiles at native scale. Partial edge tiles are padded and only valid pixels stitched. Small images are upsampled with aspect ratio preserved, and predictions restored to original dimensions. This entry does not automatically enable overlapping or multiscale inference.

## Fifty validation images per dataset

```bash
python -m ovrs_segmentation.inference.sam3_multilevel_suite \
  --suite configs/inference_val.yaml --data-root /path/to/RSOD \
  --checkpoint checkpoints/sam3_multilevel/best.pt \
  --sam3-checkpoint checkpoints/sam3.pt \
  --samples-per-dataset 50 --seed 42 --threshold 0.5 \
  --output-dir outputs/sam3_multilevel_val50
```

This suite covers 20 validation sets, selects samples reproducibly, and queries every dataset class. It uses the maximum unified activation above 0.5, exports class-ID masks and overlays with legends, and records sample IDs. It is a visualization protocol, not the final manual semantic test.

For native SAM3 with matching output structure, add `--model-type native` and use `--output-dir outputs/sam3_native_val50`. Native suite pixels use raw Semantic logits, without adaptation or the learned Router. A completed adapted summary can be passed with `--reference-summary PATH` to align samples. Keep existing result directories separate; the suite guards against overwriting a previous summary.

## Georeferenced large scenes

`ovrs_segmentation.inference.georaster` and `.georaster_multiscale` provide separate raster workflows. Inspect each module's `--help` for overlap, batch, raster, and scale arguments. Install `.[geo]` for rasterio/pyproj support. Keep raster CRS, transform, and grid metadata with exported results.

The NAIP workflow retains 0.5, 1, 2, and 5 m predictions on the server and fuses each class onto a 1 m grid by **maximum probability across scales**. The 0.5 m input grid is resampled from the selected 0.6 m NAIP source. Probability exports use probabilities in [0,1], with Presence gating where configured; uint8 maps store probability × 255.
