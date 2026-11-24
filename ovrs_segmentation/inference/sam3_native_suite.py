"""Unmodified SAM3 Semantic inference for the shared validation rendering suite."""
from __future__ import annotations

import torch
from torch.nn import functional as F

from ovrs_segmentation.evaluation.multilevel_validation import _autocast
from ovrs_segmentation.modeling import NativeSAM3EvaluationModel
from .tiling import image_tiles, upsample_small_image


def load_native_model(args, device):
    if args.load_from_hf:
        raise ValueError("Native comparison requires a local base checkpoint for provenance")
    checkpoint = args.sam3_checkpoint.resolve(strict=True)
    model = NativeSAM3EvaluationModel.from_pretrained(
        str(checkpoint),
        bpe_path=str(args.sam3_bpe_path) if args.sam3_bpe_path else None,
        device=device,
        sam3_input_size=args.sam3_input_size or 518,
        prompt_chunk_size=args.prompt_chunk_size,
    ).eval()
    # Match the tuple interface used by the shared image tiling helpers.
    model.sam3_input_size = (model.sam3_input_size, model.sam3_input_size)
    return model, checkpoint


@torch.inference_mode()
def forward_native_tile(model, batch, device, precision):
    """Decode raw Semantic and Presence outputs with explicit prompt/image ownership."""
    images = batch["images"].to(device)
    if tuple(images.shape[-2:]) != model.sam3_input_size:
        raise ValueError("Native tile input must match the configured input size")
    prompts = batch["prompt_texts"]
    owners = torch.as_tensor(batch["prompt_image_indices"], device=device, dtype=torch.long)
    if owners.shape != (len(prompts),) or torch.any((owners < 0) | (owners >= len(images))):
        raise ValueError("Every native prompt must reference a valid image")
    chunk_size = batch.get("prompt_chunk_size", model.prompt_chunk_size)
    pixels, presence = [], []
    with _autocast(device, precision):
        backbone = model.backbone.forward_image((images - model.image_mean) / model.image_std)
        for start in range(0, len(prompts), chunk_size):
            text = prompts[start:start + chunk_size]
            native = model._decode_grounding(backbone, model.backbone.forward_text(text, device=device),
                                            owners[start:start + len(text)])
            semantic = model._native_semantic_logits(native, len(text))
            pixels.append(F.interpolate(semantic[:, None].float(), size=model.sam3_input_size,
                                        mode="bilinear", align_corners=False)[:, 0])
            presence.append(model._presence_logits(native, len(text)).float())
    return {"unified_pixel_logits": torch.cat(pixels), "concept_logits": torch.cat(presence)}


@torch.inference_mode()
def forward_native(model, batch, device, precision):
    images = batch["images"]
    original = images.shape[-2:]
    images = upsample_small_image(images, model.sam3_input_size)
    height, width = images.shape[-2:]
    stitched = None
    for top, left, valid_h, valid_w, tile in image_tiles(images, model.sam3_input_size):
        normalized = (tile.to(device) - model.image_mean) / model.image_std
        chunks = []
        with _autocast(device, precision):
            backbone = model.backbone.forward_image(normalized)
            for start in range(0, len(batch["prompt_texts"]), model.prompt_chunk_size):
                prompts = batch["prompt_texts"][start:start + model.prompt_chunk_size]
                text = model.backbone.forward_text(prompts, device=device)
                owners = torch.zeros(len(prompts), device=device, dtype=torch.long)
                native = model._decode_grounding(backbone, text, owners)
                semantic = model._native_semantic_logits(native, len(prompts))
                chunks.append(F.interpolate(semantic[:, None].float(), size=model.sam3_input_size,
                                            mode="bilinear", align_corners=False)[:, 0])
        pixels = torch.cat(chunks)
        if stitched is None:
            stitched = pixels.new_empty((len(pixels), height, width))
        stitched[:, top:top + valid_h, left:left + valid_w] = pixels[:, :valid_h, :valid_w]
    if (height, width) != original:
        stitched = F.interpolate(stitched[:, None], size=original, mode="bilinear", align_corners=False)[:, 0]
    return {"unified_pixel_logits": stitched}
