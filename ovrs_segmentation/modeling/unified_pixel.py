"""One level-agnostic pixel output shared by training and inference."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def score_weighted_query_probabilities(query_mask_logits, query_logits, *,
                                       output_size=None, top_k=8, epsilon=1e-5):
    """Pool class- and stability-weighted query masks into a concept map."""
    if query_mask_logits.ndim != 4:
        raise ValueError("Expected P x Q x H x W query masks")
    confidence = quality_weighted_query_scores(query_mask_logits, query_logits)
    count = min(max(int(top_k), 1), confidence.shape[1])
    probabilities, indices = confidence.topk(count, dim=1)
    gather = indices[:, :, None, None].expand(-1, -1, *query_mask_logits.shape[-2:])
    masks = query_mask_logits.gather(1, gather).float().sigmoid()
    combined = (probabilities[:, :, None, None] * masks).amax(dim=1)
    if output_size is not None and combined.shape[-2:] != output_size:
        combined = F.interpolate(combined[:, None], size=output_size,
                                 mode="bilinear", align_corners=False)[:, 0]
    return combined.clamp(epsilon, 1.0 - epsilon)


class PixelwiseSemanticQueryFusion(nn.Module):
    """Normpixelhead: calibrated logits with a learned local 5 x 5 correction."""

    def __init__(self, query_top_k=8):
        super().__init__()
        self.query_top_k = int(query_top_k)
        self.log_temperatures = nn.Parameter(torch.zeros(2))
        self.biases = nn.Parameter(torch.zeros(2))
        self.router = nn.Sequential(nn.Conv2d(2, 16, 3, padding=1), nn.GELU(),
                                    nn.Conv2d(16, 1, 3, padding=1))
        nn.init.zeros_(self.router[-1].weight)
        nn.init.zeros_(self.router[-1].bias)

    def forward(self, semantic_logits, query_logits, query_mask_logits, *,
                output_size=None, detach_evidence=False):
        # The main objective backpropagates through fusion into both branches.
        if output_size is None:
            output_size = tuple(semantic_logits.shape[-2:])
        semantic = semantic_logits.float()
        if semantic.shape[-2:] != output_size:
            semantic = F.interpolate(semantic[:, None], size=output_size,
                                     mode="bilinear", align_corners=False)[:, 0]
        temperatures = self.log_temperatures.exp().clamp(0.05, 20.0)
        semantic = semantic / temperatures[0] + self.biases[0]
        query = score_weighted_query_probabilities(query_mask_logits, query_logits,
                    output_size=output_size, top_k=self.query_top_k)
        query = torch.logit(query) / temperatures[1] + self.biases[1]
        evidence = torch.stack((semantic, query), dim=1)
        fused = (semantic + query) * 0.5 + self.router(evidence)[:, 0]
        return {"unified_pixel_logits": fused}


def quality_weighted_query_scores(
    query_mask_logits: torch.Tensor,
    query_logits: torch.Tensor,
    *,
    stability_offset: float = 0.2,
) -> torch.Tensor:
    """Class confidence times SAM-style mask stability for every query."""
    scores = query_logits
    while scores.ndim > 2 and scores.shape[-1] == 1:
        scores = scores.squeeze(-1)
    if scores.ndim != 2 or scores.shape[:2] != query_mask_logits.shape[:2]:
        raise ValueError("Query scores and masks must have matching P x Q axes")
    masks = query_mask_logits
    inner = masks.gt(float(stability_offset)).flatten(2).sum(2).float()
    outer = masks.gt(-float(stability_offset)).flatten(2).sum(2).float()
    stability = torch.where(outer > 0, inner / outer.clamp_min(1), outer)
    return scores.float().sigmoid() * stability


def normalize_object_semantic(probabilities, valid_mask, *, lower_quantile=.02, upper_quantile=.98):
    """Normalize frozen Object Semantic probabilities per image/concept."""
    result = probabilities.detach().float().clone()
    for row in range(len(result)):
        valid = valid_mask[row].bool()
        if valid.any():
            lower, upper = torch.quantile(result[row][valid], result.new_tensor([lower_quantile, upper_quantile]))
            result[row] = ((result[row] - lower) / (upper - lower).clamp_min(1e-6)).clamp(0, 1)
        else:
            result[row].zero_()
    return result


def normalized_scene_pseudo_targets(
    activation_probabilities: torch.Tensor,
    positive_rows: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    sample_indices: torch.Tensor | None = None,
    lower_quantile: float = 0.02,
    upper_quantile: float = 0.98,
    threshold: float = 0.5,
    ignore_id: int = 255,
) -> torch.Tensor:
    """Build binary Scene pseudo masks from frozen native SAM3 activations.

    Positive GT rows supervise their complete pseudo mask. Other text prompts
    for the same image receive negative supervision inside the GT-positive
    region and remain ignored elsewhere.
    """
    if activation_probabilities.ndim != 3:
        raise ValueError("activation_probabilities must be P x H x W")
    if positive_rows.shape != activation_probabilities.shape[:1]:
        raise ValueError("positive_rows must contain one flag per prompt")
    if valid_mask.shape != activation_probabilities.shape:
        raise ValueError("valid_mask must match activation_probabilities")
    if not 0.0 <= lower_quantile < upper_quantile <= 1.0:
        raise ValueError("Scene pseudo quantiles must satisfy 0 <= lower < upper <= 1")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("Scene pseudo threshold must be between zero and one")
    targets = torch.full(
        activation_probabilities.shape,
        int(ignore_id),
        dtype=torch.uint8,
        device=activation_probabilities.device,
    )
    valid_mask = valid_mask.to(device=targets.device, dtype=torch.bool)
    positive_rows = positive_rows.to(device=targets.device, dtype=torch.bool)
    if sample_indices is None:
        sample_indices = torch.arange(len(targets), device=targets.device)
    else:
        sample_indices = sample_indices.to(device=targets.device, dtype=torch.long)
        if sample_indices.shape != positive_rows.shape:
            raise ValueError("sample_indices must contain one image owner per prompt")
    probabilities = activation_probabilities.detach().float()
    for index in positive_rows.nonzero(as_tuple=False).flatten().tolist():
        valid = valid_mask[index]
        values = probabilities[index][valid]
        if values.numel() < 2:
            continue
        lower = torch.quantile(values, lower_quantile)
        upper = torch.quantile(values, upper_quantile)
        if not bool(torch.isfinite(lower) and torch.isfinite(upper)):
            continue
        span = upper - lower
        if float(span) <= 1e-8:
            continue
        normalized = ((probabilities[index] - lower) / span).clamp_(0.0, 1.0)
        targets[index][valid] = normalized[valid].ge(threshold).to(torch.uint8)

    # A wrong text prompt is a spatial negative exactly where a GT text prompt
    # for the same image says the object/scene evidence lies. Outside this
    # region there is no reliable negative pixel label, so keep ignore_id.
    for sample_index in sample_indices.unique().tolist():
        owned = sample_indices.eq(sample_index)
        positive_owned = owned & positive_rows
        if not bool(positive_owned.any()):
            continue
        gt_region = targets[positive_owned].eq(1).any(dim=0)
        for index in (
            (owned & ~positive_rows).nonzero(as_tuple=False).flatten().tolist()
        ):
            negative_region = gt_region & valid_mask[index]
            targets[index][negative_region] = 0
    return targets
