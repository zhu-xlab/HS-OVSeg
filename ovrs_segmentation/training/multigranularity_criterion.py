"""HS-OVSeg: Presence BCE and end-to-end fused pixel BCE/Dice/CE."""
from dataclasses import dataclass
from typing import Any
import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class MultiGranularityLossWeights:
    presence: float = 1.0
    semantic_bce: float = 1.0
    semantic_dice: float = 1.0
    multiclass_ce: float = 1.0


class MultiGranularityCriterion(nn.Module):
    """The Normpixelhead P + G(F) objective at all annotation granularities."""

    def __init__(self, *, weights=None, ignore_id=255):
        super().__init__()
        self.weights = weights or MultiGranularityLossWeights()
        if any(value < 0 for value in vars(self.weights).values()):
            raise ValueError('Loss weights must be non-negative')
        self.ignore_id = int(ignore_id)

    def forward(self, outputs, batch):
        level = batch['task_type']
        logits = outputs['unified_pixel_logits'].float()
        device = logits.device
        if level == 'scene':
            target = outputs['scene_pseudo_targets'].detach().to(device)
            valid = target.ne(self.ignore_id)
            labels = batch['scene_targets'].to(device).float()
        elif level == 'object':
            target = outputs['object_unified_targets'].detach().to(device).ge(.5).long()
            valid = outputs['object_unified_valid_mask'].to(device).bool()
            labels = batch['object_presence_targets'].to(device).float()
        else:
            target = batch['pixel_targets'].to(device)
            valid = target.ne(self.ignore_id)
            labels = None
        valid = valid & batch['valid_mask'].to(device).bool()
        target = target.masked_fill(~valid, self.ignore_id)
        if labels is None:
            labels = (target.eq(1) & valid).flatten(1).any(1).float()
        if logits.shape[-2:] != target.shape[-2:]:
            logits = F.interpolate(logits[:, None], size=target.shape[-2:], mode='bilinear', align_corners=False)[:, 0]
        active = valid.flatten(1).any(1)
        if bool(active.any()):
            bce, dice = _soft_binary_loss(logits[active], target[active].float().clamp(0, 1), valid[active])
        else:
            bce = dice = logits.sum() * 0.
        ce = _pixel_candidate_multiclass_ce(logits, target, batch, ignore_id=self.ignore_id)
        presence_logits = outputs['concept_logits'].float().reshape(-1)
        presence_terms = F.binary_cross_entropy_with_logits(presence_logits, labels.reshape(-1), reduction='none')
        if level == 'lulc':
            presence = (presence_terms * active.float()).sum() / active.float().sum().clamp_min(1)
        else:
            presence = presence_terms.mean()
        total = (self.weights.presence * presence + self.weights.semantic_bce * bce
                 + self.weights.semantic_dice * dice + self.weights.multiclass_ce * ce)
        segmentation = (self.weights.semantic_bce * bce + self.weights.semantic_dice * dice
                        + self.weights.multiclass_ce * ce)
        return {f'loss_{level}_presence': presence, 'loss_unified_pixel_bce': bce,
                'loss_unified_pixel_dice': dice, 'loss_unified_pixel_multiclass_ce': ce,
                'loss_unified_total': segmentation,
                'loss_total': total}

def _soft_binary_loss(
    logits: torch.Tensor, targets: torch.Tensor, valid_mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """BCE and soft Dice for online native-SAM3 activation distillation."""
    if not logits.numel():
        zero = logits.sum() * 0.0
        return zero, zero
    targets = targets.to(device=logits.device, dtype=logits.dtype)
    valid = valid_mask.to(device=logits.device, dtype=logits.dtype)
    count = valid.flatten(1).sum(1).clamp_min(1)
    bce_map = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    bce = ((bce_map * valid).flatten(1).sum(1) / count).mean()
    probabilities = logits.sigmoid() * valid
    targets = targets * valid
    intersection = (probabilities * targets).flatten(1).sum(1)
    denominator = probabilities.flatten(1).sum(1) + targets.flatten(1).sum(1)
    dice = (1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)).mean()
    return bce, dice

def _pixel_candidate_multiclass_ce(
    logits: torch.Tensor,
    pixel_targets: torch.Tensor,
    batch: dict[str, Any],
    *,
    ignore_id: int,
) -> torch.Tensor:
    """CE across prompt maps on conflict-free labeled foreground pixels."""
    owners = batch["annotation_sample_indices"].to(
        device=logits.device, dtype=torch.long
    )
    losses: list[torch.Tensor] = []
    for image_index in owners.unique(sorted=True):
        row_indices = owners.eq(image_index).nonzero(as_tuple=False).flatten()
        class_target = torch.full(
            logits.shape[-2:], ignore_id, device=logits.device, dtype=torch.long
        )
        conflict = torch.zeros_like(class_target, dtype=torch.bool)
        for local_class, row_index in enumerate(row_indices):
            foreground = pixel_targets[row_index].eq(1)
            conflict |= (
                foreground & class_target.ne(ignore_id) & class_target.ne(local_class)
            )
            class_target[foreground & class_target.eq(ignore_id)] = local_class
        class_target[conflict] = ignore_id
        if bool(class_target.ne(ignore_id).any()):
            losses.append(
                F.cross_entropy(
                    logits.index_select(0, row_indices)[None],
                    class_target[None],
                    ignore_index=ignore_id,
                )
            )
    return torch.stack(losses).mean() if losses else logits.sum() * 0.0
