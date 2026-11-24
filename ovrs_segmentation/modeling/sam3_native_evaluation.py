from __future__ import annotations

from typing import Any, Sequence

import torch
from torch import nn

from .sam3_utils import configure_sam3_vision_input_size, ensure_sam3_importable


class NativeSAM3EvaluationModel(nn.Module):
    """Thin evaluation-only adapter around an unmodified official SAM3 model.

    This class adds only the interface required by the repository evaluators.
    It registers no LoRA parametrizations and freezes every official weight.
    """

    def __init__(
        self,
        native_model: nn.Module,
        *,
        sam3_input_size: int = 518,
        prompt_chunk_size: int = 8,
        image_mean: Sequence[float] = (0.5, 0.5, 0.5),
        image_std: Sequence[float] = (0.5, 0.5, 0.5),
    ) -> None:
        super().__init__()
        if sam3_input_size < 14 or sam3_input_size % 14:
            raise ValueError("sam3_input_size must be a positive multiple of 14")
        if prompt_chunk_size < 1:
            raise ValueError("prompt_chunk_size must be positive")
        if getattr(native_model, "segmentation_head", None) is None:
            raise ValueError("Official SAM3 must be built with enable_segmentation=True")
        self.native_model = native_model
        self.sam3_input_size = int(sam3_input_size)
        self.prompt_chunk_size = int(prompt_chunk_size)
        self.register_buffer(
            "image_mean",
            torch.tensor(image_mean, dtype=torch.float32)[None, :, None, None],
        )
        self.register_buffer(
            "image_std",
            torch.tensor(image_std, dtype=torch.float32)[None, :, None, None],
        )
        self.native_model.requires_grad_(False)
        self.native_model.eval()
        self.vision_grid = configure_sam3_vision_input_size(
            self.backbone, (self.sam3_input_size, self.sam3_input_size)
        )

    @property
    def backbone(self) -> nn.Module:
        return self.native_model.backbone

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: str | None = None,
        *,
        bpe_path: str | None = None,
        load_from_hf: bool = False,
        device: str | torch.device = "cpu",
        **kwargs: Any,
    ) -> "NativeSAM3EvaluationModel":
        if bool(checkpoint_path) == bool(load_from_hf):
            raise ValueError("Provide exactly one of checkpoint_path or load_from_hf=True")
        ensure_sam3_importable()
        try:
            from sam3.model_builder import build_sam3_image_model
        except ImportError as error:
            raise ImportError(
                "Official SAM3 is not importable. Install Meta's sam3 package "
                "in the Python environment used for evaluation."
            ) from error
        builder_kwargs: dict[str, Any] = {
            "checkpoint_path": checkpoint_path,
            "load_from_HF": load_from_hf,
            "device": str(device),
            "eval_mode": True,
            "enable_segmentation": True,
        }
        if bpe_path:
            builder_kwargs["bpe_path"] = bpe_path
        native_model = build_sam3_image_model(**builder_kwargs)
        native_model.eval()
        return cls(native_model, **kwargs).to(device)

    def train(self, mode: bool = True) -> "NativeSAM3EvaluationModel":
        # This adapter is deliberately inference-only, even if a caller invokes train().
        super().train(False)
        self.native_model.eval()
        return self

    @staticmethod
    def _presence_logits(
        output: dict[str, torch.Tensor], count: int
    ) -> torch.Tensor:
        value = output.get("presence_logit_dec")
        if value is None:
            return output["pred_logits"].new_zeros(count)
        if isinstance(value, (list, tuple)):
            if not value:
                return output["pred_logits"].new_zeros(count)
            value = value[-1]
        if not isinstance(value, torch.Tensor):
            raise ValueError("Native presence_logit_dec is not a tensor")
        while value.ndim > 1 and value.shape[-1] == 1:
            value = value.squeeze(-1)
        if value.numel() == count:
            return value.reshape(count)
        prompt_axes = [axis for axis, size in enumerate(value.shape) if size == count]
        if not prompt_axes:
            raise ValueError(f"Unexpected native presence shape: {tuple(value.shape)}")
        return value.movedim(max(prompt_axes), -1).reshape(-1, count)[-1]

    @staticmethod
    def _native_semantic_logits(
        native: dict[str, torch.Tensor], count: int
    ) -> torch.Tensor:
        value = native.get("semantic_seg")
        if not isinstance(value, torch.Tensor):
            raise KeyError(
                "Native SAM3 returned no semantic_seg; build with "
                "enable_segmentation=True"
            )
        if value.ndim == 4 and value.shape[1] == 1:
            value = value[:, 0]
        if value.ndim != 3 or value.shape[0] != count:
            raise ValueError(f"Unexpected semantic_seg shape: {tuple(value.shape)}")
        return value

    def _prompt_outputs(
        self, native: dict[str, torch.Tensor], count: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        query_scores = native["pred_logits"]
        if query_scores.ndim == 3 and query_scores.shape[-1] == 1:
            query_scores = query_scores[..., 0]
        return query_scores, self._presence_logits(native, count)

    def _decode_grounding(
        self,
        backbone_out: dict[str, Any],
        text_outputs: dict[str, torch.Tensor],
        sample_indices: torch.Tensor,
        *,
        decoder_grad: bool = False,
    ) -> dict[str, torch.Tensor]:
        if decoder_grad:
            raise ValueError("NativeSAM3EvaluationModel does not support gradients")
        try:
            from sam3.model.data_misc import FindStage
        except ImportError as error:
            raise ImportError("Official SAM3 FindStage is unavailable") from error
        encoded = dict(backbone_out)
        encoded.update(text_outputs)
        count = int(sample_indices.numel())
        find_input = FindStage(
            img_ids=sample_indices,
            text_ids=torch.arange(count, device=sample_indices.device, dtype=torch.long),
            input_boxes=None,
            input_boxes_mask=None,
            input_boxes_label=None,
            input_points=None,
            input_points_mask=None,
        )
        geometric_prompt = self.native_model._get_dummy_prompt(num_prompts=count)
        with torch.inference_mode():
            return self.native_model.forward_grounding(
                backbone_out=encoded,
                find_input=find_input,
                find_target=None,
                geometric_prompt=geometric_prompt,
            )
