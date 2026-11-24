from __future__ import annotations

from collections.abc import Iterable, Sequence
from contextlib import contextmanager
import math
from types import MethodType
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import parametrize

from .sam3_utils import configure_sam3_vision_input_size, ensure_sam3_importable
from .unified_pixel import (
    normalized_scene_pseudo_targets,
    normalize_object_semantic,
)


def _trainable_vit_mlp_forward(module: nn.Module, x: torch.Tensor) -> torch.Tensor:
    if not torch.is_grad_enabled():
        return module._ovrs_inference_forward(x)
    x = module.act(module.fc1(x))
    x = module.drop1(x)
    x = module.norm(x)
    x = module.fc2(x)
    return module.drop2(x)


def _enable_trainable_vit_mlp(native_model: nn.Module) -> None:
    """Use standard autograd for SAM3 ViT MLPs and fused math for inference."""
    from sam3.model.vitdet import Mlp

    for module in native_model.modules():
        if isinstance(module, Mlp):
            object.__setattr__(module, "_ovrs_inference_forward", module.forward)
            object.__setattr__(
                module,
                "forward",
                MethodType(_trainable_vit_mlp_forward, module),
            )


class LowRankWeightDelta(nn.Module):
    """LoRA parametrization that can be disabled without merging weights."""

    def __init__(
        self,
        out_features: int,
        in_features: int,
        *,
        rank: int = 8,
        scale: float = 1.0,
        weight_shape: Sequence[int] | None = None,
    ) -> None:
        super().__init__()
        if rank < 1:
            raise ValueError("LoRA rank must be positive")
        if scale <= 0:
            raise ValueError("LoRA scale must be positive")
        self.rank = min(int(rank), int(out_features), int(in_features))
        self.scale = float(scale)
        self.enabled = True
        self.weight_shape = tuple(weight_shape or (out_features, in_features))
        if math.prod(self.weight_shape[1:]) != int(in_features):
            raise ValueError(
                "LoRA flattened input size does not match the weight shape"
            )
        self.down = nn.Parameter(torch.empty(self.rank, in_features))
        self.up = nn.Parameter(torch.zeros(out_features, self.rank))
        nn.init.kaiming_uniform_(self.down, a=math.sqrt(5))

    def forward(self, original: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return original
        delta = self.up @ self.down
        delta = delta.reshape(self.weight_shape)
        return original + self.scale * delta.to(original.dtype)


class SharedSAM3LoRA(nn.Module):
    """Run SAM3's native decoder while adapting every matrix/kernel weight."""

    def __init__(
        self,
        native_decoder: nn.Module,
        *,
        native_model: nn.Module,
        rank: int = 8,
        scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.native_decoder = native_decoder
        self.rank = int(rank)
        self.scale = float(scale)
        object.__setattr__(self, "_weight_deltas", [])
        object.__setattr__(self, "_target_names", [])
        self._inject_all_weights(native_model)

    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)
        except AttributeError:
            native_decoder = super().__getattr__("native_decoder")
            return getattr(native_decoder, name)

    def _register(
        self,
        module: nn.Module,
        parameter_name: str,
        target_name: str,
    ) -> None:
        weight = getattr(module, parameter_name, None)
        if not isinstance(weight, nn.Parameter) or weight.ndim < 2:
            raise ValueError(f"{target_name} is not a matrix weight")
        delta = LowRankWeightDelta(
            weight.shape[0],
            math.prod(weight.shape[1:]),
            rank=self.rank,
            scale=self.scale,
            weight_shape=weight.shape,
        ).to(device=weight.device)
        parametrize.register_parametrization(module, parameter_name, delta)
        self._weight_deltas.append(delta)
        self._target_names.append(target_name)

    def _inject_all_weights(self, native_model: nn.Module) -> None:
        """Apply LoRA to every native weight with at least two dimensions."""
        seen: set[int] = set()
        modules = list(native_model.named_modules())
        for module_name, module in modules:
            for parameter_name, weight in list(module.named_parameters(recurse=False)):
                if (
                    not parameter_name.endswith("weight")
                    or weight.ndim < 2
                    or id(weight) in seen
                ):
                    continue
                seen.add(id(weight))
                prefix = f"{module_name}." if module_name else ""
                self._register(
                    module,
                    parameter_name,
                    f"{prefix}{parameter_name}",
                )

    @property
    def target_names(self) -> tuple[str, ...]:
        return tuple(self._target_names)

    @property
    def target_count(self) -> int:
        return len(self._target_names)

    def adapter_parameters(self) -> Iterable[nn.Parameter]:
        for delta in self._weight_deltas:
            yield from delta.parameters()

    def set_enabled(self, enabled: bool) -> None:
        for delta in self._weight_deltas:
            delta.enabled = bool(enabled)

    @property
    def enabled(self) -> bool:
        return self._weight_deltas[0].enabled

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.native_decoder(*args, **kwargs)


class SAM3NativeMultilevelModel(nn.Module):
    """Frozen SAM3 plus shared LoRA for Scene, LULC, and Object tasks."""

    def __init__(
        self,
        native_model: nn.Module,
        *,
        sam3_input_size: int | tuple[int, int] = 518,
        lora_rank: int = 8,
        lora_scale: float = 1.0,
        scene_pseudo_lower_quantile: float = 0.02,
        scene_pseudo_upper_quantile: float = 0.98,
        scene_pseudo_threshold: float = 0.5,
        object_teacher_chunk_size: int = 8,
        prompt_chunk_size: int = 4,
        image_mean: Sequence[float] = (0.5, 0.5, 0.5),
        image_std: Sequence[float] = (0.5, 0.5, 0.5),
    ) -> None:
        super().__init__()
        if getattr(native_model, "segmentation_head", None) is None:
            raise ValueError("Native SAM3 must be built with enable_segmentation=True")
        transformer = getattr(native_model, "transformer", None)
        decoder = getattr(transformer, "decoder", None)
        if not isinstance(decoder, nn.Module):
            raise ValueError(
                "Native SAM3 transformer does not expose its official decoder"
            )
        self.native_model = native_model
        lora = SharedSAM3LoRA(
            decoder,
            native_model=native_model,
            rank=lora_rank,
            scale=lora_scale,
        )
        transformer.decoder = lora
        self.lora_rank = int(lora_rank)
        self.lora_scale = float(lora_scale)
        if not 0.0 <= scene_pseudo_lower_quantile < scene_pseudo_upper_quantile <= 1.0:
            raise ValueError(
                "Scene pseudo quantiles must satisfy 0 <= lower < upper <= 1"
            )
        if not 0.0 <= scene_pseudo_threshold <= 1.0:
            raise ValueError("scene_pseudo_threshold must be between zero and one")
        self.scene_pseudo_lower_quantile = float(scene_pseudo_lower_quantile)
        self.scene_pseudo_upper_quantile = float(scene_pseudo_upper_quantile)
        self.scene_pseudo_threshold = float(scene_pseudo_threshold)
        if object_teacher_chunk_size < 1:
            raise ValueError("object_teacher_chunk_size must be positive")
        self.object_teacher_chunk_size = int(object_teacher_chunk_size)
        if prompt_chunk_size < 1:
            raise ValueError("prompt_chunk_size must be positive")
        self.prompt_chunk_size = int(prompt_chunk_size)
        if isinstance(sam3_input_size, int):
            sam3_input_size = (sam3_input_size, sam3_input_size)
        self.sam3_input_size = tuple(map(int, sam3_input_size))
        if len(self.sam3_input_size) != 2 or min(self.sam3_input_size) < 1:
            raise ValueError("sam3_input_size must contain two positive dimensions")
        self.register_buffer(
            "image_mean",
            torch.tensor(image_mean, dtype=torch.float32)[None, :, None, None],
        )
        self.register_buffer(
            "image_std",
            torch.tensor(image_std, dtype=torch.float32)[None, :, None, None],
        )
        from .unified_pixel import PixelwiseSemanticQueryFusion
        self.pixel_router = PixelwiseSemanticQueryFusion()
        self.configure_trainable()

    @property
    def backbone(self) -> nn.Module:
        return self.native_model.backbone

    @property
    def lora(self) -> SharedSAM3LoRA:
        return self.native_model.transformer.decoder

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: str | None = None,
        *,
        bpe_path: str | None = None,
        load_from_hf: bool = False,
        device: str | torch.device = "cpu",
        **kwargs: Any,
    ) -> "SAM3NativeMultilevelModel":
        if bool(checkpoint_path) == load_from_hf:
            raise ValueError("Use exactly one of checkpoint_path and load_from_hf")
        ensure_sam3_importable()
        from sam3.model_builder import build_sam3_image_model

        builder_kwargs: dict[str, Any] = {
            "checkpoint_path": checkpoint_path,
            "load_from_HF": load_from_hf,
            "device": str(device),
            "eval_mode": False,
            "enable_segmentation": True,
        }
        if bpe_path:
            builder_kwargs["bpe_path"] = bpe_path
        native_model = build_sam3_image_model(**builder_kwargs)
        _enable_trainable_vit_mlp(native_model)
        native_model.eval()
        model = cls(native_model, **kwargs)
        model.vision_grid = configure_sam3_vision_input_size(
            model.backbone, model.sam3_input_size
        )
        return model.to(device)

    def configure_trainable(self) -> dict[str, int]:
        """Freeze native SAM3 and enable every LoRA parameter."""
        self.native_model.requires_grad_(False)
        lora_parameters = list(self.lora.adapter_parameters())
        router_parameters = list(self.pixel_router.parameters())
        trainable_parameters = lora_parameters + router_parameters
        for parameter in trainable_parameters:
            parameter.requires_grad_(True)
        return {
            "trainable": sum(p.numel() for p in trainable_parameters),
            "total": sum(p.numel() for p in self.parameters()),
            "router_parameters": sum(p.numel() for p in router_parameters),
            "lora_targets": self.lora.target_count,
        }

    def parameter_groups(self) -> dict[str, list[nn.Parameter]]:
        return {
            "trainable": [p for p in self.lora.adapter_parameters() if p.requires_grad]
            + [p for p in self.pixel_router.parameters() if p.requires_grad]
        }

    def _trainable_named_parameters(self) -> dict[str, nn.Parameter]:
        adapter_ids = {id(parameter) for parameter in self.lora.adapter_parameters()}
        parameters = {
            name: parameter
            for name, parameter in self.named_parameters()
            if id(parameter) in adapter_ids
        }
        parameters.update({f"pixel_router.{name}": parameter
                           for name, parameter in self.pixel_router.named_parameters()})
        return parameters

    @staticmethod
    def _clone_parameters(
        parameters: dict[str, nn.Parameter]
    ) -> dict[str, torch.Tensor]:
        return {
            name: parameter.detach().cpu().clone()
            for name, parameter in parameters.items()
        }

    @staticmethod
    def _load_parameters(
        expected: dict[str, nn.Parameter],
        state: dict[str, torch.Tensor],
        *,
        label: str,
    ) -> None:
        missing = sorted(set(expected) - set(state))
        unexpected = sorted(set(state) - set(expected))
        if missing or unexpected:
            raise RuntimeError(
                f"{label} state mismatch: missing={missing[:10]}, "
                f"unexpected={unexpected[:10]}"
            )
        with torch.no_grad():
            for name, parameter in expected.items():
                value = state[name]
                if (
                    not isinstance(value, torch.Tensor)
                    or value.shape != parameter.shape
                ):
                    raise ValueError(f"{label} shape/type mismatch for {name}")
                parameter.copy_(
                    value.to(device=parameter.device, dtype=parameter.dtype)
                )

    def trainable_state_dict(self) -> dict[str, torch.Tensor]:
        return self._clone_parameters(self._trainable_named_parameters())

    def load_trainable_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        self._load_parameters(
            self._trainable_named_parameters(),
            state,
            label="trainable model",
        )

    def set_lora_enabled(self, enabled: bool) -> None:
        self.lora.set_enabled(enabled)

    def train(self, mode: bool = True) -> "SAM3NativeMultilevelModel":
        super().train(mode)
        self.native_model.eval()
        return self

    @staticmethod
    def _presence_logits(output: dict[str, torch.Tensor], count: int) -> torch.Tensor:
        value = output["presence_logit_dec"]
        if isinstance(value, (list, tuple)):
            if not value:
                raise ValueError("Native presence_logit_dec is empty")
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
        prompt_axis = max(prompt_axes)
        rows = value.movedim(prompt_axis, -1).reshape(-1, count)
        return rows[-1]

    def _student_outputs(
        self,
        backbone_out: dict[str, Any],
        prompts: Sequence[str],
        sample_indices: torch.Tensor,
        output_size: tuple[int, int],
    ) -> dict[str, torch.Tensor]:
        """Decode prompts while reusing the image features."""
        chunks: dict[str, list[torch.Tensor]] = {
            "concept_logits": [],
            "semantic_logits": [],
            "query_logits": [],
            "query_mask_logits": [],
            "unified_pixel_logits": [],
        }
        decoder_grad = self.training and any(
            parameter.requires_grad for parameter in self.parameters()
        )
        chunk_size = self.prompt_chunk_size if not self.training else len(prompts)
        for start in range(0, len(prompts), chunk_size):
            stop = min(start + chunk_size, len(prompts))
            text_outputs = self.backbone.forward_text(
                prompts[start:stop], device=sample_indices.device
            )
            native = self._decode_grounding(
                backbone_out,
                text_outputs,
                sample_indices[start:stop],
                decoder_grad=decoder_grad,
            )
            semantic_logits = self._resize_logits(
                self._native_semantic_logits(native, stop - start), output_size
            )
            query_scores, presence_logits = self._prompt_outputs(
                native, stop - start
            )
            values = {
                **self.pixel_router(semantic_logits, query_scores, native["pred_masks"],
                                    output_size=output_size),
                "concept_logits": presence_logits.float().reshape(-1),
                "semantic_logits": semantic_logits.float(),
                "query_logits": query_scores,
                "query_mask_logits": native["pred_masks"],
            }
            for name in chunks:
                chunks[name].append(values[name])
        return {name: torch.cat(values) for name, values in chunks.items()}

    def _decode_grounding(
        self,
        backbone_out: dict[str, Any],
        text_outputs: dict[str, torch.Tensor],
        sample_indices: torch.Tensor,
        *,
        decoder_grad: bool,
        geometric_prompt: Any | None = None,
    ) -> dict[str, torch.Tensor]:
        from sam3.model.data_misc import FindStage

        encoded = dict(backbone_out)
        encoded.update(text_outputs)
        count = int(sample_indices.numel())
        find_input = FindStage(
            img_ids=sample_indices,
            text_ids=torch.arange(
                count, device=sample_indices.device, dtype=torch.long
            ),
            input_boxes=None,
            input_boxes_mask=None,
            input_boxes_label=None,
            input_points=None,
            input_points_mask=None,
        )
        with torch.set_grad_enabled(decoder_grad):
            return self.native_model.forward_grounding(
                backbone_out=encoded,
                find_input=find_input,
                find_target=None,
                geometric_prompt=(
                    geometric_prompt
                    if geometric_prompt is not None
                    else self.native_model._get_dummy_prompt(num_prompts=count)
                ),
            )

    def _prompt_outputs(
        self, native: dict[str, torch.Tensor], count: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        query_scores = native["pred_logits"]
        if query_scores.ndim == 3 and query_scores.shape[-1] == 1:
            query_scores = query_scores[..., 0]
        presence_logits = self._presence_logits(native, count)
        return query_scores, presence_logits

    @staticmethod
    def _native_semantic_logits(
        native: dict[str, torch.Tensor], count: int
    ) -> torch.Tensor:
        value = native.get("semantic_seg")
        if not isinstance(value, torch.Tensor):
            raise KeyError("SAM3 did not return semantic_seg")
        if value.ndim == 4 and value.shape[1] == 1:
            value = value[:, 0]
        if value.ndim != 3 or value.shape[0] != count:
            raise ValueError(f"Unexpected semantic_seg shape: {tuple(value.shape)}")
        return value

    @staticmethod
    def _resize_logits(logits: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        if logits.shape[-2:] == size:
            return logits
        return F.interpolate(
            logits[:, None], size=size, mode="bilinear", align_corners=False
        )[:, 0]

    @staticmethod
    def _boxes_xyxy(boxes: torch.Tensor) -> torch.Tensor:
        cx, cy, width, height = boxes.unbind(-1)
        return torch.stack(
            (cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2),
            dim=-1,
        )

    @contextmanager
    def _native_teacher(self):
        enabled = self.lora.enabled
        self.set_lora_enabled(False)
        try:
            with torch.no_grad():
                yield
        finally:
            self.set_lora_enabled(enabled)

    def _scene_training_output(
        self,
        batch: dict[str, Any],
        teacher: dict[str, torch.Tensor],
        semantic_logits: torch.Tensor,
        sample_indices: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        teacher_logits = self._native_semantic_logits(
            teacher, len(batch["prompt_texts"])
        ).float()
        teacher_logits = self._resize_logits(teacher_logits, semantic_logits.shape[-2:])
        activation = teacher_logits.sigmoid()
        valid_mask = batch["valid_mask"].to(device=activation.device, dtype=torch.bool)
        pseudo_targets = normalized_scene_pseudo_targets(
            activation,
            batch["scene_targets"].to(activation.device).eq(1),
            valid_mask,
            sample_indices=sample_indices,
            lower_quantile=self.scene_pseudo_lower_quantile,
            upper_quantile=self.scene_pseudo_upper_quantile,
            threshold=self.scene_pseudo_threshold,
        )
        return {"scene_pseudo_targets": pseudo_targets}

    @staticmethod
    def _box_iou(boxes: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        top_left = torch.maximum(boxes[..., :2], target[:2])
        bottom_right = torch.minimum(boxes[..., 2:], target[2:])
        intersection = (bottom_right - top_left).clamp_min(0).prod(-1)
        box_area = (boxes[..., 2:] - boxes[..., :2]).clamp_min(0).prod(-1)
        target_area = (target[2:] - target[:2]).clamp_min(0).prod()
        return intersection / (box_area + target_area - intersection).clamp_min(1e-6)

    def _object_teacher(
        self,
        batch: dict[str, Any],
        images: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """Run frozen SAM3 once per GT box with matching class text + box prompt."""
        prompts, owners, parent_rows, boxes = [], [], [], []
        sample_indices = batch["annotation_sample_indices"].to(images.device)
        for row, row_boxes in enumerate(batch["boxes"]):
            for box in row_boxes:
                prompts.append(batch["prompt_texts"][row])
                owners.append(int(sample_indices[row]))
                parent_rows.append(row)
                boxes.append(box)
        if not boxes:
            empty = torch.empty(0, device=sample_indices.device, dtype=torch.long)
            return {}, empty

        device = sample_indices.device
        owners_tensor = torch.tensor(owners, device=device, dtype=torch.long)
        parent_tensor = torch.tensor(parent_rows, device=device, dtype=torch.long)
        box_tensor = torch.stack(boxes).to(device=device, dtype=torch.float32)
        semantic_masks, query_masks = [], []
        with self._native_teacher():
            backbone_out = self.backbone.forward_image(images)
            for start in range(0, len(boxes), self.object_teacher_chunk_size):
                stop = min(start + self.object_teacher_chunk_size, len(boxes))
                chunk_boxes = box_tensor[start:stop]
                text_outputs = self.backbone.forward_text(
                    prompts[start:stop], device=device
                )
                geometric_prompt = self.native_model._get_dummy_prompt(
                    num_prompts=stop - start
                )
                geometric_prompt.append_boxes(
                    chunk_boxes[None],
                    torch.ones(1, stop - start, device=device, dtype=torch.bool),
                )
                output = self._decode_grounding(
                    backbone_out,
                    text_outputs,
                    owners_tensor[start:stop],
                    decoder_grad=False,
                    geometric_prompt=geometric_prompt,
                )
                output_boxes = self._boxes_xyxy(output["pred_boxes"])
                output_scores = output["pred_logits"].float().squeeze(-1).sigmoid()
                output_masks = output["pred_masks"]
                selected = []
                for index, box in enumerate(chunk_boxes):
                    target = self._boxes_xyxy(box[None])[0]
                    quality = self._box_iou(output_boxes[index], target)
                    query = int((quality + output_scores[index]).argmax())
                    selected.append(output_masks[index, query])
                query_masks.append(torch.stack(selected))
                semantic_masks.append(
                    self._native_semantic_logits(output, stop - start)
                )
        teacher = {
            "semantic_seg": torch.cat(semantic_masks),
            "selected_masks": torch.cat(query_masks),
        }
        return teacher, parent_tensor

    @staticmethod
    def _ignore_box_regions(
        valid_mask: torch.Tensor, ignored_boxes: Sequence[torch.Tensor]
    ) -> torch.Tensor:
        valid_mask = valid_mask.clone()
        height, width = valid_mask.shape[-2:]
        for row, boxes in enumerate(ignored_boxes):
            for center_x, center_y, box_width, box_height in boxes.tolist():
                left = max(0, math.floor((center_x - box_width / 2) * width + 1e-6))
                top = max(0, math.floor((center_y - box_height / 2) * height + 1e-6))
                right = min(width, math.ceil((center_x + box_width / 2) * width - 1e-6))
                bottom = min(height, math.ceil((center_y + box_height / 2) * height - 1e-6))
                valid_mask[row, top:bottom, left:right] = False
        return valid_mask

    def _object_training_output(
        self,
        batch: dict[str, Any],
        teacher: dict[str, torch.Tensor],
        parent_rows: torch.Tensor,
        semantic_logits: torch.Tensor,
        output_size: tuple[int, int],
    ) -> dict[str, Any]:
        valid_mask = self._ignore_box_regions(
            batch["valid_mask"].to(device=semantic_logits.device, dtype=torch.bool),
            batch["ignored_boxes"],
        )
        result: dict[str, Any] = {
            "object_unified_targets": torch.zeros_like(semantic_logits).detach(),
            "object_unified_valid_mask": valid_mask,
        }
        if not teacher:
            return result

        teacher_selected = teacher["selected_masks"].sigmoid()
        positive_rows = parent_rows.unique(sorted=True)
        semantic_targets = []
        teacher_semantic = teacher["semantic_seg"]
        for row in positive_rows.tolist():
            row_targets = []
            row_semantic = teacher_semantic[parent_rows.eq(row)]
            for start in range(0, len(row_semantic), self.object_teacher_chunk_size):
                resized = F.interpolate(
                    row_semantic[
                        start : start + self.object_teacher_chunk_size, None
                    ].float(),
                    size=output_size,
                    mode="bilinear",
                    align_corners=False,
                )[:, 0].sigmoid()
                row_targets.append(resized.amax(0))
            semantic_targets.append(torch.stack(row_targets).amax(0))
        semantic_targets = torch.stack(semantic_targets)
        semantic_targets = normalize_object_semantic(
            semantic_targets, valid_mask.index_select(0, positive_rows),
            lower_quantile=self.scene_pseudo_lower_quantile,
            upper_quantile=self.scene_pseudo_upper_quantile,
        )
        unified_targets = torch.zeros_like(semantic_logits)
        for target_index, row in enumerate(positive_rows.tolist()):
            query_target = teacher_selected[parent_rows.eq(row)].amax(0).float()
            if query_target.shape != output_size:
                query_target = F.interpolate(
                    query_target[None, None],
                    size=output_size,
                    mode="bilinear",
                    align_corners=False,
                )[0, 0]
            semantic_target = semantic_targets[target_index]
            unified_targets[row] = 1.0 - (1.0 - semantic_target) * (1.0 - query_target)
        result.update(
            {
                "object_unified_targets": unified_targets.detach(),
            }
        )
        return result

    def forward(self, batch: dict[str, Any]) -> dict[str, Any]:
        images = batch["images"]
        prompts = batch["prompt_texts"]
        level = batch["task_type"] if self.training else None
        sample_indices = batch["annotation_sample_indices"].to(images.device)
        if images.shape[-2:] != self.sam3_input_size:
            images = F.interpolate(
                images, size=self.sam3_input_size, mode="bilinear", align_corners=False
            )
        images = (images - self.image_mean) / self.image_std
        output_size = tuple(batch["pixel_targets"].shape[-2:])
        backbone_grad = self.training and any(
            parameter.requires_grad for parameter in self.backbone.parameters()
        )
        with torch.set_grad_enabled(backbone_grad):
            backbone_out = self.backbone.forward_image(images)

        teacher = None
        parent_rows = None
        if self.training and level == "scene":
            with self._native_teacher():
                teacher_backbone = self.backbone.forward_image(images)
                teacher_text = self.backbone.forward_text(prompts, device=images.device)
                teacher = self._decode_grounding(
                    teacher_backbone,
                    teacher_text,
                    sample_indices,
                    decoder_grad=False,
                )
        elif self.training and level == "object":
            teacher, parent_rows = self._object_teacher(batch, images)
        shared = self._student_outputs(
            backbone_out, prompts, sample_indices, output_size
        )
        semantic_logits = shared["semantic_logits"]
        if self.training:
            if level == "object":
                assert parent_rows is not None
                output = self._object_training_output(
                    batch,
                    teacher,
                    parent_rows,
                    semantic_logits,
                    output_size,
                )
                output.update(shared)
                return output
            if level == "lulc":
                return shared
            assert teacher is not None
            output = self._scene_training_output(
                batch, teacher, semantic_logits, sample_indices
            )
            output.update(shared)
            return output
        return shared
