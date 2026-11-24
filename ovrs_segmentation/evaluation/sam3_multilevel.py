#!/usr/bin/env python
"""Evaluate a trained SAM3 multilevel LoRA checkpoint on one dataset level."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from ovrs_segmentation.modeling import SAM3NativeMultilevelModel
from .sam3_common import (
    add_evaluation_args,
    close_distributed,
    distributed_context,
    resolve_file,
    run_evaluation,
    validate_evaluation_args,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_evaluation_args(parser, input_size_default=None)
    parser.add_argument("--checkpoint", type=Path, required=True)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--sam3-checkpoint", type=Path)
    source.add_argument("--load-from-hf", action="store_true")
    parser.add_argument("--lora-rank", type=int)
    parser.add_argument("--lora-scale", type=float)
    return parser.parse_args()


def _checkpoint_settings(
    args: argparse.Namespace, state: dict[str, Any]
) -> dict[str, Any]:
    saved = state["args"]
    load_from_hf = bool(args.load_from_hf or saved.get("load_from_hf", False))
    base = args.sam3_checkpoint or saved.get("sam3_checkpoint")
    bpe = args.sam3_bpe_path or saved.get("sam3_bpe_path")
    if args.load_from_hf:
        base = None
        if args.sam3_bpe_path is None:
            bpe = None
    return {
        "base_checkpoint": None if base is None else Path(base),
        "bpe_path": None if bpe is None else Path(bpe),
        "load_from_hf": load_from_hf,
        "sam3_input_size": int(args.sam3_input_size or saved.get("sam3_input_size", 518)),
        "lora_rank": int(args.lora_rank or saved["lora_rank"]),
        "lora_scale": float(args.lora_scale or saved["lora_scale"]),
    }


def load_adapted_model(
    args: argparse.Namespace, device: torch.device
) -> tuple[SAM3NativeMultilevelModel, dict[str, Any], dict[str, Any], Path]:
    checkpoint = resolve_file(args.checkpoint, "--checkpoint")
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)

    settings = _checkpoint_settings(args, state)
    base = resolve_file(settings["base_checkpoint"], "--sam3-checkpoint")
    if base is None and not settings["load_from_hf"]:
        raise ValueError(
            "Pass --sam3-checkpoint; it is not stored in the training checkpoint"
        )
    bpe = resolve_file(settings["bpe_path"], "--sam3-bpe-path")
    model = SAM3NativeMultilevelModel.from_pretrained(
        None if base is None else str(base),
        bpe_path=None if bpe is None else str(bpe),
        load_from_hf=settings["load_from_hf"],
        device=device,
        sam3_input_size=settings["sam3_input_size"],
        lora_rank=settings["lora_rank"],
        lora_scale=settings["lora_scale"],
    )
    model.load_trainable_state_dict(state["model"])
    model.requires_grad_(False).eval()
    model.prompt_chunk_size = args.prompt_chunk_size
    return model, state, settings, checkpoint


def default_output(root: Path, split: str, level: str) -> Path:
    return Path("outputs/sam3_multilevel_eval") / f"{root.name}_{split}_{level}.json"


def main() -> None:
    args = parse_args()
    rank, world_size, device = distributed_context(args.device)
    try:
        model, state, settings, checkpoint = load_adapted_model(args, device)
        input_size = settings["sam3_input_size"]
        validate_evaluation_args(args, input_size)
        run_evaluation(
            args,
            model,
            rank=rank,
            world_size=world_size,
            device=device,
            input_size=input_size,
            output=args.output_json
            or default_output(args.dataset_root, args.split, args.level),
            model_fields={
                "model": "sam3_multilevel_adapted",
                "adapter_or_finetuned_weights_loaded": True,
                "trained_checkpoint": str(checkpoint),
                "checkpoint_epoch": int(state["epoch"]) + 1,
                "checkpoint_global_step": int(state["global_step"]),
                "trainable_scope": "lora_presence_semantic",
                "lora_rank": settings["lora_rank"],
                "lora_scale": settings["lora_scale"],
            },
        )
    finally:
        close_distributed()


if __name__ == "__main__":
    main()
