#!/usr/bin/env python
"""Evaluate unmodified native SAM3 on one prepared dataset split and level."""
from __future__ import annotations

import argparse
from pathlib import Path

from ovrs_segmentation.modeling import NativeSAM3EvaluationModel
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
    add_evaluation_args(parser, input_size_default=518)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--sam3-checkpoint", type=Path)
    source.add_argument("--load-from-hf", action="store_true")
    return parser.parse_args()


def default_output(root: Path, split: str, level: str) -> Path:
    return Path("outputs/sam3_native_eval") / f"{root.name}_{split}_{level}.json"


def main() -> None:
    args = parse_args()
    validate_evaluation_args(args, args.sam3_input_size)
    rank, world_size, device = distributed_context(args.device)
    try:
        checkpoint = resolve_file(args.sam3_checkpoint, "--sam3-checkpoint")
        bpe_path = resolve_file(args.sam3_bpe_path, "--sam3-bpe-path")
        model = NativeSAM3EvaluationModel.from_pretrained(
            None if checkpoint is None else str(checkpoint),
            bpe_path=None if bpe_path is None else str(bpe_path),
            load_from_hf=args.load_from_hf,
            device=device,
            sam3_input_size=args.sam3_input_size,
            prompt_chunk_size=args.prompt_chunk_size,
        ).eval()
        run_evaluation(
            args,
            model,
            rank=rank,
            world_size=world_size,
            device=device,
            input_size=args.sam3_input_size,
            output=args.output_json or default_output(args.dataset_root, args.split, args.level),
            model_fields={
                "model": "official_native_sam3_unmodified",
                "adapter_or_finetuned_weights_loaded": False,
            },
        )
    finally:
        close_distributed()


if __name__ == "__main__":
    main()
