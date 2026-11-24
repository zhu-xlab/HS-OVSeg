from __future__ import annotations

from importlib.util import find_spec
from pathlib import Path
import sys
from typing import Any

import torch
from torch import nn


def ensure_sam3_importable() -> None:
    """Use installed SAM3, or expose this repository's vendored checkout."""
    if find_spec("sam3") is not None:
        return
    source = Path(__file__).resolve().parents[2] / "third_party" / "sam3"
    if not (source / "sam3").is_dir():
        raise ModuleNotFoundError(
            "Official SAM3 is unavailable. Expected its source at "
            f"{source}. Run `python tools/install_sam3_local.py` from this project "
            "or install SAM3 in the active Python environment."
        )
    sys.path.insert(0, str(source))


def configure_sam3_vision_input_size(
    backbone: nn.Module, input_size: tuple[int, int]
) -> dict[str, Any]:
    """Rebuild global-attention RoPE buffers for a non-default image grid."""
    trunk = backbone.vision_backbone.trunk
    stride = trunk.patch_embed.proj.stride
    stride = (stride, stride) if isinstance(stride, int) else tuple(map(int, stride))
    if any(size % step for size, step in zip(input_size, stride)):
        raise ValueError(
            f"SAM3 input {input_size} must be divisible by patch stride {stride}"
        )
    grid_size = tuple(size // step for size, step in zip(input_size, stride))
    rebuilt = 0
    for block in trunk.blocks:
        attention = block.attn
        if int(getattr(block, "window_size", 0)) or not getattr(
            attention, "use_rope", False
        ):
            continue
        device = next(attention.parameters()).device
        attention.input_size = grid_size
        attention._setup_rope_freqs()
        for name in ("freqs_cis", "freqs_cis_real", "freqs_cis_imag"):
            value = getattr(attention, name, None)
            if torch.is_tensor(value):
                setattr(attention, name, value.to(device))
        rebuilt += 1
    if not rebuilt:
        raise ValueError("No global SAM3 RoPE attention block was found")
    return {
        "input_size": input_size,
        "patch_stride": stride,
        "grid_size": grid_size,
        "rebuilt_global_rope_blocks": rebuilt,
    }
