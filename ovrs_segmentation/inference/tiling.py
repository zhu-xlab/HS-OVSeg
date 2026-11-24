"""Native-resolution windows shared by validation and inference."""
from torch.nn import functional as F
import math


def minimum_size(size, minimum):
    scale = max(1.0, minimum[0] / size[0], minimum[1] / size[1])
    return max(minimum[0], math.ceil(size[0] * scale)), max(minimum[1], math.ceil(size[1] * scale))


def upsample_small_image(images, minimum):
    size = minimum_size(images.shape[-2:], minimum)
    if tuple(images.shape[-2:]) == size:
        return images
    return F.interpolate(images, size=size, mode="bilinear", align_corners=False, antialias=True)


def image_tiles(images, size, *, stride=None):
    tile_height, tile_width = size
    height, width = images.shape[-2:]
    stride_height, stride_width = size if stride is None else stride
    if not (0 < stride_height <= tile_height and 0 < stride_width <= tile_width):
        raise ValueError("Tile stride must be positive and no larger than the window")
    # Keep a regular grid and pad the last window; do not add windows after
    # the preceding window already covers the end of an axis.
    rows = range(0, max(0, height - tile_height) + stride_height, stride_height)
    columns = range(0, max(0, width - tile_width) + stride_width, stride_width)
    for top in rows:
        for left in columns:
            tile = images[..., top:top + tile_height, left:left + tile_width]
            valid_height, valid_width = tile.shape[-2:]
            tile = F.pad(tile, (0, tile_width - valid_width, 0, tile_height - valid_height), value=0.5)
            yield top, left, valid_height, valid_width, tile
