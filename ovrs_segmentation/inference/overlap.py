"""Weighted label voting for overlapping inference tiles."""
import torch


def overlapping_vote(image, tile_size, stride, classes, predict):
    if not 0 < stride <= tile_size:
        raise ValueError('Tile stride must be between 1 and tile size')
    height, width = image.shape[-2:]
    if min(height, width) < tile_size:
        raise ValueError('Upsample small images before overlapping inference')

    def starts(length):
        return sorted(set(range(0, length - tile_size + 1, stride)) | {length - tile_size})

    axis = torch.hann_window(tile_size, periodic=False).clamp_min(.05)
    weight = axis[:, None] * axis[None, :]
    votes = torch.zeros(classes, height, width)
    for top in starts(height):
        for left in starts(width):
            tile = image[..., top:top + tile_size, left:left + tile_size]
            labels = predict(tile).detach().cpu().long()
            if tuple(labels.shape) != (tile_size, tile_size) or labels.min() < 0 or labels.max() >= classes:
                raise ValueError('Invalid tile prediction shape or class ID')
            votes[:, top:top + tile_size, left:left + tile_size].scatter_add_(
                0, labels[None], weight[None])
    return votes.argmax(0)
