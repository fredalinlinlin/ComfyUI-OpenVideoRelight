"""Fit a clip to a video model's size and frame grid, and put the model's output back on the clip.

A centre crop would zoom the picture and shift it a few pixels against the original, which shows as ghosting, so the
clip is scaled and padded instead: the picture keeps its full view, and unpad() lines the result up pixel for pixel.
"""

import math

import torch
import torch.nn.functional as F

FRAME_GRIDS = {"keep": None, "17k+5 (MiniMax H3)": (17, 5)}


def grid_length(n, grid):
    """Smallest frame count >= n on a video model's grid (k*step + offset); None keeps n."""
    if not grid:
        return n
    step, offset = grid
    return offset + max(0, math.ceil((n - offset) / step)) * step


def fit_pad(images, megapixels, multiple=32, frames="keep"):
    """(T, H, W, C) -> (padded (N, th, tw, 3), tw, th, N, pad_info): about `megapixels` (never scaled up), padded with
    edge pixels to `multiple`, with the last frame repeated up to the model's frame grid (MiniMax H3 takes 17k+5)."""
    count, H, W, _ = images.shape
    target = min(1.0, (megapixels * 1e6 / (W * H)) ** 0.5) if megapixels > 0 else 1.0

    def fit_box(side):   # the first multiple above the scaled side
        return max(multiple, (int(side * target) // multiple + 1) * multiple)

    def pad_to(side):
        return math.ceil(side / multiple) * multiple

    scale = min(1.0, fit_box(W) / W, fit_box(H) / H)
    w, h = max(multiple, round(W * scale)), max(multiple, round(H * scale))
    tw, th = pad_to(w), pad_to(h)
    picture = images[..., :3].movedim(-1, 1)
    if (h, w) != (H, W):
        picture = F.interpolate(picture, size=(h, w), mode="bilinear", align_corners=False, antialias=True)
    left, top = (tw - w) // 2, (th - h) // 2
    padded = F.pad(picture, (left, tw - w - left, top, th - h - top), mode="replicate").movedim(1, -1)
    length = grid_length(count, FRAME_GRIDS.get(frames))
    if length > count:
        padded = torch.cat([padded, padded[-1:].repeat(length - count, 1, 1, 1)])
    return padded, tw, th, length, {"orig": (count, H, W), "box": (left, top, w, h)}


def unpad(images, pad_info, to_original_size=True):
    """Undo fit_pad on images (N, th, tw, C): remove the padding, back to the clip's frame count and, optionally, size."""
    count, H, W = pad_info["orig"]
    left, top, w, h = pad_info["box"]
    picture = images[:count, top:top + h, left:left + w, :3]
    if picture.shape[0] < count:
        picture = torch.cat([picture, picture[-1:].repeat(count - picture.shape[0], 1, 1, 1)])
    if to_original_size and (h, w) != (H, W):
        picture = F.interpolate(picture.movedim(-1, 1), size=(H, W), mode="bicubic", align_corners=False,
                                antialias=True).movedim(1, -1)
    return picture.clamp(0, 1).float()
