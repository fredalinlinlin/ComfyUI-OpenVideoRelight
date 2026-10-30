"""A contact sheet of the clip for the light stage editor, which draws the frame at the playhead in the
middle of the sphere (so lights are placed against the picture itself)."""

import math
import os
import uuid

import numpy as np
from PIL import Image

import folder_paths

MAX_TILES = 240     # longer clips show every n-th frame
TILE_H = 135


def clip_sheet(images, fps):
    """images (T, H, W, C) in [0, 1]. Saves one JPEG grid of small frames to the temp folder and returns
    what the editor needs to find a frame in it."""
    T, H, W = images.shape[:3]
    step = max(1, math.ceil(T / MAX_TILES))
    idx = list(range(0, T, step))
    th = min(TILE_H, H)
    tw = max(1, round(W * th / H))
    cols = max(1, min(len(idx), 4096 // tw))
    rows = math.ceil(len(idx) / cols)
    sheet = Image.new("RGB", (cols * tw, rows * th))
    for i, f in enumerate(idx):
        frame = (images[f, ..., :3].float().clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        tile = Image.fromarray(frame).resize((tw, th), Image.BILINEAR)
        sheet.paste(tile, ((i % cols) * tw, (i // cols) * th))
    name = f"openrelight_clip_{uuid.uuid4().hex[:10]}.jpg"
    os.makedirs(folder_paths.get_temp_directory(), exist_ok=True)
    sheet.save(os.path.join(folder_paths.get_temp_directory(), name), quality=82)
    return {"filename": name, "subfolder": "", "type": "temp", "frames": T, "fps": fps, "step": step,
            "count": len(idx), "cols": cols, "tile_w": tw, "tile_h": th, "width": W, "height": H}
