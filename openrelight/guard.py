"""Light guard: a generated frame may only be brighter than the original where the rig's light lands.

Image models invent light sources: given "a spotlight beam", Qwen drew a visible cone from the
ceiling lamp in the shot, and H3 copied it. The light map (shading.light_map) says where the rig's
light actually falls, from the scene's geometry. Per pixel, the generated luminance is capped at

    original * (1 + margin + allowance * light_map)

so the new light can brighten what it reaches (the mockup is original * (0.35 + 4 * map), the
allowance leaves room above that) and nothing else can get brighter than the original by more than
`margin`. Darkening is never touched. The cap is applied as a smooth per-pixel scale, so edges of the
cut don't show.
"""

import torch
import torch.nn.functional as F

from .facelock import gaussian_blur
from .shading import linear_to_srgb, luminance, srgb_to_linear


def _resize(x, H, W):
    """(N, h, w, C) -> (N, H, W, C)"""
    if x.shape[1:3] == (H, W):
        return x
    return F.interpolate(x.permute(0, 3, 1, 2), size=(H, W), mode="bilinear", align_corners=False).permute(0, 2, 3, 1)


def light_guard(image, original, light_map, margin=0.15, allowance=12.0, softness=4.0, floor=0.0005, chunk=16,
                device=None):
    """image (N, H, W, 3) sRGB generated, original (N', h, w, 3) sRGB, light_map (N'', h', w', 3) sRGB
    (Light Map node output). All are matched to the image's size and frame count (the last frame repeats).
    Returns (N, H, W, 3) sRGB."""
    device = device or image.device
    N, H, W = image.shape[:3]
    out = []
    for s in range(0, N, chunk):
        e = min(N, s + chunk)
        pick = lambda x: x[[min(i, x.shape[0] - 1) for i in range(s, e)]].to(device).float()  # noqa: E731
        g = pick(image)[..., :3]
        o = _resize(pick(original)[..., :3], H, W)
        m = _resize(pick(light_map)[..., :3], H, W)
        gl, ol = srgb_to_linear(g), srgb_to_linear(o)
        lum_g = luminance(gl)
        cap = luminance(ol) * (1.0 + margin + allowance * luminance(srgb_to_linear(m))) + floor
        scale = (cap / lum_g.clamp_min(1e-5)).clamp(max=1.0)
        # soften: take the smallest scale nearby, then blur, so a capped area fades in instead of cutting
        k = 2 * max(1, int(softness)) + 1
        scale = -F.max_pool2d(-scale[:, None], k, 1, k // 2)
        scale = gaussian_blur(scale, softness)[:, 0].clamp(0, 1)
        out.append(linear_to_srgb(gl * scale.unsqueeze(-1)).clamp(0, 1).cpu())
    return torch.cat(out, 0)
