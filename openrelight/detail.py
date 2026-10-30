"""DetailLock: keep the original's identity and detail, take only the light from the generation.

In log-linear space an image is roughly  log(albedo/detail) + log(lighting),
and lighting is the low-frequency part. So:

    out = exp( logO - low(logO) + low(logG) )

High frequencies (pores, eyes, hair, fabric) come from the original; the light's
direction, colour and falloff come from the generated video. The low-pass is a
guided filter steered by the original, so light boundaries snap to the original's
edges instead of haloing. Composite-mode lights are added afterwards as a linear layer.
"""

import torch
import torch.nn.functional as F

from .shading import linear_to_srgb, luminance, rolloff, srgb_to_linear


def _box(x, r):
    """Mean filter over a (2r+1)^2 window, x: (N, C, H, W)."""
    k = 2 * r + 1
    return F.avg_pool2d(F.pad(x, (r, r, r, r), mode="replicate"), k, stride=1)


def guided_filter(guide, src, r, eps):
    """He et al. guided filter. guide: (N, 1, H, W), src: (N, C, H, W)."""
    mean_i = _box(guide, r)
    mean_p = _box(src, r)
    cov_ip = _box(guide * src, r) - mean_i * mean_p
    var_i = _box(guide * guide, r) - mean_i * mean_i
    a = cov_ip / (var_i + eps)
    b = mean_p - a * mean_i
    return _box(a, r) * guide + _box(b, r)


def _low(log_img, guide, r, eps, scale=4):
    """Guided filter at reduced resolution (fast, big radii), upsampled back."""
    N, C, H, W = log_img.shape
    if scale > 1 and min(H, W) // scale >= 32:
        h, w = H // scale, W // scale
        g = F.interpolate(guide, size=(h, w), mode="area")
        s = F.interpolate(log_img, size=(h, w), mode="area")
        out = guided_filter(g, s, max(1, r // scale), eps)
        return F.interpolate(out, size=(H, W), mode="bilinear", align_corners=False)
    return guided_filter(guide, log_img, r, eps)


def _resample_frames(x, count):
    if x.shape[0] == count:
        return x
    idx = torch.linspace(0, x.shape[0] - 1, count).round().long()
    return x[idx]


def detail_lock(original, generated, mask=None, layer=None, radius=0.03, strength=0.85,
                highlights=0.5, layer_strength=1.0, eps=0.02, chunk=8, device=None, progress=None):
    """original: (T, H, W, 3) sRGB. generated: (T', h, w, 3) sRGB, any size/length.
    mask: optional (T, H, W) where the lock applies (outside it the generation is kept).
    layer: optional linear additive light layer (T'', h'', w'', 3) from LightPass."""
    device = device or original.device
    T, H, W, _ = original.shape
    gen = _resample_frames(generated, T)
    if layer is not None:
        layer = _resample_frames(layer, T)
    if mask is not None:
        mask = _resample_frames(mask, T)
    r = max(2, int(radius * max(H, W)))
    out = []
    for s in range(0, T, chunk):
        o = original[s:s + chunk].to(device).float().permute(0, 3, 1, 2)
        g = gen[s:s + chunk].to(device).float().permute(0, 3, 1, 2)
        if g.shape[-2:] != (H, W):
            g = F.interpolate(g, size=(H, W), mode="bicubic", align_corners=False, antialias=True).clamp(0, 1)
        lo, lg = srgb_to_linear(o), srgb_to_linear(g)
        log_o, log_g = torch.log(lo + 1e-3), torch.log(lg + 1e-3)
        guide = torch.log(luminance(lo.permute(0, 2, 3, 1)) + 1e-3).unsqueeze(1)
        low_o = _low(log_o, guide, r, eps)
        low_g = _low(log_g, guide, r, eps)
        locked = log_o - low_o + low_g

        if highlights > 0:
            # specular sparkle the generation added that the original doesn't have
            hi_g = log_g - low_g
            hi_o = log_o - low_o
            bright = (luminance(lg.permute(0, 2, 3, 1)) > 0.25).float().unsqueeze(1)
            locked = locked + highlights * (hi_g - hi_o).clamp_min(0) * bright

        res = torch.exp(locked) - 1e-3
        if mask is not None:
            m = mask[s:s + chunk].to(device).float().unsqueeze(1)
            if m.shape[-2:] != (H, W):
                m = F.interpolate(m, size=(H, W), mode="bilinear", align_corners=False)
            m = m * strength
        else:
            m = torch.full_like(res[:, :1], strength)
        res = lg * (1 - m) + res * m

        if layer is not None and layer_strength > 0:
            ly = layer[s:s + chunk].to(device).float().permute(0, 3, 1, 2)
            if ly.shape[-2:] != (H, W):
                ly = F.interpolate(ly, size=(H, W), mode="bilinear", align_corners=False)
            res = res + ly * layer_strength

        out.append(linear_to_srgb(rolloff(res, lo)).clamp(0, 1).permute(0, 2, 3, 1).cpu())
        if progress:
            progress(min(s + chunk, T), T)
    return torch.cat(out)
