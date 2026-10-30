"""Face detail lock: faces keep the original's features, only the light comes from the generated clip.

The video model redraws faces (at 0.5-0.9 MP a face is only 60-120 px), so eyes, mouth and skin
change. Inside a mask (or the whole frame) the generated clip, aligned to the original by optical flow,
is rebuilt at the original's resolution, two ways:

guided (default): a guided filter with the original's luminance as the guide. In every small window
the result is a*original + b fitted to the generated clip, so it takes the original's edges and
features where the two agree, and the generated light, shadows and colour everywhere. Where the new
light changed the local contrast (a face edge against a background that became brighter) the fit
follows the generated clip, so no outline appears.

frequency: out = low(generated) + (original - low(original)) * gain. Sharper, but the detail layer
also carries the original's own small-scale lighting (hard shadow edges, a bright step along a face
against a dark background), which the gain amplifies into outlines and harsh shadows (2026-10-28).
"""

import torch
import torch.nn.functional as F

from .detail import _box, guided_filter
from .shading import linear_to_srgb, luminance, srgb_to_linear


def gaussian_blur(x, sigma):
    """x (N, C, H, W); separable gaussian, reflect padding."""
    if sigma <= 0:
        return x
    r = max(1, int(sigma * 3 + 0.5))
    k = torch.arange(-r, r + 1, device=x.device, dtype=x.dtype)
    k = torch.exp(-0.5 * (k / sigma) ** 2)
    k = k / k.sum()
    C = x.shape[1]
    x = F.pad(x, (r, r, 0, 0), mode="reflect")
    x = F.conv2d(x, k.view(1, 1, 1, -1).expand(C, 1, 1, -1).contiguous(), groups=C)
    x = F.pad(x, (0, 0, r, r), mode="reflect")
    return F.conv2d(x, k.view(1, 1, -1, 1).expand(C, 1, -1, 1).contiguous(), groups=C)


def _resize(x, H, W):
    """(T, h, w, C) -> (T, H, W, C)."""
    if x.shape[1:3] == (H, W):
        return x
    return F.interpolate(x.permute(0, 3, 1, 2), size=(H, W), mode="bicubic", align_corners=False,
                         antialias=True).permute(0, 2, 3, 1).clamp(0, 1)


def soften_mask(mask, grow, feather):
    """mask (T, H, W) -> grown by `grow` px and feathered by `feather` px (gaussian sigma)."""
    m = mask.float()[:, None]
    if grow > 0:
        k = int(grow) * 2 + 1
        m = F.max_pool2d(m, k, 1, k // 2)
    return gaussian_blur(m, feather)[:, 0].clamp(0, 1)


def edge_agreement(a, b, r, lo=0.2, hi=0.7):
    """Per-pixel 0..1: how much the local structure of a and b (N, 1, H, W) agrees, from their
    correlation over (2r+1)^2 windows (lo -> 0, hi -> 1), smoothed so the switch has no seam."""
    ma, mb = _box(a, r), _box(b, r)
    cov = _box(a * b, r) - ma * mb
    va = (_box(a * a, r) - ma * ma).clamp_min(0)
    vb = (_box(b * b, r) - mb * mb).clamp_min(0)
    rho = cov / (va.sqrt() * vb.sqrt() + 1e-4)
    return _box(((rho - lo) / (hi - lo)).clamp(0, 1), max(1, r // 2))


def face_detail_lock(original, generated, mask, detail_px=3.0, max_gain=3.0, grow=6, feather=4.0,
                     chunk=8, device=None, method="guided", radius=8, eps=3e-4):
    """original (T, H, W, 3) sRGB at full resolution, generated (T, h, w, 3) sRGB at any resolution,
    mask (T, H', W') faces (any resolution). Returns (T, H, W, 3) sRGB at the original's resolution:
    the generated clip upscaled, with original detail inside the faces."""
    device = device or original.device
    T = min(original.shape[0], generated.shape[0])
    H, W = original.shape[1:3]
    out = []
    for s in range(0, T, chunk):
        e = min(T, s + chunk)
        o = original[s:e, ..., :3].to(device).float()
        g = _resize(generated[s:e, ..., :3].to(device).float(), H, W)
        m = mask[min(s, mask.shape[0] - 1):min(e, mask.shape[0])].to(device).float()
        if m.shape[0] < e - s:
            m = torch.cat([m, m[-1:].expand(e - s - m.shape[0], -1, -1)], 0)
        if m.shape[1:3] != (H, W):
            m = F.interpolate(m[:, None], size=(H, W), mode="bilinear", align_corners=False)[:, 0]
        m = soften_mask(m, grow, feather)
        if m.max() <= 0:
            out.append(g.cpu())
            continue
        if method == "guided":
            guide = luminance(o).unsqueeze(1)
            p = g.permute(0, 3, 1, 2)
            q = guided_filter(guide, p, radius, eps)
            # where the original and the generated clip change in opposite directions (a head against a
            # window the new light turned from bright to dark), the fit would paste the original's edge on
            # inverted, as a cut-out outline: there the generated clip is kept. Faces, where the two agree,
            # keep the original's detail.
            w = edge_agreement(guide, luminance(g).unsqueeze(1), radius)
            locked = (p + (q - p) * w).permute(0, 2, 3, 1).clamp(0, 1)
            out.append((g + (locked - g) * m.unsqueeze(-1)).cpu())
            continue
        ol, gl = srgb_to_linear(o), srgb_to_linear(g)
        o_low = gaussian_blur(ol.permute(0, 3, 1, 2), detail_px).permute(0, 2, 3, 1)
        g_low = gaussian_blur(gl.permute(0, 3, 1, 2), detail_px).permute(0, 2, 3, 1)
        # how much the new light brightened this area: one factor for all channels, clamped so dark
        # noise is never blown up more than max_gain
        gain = (luminance(g_low) / luminance(o_low).clamp_min(2e-3)).clamp(1.0 / max_gain, max_gain)
        rebuilt = (g_low + (ol - o_low) * gain.unsqueeze(-1)).clamp_min(0)
        # anti-halo: stay within the generated clip's own local range (plus a margin for the original's
        # finer detail). A bright small feature (a cap badge) brightened by `gain` otherwise overshoots
        # into a glow; ordinary skin and fabric detail sits well inside the range.
        k = 2 * max(1, round(2 * detail_px)) + 1
        g4 = gl.permute(0, 3, 1, 2)
        hi = F.max_pool2d(g4, k, 1, k // 2).permute(0, 2, 3, 1)
        lo = -F.max_pool2d(-g4, k, 1, k // 2).permute(0, 2, 3, 1)
        rebuilt = torch.minimum(torch.maximum(rebuilt, lo * 0.7), hi * 1.3 + 0.004)
        locked = linear_to_srgb(rebuilt).clamp(0, 1)
        out.append((g + (locked - g) * m.unsqueeze(-1)).cpu())
    return torch.cat(out, 0)
