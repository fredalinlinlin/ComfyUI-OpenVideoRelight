"""Green / blue screen: a colour-difference matte, despill and a composite over a background plate.

The relight then sees the clip as it will be shot against the new background (the plate also goes to the LM
lighting director as the "new background", so the subject is lit as that place would light it).
"""
import torch
import torch.nn.functional as F

from .facelock import gaussian_blur


def _cover(plate, H, W):
    """The plate scaled to cover H x W (no stretch) and centre-cropped. plate: (1 or T, h, w, 3)."""
    _, h, w, _ = plate.shape
    s = max(H / h, W / w)
    nh, nw = max(H, round(h * s)), max(W, round(w * s))
    x = F.interpolate(plate.movedim(-1, 1), size=(nh, nw), mode="bilinear", align_corners=False)
    y0, x0 = (nh - H) // 2, (nw - W) // 2
    return x[:, :, y0:y0 + H, x0:x0 + W].movedim(1, -1)


def screen_matte(images, screen="green", low=0.2, high=0.7):
    """Foreground alpha (T, H, W) from how much more of the screen colour a pixel has than the other two
    channels, relative to the screen itself (a high quantile of that difference over the clip)."""
    r, g, b = images[..., 0], images[..., 1], images[..., 2]
    d = g - torch.maximum(r, b) if screen == "green" else b - torch.maximum(r, g)
    flat = d.flatten()
    step = max(1, flat.numel() // 4_000_000)
    ref = torch.quantile(flat[::step].float(), 0.98).clamp_min(0.04)
    t = d / ref                                      # 0 = no screen colour, 1 = the screen
    screen_amount = ((t - low) / max(high - low, 1e-3)).clamp(0, 1)
    return 1.0 - screen_amount


def despill(images, screen="green"):
    """The screen's colour pushed back to the other channels' level (keeps yellows / cyans, kills the cast)."""
    r, g, b = images[..., 0], images[..., 1], images[..., 2]
    if screen == "green":
        g = torch.minimum(g, torch.maximum(r, b))
    else:
        b = torch.minimum(b, torch.maximum(r, g))
    return torch.stack([r, g, b], -1)


def key_composite(images, plate, screen="green", low=0.2, high=0.7, matte_blur=1.0, plate_blur=0.0, chunk=8, erode=1,
                  matte=None):
    """(composite, alpha). images (T, H, W, 3) 0-1; plate (1 or T, h, w, 3), a still plate is held.
    matte: a foreground matte made elsewhere (1 = keep), used instead of the colour key (a hard shot: spill,
    tracking markers, glass)."""
    T, H, W, _ = images.shape
    images = images[..., :3]
    if matte is not None:
        m = matte if matte.dim() == 3 else matte[None]
        if m.shape[-2:] != (H, W):
            m = F.interpolate(m[:, None].float(), size=(H, W), mode="bilinear", align_corners=False)[:, 0]
        alpha = (m if m.shape[0] >= T else m[-1:].expand(T, -1, -1))[:T].float().to(images.device)
    else:
        alpha = shrink(screen_matte(images, screen, low, high), erode)
    bg = _cover(plate[..., :3].float(), H, W)
    if plate_blur > 0:
        bg = gaussian_blur(bg.movedim(-1, 1), plate_blur * min(H, W) / 100.0).movedim(1, -1)
    outs, alphas = [], []
    for i in range(0, T, chunk):
        a = alpha[i:i + chunk]
        if matte_blur > 0:
            a = gaussian_blur(a[:, None], matte_blur)[:, 0]
        fg = despill(images[i:i + chunk], screen)
        b = bg[min(i, bg.shape[0] - 1):min(i, bg.shape[0] - 1) + a.shape[0]] if bg.shape[0] > 1 else bg
        outs.append(fg * a[..., None] + b * (1 - a[..., None]))
        alphas.append(a)
    return torch.cat(outs).clamp(0, 1), torch.cat(alphas)


def shrink(alpha, px=1):
    """The foreground matte pulled in by px pixels (min filter): drops a spill fringe along screen edges."""
    if px <= 0:
        return alpha
    k = 2 * px + 1
    return -F.max_pool2d(-alpha[:, None], k, stride=1, padding=px)[:, 0]


def restore_background(images, alpha, plate, plate_blur=0.0):
    """After the relight, the plate back where the screen was: the generators repaint the background (darker,
    different clouds), the composite keeps the plate the user chose. images (T, H, W, 3); alpha (T or 1, h, w)."""
    T, H, W, _ = images.shape
    if alpha.shape[-2:] != (H, W):
        alpha = F.interpolate(alpha[:, None].float(), size=(H, W), mode="bilinear", align_corners=False)[:, 0]
    if alpha.shape[0] == 1 and T > 1:
        alpha = alpha.expand(T, -1, -1)
    bg = _cover(plate[..., :3].float(), H, W)
    if plate_blur > 0:
        bg = gaussian_blur(bg.movedim(-1, 1), plate_blur * min(H, W) / 100.0).movedim(1, -1)
    a = alpha[:T, ..., None].to(images.device)
    return (images[..., :3] * a + bg.to(images.device) * (1 - a)).clamp(0, 1)
