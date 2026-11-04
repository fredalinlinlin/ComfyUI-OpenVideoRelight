"""Reads the original light from MoGe geometry: sRGB to linear light, luminance, an order-1 spherical
harmonics fit of brightness against surface direction, and the dominant light direction from that fit
(used by Detect Original Light with geometry connected)."""

import math

import torch
import torch.nn.functional as F


LUMA = (0.2126, 0.7152, 0.0722)


def srgb_to_linear(x):
    return torch.where(x <= 0.04045, x / 12.92, ((x.clamp_min(0) + 0.055) / 1.055) ** 2.4)


def luminance(x):
    w = torch.tensor(LUMA, device=x.device, dtype=x.dtype)
    return (x * w).sum(-1)


def fit_sh1(lin, normal, region):
    """Least-squares fit of luminance ~ a0 + a.N over region. Returns (4,) coeffs; a = dominant light dir."""
    y = luminance(lin)[region]
    n = normal[region]
    if y.numel() < 64:
        return torch.tensor([y.mean() if y.numel() else 0.5, 0, 0, 0], device=lin.device)
    if y.numel() > 200_000:
        step = y.numel() // 200_000 + 1
        y, n = y[::step], n[::step]
    A = torch.cat([torch.ones_like(y).unsqueeze(-1), n], dim=-1)
    sol = torch.linalg.lstsq(A, y.unsqueeze(-1)).solution.squeeze(-1)
    return sol


def detect_dominant_light(coeffs):
    """Turn averaged SH1 coefficients into an (azimuth, elevation, strength) estimate."""
    a = coeffs[:, 1:].mean(0)
    strength = float(a.norm() / coeffs[:, 0].mean().abs().clamp_min(1e-4))
    d = F.normalize(a, dim=0)
    # camera space -> rig angles (inverse of rig._dir_from_angles)
    el = math.degrees(math.asin(max(-1.0, min(1.0, -float(d[1])))))
    az = math.degrees(math.atan2(float(d[0]), -float(d[2])))
    return az, el, strength


