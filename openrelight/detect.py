"""Detect the clip's own light and turn it into "original" lights on the light stage.

A first-order spherical-harmonic fit of luminance against MoGe's normals gives the dominant light
direction (shading ~ a0 + a.N). It is a starting point the user checks, not a measurement:
  - only the foreground is used (nearest part of the scene), so a large flat backdrop does not drown
    the subject's shading;
  - colour comes from how the bright areas differ from the whole frame (the light's tint), which
    removes much of the surface colour (skin, clothing) a plain average of bright pixels would carry;
  - a weak directional fit means even light, and no light is made. Softness is not estimated (starts at 0.5).
Known weak cases: several lights of similar strength, strong backlight, very dark or very bright
surfaces, coloured light, and stylised (non-photographic) shading.
"""

import copy

import torch

from .geometry import sample_quantile
from .rig import kelvin_to_rgb, load_rig
from .shading import detect_dominant_light, fit_sh1, luminance, srgb_to_linear

_KELVINS = list(range(1800, 12001, 100))
_KELVIN_RGB = torch.tensor([kelvin_to_rgb(k) for k in _KELVINS])


def _kelvin_of(rgb):
    """Nearest blackbody colour temperature to an RGB tint (chromaticity only)."""
    v = torch.as_tensor(rgb, dtype=torch.float32)
    v = v / v.sum().clamp_min(1e-6)
    table = _KELVIN_RGB / _KELVIN_RGB.sum(-1, keepdim=True)
    return _KELVINS[int((table - v).abs().sum(-1).argmin())]


def detect_original_lights(images, geo, samples=8, min_strength=0.15):
    """images (T, H, W, 3) sRGB at the geometry's frames. Returns (rig dict, report text)."""
    dev = geo["normal"].device
    T = min(images.shape[0], geo["normal"].shape[0])
    H, W = geo["normal"].shape[1:3]
    coeffs, tints, strengths = [], [], []
    for t in sorted({round(i * (T - 1) / max(samples - 1, 1)) for i in range(samples)}):
        img = images[t].to(dev).float()
        if img.shape[:2] != (H, W):
            img = torch.nn.functional.interpolate(img.permute(2, 0, 1)[None], size=(H, W), mode="area")[0].permute(1, 2, 0)
        lin = srgb_to_linear(img.clamp(0, 1))
        valid = geo["valid"][t] & (geo["alpha"][t] > 0.5)
        depth = geo["depth"][t]
        if valid.sum() < 256:
            continue
        # foreground: the nearer 60 % of the valid points (a close-up's face, not the backdrop behind it)
        near = valid & (depth <= sample_quantile(depth[valid], 0.6))
        region = near if near.sum() > 256 else valid
        c = fit_sh1(lin, geo["normal"][t], region)
        coeffs.append(c)
        lum = luminance(lin)
        strengths.append(float(c[1:].norm() / lum[region].mean().clamp_min(1e-4)))
        bright = region & (lum >= sample_quantile(lum[region], 0.9))
        if bright.sum() > 32:
            tints.append(lin[bright].mean(0) / lin[region].mean(0).clamp_min(1e-4))
    if not coeffs:
        return load_rig({"lights": []}), "Not enough depth information to work out the original light."
    az, el, _ = detect_dominant_light(torch.stack(coeffs))
    # how much brightness changes with surface direction, relative to the average: ~0.2 even light, >1 hard key
    strength = sum(strengths) / len(strengths)
    kelvin = _kelvin_of(torch.stack(tints).mean(0).cpu()) if tints else 5600
    if strength < min_strength:
        return load_rig({"lights": [], "detect_strength": strength}), (f"The clip's light is even, with no clear main light (directional strength {strength:.2f}). "
                                                                       "No original light was created. You can add new lights directly.")
    # softness is not told apart reliably (a soft frontal light on a face measured 1.17, a hard side light on
    # a test sphere 1.79): start in the middle and let the user set it
    softness = 0.5
    light = {
        "id": "original0", "name": "Original light", "type": "area", "mode": "generate", "enabled": True, "original": True,
        "color": {"kelvin": kelvin}, "softness": softness, "cone": {"angle": 40.0, "feather": 0.4}, "shadow": True,
        "motion": None, "prompt": "",
        "keys": [{"sec": 0, "azimuth": round(az), "elevation": round(el), "distance": 1.5, "intensity_ev": 0.0,
                  "kelvin": kelvin, "ease": "smooth"}],
    }
    light["detected"] = copy.deepcopy(light)
    rig = load_rig({"lights": [light], "detect_strength": strength})
    side = "right" if az > 15 else "left" if az < -15 else "centre"
    front = "front" if abs(az) < 90 else "back"
    report = (f"Original light detected: azimuth {az:.0f}° ({front} {side}), elevation {el:.0f}°, about {kelvin}K, "
              f"directional strength {strength:.2f} (higher is surer). Softness cannot be read reliably, so it starts at medium. This is a starting point: check it on the light stage.")
    return rig, report


# ---------------------------------------------------------------- lights from a lighting reference image


