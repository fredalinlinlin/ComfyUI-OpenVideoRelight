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
import math

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

def _hex_of(rgb_lin):
    """Linear RGB tint -> '#rrggbb' at full brightness (the colour of the light, not of the pixel)."""
    v = rgb_lin.clamp_min(0)
    v = v / v.max().clamp_min(1e-6)
    s = torch.where(v <= 0.0031308, v * 12.92, 1.055 * v.clamp_min(1e-8) ** (1 / 2.4) - 0.055)
    return "#" + "".join(f"{int(round(float(c) * 255)):02x}" for c in s.clamp(0, 1))


def _kmeans(x, w, k, iters=20):
    """Weighted k-means on rows of x (N, D). Returns centres (k, D) and labels (N,)."""
    order = torch.argsort(w, descending=True)
    centres = [x[order[0]]]
    for _ in range(1, k):                                   # farthest point start, among the brighter half
        d = torch.stack([(x - c).pow(2).sum(-1) for c in centres]).min(0).values * w
        centres.append(x[int(d.argmax())])
    centres = torch.stack(centres)
    for _ in range(iters):
        lab = torch.cdist(x, centres).argmin(-1)
        for j in range(k):
            m = lab == j
            if m.any():
                centres[j] = (x[m] * w[m, None]).sum(0) / w[m].sum().clamp_min(1e-8)
    return centres, torch.cdist(x, centres).argmin(-1)


def _has_blend(centres, tol=0.025):
    """True when one colour group lies on the line between two others: it is their overlap, not a light."""
    k = centres.shape[0]
    for i in range(k):
        for a in range(k):
            for b in range(a + 1, k):
                if i in (a, b):
                    continue
                ab = centres[b] - centres[a]
                t = ((centres[i] - centres[a]) * ab).sum() / (ab * ab).sum().clamp_min(1e-8)
                if 0 < t < 1 and (centres[a] + t * ab - centres[i]).norm() < tol:
                    return True
    return False


def detect_reference_lights(image, geo, max_lights=3, min_separation=0.07):
    """Lights of a lighting reference image, to put on the light stage.

    image (H, W, 3) sRGB, geo: prepare_geometry of that one image. The lit surfaces are grouped by the colour
    of their light (chromaticity, weighted by brightness): one group = one light. For each group, a first-order
    spherical-harmonic fit of that group's share of the brightness against the normals gives its direction,
    the group's total brightness its level relative to the brightest, and its mean chromaticity its colour
    (a gel colour when saturated, else a colour temperature). How dark the unlit part is sets the ambient.
    A starting point to check, like detect_original_lights: softness is not estimated."""
    dev = geo["normal"].device
    H, W = geo["normal"].shape[1:3]
    img = image.to(dev).float()
    if img.shape[:2] != (H, W):
        img = torch.nn.functional.interpolate(img.permute(2, 0, 1)[None], size=(H, W), mode="area")[0].permute(1, 2, 0)
    lin = srgb_to_linear(img.clamp(0, 1))
    valid = geo["valid"][0] & (geo["alpha"][0] > 0.5)
    if valid.sum() < 256:
        return load_rig({"lights": []}), "The reference image has too little depth information to read its light."
    depth = geo["depth"][0]
    near = valid & (depth <= sample_quantile(depth[valid], 0.85))
    region = near if near.sum() > 256 else valid
    lum = luminance(lin)
    # lit by total RGB, not luminance: luminance would drop a red / magenta light next to a green / cyan one
    tot = lin.sum(-1)
    lit = region & (tot >= sample_quantile(tot[region], 0.4))
    chroma = lin[lit] / lin[lit].sum(-1, keepdim=True).clamp_min(1e-5)
    w = tot[lit]
    # how many lights: split while the colour groups are clearly apart, each holds a real share of the light,
    # and no group is just the blend of two others (where two coloured lights overlap)
    k_best, centres, labels = 1, ((chroma * w[:, None]).sum(0) / w.sum())[None], torch.zeros_like(w, dtype=torch.long)
    for k in range(2, max_lights + 1):
        c, lab = _kmeans(chroma, w, k)
        sep = torch.cdist(c, c)[~torch.eye(k, dtype=torch.bool, device=c.device)].min()
        share = torch.stack([w[lab == j].sum() for j in range(k)]) / w.sum()
        if sep < min_separation or share.min() < 0.08 or _has_blend(c):
            break
        k_best, centres, labels = k, c, lab
    # each light's own colour: the purest pixels of its group (farthest from the other groups), not the group
    # mean, which the overlap zone pulls toward the other light
    colours = []
    for j in range(k_best):
        m = labels == j
        cj, wj = chroma[m], w[m]
        if k_best > 1:
            others = torch.cat([centres[:j], centres[j + 1:]])
            far = torch.cdist(cj, others).min(-1).values
            keep = far >= sample_quantile(far, 0.8)
            cj, wj = cj[keep], wj[keep]
        colours.append((cj * wj[:, None]).sum(0) / wj.sum().clamp_min(1e-8))
    # unmix: every pixel = sum_j s_j * colour_j (least squares, s >= 0), so s_j is light j's shading alone
    M = torch.stack(colours, 1)                                                  # (3, k)
    px = lin[region]                                                             # (N, 3)
    sj = torch.linalg.lstsq(M[None].expand(px.shape[0], -1, -1), px[..., None]).solution[..., 0].clamp_min(0)
    lights, energies = [], []
    for j in range(k_best):
        y = torch.zeros_like(lum)
        y[region] = sj[:, j]
        c = fit_sh1(y[..., None].expand(-1, -1, 3).contiguous(), geo["normal"][0], region)
        az, el, strength = detect_dominant_light(c[None])
        tint = colours[j] / colours[j].sum().clamp_min(1e-6)
        energy = float(sj[:, j].sum() * luminance(tint))
        sat = float((tint.max() - tint.min()) / tint.max().clamp_min(1e-6))
        if k_best == 1:      # one light: its tint against the frame, as for the original light (removes surface colour)
            bright = lit & (lum >= sample_quantile(lum[region], 0.9))
            tint = lin[bright].mean(0) / lin[region].mean(0).clamp_min(1e-4)
            tint = tint / tint.sum().clamp_min(1e-6)
            sat = float((tint.max() - tint.min()) / tint.max().clamp_min(1e-6))
        color = {"hex": _hex_of(tint)} if sat > 0.35 else {"kelvin": _kelvin_of(tint.cpu())}
        lights.append({"az": az, "el": el, "strength": strength, "color": color, "energy": energy})
        energies.append(energy)
    top = max(energies)
    rig_lights = []
    for i, L in enumerate(sorted(lights, key=lambda L: -L["energy"])):
        ev = 0.5 + math.log2(max(L["energy"], 1e-6) / max(top, 1e-6))
        kelvin = L["color"].get("kelvin", 5600)
        rig_lights.append({
            "id": f"ref{i}", "name": f"Reference light {i + 1}", "type": "area", "mode": "generate", "enabled": True,
            "reference": True, "color": L["color"], "softness": 0.5, "cone": {"angle": 40.0, "feather": 0.4},
            "shadow": True, "motion": None, "prompt": "",
            "keys": [{"sec": 0, "azimuth": round(L["az"]), "elevation": round(L["el"]), "distance": 1.5,
                      "intensity_ev": round(ev, 1), "kelvin": kelvin, "ease": "smooth"}]})
    # the reference's darkness away from its lights -> the scene's base exposure
    dark, bright = sample_quantile(lum[region], 0.15), sample_quantile(lum[region], 0.9)
    amb_ev = max(-4.0, min(0.0, math.log2(float(dark.clamp_min(1e-5) / bright.clamp_min(1e-5))) + 2.0))
    rig = load_rig({"lights": rig_lights,
                    "ambient": {"kelvin": 6500, "intensity_ev": round(amb_ev, 1)} if amb_ev < -0.5 else None})
    desc = []
    for l in rig_lights:
        k = l["keys"][0]
        col = l["color"].get("hex") or f"{l['color']['kelvin']}K"
        side = "right" if k["azimuth"] > 15 else "left" if k["azimuth"] < -15 else "centre"
        front = "front" if abs(k["azimuth"]) < 90 else "back"
        desc.append(f"{l['name']}: {front} {side}, azimuth {k['azimuth']}°, elevation {k['elevation']}°, {col}, "
                    f"{k['intensity_ev']:+.1f} stops")
    report = (str(len(rig_lights)) + " lights found in the reference: " + "; ".join(desc) +
              (f". Shadows at about {amb_ev:+.1f} stops, set as ambient" if rig["ambient"] else "") +
              ". Softness cannot be read, so it starts at medium. Applying these switches the original lights off; you can adjust them afterwards.")
    return rig, report
