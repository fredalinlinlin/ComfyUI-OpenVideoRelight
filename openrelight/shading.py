"""Physical light pass: relight every frame from the LightRig, normals and depth.

Per pixel, in linear light:

    albedo   = rgb / mix(1, S0, delight)          S0 = original shading (SH order-1 fit), mean 1
    diffuse  = sum_i  colour_i * I_i * wrap(N.L_i, softness) * falloff * cone * shadow
    specular = sum_i  colour_i * I_i * blinn(N, L_i, V) * falloff * cone * shadow
    relit    = albedo * (keep * mix(1, S0, delight) + ambient + diffuse) + specular

Lights whose mode is "generate" or "both" go into the guide the video model sees.
"composite" lights are rendered into a separate additive layer that DetailLock adds
after generation, so their timing is exact no matter what the model does.
The same maths runs in the browser preview, so what you see is what is rendered.
"""

import math

import torch
import torch.nn.functional as F

from .geometry import smooth_1d
from .rig import evaluate_rig

LUMA = (0.2126, 0.7152, 0.0722)


def srgb_to_linear(x):
    return torch.where(x <= 0.04045, x / 12.92, ((x.clamp_min(0) + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(x):
    x = x.clamp_min(0)
    return torch.where(x <= 0.0031308, x * 12.92, 1.055 * x ** (1 / 2.4) - 0.055)


def rolloff(x, ref=None, knee=0.75):
    """Film-like shoulder: values above the knee are compressed into (knee, 1) instead of clipping.
    With ref (the original pixel), the knee is never below the original, so untouched pixels stay identical."""
    k = torch.full_like(x, knee) if ref is None else ref.clamp(max=0.999).clamp_min(knee)
    head = (1.0 - k).clamp_min(1e-3)
    over = (x - k).clamp_min(0)
    return torch.where(x > k, k + head * (1 - torch.exp(-over / head)), x)


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


def original_shading(normal, coeffs, region):
    s = coeffs[0] + (normal * coeffs[1:]).sum(-1)
    floor = coeffs[0].abs() * 0.15 + 1e-3
    s = s.clamp_min(floor)
    mean = s[region].mean() if region.any() else s.mean()
    return s / mean.clamp_min(1e-4)


def _project(q, K):
    """Camera-space points (..., 3) -> grid_sample coords in [-1, 1]."""
    z = q[..., 2].clamp_min(1e-4)
    u = K[0, 0] * q[..., 0] / z + K[0, 2]
    v = K[1, 1] * q[..., 1] / z + K[1, 2]
    return torch.stack([u * 2 - 1, v * 2 - 1], dim=-1)


def screen_shadow(P, L_vec, depth, K, march_len, steps=12, thickness=0.08):
    """Screen-space ray march toward the light. Returns visibility (H, W) in [0, 1]."""
    H, W = depth.shape
    dmap = depth.view(1, 1, H, W)
    occ = torch.zeros(H, W, device=P.device)
    for s in range(1, steps + 1):
        q = P + L_vec * (march_len * s / steps)
        grid = _project(q, K).view(1, H, W, 2)
        scene = F.grid_sample(dmap, grid, mode="bilinear", padding_mode="border", align_corners=False).view(H, W)
        gap = q[..., 2] - scene
        hit = (gap > 0.01 * q[..., 2]) & (gap < thickness * q[..., 2]) & (grid.view(H, W, 2).abs() <= 1).all(-1)
        occ = torch.maximum(occ, hit.float() * (1.0 - (s - 1) / steps * 0.5))
    # MoGe's thin structures and depth edges give isolated false hits; a small blur turns
    # them from blotches into soft penumbra instead of hard rectangular patches
    k = max(3, (min(H, W) // 90) * 2 + 1)
    occ = F.avg_pool2d(occ.view(1, 1, H, W), k, stride=1, padding=k // 2).view(H, W)
    return 1.0 - occ


def _smoothstep(e0, e1, x):
    t = ((x - e0) / (e1 - e0 + 1e-6)).clamp(0, 1)
    return t * t * (3 - 2 * t)


def light_contribution(lt, P, N, V, anchor, depth, K, params):
    """Diffuse (H, W, 3) and specular (H, W, 3) for one evaluated light."""
    dev = P.device
    d_unit = torch.tensor(lt["direction"], device=dev)
    subject_dist = anchor.norm().clamp_min(1e-3)
    if lt["type"] == "directional":
        L = d_unit.view(1, 1, 3).expand_as(P)
        atten = torch.ones(P.shape[:2], device=dev)
        lp = None
    else:
        lp = anchor + d_unit * lt["distance"] * subject_dist
        to_light = lp.view(1, 1, 3) - P
        dist = to_light.norm(dim=-1).clamp_min(1e-4)
        L = to_light / dist.unsqueeze(-1)
        ref = (lp - anchor).norm().clamp_min(1e-4)
        atten = (ref / dist).pow(lt["decay"]).clamp(max=6.0)

    cone = torch.ones_like(atten)
    if lt["type"] == "spot" and lp is not None:
        aim = F.normalize(anchor - lp, dim=0)
        cos_a = (-L * aim.view(1, 1, 3)).sum(-1)
        half = math.radians(lt["cone_angle"]) / 2
        inner = half * (1.0 - lt["cone_feather"])
        cone = _smoothstep(math.cos(half), math.cos(inner), cos_a)

    wrap = lt["softness"] * 1.0
    ndl = (N * L).sum(-1)
    diff = ((ndl + wrap) / (1 + wrap)).clamp_min(0)
    # relief < 1 flattens the shading toward an even wash (1 = full form from the normals)
    relief = params.get("relief", 1.0)
    if relief < 1.0:
        diff = diff * relief + (1.0 - relief)

    vis = torch.ones_like(atten)
    if params.get("shadows") and lt["shadow"]:
        march = subject_dist * params.get("shadow_range", 0.6)
        vis = screen_shadow(P, L, depth, K, march, steps=params.get("shadow_steps", 12))
        soft = 0.25 + 0.6 * lt["softness"]
        vis = 1.0 - (1.0 - vis) * params.get("shadow_strength", 0.85) * (1.0 - soft * 0.5)

    rough = params.get("roughness", 0.45)
    shin = max(4.0, min(256.0, 2.0 / max(rough, 0.05) ** 4 - 2.0))
    Hv = F.normalize(L + V, dim=-1)
    spec = (N * Hv).sum(-1).clamp_min(0).pow(shin) * (ndl > 0).float()
    spec = spec * params.get("spec_strength", 0.3) * lt["specular"] * (shin + 8) / (8 * math.pi) * 0.1

    color = torch.tensor(lt["color"], device=dev) * lt["intensity"]
    k = (atten * cone * vis).unsqueeze(-1)
    return color * diff.unsqueeze(-1) * k, color * spec.unsqueeze(-1) * k


def render(images, geo, rig, fps, params, progress=None):
    """images: (T, H, W, 3) sRGB in [0, 1] at the geometry's resolution.
    Returns sRGB guide / full preview, plus linear composite layer and shading."""
    dev = geo["normal"].device
    T, H, W = geo["depth"].shape
    imgs = images.to(dev).float()
    if imgs.shape[1:3] != (H, W):
        imgs = F.interpolate(imgs.permute(0, 3, 1, 2), size=(H, W), mode="bilinear",
                             align_corners=False).permute(0, 2, 3, 1)
    frame_count = imgs.shape[0]
    T = min(T, frame_count)
    K = geo["intrinsics"].to(dev)
    delight = params.get("delight", 0.6)
    keep = params.get("keep", 0.35)
    scope_all = params.get("scope", "subject") == "all"

    # pass 1: original shading per frame, smoothed over time so it doesn't flicker
    coeffs = []
    for t in range(T):
        lin = srgb_to_linear(imgs[t])
        region = geo["valid"][t] & (geo["alpha"][t] > 0.5) if not scope_all else geo["valid"][t]
        coeffs.append(fit_sh1(lin, geo["normal"][t], region))
    coeffs = smooth_1d(torch.stack(coeffs), sigma=4.0)

    out_guide, out_full, out_layer, out_shading = [], [], [], []
    for t in range(T):
        lin = srgb_to_linear(imgs[t])
        P, N = geo["points"][t], geo["normal"][t]
        V = F.normalize(-P, dim=-1)
        valid = geo["valid"][t]
        alpha = geo["alpha"][t] if not scope_all else valid.float()
        region = valid & (alpha > 0.5)

        S0 = original_shading(N, coeffs[t], region)
        S0m = (1.0 - delight) + delight * S0
        albedo = lin / S0m.unsqueeze(-1)

        lights, ambient = evaluate_rig(rig, float(t), frame_count, fps)
        ambient = torch.tensor(ambient, device=dev)
        dg = torch.zeros_like(lin)
        sg = torch.zeros_like(lin)
        dc = torch.zeros_like(lin)
        sc = torch.zeros_like(lin)
        for lt in lights:
            d, s = light_contribution(lt, P, N, V, geo["anchor"][t], geo["depth"][t], K, params)
            if lt["mode"] == "composite":
                dc += d
                sc += s
            else:
                dg += d
                sg += s

        base = keep * S0m.unsqueeze(-1) + ambient
        guide = albedo * (base + dg) + sg
        layer = albedo * dc + sc
        full = guide + layer

        m = (alpha * valid.float()).unsqueeze(-1)
        exposure = 2.0 ** params.get("exposure_ev", 0.0)
        guide = lin * (1 - m) + guide * exposure * m
        full = lin * (1 - m) + full * exposure * m
        layer = layer * exposure * m

        if params.get("rolloff", True):
            guide, full = rolloff(guide, lin), rolloff(full, lin)
        out_guide.append(linear_to_srgb(guide).clamp(0, 1))
        out_full.append(linear_to_srgb(full).clamp(0, 1))
        out_layer.append(layer)
        out_shading.append(linear_to_srgb((base + dg + dc) * m / 2.0).clamp(0, 1))
        if progress:
            progress(t + 1, T)

    return (torch.stack(out_guide), torch.stack(out_full), torch.stack(out_layer),
            torch.stack(out_shading), coeffs)


def detect_dominant_light(coeffs):
    """Turn averaged SH1 coefficients into an (azimuth, elevation, strength) estimate."""
    a = coeffs[:, 1:].mean(0)
    strength = float(a.norm() / coeffs[:, 0].mean().abs().clamp_min(1e-4))
    d = F.normalize(a, dim=0)
    # camera space -> rig angles (inverse of rig._dir_from_angles)
    el = math.degrees(math.asin(max(-1.0, min(1.0, -float(d[1])))))
    az = math.degrees(math.atan2(float(d[0]), -float(d[2])))
    return az, el, strength


def suppress_depth_edges(m, depth, rel=0.02, k=7):
    """Take the darker neighbour at depth jumps (silhouettes).

    MoGe gives silhouette pixels a depth between the person and the background and a normal facing
    the camera, so a frontal light draws a thin bright outline around every person; the image editor
    then copies it as a real edge line. MoGe smears some edges over several pixels, so the jump is
    measured over a 5x5 window. m: (H, W, C), depth: (H, W).
    """
    d = depth[None, None]
    dmax = F.max_pool2d(d, 5, 1, 2)
    dmin = -F.max_pool2d(-d, 5, 1, 2)
    edge = (dmax - dmin) / dmin.clamp_min(1e-4) > rel
    edge = F.max_pool2d(edge.float(), 5, 1, 2)[0, 0] > 0
    x = m.permute(2, 0, 1)[None]
    darkest = -F.max_pool2d(-x, k, 1, k // 2)
    return torch.where(edge.unsqueeze(-1), darkest[0].permute(1, 2, 0), m)


def light_map(geo, rig, frame_count, fps, include_composite=True, progress=None,
              shadow_strength=0.0, shadow_length=0.6, relief=1.0, normalize=True):
    """Where the rig's lights land, nothing else: (T, H, W, 3) linear, 0 = no new light, ~1 = the brightest spot.

    A reference for an image editor ("the light falls here"), not a picture to copy, so there is
    no albedo, no original shading and no specular. Normalised over the whole clip so dimming and
    flicker stay visible from frame to frame.

    shadow_strength: screen-space cast shadows from MoGe's depth (0 = off). Only what is visible in the
    frame casts them. shadow_length: how far shadows reach, as a fraction of the camera-to-subject distance.
    relief: 1 = full form from the normals, 0 = an even wash inside the beam.
    """
    from .geometry import sample_quantile

    dev = geo["normal"].device
    T = min(geo["depth"].shape[0], frame_count)
    K = geo["intrinsics"].to(dev)
    # longer shadows march further, so take more steps to keep thin occluders from being skipped
    params = {"shadows": shadow_strength > 0, "shadow_strength": shadow_strength, "shadow_range": shadow_length,
              "shadow_steps": max(12, round(20 * shadow_length)), "relief": relief}
    maps = []
    for t in range(T):
        P, N = geo["points"][t], geo["normal"][t]
        V = F.normalize(-P, dim=-1)
        lights, _ = evaluate_rig(rig, float(t), frame_count, fps)
        acc = torch.zeros_like(P)
        for lt in lights:
            if lt["mode"] == "composite" and not include_composite:
                continue
            d, _ = light_contribution(lt, P, N, V, geo["anchor"][t], geo["depth"][t], K, params)
            acc += d
        maps.append(suppress_depth_edges(acc, geo["depth"][t]) * geo["valid"][t].float().unsqueeze(-1))
        if progress:
            progress(t + 1, T)
    maps = torch.stack(maps)
    if not normalize:
        return maps
    peak = sample_quantile(luminance(maps), 0.995).clamp_min(1e-4)
    return (maps / peak).clamp(0, 1)


def light_mockup(images, maps, ambient=0.35, gain=4.0):
    """Paint the light map onto the frames: a rough photo of the target lighting.

    An image editor ignores an abstract grey map, so it edits this instead: where the light falls is then
    fixed by the geometry. The original is dimmed to `ambient` and the new light multiplies it up. Output is
    at the images' resolution (the map is upsampled), so keyframes keep full-resolution detail.
    """
    dev = maps.device
    n = min(images.shape[0], maps.shape[0])
    lin = srgb_to_linear(images[:n].to(dev).float())
    m = maps[:n]
    if m.shape[1:3] != lin.shape[1:3]:
        m = F.interpolate(m.permute(0, 3, 1, 2), size=lin.shape[1:3], mode="bilinear",
                          align_corners=False).permute(0, 2, 3, 1)
    out = lin * (ambient + gain * m)
    return linear_to_srgb(rolloff(out)).clamp(0, 1)


def relight_mockup(images, geo, rig, frame_count, fps, ambient=0.3, soften=0.03, progress=None, **shadow):
    """The clip with the stage's lighting painted on: a preview, and a rough direction guide for the editor.

    The scene is lit twice on MoGe's geometry: by the light the clip already has (the detected "original"
    lights, as detected) and by the stage as it is now. Each pixel changes by the ratio of the two, so
    lights left as detected change nothing, a moved light darkens where it was and brightens where it
    goes, a deleted one darkens what it lit, and new lights add on top. `ambient` is the share of the
    clip's light no detected light explains (it is never removed); the stage's ambient setting scales it.
    The ratio is blurred by `soften` (fraction of the frame) and clamped: the geometry knows directions,
    not materials, so the guide keeps where the light goes and drops the hard CG terminator.
    Returns (light map for the preview, sRGB (T,H,W,3) at the geometry's size; mock-up, sRGB at the images' size).
    """
    from .geometry import sample_quantile
    from .facelock import gaussian_blur
    from .rig import light_color, original_rig

    target = light_map(geo, rig, frame_count, fps, progress=progress, normalize=False, **shadow)
    orig_rig = original_rig(rig)
    if orig_rig["lights"]:
        before = light_map(geo, orig_rig, frame_count, fps, normalize=False, **shadow)
        ref = sample_quantile(luminance(before), 0.995).clamp_min(1e-4)
    else:
        before = torch.zeros_like(target)
        ref = sample_quantile(luminance(target), 0.995).clamp_min(1e-4)
    amb = rig.get("ambient")
    if amb:
        tint = torch.tensor(light_color({"kelvin": amb.get("kelvin", 6500)}), device=target.device)
        tint = tint / torch.tensor(light_color({"kelvin": 6500}), device=target.device)
        a_now = ambient * 2.0 ** float(amb.get("intensity_ev", 0.0)) * tint
    else:
        a_now = torch.full((3,), ambient, device=target.device)
    ratio = (a_now + target / ref) / (ambient + before / ref)
    # soften in log space (a ratio), at the geometry's resolution, then clamp
    T, H, W, _ = ratio.shape
    sigma = max(0.5, soften * min(H, W))
    logr = gaussian_blur(ratio.clamp(1e-3, 100).log().permute(0, 3, 1, 2), sigma).permute(0, 2, 3, 1)
    ratio = logr.exp().clamp(0.1, 8.0)
    preview = linear_to_srgb((target / ref).clamp(0, 1))

    # a few frames at a time, each kept on the CPU: the whole clip at once at the images' size ran out of GPU
    # memory (96 frames of 1920x1080 is ~10 GB of intermediates, 2026-10-30)
    n = min(images.shape[0], T)
    out = []
    for i in range(0, n, 8):
        lin = srgb_to_linear(images[i:min(i + 8, n)].to(ratio.device).float())
        r = ratio[i:i + lin.shape[0]]
        if r.shape[1:3] != lin.shape[1:3]:
            r = F.interpolate(r.permute(0, 3, 1, 2), size=lin.shape[1:3], mode="bilinear",
                              align_corners=False).permute(0, 2, 3, 1)
        out.append(linear_to_srgb(rolloff(lin * r)).clamp(0, 1).cpu())
    return preview, torch.cat(out)
