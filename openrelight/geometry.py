"""Turn per-frame MoGe output into temporally stable geometry for relighting.

MoGe runs one frame at a time, so its scale and normals jitter between frames.
prepare_geometry() fixes the per-frame scale, smooths points/normals over time
(aligned by optical flow when the clip is given, else bilaterally, so moving parts
don't ghost), fills invalid pixels, and finds a per-frame subject anchor that lights
orbit around.

All tensors are in MoGe's OpenCV camera space: X right, Y down, Z into the scene.
"""

import math

import torch
import torch.nn.functional as F


def _gauss_weights(sigma, device):
    radius = max(1, int(math.ceil(sigma * 2.5)))
    x = torch.arange(-radius, radius + 1, device=device, dtype=torch.float32)
    w = torch.exp(-0.5 * (x / max(sigma, 1e-6)) ** 2)
    return w / w.sum(), radius


def smooth_1d(values, sigma):
    """Gaussian smoothing along dim 0 of a (T, ...) tensor, edge-clamped."""
    if sigma <= 0 or values.shape[0] < 2:
        return values
    w, r = _gauss_weights(sigma, values.device)
    T = values.shape[0]
    out = torch.zeros_like(values)
    for k in range(-r, r + 1):
        idx = torch.clamp(torch.arange(T, device=values.device) + k, 0, T - 1)
        out += w[k + r] * values[idx]
    return out


def temporal_bilateral(x, sigma_t, sigma_v, valid=None):
    """Smooth a (T, H, W, C) tensor over time, ignoring neighbours that differ by more than ~sigma_v."""
    if sigma_t <= 0 or x.shape[0] < 2:
        return x
    w, r = _gauss_weights(sigma_t, x.device)
    T = x.shape[0]
    out = torch.empty_like(x)
    for t in range(T):
        acc = torch.zeros_like(x[t])
        wsum = torch.zeros_like(x[t][..., :1])
        for k in range(-r, r + 1):
            s = min(max(t + k, 0), T - 1)
            diff = ((x[s] - x[t]) ** 2).sum(-1, keepdim=True)
            wk = w[k + r] * torch.exp(-diff / (2 * sigma_v ** 2))
            if valid is not None:
                wk = wk * valid[s].unsqueeze(-1)
            acc += wk * x[s]
            wsum += wk
        out[t] = torch.where(wsum > 1e-6, acc / wsum.clamp_min(1e-6), x[t])
    return out


def _rays(intrinsics, H, W, device):
    """Per-pixel unit-depth rays (z = 1) from MoGe's normalised intrinsics (fx by width, fy by height)."""
    fx, fy = intrinsics[0, 0].item(), intrinsics[1, 1].item()
    cx, cy = intrinsics[0, 2].item(), intrinsics[1, 2].item()
    u = (torch.arange(W, device=device, dtype=torch.float32) + 0.5) / W
    v = (torch.arange(H, device=device, dtype=torch.float32) + 0.5) / H
    vv, uu = torch.meshgrid(v, u, indexing="ij")
    return torch.stack([(uu - cx) / fx, (vv - cy) / fy, torch.ones_like(uu)], dim=-1)


def _normals_from_points(points):
    dx = points[:, :, 2:, :] - points[:, :, :-2, :]
    dy = points[:, 2:, :, :] - points[:, :-2, :, :]
    dx = F.pad(dx.permute(0, 3, 1, 2), (1, 1, 0, 0), mode="replicate").permute(0, 2, 3, 1)
    dy = F.pad(dy.permute(0, 3, 1, 2), (0, 0, 1, 1), mode="replicate").permute(0, 2, 3, 1)
    return F.normalize(torch.cross(dy, dx, dim=-1), dim=-1)


def default_intrinsics(H, W, fov_x_deg=55.0):
    fx = 0.5 / math.tan(math.radians(fov_x_deg) / 2)
    return torch.tensor([[fx, 0, 0.5], [0, fx * W / H, 0.5], [0, 0, 1]], dtype=torch.float32)


def prepare_geometry(moge, alpha=None, temporal_sigma=1.5, stabilize_scale=True, device=None, progress=None):
    """moge: MOGE_GEOMETRY dict. alpha: optional (T, H, W) subject matte in [0, 1]."""
    device = device or (moge["points"].device if "points" in moge else torch.device("cpu"))
    points = moge["points"].float().to(device).clone()
    T, H, W, _ = points.shape
    valid = torch.isfinite(points).all(-1)
    if "mask" in moge:
        valid &= moge["mask"].to(device).bool()
    valid &= points[..., 2] > 0

    intr = moge.get("intrinsics")
    K = intr[0].float() if intr is not None else default_intrinsics(H, W)
    rays = _rays(K, H, W, device)

    depth = torch.where(valid, points[..., 2], torch.zeros_like(points[..., 2]))

    if alpha is not None:
        alpha = alpha.float().to(device)
        if alpha.shape[1:] != (H, W):
            alpha = F.interpolate(alpha.unsqueeze(1), size=(H, W), mode="bilinear", align_corners=False).squeeze(1)
    else:
        alpha = torch.ones(T, H, W, device=device)

    # --- per-frame scale: MoGe's metric scale drifts frame to frame
    region = valid & (alpha > 0.5)
    scales = []
    for t in range(T):
        sel = depth[t][region[t]] if region[t].sum() > 64 else depth[t][valid[t]]
        scales.append(sel.median() if sel.numel() else torch.tensor(1.0, device=device))
    scales = torch.stack(scales).clamp_min(1e-4)
    if stabilize_scale and T > 1:
        target = smooth_1d(scales, sigma=max(temporal_sigma * 4, 4.0))
        depth = depth * (target / scales).view(T, 1, 1)

    # --- fill invalid (sky, holes) with a far plane so lighting math stays finite
    far = torch.stack([sample_quantile(d[v], 0.98) if v.any() else torch.tensor(10.0, device=device)
                       for d, v in zip(depth, valid)]) * 1.5
    depth = torch.where(valid, depth, far.view(T, 1, 1).expand_as(depth))
    points = rays.unsqueeze(0) * depth.unsqueeze(-1)

    # --- normals: prefer MoGe's predicted ones (v2/v3), fall back to point derivatives
    if "normal" in moge:
        normal = F.normalize(moge["normal"].float().to(device), dim=-1)
        normal = torch.where(torch.isfinite(normal), normal, torch.zeros_like(normal))
    else:
        normal = _normals_from_points(points)
    facing_away = (normal * -F.normalize(points, dim=-1)).sum(-1, keepdim=True) < -0.2
    normal = torch.where(facing_away, -normal, normal)

    # --- temporal stabilization
    validf = valid.float()
    normal = F.normalize(temporal_bilateral(normal, temporal_sigma, sigma_v=0.35, valid=validf), dim=-1)
    depth = temporal_bilateral(depth.unsqueeze(-1), temporal_sigma,
                               sigma_v=float(depth[valid].std()) * 0.1 if valid.any() else 1.0,
                               valid=validf).squeeze(-1)
    points = rays.unsqueeze(0) * depth.unsqueeze(-1)

    # --- subject anchor per frame: the point lights orbit around
    anchors = []
    for t in range(T):
        sel = region[t] if region[t].sum() > 64 else None
        if sel is None:
            box = torch.zeros(H, W, dtype=torch.bool, device=device)
            box[int(H * 0.3):int(H * 0.7), int(W * 0.3):int(W * 0.7)] = True
            sel = box & valid[t]
        pts = points[t][sel] if sel.any() else points[t].reshape(-1, 3)
        anchors.append(pts.median(dim=0).values)
    anchors = smooth_1d(torch.stack(anchors), sigma=3.0)

    return {
        "points": points,           # (T, H, W, 3)
        "normal": normal,           # (T, H, W, 3), unit, outward
        "depth": depth,             # (T, H, W)
        "valid": valid,             # (T, H, W) bool
        "alpha": alpha,             # (T, H, W)
        "anchor": anchors,          # (T, 3)
        "intrinsics": K,            # (3, 3) normalised
        "size": (H, W),
    }


def normal_preview(geo):
    n = geo["normal"] * torch.tensor([1.0, -1.0, -1.0], device=geo["normal"].device)  # OpenGL-style view
    return (n * 0.5 + 0.5).clamp(0, 1)


def sample_quantile(x, q, max_n=1_000_000):
    """torch.quantile refuses >16M elements; a strided sample is plenty for clipping ranges."""
    flat = x.reshape(-1)
    if flat.numel() > max_n:
        flat = flat[:: flat.numel() // max_n + 1]
    return flat.float().quantile(q)


def depth_preview(geo):
    d = geo["depth"]
    disp = 1.0 / d.clamp_min(1e-4)
    lo, hi = sample_quantile(disp, 0.01), sample_quantile(disp, 0.99)
    v = ((disp - lo) / (hi - lo).clamp_min(1e-6)).clamp(0, 1)
    return v.unsqueeze(-1).expand(*v.shape, 3).contiguous()
