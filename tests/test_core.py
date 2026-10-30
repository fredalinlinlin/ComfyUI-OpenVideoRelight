"""Core maths checks on a synthetic sphere.

Run with pytest, or without it: python tests/test_core.py
"""

import json
import math
import os
import re
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openrelight import lm  # noqa: E402
from openrelight import rig as rigmod  # noqa: E402
from openrelight.detail import detail_lock  # noqa: E402
from openrelight.geometry import prepare_geometry  # noqa: E402
from openrelight.prompt import direction_words, rig_to_prompt  # noqa: E402
from openrelight.shading import detect_dominant_light, fit_sh1, light_map, render, srgb_to_linear  # noqa: E402

H, W, T = 96, 128, 6


def sphere_scene(light_dir=None):
    """A unit sphere 3 units in front of a 60-degree camera, on a far back wall."""
    fx = 0.5 / math.tan(math.radians(30))
    K = torch.tensor([[fx, 0, 0.5], [0, fx * W / H, 0.5], [0, 0, 1.0]])
    u = (torch.arange(W) + 0.5) / W
    v = (torch.arange(H) + 0.5) / H
    vv, uu = torch.meshgrid(v, u, indexing="ij")
    d = torch.stack([(uu - 0.5) / K[0, 0], (vv - 0.5) / K[1, 1], torch.ones_like(uu)], -1)
    d = d / d.norm(dim=-1, keepdim=True)
    c = torch.tensor([0.0, 0.0, 3.0])
    b = (d * c).sum(-1)
    disc = b * b - (c * c).sum() + 1.0
    hit = disc > 0
    t_hit = torch.where(hit, b - disc.clamp_min(0).sqrt(), 6.0 / d[..., 2])
    P = d * t_hit.unsqueeze(-1)
    N = torch.where(hit.unsqueeze(-1), (P - c), torch.tensor([0.0, 0.0, -1.0]).expand_as(P))
    N = N / N.norm(dim=-1, keepdim=True)
    if light_dir is None:
        rgb = torch.full((H, W, 3), 0.5)
    else:
        L = torch.tensor(light_dir) / torch.tensor(light_dir).norm()
        shade = (N * L).sum(-1).clamp_min(0) * 0.8 + 0.1
        rgb = (shade.unsqueeze(-1) * torch.tensor([0.8, 0.7, 0.6])).clamp(0, 1) ** (1 / 2.2)
    moge = {"points": P.expand(T, H, W, 3).clone(), "normal": N.expand(T, H, W, 3).clone(),
            "mask": torch.ones(T, H, W, dtype=torch.bool), "intrinsics": K.expand(T, 3, 3).clone()}
    return moge, rgb.expand(T, H, W, 3).clone(), hit.float().expand(T, H, W).clone()


def _halves(img, t, mask):
    """Mean brightness of the left and right halves of the masked object."""
    lum = img[t].mean(-1)
    m = mask[t] > 0.5
    left, right = m.clone(), m.clone()
    left[:, W // 2:] = False
    right[:, : W // 2] = False
    return lum[left].mean().item(), lum[right].mean().item()


def test_direction_convention():
    assert direction_words(0, 0) == "the front"
    assert direction_words(90, 0) == "the right side"
    assert direction_words(-90, 0) == "the left side"
    assert "backlight" in direction_words(180, 0)
    d = rigmod._dir_from_angles(90, 0)
    assert d[0] > 0.99  # +X = screen right


def test_sweep_moves_light_from_right_to_front():
    moge, rgb, alpha = sphere_scene()
    geo = prepare_geometry(moge, alpha=alpha, temporal_sigma=0)
    rig = rigmod.load_rig({"lights": [{
        "id": "beam", "type": "point", "mode": "generate", "shadow": False, "softness": 0.0,
        "keys": [{"t": 0, "azimuth": 90, "elevation": 0, "distance": 2.0, "intensity_ev": 1.0},
                 {"t": 1, "azimuth": 0, "elevation": 0, "distance": 2.0, "intensity_ev": 1.0}],
    }], "ambient": {"kelvin": 6500, "intensity_ev": -8}})
    guide, full, layer, shading, _ = render(rgb, geo, rig, 24.0, {"delight": 0.0, "keep": 0.0, "shadows": False})
    left0, right0 = _halves(guide, 0, alpha)
    assert right0 > left0 * 1.5, (left0, right0)
    centre_last = guide[-1, H // 2 - 5:H // 2 + 5, W // 2 - 5:W // 2 + 5].mean().item()
    edge_last = guide[-1, H // 2 - 5:H // 2 + 5, W // 2 + 24:W // 2 + 34].mean().item()
    assert centre_last > edge_last, (centre_last, edge_last)
    assert layer.abs().max().item() == 0.0  # no composite lights


def test_composite_lights_go_to_layer_only():
    moge, rgb, alpha = sphere_scene()
    geo = prepare_geometry(moge, alpha=alpha, temporal_sigma=0)
    rig = rigmod.load_rig({"lights": [{"id": "c", "type": "point", "mode": "composite", "shadow": False,
                                       "keys": [{"frame": 0, "azimuth": -60, "elevation": 0, "intensity_ev": 1}]}]})
    guide, full, layer, _, _ = render(rgb, geo, rig, 24.0, {"delight": 0.0, "keep": 1.0, "shadows": False})
    assert layer.max().item() > 0.05
    assert (full - guide).abs().max().item() > 0.05


def test_keep_original_with_no_lights_is_identity():
    moge, rgb, alpha = sphere_scene(light_dir=[1.0, -1.0, -1.0])
    geo = prepare_geometry(moge, alpha=alpha, temporal_sigma=0)
    rig = rigmod.load_rig({"lights": [], "ambient": {"kelvin": 6500, "intensity_ev": -20}})
    guide, *_ = render(rgb, geo, rig, 24.0, {"delight": 0.7, "keep": 1.0, "shadows": False})
    assert (guide - rgb).abs().max().item() < 0.02


def test_detect_light_direction():
    # light from screen right and above: camera-space dir (+x, -y, -z)
    moge, rgb, alpha = sphere_scene(light_dir=[0.7, -0.5, -0.5])
    geo = prepare_geometry(moge, alpha=alpha, temporal_sigma=0)
    c = fit_sh1(srgb_to_linear(rgb[0]), geo["normal"][0], geo["valid"][0] & (geo["alpha"][0] > 0.5))
    az, el, _ = detect_dominant_light(c.unsqueeze(0))
    assert 20 < az < 90, az
    assert el > 5, el


def test_shadows_run():
    moge, rgb, alpha = sphere_scene()
    geo = prepare_geometry(moge, alpha=alpha, temporal_sigma=1.0)
    rig = rigmod.load_rig(None, preset="sweep_right_to_front")
    guide, *_ = render(rgb, geo, rig, 24.0, {"shadows": True, "scope": "all"})
    assert torch.isfinite(guide).all()


def test_detail_lock_keeps_original_detail_and_takes_generated_light():
    torch.manual_seed(0)
    orig = (torch.rand(4, 64, 80, 3) * 0.2 + 0.4)            # textured, flat light
    ramp = torch.linspace(0.3, 1.6, 80).view(1, 1, 80, 1)   # generated: light from the right, blurry
    gen_lin = srgb_to_linear(torch.full((4, 32, 40, 3), 0.5)) * torch.nn.functional.interpolate(
        ramp.permute(0, 3, 1, 2), size=(32, 40)).permute(0, 2, 3, 1)
    gen = gen_lin.clamp(0, 1) ** (1 / 2.2)
    out = detail_lock(orig, gen, radius=0.08, strength=1.0, highlights=0.0, device=torch.device("cpu"))
    assert out.shape == orig.shape
    left = out[:, :, :20].mean().item()
    right = out[:, :, -20:].mean().item()
    assert right > left * 1.3, (left, right)
    hf_orig = orig - torch.nn.functional.avg_pool2d(orig.permute(0, 3, 1, 2), 5, 1, 2).permute(0, 2, 3, 1)
    hf_out = out - torch.nn.functional.avg_pool2d(out.permute(0, 3, 1, 2), 5, 1, 2).permute(0, 2, 3, 1)
    corr = torch.corrcoef(torch.stack([hf_orig.flatten(), hf_out.flatten()]))[0, 1].item()
    assert corr > 0.8, corr


def test_prompt_mentions_sweep_with_times():
    rig = rigmod.load_rig(None, preset="sweep_right_to_front")
    text = rig_to_prompt(rig, frame_count=121, fps=24.0)
    assert "Only the lighting changes" in text
    assert "sweeps from the right side to the front" in text
    assert "0.0s to 3.0s" in text


def test_flicker_and_strobe_change_intensity():
    rig = rigmod.load_rig(None, preset="police")
    vals = [rigmod.evaluate_rig(rig, f, 48, 24.0)[0][0]["intensity"] for f in range(48)]
    assert min(vals) == 0.0 and max(vals) > 1.0
    rig = rigmod.load_rig(None, preset="firelight")
    vals = [rigmod.evaluate_rig(rig, f, 48, 24.0)[0][0]["intensity"] for f in range(48)]
    assert max(vals) - min(vals) > 0.3


def test_light_map_shows_where_the_light_lands():
    moge, _, alpha = sphere_scene()
    geo = prepare_geometry(moge, alpha=alpha, temporal_sigma=0)
    rig = rigmod.load_rig({"lights": [{
        "id": "beam", "type": "point", "mode": "generate", "softness": 0.0,
        "keys": [{"t": 0, "azimuth": 90, "elevation": 0, "distance": 2.0, "intensity_ev": 0.0},
                 {"t": 1, "azimuth": -90, "elevation": 0, "distance": 2.0, "intensity_ev": 0.0}],
    }]})
    maps = light_map(geo, rig, T, 24.0)
    assert maps.shape == (T, H, W, 3) and maps.min() >= 0 and maps.max() <= 1
    left0, right0 = _halves(maps, 0, alpha)
    left1, right1 = _halves(maps, T - 1, alpha)
    assert right0 > left0 * 3 and left1 > right1 * 3, (left0, right0, left1, right1)


def test_light_map_shadows_and_relief():
    moge, _, alpha = sphere_scene()
    geo = prepare_geometry(moge, alpha=alpha, temporal_sigma=0)
    rig = rigmod.load_rig({"lights": [{
        "id": "key", "type": "point", "mode": "generate", "softness": 0.0,
        "keys": [{"t": 0, "azimuth": 70, "elevation": 0, "distance": 2.0, "intensity_ev": 0.0}],
    }]})
    flat = light_map(geo, rig, T, 24.0, relief=0.0)
    full = light_map(geo, rig, T, 24.0, relief=1.0)
    # inside the sphere, away from the silhouette (edges take the darker neighbour)
    lit = torch.nn.functional.max_pool2d((alpha[0] < 0.5).float()[None, None], 9, 1, 4)[0, 0] == 0
    # an even wash has less contrast across the sphere than full form
    assert flat[0].mean(-1)[lit].std() < full[0].mean(-1)[lit].std() * 0.5
    # shadows only take light away, and strength 0 is the same as no shadows
    off = light_map(geo, rig, T, 24.0, shadow_strength=0.0)
    assert (off - full).abs().max() < 1e-6
    shad = light_map(geo, rig, T, 24.0, shadow_strength=1.0, shadow_length=1.0)
    assert shad.sum() < full.sum()


def test_no_bright_outline_at_silhouettes():
    # a pixel between the sphere and the wall, with an in-between depth and a normal facing the camera,
    # is what MoGe gives at silhouettes; under a frontal light it would draw a bright outline
    moge, _, alpha = sphere_scene()
    edge = (alpha[0] > 0.5) & ~torch.nn.functional.max_pool2d(
        (alpha[0] < 0.5).float()[None, None], 3, 1, 1)[0, 0].bool().logical_not()
    ys, xs = edge.nonzero(as_tuple=True)
    for t in range(T):
        moge["points"][t, ys, xs] = moge["points"][t, ys, xs] * 1.5
        moge["normal"][t, ys, xs] = torch.tensor([0.0, 0.0, -1.0])
    geo = prepare_geometry(moge, alpha=alpha, temporal_sigma=0)
    rig = rigmod.load_rig({"lights": [{"id": "f", "type": "point", "mode": "generate", "softness": 0.0,
                                       "keys": [{"t": 0, "azimuth": 0, "elevation": 0, "distance": 2.0}]}]})
    m = light_map(geo, rig, T, 24.0)[0].mean(-1)
    # the sphere's limb just inside the outline faces sideways and is dark; the outline must not outshine it
    near = (torch.nn.functional.max_pool2d((alpha[0] < 0.5).float()[None, None], 9, 1, 4)[0, 0] > 0)         & (alpha[0] > 0.5) & ~edge
    assert m[edge].mean() < m[near].mean(), (m[edge].mean(), m[near].mean())


def test_fit_pad_and_unpad_line_up_pixel_for_pixel():
    from openrelight.fit import fit_pad, unpad
    clip = torch.rand(6, 720, 1280, 3)
    x, w, h, n, info = fit_pad(clip, 0.5)
    assert (w, h, n) == (960, 544, 6) and w % 32 == 0 and h % 32 == 0
    back = unpad(x, info)
    assert back.shape == clip.shape
    # no crop, no shift: the round trip only blurs (downscale), the mean of each region stays put
    blur = lambda t: torch.nn.functional.avg_pool2d(t.permute(0, 3, 1, 2), 16)  # noqa: E731
    assert (blur(back) - blur(clip)).abs().max() < 0.03
    x, w, h, n, info = fit_pad(clip, 0)
    assert (w, h) == (1280, 736) and torch.equal(unpad(x, info), clip)


def test_fit_pad_keeps_every_frame_on_h3_grid():
    from openrelight.fit import fit_pad, grid_length, unpad
    assert [grid_length(n, (17, 5)) for n in (1, 5, 6, 39, 40, 49, 56)] == [5, 5, 22, 39, 56, 56, 56]
    clip = torch.rand(49, 64, 96, 3)                       # 2 s at 24 fps: H3 needs 56, the old prep cut it to 39
    x, w, h, n, info = fit_pad(clip, 0, frames="17k+5 (MiniMax H3)")
    assert n == 56 and x.shape[0] == 56 and torch.equal(x[48:], x[48:49].expand(8, -1, -1, -1))
    assert unpad(x, info).shape[0] == 49


def test_face_detail_lock_keeps_original_detail_and_takes_new_light():
    from openrelight.facelock import face_detail_lock, gaussian_blur
    torch.manual_seed(0)
    Tn, Hn, Wn = 3, 64, 96
    # original: dark, with fine texture (the "features"); generated: half resolution, brighter on the
    # right (the new light), its own different texture (a redrawn face)
    orig = (0.15 + 0.08 * torch.rand(Tn, Hn, Wn, 3)).clamp(0, 1)
    ramp = torch.linspace(1.0, 3.0, Wn // 2).view(1, 1, -1, 1)
    gen_lin = srgb_to_linear(torch.full((Tn, Hn // 2, Wn // 2, 3), 0.17)) * ramp
    gen = (gen_lin.clamp(0, 1) ** (1 / 2.2) + 0.05 * torch.rand(Tn, Hn // 2, Wn // 2, 3)).clamp(0, 1)
    mask = torch.zeros(Tn, Hn, Wn)
    mask[:, 16:48, 24:72] = 1.0
    out = face_detail_lock(orig, gen, mask, detail_px=2.0, grow=0, feather=0.0, method="frequency")
    assert out.shape == orig.shape
    inside = (slice(None), slice(20, 44), slice(28, 68))
    hf = lambda x: (x - gaussian_blur(x.permute(0, 3, 1, 2), 2.0).permute(0, 2, 3, 1))[inside]  # noqa: E731
    corr = torch.corrcoef(torch.stack([hf(orig).flatten(), hf(out).flatten()]))[0, 1].item()
    assert corr > 0.8, corr                                               # features are the original's
    up = torch.nn.functional.interpolate(gen.permute(0, 3, 1, 2), size=(Hn, Wn), mode="bicubic",
                                         align_corners=False, antialias=True).permute(0, 2, 3, 1).clamp(0, 1)
    # light is the generated one: same left/right brightness ratio as the generated clip, not the flat original
    ratio = lambda x: x[:, 20:44, 56:68].mean().item() / x[:, 20:44, 28:40].mean().item()  # noqa: E731
    assert abs(ratio(out) - ratio(up)) < 0.03 and ratio(out) > 1.08, (ratio(out), ratio(up), ratio(orig))
    assert (out[:, :8] - up[:, :8]).abs().max() < 1e-4                   # outside the mask: generated, upscaled


def test_guided_lock_takes_original_edges_without_outlines():
    from openrelight.facelock import face_detail_lock
    Tn, Hn, Wn = 2, 64, 96
    # original: a sharp dark/bright edge (a face profile) in flat dim light
    orig = torch.full((Tn, Hn, Wn, 3), 0.15)
    orig[:, :, 48:] = 0.35
    # generated, half resolution: the same edge but blurry, and the new light brightened the dark side
    # more than the face (so the original's step must not be pasted on at its old contrast)
    gen = torch.full((Tn, Hn // 2, Wn // 2, 3), 0.45)
    gen[:, :, 24:] = 0.6
    gen = torch.nn.functional.avg_pool2d(gen.permute(0, 3, 1, 2), 5, 1, 2,
                                         count_include_pad=False).permute(0, 2, 3, 1)
    out = face_detail_lock(orig, gen, torch.ones(Tn, Hn, Wn), grow=0, feather=0.0, method="guided", radius=4, eps=1e-4)
    row = out[0, 32, :, 0]
    assert abs(row[:40].mean() - 0.45) < 0.03 and abs(row[56:].mean() - 0.6) < 0.03, row   # generated light
    up = torch.nn.functional.interpolate(gen.permute(0, 3, 1, 2), size=(Hn, Wn), mode="bicubic",
                                         align_corners=False)[0, 0, 32]
    assert row[48] - row[47] > 3 * up.diff().abs().max(), (row[48] - row[47], up.diff().abs().max())  # sharp edge
    assert row.min() > 0.4 and row.max() < 0.65, (row.min(), row.max())                    # no outline
    # reversed contrast (a head against a window that the new light made darker than the head): the
    # original's edge must not be pasted on inverted; the generated edge is kept
    rev = torch.full((Tn, Hn // 2, Wn // 2, 3), 0.6)
    rev[:, :, 24:] = 0.3
    rev = torch.nn.functional.avg_pool2d(rev.permute(0, 3, 1, 2), 5, 1, 2,
                                         count_include_pad=False).permute(0, 2, 3, 1)
    out = face_detail_lock(orig, rev, torch.ones(Tn, Hn, Wn), grow=0, feather=0.0, method="guided", radius=4, eps=1e-4)
    up = torch.nn.functional.interpolate(rev.permute(0, 3, 1, 2), size=(Hn, Wn), mode="bicubic",
                                         align_corners=False, antialias=True).permute(0, 2, 3, 1).clamp(0, 1)
    assert (out[0, 32, 40:56, 0] - up[0, 32, 40:56, 0]).abs().max() < 0.03


def test_light_guard_removes_invented_light_only():
    from openrelight.guard import light_guard
    Tn, Hn, Wn = 2, 48, 64
    orig = torch.full((Tn, Hn, Wn, 3), 0.2)
    lmap = torch.zeros(Tn, Hn, Wn, 3)
    lmap[:, :, 40:] = 1.0                       # the rig's light lands on the right (map peak)
    gen = orig.clone()
    gen[:, :, 40:] = 0.6                        # real new light
    gen[:, :20, 5:20] = 0.7                     # invented beam where no light lands
    gen[:, 30:, :20] = 0.1                      # darkening is allowed
    out = light_guard(gen, orig, lmap, softness=1.0)
    assert (out[:, :, 46:] - gen[:, :, 46:]).abs().max() < 0.01           # real light kept
    assert out[:, 4:16, 8:16].mean() < 0.23                                # invented beam capped near the original
    assert (out[:, 34:, 2:16] - gen[:, 34:, 2:16]).abs().max() < 0.01     # darker areas untouched


def test_light_stage_json_is_read_as_written():
    # what web/js/lightstage.js writes: keys in seconds with every animated field, hex gels, effects
    ui = {"version": 1, "duration_sec": 4, "ambient": {"kelvin": 7000, "intensity_ev": -3},
          "look": {"preset": "neo_noir", "film": "", "grade": ""}, "user_prompt": "rainy night",
          "lights": [{"id": "key0", "name": "主光", "type": "spot", "mode": "generate", "enabled": True,
                      "color": {"hex": "#ff5533"}, "softness": 0.2, "cone": {"angle": 30, "feather": 0.4},
                      "shadow": True, "motion": {"type": "strobe", "rate": 2, "duty": 0.5, "phase": 0}, "prompt": "",
                      "keys": [{"sec": 0, "azimuth": 90, "elevation": 10, "distance": 2, "intensity_ev": 1,
                                "kelvin": 5600, "ease": "linear"},
                               {"sec": 2, "azimuth": 0, "elevation": 10, "distance": 2, "intensity_ev": 1,
                                "kelvin": 5600, "ease": "smooth"}]}]}
    rig = rigmod.load_rig(json.dumps(ui))
    mid = rigmod.evaluate_light(rig["lights"][0], 24, 97, 24.0)             # 1 s = half way, linear
    assert abs(mid["azimuth"] - 45) < 1e-6
    end = rigmod.evaluate_light(rig["lights"][0], 72, 97, 24.0)             # held after the last key
    assert end["azimuth"] == 0
    assert mid["color"][0] > 0.9 and mid["color"][2] < 0.3                  # the gel, not 5600 K
    vals = [rigmod.evaluate_light(rig["lights"][0], f, 97, 24.0)["intensity"] for f in range(24)]
    assert min(vals) == 0.0 and max(vals) > 1.0                             # strobe on and off
    text = rig_to_prompt(rig, 97, 24.0)
    assert "rainy night" in text and "0.0s to 2.0s" in text


def test_template_prompts_follow_the_rig():
    rig = rigmod.load_rig(None, preset="sweep_right_to_front")
    first, last = lm.keyframe_prompt(rig, 0, 121, 24.0), lm.keyframe_prompt(rig, 120, 121, 24.0)
    assert "toward frame-right" in first and "Key light" in first and first != last
    assert "stops" in first and "K" in first and "blinding" not in first          # units, not adjectives
    for p in (first, last):
        assert "own style and medium" in p and lm.MOCKUP_NOTE in p
        assert "spotlight beam" not in p                      # the wording that made Qwen draw a lamp cone
        assert "skin" not in p and "pores" not in p           # generic: no realism wording for stylised clips
    video = lm.video_prompt(rig, 121, 24.0)
    assert video.startswith("Keep the content") and "0.0s to 3.0s" in video and "skin" not in video
    # neutral by default: no style and no ambient sentence unless the user set them
    plain = rigmod.load_rig({"lights": [{"id": "k", "type": "area", "keys": [{"t": 0, "azimuth": -40, "elevation": 20}]}]})
    for p in (lm.keyframe_prompt(plain, 0, 24, 24.0), lm.video_prompt(plain, 24, 24.0)):
        assert "ambient" not in p.lower() and "noir" not in p and "darker than" not in p


def _lit_sphere_from_right():
    # a matte sphere lit from screen right (+x) and a little above, at eye level in front of the camera
    moge, rgb, alpha = sphere_scene(light_dir=[0.9, -0.2, -0.35])
    geo = prepare_geometry(moge, alpha=alpha, temporal_sigma=0)
    return geo, rgb, alpha


def test_detect_original_light_and_difference_mockup():
    import copy

    from openrelight.detect import detect_original_lights
    from openrelight.shading import relight_mockup
    geo, rgb, alpha = _lit_sphere_from_right()
    rig, report = detect_original_lights(rgb, geo)
    assert len(rig["lights"]) == 1 and rig["lights"][0]["original"], report
    k = rig["lights"][0]["keys"][0]
    assert 30 < k["azimuth"] < 120, report                              # from the right
    m = alpha[0] > 0.5

    def halves(img):
        return _halves(img, 0, alpha)

    # left as detected: the picture does not change
    _, same = relight_mockup(rgb, geo, rig, T, 24.0, soften=0.0)
    assert (same[0][m] - rgb[0][m]).abs().mean() < 0.02
    # moved to the left: the right half darkens, the left half brightens
    moved = copy.deepcopy(rig)
    moved["lights"][0]["keys"][0]["azimuth"] = -k["azimuth"]
    _, mv = relight_mockup(rgb, geo, moved, T, 24.0, soften=0.0)
    (l0, r0), (l1, r1) = halves(rgb), halves(mv)
    assert l1 > l0 * 1.2 and r1 < r0 * 0.9, (l0, r0, l1, r1)
    # deleted: what it lit gets darker
    gone = copy.deepcopy(rig)
    gone["removed_originals"] = [gone["lights"].pop(0)["detected"]]
    _, dl = relight_mockup(rgb, geo, gone, T, 24.0, soften=0.0)
    assert halves(dl)[1] < r0 * 0.8                                       # the unexplained ambient share stays
    # prompts: unchanged = not mentioned, moved = a change, deleted = switched off
    assert "existing main light" not in lm.keyframe_prompt(rig, 0, T, 24.0)
    assert "existing main light, changed" in lm.keyframe_prompt(moved, 0, T, 24.0)
    assert "switched off" in lm.video_prompt(gone, T, 24.0)


def test_reference_lights_two_colours():
    from openrelight.detect import detect_reference_lights
    moge, _, alpha = sphere_scene()
    N = moge["normal"][0]
    rgb = torch.full(N.shape, 0.02)
    for d, col in (([-0.9, -0.15, -0.4], [1.0, 0.15, 0.85]), ([0.9, -0.15, -0.4], [0.1, 0.8, 1.0])):
        L = torch.tensor(d) / torch.tensor(d).norm()
        rgb = rgb + (N * L).sum(-1).clamp_min(0)[..., None] * torch.tensor(col) * 0.8
    rgb = rgb.clamp(0, 1) ** (1 / 2.2)
    geo = prepare_geometry(moge, alpha=alpha, temporal_sigma=0)
    rig, report = detect_reference_lights(rgb, geo)
    lights = rig["lights"]
    assert len(lights) == 2, report
    by_side = sorted(lights, key=lambda l: l["keys"][0]["azimuth"])
    left, right = by_side
    assert left["keys"][0]["azimuth"] < -30 and right["keys"][0]["azimuth"] > 30, report
    hexes = [l["color"].get("hex", "") for l in (left, right)]
    r, b = int(hexes[0][1:3], 16), int(hexes[0][5:7], 16)       # magenta: red and blue high
    g = int(hexes[1][3:5], 16)                                   # cyan: green high
    assert r > 180 and b > 150 and g > 150, (hexes, report)
    assert all(l["reference"] for l in lights)


def test_detect_reports_even_light():
    from openrelight.detect import detect_original_lights
    moge, _, alpha = sphere_scene()                                       # flat grey: no direction
    geo = prepare_geometry(moge, alpha=alpha, temporal_sigma=0)
    rig, report = detect_original_lights(torch.full((T, H, W, 3), 0.5), geo)
    assert rig["lights"] == [] and "even" in report


def test_offscreen_note():
    rig = rigmod.load_rig(None, preset="sweep_right_to_front")
    assert lm.offscreen_note(rig, 121, 24.0) == lm.OFFSCREEN_NOTE
    assert lm.OFFSCREEN_NOTE in lm.keyframe_brief(rig, 60, 121, 24.0)
    back = rigmod.load_rig({"lights": [{"id": "b", "type": "spot", "mode": "generate",
                                        "keys": [{"t": 0, "azimuth": 180, "elevation": 5, "distance": 2.0}]}]})
    assert lm.offscreen_note(back, 24, 24.0) == ""


def test_keyframe_brief_follows_the_sweep():
    rig = rigmod.load_rig(None, preset="sweep_right_to_front")
    first = lm.keyframe_brief(rig, 0, 121, 24.0)
    last = lm.keyframe_brief(rig, 120, 121, 24.0)
    assert "right edge of the frame" in first and "split lighting" in first, first
    assert "from the camera position" in last and "head-on" in last, last


def test_clean_lm_output_keeps_only_the_answer():
    answer = "Relight the input image with a cool spotlight from the right."
    assert lm.clean_lm_output("</think>\n\n" + answer) == answer
    reasoning = "The user's instruction is in English.\n\nLet me analyze.\n\nDraft: ...\n\n\n" + answer
    assert lm.clean_lm_output(reasoning) == answer
    assert lm.clean_lm_output('{"' + answer + '"}') == answer


class _StubLM:
    """Stands in for the Qwen3-VL clip: returns a canned director answer, or one of a list call by call."""
    def __init__(self, text):
        self.text = text
        self.asked = []

    def tokenize(self, text, images=None):
        self.asked.append(text)
        return {"prompt": text}

    def generate(self, tokens, **kw):
        return [0]

    def decode(self, ids):
        return self.text if isinstance(self.text, str) else self.text[len(self.asked) - 1]


def _panel_rig(user_prompt=""):
    return rigmod.load_rig({"user_prompt": user_prompt, "lights": [{
        "id": "k", "type": "area", "shape": "disc", "size": 0.5, "softness": 0.4,
        "keys": [{"sec": 0, "azimuth": -70, "elevation": 0, "distance": 1.5}]}]})


def test_director_keeps_light_sources_out_of_frame():
    answer = ("KEYFRAME 1: Soft warm light from frame left lights the frame-left side of the subject; the frame-right "
              "side falls into shadow and the background stays dim.\n"
              "VIDEO: From 0.0s to 2.0s soft warm light from frame left lights the frame-left side; the other side "
              "stays in shadow and the background dim.")
    frames = torch.zeros(1, 8, 8, 3)
    kf, video, _, notes = lm.direct(_StubLM(answer), _panel_rig(), frames, [0], 49, 24.0, mockup=False)
    assert not notes, notes
    assert lm.NO_SOURCE_NOTE in kf[0] and lm.NO_SOURCE_NOTE in video
    # the brief describes the light, not a fixture to put in the picture
    brief = lm.stage_brief(_panel_rig(), 49, 24.0)
    assert "source of broad light" in brief and "panel" not in brief
    # the user asking to see the light source drops the note
    kf, video, _, _ = lm.direct(_StubLM(answer), _panel_rig("燈具入鏡，看得到燈"), frames, [0], 49, 24.0, mockup=False)
    assert lm.NO_SOURCE_NOTE not in kf[0] and lm.NO_SOURCE_NOTE not in video
    assert not lm.wants_visible_source(_panel_rig("黑色深邃的環境空間，水波流動的光感"))
    # a keyframe that points at another ("unchanged from Keyframe 1") loses that opening: Qwen sees one image
    two = ["LIGHT 1: soft warm light from frame left.\nVIDEO: From 0.0s to 2.0s soft warm light from frame left lights "
           "the face and the frame-right side stays in soft shadow.",
           "KEYFRAME 1: Soft warm light from frame left lights the frame-left side of the face; the frame-right "
           "side falls into shadow and the background stays dim.",
           "KEYFRAME 1: The lighting remains unchanged from Keyframe 1: the frame-left side of the face is lit and the "
           "frame-right side is in soft shadow, the background dim."]
    stub = _StubLM(two)
    kf, _, raw, _ = lm.direct(stub, _panel_rig(), torch.zeros(2, 8, 8, 3), [0, 48], 49, 24.0, mockup=False)
    assert "keyframe 1" not in kf[1].lower() and "The frame-left side of the face is lit" in kf[1], kf[1]
    # the plan comes first, then each keyframe alone with its one image, and the raw text keeps them apart
    assert len(stub.asked) == 3 and "soft warm light from frame left" in stub.asked[1] and "KEYFRAME 2:" in raw
    assert stub.asked[0].count("<|image_pad|>") == 2 and stub.asked[2].count("<|image_pad|>") == 1


def test_green_screen_key_and_restore():
    from openrelight.key import key_composite, restore_background
    img = torch.zeros(2, 40, 60, 3)
    img[..., 1] = 0.8                                   # green screen
    img[:, 10:30, 20:40] = torch.tensor([0.5, 0.45, 0.4])   # a grey-brown subject
    plate = torch.zeros(1, 20, 30, 3)
    plate[..., 2] = 1.0                                 # blue sky
    comp, alpha = key_composite(img, plate, erode=0, matte_blur=0)
    assert alpha[:, 15:25, 25:35].min() > 0.99 and alpha[:, :5, :5].max() < 0.01
    assert comp[0, 2, 2, 2] > 0.99 and abs(float(comp[0, 20, 30, 0]) - 0.5) < 1e-3
    # after a relight the plate goes back over the screen area, the subject keeps its new light
    relit = torch.full_like(img, 0.2)
    back = restore_background(relit, alpha, plate)
    assert back[0, 2, 2, 2] > 0.99 and abs(float(back[0, 20, 30, 0]) - 0.2) < 1e-3


def test_lm_light_reading_maps_and_merges():
    from openrelight.read_light import parse_reading, reading_to_rig
    raw = ('{"lights": [{"role": "key", "horizontal": "behind", "vertical": "level", "colour": "green", "hardness": "soft", '
           '"strength": 1.0, "source": "green screen"}, {"role": "fill", "horizontal": "front", "vertical": "level", '
           '"colour": "warm", "hardness": "soft", "strength": 0.3, "source": "ceiling tubes"}, {"role": "back", '
           '"horizontal": "behind", "vertical": "level", "colour": "green", "strength": 0.5, "source": "screen reflection"}],'
           ' "summary": "x"}')
    rig, report = reading_to_rig(parse_reading("Here: " + raw))
    keys = [(l["keys"][0]["azimuth"], l["keys"][0]["elevation"]) for l in rig["lights"]]
    assert keys == [(180, 10), (0, 75)], keys          # the duplicate screen dropped, the ceiling tubes overhead
    assert rig["lights"][0]["color"].get("hex") and all(l["original"] for l in rig["lights"])
    assert rig["lights"][1]["keys"][0]["intensity_ev"] < -1
    assert parse_reading("no json here") is None


def test_keyframes_follow_the_light():
    from openrelight.rig import load_rig, plan_keyframes
    still = load_rig({"lights": [{"id": "k", "type": "area", "keys": [{"sec": 0, "azimuth": 45, "elevation": 30}]}]})
    assert plan_keyframes(still, 48, 24.0) == [0, 47]                     # nothing changes: first and last only
    # the user's red sweep (2026-10-30): left to right by 1.27 s, a flash at 1.47 s, dim at 2.08 s, on a 2 s clip
    red = load_rig({"lights": [{"id": "r", "type": "area", "color": {"hex": "#ff0000"}, "keys": [
        {"sec": 0, "azimuth": -38, "elevation": 31, "intensity_ev": -3.2},
        {"sec": 1.27, "azimuth": 57, "elevation": 37, "intensity_ev": -2, "ease": "linear"},
        {"sec": 1.473, "azimuth": 57, "elevation": 37, "intensity_ev": 2.7},
        {"sec": 2.083, "azimuth": 57, "elevation": 37, "intensity_ev": -2, "ease": "linear"}]}]})
    plan = plan_keyframes(red, 48, 24.0)
    assert len(plan) == 4 and plan[0] == 0 and plan[-1] == 47, plan
    assert any(8 <= f <= 24 for f in plan), plan                          # one inside the sweep (0-30), not only its ends
    assert any(28 <= f <= 38 for f in plan), plan                         # one by the sweep's end and the flash
    # start -> end over the whole clip: keyframes at thirds of the move, evenly apart
    sweep = load_rig({"lights": [{"id": "s", "type": "spot", "keys": [
        {"sec": 0, "azimuth": -45, "elevation": 30, "ease": "linear"}, {"sec": 47 / 24, "azimuth": 45, "elevation": 30}]}]})
    plan = plan_keyframes(sweep, 48, 24.0)
    assert len(plan) == 4 and all(abs(f - g) <= 2 for f, g in zip(plan, [0, 16, 31, 47])), plan
    # a move that ends early keeps its end as a keyframe (it shows at its time), one more splits the move
    early = load_rig({"lights": [{"id": "e", "type": "spot", "keys": [
        {"sec": 0, "azimuth": -45, "elevation": 30, "ease": "linear"}, {"sec": 1.0, "azimuth": 45, "elevation": 30}]}]})
    plan = plan_keyframes(early, 48, 24.0)
    assert len(plan) == 4 and any(abs(f - 24) <= 1 for f in plan) and any(8 <= f <= 16 for f in plan), plan
    # a light that comes on at 0.5 s and goes out at 1.5 s: keyframes where it does
    blink = load_rig({"lights": [{"id": "b", "type": "spot", "start_sec": 0.5, "end_sec": 1.5,
                                  "keys": [{"sec": 0, "azimuth": 0, "elevation": 30}]}]})
    plan = plan_keyframes(blink, 48, 24.0)
    assert any(abs(f - 12) <= 1 for f in plan) and any(abs(f - 36) <= 1 for f in plan), plan


def test_three_keyframes_keep_the_middle_of_a_move():
    from openrelight.rig import load_rig, plan_keyframes
    # the user's sweep held to the end of the clip: the slot next to the hold goes to the middle of the sweep
    early = load_rig({"lights": [{"id": "e", "type": "spot", "keys": [
        {"sec": 0, "azimuth": -45, "elevation": 30, "ease": "linear"}, {"sec": 1.27, "azimuth": 57, "elevation": 37},
        {"sec": 2.083, "azimuth": 57, "elevation": 37}]}]})
    plan = plan_keyframes(early, 48, 24.0, max_count=3)
    assert len(plan) == 3 and plan[0] == 0 and plan[-1] == 47 and 12 <= plan[1] <= 18, plan
    # a light that only switches on cannot be cut in two: its keyframe stays where it comes on
    on = load_rig({"lights": [{"id": "b", "type": "spot", "start_sec": 1.0, "keys": [{"sec": 0, "azimuth": 0, "elevation": 30}]}]})
    plan = plan_keyframes(on, 48, 24.0, max_count=3)
    assert len(plan) == 3 and abs(plan[1] - 24) <= 1, plan


def _sweep_note(note="", **extra):
    return dict({"id": "r", "name": "紅", "type": "area", "shape": "disc", "size": 1.2, "softness": 0.5, "prompt": note,
                 "color": {"hex": "#ff0000"}, "keys": [
                     {"sec": 0, "azimuth": -43, "elevation": 29, "intensity_ev": -3.2},
                     {"sec": 1.27, "azimuth": 57, "elevation": 37, "intensity_ev": -2},
                     {"sec": 2.0, "azimuth": 57, "elevation": 37, "intensity_ev": -2}]}, **extra)


def test_levels_are_told_against_the_lights_left():
    gone = {"id": "original0", "original": True, "keys": [{"sec": 0, "azimuth": 0, "elevation": 10, "intensity_ev": 0.0}]}
    kept = {"id": "original1", "name": "天光", "type": "area", "original": True,
            "keys": [{"sec": 0, "azimuth": 64, "elevation": 11, "intensity_ev": -3.2}]}
    kept["detected"] = dict(kept)
    rig = rigmod.load_rig({"lights": [kept, _sweep_note()], "removed_originals": [gone]})
    brief = lm.stage_brief(rig, 48, 24.0)
    assert "dim:" not in brief and "stronger than the scene's main light" in brief, brief
    # nothing removed: the scale stays the original main light's, and a low light is still low
    rig = rigmod.load_rig({"lights": [dict(kept, detected=dict(kept, keys=[dict(kept["keys"][0], intensity_ev=0.0)])), _sweep_note()]})
    assert "dim:" in lm.stage_brief(rig, 48, 24.0)


def test_light_brief_tells_the_beam_and_the_whole_path():
    # the user's words come first, a searchlight on a panel is a beam, a hold is told as a hold
    rig = rigmod.load_rig({"lights": [_sweep_note("紅色探照燈光束從船艙外打進室內")]})
    brief = lm.stage_brief(rig, 48, 24.0)
    assert brief.index("紅色探照燈") < brief.index("narrow beam") and "broad light" not in brief, brief
    assert "then it holds there" in brief and "from frame-left to frame-right" in brief, brief
    assert "moves quickly" in brief and "about two-thirds of the way through the clip" in brief, brief
    # no seconds and no keyframe positions: the plan and the video prompt tell a path, not a timetable
    assert not re.search(r"\d\.\d+s\b", brief) and "Keyframe 1" not in brief, brief
    assert "its pool falls roughly on the frame-left part" in lm.director_facts(rig, 0, 48, 24.0)
    # a light that goes out and comes back ends where it started: it still moved
    back = rigmod.load_rig({"lights": [{"id": "o", "type": "spot", "keys": [
        {"sec": 0, "azimuth": -60, "elevation": 20}, {"sec": 1, "azimuth": 60, "elevation": 20},
        {"sec": 2, "azimuth": -60, "elevation": 20}]}]})
    brief = lm.stage_brief(back, 48, 24.0)
    assert "from frame-left to frame-right to frame-left" in brief, brief
    # when a light comes on and goes out is told in words too
    blink = rigmod.load_rig({"lights": [dict(_sweep_note(), start_sec=0.8, end_sec=1.1)]})
    brief = lm.stage_brief(blink, 48, 24.0)
    assert "comes on about a third of the way through the clip" in brief, brief
    assert "goes out about halfway through the clip" in brief and not re.search(r"\d\.\d+s\b", brief), brief


def test_keyframes_open_with_where_the_light_is_now():
    rig = rigmod.load_rig({"lights": [_sweep_note("紅色探照燈光束")]})
    assert lm.moment_opening(rig, 0, 48, 24.0) == "In this frame the vivid red beam lands only on the frame-left part of the scene, where"
    assert "the frame-right part" in lm.moment_opening(rig, 47, 48, 24.0)
    # a flood is told by its side; a beam far weaker than it does not lead
    flood = {"id": "f", "type": "area", "color": {"hex": "#8040ff"}, "keys": [{"sec": 0, "azimuth": 141, "elevation": -11, "intensity_ev": 4}]}
    rig = rigmod.load_rig({"lights": [flood, _sweep_note("紅色探照燈光束")]})
    opening = lm.moment_opening(rig, 0, 48, 24.0)
    assert "strongest toward the frame-right of the picture" in opening and "fills the background" in opening, opening
    weak = dict(flood, keys=[dict(flood["keys"][0], intensity_ev=0)])
    assert "fills the background" not in lm.moment_opening(rigmod.load_rig({"lights": [weak]}), 0, 48, 24.0)
    assert lm.moment_opening(rigmod.load_rig({"lights": []}), 0, 48, 24.0) == ""
    # the opening reaches the prompt with what the LM adds, cut back to a whole sentence at the token limit
    answers = ["LIGHT 1: red.\nVIDEO: From 0.0s to 2.0s the red beam crosses the room from frame left to frame right.",
               "the man's shoulders get brighter than the wall behind him, with a hard edge. The rest of the room stays as"]
    rig = rigmod.load_rig({"lights": [_sweep_note("紅色探照燈光束")]})
    kf, _, _, _ = lm.direct(_StubLM(answers), rig, torch.zeros(1, 8, 8, 3), [0], 48, 24.0, mockup=False)
    assert "frame-left part of the scene, where the man's shoulders" in kf[0] and "The rest of the room" not in kf[0], kf[0]


def test_key_pattern_is_only_for_faces_the_camera_sees():
    rig = rigmod.load_rig({"lights": [{"id": "k", "type": "area", "keys": [{"sec": 0, "azimuth": 45, "elevation": 40}]}]})
    text = lm.director_facts(rig, 0, 48, 24.0)
    assert "On any face the camera can see" in text and "On the people" not in text, text
    assert "On any face" not in lm.director_facts(rig, 0, 48, 24.0, scene="People: three men\nFaces: none")
    assert "On any face" in lm.director_facts(rig, 0, 48, 24.0, scene="People: one woman\nFaces: large")
    assert "On any face" in lm.keyframe_request(rig, 0, 48, 24.0, "LIGHT 1: x.", scene="Faces: large")


def test_a_beam_the_user_asks_for_is_not_forbidden():
    answer = ("KEYFRAME 1: A red beam comes in through the window and lands on the console, whose top gets much brighter "
              "than the wall around it.\nVIDEO: From 0.0s to 2.0s the red beam crosses the console from frame left to right.")
    frames = torch.zeros(1, 8, 8, 3)
    asked = rigmod.load_rig({"lights": [_sweep_note("紅色探照燈光束")]})
    kf, video, _, _ = lm.direct(_StubLM(answer), asked, frames, [0], 48, 24.0, mockup=False)
    assert "bulb or sun for them" in kf[0] and "or beam" not in kf[0] and "or beam" not in video, kf[0]
    assert "glow, bloom" not in kf[0] and "tints" not in kf[0], kf[0]
    plain = rigmod.load_rig({"lights": [_sweep_note()]})
    kf, _, _, _ = lm.direct(_StubLM(answer), plain, frames, [0], 48, 24.0, mockup=False)
    assert lm.NO_SOURCE_NOTE in kf[0] and "no haze, fog, glow, bloom or milky layer" in kf[0], kf[0]
    # the look of the rest follows the new lighting, not the original
    assert "follow the new lighting" in kf[0] and "keeps its own colour" not in kf[0], kf[0]


def test_the_users_ambient_goes_on_every_prompt():
    answer = ("LIGHT 1: red.\nKEYFRAME 1: the wall gets much brighter than what is around it, with a hard edge and a shadow.\n"
              "VIDEO: A red searchlight moves across the room from left to right and settles there in the tense dark.")
    frames = torch.zeros(1, 8, 8, 3)
    dark = rigmod.load_rig({"lights": [_sweep_note()], "ambient": {"kelvin": 6500, "intensity_ev": -2.5}})
    kf, video, _, _ = lm.direct(_StubLM(answer), dark, frames, [0], 48, 24.0, mockup=False)
    assert "clearly darker than the original" in kf[0] and "clearly darker than the original" in video, kf[0]
    plain = rigmod.load_rig({"lights": [_sweep_note()]})
    kf, _, _, _ = lm.direct(_StubLM(answer), plain, frames, [0], 48, 24.0, mockup=False)
    assert "Everywhere none of these lights reach" not in kf[0], kf[0]


def test_lm_rules_name_no_material_of_their_own():
    # the footage decides which things take the light in their own way: the rules must not assume glass, metal or windows
    from openrelight.read_light import READ_SYSTEM
    for text in (lm.DIRECTOR_SYSTEM, lm.DIRECTOR_KEYFRAME):
        low = text.lower()
        assert "glass" not in low and "metal" not in low and "pane" not in low, text
    assert '"materials"' in READ_SYSTEM and '"shiny"' not in READ_SYSTEM
    # new light is part of the place, in both the plan and each keyframe
    for text in (lm.DIRECTOR_SYSTEM, lm.DIRECTOR_KEYFRAME):
        assert "never across them" in text and "shadow side" in text, text


def test_video_prompt_is_the_lm_prose_with_no_timetable():
    answers = ["MOOD: tense.\nLIGHT 1: 紅. Real source: a searchlight. Comes through: directly. Look: hard. Path: a quick "
               "sweep. Answers: the console.\nVIDEO: A hard red searchlight sweeps quickly across the room and settles to "
               "the right, its reflection sliding along the glass while the crew stand still in the tense dark.",
               "the console's top catches a hard red edge against the dark room, and its metal throws the light back."]
    rig = rigmod.load_rig({"lights": [_sweep_note("紅色探照燈光束")]})
    stub = _StubLM(answers)
    kf, video, _, notes = lm.direct(stub, rig, torch.zeros(1, 8, 8, 3), [0], 48, 24.0, mockup=False)
    assert not notes and "its reflection sliding along the glass" in video and not re.search(r"\d\.\d+s", video), (notes, video)
    # a still keyframe gets the mood and what answers the light, not the path, and the path's words stay out of it
    assert "MOOD: tense." in stub.asked[1] and "Answers: the console." in stub.asked[1], stub.asked[1]
    assert "a quick sweep" not in stub.asked[1] and "its pool falls roughly on the frame-left part" in stub.asked[1]


def test_scene_fields_reach_the_director_as_lines():
    from openrelight.read_light import reading_to_rig, scene_text
    assert scene_text({"openings": "five windows across the far wall", "people": " "}) == \
        "Openings: five windows across the far wall"
    assert scene_text("a room") == "a room" and scene_text(None) == ""
    assert scene_text({"people": [{"location": "centre", "description": "a man"}, {"location": "left", "description": "a boy"}]}) == \
        "People: centre, a man; left, a boy"
    rig, _ = reading_to_rig({"lights": [{"horizontal": "front", "strength": 1.0}], "scene": {"far side": "a wall"}})
    assert rig["scene"] == "Far side: a wall"
    plan = lm.parse_plan("SCENE: a room.\nLIGHT 1: red. Reach: beam.\n**LIGHT 2**: white.\nKEYFRAME 1: words")
    assert plan == "SCENE: a room.\nLIGHT 1: red. Reach: beam.\nLIGHT 2: white.", plan


if __name__ == "__main__":
    failed = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("PASS", name)
            except Exception as e:  # noqa: BLE001
                failed += 1
                print("FAIL", name, type(e).__name__, e)
    sys.exit(1 if failed else 0)
