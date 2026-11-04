"""OpenRelight ComfyUI nodes (V3 API)."""

import hashlib
import json

import torch
from typing_extensions import override

import comfy.model_management
import comfy.utils
from comfy_api.latest import ComfyExtension, io, ui

from .openrelight import lm
from .openrelight import rig as rigmod
from .openrelight.fit import FRAME_GRIDS, fit_pad, unpad
from .openrelight.geometry import depth_preview, normal_preview, prepare_geometry
from .openrelight.detect import detect_original_lights
from .openrelight.clip import clip_sheet

LightRigType = io.Custom("OR_LIGHT_RIG")
GeometryType = io.Custom("OR_GEOMETRY")
MoGeGeometry = io.Custom("MOGE_GEOMETRY")

CATEGORY = "OpenRelight"


def _out(t):
    return t.to(device=comfy.model_management.intermediate_device(),
                dtype=comfy.model_management.intermediate_dtype())


class LightRig(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="OR_LightRig",
            display_name="Light Rig (OpenRelight)",
            category=CATEGORY,
            description="The light stage: every light of the shot, set on the sphere editor (direction, motion, colour, "
                        "hardness, effects). Connect Detect Original Light to start from the clip's own light. "
                        "An output node, so it can be run on its own (select it, ▶) to analyse the clip without generating.",
            is_output_node=True,
            inputs=[
                io.Combo.Input("preset", options=["custom"] + rigmod.preset_names(), default="custom",
                               tooltip="'custom' = what the light stage shows. A preset name loads that preset instead."),
                io.String.Input("rig_json", multiline=True, default="",
                                tooltip="Written by the light stage. Empty + 'detected' connected = start from the detected light."),
                LightRigType.Input("detected", optional=True,
                                   tooltip="Detect Original Light. Used while the light stage is still empty; the editor "
                                           "also offers it to apply after each analysis run."),
                LightRigType.Input("reference_lights", optional=True,
                                   tooltip="Optional: another light rig to add as reference lights. They join the "
                                           "rig in this run and the original light is switched off."),
            ],
            outputs=[LightRigType.Output(display_name="light_rig"), io.String.Output(display_name="rig_json")],
        )

    @classmethod
    def execute(cls, preset, rig_json, detected=None, reference_lights=None) -> io.NodeOutput:
        if preset == "custom" and not (rig_json or "").strip() and detected is not None:
            rig = rigmod.load_rig(detected)
        else:
            rig = rigmod.load_rig(rig_json, preset=None if preset == "custom" else preset)
        # a new reference is applied in this same run (the editor only hears of it after the run): its lights
        # replace earlier reference lights and the original light is switched off. The same reference again
        # (same signature as the editor keeps) leaves the stage as the user edited it.
        ref = (reference_lights or {}).get("lights") or []
        if ref:
            sig = rigmod.reference_signature(ref)
            if rig.get("reference_sig") != sig:
                rig["lights"] = [l for l in rig["lights"] if not l.get("reference")]
                for light in rig["lights"]:
                    if light.get("original"):
                        light["enabled"] = False
                rig["lights"] += [dict(l, reference=True) for l in rigmod.load_rig({"lights": ref})["lights"]]
                if reference_lights.get("ambient"):
                    rig["ambient"] = dict(reference_lights["ambient"])
                rig["reference_sig"] = sig
        return io.NodeOutput(rig, json.dumps(rig, indent=2, ensure_ascii=False))


class GeometryPrep(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="OR_GeometryPrep",
            display_name="Relight Geometry Prep (OpenRelight)",
            category=CATEGORY,
            description="Stabilise per-frame MoGe geometry over time and find the subject anchor the lights orbit.",
            inputs=[
                MoGeGeometry.Input("moge_geometry"),
                io.Mask.Input("subject_mask", optional=True, tooltip="Subject matte (white = subject). Lights aim at and anchor to it."),
                io.Float.Input("temporal_sigma", default=1.5, min=0.0, max=6.0, step=0.1,
                               tooltip="Frames of temporal smoothing for normals/depth. 0 = off."),
                io.Boolean.Input("stabilize_scale", default=True, tooltip="Remove MoGe's per-frame depth scale jitter."),
            ],
            outputs=[GeometryType.Output(display_name="geometry"),
                     io.Image.Output(display_name="normals"), io.Image.Output(display_name="depth")],
        )

    @classmethod
    def execute(cls, moge_geometry, temporal_sigma, stabilize_scale, subject_mask=None) -> io.NodeOutput:
        device = comfy.model_management.get_torch_device()
        pbar = comfy.utils.ProgressBar(moge_geometry["points"].shape[0])
        geo = prepare_geometry(moge_geometry, alpha=subject_mask, temporal_sigma=temporal_sigma,
                               stabilize_scale=stabilize_scale, device=device,
                               progress=lambda i, n: pbar.update_absolute(i, n))
        return io.NodeOutput(geo, _out(normal_preview(geo)), _out(depth_preview(geo)))


PadInfoType = io.Custom("OR_LTX_PAD_INFO")


class FitPad(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="OR_FitPad",
            display_name="Fit & Pad for Video Model (OpenRelight)",
            category=CATEGORY,
            description="Scale the clip to about N megapixels and pad to multiples of 32 with edge pixels, instead of "
                        "centre-cropping. The full picture is kept, so Unpad puts the video model's result back on the "
                        "original pixel for pixel (a crop + stretch shifts everything and shows as ghosting).",
            inputs=[
                io.Image.Input("images"),
                io.Float.Input("megapixels", default=0.5, min=0.0, max=4.0, step=0.05, tooltip="0 = keep the size."),
                io.Int.Input("multiple", default=32, min=8, max=128, step=8),
                io.Combo.Input("frames", options=list(FRAME_GRIDS), default="keep",
                               tooltip="Pad the clip with its last frame to the model's frame grid (MiniMax H3: "
                                       "17k+5). Unpad cuts the result back, so every frame of the clip is kept."),
            ],
            outputs=[io.Image.Output(display_name="images"), io.Int.Output(display_name="width"),
                     io.Int.Output(display_name="height"), io.Int.Output(display_name="length"),
                     PadInfoType.Output(display_name="pad_info")],
        )

    @classmethod
    def execute(cls, images, megapixels, multiple, frames="keep") -> io.NodeOutput:
        x, w, h, n, info = fit_pad(images, megapixels, multiple, frames)
        return io.NodeOutput(_out(x), w, h, n, info)


class Unpad(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="OR_Unpad",
            display_name="Unpad (OpenRelight)",
            category=CATEGORY,
            description="Undo Fit & Pad on the video model's output: remove the padding and (optionally) scale back to "
                        "the original size.",
            inputs=[io.Image.Input("images"), PadInfoType.Input("pad_info"),
                    io.Boolean.Input("original_size", default=True,
                                     tooltip="Scale back to the original resolution. Off = keep the model's resolution.")],
            outputs=[io.Image.Output(display_name="images")],
        )

    @classmethod
    def execute(cls, images, pad_info, original_size) -> io.NodeOutput:
        return io.NodeOutput(_out(unpad(images, pad_info, original_size)))


class SourceClip(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="OR_SourceClip",
            display_name="Source Clip (OpenRelight)",
            category=CATEGORY,
            description="The clip to relight: every frame (no crop, no resampling) with its own frame rate. Shows the "
                        "clip in the middle of the light stage, and the light stage's timeline takes its length.",
            is_output_node=True,
            inputs=[io.Video.Input("video"),
                    io.Float.Input("max_seconds", default=10.0, min=0.0, max=60.0, step=0.5,
                                   tooltip="Use at most this much of the clip (0 = all of it). Trim the clip "
                                           "first (Trim Video) to choose which part."),
                    io.Int.Input("max_side", default=1920, min=256, max=8192, step=16,
                                 tooltip="Scale a larger clip down to this long side (a 4K clip in memory is ~10 GB; "
                                         "the relight works at 0.5-1 MP anyway).")],
            outputs=[io.Image.Output(display_name="images"), io.Int.Output(display_name="frame_count"),
                     io.Float.Output(display_name="fps"), io.Float.Output(display_name="seconds")],
        )

    @classmethod
    def execute(cls, video, max_seconds, max_side=1920) -> io.NodeOutput:
        comps = video.get_components()
        fps = float(comps.frame_rate) or 24.0
        images = comps.images
        if max_seconds > 0:
            images = images[:max(1, round(max_seconds * fps))]
        H, W = images.shape[1:3]
        if max(H, W) > max_side:
            s = max_side / max(H, W)
            w2, h2 = round(W * s / 2) * 2, round(H * s / 2) * 2
            images = torch.cat([comfy.utils.common_upscale(images[i:i + 16].movedim(-1, 1), w2, h2, "area", "disabled")
                                .movedim(1, -1) for i in range(0, images.shape[0], 16)])
        n = images.shape[0]
        # the light stage editor shows the clip from this (web/js/lightstage.js)
        sheet = clip_sheet(images, fps)
        return io.NodeOutput(_out(images), n, fps, n / fps, ui={"openrelight_clip": [json.dumps(sheet)]})


class DetectOriginalLight(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="OR_DetectOriginalLight",
            display_name="Detect Original Light (OpenRelight)",
            category=CATEGORY,
            description="The clip's own lights, read by the vision LM from three frames (windows, lamps, green-screen "
                        "spill, studio lights), as lights marked Original on the light stage, and the space they light (for "
                        "the Lighting Director). A starting point to check on the stage.",
            is_output_node=True,
            inputs=[GeometryType.Input("geometry", optional=True,
                                       tooltip="Optional (MoGe + Geometry Prep): refines the main light's angle on close-ups."),
                    io.Image.Input("images", tooltip="The clip (Source Clip)."),
                    io.Float.Input("fps", default=24.0, min=1.0, max=240.0,
                                   tooltip="The clip's frame rate; sets the light stage's timeline to the clip's length."),
                    io.Combo.Input("method", options=["LM", "LM + geometry", "geometry only"], default="LM",
                                   tooltip="LM: the vision LM reads up to three lights. + geometry: with MoGe geometry "
                                           "connected, its fit refines the main light's angle where they agree. Geometry "
                                           "only: one light from brightness against surface direction."),
                    io.Clip.Input("clip", optional=True, lazy=True,
                                  tooltip="CLIP Loader: qwen3vl_8b (type qwen_image), the one Qwen-Image uses.")],
            outputs=[LightRigType.Output(display_name="detected"), io.String.Output(display_name="lights",
                                                                               tooltip="The lights found, one per line."),
                     io.String.Output(display_name="scene", tooltip="The space as the LM read it (what is behind the people, "
                                                                    "windows, surfaces): for the Lighting Director.")],
        )

    @classmethod
    def check_lazy_status(cls, method="LM", **kwargs):
        # an unconnected input is absent here (connected but not run yet = None); asking ComfyUI for an
        # unconnected one fails the run (execute then says to connect it)
        return ["clip"] if method.startswith("LM") and "clip" in kwargs and kwargs["clip"] is None else []

    @classmethod
    def execute(cls, images, fps=24.0, method="LM", clip=None, geometry=None) -> io.NodeOutput:
        rig = {"lights": []}
        if geometry is not None:
            rig, _ = detect_original_lights(images, geometry)
        if method.startswith("LM"):
            if clip is None:
                raise ValueError("Detect Original Light: connect the CLIP Loader with qwen3vl_8b (type qwen_image) to clip.")
            from .openrelight.read_light import read_original_lights
            guess = None
            if geometry is not None and rig["lights"] and method == "LM + geometry":
                k = rig["lights"][0]["keys"][0]
                guess = (k["azimuth"], k["elevation"])
            T = images.shape[0]
            frames = _fit_long_side(images[[0, T // 2, T - 1]][..., :3].float(), 640)
            lm_rig, _, _ = read_original_lights(clip, frames, guess, close_up=float(rig.get("detect_strength", 0)) >= 1.0)
            if lm_rig is not None:
                rig = lm_rig
        elif geometry is None:
            raise ValueError("Detect Original Light: 'geometry only' needs MoGe geometry connected.")
        rig["duration_sec"] = round(images.shape[0] / fps, 2)
        rig["clip_id"] = _clip_id(images)
        # the lights found and their settings, one per line (no report: the user wants the settings, 2026-10-30)
        listed = "\n".join(_light_line(l) for l in rig["lights"]) or "No clear original light"
        # the light stage editor picks the rig up from this message (web/js/lightstage.js)
        return io.NodeOutput(rig, listed, rig.get("scene", ""),
                             ui={"text": [listed], "openrelight_rig": [json.dumps(rig, ensure_ascii=False)]})


def _light_line(light):
    k = light["keys"][0]
    c = light.get("color") or {}
    colour = c.get("hex") or f"{int(c.get('kelvin', k.get('kelvin', 5600)))}K"
    src = light.get("source") or ""
    return (f"{light.get('name', 'Original light')}{' (' + src + ')' if src else ''}: azimuth {round(k['azimuth'])}°, elevation "
            f"{round(k['elevation'])}° | {colour} | {k.get('intensity_ev', 0):+.1f} stops | "
            f"{'hard' if float(light.get('softness', 0.5)) < 0.4 else 'soft'} light")


class KeyframeSeeds(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="OR_KeyframeSeeds",
            display_name="Keyframe Seeds (OpenRelight)",
            category=CATEGORY,
            description="One seed per keyframe. Not happy with one keyframe: change its seed and run again; only that "
                        "keyframe is redrawn.",
            inputs=[io.Int.Input(f"seed{i}", default=424242 + i, min=0, max=2 ** 31 - 1) for i in range(4)],
            outputs=[io.Int.Output(display_name=f"seed{i}") for i in range(4)],
        )

    @classmethod
    def execute(cls, seed0, seed1, seed2, seed3) -> io.NodeOutput:
        return io.NodeOutput(seed0, seed1, seed2, seed3)


class KeyframeReview(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="OR_KeyframeReview",
            display_name="Keyframes (OpenRelight)",
            category=CATEGORY,
            description="The relit keyframes in one place. Select this node and press ▶ to make only the keyframes "
                        "(no video); redraw any one with its button. When they look right, press Run: the video is "
                        "made from these same keyframes (cached, not redrawn).",
            is_output_node=True,
            inputs=[io.Image.Input("kf0"), io.Image.Input("kf1", optional=True, lazy=True),
                    io.Image.Input("kf2", optional=True, lazy=True), io.Image.Input("kf3", optional=True, lazy=True),
                    io.String.Input("frames", default="", optional=True, tooltip="Keyframe Plan's frame list, for the labels."),
                    io.Float.Input("fps", default=24.0, min=1.0, max=240.0, optional=True),
                    io.Int.Input("count", default=4, min=1, max=4, optional=True,
                                 tooltip="Keyframe Plan's count: only that many keyframes are made, the other "
                                         "branches do not run.")],
            outputs=[io.Image.Output(display_name=f"kf{i}") for i in range(4)]
                    + [io.Image.Output(display_name="keyframes", tooltip="The keyframes made, as one batch.")],
        )

    @classmethod
    def check_lazy_status(cls, count=4, **kwargs):
        # only the keyframes the plan uses: the branches past `count` are never run (a still light needs two)
        return [f"kf{i}" for i in range(1, max(1, min(int(count), 4))) if f"kf{i}" in kwargs and kwargs[f"kf{i}"] is None]

    @classmethod
    def execute(cls, kf0, kf1=None, kf2=None, kf3=None, frames="", fps=24.0, count=4) -> io.NodeOutput:
        from comfy_execution.graph_utils import ExecutionBlocker
        shown = [kf0]
        for k in (kf1, kf2, kf3)[:max(0, int(count) - 1)]:
            if k is None:
                break
            shown.append(k)
        idx = [int(f) for f in (frames or "").split(",") if f.strip().lstrip("-").isdigit()]
        cards = []
        for i, k in enumerate(shown):
            saved = ui.PreviewImage(_out(k[:1]).float().cpu(), cls=cls).as_dict()["images"][0]
            f = idx[i] if i < len(idx) else None
            cards.append(dict(saved, index=i, frame=f, sec=None if f is None else round(f / fps, 2)))
        H, W = kf0.shape[1:3]
        batch = torch.cat([k[:1] if k.shape[1:3] == (H, W) else comfy.utils.common_upscale(
            k[:1].movedim(-1, 1), W, H, "lanczos", "disabled").movedim(1, -1) for k in shown])
        # keyframes not made block what hangs off them (their Save Image) instead of saving a copy of the first
        kfs = shown + [ExecutionBlocker(None)] * (4 - len(shown))
        # the node's own UI (web/js/keyframes.js) shows them large with a redraw button each
        return io.NodeOutput(*kfs, batch, ui={"openrelight_keyframes": [json.dumps(cards)]})


class EncodeFirst(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="OR_EncodeFirst",
            display_name="Encode Every Keyframe First (OpenRelight)",
            category=CATEGORY,
            description="Passes each keyframe's conditioning through once every keyframe's prompt is encoded, so the "
                        "text encoder and then the image model each load once. Run one keyframe after another, they "
                        "took turns and the image model was set up again for every keyframe (~10 s each).",
            inputs=[io.Conditioning.Input("c0"), io.Conditioning.Input("c1", optional=True, lazy=True),
                    io.Conditioning.Input("c2", optional=True, lazy=True),
                    io.Conditioning.Input("c3", optional=True, lazy=True),
                    io.Int.Input("count", default=4, min=1, max=4, optional=True, tooltip="Keyframe Plan's count.")],
            outputs=[io.Conditioning.Output(display_name=f"c{i}") for i in range(4)],
        )

    @classmethod
    def check_lazy_status(cls, count=4, **kwargs):
        return [f"c{i}" for i in range(1, max(1, min(int(count), 4))) if f"c{i}" in kwargs and kwargs[f"c{i}"] is None]

    @classmethod
    def execute(cls, c0, c1=None, c2=None, c3=None, count=4) -> io.NodeOutput:
        from comfy_execution.graph_utils import ExecutionBlocker
        return io.NodeOutput(*[c if c is not None and i < int(count) else ExecutionBlocker(None)
                               for i, c in enumerate((c0, c1, c2, c3))])


class AddKeyframeGuides(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="OR_AddKeyframeGuides",
            display_name="Add Keyframe Guides for MiniMax H3 (OpenRelight)",
            category=CATEGORY,
            description="Anchors each keyframe at its frame, placed exactly where the reference clip sits in the video "
                        "model's padded frame (Fit & Pad's pad_info), however many keyframes the plan made.",
            inputs=[io.Conditioning.Input("positive"), io.Latent.Input("latent"), io.Vae.Input("vae"),
                    io.Image.Input("keyframes", tooltip="Keyframes' 'keyframes' batch."),
                    io.String.Input("frames", force_input=True, tooltip="Keyframe Plan's frame list."),
                    PadInfoType.Input("pad_info", tooltip="The reference clip's Fit & Pad: the same scale and padding.")],
            outputs=[io.Conditioning.Output(display_name="positive")],
        )

    @classmethod
    def execute(cls, positive, latent, vae, keyframes, frames, pad_info) -> io.NodeOutput:
        from comfy_extras.nodes_minimax_h3 import MiniMaxH3AddGuide
        video = latent["samples"].tensors[0]
        th, tw = video.shape[3] * 16, video.shape[4] * 16      # the padded frame the video model makes
        px, py, w, h = pad_info["box"]                          # where the clip sits in it
        idx = [int(f) for f in frames.split(",") if f.strip().lstrip("-").isdigit()]
        for i, f in enumerate(idx[:keyframes.shape[0]]):
            img = comfy.utils.common_upscale(keyframes[i:i + 1].movedim(-1, 1), w, h, "lanczos", "disabled")
            img = torch.nn.functional.pad(img, (px, tw - w - px, py, th - h - py), mode="replicate").movedim(1, -1)
            positive = MiniMaxH3AddGuide.execute(positive=positive, latent=latent, frame_idx=f, vae=vae,
                                                 image=_out(img)).args[0]
        return io.NodeOutput(positive)


class KeyframePlan(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="OR_KeyframePlan",
            display_name="Keyframe Plan (OpenRelight)",
            category=CATEGORY,
            description="Where the relit keyframes go. With the light rig: where its light changes (a moving light gets "
                        "keyframes along its path, up to 'count'; a still one the first and last frame). Without: spread "
                        "evenly over the clip. Wire kf0..kf3 to the keyframe nodes and 'frames' to the prompt nodes.",
            inputs=[io.Int.Input("frame_count", default=121, min=1, max=10000, tooltip="The clip's frame count."),
                    io.Int.Input("count", default=3, min=1, max=4,
                                 tooltip="With the light rig: at most this many (3: a moving light gets its start, middle "
                                         "and end). Without: exactly this many."),
                    LightRigType.Input("light_rig", optional=True),
                    io.Float.Input("fps", default=24.0, min=1.0, max=240.0, optional=True)],
            outputs=[io.String.Output(display_name="frames")] + [io.Int.Output(display_name=f"kf{i}") for i in range(4)]
                    + [io.Int.Output(display_name="count", tooltip="How many keyframes the plan made.")],
        )

    @classmethod
    def execute(cls, frame_count, count, light_rig=None, fps=24.0) -> io.NodeOutput:
        last = max(frame_count - 1, 0)
        if light_rig is not None:
            frames = rigmod.plan_keyframes(light_rig, frame_count, fps, max_count=count)
        else:
            frames = [round(last * i / max(count - 1, 1)) for i in range(count)] if count > 1 else [0]
        padded = frames + [frames[-1]] * (4 - len(frames))
        return io.NodeOutput(",".join(map(str, frames)), *padded, len(frames))


PROMPT_SEP = "\n\n=====\n\n"


def _reset_between_runs():
    """What ComfyUI does between nodes (execution.py). Calling clip.generate twice in one node without it
    leaves the prefetch / cast-buffer state of the first run and the second crashes on a CUDA index assert."""
    import comfy.memory_management
    import comfy.model_prefetch
    if comfy.memory_management.aimdo_enabled:
        import comfy_aimdo.model_vbar
        comfy.model_prefetch.cleanup_prefetch_queues()
        comfy.model_management.reset_cast_buffers()
        comfy_aimdo.model_vbar.vbars_reset_watermark_limits()


class LightingDirector(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="OR_LightingDirector",
            display_name="LM Lighting Director (OpenRelight)",
            category=CATEGORY,
            description="A vision LM (Qwen3-VL 8B, the Qwen-Image 2.1 text encoder already loaded for the keyframes) "
                        "looks at the keyframes' original frames (and a lighting reference or new background), reads the "
                        "light stage's plan and writes what it looks like on this footage: technical terms kept, the "
                        "visible result after each. One LM run for the plan and the video, one for each keyframe. Any part it gets wrong "
                        "(missing colour, missing timestamp) falls back to the template prompt. Off: the template prompts. "
                        "prompt_1..3 are the keyframes' prompts, for their Qwen encoders.",
            is_output_node=True,
            inputs=[
                io.String.Input("keyframe_prompts", force_input=True, optional=True,
                                tooltip="Optional: your own fallback prompts. Unconnected, the director writes them."),
                io.String.Input("video_prompt", force_input=True, optional=True),
                LightRigType.Input("light_rig"),
                io.Image.Input("images", tooltip="The clip; the LM sees the frames at the keyframes."),
                io.String.Input("keyframes", default="0,60,123", tooltip="Frame numbers (Keyframe Plan 'frames')."),
                io.Int.Input("frame_count", default=121, min=1, max=10000),
                io.Float.Input("fps", default=24.0, min=1.0, max=120.0, step=0.001),
                io.Boolean.Input("enabled", default=True),
                io.Clip.Input("clip", optional=True, lazy=True,
                              tooltip="CLIP Loader: qwen3vl_8b (type qwen_image), the same one Qwen-Image uses."),
                io.Image.Input("reference", optional=True, tooltip="Lighting reference or new background image."),
                io.Combo.Input("reference_kind", options=["lighting reference", "new background", "green screen"],
                               default="lighting reference",
                               tooltip="green screen: the reference is the plate that replaces the green screen; connect it "
                                       "to the keyframes' Qwen encoders as image 2 too."),
                io.Boolean.Input("mockup", default=False,
                                 tooltip="Only for keyframe images with a light guide painted on them: adds a note that the "
                                         "painted light is a guide. Leave off when the keyframes edit the original frames."),
                io.Int.Input("seed", default=7, min=0, max=2**31 - 1),
                io.String.Input("scene", force_input=True, optional=True,
                                tooltip="Detect Original Light's scene: the director matches the stage's lights to it (a light "
                                        "behind the people = through the windows there)."),
            ],
            outputs=[io.String.Output(display_name="keyframe_prompts"), io.String.Output(display_name="video_prompt"),
                     io.String.Output(display_name="lm_text", tooltip="What the LM wrote, and any fallback.")]
                    + [io.String.Output(display_name=f"prompt_{i + 1}", tooltip=f"Keyframe {i + 1}'s prompt.") for i in range(3)],
        )

    @classmethod
    def check_lazy_status(cls, enabled, **kwargs):
        # only a connected clip can be asked for (an unconnected one is absent here): no clip = template prompts
        return ["clip"] if enabled and "clip" in kwargs and kwargs["clip"] is None else []

    @classmethod
    def execute(cls, light_rig, images, keyframes, frame_count, fps, enabled, keyframe_prompts=None, video_prompt=None,
                clip=None, reference=None, reference_kind="lighting reference", mockup=True, seed=7, scene="") -> io.NodeOutput:
        kfs = [int(x) for x in keyframes.replace(" ", "").split(",") if x]
        if keyframe_prompts is None:
            keyframe_prompts = PROMPT_SEP.join(lm.keyframe_prompt(light_rig, f, frame_count, fps, mockup=mockup) for f in kfs)
        if video_prompt is None:
            video_prompt = lm.video_prompt(light_rig, frame_count, fps)

        def out(kf, video, text, ui=None):
            each = kf.split(PROMPT_SEP)
            each = [each[min(i, len(each) - 1)] for i in range(3)]
            return io.NodeOutput(kf, video, text, *each, ui=ui)

        if not enabled or clip is None:
            return out(keyframe_prompts, video_prompt, "(LM off: using the template prompts)")
        idx = [min(f, images.shape[0] - 1) for f in kfs]
        small = lambda im: _fit_long_side(im[..., :3], 640)  # noqa: E731
        kf, video, raw, notes = lm.direct(clip, light_rig, small(images[idx]), kfs, frame_count, fps,
                                          reference_image=None if reference is None else small(reference[:1]),
                                          reference_kind=reference_kind, mockup=mockup, seed=seed,
                                          between=_reset_between_runs, scene=scene or "")
        # shown whole: the plan (what light this is in this scene, what it does) and what each keyframe was told
        report = raw.split("</think>")[-1].strip() + ("\n\n" + "\n".join(notes) if notes else "")
        return out(PROMPT_SEP.join(kf), video, report, ui={"text": [report]})


def _clip_id(images):
    """The same clip (and trim) always gives the same id. The light stage replaces its original lights only when this
    changes: the LM reads one clip a little differently run to run (2 lights, then 3), so its reading cannot tell
    clips apart, and taking it for a new clip put the original lights back over the user's edits (2026-10-30)."""
    T = images.shape[0]
    small = torch.nn.functional.adaptive_avg_pool2d(images[[0, T // 2, T - 1]][..., :3].float().movedim(-1, 1), 8)
    data = (small.clamp(0, 1) * 255).round().to(torch.uint8).cpu().numpy().tobytes()
    return hashlib.sha1(data + str(tuple(images.shape)).encode()).hexdigest()[:12]


def _fit_long_side(images, size):
    H, W = images.shape[1:3]
    s = size / max(H, W)
    if s >= 1:
        return images
    return comfy.utils.common_upscale(images.movedim(-1, 1), round(W * s), round(H * s), "area", "disabled").movedim(1, -1)


class OpenRelightExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [SourceClip, LightRig, GeometryPrep, DetectOriginalLight, KeyframeSeeds, KeyframeReview, EncodeFirst, AddKeyframeGuides, KeyframePlan, LightingDirector, FitPad, Unpad]
