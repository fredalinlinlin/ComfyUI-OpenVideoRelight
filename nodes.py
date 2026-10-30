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
from .openrelight.detail import detail_lock
from .openrelight.guard import light_guard
from .openrelight.geometry import depth_preview, normal_preview, prepare_geometry
from .openrelight.prompt import rig_to_prompt
from .openrelight.detect import detect_original_lights, detect_reference_lights
from .openrelight.clip import clip_sheet
from .openrelight.shading import (detect_dominant_light, fit_sh1, light_map, light_mockup, linear_to_srgb, relight_mockup,
                                  render, srgb_to_linear)

LightRigType = io.Custom("OR_LIGHT_RIG")
GeometryType = io.Custom("OR_GEOMETRY")
MoGeGeometry = io.Custom("MOGE_GEOMETRY")
MoGeModelType = io.Custom("MOGE_MODEL")

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
                                   tooltip="Reference Lights: the lights found in a lighting reference image. The "
                                           "editor puts them on the stage (switching the original light off)."),
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


class LightPass(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="OR_LightPass",
            display_name="Light Pass (OpenRelight)",
            category=CATEGORY,
            description="Relight every frame from the rig. 'guide' feeds the video model; 'composite_layer' holds composite-mode lights for DetailLock.",
            inputs=[
                io.Image.Input("images"),
                GeometryType.Input("geometry"),
                LightRigType.Input("light_rig"),
                io.Float.Input("fps", default=24.0, min=1.0, max=120.0, step=0.001),
                io.Float.Input("delight", default=0.6, min=0.0, max=1.0, step=0.05,
                               tooltip="How much of the original light to estimate and remove. 0 = only add light."),
                io.Float.Input("keep_original", default=0.35, min=0.0, max=1.5, step=0.05,
                               tooltip="How much of the original lighting stays."),
                io.Combo.Input("scope", options=["subject", "all"], default="subject",
                               tooltip="'subject' lights only inside the mask; 'all' relights the whole frame (interiors)."),
                io.Boolean.Input("shadows", default=False,
                                 tooltip="Experimental screen-space shadows. The video model copies guide artefacts, so leave off unless the shot needs cast shadows."),
                io.Float.Input("shadow_strength", default=0.85, min=0.0, max=1.0, step=0.05),
                io.Float.Input("specular", default=0.1, min=0.0, max=2.0, step=0.05,
                               tooltip="Specular in the guide. Keep low: the video model copies it, and fabric/skin turn plastic above ~0.2."),
                io.Float.Input("roughness", default=0.6, min=0.05, max=1.0, step=0.05),
                io.Float.Input("exposure_ev", default=0.0, min=-4.0, max=4.0, step=0.1),
            ],
            outputs=[io.Image.Output(display_name="guide"), io.Image.Output(display_name="full_preview"),
                     io.Image.Output(display_name="composite_layer"), io.Image.Output(display_name="shading")],
        )

    @classmethod
    def execute(cls, images, geometry, light_rig, fps, delight, keep_original, scope, shadows, shadow_strength,
                specular, roughness, exposure_ev) -> io.NodeOutput:
        params = {"delight": delight, "keep": keep_original, "scope": scope, "shadows": shadows,
                  "shadow_strength": shadow_strength, "spec_strength": specular, "roughness": roughness,
                  "exposure_ev": exposure_ev}
        pbar = comfy.utils.ProgressBar(images.shape[0])
        guide, full, layer, shading, _ = render(images, geometry, light_rig, fps, params,
                                                progress=lambda i, n: pbar.update_absolute(i, n))
        return io.NodeOutput(_out(guide), _out(full), _out(layer), _out(shading))


class RigToPrompt(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="OR_RigToPrompt",
            display_name="Rig to Prompt (OpenRelight)",
            category=CATEGORY,
            description="Describe the rig in words, with timestamps for moving lights, for MiniMax H3 / LTX.",
            inputs=[
                LightRigType.Input("light_rig"),
                io.Int.Input("frame_count", default=121, min=1, max=10000),
                io.Float.Input("fps", default=24.0, min=1.0, max=120.0, step=0.001),
                io.String.Input("ref_tag", default="<Video 1>", tooltip="How the prompt refers to the original clip."),
                io.Boolean.Input("preamble", default=True, tooltip="Start with 'keep identity, motion and timing; only lighting changes'."),
                io.Boolean.Input("include_composite", default=False,
                                 tooltip="Also describe composite-mode lights (normally they are added after generation)."),
                io.String.Input("extra", multiline=True, default=""),
            ],
            outputs=[io.String.Output(display_name="prompt")],
        )

    @classmethod
    def execute(cls, light_rig, frame_count, fps, ref_tag, preamble, include_composite, extra) -> io.NodeOutput:
        return io.NodeOutput(rig_to_prompt(light_rig, frame_count, fps, ref_tag=ref_tag, preamble=preamble,
                                           include_composite=include_composite, extra=extra))


class LightingBrief(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="OR_LightingBrief",
            display_name="Lighting Brief for LM (OpenRelight)",
            category=CATEGORY,
            description="Turn the rig into a brief plus system prompt for core 'Generate Text'. "
                        "keyframe: the Qwen-Image 2.1 prompt enhancer writes the edit for one frame. "
                        "video: Qwen3-VL writes the MiniMax H3 prompt, keeping the timestamps.",
            inputs=[
                LightRigType.Input("light_rig"),
                io.Combo.Input("target", options=["keyframe", "video"], default="keyframe"),
                io.Int.Input("frame_idx", default=0, min=0, max=10000, tooltip="keyframe: which frame of the clip is relit."),
                io.Int.Input("frame_count", default=121, min=1, max=10000),
                io.Float.Input("fps", default=24.0, min=1.0, max=120.0, step=0.001),
                io.Boolean.Input("mockup", default=True,
                                 tooltip="keyframe: <image1> is the Light Map 'mockup' frame (light already painted on), "
                                         "so where the light falls comes from the geometry, not from words."),
                io.String.Input("extra", multiline=True, default=""),
            ],
            outputs=[io.String.Output(display_name="brief"), io.String.Output(display_name="system_prompt"),
                     io.String.Output(display_name="after_lm",
                                      tooltip="Appended after the LM's rewrite (Clean LM Text 'append'), so the LM can't drop it.")],
        )

    @classmethod
    def execute(cls, light_rig, target, frame_idx, frame_count, fps, mockup, extra) -> io.NodeOutput:
        if target == "keyframe":
            after = " ".join(t for t in (lm.MOCKUP_NOTE if mockup else "",
                                         lm.offscreen_note(light_rig, frame_count, fps, frames=[frame_idx])) if t)
            return io.NodeOutput(lm.keyframe_brief(light_rig, frame_idx, frame_count, fps, mockup=mockup, extra=extra),
                                 lm.qwen_edit_system_prompt(), after)
        return io.NodeOutput(lm.video_brief(light_rig, frame_count, fps, extra=extra), lm.VIDEO_LM_SYSTEM,
                             lm.offscreen_note(light_rig, frame_count, fps))


class LightMap(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="OR_LightMap",
            display_name="Light Map (OpenRelight)",
            category=CATEGORY,
            description="Preview of the light stage on the clip, and 'mockup': the clip lit as the stage says, for the "
                        "image editor as a rough guide to where the light comes from (softened: the geometry knows "
                        "directions, not materials). Detected lights left as detected change nothing; moved ones darken "
                        "where they were and brighten where they go; deleted ones darken what they lit.",
            inputs=[
                GeometryType.Input("geometry"),
                LightRigType.Input("light_rig"),
                io.Int.Input("frame_count", default=121, min=1, max=10000),
                io.Float.Input("fps", default=24.0, min=1.0, max=120.0, step=0.001),
                io.Image.Input("images", optional=True,
                               tooltip="The clip to paint the light onto for 'mockup', at any resolution (full res for keyframes)."),
                io.Float.Input("ambient", default=0.3, min=0.05, max=1.0, step=0.05,
                               tooltip="Share of the clip's light that no detected light explains (it always stays). "
                                       "Lower = new lights look stronger."),
                io.Float.Input("soften", default=0.03, min=0.0, max=0.1, step=0.005,
                               tooltip="Blur of the painted light, as a fraction of the frame: keeps where the light goes, "
                                       "drops the hard CG shading edge. 0 = sharp."),
                io.Float.Input("shadow_strength", default=0.8, min=0.0, max=1.0, step=0.05,
                               tooltip="Cast shadows from the scene's depth: what is in front blocks the light for what is "
                                       "behind it. 0 = off. Only things visible in the frame cast shadows."),
                io.Float.Input("shadow_length", default=0.6, min=0.05, max=2.0, step=0.05,
                               tooltip="How far shadows reach, as a fraction of the camera-to-subject distance."),
                io.Float.Input("relief", default=1.0, min=0.0, max=1.0, step=0.05,
                               tooltip="How much the light follows the shape of faces and bodies. "
                                       "1 = full form (bright where surfaces face the light), 0 = an even wash inside the beam."),
                io.Int.Input("preview_frames", default=4, min=0, max=12,
                             tooltip="Frames shown on the node (light map above, mockup below), evenly spread over the clip. 0 = none."),
            ],
            outputs=[io.Image.Output(display_name="light_map"),
                     io.Image.Output(display_name="mockup",
                                     tooltip="The clip with the light painted on, at the images' resolution: the image editor's <image1>.")],
        )

    @classmethod
    def execute(cls, geometry, light_rig, frame_count, fps, ambient, soften, shadow_strength, shadow_length, relief,
                preview_frames, images=None) -> io.NodeOutput:
        pbar = comfy.utils.ProgressBar(frame_count)
        srgb, mockup = relight_mockup(images, geometry, light_rig, frame_count, fps, ambient=ambient, soften=soften,
                                      progress=lambda i, n: pbar.update_absolute(i, n), shadow_strength=shadow_strength,
                                      shadow_length=shadow_length, relief=relief)
        if preview_frames <= 0:
            return io.NodeOutput(_out(srgb), _out(mockup))
        return io.NodeOutput(_out(srgb), _out(mockup), ui=ui.PreviewImage(_preview_sheet(srgb, mockup, preview_frames), cls=cls))


class LightGuard(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="OR_LightGuard",
            display_name="Light Guard (OpenRelight)",
            category=CATEGORY,
            description="Stop invented light: a generated frame may only get brighter than the original where the rig's "
                        "light lands (Light Map). Removes light sources and beams the image/video model made up, e.g. a "
                        "cone from a lamp in the shot. Never darkens. Works on keyframes and on whole clips.",
            inputs=[
                io.Image.Input("images", tooltip="Generated / relit frames."),
                io.Image.Input("original", tooltip="The original frames (same frames, any resolution)."),
                io.Image.Input("light_map", tooltip="Light Map 'light_map' output for the same frames."),
                io.Float.Input("margin", default=0.15, min=0.0, max=2.0, step=0.05,
                               tooltip="How much brighter than the original areas without new light may get."),
                io.Float.Input("allowance", default=12.0, min=0.0, max=64.0, step=0.5,
                               tooltip="How much brighter areas with new light may get, per unit of light map."),
                io.Float.Input("softness", default=4.0, min=0.0, max=32.0, step=0.5,
                               tooltip="Pixels over which a capped area fades in."),
            ],
            outputs=[io.Image.Output(display_name="images")],
        )

    @classmethod
    def execute(cls, images, original, light_map, margin, allowance, softness) -> io.NodeOutput:
        out = light_guard(images, original, light_map, margin=margin, allowance=allowance, softness=softness,
                          device=comfy.model_management.get_torch_device())
        return io.NodeOutput(_out(out))


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


def _preview_sheet(maps, mockup, count):
    """One image: chosen frames side by side, light map on top, mockup below (at the map's size)."""
    n = min(maps.shape[0], mockup.shape[0])
    idx = torch.linspace(0, n - 1, min(count, n)).round().long().tolist()
    H, W = maps.shape[1:3]
    mock = mockup[idx].to(maps.device).float()
    if mock.shape[1:3] != (H, W):
        mock = torch.nn.functional.interpolate(mock.permute(0, 3, 1, 2), size=(H, W), mode="bilinear",
                                               align_corners=False).permute(0, 2, 3, 1)
    top = torch.cat(list(maps[idx].float()), dim=1)
    bottom = torch.cat(list(mock), dim=1)
    return torch.cat([top, bottom], dim=0).unsqueeze(0).clamp(0, 1).cpu()


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


class GreenScreen(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="OR_GreenScreen",
            display_name="Green Screen Composite (OpenRelight)",
            category=CATEGORY,
            description="Key a green (or blue) screen out of the clip, despill it and put the background plate behind. "
                        "Feed the composite to the relight, and the plate to the LM Lighting Director as its 'new "
                        "background', so the subject is lit as that place would light it.",
            is_output_node=True,
            inputs=[io.Image.Input("images"),
                    io.Image.Input("background", tooltip="The new background (a still plate, or a clip of the same length)."),
                    io.Combo.Input("screen", options=["green", "blue"], default="green"),
                    io.Float.Input("matte_low", default=0.2, min=0.0, max=1.0, step=0.01,
                                   tooltip="Below this much screen colour a pixel stays fully foreground."),
                    io.Float.Input("matte_high", default=0.7, min=0.0, max=1.0, step=0.01,
                                   tooltip="Above this much screen colour a pixel is fully background."),
                    io.Float.Input("matte_blur", default=1.0, min=0.0, max=10.0, step=0.1, tooltip="Matte softness, pixels."),
                    io.Float.Input("background_blur", default=0.0, min=0.0, max=5.0, step=0.1,
                                   tooltip="Defocus the plate (percent of the frame)."),
                    io.Mask.Input("external_matte", optional=True,
                                  tooltip="A foreground matte from your compositing software (white = keep). Used instead "
                                          "of the built-in colour key for hard shots (heavy spill, tracking markers, glass).")],
            outputs=[io.Image.Output(display_name="images"), io.Mask.Output(display_name="matte")],
        )

    @classmethod
    def execute(cls, images, background, screen, matte_low, matte_high, matte_blur, background_blur,
                external_matte=None) -> io.NodeOutput:
        from .openrelight.key import key_composite
        comp, alpha = key_composite(images.float(), background.float(), screen, matte_low, matte_high, matte_blur,
                                    background_blur, matte=external_matte)
        mid = comp.shape[0] // 2
        preview = ui.PreviewImage(torch.cat([comp[mid:mid + 1], alpha[mid:mid + 1, ..., None].expand(-1, -1, -1, 3)]))
        return io.NodeOutput(_out(comp), alpha, ui=preview)


class RestoreBackground(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="OR_RestoreBackground",
            display_name="Restore Background Plate (OpenRelight)",
            category=CATEGORY,
            description="After the relight, put the green-screen plate back where the screen was (Green Screen "
                        "Composite's matte): Qwen and H3 repaint the background (darker sky, other clouds); this keeps "
                        "the plate exactly as chosen while the subject keeps its new light.",
            inputs=[io.Image.Input("images"), io.Mask.Input("matte"), io.Image.Input("background"),
                    io.Int.Input("frame_index", default=-1, min=-1, max=100000,
                                 tooltip="-1: the matte has a frame per image. Otherwise the matte frame to use for a "
                                         "single image (a keyframe)."),
                    io.Float.Input("background_blur", default=0.0, min=0.0, max=5.0, step=0.1)],
            outputs=[io.Image.Output(display_name="images")],
        )

    @classmethod
    def execute(cls, images, matte, background, frame_index, background_blur) -> io.NodeOutput:
        from .openrelight.key import restore_background
        m = matte if matte.dim() == 3 else matte[None]
        if frame_index >= 0:
            m = m[min(frame_index, m.shape[0] - 1)][None]
        return io.NodeOutput(_out(restore_background(images.float(), m.float(), background.float(), background_blur)))


class ReferenceLights(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="OR_ReferenceLights",
            display_name="Reference Lights (OpenRelight)",
            category=CATEGORY,
            description="Find the lights of a lighting reference image (how many, from where, what colour, how bright "
                        "relative to each other, how dark the rest is) with MoGe, and put them on the light stage. No "
                        "image connected = nothing. A starting point to adjust; softness is not estimated.",
            is_output_node=True,
            inputs=[MoGeModelType.Input("moge_model"),
                    io.Image.Input("image", optional=True, tooltip="The lighting reference (Load Image)."),
                    io.Int.Input("max_lights", default=3, min=1, max=4)],
            outputs=[LightRigType.Output(display_name="reference_lights"), io.String.Output(display_name="report")],
        )

    @classmethod
    def execute(cls, moge_model, max_lights, image=None) -> io.NodeOutput:
        if image is None:
            return io.NodeOutput(None, "(no reference image)")
        from comfy_extras.nodes_moge import MoGeInference
        img = image[:1, ..., :3]
        H, W = img.shape[1:3]
        s = min(1.0, (0.5e6 / (H * W)) ** 0.5)
        if s < 1.0:
            img = comfy.utils.common_upscale(img.movedim(-1, 1), round(W * s), round(H * s), "area", "disabled").movedim(1, -1)
        moge = MoGeInference.execute(moge_model, img, 4, 0.0, 1, True, True, 1).result[0]
        geo = prepare_geometry(moge, temporal_sigma=0, stabilize_scale=False,
                               device=comfy.model_management.get_torch_device())
        rig, report = detect_reference_lights(img[0], geo, max_lights=max_lights)
        # the light stage editor applies these from this message (web/js/lightstage.js)
        return io.NodeOutput(rig, report, ui={"text": [report], "openrelight_reference": [json.dumps(rig, ensure_ascii=False)]})


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


class ApproveKeyframes(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="OR_ApproveKeyframes",
            display_name="Approve Keyframes (OpenRelight)",
            category=CATEGORY,
            description="Shows the relit keyframes and stops here until 'approved' is on. Check them, redraw any with "
                        "Keyframe Seeds, then turn 'approved' on and run again: the video is made from the cached keyframes.",
            is_output_node=True,
            inputs=[io.Image.Input("kf0"), io.Image.Input("kf1", optional=True), io.Image.Input("kf2", optional=True),
                    io.Image.Input("kf3", optional=True),
                    io.Boolean.Input("approved", default=False, tooltip="On = go on to the video.")],
            outputs=[io.Image.Output(display_name=f"kf{i}") for i in range(4)],
        )

    @classmethod
    def fingerprint_inputs(cls, approved, **kwargs):
        return approved

    @classmethod
    def execute(cls, kf0, approved, kf1=None, kf2=None, kf3=None) -> io.NodeOutput:
        from comfy_execution.graph_utils import ExecutionBlocker
        kfs = [kf0] + [k if k is not None else kf0 for k in (kf1, kf2, kf3)]
        shown = [k for k in (kf0, kf1, kf2, kf3) if k is not None]
        H, W = shown[0].shape[1:3]
        sheet = torch.cat([torch.nn.functional.interpolate(k[:1].permute(0, 3, 1, 2), size=(H, W), mode="bilinear",
                                                           align_corners=False).permute(0, 2, 3, 1) for k in shown], 2)
        preview = ui.PreviewImage(_out(sheet).float().cpu(), cls=cls)
        if not approved:
            return io.NodeOutput(*[ExecutionBlocker(None)] * 4, ui=preview)
        return io.NodeOutput(*kfs, ui=preview)


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


class PolishPrompts(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="OR_PolishPrompts",
            display_name="Polish Prompts with LM (OpenRelight)",
            category=CATEGORY,
            description="Optional, off by default. An LM looks at one frame and the user's notes (light stage 'extra "
                        "description'), writes the original's style and medium, and rewrites the notes as clear prompt "
                        "language. It never changes the lights. Off: the prompts pass through and no LM is loaded.",
            inputs=[
                io.String.Input("keyframe_prompts", force_input=True),
                io.String.Input("video_prompt", force_input=True),
                LightRigType.Input("light_rig"),
                io.Image.Input("images", tooltip="The clip; the LM sees its middle frame."),
                io.Boolean.Input("enabled", default=False),
                io.Combo.Input("engine", options=["built-in Gemma 4 E2B", "LM Studio"], default="built-in Gemma 4 E2B",
                               tooltip="Built-in: connect a CLIP Loader with gemma4_e2b_it (type ltxv). LM Studio: its local "
                                       "server must be running with a vision model loaded."),
                io.Clip.Input("clip", optional=True, lazy=True, tooltip="Built-in engine only."),
                io.String.Input("lmstudio_url", default="http://127.0.0.1:1234"),
                io.String.Input("lmstudio_model", default="", tooltip="Empty = the model LM Studio has loaded."),
            ],
            outputs=[io.String.Output(display_name="keyframe_prompts"), io.String.Output(display_name="video_prompt"),
                     io.String.Output(display_name="lm_notes", tooltip="What the LM wrote (style + polished notes).")],
        )

    @classmethod
    def check_lazy_status(cls, enabled, engine, **kwargs):
        return ["clip"] if enabled and engine.startswith("built-in") and "clip" in kwargs and kwargs["clip"] is None else []

    @classmethod
    def execute(cls, keyframe_prompts, video_prompt, light_rig, images, enabled, engine, lmstudio_url, lmstudio_model,
                clip=None) -> io.NodeOutput:
        if not enabled:
            return io.NodeOutput(keyframe_prompts, video_prompt, "(LM off)")
        frame = images[images.shape[0] // 2][None]
        notes = (light_rig.get("user_prompt") or "").strip()
        if engine.startswith("built-in"):
            if clip is None:
                raise ValueError("Polish Prompts: connect a CLIP Loader with gemma4_e2b_it (type ltxv) for the built-in engine.")
            raw = lm.polish_with_gemma(clip, frame, notes)
        else:
            raw = lm.polish_with_lmstudio(lmstudio_url, lmstudio_model, frame, notes)
        style, polished = lm.parse_polish(lm.clean_lm_output(raw) if "STYLE" not in raw else raw)
        kf = PROMPT_SEP.join(lm.apply_polish(p, notes, style, polished) for p in keyframe_prompts.split(PROMPT_SEP))
        return io.NodeOutput(kf, lm.apply_polish(video_prompt, notes, style, polished),
                             f"STYLE: {style}\nNOTES: {polished or '(none)'}")


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


class RelightPrompts(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="OR_RelightPrompts",
            display_name="Relight Prompts (OpenRelight)",
            category=CATEGORY,
            description="Every keyframe prompt (Qwen-Image 2.1 edit) and the MiniMax H3 prompt, written straight from the "
                        "light rig with fixed templates: no LM, instant. Qwen and H3 read the frames themselves. Same "
                        "outputs as LM Rewrite All; pick a keyframe's prompt with Pick Prompt.",
            inputs=[
                LightRigType.Input("light_rig"),
                io.String.Input("keyframes", default="0,60,123", tooltip="Frame numbers (Keyframe Plan 'frames')."),
                io.Int.Input("frame_count", default=121, min=1, max=10000),
                io.Float.Input("fps", default=24.0, min=1.0, max=120.0, step=0.001),
                io.Boolean.Input("mockup", default=True, tooltip="Keyframes edit the Light Map mockup (light painted on)."),
            ],
            outputs=[io.String.Output(display_name="keyframe_prompts", tooltip="All keyframe prompts; use Pick Prompt."),
                     io.String.Output(display_name="video_prompt")],
        )

    @classmethod
    def execute(cls, light_rig, keyframes, frame_count, fps, mockup) -> io.NodeOutput:
        kfs = [int(x) for x in keyframes.replace(" ", "").split(",") if x]
        prompts = [lm.keyframe_prompt(light_rig, f, frame_count, fps, mockup=mockup) for f in kfs]
        return io.NodeOutput(PROMPT_SEP.join(prompts), lm.video_prompt(light_rig, frame_count, fps))


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
                                tooltip="Optional: other fallback prompts (Relight Prompts). Unconnected, the director writes them."),
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
                io.Boolean.Input("mockup", default=True, tooltip="Keyframes edit the Light Map mockup (adds its note)."),
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


class LMRewriteAll(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="OR_LMRewriteAll",
            display_name="LM Rewrite All Prompts (OpenRelight)",
            category=CATEGORY,
            description="Write every keyframe prompt (Qwen-Image 2.1 prompt enhancer) and the MiniMax H3 prompt in one "
                        "node, from two LM rewrites: one shared by all keyframes (each adds where the light comes from "
                        "at its frame) and one for the video. Pick a keyframe's prompt with Pick Prompt.",
            inputs=[
                io.Clip.Input("clip", tooltip="CLIP Loader: qwen3.5_9b_qwen_image_2.1_pe_i2i, type qwen_image."),
                LightRigType.Input("light_rig"),
                io.Image.Input("images", tooltip="The clip (any resolution); the LM looks at each keyframe's frame."),
                io.String.Input("keyframes", default="0,12,24,36,48,60,72,84", tooltip="Frame numbers, comma separated."),
                io.Int.Input("frame_count", default=121, min=1, max=10000),
                io.Float.Input("fps", default=24.0, min=1.0, max=120.0, step=0.001),
                io.Boolean.Input("mockup", default=True, tooltip="Keyframes edit the Light Map mockup (see Lighting Brief)."),
                io.String.Input("video_frames", default="0,24,48,72",
                                tooltip="Frames the LM sees to write the H3 prompt. Empty = no video prompt."),
                io.Int.Input("seed", default=7, min=0, max=2 ** 31 - 1),
                io.Int.Input("max_length", default=6144, min=256, max=16384,
                             tooltip="Token limit per rewrite. The enhancer reasons for thousands of tokens before it "
                                     "answers; past the limit the brief is used as it is."),
            ],
            outputs=[io.String.Output(display_name="keyframe_prompts", tooltip="All keyframe prompts; use Pick Prompt."),
                     io.String.Output(display_name="video_prompt")],
        )

    @classmethod
    def execute(cls, clip, light_rig, images, keyframes, frame_count, fps, mockup, video_frames, seed,
                max_length) -> io.NodeOutput:
        kfs = [int(x) for x in keyframes.replace(" ", "").split(",") if x]
        vf = [int(x) for x in video_frames.replace(" ", "").split(",") if x]
        video_images = images[[min(f, images.shape[0] - 1) for f in vf]] if vf else None
        pbar = comfy.utils.ProgressBar(len(kfs) + (1 if vf else 0))
        prompts, video = lm.rewrite_all(clip, light_rig, images, kfs, frame_count, fps, mockup=mockup,
                                        video_images=video_images, seed=seed, max_length=max_length,
                                        progress=lambda i, n: pbar.update_absolute(i, n), between=_reset_between_runs)
        return io.NodeOutput(PROMPT_SEP.join(prompts), video)


class PickPrompt(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="OR_PickPrompt",
            display_name="Pick Prompt (OpenRelight)",
            category=CATEGORY,
            description="One keyframe's prompt out of LM Rewrite All's keyframe_prompts (index 0 = first keyframe).",
            inputs=[io.String.Input("prompts", force_input=True), io.Int.Input("index", default=0, min=0, max=99)],
            outputs=[io.String.Output(display_name="prompt")],
        )

    @classmethod
    def execute(cls, prompts, index) -> io.NodeOutput:
        items = prompts.split(PROMPT_SEP)
        return io.NodeOutput(items[min(index, len(items) - 1)])


class CleanLMText(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="OR_CleanLMText",
            display_name="Clean LM Text (OpenRelight)",
            category=CATEGORY,
            description="Keep only the final answer from 'Generate Text': drops <think> blocks and reasoning written before the answer.",
            inputs=[io.String.Input("text", force_input=True),
                    io.String.Input("append", force_input=True, optional=True,
                                    tooltip="Added after the answer, e.g. 'after_lm' from Lighting Brief.")],
            outputs=[io.String.Output(display_name="text")],
        )

    @classmethod
    def execute(cls, text, append="") -> io.NodeOutput:
        out = lm.clean_lm_output(text)
        return io.NodeOutput(f"{out} {append.strip()}" if append and append.strip() else out)


class DetailLock(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="OR_DetailLock",
            display_name="Detail Lock (OpenRelight)",
            category=CATEGORY,
            description="Keep the original's face and detail; take only the lighting from the generated video. Adds composite-mode lights.",
            inputs=[
                io.Image.Input("original"),
                io.Image.Input("generated", tooltip="Any resolution or frame count; resampled to the original."),
                io.Mask.Input("mask", optional=True, tooltip="Where to lock detail (usually the subject). Outside it the generation is kept."),
                io.Image.Input("composite_layer", optional=True, tooltip="'composite_layer' from Light Pass."),
                io.Float.Input("radius", default=0.03, min=0.005, max=0.15, step=0.005,
                               tooltip="Low-pass radius as a fraction of the frame. Smaller = more of the generation's shading detail."),
                io.Float.Input("strength", default=0.85, min=0.0, max=1.0, step=0.05, tooltip="0 = pure generation, 1 = fully locked."),
                io.Float.Input("highlights", default=0.5, min=0.0, max=1.5, step=0.05,
                               tooltip="Bring back specular highlights the generation added."),
                io.Float.Input("layer_strength", default=1.0, min=0.0, max=3.0, step=0.05),
            ],
            outputs=[io.Image.Output(display_name="image")],
        )

    @classmethod
    def execute(cls, original, generated, radius, strength, highlights, layer_strength, mask=None,
                composite_layer=None) -> io.NodeOutput:
        pbar = comfy.utils.ProgressBar(original.shape[0])
        out = detail_lock(original, generated, mask=mask, layer=composite_layer, radius=radius, strength=strength,
                          highlights=highlights, layer_strength=layer_strength,
                          device=comfy.model_management.get_torch_device(),
                          progress=lambda i, n: pbar.update_absolute(i, n))
        return io.NodeOutput(_out(out))


class DetectLights(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="OR_DetectLights",
            display_name="Detect Lights (OpenRelight)",
            category=CATEGORY,
            description="Estimate the clip's dominant light direction and add it to a rig as a starting key light.",
            inputs=[
                io.Image.Input("images"),
                GeometryType.Input("geometry"),
                LightRigType.Input("light_rig", optional=True),
            ],
            outputs=[LightRigType.Output(display_name="light_rig"), io.String.Output(display_name="report")],
        )

    @classmethod
    def execute(cls, images, geometry, light_rig=None) -> io.NodeOutput:
        dev = geometry["normal"].device
        coeffs = []
        step = max(1, images.shape[0] // 24)
        for t in range(0, min(images.shape[0], geometry["normal"].shape[0]), step):
            lin = srgb_to_linear(images[t].to(dev).float())
            region = geometry["valid"][t] & (geometry["alpha"][t] > 0.5)
            coeffs.append(fit_sh1(lin, geometry["normal"][t], region))
        az, el, strength = detect_dominant_light(torch.stack(coeffs))
        rig = rigmod.load_rig(light_rig or {})
        detected = rigmod.load_rig({"lights": [{
            "id": "detected_key", "name": "Detected key", "type": "area", "mode": "generate",
            "color": {"kelvin": 5600}, "softness": 0.5,
            "keys": [{"frame": 0, "azimuth": round(az, 1), "elevation": round(el, 1), "distance": 1.5,
                      "intensity_ev": 0.0}],
        }]})["lights"][0]
        rig["lights"] = [detected] + [l for l in rig["lights"] if l.get("id") != "detected_key"]
        report = f"Dominant light: azimuth {az:.0f} deg, elevation {el:.0f} deg, directional strength {strength:.2f}"
        return io.NodeOutput(rig, report)


class OpenRelightExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [SourceClip, GreenScreen, RestoreBackground, ReferenceLights, LightRig, GeometryPrep, LightPass, RigToPrompt, LightingBrief, LightMap, CleanLMText, DetectOriginalLight, KeyframeSeeds, KeyframeReview, EncodeFirst, AddKeyframeGuides, ApproveKeyframes, KeyframePlan, RelightPrompts, LightingDirector, PolishPrompts, LMRewriteAll, PickPrompt, LightGuard, FitPad, Unpad, DetailLock, DetectLights]
