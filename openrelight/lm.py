"""Briefs and system prompts for the LM step.

The rig is turned into a deterministic brief here; core TextGenerate then rewrites
it with a vision LM that sees the actual frame:
  - keyframe brief -> Qwen-Image 2.1 prompt enhancer -> Qwen-Image 2.1 Edit (one relit frame)
  - video brief    -> Qwen3-VL                        -> MiniMax H3 prompt (timestamps kept)
"""

import glob
import json
import math
import os
import re

from .prompt import (FILMS, GRADES, LOOKS, PREAMBLE, TYPE_WORDS, _color_words, ambient_words, as_change, intensity_words,
                     kelvin_words, removed_original_words, rig_to_prompt, stage_lights)
from .rig import evaluate_light, filled_keys
from . import pro


def frame_direction_words(az, el):
    """Direction as seen in the frame (image editors think in screen space, not the subject's left/right)."""
    a = ((az + 180.0) % 360.0) - 180.0
    side = "right" if a > 0 else "left"
    m = abs(a)
    if m <= 22.5:
        horiz = "from the camera position, straight onto the people"
    elif m <= 67.5:
        horiz = f"from the front, off to the {side} of the frame"
    elif m <= 112.5:
        horiz = f"from the {side} edge of the frame, at a right angle to the camera"
    elif m <= 157.5:
        horiz = f"from behind the people, on the {side} of the frame"
    else:
        horiz = "from directly behind the people, facing the camera"
    if el > 60:
        return horiz + ", from high overhead"
    if el > 25:
        return horiz + ", angled down from above"
    if el < -25:
        return horiz + ", angled up from below"
    return horiz + ", at head height"


def landing_words(az):
    """Where the light lands on the people: the part an editing model must get right."""
    a = ((az + 180.0) % 360.0) - 180.0
    side = "right" if a > 0 else "left"
    other = "left" if a > 0 else "right"
    m = abs(a)
    if m <= 22.5:
        return "faces and fronts of bodies are lit head-on, with little shadow on the faces"
    if m <= 67.5:
        return (f"the {side}-facing half of each face is lit and the {other} half falls into soft shadow "
                "(classic key light)")
    if m <= 112.5:
        return (f"only the {side}-facing side of each face, ear and shoulder is lit; the {other} side of "
                "each face stays dark (split lighting)")
    if m <= 157.5:
        return f"a bright rim outlines the {side} edges of heads and shoulders; faces stay mostly in shadow"
    return "a bright rim outlines every head and shoulder; faces and fronts stay in shadow (silhouette)"


def source_in_frame(s, fov_x=55.0, aspect=16 / 9):
    """Whether an evaluated light's source would show in the frame.

    Approximate: the subject sits 1 unit in front of the camera (lights are placed relative to it,
    as in shading.light_contribution) and the lens is a typical ~55 degrees wide.
    """
    dx, dy, dz = s["direction"]
    if s["type"] == "directional":
        x, y, z = dx, dy, dz
    else:
        x, y, z = dx * s["distance"], dy * s["distance"], 1.0 + dz * s["distance"]
    if z <= 0.05:
        return False
    tx = math.tan(math.radians(fov_x) / 2)
    return abs(x / z) < tx and abs(y / z) < tx / aspect


# Naming the lamps in the frame ("they keep their brightness") made Qwen draw the beam as a visible cone
# from the ceiling lamp; saying the beam itself is invisible did not (A/B on a test frame, 2026-10-28).
OFFSCREEN_NOTE = ("The new light comes from off-screen, so neither the light source nor the beam itself is visible; "
                  "the light only shows where it lands on the people and objects.")


def offscreen_note(rig, frame_count, fps, frames=None):
    """OFFSCREEN_NOTE if every generated light stays out of the frame at the given frames (default: the whole clip)."""
    frames = range(0, max(frame_count, 1), max(1, frame_count // 24)) if frames is None else frames
    lights = [l for l in rig["lights"] if l.get("enabled", True) and l.get("mode", "generate") == "generate"]
    if not lights:
        return ""
    for light in lights:
        for f in frames:
            if source_in_frame(evaluate_light(light, f, frame_count, fps)):
                return ""
    return OFFSCREEN_NOTE


def light_state_words(light, frame, frame_count, fps, landing=True, direction=True):
    """landing=False leaves out where the light falls on the people (a mock-up already shows it).
    direction=False also leaves out where the light comes from (added per keyframe by direction_words)."""
    s = evaluate_light(light, frame, frame_count, fps)
    keys = filled_keys(light, frame_count, fps)
    ev_now = math.log2(max(s["intensity"], 1e-4))
    if ev_now < max(k["intensity_ev"] for k in keys) - 3:
        return f"The {TYPE_WORDS.get(s['type'], 'light')} is off at this moment."
    kelvin = keys[0]["kelvin"]
    for k in keys:
        if k["_frame"] <= frame:
            kelvin = k["kelvin"]
    softness = s["softness"]
    quality = ("hard-edged, with crisp shadow edges" if softness < 0.3
               else "soft and diffused, with gentle shadow edges" if softness > 0.6 else "medium-soft")
    # with a mock-up the image already shows the pool of light; "spotlight beam" made the editor draw a
    # visible cone from a lamp in the shot, so the light is only named as a hard / soft light then
    kind = TYPE_WORDS.get(s["type"], "light") if landing else ("light" if s["type"] != "spot" else "key light")
    head = " ".join(w for w in (intensity_words(ev_now), _color_words(light, kelvin), kind) if w)
    where = f" {frame_direction_words(s['azimuth'], s['elevation'])}" if direction else ""
    text = f"A {head} shines{where}; it is {quality}."
    if not direction:
        # a shared rewrite: each keyframe adds its own direction, so the LM must not invent one
        text += " Do not say which direction it comes from; that is added separately for each frame."
    if landing:
        text += f" On the people, {landing_words(s['azimuth'])}."
    if s["type"] == "spot" and landing:
        text += " It is a focused beam: a pool of light with falloff to darkness around it, not an even wash."
    if light.get("prompt"):
        text += f" ({light['prompt']})"
    return text


# The mock-up is MoGe geometry lit by the rig: it knows directions, not materials. Copied exactly it
# looks like CG (hard terminators, plaster shading), so the editor only takes direction and coverage from it.
MOCKUP_NOTE = ("The light roughly painted onto this image is only a guide to where the new light comes from and "
               "which areas it reaches; render it as natural light in the image's own style, not as a copy of "
               "the painted shading.")


def direction_words(rig, frame, frame_count, fps):
    """Where each lit light comes from at this frame, one sentence each (appended to a shared LM prompt)."""
    out = []
    for light in rig["lights"]:
        if not light.get("enabled", True):
            continue
        s = evaluate_light(light, frame, frame_count, fps)
        keys = filled_keys(light, frame_count, fps)
        if math.log2(max(s["intensity"], 1e-4)) < max(k["intensity_ev"] for k in keys) - 3:
            continue
        out.append(f"At this moment the new light comes {frame_direction_words(s['azimuth'], s['elevation'])}.")
    return " ".join(out)


def keyframe_brief(rig, frame, frame_count, fps, mockup=False, extra="", direction=True):
    """Instruction for an image editor relighting one frame of the clip (the frame is <image1>).

    mockup: <image1> is the frame with the rig's light already painted on (Light Map 'mockup'). Where the
    light falls then comes from the scene's geometry, and the text only asks for real-looking light.
    Without it the text has to say where the light lands, guessed from the light's angle alone, which is
    wrong whenever people don't face the camera (a light from the right is a rim light on people facing left).
    """
    if mockup:
        lines = ["<image1> is a frame of a film whose new lighting has been roughly painted on. Turn it into a real "
                 "photograph lit by real film lights: keep every person's face, identity, expression, pose, hair, "
                 "clothing, every object, the framing and the camera exactly as they are. Real skin texture, real "
                 "fabric, film-like contrast, no plastic, painted or CGI look."]
    else:
        lines = ["Relight <image1>. Change only the lighting: keep every person's face, identity, expression, pose, "
                 "hair, clothing, every object, the framing and the camera exactly as they are. Keep it photographic "
                 "and real: real skin texture, real fabric, film-like contrast, no plastic or CGI look."]
    for light in rig["lights"]:
        if light.get("enabled", True):
            lines.append(light_state_words(light, frame, frame_count, fps, landing=not mockup, direction=direction))
    lines.append(ambient_words(rig))
    look = rig.get("look") or {}
    looks = [w for w in (LOOKS.get(look.get("preset", ""), look.get("preset", "")),
                         FILMS.get(look.get("film", ""), look.get("film", "")),
                         GRADES.get(look.get("grade", ""), look.get("grade", ""))) if w]
    if looks:
        lines.append(", ".join(looks).capitalize() + ".")
    lines.append(offscreen_note(rig, frame_count, fps, frames=[frame]))
    for text in (rig.get("user_prompt", ""), extra):
        if text and text.strip():
            lines.append(text.strip())
    return " ".join(l for l in lines if l)


def video_brief(rig, frame_count, fps, extra=""):
    """Brief for the LM that writes the video prompt: rig_to_prompt without the preamble."""
    text = "Lighting brief from the director: " + rig_to_prompt(rig, frame_count, fps, preamble=False,
                                                                include_composite=False, extra=extra)
    note = offscreen_note(rig, frame_count, fps)
    return f"{text} {note}" if note else text


VIDEO_LM_SYSTEM = """You write prompts for MiniMax H3, a reference-to-video model that re-renders a reference clip.
<Video 1> is the original clip. Anchored keyframes already show the target lighting at certain moments.
You receive frames of the clip and a lighting brief with timestamps from the director.

Write ONE paragraph in English, 90 to 160 words:
1. Start with: "Keep every person's face, identity, expression, body motion, wardrobe, framing, camera movement and timing exactly the same as <Video 1>. Only the lighting changes."
2. Name the setting in a few words from what you see (e.g. "a dark ship cabin with several crew members").
3. Describe each light as a real film light: type, colour, hardness, and where it lands on the people (which side of the faces, rim, pool of light). If the brief says the light comes from off-screen, say that neither the source nor the beam itself is visible, only where the light lands.
4. For moving or changing lights, describe the change in chronological order WITH the director's timestamps ("from 0.0s to 3.0s the beam sweeps from ... to ..."). Never drop or change a timestamp.
5. End with the look in a few words (film stock, contrast, grade) if the brief has one, and "photographic, real skin texture".
Do not invent lights, people, objects or camera moves. No headings, no lists, no quotes around the answer."""

def clean_lm_output(text):
    """Keep only the final answer of an LM rewrite.

    The Qwen 2.1 prompt enhancer sometimes writes its reasoning in plain text before the answer
    (no <think> tags); the answer is then the last paragraph, after a blank-line gap.
    """
    text = text.split("</think>")[-1].strip()
    if "\n\n\n" in text:
        text = text.split("\n\n\n")[-1]
    elif "\n\n" in text:
        paras = [p.strip() for p in text.split("\n\n") if p.strip()]
        text = paras[-1] if paras else text
    return text.strip().strip('{}"').strip()


_PE_FALLBACK = ("You rewrite image-editing instructions into one precise, unambiguous editing directive anchored on "
                "what the input image shows. Keep everything the user did not ask to change. Output only the directive.")


def qwen_edit_system_prompt():
    """The official Qwen-Image 2.1 edit prompt-enhancer system prompt, read from the installed workflow templates.

    The PE model is tuned to it, so we reuse it rather than ship our own copy. Falls back to a short prompt.
    """
    try:
        import comfyui_workflow_templates_json as pkg
        base = os.path.dirname(pkg.__file__)
    except Exception:
        return _PE_FALLBACK
    for path in sorted(glob.glob(os.path.join(base, "templates", "image_qwen_image_2_1*edit*.json"))):
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue
        for sg in data.get("definitions", {}).get("subgraphs", []):
            for node in sg.get("nodes", []):
                if node.get("type") == "PrimitiveStringMultiline" and "Prompt Enhancer" in str(node.get("widgets_values")):
                    return node["widgets_values"][0]
    return _PE_FALLBACK


# Sampling the Qwen 2.1 prompt enhancer is tuned for (its workflow template). Greedy decoding writes its
# reasoning into the answer; thinking mode tucks the answer inside the reasoning.
PE_SAMPLING = {"do_sample": True, "temperature": 1.0, "top_k": 20, "top_p": 0.95, "min_p": 0.0,
               "repetition_penalty": 1.0, "presence_penalty": 0.0}


_REASONING = ("the user", "let me", "okay", "first,", "we need", "i need", "draft", "wait", "also", "check",
              "one more", "i must", "the instruction", "hmm", "actually", "or\"", "avoid")


def looks_unfinished(text, limit_chars=2500):
    """The enhancer reasons in plain text for thousands of tokens before answering; cut off by max_length,
    the 'answer' is a piece of that reasoning ("Wait - ...", "Also ...", "Check draft: ...")."""
    t = text.strip()
    low = t.lower()
    return (not t or len(t) > limit_chars or low.startswith(_REASONING)
            or any(w in low for w in ("draft", "the instruction says", "let me ", "i must ")))


def rewrite(clip, prompt, system, image=None, seed=7, max_length=2048):
    """One LM rewrite with a ComfyUI text-generation CLIP (what core 'Generate Text' does), cleaned.
    Returns the brief itself when the LM rambles past max_length or never reaches an answer."""
    tokens = clip.tokenize(prompt, image=image, skip_template=False, min_length=1, thinking=False,
                           system_prompt=system)
    ids = clip.generate(tokens, max_length=max_length, seed=seed, mtp=True, **PE_SAMPLING)
    text = clean_lm_output(clip.decode(ids))
    return prompt if looks_unfinished(text) else text


def rewrite_all(clip, rig, frames, keyframes, frame_count, fps, mockup=True, video_images=None, seed=7,
                max_length=6144, progress=None, between=None):
    """The keyframe prompts and the video prompt from two LM rewrites.

    The Qwen 2.1 prompt enhancer reasons for 3-5k tokens before it answers, so one rewrite per keyframe
    was ~12 min for 8 keyframes. The keyframes differ only in where the light comes from (and the mock-up
    already shows where it lands), so one rewrite of the brief without direction is shared, and each
    keyframe appends its own direction sentence. frames: (T, H, W, 3) the clip the LM looks at.
    Returns (list of keyframe prompts, video prompt or "")."""
    n = 1 + (1 if video_images is not None else 0)
    mid = keyframes[len(keyframes) // 2]
    shared = rewrite(clip, keyframe_brief(rig, mid, frame_count, fps, mockup=mockup, direction=False),
                     qwen_edit_system_prompt(), frames[min(mid, frames.shape[0] - 1)][None], seed, max_length)
    if between:
        between()
    if progress:
        progress(1, n)
    prompts = []
    for f in keyframes:
        after = " ".join(t for t in (direction_words(rig, f, frame_count, fps), MOCKUP_NOTE if mockup else "",
                                     offscreen_note(rig, frame_count, fps, frames=[f])) if t)
        prompts.append(f"{shared} {after}".strip())
    video = ""
    if video_images is not None:
        after = offscreen_note(rig, frame_count, fps)
        text = rewrite(clip, video_brief(rig, frame_count, fps), VIDEO_LM_SYSTEM, video_images, seed, max_length)
        video = f"{text} {after}".strip()
        if progress:
            progress(n, n)
    return prompts, video


# ---------------------------------------------------------------- fixed templates (no LM)
# Generic on purpose: the clip may be live action, animation, painting or CG, with or without people, and
# the user writes their own prompt. The template only frames the task (change the light, keep everything
# else including the image's own style), says what the light stage set, and adds the user's text. No
# realism or skin wording: that would push stylised footage toward photographs. (The LM rewrite this
# replaced added nothing the rig did not say; Qwen-Image and H3 read the frames themselves.)
KEYFRAME_TASK = ("Relight this image. Change only the lighting: keep the content, every person and their identity, "
                 "expression and pose, the composition, the camera, and the image's own style and medium "
                 "(photographic, animated, painted or CG) exactly as they are.")


def _looks(rig):
    look = rig.get("look") or {}
    looks = [w for w in (LOOKS.get(look.get("preset", ""), look.get("preset", "")),
                         FILMS.get(look.get("film", ""), look.get("film", "")),
                         GRADES.get(look.get("grade", ""), look.get("grade", ""))) if w]
    return ", ".join(looks).capitalize() + "." if looks else ""


def _user(rig):
    text = (rig.get("user_prompt") or "").strip()
    return text if not text or text[-1] in ".!?。！？" else text + "."


def keyframe_prompt(rig, frame, frame_count, fps, mockup=True):
    """Qwen-Image 2.1 edit prompt for one keyframe, from the light stage alone: the task, the user's own words
    first, then the stage as a lighting plot in units (pro.py). A lighting reference image reaches the prompt
    as lights on the stage (Reference Lights): given to Qwen as image 2 it changed almost nothing."""
    lines = [KEYFRAME_TASK, _user(rig)]
    described = list(stage_lights(rig, frame_count, fps))
    if described:
        lines.append("Lighting setup:")
        lines += pro.setup_lines([l for l, _ in described], frame, frame_count, fps,
                                 originals={l.get("id") for l, o in described if o})
    removed = removed_original_words(rig, frame_count, fps)
    lines.append(removed)
    lines.append(pro.ambient_spec(rig))
    lines.append(_looks(rig))
    if mockup and (described or removed):
        lines.append(MOCKUP_NOTE)
    if described:
        lines.append(offscreen_note(rig, frame_count, fps, frames=[frame]))
    return " ".join(l for l in lines if l)


def video_prompt(rig, frame_count, fps):
    """MiniMax H3 prompt from the light stage alone: keep-everything preamble, the user's words, the lighting
    plot at the start and each light's timed changes, off-screen note."""
    lines = [PREAMBLE.format(ref="<Video 1>"), _user(rig)]
    described = list(stage_lights(rig, frame_count, fps, include_composite=False))
    lights = [l for l, _ in described]
    if lights:
        lines.append("Lighting setup at the start:")
        lines += pro.setup_lines(lights, 0, frame_count, fps, originals={l.get("id") for l, o in described if o})
        names, _, _ = pro.roles([evaluate_light(l, 0, frame_count, fps) for l in lights])
        for light, name in zip(lights, names):
            moves = pro.move_spec(light, frame_count, fps)
            if moves:
                lines.append(f"The {name}: {moves}.")
    lines.append(removed_original_words(rig, frame_count, fps))
    lines.append(pro.ambient_spec(rig))
    lines.append(_looks(rig))
    if lights:
        lines.append(offscreen_note(rig, frame_count, fps))
    return " ".join(l for l in lines if l)


# ---------------------------------------------------------------- optional LM polish (off by default)
# What the user asked the LM for: polish the user's own notes, and fill in what they leave out about the
# original's style, so the edit stays in that style. It never decides the lighting: that is the stage's.
POLISH_SYSTEM = (
    "You help a video relighting tool. You see one frame of the original video and the user's own notes.\n"
    "Reply with exactly two lines and nothing else:\n"
    "STYLE: up to 25 words on the original's visual style and medium (for example live-action film, anime, 3D "
    "animation, oil painting), its colour palette and contrast. Do not describe the lighting direction.\n"
    "NOTES: the user's notes rewritten as clear, concrete prompt language for an image or video model, keeping "
    "their meaning. Add no lights, directions or colours the user did not write. If the notes are empty, write NONE.")


def parse_polish(text):
    """'STYLE: ...\nNOTES: ...' -> (style, notes); missing parts come back empty."""
    style, notes = "", ""
    for line in text.splitlines():
        head, _, rest = line.partition(":")
        if head.strip().upper() == "STYLE":
            style = rest.strip()
        elif head.strip().upper() == "NOTES":
            notes = rest.strip()
    if notes.upper().strip(". ") == "NONE":
        notes = ""
    return style, notes


def polish_with_gemma(clip, frame, notes, max_length=300):
    """Built-in Gemma 4 (ComfyUI text generation), Gemma 4 chat format with one image."""
    prompt = (f"<|turn>system\n{POLISH_SYSTEM}<turn|>\n"
              f"<|turn>user\n<|image><|image|><image|>\n\nUser notes: {notes or '(empty)'}<turn|>\n"
              f"<|turn>model\n<|channel>final\n")
    tokens = clip.tokenize(prompt, image=frame, skip_template=True, min_length=1, thinking=False)
    ids = clip.generate(tokens, do_sample=False, max_length=max_length, temperature=1.0, top_k=50, top_p=1.0,
                        min_p=0.0, repetition_penalty=1.0, presence_penalty=0.0, seed=0, mtp=True)
    return clip.decode(ids)


def polish_with_lmstudio(url, model, frame, notes, max_tokens=300, timeout=180):
    """LM Studio's local OpenAI-compatible server; asks it to unload the model a minute after use (ttl) so it
    does not hold GPU memory while ComfyUI runs the image and video models."""
    import base64
    import io as _io
    import urllib.request

    from PIL import Image
    buf = _io.BytesIO()
    Image.fromarray((frame[0].clamp(0, 1).cpu().numpy() * 255).astype("uint8")).save(buf, format="JPEG", quality=90)
    body = {"messages": [{"role": "system", "content": POLISH_SYSTEM},
                         {"role": "user", "content": [
                             {"type": "text", "text": f"User notes: {notes or '(empty)'}"},
                             {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,"
                                                                       + base64.b64encode(buf.getvalue()).decode()}}]}],
            "max_tokens": max_tokens, "temperature": 0.2, "ttl": 60}
    if model:
        body["model"] = model
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)["choices"][0]["message"]["content"]


def apply_polish(prompt, user_notes, style, notes):
    """Swap the user's raw notes for the polished ones and add the original's style."""
    out = prompt
    if notes and user_notes and user_notes.strip() in out:
        out = out.replace(user_notes.strip(), notes)
    elif notes:
        out = f"{out} {notes}"
    if style:
        out = f"{out} Keep the original's look: {style.rstrip('.')}."
    return out


# ---------------------------------------------------------------- LM lighting director (2026-10-29)
# The light stage says what the lights are (directions, stops, colours, timings); it cannot say what they will
# look like on this footage. The director is a vision LM (Qwen3-VL 8B: Qwen-Image 2.1's own text encoder, so no
# extra model is loaded) that sees the keyframes' original frames (and a lighting reference or new background),
# reads the stage's plan, and describes the result in natural photographic language. First version kept the
# technical terms (stops, hex, angles, "crisp narrow rim"): Qwen drew them literally (a drawn outline, crushed
# blacks), while the user's few plain words ("light behind the person, natural, beautiful rim light") gave a
# natural rim. So the plan's numbers stay with the LM and the prompt is a plain visual description.
# The LM reads the user's intent and the light's effect on this footage (mood, character, path, what answers it) and has
# room to interpret: told to report where the light is and what it lights at which second, it wrote stiff prompts and the
# video came out stiff with them (the user, 2026-10-30). The guide frames hold the timing, so the video prompt has none.
# The answer still starts with fixed slots for each light: free sentences only repeated the settings.
# Asked for "beams in the haze", Qwen laid two see-through colour columns across the dancers while their skin stayed in
# the original's orange: the light looked pasted on, so the rules now say it belongs to the place (2026-10-30).
DIRECTOR_SYSTEM = """You are a cinematographer and gaffer. A user lit a shot on a light stage: a sphere around the people, seen from the camera. They want that lighting on their real footage, and you write what an image editor and a video model will follow.
You see the original frames (in order) and possibly one more image: a lighting reference or a new background. You get: what the scene looks like (read from the clip), each light the user set (the user's own words for it, its kind, colour, strength and path), how dark everything else should be, and the user's own request.

Read the footage and the user's lights together, work out what the user is after, and describe the light the way a film would show it. You are free to interpret, as long as every light keeps its colour, strength and path.

What to work out:
- The intent: the mood and tension of this shot, and the kind of film moment this lighting makes.
- What each light really is. The user's own words for it and their request decide what it is, where it comes from and what it is aimed at; where they say where it comes from, that replaces the stage's "in front" or "behind". Without such words, match the stage's place to the scene: a light behind the people comes from whatever is on the far side (through an opening, from outside or the next room; on a wall, from a lamp or lit surface there; in the open, from the sky or the sun), and a light in front comes from the camera's side of the space. Never invent a source the scene gives no reason for.
- Its character: how bright it is against the scene, its shape and reach, how hard its edge is, how it falls off.
- Its path: how it moves through the picture, the way a camera operator would say it: its direction and pace, any pause, flicker or breathing the user's words ask for.
- What in this footage answers the light: find the things in these frames with a particular surface or material, the ones that take light in their own way, and work out how each one really takes this light: what it reflects, lets through, bends, scatters or soaks up, what it bounces onto what is near, and the shadows it throws. Which things these are depends on this footage alone.

What is physically true:
- Light adds brightness; it does not paint. Where it lands, things get brighter than what is around them, with a shape and an edge, and its colour comes with that brightness. The rest of the picture does not turn that colour: it settles into the new lighting, its bounce and the environment it creates. A dim light does the same, only less bright.
- New light belongs to this place; it is never laid over the picture. It shapes each form with a lit side, a falloff and a shadow side, it takes over from the old light on whatever it reaches, so skin and clothes there carry its colour, and it spills onto what is around them.
- Whatever is transparent stays transparent: what is behind it stays visible unless a light far stronger than the scene's own fills it as well.
- A flood reaches everything that faces it. A beam reaches only what is in its path.
- Which way things face decides what the camera sees. A light from the far side shines toward the camera: it shows on top surfaces, edges and floors, and along the edges of hair, ears and shoulders. A light from the camera's side lands on what we see, and shadows fall away from it.
- Go by which way each person faces in the frames. A face the camera cannot see is never lit.
- How dark the places none of these lights reach get is set only by the line about everywhere else.
- Every light the user set shows in the result; the strongest one leads the look. The user's request adds what else changes (a new background, lights removed, the mood); it never replaces or dims a light the user set.
- Describe only the lights in the settings and what they cause: never add a rim, fill or any other light that is not listed. Visible beams in the air, haze or fog only if the user's words ask for a beam, haze or fog, and a beam seen in the air stays behind the people in the depth of the scene, never across them.
- The scene's own lights: the settings say which are kept, changed or switched off. One that is switched off or replaced is gone, and the subject keeps none of that light or its colour. Whatever the user asks to change or remove in the picture (a green screen, studio lights, their reflections or spill, a new background) is done: say it plainly.
- Describe the light, never the fixture: the lamps stay out of frame unless the user asks to see them. Keep everything already in the image, including objects that give off light themselves (lamps, screens, signs, headlights, windows).

How to write:
- Natural film language, the way a cinematographer talks about a shot: what it feels like and what the light does. Never a list of what is lit, and never when each thing is lit.
- No numbers, seconds, degrees, stops, hex codes, Kelvin or equipment names. Colours as the settings say (vivid stays vivid). Never use these words: tint, wash, stain, veil, overlay, hue, bathe, glow.
- Left and right always mean frame left / frame right, as the viewer sees the image.
- A rim from behind is natural light on the edges of hair, ears and shoulders, never an outline, a line or a halo.
- Refer to people and things generically for what they are in the frames. Do not invent objects or people, and do not change content, framing or style.
- A lighting reference: take its light (direction, colour, contrast, mood) as it falls on its subject, nothing else. A new background: light the subject as that place would.

Reply in exactly this format and nothing else:
SCENE: <one short sentence: the picture once the user's request is done>
MOOD: <one sentence: what this lighting should make the viewer feel>
LIGHT 1: <its name>. Real source: <what it is and where it really is>. Comes through: <which openings, or directly>. Look: <how bright, its shape and reach, its edge and falloff>. Path: <how it moves through the picture, in natural words>. Answers: <the few things with a particular surface or material in this footage, and how each one takes this light>.
LIGHT 2: ...
(one line per light the user set, strongest first)
VIDEO: <50-90 words of natural prose for a video model: the mood, the light's character and what answers it, and its movement as one continuous action with its pace, with any flicker or breathing the user's words ask for. The guide frames already hold the timing and the positions: no times, no mention of the clip or how far through it anything happens, and no account of what is lit when>"""

# One call per keyframe, with its own short rules: only that image, the mood, what each light is and where it is at that
# moment. The answer's first words are written here (moment_opening) and the LM goes on from them, looking at the image:
# told where the light is in a sentence of the request it retold the whole sweep instead, all three people lit in each
# keyframe and three identical texts on a face; asked to list what is lit and what is not it looped on the list until
# the token limit (2026-10-30).
DIRECTOR_KEYFRAME = """You write one instruction for an image editor that relights one film frame. You see the frame as it was shot. You are told the mood, what each light is, and where each light is at this moment.

Write 60 to 90 words of natural prose, the way a cinematographer describes a frame, going on from the words you are given:
- Follow the mood. Say how the light looks here: how bright, its shape and edge, how it falls off.
- Pick the few things in this image with a particular surface or material and say how each one really takes this light: what it reflects, lets through, bends, scatters or soaks up, and the shadow it throws. Do not go through everything that is lit.
- Light adds brightness and brings its colour with it; the rest of the frame settles into the new lighting and the environment it creates. A beam lights nothing outside its part of the picture.
- The light belongs to this place and is never laid over the image: it shapes each form with a lit side and a shadow side, it replaces the old light on the skin and clothes it reaches, and it spills onto what is around them. A beam seen in the air stays behind the people, never across them.
- Whatever is transparent stays transparent, unless a light much stronger than the scene's main light fills what is behind it as well.
- Then each other light in one short phrase.
- This is a still image: never write sweeps, moves, travels, swings, pulses or crosses.
- Someone seen from behind shows light only on hair, ears, shoulders and back.
- Frame left and frame right as the viewer sees it. No numbers, stops, hex codes, Kelvin or equipment names, and no lamp or fixture. Never use these words: tint, wash, stain, veil, overlay, hue, bathe, glow. No lists; never repeat a phrase.

Reply with exactly one line and nothing else:
KEYFRAME 1: <the text>"""

# Every director prompt ends with this unless the user asks to see the light source: told "a medium round panel
# light from frame left", Qwen drew the panel into the picture (2026-10-29).
# Only about the NEW lights: "no lamp ... appears anywhere in the image" made Qwen delete a car with its headlights
# on and dim a neon sign (2026-10-29); what is already in the picture stays, light sources included.
NO_SOURCE_NOTE = ("The new lights' sources are outside the frame: do not add any lamp, panel, softbox, bulb, sun or "
                  "beam for them, only their light on the subject and the scene. Apart from what the description asks to "
                  "change or remove, everything already in the image stays as it is, including objects that give off "
                  "light themselves (lamps, signs, car headlights, screens), which keep shining.")
_SHOW_SOURCE = ("光源入鏡", "燈入鏡", "燈具入鏡", "看得到燈", "看到燈", "看見燈", "出現燈", "燈出現", "光源出現",
                "出現光源", "拍到燈", "看得到光源", "看到光源", "露出燈", "visible lamp", "lamp in frame",
                "light in frame", "light source visible", "visible light source", "show the lamp", "show the light source",
                "lamp in shot", "in shot")


# words that make a light a beam, the first ones a beam the user wants to see (the rest light a pool: "spotlight beam"
# in the prompt made Qwen draw a cone from a lamp in the shot, 2026-10-28)
_VISIBLE_BEAM = ("光束", "探照", "searchlight", "beam")
_BEAM_WORDS = _VISIBLE_BEAM + ("聚光", "手電筒", "spotlight", "flashlight")


def wants_visible_source(rig):
    """The user's own words (prompt box, per-light notes) ask for the light source itself in the picture."""
    text = " ".join([rig.get("user_prompt") or ""] + [l.get("prompt") or "" for l in rig.get("lights", [])]).lower()
    return any(k.lower() in text for k in _SHOW_SOURCE)


# the director's keyframe prompts start with this instead of the long KEYFRAME_TASK
DIRECTOR_TASK = ("Relight this image as described, keeping the people, their pose and expression, the composition, the "
                 "camera and the style; change nothing else unless the description asks for it.")
# A strong coloured light over a darkened scene came out as a milky colour veil with no skin texture (2026-10-29):
# the image's quality is asked for outright on every keyframe.
# "coloured light tints only what it lights" asked for a tint, and Qwen painted one. Fixed blacks and "keeps its own colour"
# held the picture to the original: the blacks, contrast and colour of the rest belong to the new lighting (2026-10-30).
QUALITY_NOTE = ("Keep all the image's detail and texture, with no {air}milky layer over the picture; its blacks, contrast "
                "and the colour of the areas outside the light follow the new lighting and the environment it creates.")

DIRECTOR_REF = {"lighting reference": "The last image is a LIGHTING REFERENCE. Read its light yourself: where the light "
                                      "falling on its main subject comes from (frame left or right, front, behind, above), "
                                      "its exact colour (e.g. blue-violet rather than just blue), how hard or soft it is, "
                                      "its shape (an even wash, a soft pool, or sharp-edged patches and patterns of light "
                                      "as through a cutout, blinds or leaves, and where those patches fall), the contrast, "
                                      "how dark the scene is. Only light "
                                      "that falls on the subject counts: the light's colour is the colour of the subject's "
                                      "lit edges and surfaces, never of parts that glow by themselves (eyes, veins, a glowing "
                                      "chest, signs, screens, lightning bolts, lamps), which are not lights on our subject. "
                                      "Light our footage "
                                      "that way and take nothing else from the reference (no content, objects or style).",
                "new background": "The last image is the NEW BACKGROUND the subject will be placed in: light the subject "
                                  "as that environment would.",
                "green screen": "The footage was shot against a green screen. The last image is the NEW BACKGROUND that "
                                "replaces every green area (behind the subject, through windows and glass). Describe the "
                                "scene as it will look in that place: the light it gets from that environment (its "
                                "direction, softness, colour and brightness), with no green spill left anywhere; do not "
                                "describe the green screen."}

# Green screen replaced by the generator itself (2026-10-29): keying, pasting the plate and relighting after looked
# cut out (hard edges, spill, a plate that never sat in the scene). Qwen gets the green-screen frame and the plate
# (image 2) and does the replacement and the relight in one edit, so the subject and the new place are lit as one.
GREENSCREEN_TASK = ("Replace the green screen with the background from image 2 and relight the whole image to match "
                    "it, keeping every person, their pose and expression, the camera and every object in the foreground "
                    "exactly as they are.")
GREENSCREEN_NOTE = ("Every green area (behind the subject, through windows and glass, reflected in glossy surfaces) "
                    "becomes the background of image 2 with matching perspective, scale and focus; no green screen, "
                    "tracking marker or green spill remains on the subject or the set.")
GREENSCREEN_VIDEO = ("The green screen of <Video 1> is replaced by the background shown in the guide frames in every "
                     "frame; no green, tracking marker or green spill remains.")


def _strip(text, *parts):
    for p in parts:
        if p:
            text = text.replace(p, "")
    return " ".join(text.split())


# The LM gets plain facts, not the template's prose: given "crisp, well-defined shadow edges (narrow penumbra)"
# it wrote "a crisp, well-defined rim" whatever its rules said, and Qwen drew a line around the subject.
def _plain_color(light, kelvin):
    import colorsys
    hexc = (light.get("color") or {}).get("hex")
    if hexc:
        h = hexc.lstrip("#")
        r, g, b = (int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
        hue, lum, sat = colorsys.rgb_to_hls(r, g, b)
        if sat < 0.12:
            return "white"
        name = next(n for limit, n in pro._HUES if hue * 360.0 < limit)
        return ("pale " if lum > 0.75 else "deep " if lum < 0.3 else "vivid " if sat > 0.6 else "") + name
    k = kelvin or 5600
    return ("warm amber" if k < 3000 else "warm" if k < 4300 else "neutral white" if k < 6200
            else "slightly cool" if k < 8000 else "cool, bluish white" if k < 11000 else "cool blue")


def dirOf_py(k):
    """Unit direction of a key (the stage's convention: x right, y up, z toward the camera)."""
    az, el = math.radians(k["azimuth"]), math.radians(k["elevation"])
    return (math.sin(az) * math.cos(el), math.sin(el), math.cos(az) * math.cos(el))


def _narrow_beam(light):
    # the stage's spotlight starts at 20 degrees; 30 used to be its default and read as a side light, not a beam.
    # A beam is also what the user calls one, or a small hard source: a red "searchlight" set as a small panel was told
    # as "a small round source of broad light" and came out as a red stain (2026-10-30)
    if light.get("type") == "spot" and float((light.get("cone") or {}).get("angle", 40)) <= 35:
        return True
    if light.get("type") == "area" and float(light.get("size") or 1.0) < 0.4 and float(light.get("softness", 0.5)) < 0.3:
        return True
    return any(w in (light.get("prompt") or "").lower() for w in _BEAM_WORDS)


def _pool_part(az):
    """Which part of the frame a narrow beam's pool covers, from the side the light comes from. Five places, not
    three: a left-to-right sweep over three places jumped from one to the next (2026-10-30)."""
    return ("the frame-left part of the scene" if az < -36 else "the area just left of the centre of the scene"
            if az < -13 else "the centre of the scene" if az <= 13 else "the area just right of the centre of the scene"
            if az <= 36 else "the frame-right part of the scene")


def _pool_size(light):
    a = float((light.get("cone") or {}).get("angle", 40)) if light.get("type") == "spot" else 20.0
    return ("about a tenth of the frame wide" if a <= 15 else "about a fifth of the frame wide" if a <= 25
            else "about a third of the frame wide")


def _plain_ambient(rig):
    """How dark everything no light reaches gets: the user's own ambient setting, and only that (thresholds on a dim
    light's level gave "the frame-left side falls into deep shadow" though the user had set the room only a little
    darker, 2026-10-30)."""
    amb = rig.get("ambient") or {}
    ev = float(amb.get("intensity_ev", 0.0))
    k = float(amb.get("kelvin", 6500))
    if not amb or abs(ev) < 0.25:
        look = "it keeps the original's own light"
    elif ev > 0:
        look = "a little brighter than the original" if ev < 1.5 else "clearly brighter than the original"
    elif ev > -1.5:
        look = "a little darker than the original, every detail still clear"
    elif ev > -3:
        look = "clearly darker than the original, shapes still readable"
    else:
        look = "very dark, close to a silhouette"
    # only a clear tint is worth a word: "a warm tint" for 4800K made Qwen light the face warm
    tint = "" if 4000 <= k <= 8500 else (", with a warm tint" if k < 4000 else ", with a cool tint")
    return f"Everywhere none of these lights reach: {look}{tint}."


def _stage_place(az, el, dist=None):
    """Where a light sits on the stage, in words the LM can match to the scene: the stage's front is the camera's
    side, its back the far side of the scene behind the people (the background)."""
    a = ((az + 180) % 360) - 180
    side = "frame-left" if a < 0 else "frame-right"
    where = ("on the camera's side, in front of the people" if abs(a) <= 25
             else f"in front of the people, off to the {side}" if abs(a) <= 70
             else f"at the {side} side of the people" if abs(a) <= 110
             else f"behind the people toward the {side}: on the far side of the scene, the background side" if abs(a) <= 155
             else "straight behind the people: on the far side of the scene, the background side")
    height = ("high overhead" if el >= 60 else "above them, shining down" if el >= 25
              else "low, shining up from below" if el <= -15 else "around head height")
    near = ", very close to the people" if dist is not None and dist <= 1.0 else ""
    return f"{where}, {height}{near}"


def _strength(ev):
    """A light's level against the scene's main light (stops above _level_ref), in words."""
    return ("much stronger than the scene's main light: it overpowers it where it lands" if ev >= 2 else
            "stronger than the scene's main light" if ev >= 0.75 else
            "about as strong as the scene's main light" if ev > -0.75 else
            "weaker than the scene's main light" if ev > -2 else
            "dim: it adds only a little brightness where it lands")


def _peak(light, frame_count, fps):
    return max(evaluate_light(dict(light, motion=None), f, frame_count, fps)["intensity"]
               for f in range(0, max(frame_count, 1), max(frame_count // 8, 1)))


def _level_ref(rig, lights, frame_count, fps):
    """The level the lights are told against. Levels are stops from the scene's main light as detected; once the user
    removes that light they are told against what is left: with the green screen gone, a searchlight at -2 stops, the
    strongest light left, was told as faint and came out as a red stain (2026-10-30). lights: (light, name) pairs."""
    stage = rig.get("lights", [])
    if not rig.get("removed_originals") and not any(l.get("original") and not l.get("enabled", True) for l in stage):
        return 0.0
    kept = [l["detected"] for l in stage if l.get("original") and l.get("enabled", True) and l.get("detected")]
    if kept:
        return max(filled_keys(l, frame_count, fps)[0]["intensity_ev"] for l in kept)
    return max((math.log2(max(_peak(l, frame_count, fps), 1e-4)) for l, _ in lights), default=0.0)


def _light_kind(light):
    """What kind of light it is, as set on the stage (the fixture itself stays out of frame)."""
    kind = light.get("type", "spot")
    soft = float(light.get("softness", 0.5))
    edges = "soft, gentle shadows" if soft > 0.6 else "hard, clear-edged shadows" if soft < 0.3 else "fairly soft shadows"
    if _narrow_beam(light):
        return f"a narrow beam like a searchlight: it lights a pool {_pool_size(light)}, not everything facing it ({edges})"
    if kind == "area":
        d = float((light.get("keys") or [{}])[0].get("distance", 1.5))
        size = light.get("size") or 2 * d * math.tan(math.radians((2 + 43 * soft) / 2))
        word = "small" if size < 0.4 else "medium" if size < 1.0 else "large"
        shape = "round" if light.get("shape") == "disc" else "square"
        return f"a {word} {shape} source of broad light ({edges})"
    if kind == "spot":
        return f"a wide spotlight beam with a defined edge ({edges})"
    if kind == "point":
        return f"a bare bulb: bright close to it, falling off quickly with distance ({edges})"
    return "sunlight-like parallel light: equally strong everywhere it reaches, hard shadows"


_EASE_WORDS = {"linear": "at a steady speed", "smooth": "smoothly", "easeInOut": "starting and ending slowly",
               "easeIn": "starting slowly", "easeOut": "slowing down at the end", "hold": "staying put, then jumping"}


def _across(az, el):
    """Where a direction sits across the picture, whatever its depth."""
    x = math.sin(math.radians(az)) * math.cos(math.radians(el))
    return "frame-left" if x < -0.3 else "frame-right" if x > 0.3 else "the middle"


def _turn(k0, k1):
    """Degrees between the directions of two keys."""
    return math.degrees(math.acos(max(-1.0, min(1.0, sum(a * b for a, b in zip(dirOf_py(k0), dirOf_py(k1)))))))


def _when(part):
    """A moment of the clip in words (part: 0..1): seconds in the brief came back as "from 0.0s to 1.3s ..." in the video
    prompt, and the keyframes hold the timing anyway (2026-10-30)."""
    return ("early in the clip" if part < 0.2 else "about a third of the way through the clip" if part < 0.42
            else "about halfway through the clip" if part < 0.6 else "about two-thirds of the way through the clip"
            if part < 0.8 else "near the end of the clip")


def _pace(deg_per_sec):
    """A light's speed in words: the frame is only some 55 degrees wide, so 45 degrees a second crosses it in about a second."""
    return "slowly" if deg_per_sec < 20 else "steadily" if deg_per_sec < 45 else "quickly"


def _light_brief(light, name, frame_count, fps, ref=0.0):
    """One light over the whole clip: the user's words for it first (they say what it really is, and read after the
    settings they lost to them), kind, colour, strength, where it is and how it moves, when it is on."""
    from .rig import original_changed
    last = max(frame_count - 1, 0)
    plain = dict(light, motion=None)
    on = [f for f in range(last + 1) if evaluate_light(plain, f, frame_count, fps)["intensity"] > 0] or [0]
    first, end = evaluate_light(plain, on[0], frame_count, fps), evaluate_light(plain, on[-1], frame_count, fps)
    k = filled_keys(light, frame_count, fps)
    if k[-1]["_frame"] > last:   # keys past the clip's end (an end set at 2.0 s on a 1.96 s clip) stop at its last frame
        k = [x for x in k if x["_frame"] < last] + [dict(k[-1], _frame=float(last), azimuth=end["azimuth"],
                                                         elevation=end["elevation"])]
    who = name
    if light.get("original"):
        src = (light.get("detected") or {}).get("source") or light.get("source") or ""
        who += f" (the scene's own light{': ' + src if src else ''}, " + (
            "changed by the user)" if original_changed(light, frame_count, fps) else "kept as it is in the clip)")
    colour = _plain_color(light, k[0].get("kelvin"))
    ev0, ev1 = (math.log2(max(s["intensity"], 1e-4)) - ref for s in (first, end))
    note = (light.get("prompt") or "").strip()
    line = (f"{who}: " + (f"the user's own words for it: {note}. Its settings: " if note else "")
            + f"{_light_kind(light)}; {colour} light; {_strength(ev0)}; "
            + _stage_place(first["azimuth"], first["elevation"], first["distance"]))
    # the whole path, not its two ends: a light that goes out and comes back ends where it started
    if sum(_turn(a, b) for a, b in zip(k, k[1:])) > 10:
        # turning points of a path (an old recording has a key every few frames; >= 25 degrees apart are kept)
        main = [k[0]]
        for kk in k[1:-1]:
            if _turn(main[-1], kk) >= 25:
                main.append(kk)
        main.append(k[-1])
        steps = []
        for k0, k1 in zip(main, main[1:]):
            end = min(k1["_frame"], last)
            if _turn(k0, k1) < 3:
                steps.append("then it holds there")
                continue
            speed = _turn(k0, k1) / max((end - k0["_frame"]) / fps, 1e-3)
            steps.append(f"it moves {_pace(speed)}, {_EASE_WORDS.get(k0.get('ease', 'smooth'), 'smoothly')}, to "
                         f"{_stage_place(k1['azimuth'], k1['elevation'], k1['distance'])}, arriving {_when(end / max(last, 1))}")
        line += ". Its path: " + "; ".join(steps)
        sides = [_across(x["azimuth"], x["elevation"]) for x in main]
        sides = [s for i, s in enumerate(sides) if i == 0 or s != sides[i - 1]]
        if len(sides) > 1:
            line += "; across the picture it travels from " + " to ".join(sides)
    if _strength(ev1) != _strength(ev0):
        line += f"; by the end it is {_strength(ev1).split(':')[0]}"
    start, stop = light.get("start_sec"), light.get("end_sec")
    if start is not None and float(start) * fps > 0.5:
        line += f"; it is off at first and comes on {_when(float(start) * fps / max(last, 1))}"
    if stop is not None and float(stop) * fps < last - 0.5:
        line += f"; it goes out {_when(float(stop) * fps / max(last, 1))} and stays off"
    return line + "."


# faces too small or absent have no pattern to light; the LM copied one onto a man seen from behind (2026-10-30)
_NO_FACES = re.compile(r"^Faces:\s*(none|small)\b", re.I | re.M)


def director_facts(rig, frame, frame_count, fps, reference=False, scene=""):
    """Where each light the user set is at one frame (the light's kind, colour and path are in stage_brief)."""
    lights = _brief_lights(rig, frame_count, fps, reference)
    ref = _level_ref(rig, lights, frame_count, fps)
    out = []
    for light, name in lights:
        st = evaluate_light(dict(light, motion=None), frame, frame_count, fps)
        if st["intensity"] <= 0:
            out.append(f"{name}: off at this moment.")
            continue
        line = (f"{name}: {_stage_place(st['azimuth'], st['elevation'], st['distance'])}; "
                f"{_strength(math.log2(st['intensity']) - ref).split(':')[0]}")
        a = ((st["azimuth"] + 180) % 360) - 180
        if _narrow_beam(light) and abs(a) < 120:
            line += f"; seen from the camera its pool falls roughly on {_pool_part(a)}"
        elif _across(st["azimuth"], st["elevation"]) != "the middle":
            line += f"; it is strongest toward the {_across(st['azimuth'], st['elevation'])} of the picture"
        out.append(line + ".")
    pattern = "" if _NO_FACES.search(scene or "") else _key_pattern(
        [(l, evaluate_light(dict(l, motion=None), frame, frame_count, fps)) for l, _ in lights], ref)
    return " ".join(out + ([pattern] if pattern else [])) or "No light from the stage at this moment."


def _key_pattern(states, ref=0.0):
    """Which side of a face the strongest light in front lights, and how dark the other side is. Not a key: lights
    behind (rims; a scene lit mainly from behind has no key in front: a faint cabin fill under a sunset became a
    Rembrandt key), very faint lights (an accent), and a beam unless it is clearly bright (a faint red beam is a pool,
    said with the light; a car's headlight beam on a face is its key). Only for faces the camera sees: said of "the
    people" in a shot of backs of heads, it put the red on a cap (2026-10-30). ref: see _level_ref."""
    wrap = lambda az: ((az + 180) % 360) - 180  # noqa: E731
    on = [(l, math.log2(st["intensity"]) - ref, wrap(st["azimuth"]), st["elevation"]) for l, st in states if st["intensity"] > 0]
    # behind = past 110 degrees, as _stage_place says it
    front = [(ev, az, el) for l, ev, az, el in on if abs(az) <= 110 and (not _narrow_beam(l) or ev > -1)]
    back = [ev for l, ev, az, el in on if abs(az) > 110]
    if not front or max(front)[0] <= -2.5 or (back and max(back) > max(front)[0] + 1):
        return ""
    kev, kaz, kel = max(front)
    who = "On a face the beam reaches" if any(_narrow_beam(l) and ev == kev and az == kaz for l, ev, az, _ in on) \
        else "On any face the camera can see"
    lit, dark = ("frame-left", "frame-right") if kaz < 0 else ("frame-right", "frame-left")
    other = [ev for ev, az, _ in front if az * kaz < 0 and abs(az) > 10]
    if other and kev - max(other) < 0.8:
        return (f"{who}: two-sided light, each side lit by its own light in that light's colour; neither side "
                "is in deep shadow.")
    if kel <= -20 and abs(kaz) < 45:
        shape = "light from below: the undersides of every form are lit (chin, jaw, under the nose), shadows fall upward"
    elif kel >= 60:
        shape = "top light: the tops of forms are lit (head, brow, nose bridge), the eye sockets and under the chin fall into shadow"
    elif abs(kaz) < 25:
        shape = "near-frontal light: both sides lit about equally, the shading runs from the lit front to the edges"
    elif 25 <= abs(kaz) <= 60 and 30 <= kel <= 60:
        shape = (f"a Rembrandt-style key, high and to one side: the {lit} side of the face is lit, the {dark} side is "
                 "in shadow with a small triangle of light on that cheek")
    elif 75 <= abs(kaz) <= 105 and not other:
        shape = f"split light: the {lit} half of the face is lit, the {dark} half is in shadow"
    else:
        shape = f"side light from the {lit}: the {lit} side of the face and body is lit, the {dark} side turns away into shadow"
    if abs(kaz) < 25:
        rest = ""
    elif other:
        d = kev - max(other)
        rest = f"; the {dark} side is about {d:.0f} stops darker (about {2 ** d:.0f}:1), darker but with detail"
    else:
        rest = f"; nothing fills the {dark} side: it is as dark as everywhere else (the line about everywhere else)"
    return f"{who}: {shape}{rest}."


def _brief_lights(rig, frame_count, fps, reference=False):
    """(light, name) for every light on the stage the director hears about: enabled ones, the scene's own included
    (kept or changed), the reference's estimates left out when the LM sees the reference itself (from a stylised
    reference the estimate took a glowing dragon's chest for a red front light)."""
    lights = [l for l in rig.get("lights", []) if l.get("enabled", True) and not (reference and l.get("reference"))]
    names = []
    for i, l in enumerate(lights):
        n = (l.get("name") or f"light {i + 1}").strip()
        names.append(n if n not in names else f"{n} ({i + 1})")
    # the strongest first: listed after two faint lights, a +4 stop violet light was taken for faint daylight
    # (2026-10-30)
    return sorted(zip(lights, names), key=lambda ln: -_peak(ln[0], frame_count, fps))


def stage_brief(rig, frame_count, fps, reference=False, scene=""):
    """What the director plans from: the scene, the user's lights in words it can match to the scene, the scene's own
    lights switched off, how dark the rest gets and the user's request. Where each light is at each keyframe goes only
    to that keyframe's own call: the plan and the video prompt are about the light, not its position in each frame."""
    out = []
    if (scene or "").strip():
        out.append(f"The scene (read from the clip): {scene.strip()}")
    if reference and any(l.get("reference") for l in rig.get("lights", [])):
        out.append("The main lighting comes from the lighting reference: read it from that image.")
    lights = _brief_lights(rig, frame_count, fps, reference)
    ref = _level_ref(rig, lights, frame_count, fps)
    out.append("The lights the user set:" if lights else "The user set no light.")
    out += [f"- {_light_brief(l, n, frame_count, fps, ref)}" for l, n in lights]
    gone = list(rig.get("removed_originals") or [])
    gone += [l for l in rig.get("lights", []) if l.get("original") and not l.get("enabled", True)]
    for g in gone:
        k0 = filled_keys(g.get("detected") or g, frame_count, fps)[0]
        src = (g.get("detected") or g).get("source") or ""
        out.append(f"- The scene's own light {_stage_place(k0['azimuth'], k0['elevation'])}"
                   f"{' (' + src + ')' if src else ''} is switched off by the user: it is gone.")
    out.append(_plain_ambient(rig))
    if (rig.get("user_prompt") or "").strip():
        out.append(f"The user's request: {rig['user_prompt'].strip()}")
    return "\n".join(out)


def director_request(rig, keyframes, frame_count, fps, reference=None, scene=""):
    """The first user turn: which image is which, then the brief."""
    out = [f"Images 1-{len(keyframes)}: the original frames of the clip, in order."]
    if reference:
        out.append(DIRECTOR_REF[reference])
    out.append(stage_brief(rig, frame_count, fps, reference=bool(reference), scene=scene))
    return "\n\n".join(out)


def moment_opening(rig, frame, frame_count, fps, reference=False):
    """The first words of a keyframe's text: where the light that matters most is at this moment, from the stage, not
    from the LM. A beam close enough to the strongest light goes first: where it lands is what tells the moments apart.
    It says no more than that: "comes in through the windows" painted the panes solid red, "a shaft of light" drew a thin
    red laser line down the frame, and "window glass stays clear" made the beam all but vanish (2026-10-30)."""
    on = [(l, evaluate_light(dict(l, motion=None), frame, frame_count, fps))
          for l, _ in _brief_lights(rig, frame_count, fps, reference)]
    on = [(l, st) for l, st in on if st["intensity"] > 0]
    if not on:
        return ""
    top = max(st["intensity"] for _, st in on)
    wrap = lambda az: ((az + 180) % 360) - 180  # noqa: E731
    beams = [o for o in on if _narrow_beam(o[0]) and abs(wrap(o[1]["azimuth"])) < 120 and o[1]["intensity"] >= top / 4]
    light, st = beams[0] if beams else max(on, key=lambda o: o[1]["intensity"])
    colour = _plain_color(light, filled_keys(light, frame_count, fps)[0].get("kelvin"))
    if beams:
        return f"In this frame the {colour} beam lands only on {_pool_part(wrap(st['azimuth']))}, where"
    side = _across(st["azimuth"], st["elevation"])
    # a flood that overpowers the scene from the far side is the one light that changes the background too; left to the LM
    # a +4 stop violet light through the windows left the sea "unchanged" (2026-10-30)
    lights = _brief_lights(rig, frame_count, fps, reference)
    if abs(wrap(st["azimuth"])) > 110 and math.log2(top) - _level_ref(rig, lights, frame_count, fps) >= 2:
        where = "" if side == "the middle" else f", strongest toward the {side} of the picture"
        return f"In this frame the {colour} light overpowers everything from the far side{where}, fills the background, and"
    return (f"In this frame the {colour} light" if side == "the middle"
            else f"In this frame the {colour} light is strongest toward the {side} of the picture, where")


def _whole_sentences(text):
    """Text cut off by the token limit, back to its last full sentence."""
    text = text.strip()
    end = max(text.rfind(c) for c in ".!?")
    return text[:end + 1] if end > 0 and len(text) - end < 200 else text


def keyframe_request(rig, frame, frame_count, fps, plan, reference=None, scene=""):
    """One keyframe's user turn: its image, the plan without the lights' paths (a still has none) and where each light
    is at this moment."""
    out = ["The image: one frame of the clip."]
    if reference:
        out.append(DIRECTOR_REF[reference])
    what = "\n".join(re.sub(r"\bPath:.*?(?=\bAnswers:|$)", "", line).strip() for line in plan.splitlines())
    out.append(f"The scene, the mood and each light:\n{what}")
    out.append(_plain_ambient(rig))
    if (rig.get("user_prompt") or "").strip():
        out.append(f"The user's request: {rig['user_prompt'].strip()}")
    out.append(f"Where each light is at this moment: {director_facts(rig, frame, frame_count, fps, bool(reference), scene)}")
    return "\n\n".join(out)


def parse_plan(text):
    """The SCENE, MOOD and LIGHT lines of the director's answer, as one text ('' when missing)."""
    text = text.split("</think>")[-1]
    got = re.findall(r"^\s*\**\s*(SCENE|MOOD|LIGHT\b[^:*\n]*)\s*\**\s*:\s*\**\s*(.+)$", text, re.I | re.M)
    return "\n".join(f"{k.strip().upper()}: {v.strip()}" for k, v in got)


def parse_director(text, n):
    """'KEYFRAME i: ...' lines and a 'VIDEO: ...' line -> ([n keyframe texts or None], video text or None)."""
    text = text.split("</think>")[-1]
    parts, cur = {}, None
    for line in text.splitlines():
        m = re.match(r"\s*\**\s*(KEYFRAME\s*(\d+)|VIDEO)\s*\**\s*:\s*\**\s*(.*)", line, re.I)
        if m:
            cur = f"k{int(m.group(2))}" if m.group(2) else "video"
            parts[cur] = m.group(3).strip()
        elif cur and line.strip():
            parts[cur] += " " + line.strip()
    return [parts.get(f"k{i + 1}") or None for i in range(n)], parts.get("video") or None


_BARE_SIDE = re.compile(r"(?<![-\w])(left|right)(?=[ -](cheek|eye|ear|side|half|shoulder|temple|jaw|edge|brow|"
                        r"forehead|hand|arm|profile)s?\b)", re.I)


def frame_sides(text):
    """'the left cheek' -> 'the frame-left cheek': Qwen may read a bare left/right as the subject's own side,
    the mirror of what the stage set (the LM writes them despite being told to say frame left / right)."""
    return _BARE_SIDE.sub(lambda m: "frame-" + m.group(1).lower(), text)


def director_checks(text):
    """A usable answer: a real description, not a stub or a ramble."""
    return 12 <= len((text or "").split()) <= 220


_HAZE_WORDS = ("霧", "煙", "光暈", "眩光", "haze", "fog", "mist", "smoke", "volumetric", "god ray", "glow", "bloom") + _VISIBLE_BEAM


def direct(clip, rig, frames, keyframes, frame_count, fps, reference_image=None, reference_kind="lighting reference",
           mockup=True, seed=7, max_length=1600, between=None, scene=""):
    """The director's prompts. frames: the original frames at the keyframes (K, H, W, 3), small is fine. scene: the
    clip read by Detect Original Light (what is where, what is behind the people).

    One call first reads the user's intent in this scene, light by light (SCENE, MOOD, LIGHT n: fixed slots, since a free
    sentence only repeated the settings), and writes the video prompt. Told only "a purple panel light behind the people",
    it wrote a purple glow behind them, not light pouring in through the windows that were there (2026-10-30). Each
    keyframe is then written from that plan with its own image and moment, so they follow one light on one path.
    Returns (keyframe prompts, video prompt, raw LM text, notes); what the LM gets wrong falls back to the template."""
    kf_templates = [keyframe_prompt(rig, f, frame_count, fps, mockup=mockup) for f in keyframes]
    video_template = video_prompt(rig, frame_count, fps)
    images = [frames[i:i + 1] for i in range(frames.shape[0])]
    kind = None
    if reference_image is not None:
        images.append(reference_image[:1])
        kind = reference_kind
    vision = "<|vision_start|><|image_pad|><|vision_end|>"
    calls = []

    def ask(imgs, request, system, opening="", limit=max_length):
        # a second generate in the same node needs ComfyUI's between-nodes reset (nodes._reset_between_runs),
        # or it crashes on a CUDA index assert
        if calls and between:
            between()
        calls.append(1)
        text = (f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n" + vision * len(imgs)
                + f"\n{request}<|im_end|>\n<|im_start|>assistant\n{opening}")
        tokens = clip.tokenize(text, images=imgs)
        ids = clip.generate(tokens, do_sample=False, max_length=limit, temperature=1.0, top_k=50, top_p=1.0,
                            min_p=0.0, repetition_penalty=1.05, seed=seed)
        return clip.decode(ids)

    ref = images[len(keyframes):]
    brief = stage_brief(rig, frame_count, fps, reference=bool(kind), scene=scene)
    raw = ask(images, director_request(rig, keyframes, frame_count, fps, kind, scene=scene), DIRECTOR_SYSTEM)
    video_lm = parse_director(raw, 0)[1]
    plan = parse_plan(raw)
    kf_lm = []
    for i, f in enumerate(keyframes):
        head = moment_opening(rig, f, frame_count, fps, bool(kind))
        r = ask([images[i]] + ref, keyframe_request(rig, f, frame_count, fps, plan or brief, kind, scene), DIRECTOR_KEYFRAME,
                opening=f"KEYFRAME 1: {head}", limit=240)
        body = re.sub(r"^\s*\**\s*KEYFRAME\s*\d*\s*\**\s*:\s*\**", "", r.split("</think>")[-1].strip())
        text = _whole_sentences(f"{head} {body}".strip())
        raw += f"\n\nKEYFRAME {i + 1}: {text}"
        kf_lm.append(text)
    # "The lighting remains unchanged from Keyframe 1: ..." despite the rule: Qwen sees one image, the reference to
    # another keyframe means nothing to it, so that opening goes
    kf_lm = [re.sub(r"^[^.:]*\b(unchanged|identical|the same|same as)\b[^.:]*\bkeyframe\s*\d+\s*[.:]?\s*", "", t,
                    flags=re.I) if t else t for t in kf_lm]
    kf_lm = [frame_sides(t[:1].upper() + t[1:]) if t else t for t in kf_lm]
    video_lm = frame_sides(video_lm) if video_lm else video_lm
    described = any(True for _ in stage_lights(rig, frame_count, fps))
    words = " ".join([rig.get("user_prompt") or ""] + [l.get("prompt") or "" for l in rig.get("lights", [])]).lower()
    no_source = "" if wants_visible_source(rig) or not described else NO_SOURCE_NOTE
    if any(w in words for w in _VISIBLE_BEAM):   # asked for in the system prompt, then forbidden here (2026-10-30)
        no_source = no_source.replace("bulb, sun or beam", "bulb or sun")
    # the user's own words go on every prompt as written: with one LM call per keyframe the LM dropped "replace
    # the green screen" from one keyframe and that keyframe came back green (2026-10-29)
    asked = (rig.get("user_prompt") or "").strip()
    asked = f"User's request (must be followed): {asked}" if asked else ""
    # the user's own setting, so it goes on every prompt as well: left to the LM it was dropped, and -1 and -2.5 stops of
    # ambient came out equally dark (2026-10-30)
    ambient = _plain_ambient(rig) if abs(float((rig.get("ambient") or {}).get("intensity_ev", 0))) >= 0.25 else ""
    quality = QUALITY_NOTE.format(air="" if any(w in words for w in _HAZE_WORDS) else "haze, fog, glow, bloom or ")
    notes, kf_out = [], []
    gs = kind == "green screen"
    for i, (lm_text, tmpl) in enumerate(zip(kf_lm, kf_templates)):
        if lm_text and director_checks(lm_text):
            kf_out.append(" ".join(t for t in [GREENSCREEN_TASK if gs else DIRECTOR_TASK, lm_text, ambient, asked,
                                               GREENSCREEN_NOTE if gs else "", MOCKUP_NOTE if mockup and described else "",
                                               no_source, quality] if t))
        else:
            kf_out.append(tmpl)
            notes.append(f"keyframe {i + 1}: LM output unusable, template used")
    if video_lm and director_checks(video_lm):
        video_out = " ".join(t for t in [PREAMBLE.format(ref="<Video 1>"), video_lm, ambient, asked,
                                         GREENSCREEN_VIDEO if gs else "", no_source] if t)
    else:
        video_out = video_template
        notes.append("video: LM output unusable, template used")
    return kf_out, video_out, raw, notes
