"""LightRig -> text the video model can follow, with timestamps for moving lights.

Deterministic templates on purpose: the same rig always gives the same prompt,
so a result can be reproduced and compared. The LM step that rewrites these
into final prompts lives in lm.py.
"""

from .rig import filled_keys

PREAMBLE = ("Keep the content, every person's face, identity, wardrobe and motion, the framing, camera movement, "
            "timing and the video's own style and medium exactly the same as {ref}. Only the lighting changes.")

LOOKS = {
    "neo_noir": "neo-noir mood, deep shadows, high contrast",
    "natural": "natural documentary color",
    "warm_film": "warm cinematic film look",
    "cool_scifi": "cool science-fiction film look",
    "high_key": "bright high-key look, soft shadows",
    "low_key": "dramatic low-key look, mostly darkness",
}
FILMS = {
    "tungsten_500t": "tungsten 500T film stock, fine grain, halation around highlights",
    "daylight_250d": "daylight 250D film stock",
    "bw_400": "black-and-white 400 film",
}
GRADES = {
    "teal_orange": "teal and orange grade",
    "bleach_bypass": "bleach bypass grade, desaturated",
    "pastel": "soft pastel grade",
}

TYPE_WORDS = {
    "spot": "spotlight",  # not "beam": it made the models draw a visible cone from a lamp in the shot
    "point": "practical light",
    "area": "soft panel light",
    "directional": "sunlight-like directional light",
}


def direction_words(az, el):
    a = ((az + 180.0) % 360.0) - 180.0
    side = "right" if a > 0 else "left"
    m = abs(a)
    if m <= 22.5:
        horiz = "the front"
    elif m <= 67.5:
        horiz = f"the front-{side}"
    elif m <= 112.5:
        horiz = f"the {side} side"
    elif m <= 157.5:
        horiz = f"behind on the {side} (rim light)"
    else:
        horiz = "directly behind (backlight)"
    if el > 60:
        return "directly above"
    if el > 25:
        return f"{horiz}, from above"
    if el < -25:
        return f"{horiz}, from below"
    return horiz


def kelvin_words(k):
    if k < 2300:
        return f"deep orange {int(k)}K firelight-warm"
    if k < 3500:
        return f"warm tungsten {int(k)}K"
    if k < 4500:
        return f"warm white {int(k)}K"
    if k < 6000:
        return f"neutral daylight {int(k)}K"
    if k < 7500:
        return f"cool {int(k)}K"
    return f"cold blue {int(k)}K"


def intensity_words(ev):
    if ev <= -2.5:
        return "faint"
    if ev <= -1:
        return "soft"
    if ev < 0.75:
        return "strong"
    if ev < 2:
        return "bright"
    return "harsh, blinding"


def _color_words(light, kelvin):
    color = light.get("color") or {}
    if color.get("hex"):
        return f"colored ({color['hex']})"
    return kelvin_words(kelvin)


def _motion_words(light):
    motion = light.get("motion") or {}
    kind = motion.get("type")
    if kind == "flicker":
        return "flickering irregularly like firelight"
    if kind == "strobe":
        return f"strobing on and off about {motion.get('rate', 2.0):g} times per second"
    if kind == "pulse":
        return "slowly pulsing"
    return ""


def describe_light(light, frame_count, fps):
    keys = filled_keys(light, frame_count, fps)
    first, last = keys[0], keys[-1]
    softness = float(light.get("softness", 0.3))
    quality = "hard-edged" if softness < 0.3 else ("soft, diffused" if softness > 0.6 else "")
    head = " ".join(w for w in (intensity_words(first["intensity_ev"]), quality,
                                 _color_words(light, first["kelvin"]),
                                 TYPE_WORDS.get(light.get("type"), "light")) if w)
    text = f"A {head} from {direction_words(first['azimuth'], first['elevation'])}"

    moves = []
    for k0, k1 in zip(keys, keys[1:]):
        t0, t1 = k0["_frame"] / fps, k1["_frame"] / fps
        parts = []
        if abs(k0["azimuth"] - k1["azimuth"]) > 15 or abs(k0["elevation"] - k1["elevation"]) > 15:
            parts.append(f"sweeps from {direction_words(k0['azimuth'], k0['elevation'])} "
                         f"to {direction_words(k1['azimuth'], k1['elevation'])}")
        if k1["intensity_ev"] - k0["intensity_ev"] > 0.75:
            parts.append("brightens")
        elif k0["intensity_ev"] - k1["intensity_ev"] > 0.75:
            parts.append("dims")
        if abs(k1["kelvin"] - k0["kelvin"]) > 600 and not (light.get("color") or {}).get("hex"):
            parts.append(f"shifts from {kelvin_words(k0['kelvin'])} to {kelvin_words(k1['kelvin'])}")
        if parts:
            moves.append(f"from {t0:.1f}s to {t1:.1f}s it " + " and ".join(parts))
    if moves:
        text += "; " + "; ".join(moves)
        text += ", passing across the subject" if any("front" in m for m in moves) else ""
    motion = _motion_words(light)
    if motion:
        text += f", {motion}"
    if light.get("prompt"):
        text += f" ({light['prompt']})"
    return text + "."


def removed_original_words(rig, frame_count, fps):
    """Detected lights the user deleted or switched off: the light the clip had from there goes away."""
    from .rig import filled_keys
    gone = list(rig.get("removed_originals") or [])
    gone += [l.get("detected") or l for l in rig.get("lights", []) if l.get("original") and not l.get("enabled", True)]
    out = []
    for snap in gone:
        k = filled_keys(snap, frame_count, fps)[0]
        from .pro import angle_spec
        out.append(f"The scene's existing main light ({angle_spec(k['azimuth'], k['elevation'])}) is switched off; "
                   "the surfaces it lit fall into shadow.")
    return " ".join(out)


def stage_lights(rig, frame_count, fps, include_composite=True):
    """Lights the prompt describes: enabled ones, except detected lights left exactly as detected (the clip
    already has them). Yields (light, is_original)."""
    from .rig import original_changed
    for light in rig["lights"]:
        if not light.get("enabled", True):
            continue
        if light.get("mode") == "composite" and not include_composite:
            continue
        if light.get("original") and not original_changed(light, frame_count, fps):
            continue
        yield light, bool(light.get("original"))


def as_change(text):
    """'A soft ... light from ...' -> 'The scene's existing light changes: a soft ... light from ...'"""
    return "The scene's existing light changes: " + (text[:1].lower() + text[1:] if text else text)


def ambient_words(rig):
    """The rest of the scene's light, only when the user set it (default: the original's own light stays)."""
    amb = rig.get("ambient")
    if not amb:
        return ""
    ev = float(amb.get("intensity_ev", 0.0))
    level = ("much darker than in the original" if ev <= -3 else "darker than in the original" if ev <= -1
             else "brighter than in the original" if ev >= 1 else "about as bright as in the original")
    return (f"Everywhere the lights described here do not reach, the scene's own light is {level}, with a "
            f"{kelvin_words(float(amb.get('kelvin', 6500)))} tint.")


def rig_to_prompt(rig, frame_count, fps, ref_tag="<Video 1>", preamble=True, include_composite=False, extra=""):
    lines = []
    if preamble:
        lines.append(PREAMBLE.format(ref=ref_tag))
    for light, original in stage_lights(rig, frame_count, fps, include_composite):
        text = describe_light(light, frame_count, fps)
        lines.append(as_change(text) if original else text)
    lines.append(removed_original_words(rig, frame_count, fps))
    amb = ambient_words(rig)
    if amb:
        lines.append(amb)
    look = rig.get("look") or {}
    looks = [LOOKS.get(look.get("preset", ""), look.get("preset", "")),
             FILMS.get(look.get("film", ""), look.get("film", "")),
             GRADES.get(look.get("grade", ""), look.get("grade", ""))]
    looks = [w for w in looks if w]
    if looks:
        lines.append(", ".join(looks).capitalize() + ".")
    for text in (rig.get("user_prompt", ""), extra):
        if text and text.strip():
            lines.append(text.strip())
    return " ".join(l for l in lines if l)
