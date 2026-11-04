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


