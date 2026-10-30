"""The clip's own lights read by the vision LM (Qwen3-VL), for the light stage's starting point.

The geometry fit (detect.detect_original_lights) assumes one light, one material and a close-up: brightness that
changes with surface direction is read as the light's direction. In a wide shot with dark clothes on pale walls,
glowing green-screen windows and two kinds of light it gave "from below" for a bridge lit by its windows and its
ceiling tubes (2026-10-29). The LM sees what lights the scene (windows, lamps, a green screen's spill, studio
lights) and names up to three lights in a fixed form; the geometry only refines the main light's angle where the
two agree.
"""
import json
import math
import re

from .rig import load_rig

# The scene goes to the lighting director, so it has to say where the openings are: as free sentences it came back as "a
# large green screen with the ship's wheel behind it" for a row of windows, with no window named (2026-10-30). The fields
# that asked for Chinese are gone: nothing shows them any more, and they pulled the scene into Chinese too.
READ_SYSTEM = """You are a gaffer studying frames from one shot. Work out which lights light this scene as it was filmed, and how the space is laid out.
Look at where highlights and shadows fall on people and objects, which sides are bright, the shadow edges, reflections, light sources visible in the frame or just outside it (windows, lamps, screens, signs, the sky, a green screen glowing behind the subject, studio lights reflected in glass), and the colour of the light (not the colour of the paint or clothes).

Reply with JSON only, in English, no other text:
{"lights": [{"role": "key|fill|back|top|practical",
             "horizontal": "front|front-left|frame-left|behind-left|behind|behind-right|frame-right|front-right",
             "vertical": "overhead|high|level|low",
             "colour": "warm|neutral|cool|green|blue|red|magenta|other",
             "hardness": "hard|soft",
             "strength": 0.0-1.0,
             "source": "a few words: what this light is (e.g. window daylight, ceiling tubes, green screen spill, studio lamp, street lamp)"}],
 "scene": {"camera": "where the camera is and what it looks toward",
           "far side": "what is on the far side of the scene, behind the people",
           "openings": "every window, doorway, hatch or skylight: how many, where each is in the frame, and what is seen through it",
           "surfaces": "the surfaces that could catch new light (floor, ceiling, walls, tables, consoles): where each is in the frame and which way it faces (toward the camera, up, down, toward the far side)",
           "materials": "the things with a particular surface or material that would take new light in their own way, where each is in the frame and what it is made of",
           "people": "each person: where in the frame, and which way they face (toward the camera, away from it, side-on)",
           "faces": "none|small|large: the biggest face the camera sees (small = under a fifth of the frame height)"}}
Rules: 1 to 3 lights, the strongest first (strength 1.0); a coloured or bright background is not a light unless it visibly lights the subject (its edges or front); left and right as seen in the frame; "behind" means behind the subject, facing the camera; the camera is "front". Do not list a light you cannot see evidence for. A framed area filled with flat bright green or blue is a window or doorway with a screen outside it: list it under openings as that window or doorway, never as a screen in the room."""

_H = {"front": 0, "front-left": -45, "frame-left": -90, "behind-left": -135, "behind": 180, "behind-right": 135,
      "frame-right": 90, "front-right": 45}
_V = {"overhead": 75, "high": 40, "level": 10, "low": -30}
_KELVIN = {"warm": 3200, "neutral": 5600, "cool": 7500}
_HEX = {"green": "#5cff7a", "blue": "#5a8cff", "red": "#ff4a3a", "magenta": "#ff4ad8"}


def _ask(clip, frames, seed=7, max_length=1400):   # the scene description needs room
    vision = "<|vision_start|><|image_pad|><|vision_end|>"
    text = (f"<|im_start|>system\n{READ_SYSTEM}<|im_end|>\n<|im_start|>user\n" + vision * frames.shape[0]
            + f"\nThese are {frames.shape[0]} frames of the same shot, in order.<|im_end|>\n<|im_start|>assistant\n")
    tokens = clip.tokenize(text, images=[frames[i:i + 1] for i in range(frames.shape[0])])
    ids = clip.generate(tokens, do_sample=False, max_length=max_length, temperature=1.0, top_k=50, top_p=1.0,
                        min_p=0.0, repetition_penalty=1.05, seed=seed)
    return clip.decode(ids)


def parse_reading(raw):
    """The LM's JSON (the first {...} block), or None."""
    raw = raw.split("</think>")[-1]
    a, b = raw.find("{"), raw.rfind("}")
    if a < 0 or b <= a:
        return None
    try:
        data = json.loads(raw[a:b + 1])
    except json.JSONDecodeError:
        return None
    return data if isinstance(data.get("lights"), list) and data["lights"] else None


def _angle(a, b):
    """Degrees between two (azimuth, elevation) directions."""
    def vec(az, el):
        az, el = math.radians(az), math.radians(el)
        return (math.sin(az) * math.cos(el), math.sin(el), math.cos(az) * math.cos(el))
    va, vb = vec(*a), vec(*b)
    return math.degrees(math.acos(max(-1.0, min(1.0, sum(x * y for x, y in zip(va, vb))))))


def _plain(value):
    """A field as words: the LM answers some fields as lists of {location, description}."""
    if isinstance(value, dict):
        return ", ".join(t for t in map(_plain, value.values()) if t)
    if isinstance(value, list):
        return "; ".join(t for t in map(_plain, value) if t)
    return str(value).strip()


def scene_text(scene):
    """The scene's fields as labelled lines for the director; a plain string is taken as it is."""
    if isinstance(scene, dict):
        return "\n".join(f"{str(k).capitalize()}: {_plain(v)}" for k, v in scene.items() if _plain(v))
    return str(scene or "").strip()


def reading_to_rig(data, geometry_guess=None, close_up=False):
    """The LM's lights as the stage's original lights. geometry_guess: (azimuth, elevation) of the geometry fit;
    on a close-up, when it agrees with the LM's main light within 45 degrees, its finer angle is used."""
    lights, lines = [], []
    items, seen = [], []
    for l in (x for x in data["lights"] if isinstance(x, dict)):
        # the same light twice (the screen and "the screen's reflected light" from the same side): keep the first
        d = (_H.get(str(l.get("horizontal", "front")).lower(), 0), _V.get(str(l.get("vertical", "level")).lower(), 10))
        c = str(l.get("colour", "neutral")).lower()
        if any(_angle(d, d0) < 25 and c == c0 for d0, c0 in seen):
            continue
        seen.append((d, c))
        items.append(l)
    items = items[:3]
    top = max(float(l.get("strength", 1.0) or 0.0) for l in items) or 1.0
    for i, l in enumerate(items):
        az = _H.get(str(l.get("horizontal", "front")).lower(), 0)
        el = _V.get(str(l.get("vertical", "level")).lower(), 10)
        # a source it names as overhead is overhead, whatever height it put down ("ceiling tubes" came back "level")
        src = str(l.get("source", "")).lower()
        if any(w in src for w in ("ceiling", "overhead", "天花板", "頂燈", "吊燈", "skylight")):
            el = max(el, _V["overhead"])
        refined = ""
        if i == 0 and geometry_guess and close_up and _angle((az, el), geometry_guess) <= 45:
            az, el = geometry_guess
            refined = " (angle refined from the geometry)"
        colour = str(l.get("colour", "neutral")).lower()
        color = {"hex": _HEX[colour]} if colour in _HEX else {"kelvin": _KELVIN.get(colour, 5600)}
        kelvin = color.get("kelvin", 5600)
        strength = max(float(l.get("strength", 1.0) or 0.05), 0.05)
        role = str(l.get("role", "key")).lower()
        light = {"id": f"original{i}", "name": "Original light" if i == 0 else f"Original light {i + 1}", "type": "area", "mode": "generate",
                 "enabled": True, "original": True, "color": color,
                 "softness": 0.2 if str(l.get("hardness", "soft")).lower() == "hard" else 0.7,
                 "cone": {"angle": 40.0, "feather": 0.4}, "shadow": True, "motion": None, "prompt": "",
                 "source": str(l.get("source", "")),
                 "keys": [{"sec": 0, "azimuth": round(az), "elevation": round(el), "distance": 1.5,
                           "intensity_ev": round(math.log2(strength / top), 1), "kelvin": kelvin, "ease": "smooth"}]}
        lights.append(light)
        lines.append(f"{light['name']} ({role}): {l.get('source', '')}, azimuth {round(az)}°, elevation "
                     f"{round(el)}°, {'hard' if light['softness'] < 0.5 else 'soft'} light{refined}.")
    for light in lights:
        light["detected"] = {k: v for k, v in light.items() if k != "detected"}
    rig = load_rig({"lights": lights})
    # the space as the LM saw it, for the lighting director: what a light "behind the people" really is in this scene
    rig["scene"] = scene_text(data.get("scene"))
    report = "Original lights read by the LM:\n" + "\n".join(lines) + "\nThis is a starting point: check it on the light stage."
    return rig, report


def read_original_lights(clip, frames, geometry_guess=None, close_up=False, seed=7):
    """(rig, report, raw) or (None, None, raw) when the LM gave nothing usable. frames: (K, H, W, 3), small."""
    raw = _ask(clip, frames, seed)
    data = parse_reading(raw)
    if data is None:
        return None, None, raw
    rig, report = reading_to_rig(data, geometry_guess, close_up)
    return rig, report, raw
