"""LightRig: the single source of truth for every light in a shot.

A rig is plain JSON (see examples/ and DEFAULT_RIG). Light positions are spherical
coordinates around the subject, in degrees:

    azimuth   0 = front (camera side), +90 = screen right, -90 = screen left, 180 = behind
    elevation 0 = eye level, +90 = straight above, -90 = straight below
    distance  multiple of the subject's distance from the camera (1.0 = as far as the camera)

Keyframes are placed on the clip's own frames. A key may give its time as
"frame", "sec" or "t" (0..1 of the clip), so presets work on clips of any length.
"""

import copy
import json
import math

import torch

DEFAULT_LIGHT = {
    "id": "light",
    "name": "Light",
    "type": "spot",            # spot | point | area | directional
    "mode": "generate",        # generate | composite | both
    "enabled": True,
    "color": {"kelvin": 5600},
    "softness": 0.3,           # 0 = hard source, 1 = very soft (wrap + soft shadow)
    "falloff": {"decay": 2.0},
    "cone": {"angle": 40.0, "feather": 0.4},
    "shadow": True,
    "specular": 1.0,
    "motion": None,            # {"type": "flicker" | "strobe" | "pulse", ...}
    "prompt": "",
    "keys": [{"frame": 0, "azimuth": 45.0, "elevation": 20.0, "distance": 1.5, "intensity_ev": 0.0}],
}

DEFAULT_RIG = {
    "version": 1,
    # None = the original's own light stays as it is; set only when the user changes it on the light stage
    "ambient": None,
    "lights": [],
    "look": {"preset": "", "film": "", "grade": ""},
    "user_prompt": "",
}

ANIMATED = ("azimuth", "elevation", "distance", "intensity_ev", "kelvin")

GELS = {
    "cto_full": (1.0, 0.72, 0.42),
    "cto_half": (1.0, 0.84, 0.66),
    "ctb_full": (0.62, 0.78, 1.0),
    "ctb_half": (0.80, 0.89, 1.0),
    "steel_blue": (0.70, 0.85, 1.0),
    "magenta": (1.0, 0.55, 0.95),
    "green": (0.70, 1.0, 0.65),
    "red": (1.0, 0.25, 0.2),
}


def _light(id, name, type, mode, kelvin, keys, **extra):
    light = copy.deepcopy(DEFAULT_LIGHT)
    light.update({"id": id, "name": name, "type": type, "mode": mode, "color": {"kelvin": kelvin}, "keys": keys})
    light.update(extra)
    return light


def _rig(lights, ambient_kelvin=6500, ambient_ev=-3.0, **extra):
    rig = copy.deepcopy(DEFAULT_RIG)
    rig["lights"] = lights
    rig["ambient"] = {"kelvin": ambient_kelvin, "intensity_ev": ambient_ev}
    rig.update(extra)
    return rig


def _key(t, az, el, dist=1.5, ev=0.0, **extra):
    key = {"t": t, "azimuth": az, "elevation": el, "distance": dist, "intensity_ev": ev}
    key.update(extra)
    return key


PRESETS = {
    "sweep_right_to_front": _rig([
        _light("key", "Key", "area", "generate", 4300, [_key(0, -40, 20, 1.6, -0.5)], softness=0.6, shadow=False),
        _light("beam", "Searchlight", "spot", "both", 7000,
               [_key(0.0, 95, 12, 2.0, 1.0, ease="easeInOut"), _key(0.6, 0, 12, 2.0, 1.0)],
               softness=0.1, cone={"angle": 26.0, "feather": 0.35},
               prompt="harsh cold searchlight beam with faint visible haze"),
    ], ambient_kelvin=7500, ambient_ev=-3.5, look={"preset": "neo_noir", "film": "", "grade": ""}),
    "rembrandt": _rig([
        _light("key", "Key", "area", "generate", 3400, [_key(0, -45, 35, 1.5, 0.0)], softness=0.5),
        _light("fill", "Fill", "area", "generate", 5000, [_key(0, 50, 5, 2.0, -2.5)], softness=0.9, shadow=False),
    ], ambient_ev=-4.0),
    "silhouette": _rig([
        _light("back", "Backlight", "area", "generate", 3200, [_key(0, 180, 15, 1.5, 1.5)], softness=0.4),
    ], ambient_ev=-5.0),
    "contre_jour": _rig([
        _light("sun", "Sun", "directional", "generate", 4000, [_key(0, 160, 25, 1.0, 1.2)], softness=0.2),
        _light("bounce", "Bounce", "area", "generate", 5000, [_key(0, 0, -10, 2.0, -2.0)], softness=1.0, shadow=False),
    ], ambient_ev=-3.0),
    "window": _rig([
        _light("window", "Window", "area", "generate", 6500, [_key(0, -75, 15, 1.4, 0.3)], softness=0.8),
    ], ambient_kelvin=7000, ambient_ev=-3.0),
    "firelight": _rig([
        _light("fire", "Fire", "point", "both", 1900, [_key(0, 60, -20, 1.0, 0.8)], softness=0.6,
               motion={"type": "flicker", "amount": 0.45, "speed": 7.0},
               prompt="warm flickering firelight from below"),
    ], ambient_kelvin=9000, ambient_ev=-4.5),
    "police": _rig([
        _light("red", "Red", "point", "composite", 3000, [_key(0, -70, 10, 1.5, 1.0)], color={"hex": "#ff2a1a"},
               motion={"type": "strobe", "rate": 2.0, "phase": 0.0, "duty": 0.5}),
        _light("blue", "Blue", "point", "composite", 3000, [_key(0, 70, 10, 1.5, 1.0)], color={"hex": "#1a4dff"},
               motion={"type": "strobe", "rate": 2.0, "phase": 0.5, "duty": 0.5}),
    ], ambient_kelvin=8000, ambient_ev=-3.5),
    "sunset_fade": _rig([
        _light("sun", "Sun", "directional", "generate", 5200,
               [_key(0, 70, 30, 1.0, 0.5, kelvin=5200), _key(1, 85, 3, 1.0, -0.5, kelvin=2400)], softness=0.2),
    ], ambient_kelvin=8000, ambient_ev=-3.0),
}


# Professional lighting setups (2026-10-29): the editor's "+ Add preset" puts a whole setup on the sphere, the
# user then only moves / dims / recolours. Applying one switches the detected original light off (it can be
# switched back on), since the setup is meant to replace the clip's light, and leaves the ambient as it was.
# Levels are stops relative to the original main light; ratios follow common practice (key:fill 2-4:1).
def _setup(label, lights):
    rig = _rig(lights)
    rig["ambient"] = None
    rig["setup"] = label
    return rig


def _gel(id, name, hex_color, keys, **extra):
    light = _light(id, name, "area", "generate", 5600, keys, **extra)
    light["color"] = {"hex": hex_color}
    return light


SETUPS = {
    "rembrandt": _setup("Rembrandt (high 45° key, weak fill, 4:1)", [
        _light("key", "Key", "area", "generate", 4300, [_key(0, -45, 40, 1.5, 0.5)], softness=0.45),
        _light("fill", "Fill", "area", "generate", 5000, [_key(0, 35, 0, 2.0, -1.5)], softness=0.9, shadow=False),
    ]),
    "loop": _setup("Loop (30° key, fill, 3:1)", [
        _light("key", "Key", "area", "generate", 5000, [_key(0, -30, 30, 1.5, 0.5)], softness=0.5),
        _light("fill", "Fill", "area", "generate", 5000, [_key(0, 25, 0, 2.0, -1.1)], softness=0.9, shadow=False),
    ]),
    "butterfly": _setup("Butterfly (high front key, reflector below)", [
        _light("key", "Key", "area", "generate", 5000, [_key(0, 0, 45, 1.5, 0.5)], softness=0.4),
        _light("fill", "Reflector", "area", "generate", 5000, [_key(0, 0, -30, 1.5, -1.5)], softness=1.0, shadow=False),
    ]),
    "split": _setup("Split (90° side light)", [
        _light("key", "Key", "area", "generate", 4500, [_key(0, -90, 10, 1.5, 0.5)], softness=0.4),
    ]),
    "three_point": _setup("Three-point (key, fill, back)", [
        _light("key", "Key", "area", "generate", 4800, [_key(0, -40, 30, 1.5, 0.5)], softness=0.55),
        _light("fill", "Fill", "area", "generate", 5000, [_key(0, 40, 10, 2.0, -1.5)], softness=0.9, shadow=False),
        _light("back", "Back", "area", "generate", 5600, [_key(0, 150, 45, 1.5, 0.3)], softness=0.3),
    ]),
    "key_rim": _setup("Key and cool rim", [
        _light("key", "Key", "area", "generate", 3800, [_key(0, -40, 25, 1.5, 0.5)], softness=0.6),
        _light("rim", "Rim", "area", "generate", 7000, [_key(0, 140, 30, 1.5, 0.5)], softness=0.25),
    ]),
    "neon": _setup("Two-colour neon (magenta left, cyan right)", [
        _gel("magenta", "Magenta", "#ff2bd6", [_key(0, -80, 10, 1.3, 0.5)], softness=0.5),
        _gel("cyan", "Cyan", "#19d3ff", [_key(0, 80, 10, 1.3, 0.5)], softness=0.5),
    ]),
    "window": _setup("Window (large soft side light)", [
        _light("window", "Window", "area", "generate", 6000, [_key(0, -70, 20, 1.4, 0.5)], softness=0.9),
    ]),
    "top": _setup("Top light", [
        _light("top", "Top", "area", "generate", 5000, [_key(0, 0, 85, 1.5, 0.5)], softness=0.5),
    ]),
    "under": _setup("Under light", [
        _light("under", "Under", "area", "generate", 3200, [_key(0, 0, -50, 1.3, 0.3)], softness=0.3),
    ]),
    "silhouette": _setup("Silhouette (light from behind only)", [
        _light("back", "Back", "area", "generate", 3200, [_key(0, 180, 15, 1.5, 1.5)], softness=0.4),
    ]),
    "golden_hour": _setup("Golden hour (low warm sun, cool sky fill)", [
        _light("sun", "Low sun", "directional", "generate", 2800, [_key(0, -70, 8, 1.0, 0.8)], softness=0.2),
        _light("sky", "Sky fill", "area", "generate", 8000, [_key(0, 40, 50, 2.0, -2.0)], softness=1.0, shadow=False),
    ]),
}
PRESETS.update(SETUPS)
PRESET_LABELS = {name: rig["setup"] for name, rig in SETUPS.items()}
PRESET_LABELS.update({"sweep_right_to_front": "Searchlight sweep", "contre_jour": "Backlit sun with ground bounce",
                      "firelight": "Firelight", "police": "Red and blue warning lights", "sunset_fade": "Sunset fade"})


def reference_signature(lights):
    """Which reference the stage's reference lights came from; the same string as the editor's (lightstage.js
    applyReference: JSON.stringify of colour, azimuth, elevation, level of each light)."""
    rows = [[l.get("color"), l["keys"][0].get("azimuth"), l["keys"][0].get("elevation"), l["keys"][0].get("intensity_ev")]
            for l in lights]
    return json.dumps(rows, separators=(",", ":"), ensure_ascii=False)


def preset_names():
    return list(PRESETS.keys())


def load_rig(text_or_dict, preset=None):
    """Parse a rig and fill every missing field with defaults."""
    if preset and preset in PRESETS:
        raw = copy.deepcopy(PRESETS[preset])
    elif isinstance(text_or_dict, dict):
        raw = copy.deepcopy(text_or_dict)
    else:
        text = (text_or_dict or "").strip()
        raw = json.loads(text) if text else copy.deepcopy(DEFAULT_RIG)

    rig = copy.deepcopy(DEFAULT_RIG)
    rig.update({k: v for k, v in raw.items() if k != "lights"})
    lights = []
    for i, src in enumerate(raw.get("lights", [])):
        light = copy.deepcopy(DEFAULT_LIGHT)
        light.update(src)
        light.setdefault("id", f"light{i}")
        for sub in ("cone", "falloff"):
            merged = copy.deepcopy(DEFAULT_LIGHT[sub])
            merged.update(src.get(sub) or {})
            light[sub] = merged
        if not light["keys"]:
            light["keys"] = copy.deepcopy(DEFAULT_LIGHT["keys"])
        lights.append(light)
    rig["lights"] = lights
    return rig


# ---------------------------------------------------------------- colour

def kelvin_to_rgb(kelvin):
    """Blackbody approximation (Tanner Helland), normalised to unit luminance."""
    t = max(1000.0, min(40000.0, float(kelvin))) / 100.0
    if t <= 66:
        r = 255.0
        g = 99.4708025861 * math.log(t) - 161.1195681661
        b = 0.0 if t <= 19 else 138.5177312231 * math.log(t - 10) - 305.0447927307
    else:
        r = 329.698727446 * ((t - 60) ** -0.1332047592)
        g = 288.1221695283 * ((t - 60) ** -0.0755148492)
        b = 255.0
    rgb = [max(0.0, min(255.0, c)) / 255.0 for c in (r, g, b)]
    rgb = [c ** 2.2 for c in rgb]  # to linear
    lum = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
    return [c / max(lum, 1e-6) for c in rgb]


def _hex_to_rgb(hexstr):
    h = hexstr.lstrip("#")
    rgb = [int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)]
    rgb = [c ** 2.2 for c in rgb]
    lum = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
    return [c / max(lum, 1e-6) for c in rgb]


def light_color(color, kelvin_override=None):
    color = color or {}
    if color.get("hex"):
        rgb = _hex_to_rgb(color["hex"])
    else:
        rgb = kelvin_to_rgb(kelvin_override if kelvin_override is not None else color.get("kelvin", 5600))
    gel = GELS.get(color.get("gel") or "")
    if gel:
        rgb = [c * g for c, g in zip(rgb, gel)]
    return rgb


# ---------------------------------------------------------------- time

EASES = {
    "linear": lambda u: u,
    "easeIn": lambda u: u * u,
    "easeOut": lambda u: 1 - (1 - u) * (1 - u),
    "easeInOut": lambda u: 2 * u * u if u < 0.5 else 1 - (-2 * u + 2) ** 2 / 2,
    "smooth": lambda u: u * u * (3 - 2 * u),
    "hold": lambda u: 0.0,
}


def key_frame(key, frame_count, fps):
    if "frame" in key:
        return float(key["frame"])
    if "sec" in key:
        return float(key["sec"]) * fps
    if "t" in key:
        return float(key["t"]) * max(frame_count - 1, 1)
    return 0.0


def _dir_from_angles(az, el):
    """Unit vector (subject -> light) in MoGe/OpenCV camera space: X right, Y down, Z into the scene."""
    a, e = math.radians(az), math.radians(el)
    return [math.sin(a) * math.cos(e), -math.sin(e), -math.cos(a) * math.cos(e)]


def _slerp(v0, v1, u):
    dot = max(-1.0, min(1.0, sum(a * b for a, b in zip(v0, v1))))
    if dot > 0.9995:
        out = [a + (b - a) * u for a, b in zip(v0, v1)]
    else:
        theta = math.acos(dot) * u
        rel = [b - a * dot for a, b in zip(v0, v1)]
        n = math.sqrt(sum(c * c for c in rel)) or 1.0
        rel = [c / n for c in rel]
        out = [a * math.cos(theta) + r * math.sin(theta) for a, r in zip(v0, rel)]
    n = math.sqrt(sum(c * c for c in out)) or 1.0
    return [c / n for c in out]


def _noise(seed, x):
    """Smooth 1D value noise in [-1, 1], deterministic per seed."""
    def h(i):
        v = math.sin(i * 127.1 + seed * 311.7) * 43758.5453
        return (v - math.floor(v)) * 2 - 1
    i = math.floor(x)
    f = x - i
    u = f * f * (3 - 2 * f)
    return h(i) * (1 - u) + h(i + 1) * u


def _motion_gain(motion, frame, fps, seed):
    if not motion:
        return 1.0
    kind = motion.get("type")
    sec = frame / max(fps, 1e-6)
    if kind == "flicker":
        amount = float(motion.get("amount", 0.4))
        speed = float(motion.get("speed", 6.0))
        n = 0.65 * _noise(seed, sec * speed) + 0.35 * _noise(seed + 17, sec * speed * 2.7)
        return max(0.0, 1.0 + amount * n)
    if kind == "strobe":
        rate = float(motion.get("rate", 2.0))
        phase = float(motion.get("phase", 0.0))
        duty = float(motion.get("duty", 0.5))
        return 1.0 if ((sec * rate + phase) % 1.0) < duty else 0.0
    if kind == "pulse":
        rate = float(motion.get("rate", 1.0))
        amount = float(motion.get("amount", 0.5))
        return max(0.0, 1.0 + amount * math.sin(2 * math.pi * rate * sec))
    return 1.0


def filled_keys(light, frame_count, fps):
    """Keys sorted in time, each carrying every animated field (missing ones inherit the previous key)."""
    current = {
        "azimuth": 45.0, "elevation": 20.0, "distance": 1.5, "intensity_ev": 0.0,
        "kelvin": float((light.get("color") or {}).get("kelvin", 5600)),
    }
    out = []
    for key in sorted(light["keys"], key=lambda k: key_frame(k, frame_count, fps)):
        current = {name: float(key.get(name, current[name])) for name in ANIMATED}
        filled = dict(current)
        filled["_frame"] = key_frame(key, frame_count, fps)
        filled["ease"] = key.get("ease", "smooth")
        out.append(filled)
    return out


def light_on(light, sec):
    """Whether the light is on at this time: between its start_sec and end_sec (either empty = the clip's edge)."""
    start, end = light.get("start_sec"), light.get("end_sec")
    return (start is None or sec >= float(start) - 1e-6) and (end is None or sec <= float(end) + 1e-6)


def evaluate_light(light, frame, frame_count, fps):
    """Interpolate one light at a (fractional) frame. Returns plain floats/lists."""
    keys = filled_keys(light, frame_count, fps)
    times = [k["_frame"] for k in keys]

    if frame <= times[0] or len(keys) == 1:
        vals = dict(keys[0])
        direction = _dir_from_angles(vals["azimuth"], vals["elevation"])
    elif frame >= times[-1]:
        vals = dict(keys[-1])
        direction = _dir_from_angles(vals["azimuth"], vals["elevation"])
    else:
        i = max(j for j in range(len(keys) - 1) if times[j] <= frame)
        k0, k1 = keys[i], keys[i + 1]
        span = max(times[i + 1] - times[i], 1e-6)
        u = EASES.get(k0.get("ease", "smooth"), EASES["smooth"])((frame - times[i]) / span)
        vals = {name: k0[name] + (k1[name] - k0[name]) * u for name in ANIMATED}
        direction = _slerp(_dir_from_angles(k0["azimuth"], k0["elevation"]),
                           _dir_from_angles(k1["azimuth"], k1["elevation"]), u)
        # the angles of the direction itself, as the stage draws it: the numbers interpolated straight went from 141 to
        # -112 degrees through 0, so a light crossing behind the people was told as passing in front (2026-10-30)
        n = math.sqrt(sum(c * c for c in direction)) or 1.0
        x, y, z = (c / n for c in direction)
        vals["azimuth"] = math.degrees(math.atan2(x, -z))
        vals["elevation"] = math.degrees(math.asin(max(-1.0, min(1.0, -y))))

    seed = sum(ord(c) for c in str(light.get("id", "")))
    gain = _motion_gain(light.get("motion"), frame, fps, seed) if light_on(light, frame / max(fps, 1e-6)) else 0.0
    has_kelvin_keys = any("kelvin" in k for k in light["keys"])
    return {
        "id": light.get("id"),
        "type": light.get("type", "spot"),
        "mode": light.get("mode", "generate"),
        "direction": direction,
        "azimuth": vals["azimuth"],
        "elevation": vals["elevation"],
        "distance": max(vals["distance"], 0.05),
        "intensity": (2.0 ** vals["intensity_ev"]) * gain,
        "color": light_color(light.get("color"), vals["kelvin"] if has_kelvin_keys else None),
        "softness": float(light.get("softness", 0.3)),
        "decay": float(light["falloff"].get("decay", 2.0)),
        "cone_angle": float(light["cone"].get("angle", 40.0)),
        "cone_feather": float(light["cone"].get("feather", 0.4)),
        "shadow": bool(light.get("shadow", True)),
        "specular": float(light.get("specular", 1.0)),
    }


def evaluate_rig(rig, frame, frame_count, fps):
    lights = [evaluate_light(l, frame, frame_count, fps) for l in rig["lights"] if l.get("enabled", True)]
    amb = rig.get("ambient") or {}
    ambient = [c * 2.0 ** float(amb.get("intensity_ev", -3.0)) for c in light_color({"kelvin": amb.get("kelvin", 6500)})]
    return lights, ambient


def plan_keyframes(rig, frame_count, fps, max_count=4, still=8.0, ev_deg=10.0, min_gap=3):
    """Keyframe frames for the rig: where its light changes, not spread evenly.

    The video model fades from one keyframe to the next, so a light that moves between two keyframes is not seen to
    travel: it dims in one place and comes up in the other (a red sweep between keyframes at 0 s and 2 s flooded the
    whole room red, 2026-10-30). Keyframes go first where a change starts and stops (a move's start and end, a
    light coming on or going out), so each shows at its time; then the slots left split the changes evenly (by
    degrees turned, ev_deg degrees per stop), so the steps the fades bridge are as small as they can be: two keyframes
    inside one move put it at thirds. Keyframes too close to each other (two keyframes 5 frames apart on either side
    of a sweep's middle made it jump, 2026-10-30) are not made. A rig that does not change gets the first and last.
    With no slot left for a move's middle, the keyframe where it meets a stretch with no change gives up its slot: the
    keyframe at the far end of that stretch shows the same light, and a 1.3 s sweep held to the end of a 2 s clip got
    keyframes at 0, 1.3 and 2 s, so the fade across the sweep flooded the windows red (2026-10-30). The clip's first and
    last frames always stay: past the last keyframe the video model has nothing to hold it to the new look."""
    last = max(int(frame_count) - 1, 0)
    if last < 1:
        return [0]
    lights = [dict(l, motion=None) for l in rig.get("lights", []) if l.get("enabled", True) and l.get("keys")]
    states = [[evaluate_light(l, f, frame_count, fps) for l in lights] for f in range(last + 1)]

    def angle(v, w):
        return math.degrees(math.acos(max(-1.0, min(1.0, sum(x * y for x, y in zip(v, w))))))

    def stops(s):   # a light that is off counts as 8 stops down
        return max(math.log2(max(s["intensity"], 1e-6)), -8.0)

    change = [0.0] + [sum(angle(b["direction"], a["direction"]) + ev_deg * abs(stops(b) - stops(a))
                          for a, b in zip(states[f - 1], states[f])) for f in range(1, last + 1)]
    if sum(change) <= still:
        return [0, last]
    total = [0.0]
    for c in change[1:]:
        total.append(total[-1] + c)
    moving = [c > 0.05 for c in change]
    # a change starts or stops between f-1 and f: f-1 is the last frame before it starts or of it
    marks = [0] + [f - 1 for f in range(1, last + 1) if moving[f] != moving[f - 1]] + [last]
    chosen = []
    for f in sorted(set(marks)):   # too close: keep the later one (the light after the change), never drop the ends
        if chosen and f - chosen[-1] < min_gap:
            if chosen[-1] == 0:
                continue
            chosen.pop()
        chosen.append(f)
    while len(chosen) > max_count:   # more turning points than keyframes: drop the one that changes least
        i = min(range(1, len(chosen) - 1), key=lambda i: total[chosen[i + 1]] - total[chosen[i - 1]])
        chosen.pop(i)

    def splits(a, b):   # one keyframe between a and b can cut their change in two (a light switching on cannot be cut)
        frames = range(a + min_gap, b - min_gap + 1)
        f = min(frames, key=lambda f: abs(total[f] - (total[a] + total[b]) / 2), default=None)
        return f is not None and min(total[f] - total[a], total[b] - total[f]) > still

    if len(chosen) == max_count:
        for i in range(1, len(chosen) - 1):
            if (min(total[chosen[i]] - total[chosen[i - 1]], total[chosen[i + 1]] - total[chosen[i]]) <= still
                    and splits(chosen[i - 1], chosen[i + 1])):
                chosen.pop(i)
                break
    # the slots left: to the gaps with the most change per step, each gap cut at even fractions of its change
    slots = {i: 0 for i in range(len(chosen) - 1)}
    gap = lambda i: total[chosen[i + 1]] - total[chosen[i]]   # noqa: E731
    for _ in range(max_count - len(chosen)):
        i = max(slots, key=lambda i: gap(i) / (slots[i] + 1))
        if gap(i) / (slots[i] + 1) <= still or chosen[i + 1] - chosen[i] < (slots[i] + 2) * min_gap:
            break
        slots[i] += 1
    out = []
    for i, n in slots.items():
        a, b = chosen[i], chosen[i + 1]
        out.append(a)
        for k in range(1, n + 1):
            want = total[a] + gap(i) * k / (n + 1)
            out.append(min(range(a + min_gap, b - min_gap + 1), key=lambda f: abs(total[f] - want)))
    return sorted(set(out + [last]))


def to_tensor(v, device):
    return torch.tensor(v, dtype=torch.float32, device=device)


# ---------------------------------------------------------------- original lights
# A light detected in the clip is a normal light on the stage, flagged "original" and carrying a snapshot of
# what was detected ("detected"). The user may keep it, move / recolour it, or delete it (its snapshot then
# goes to rig["removed_originals"]). The mock-up compares the scene lit by the snapshots (the light the clip
# already has) with the scene lit by the stage as it is now; only the difference changes the picture.
ORIGINAL_FIELDS = ("azimuth", "elevation", "distance", "intensity_ev", "kelvin")


def original_rig(rig):
    """The light the clip already has: snapshots of every detected light, kept, edited or removed."""
    snaps = [copy.deepcopy(l["detected"]) for l in rig.get("lights", []) if l.get("original") and l.get("detected")]
    snaps += [copy.deepcopy(s) for s in rig.get("removed_originals", []) or []]
    for s in snaps:
        s["enabled"] = True
        s.pop("detected", None)
    return load_rig({"lights": snaps})


def original_changed(light, frame_count=121, fps=24.0, tol=(8.0, 8.0, 0.2, 0.3, 300.0)):
    """Whether a detected light was edited on the stage (moved, dimmed, recoloured, retyped or animated)."""
    snap = light.get("detected")
    if not light.get("original") or not snap:
        return True
    if not light.get("enabled", True):
        return True
    keys = filled_keys(light, frame_count, fps)
    base = filled_keys(snap, frame_count, fps)[0]
    for k in keys:
        for f, t in zip(ORIGINAL_FIELDS, tol):
            d = abs(k[f] - base[f])
            if f == "azimuth":
                d = abs(((d + 180.0) % 360.0) - 180.0)
            if d > t:
                return True
    for f in ("type", "softness", "color", "motion"):
        if light.get(f) != snap.get(f):
            return True
    return False
