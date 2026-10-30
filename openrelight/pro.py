"""Light stage -> the language of a lighting plot, for the image and video models.

The user sets lights with a few simple controls (direction on the sphere, brightness, colour, hardness).
The prompt states them the way a gaffer would write them down, in units, never in loose adjectives
("blinding" from a +3 EV slider made the model blow the whole frame out):
  - role in the setup (key, fill, rim / back, top, under), from direction and relative level;
  - fixture and source size: soft = large source (wide penumbra), hard = small source (crisp shadow edge);
  - angle off the camera axis in degrees, toward frame-left / frame-right, and height in degrees;
  - level in stops (and the lighting ratio between key and fill / scene);
  - colour as a correlated colour temperature in kelvin, or a named gel colour with its hue in degrees;
  - what it does to the subject: which side it models, where shadows fall, whether it rims the outline.
Generic on purpose: "subject" and "surfaces", no faces or cheeks, since a clip may show anything.
"""

import colorsys
import math

from .rig import evaluate_light, filled_keys

FRONT_LIMIT = 112.5     # beyond this off the camera axis a light no longer reaches the camera side


def _wrap(az):
    return ((az + 180.0) % 360.0) - 180.0


def _side(az):
    return "frame-right" if _wrap(az) > 0 else "frame-left"


def _other(az):
    return "frame-left" if _wrap(az) > 0 else "frame-right"


# ---------------------------------------------------------------- colour

_HUES = [(15, "red"), (40, "orange"), (52, "amber"), (68, "yellow"), (95, "yellow-green"), (150, "green"),
         (175, "teal"), (195, "cyan"), (220, "sky blue"), (250, "blue"), (275, "violet"), (300, "purple"),
         (330, "magenta"), (350, "pink"), (361, "red")]


def hex_color_words(hex_str):
    """'#9c66ff' -> 'saturated violet gel (#9c66ff, hue 262°)'."""
    h = hex_str.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
    hue, light, sat = colorsys.rgb_to_hls(r, g, b)
    deg = hue * 360.0
    if sat < 0.12:
        return f"neutral white light ({hex_str})"
    name = next(n for limit, n in _HUES if deg < limit)
    depth = "deep " if light < 0.3 else "pale " if light > 0.8 else ""
    strength = "saturated " if sat > 0.6 else "muted " if sat < 0.3 else ""
    return f"{strength}{depth}{name} gel ({hex_str}, hue {deg:.0f}°)"


def kelvin_spec(k):
    k = int(round(k / 50.0) * 50)
    if k < 2300:
        return f"{k}K (candle / firelight orange)"
    if k < 3500:
        return f"{k}K (tungsten, warm)"
    if k < 4500:
        return f"{k}K (warm white)"
    if k < 6000:
        return f"{k}K (daylight balanced, neutral)"
    if k < 7500:
        return f"{k}K (overcast daylight, slightly cool)"
    return f"{k}K (shade / blue hour, cool)"


def color_spec(light, kelvin):
    color = light.get("color") or {}
    return hex_color_words(color["hex"]) if color.get("hex") else kelvin_spec(kelvin)


# ---------------------------------------------------------------- source and level

def source_spec(kind, softness, cone_angle=40.0, cone_feather=0.4):
    """Fixture + apparent source size, the thing that sets how hard the shadow edges are."""
    size = 2.0 + 43.0 * max(0.0, min(1.0, softness))                 # apparent size seen from the subject
    edge = ("crisp, well-defined shadow edges (narrow penumbra)" if softness < 0.3
            else "wide, gradual shadow transitions (broad penumbra)" if softness > 0.6
            else "moderately soft shadow edges")
    if kind == "spot":
        rim = "hard-edged" if cone_feather < 0.3 else "soft-edged"
        fixture = f"a Fresnel spot with a {cone_angle:.0f}° field angle, giving a {rim} pool of light that falls off outside it"
    elif kind == "directional":
        fixture = "a distant source with parallel rays, like the sun"
    elif kind == "point":
        fixture = "a bare practical bulb"
    else:
        fixture = "a large diffused source (softbox)" if softness > 0.6 else "a panel source"
    return f"{fixture}; apparent source size about {size:.0f}°, so {edge}"


def stops(ev):
    """+1.6 -> '+1.6 stops (3.0x)'."""
    return f"{ev:+.1f} stops ({2.0 ** ev:.1f}x)"


def ratio(ev_hi, ev_lo):
    d = max(0.0, ev_hi - ev_lo)
    return f"{2.0 ** d:.0f}:1 ({d:.1f} stops)" if d >= 0.95 else f"{2.0 ** d:.1f}:1 ({d:.1f} stops)"


# ---------------------------------------------------------------- geometry

def angle_spec(az, el):
    """Position in degrees off the camera axis and above / below the subject."""
    a = _wrap(az)
    m = abs(a)
    if m <= 8:
        horiz = "on the camera axis (frontal)"
    elif m >= 172:
        horiz = "directly behind the subject, pointing toward the camera"
    elif m > 90:
        horiz = f"{m:.0f}° off the camera axis toward {_side(a)}, behind the subject"
    else:
        horiz = f"{m:.0f}° off the camera axis toward {_side(a)}"
    if el >= 75:
        return "directly overhead (90° high)"
    vert = (f"{el:.0f}° above the subject" if el > 5 else f"{-el:.0f}° below the subject" if el < -5
            else "level with the subject")
    return f"{horiz}, {vert}"


def effect_spec(az, el):
    """What the light does to the subject, in terms of any subject (a person, an object, a set)."""
    a = _wrap(az)
    m = abs(a)
    side, other = _side(a), _other(a)
    if m <= 22.5:
        text = ("frontal light: lights the camera-facing surfaces evenly with little modelling; shadows fall "
                "straight behind the subject")
    elif m <= 67.5:
        text = (f"three-quarter light: strongest on the {side}-facing surfaces, falling off gradually toward the "
                f"{other}-facing side; cast shadows fall toward {other}")
    elif m <= FRONT_LIMIT:
        text = (f"side light: strongest on the {side}-facing surfaces, wrapping around the form and fading gradually "
                f"toward {other}, with strong surface texture; cast shadows fall toward {other}")
    elif m <= 157.5:
        text = (f"rim / edge light: outlines the {side} edges of the subject and separates it from the background; "
                "it does not light the camera-facing surfaces")
    else:
        text = ("backlight: a thin rim of light around the whole outline; the camera-facing surfaces get none of "
                "this light")
    if el >= 60:
        text += "; from this height it lights the tops of surfaces and shadows fall straight down"
    elif el > 25:
        text += "; shadows also fall downward"
    elif el < -25:
        text += "; from below, so shadows fall upward (underlighting)"
    return text


def falloff_spec(kind, distance):
    if kind == "directional":
        return "no falloff across the scene"
    if distance < 1.0:
        return "very close, so it falls off quickly with distance (inverse square): what is nearer gets much more"
    if distance < 2.0:
        return "at a medium distance, falling off gradually behind the subject"
    return "far away, so it is nearly even across the scene"


# ---------------------------------------------------------------- roles

def roles(states):
    """states: list of evaluated lights (dicts with azimuth, elevation, intensity) -> list of role names."""
    evs = [math.log2(max(s["intensity"], 1e-4)) for s in states]
    front = [i for i, s in enumerate(states) if abs(_wrap(s["azimuth"])) <= FRONT_LIMIT and s["elevation"] < 75]
    key = max(front, key=lambda i: evs[i]) if front else None
    out = []
    for i, s in enumerate(states):
        m, el = abs(_wrap(s["azimuth"])), s["elevation"]
        if el >= 75:
            out.append("top light")
        elif el <= -40:
            out.append("underlight")
        elif m > FRONT_LIMIT:
            out.append("rim light" if m <= 157.5 else "backlight")
        elif i == key:
            out.append("key light")
        elif evs[i] >= evs[key] - 0.5:
            out.append("second key light")
        else:
            out.append("fill light")
    return out, evs, key


def light_spec(light, state, role, kelvin, now=True):
    """One light as a line of a lighting plot."""
    ev = math.log2(max(state["intensity"], 1e-4))
    parts = [f"{role[:1].upper() + role[1:]}: {source_spec(state['type'], state['softness'], state['cone_angle'], state['cone_feather'])}",
             f"placed {angle_spec(state['azimuth'], state['elevation'])}",
             f"{falloff_spec(state['type'], state['distance'])}",
             f"colour {color_spec(light, kelvin)}",
             f"level {stops(ev)} relative to the scene's original main light"]
    text = "; ".join(parts) + ". " + effect_spec(state["azimuth"], state["elevation"]).capitalize() + "."
    if light.get("prompt"):
        text += f" ({light['prompt']})"
    return text


def kelvin_at(light, frame, frame_count, fps):
    keys = filled_keys(light, frame_count, fps)
    k = keys[0]["kelvin"]
    for key in keys:
        if key["_frame"] <= frame:
            k = key["kelvin"]
    return k


def setup_lines(lights, frame, frame_count, fps, originals=()):
    """Lighting plot at one frame: one line per light, then the key / fill ratio.
    lights: list of rig lights; originals: ids of the scene's own (detected) lights, described as changes."""
    states = [evaluate_light(l, frame, frame_count, fps) for l in lights]
    peak = [max(k["intensity_ev"] for k in filled_keys(l, frame_count, fps)) for l in lights]
    on = [i for i, s in enumerate(states) if math.log2(max(s["intensity"], 1e-4)) >= peak[i] - 3]
    lines = []
    for i in range(len(lights)):
        if i not in on:
            lines.append(f"The light named '{lights[i].get('name') or lights[i].get('id')}' is off at this moment.")
    if not on:
        return lines
    names, evs, key = roles([states[i] for i in on])
    for j, i in enumerate(on):
        role = names[j] + (" (the scene's existing main light, changed)" if lights[i].get("id") in originals else "")
        lines.append(light_spec(lights[i], states[i], role, kelvin_at(lights[i], frame, frame_count, fps)))
    lines += blend_lines([states[i] for i in on], evs)
    fills = [j for j, n in enumerate(names) if n == "fill light"]
    if key is not None and fills:
        lines.append(f"Key-to-fill ratio {ratio(evs[key], max(evs[j] for j in fills))}.")
    return lines


def blend_lines(states, evs):
    """Lights from opposite sides of the camera axis: say how they meet. Described one by one ("lit on the left,
    shadow on the right" twice) the editor split the subject into two flat colour halves with a hard line."""
    left = [i for i, s in enumerate(states) if -FRONT_LIMIT <= _wrap(s["azimuth"]) < -15 and s["elevation"] < 75]
    right = [i for i, s in enumerate(states) if 15 < _wrap(s["azimuth"]) <= FRONT_LIMIT and s["elevation"] < 75]
    if not left or not right:
        return []
    l, r = max(left, key=lambda i: evs[i]), max(right, key=lambda i: evs[i])
    if abs(evs[l] - evs[r]) > 2.0:
        return []       # one clearly dominates: a key and its fill, already said by the ratio
    return ["The lights from frame-left and frame-right overlap across the middle of the subject: surfaces facing "
            "the camera receive both, so there the two lights mix into a blended tone and each fades gradually into "
            "the other. No sharp dividing line between them; the transition follows the curvature of the form, and "
            "each light keeps its own specular highlights on the surfaces facing it."]


def ambient_spec(rig):
    """The scene's own light outside the stage's lights, only when the user set it."""
    amb = rig.get("ambient")
    if not amb:
        return ""
    ev = float(amb.get("intensity_ev", 0.0))
    return (f"Base exposure of everything the lights above do not reach: {stops(ev)} relative to the original, "
            f"colour balance {kelvin_spec(float(amb.get('kelvin', 6500)))}.")


def move_spec(light, frame_count, fps):
    """Keyframed changes over the clip, with times: for the video model."""
    keys = filled_keys(light, frame_count, fps)
    out = []
    for k0, k1 in zip(keys, keys[1:]):
        t0, t1 = k0["_frame"] / fps, k1["_frame"] / fps
        parts = []
        if abs(k0["azimuth"] - k1["azimuth"]) > 10 or abs(k0["elevation"] - k1["elevation"]) > 10:
            parts.append(f"moves from {angle_spec(k0['azimuth'], k0['elevation'])} to "
                         f"{angle_spec(k1['azimuth'], k1['elevation'])}")
        d = k1["intensity_ev"] - k0["intensity_ev"]
        if abs(d) >= 0.5:
            parts.append(f"{'rises' if d > 0 else 'drops'} by {abs(d):.1f} stops")
        if abs(k1["kelvin"] - k0["kelvin"]) > 400 and not (light.get("color") or {}).get("hex"):
            parts.append(f"shifts from {kelvin_spec(k0['kelvin'])} to {kelvin_spec(k1['kelvin'])}")
        if parts:
            out.append(f"from {t0:.1f}s to {t1:.1f}s it " + " and ".join(parts))
    motion = light.get("motion") or {}
    kind = motion.get("type")
    if kind == "flicker":
        out.append("its level flickers irregularly like a flame")
    elif kind == "strobe":
        out.append(f"it strobes on and off at {motion.get('rate', 2.0):g} Hz")
    elif kind == "pulse":
        out.append("its level pulses slowly")
    return "; ".join(out)
