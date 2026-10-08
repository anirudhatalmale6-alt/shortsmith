"""Procedural cinematic plates.

The offline visual back-end.  No model, no network, no API key: given a prompt
it composes a 1080x1920 still out of an atmospheric sky, layered silhouettes,
a light source, haze, bokeh, grain and a vignette.

This exists because a generated-image provider is the one part of the pipeline
that is always someone else's uptime.  When the hosted generator is rate
limited or the GPU box is off, a scheduled Short still has to come out, and an
abstract plate under good narration and good captions is a publishable Short.
A broken image is not.

The composition is chosen from words in the prompt, so a horror beat gets a
black tree line under a cold moon and a sci-fi beat gets a planet over a
horizon.  The palette comes from mood words.  Everything is seeded from the
prompt text, so the same beat always renders the same plate.
"""

from __future__ import annotations

import colorsys
import hashlib
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from ..config import settings

W, H = settings.video_width, settings.video_height

# mood -> (sky hue range, accent hue offset, lightness, saturation)
PALETTES: dict[str, tuple[float, float, float, float]] = {
    "melancholy": (0.58, 0.04, 0.26, 0.30),   # cold blue, desaturated
    "warm":       (0.08, -0.02, 0.36, 0.34),  # muted amber, not orange
    "horror":     (0.66, 0.02, 0.14, 0.32),   # near-black blue
    "scifi":      (0.52, 0.10, 0.30, 0.62),   # cyan with amber accent
    "hopeful":    (0.10, 0.03, 0.52, 0.52),   # sunrise gold
    "clinical":   (0.55, 0.01, 0.46, 0.18),   # flat daylight
    "noir":       (0.60, 0.0, 0.20, 0.10),    # grey
}

MOOD_WORDS: dict[str, tuple[str, ...]] = {
    "horror": ("horror", "unsettling", "deep shadow", "torchlight", "grain", "dark", "ominous"),
    "scifi": ("sci-fi", "scifi", "volumetric", "cyan", "anamorphic", "station", "corridor",
              "panel", "visor", "orbital"),
    "hopeful": ("hopeful", "golden hour", "sunrise", "uplifting", "warm rim"),
    "warm": ("warm", "cosy", "tungsten", "intimate", "practical lighting", "golden"),
    "clinical": ("documentary", "clinical", "scientific", "instruments", "natural light"),
    "noir": ("noir", "archive", "corporate", "fluorescent", "file", "office"),
    "melancholy": ("melancholic", "melancholy", "muted", "desaturated", "rain", "grey",
                   "quiet", "emotional"),
}

SCENE_WORDS: dict[str, tuple[str, ...]] = {
    "forest": ("tree", "forest", "farmhouse", "woods", "camp", "valley"),
    "city": ("city", "street", "apartment", "office", "building", "skyline", "shop",
             "platform", "corridor", "hallway", "motel", "block"),
    "sea": ("sea", "ocean", "beach", "harbour", "lighthouse", "ferry", "reservoir",
            "coast", "water", "flooded", "dive"),
    "mountain": ("mountain", "glacier", "pass", "ridge", "cave", "dig site", "field",
                 "road", "town", "village"),
    "space": ("orbital", "station", "ship", "mars", "colony", "space", "sky", "planet"),
    "interior": ("kitchen", "table", "desk", "counter", "room", "chair", "window",
                 "door", "car", "bed", "lamp", "cup"),
}


def _seed(prompt: str, index: int) -> int:
    return int(hashlib.sha256(f"{prompt}|{index}".encode()).hexdigest()[:12], 16)


def _pick(prompt: str, mapping: dict[str, tuple[str, ...]], default: str = "") -> str:
    low = prompt.lower()
    best, hits = default, 0
    for key, words in mapping.items():
        found = sum(1 for word in words if word in low)
        if found > hits:
            best, hits = key, found
    return best


def _rgb(hue: float, lightness: float, saturation: float) -> tuple[int, int, int]:
    return tuple(int(round(c * 255)) for c in colorsys.hls_to_rgb(hue % 1.0, lightness, saturation))


def _vertical_gradient(top: tuple[int, int, int], bottom: tuple[int, int, int],
                       gamma: float = 1.0) -> Image.Image:
    ramp = np.linspace(0.0, 1.0, H, dtype=np.float32) ** gamma
    ramp = ramp[:, None, None]
    top_arr = np.array(top, dtype=np.float32)[None, None, :]
    bottom_arr = np.array(bottom, dtype=np.float32)[None, None, :]
    column = top_arr + (bottom_arr - top_arr) * ramp
    return Image.fromarray(np.repeat(column, W, axis=1).astype(np.uint8), "RGB")


def _ridge(rng: random.Random, base_y: int, amplitude: int, octaves: int = 4,
           points: int = 220) -> list[tuple[int, int]]:
    """A smooth fractal ridge line, for silhouettes.

    Summed cosine octaves rather than a random walk: a walk at this sample
    density produces a spiky comb that reads as a pine forest no matter what
    the prompt said, which is exactly the failure the first version had.
    """
    phases = [(rng.uniform(0.6, 1.6) * (2 ** octave), rng.uniform(0, math.tau),
               1.0 / (1.7 ** octave)) for octave in range(octaves)]
    raw = []
    for i in range(points):
        t = i / (points - 1)
        raw.append(sum(weight * math.cos(freq * math.tau * t + phase)
                       for freq, phase, weight in phases))
    peak = max(1e-6, max(abs(v) for v in raw))
    return [
        (int(i * W / (points - 1)), int(base_y + (v / peak) * amplitude))
        for i, v in enumerate(raw)
    ]


def _silhouette_layer(draw: ImageDraw.ImageDraw, rng: random.Random, kind: str,
                      base_y: int, amplitude: int, colour: tuple[int, int, int]) -> None:
    if kind == "city":
        x = 0
        points = [(0, H)]
        while x < W:
            width = rng.randint(50, 170)
            top = base_y + rng.randint(-amplitude, amplitude // 2)
            points += [(x, top), (x + width, top)]
            x += width + rng.randint(0, 22)
        points.append((W, H))
        draw.polygon(points, fill=colour)
    elif kind == "forest":
        draw.rectangle([0, base_y + amplitude // 2, W, H], fill=colour)
        x = -40
        while x < W + 40:
            height = rng.randint(amplitude, int(amplitude * 2.4))
            half = rng.randint(28, 62)
            tip = base_y + amplitude // 2 - height
            draw.polygon(
                [(x - half, base_y + amplitude // 2), (x, tip), (x + half, base_y + amplitude // 2)],
                fill=colour,
            )
            x += rng.randint(42, 96)
    elif kind == "sea":
        draw.rectangle([0, base_y, W, H], fill=colour)
    else:  # mountain / generic ridge
        ridge = _ridge(rng, base_y, amplitude)
        draw.polygon([(0, H)] + ridge + [(W, H)], fill=colour)


COMPOSITIONS = ("mountain", "city", "forest", "sea")


def render_plate(prompt: str, out_path: Path, index: int = 0) -> Path:
    rng = random.Random(_seed(prompt, index))
    mood = _pick(prompt, MOOD_WORDS, "melancholy")
    scene = _pick(prompt, SCENE_WORDS, "")
    if not scene:
        # Nothing in the prompt named a setting.  Rotating by beat index keeps a
        # six-beat video from being six near-identical frames, which is what the
        # first version did and it read as one long still.
        scene = COMPOSITIONS[index % len(COMPOSITIONS)]

    hue, accent_offset, lightness, saturation = PALETTES[mood]
    # Walk the hue and exposure a little per beat so consecutive scenes read as
    # a sequence rather than a repeat, while staying inside the mood.
    hue += rng.uniform(-0.02, 0.02) + 0.035 * math.sin(index * 1.1)
    lightness = min(0.72, lightness * (1.18 + 0.16 * math.sin(index * 0.7 + 1.0)))

    # --- sky ------------------------------------------------------------
    sky_top = _rgb(hue, max(0.07, lightness * 0.50), saturation * 0.9)
    sky_bottom = _rgb(hue + accent_offset, min(0.90, lightness * 1.85), saturation)
    img = _vertical_gradient(sky_top, sky_bottom, gamma=1.6)
    draw = ImageDraw.Draw(img, "RGBA")

    # Alternate a high and a low horizon so the framing changes between beats.
    horizon = int(H * (rng.uniform(0.44, 0.52) if index % 2 else rng.uniform(0.58, 0.70)))

    # --- light source ---------------------------------------------------
    light_x = int(W * rng.uniform(0.18, 0.82))
    light_y = int(horizon - H * rng.uniform(0.02, 0.22))
    glow_colour = _rgb(hue + accent_offset, min(0.92, lightness * 2.1), saturation * 0.85)

    glow = Image.new("L", (W, H), 0)
    gdraw = ImageDraw.Draw(glow)
    for step in range(10, 0, -1):
        radius = int(min(W, H) * 0.06 * step)
        gdraw.ellipse(
            [light_x - radius, light_y - radius, light_x + radius, light_y + radius],
            fill=int(190 / step),
        )
    glow = glow.filter(ImageFilter.GaussianBlur(W // 22))
    img = Image.composite(Image.new("RGB", (W, H), glow_colour), img, glow)
    draw = ImageDraw.Draw(img, "RGBA")

    if scene == "space":
        # Stars, then a planet limb; cheap, and it reads instantly as sci-fi.
        for _ in range(rng.randint(140, 260)):
            sx, sy = rng.randrange(W), rng.randrange(int(horizon))
            size = rng.choice([1, 1, 1, 2])
            brightness = rng.randint(120, 255)
            draw.ellipse([sx, sy, sx + size, sy + size],
                         fill=(brightness, brightness, brightness, 220))
        radius = int(W * rng.uniform(0.55, 0.95))
        cx = int(W * rng.uniform(0.1, 0.9))
        cy = horizon + radius - int(H * rng.uniform(0.02, 0.12))
        draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius],
                     fill=_rgb(hue + 0.02, lightness * 0.5, saturation * 0.8))

    # --- depth layers ---------------------------------------------------
    layers = 4
    for layer in range(layers):
        depth = layer / (layers - 1)
        base_y = int(horizon + (H - horizon) * (0.04 + 0.30 * depth))
        amplitude = int(H * (0.035 + 0.055 * (1 - depth)))
        # Far layers sit closer to the sky colour; near layers go to black.
        shade = _rgb(hue, max(0.045, lightness * (0.70 - 0.15 * layer)), saturation * (0.75 - 0.14 * layer))
        kind = scene if scene not in {"space", "interior"} else "mountain"
        _silhouette_layer(draw, rng, kind, base_y, amplitude, shade)

    if scene == "sea":
        # Specular path from the light to the viewer: one tapered polygon, then
        # blurred and composited, so there is no stair-stepping.
        path = Image.new("L", (W, H), 0)
        pdraw = ImageDraw.Draw(path)
        spread = int(W * 0.22)
        pdraw.polygon(
            [(light_x - 10, horizon), (light_x + 10, horizon),
             (light_x + spread, H), (light_x - spread, H)],
            fill=120,
        )
        path = path.filter(ImageFilter.GaussianBlur(W // 26))
        img = Image.composite(Image.new("RGB", (W, H), glow_colour), img, path)
        draw = ImageDraw.Draw(img, "RGBA")

    # --- haze -----------------------------------------------------------
    haze = _vertical_gradient((0, 0, 0), sky_bottom, gamma=0.5).filter(
        ImageFilter.GaussianBlur(W // 14)
    )
    img = Image.blend(img, haze, 0.18)

    # --- bokeh ----------------------------------------------------------
    bokeh = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    bdraw = ImageDraw.Draw(bokeh)
    for _ in range(rng.randint(5, 10)):
        radius = rng.randint(10, 38)
        bx = rng.randrange(-radius, W + radius)
        # Keep bokeh in the upper two thirds; blobs over the caption area read
        # as dirt on the lens once the text is burned in.
        by = rng.randrange(int(H * 0.04), int(H * 0.62))
        alpha = rng.randint(8, 22)
        bdraw.ellipse([bx - radius, by - radius, bx + radius, by + radius],
                      fill=(*glow_colour, alpha))
    img = Image.alpha_composite(img.convert("RGBA"),
                                bokeh.filter(ImageFilter.GaussianBlur(9))).convert("RGB")

    # --- grade, grain, vignette -----------------------------------------
    arr = np.asarray(img).astype(np.float32) / 255.0
    # Filmic S-curve: lift the blacks slightly, roll off the highlights.
    arr = np.clip(arr, 0, 1)
    arr = arr * arr * (3 - 2 * arr) * 0.72 + arr * 0.28
    arr = 0.045 + arr * 0.955   # lift the blacks so the lower third keeps detail
    arr[..., 2] = np.clip(arr[..., 2] * 1.03 + 0.012, 0, 1)   # cool shadows
    arr[..., 0] = np.clip(arr[..., 0] * 1.01, 0, 1)

    grain = np.random.default_rng(_seed(prompt, index)).normal(0, 0.016, arr.shape[:2])
    arr += grain[..., None]

    yy, xx = np.mgrid[0:H, 0:W]
    radial = np.sqrt(((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 1.55)) ** 2)
    arr *= np.clip(1.04 - 0.34 * radial ** 2.0, 0.45, 1.0)[..., None]

    img = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8), "RGB")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, "PNG")
    return out_path
