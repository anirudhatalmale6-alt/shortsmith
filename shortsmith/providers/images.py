"""Visuals.

Five interchangeable back-ends, picked with IMAGE_PROVIDER:

  hfspace      - a public Hugging Face Space (FLUX.1-schnell by default).
                 Free, no key, open weights, and the best quality available
                 without a GPU of your own.  This is the default.
  pollinations - another free hosted generator, no key.  A second opinion when
                 the Space's shared GPU quota is spent.
  sdwebui      - a local AUTOMATIC1111 / Forge instance.  Fully offline, your
                 own checkpoints and LoRAs, no rate limit.
  comfyui      - a local ComfyUI instance running a simple txt2img graph.
  pexels       - real stock footage stills, free API key, for topics where
                 photographs beat generated art.
  gradient     - procedural gradient + grain.  No network at all.  Always the
                 last resort so a render never dies on a provider outage.

Every provider returns a PNG at the project's video resolution.  If a provider
fails on a single scene the pipeline degrades to `gradient` for that scene only
rather than losing the whole video.
"""

from __future__ import annotations

import colorsys
import hashlib
import io
import logging
import math
import random
import shutil
import subprocess
import time
from pathlib import Path
from urllib.parse import quote

import httpx
from PIL import Image, ImageDraw, ImageFilter

from ..config import settings

log = logging.getLogger(__name__)

TARGET = (settings.video_width, settings.video_height)

NEGATIVE = (
    "text, words, letters, caption, subtitle, watermark, signature, logo, "
    "username, lowres, blurry, jpeg artifacts, extra fingers, deformed hands, "
    "disfigured, bad anatomy, cropped head, frame, border"
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _fit_vertical(img: Image.Image, trim_bottom: int = 0) -> Image.Image:
    """Cover-crop to the target aspect, then resize.

    `trim_bottom` removes a strip before cropping, which is how the hosted
    provider's corner watermark is taken off: cutting it beats blurring it,
    because a blurred rectangle is visible on a still frame.
    """
    img = img.convert("RGB")
    if trim_bottom and img.height > trim_bottom * 3:
        img = img.crop((0, 0, img.width, img.height - trim_bottom))

    tw, th = TARGET
    scale = max(tw / img.width, th / img.height)
    new = (max(tw, int(math.ceil(img.width * scale))), max(th, int(math.ceil(img.height * scale))))
    img = img.resize(new, Image.LANCZOS)
    left = (img.width - tw) // 2
    top = (img.height - th) // 2
    return img.crop((left, top, left + tw, top + th))


# One seed base per video, offset by beat index.  Without this every still is
# an independent roll of the dice and the six beats do not look like one film.
_seed_base = 0
_negative = NEGATIVE


def set_look(seed_base: int = 0, negative: str = "") -> None:
    """Called once per render to lock the seed family and the negative prompt."""
    global _seed_base, _negative
    _seed_base = int(seed_base)
    _negative = negative or NEGATIVE


def _seed_for(prompt: str, index: int) -> int:
    if _seed_base:
        # Deterministic, close together, and stable across re-renders of the
        # same video: the model lands in the same region of latent space.
        return (_seed_base + index * 101) % (2 ** 31)
    digest = hashlib.sha256(f"{prompt}|{index}".encode()).hexdigest()
    return int(digest[:8], 16)


# ---------------------------------------------------------------------------
# providers
# ---------------------------------------------------------------------------

# The free anonymous tier allows roughly one image a minute per IP and answers
# 402 (not 429) the moment you exceed it, so pace the calls rather than retrying
# into the wall.  Override with POLLINATIONS_GAP once a key removes the limit.
_POLLINATIONS_MIN_GAP = float(__import__("os").environ.get("POLLINATIONS_GAP", "32"))
_last_pollinations_call = 0.0


def _http_get_bytes(url: str, timeout: int = 240) -> tuple[int, bytes]:
    """GET through curl rather than httpx.

    Not a style preference.  image.pollinations.ai answers a Python HTTP client
    with an instant 402 and the same URL from curl with a 200, so it is keying
    on the TLS/HTTP2 client fingerprint, not on a quota.  curl is already a
    given on any box that can run ffmpeg, so shelling out is the reliable path;
    httpx stays as the fallback for an environment without curl.
    """
    if shutil.which("curl"):
        proc = subprocess.run(
            ["curl", "-sS", "-L", "--max-time", str(timeout),
             "-w", "\n%{http_code}", "--output", "-", url],
            capture_output=True,
        )
        if proc.returncode == 0 and proc.stdout:
            body, _, code = proc.stdout.rpartition(b"\n")
            try:
                return int(code.decode().strip()), body
            except ValueError:
                pass
        raise RuntimeError(
            f"curl exit {proc.returncode}: {proc.stderr.decode('utf-8', 'replace')[:160]}"
        )
    response = httpx.get(url, timeout=timeout, follow_redirects=True)
    return response.status_code, response.content


def _pollinations(prompt: str, out_path: Path, index: int) -> Path:
    global _last_pollinations_call

    seed = _seed_for(prompt, index)
    # Ask for a taller-than-target render; the service rounds dimensions down,
    # and the extra height absorbs the watermark trim without losing framing.
    # No `model` or `referrer` parameter: both move the request onto the paid
    # tier and come back 402 for an anonymous caller.
    url = (
        "https://image.pollinations.ai/prompt/"
        + quote(f"{prompt}. vertical 9:16 composition", safe="")
        + f"?width=768&height=1365&nologo=true&seed={seed}"
    )
    last_exc: Exception | None = None
    for attempt in range(6):
        try:
            wait = _POLLINATIONS_MIN_GAP - (time.time() - _last_pollinations_call)
            if wait > 0:
                time.sleep(wait)
            _last_pollinations_call = time.time()
            status, content = _http_get_bytes(url)
            if status in (402, 429, 503):
                raise RuntimeError(f"rate limited (HTTP {status})")
            if status >= 400:
                raise RuntimeError(f"HTTP {status}")
            if len(content) < 2000:
                raise RuntimeError(f"image too small ({len(content)} bytes)")
            img = Image.open(io.BytesIO(content))
            img.load()
            # The hosted service stamps a logo in the bottom-right even with
            # nologo=true, so always trim a strip proportional to the render.
            trim = max(24, int(img.height * 0.045))
            _fit_vertical(img, trim_bottom=trim).save(out_path, "PNG")
            return out_path
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            log.warning("pollinations attempt %s failed: %s", attempt + 1, exc)
            time.sleep(30 + 20 * attempt)
    raise RuntimeError(f"pollinations failed: {last_exc}")


def _hfspace(prompt: str, out_path: Path, index: int) -> Path:
    from . import hfspace

    img = hfspace.generate(prompt, seed=_seed_for(prompt, index))
    _fit_vertical(img).save(out_path, "PNG")
    return out_path


def _sdwebui(prompt: str, out_path: Path, index: int) -> Path:
    import base64

    payload = {
        "prompt": prompt,
        "negative_prompt": _negative,
        "width": 768,
        "height": 1344,
        "steps": 28,
        "cfg_scale": 6.5,
        "sampler_name": "DPM++ 2M",
        "seed": _seed_for(prompt, index),
    }
    if settings.sdwebui_model:
        payload["override_settings"] = {"sd_model_checkpoint": settings.sdwebui_model}
    response = httpx.post(f"{settings.sdwebui_url}/sdapi/v1/txt2img", json=payload, timeout=600)
    response.raise_for_status()
    images = response.json().get("images") or []
    if not images:
        raise RuntimeError("sdwebui returned no images")
    img = Image.open(io.BytesIO(base64.b64decode(images[0].split(",", 1)[-1])))
    _fit_vertical(img).save(out_path, "PNG")
    return out_path


def _comfyui(prompt: str, out_path: Path, index: int) -> Path:
    """Minimal txt2img graph posted to ComfyUI, then polled for the result."""
    import json as _json
    import uuid

    client_id = str(uuid.uuid4())
    graph = {
        "3": {
            "class_type": "KSampler",
            "inputs": {
                "seed": _seed_for(prompt, index), "steps": 28, "cfg": 6.5,
                "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 1.0,
                "model": ["4", 0], "positive": ["6", 0], "negative": ["7", 0],
                "latent_image": ["5", 0],
            },
        },
        "4": {"class_type": "CheckpointLoaderSimple",
              "inputs": {"ckpt_name": settings.sdwebui_model or "sd_xl_base_1.0.safetensors"}},
        "5": {"class_type": "EmptyLatentImage",
              "inputs": {"width": 768, "height": 1344, "batch_size": 1}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["4", 1]}},
        "7": {"class_type": "CLIPTextEncode", "inputs": {"text": _negative, "clip": ["4", 1]}},
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0], "vae": ["4", 2]}},
        "9": {"class_type": "SaveImage", "inputs": {"filename_prefix": "shortsmith", "images": ["8", 0]}},
    }
    queued = httpx.post(
        f"{settings.comfyui_url}/prompt",
        content=_json.dumps({"prompt": graph, "client_id": client_id}),
        headers={"Content-Type": "application/json"},
        timeout=60,
    )
    queued.raise_for_status()
    prompt_id = queued.json()["prompt_id"]

    deadline = time.time() + 600
    while time.time() < deadline:
        history = httpx.get(f"{settings.comfyui_url}/history/{prompt_id}", timeout=30).json()
        entry = history.get(prompt_id)
        if entry:
            for node in entry.get("outputs", {}).values():
                for image in node.get("images", []):
                    raw = httpx.get(
                        f"{settings.comfyui_url}/view",
                        params={
                            "filename": image["filename"],
                            "subfolder": image.get("subfolder", ""),
                            "type": image.get("type", "output"),
                        },
                        timeout=120,
                    ).content
                    _fit_vertical(Image.open(io.BytesIO(raw))).save(out_path, "PNG")
                    return out_path
            raise RuntimeError("comfyui finished without an image")
        time.sleep(2)
    raise RuntimeError("comfyui timed out")


_STOPWORDS = {
    "cinematic", "film", "still", "shot", "light", "lighting", "no", "text", "watermark",
    "the", "a", "an", "of", "in", "on", "with", "and", "very", "highly", "detailed",
    "palette", "depth", "field", "shallow", "focus", "photography", "realism", "35mm",
    "close", "up", "wide", "mood", "muted", "soft", "natural",
}


def _pexels(prompt: str, out_path: Path, index: int) -> Path:
    if not settings.pexels_api_key:
        raise RuntimeError("PEXELS_API_KEY is not set")
    words = [w for w in (t.strip(" ,.") for t in prompt.lower().split()) if w and w not in _STOPWORDS]
    query = " ".join(words[:4]) or "cinematic"
    search = httpx.get(
        "https://api.pexels.com/v1/search",
        headers={"Authorization": settings.pexels_api_key},
        params={"query": query, "orientation": "portrait", "per_page": 12, "size": "large"},
        timeout=60,
    )
    search.raise_for_status()
    photos = search.json().get("photos") or []
    if not photos:
        raise RuntimeError(f"pexels has no portrait result for '{query}'")
    photo = photos[_seed_for(prompt, index) % len(photos)]
    raw = httpx.get(photo["src"]["large2x"], timeout=120, follow_redirects=True).content
    _fit_vertical(Image.open(io.BytesIO(raw))).save(out_path, "PNG")
    return out_path


def _gradient(prompt: str, out_path: Path, index: int) -> Path:
    """Offline, deterministic, no network.

    Delegates to the plate compositor, which reads the prompt for mood and
    setting and builds a layered cinematic background.  Kept under the old name
    because `gradient` is the provider value stored in existing jobs.
    """
    from .plates import render_plate

    return render_plate(prompt, out_path, index)


def _flat_gradient(prompt: str, out_path: Path, index: int) -> Path:
    """The simplest possible plate: two-stop gradient, glow, grain, vignette.

    Only used if the compositor itself throws, which would mean a broken Pillow
    or numpy install rather than anything to do with this project.
    """
    rng = random.Random(_seed_for(prompt, index))
    width, height = TARGET
    base_hue = rng.random()
    accent_hue = (base_hue + rng.uniform(0.08, 0.22)) % 1.0

    top = tuple(int(c * 255) for c in colorsys.hls_to_rgb(base_hue, 0.16, 0.45))
    bottom = tuple(int(c * 255) for c in colorsys.hls_to_rgb(accent_hue, 0.42, 0.55))

    img = Image.new("RGB", (width, height))
    draw = ImageDraw.Draw(img)
    for y in range(height):
        t = y / max(1, height - 1)
        eased = t * t * (3 - 2 * t)
        draw.line(
            [(0, y), (width, y)],
            fill=tuple(int(top[i] + (bottom[i] - top[i]) * eased) for i in range(3)),
        )

    # A soft off-centre light source gives the Ken Burns move something to bite on.
    glow = Image.new("L", (width, height), 0)
    gdraw = ImageDraw.Draw(glow)
    cx = int(width * rng.uniform(0.25, 0.75))
    cy = int(height * rng.uniform(0.25, 0.6))
    radius = int(min(width, height) * rng.uniform(0.45, 0.8))
    gdraw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], fill=120)
    glow = glow.filter(ImageFilter.GaussianBlur(radius // 3))
    highlight = Image.new("RGB", (width, height),
                          tuple(int(c * 255) for c in colorsys.hls_to_rgb(accent_hue, 0.72, 0.5)))
    img = Image.composite(highlight, img, glow.point(lambda v: min(255, v)))
    img = Image.blend(img, Image.new("RGB", (width, height), (0, 0, 0)), 0.12)

    # Film grain, at a visible but not noisy strength.
    grain = Image.effect_noise((width, height), 22).convert("L")
    img = Image.blend(img, Image.merge("RGB", (grain, grain, grain)), 0.045)

    # Vignette so burned-in captions stay readable at the bottom.
    vignette = Image.new("L", (width, height), 0)
    vdraw = ImageDraw.Draw(vignette)
    vdraw.ellipse(
        [-width // 3, -height // 6, width + width // 3, height + height // 6], fill=255
    )
    vignette = vignette.filter(ImageFilter.GaussianBlur(width // 8))
    img = Image.composite(img, Image.new("RGB", (width, height), (0, 0, 0)), vignette)
    img.save(out_path, "PNG")
    return out_path


PROVIDERS = {
    "hfspace": _hfspace,
    "pollinations": _pollinations,
    "sdwebui": _sdwebui,
    "comfyui": _comfyui,
    "pexels": _pexels,
    "gradient": _gradient,
}


def generate_image(prompt: str, out_path: Path, index: int = 0, provider: str | None = None) -> tuple[Path, str]:
    """Render one still.  Returns (path, provider actually used)."""
    name = (provider or settings.image_provider or "pollinations").lower()
    fn = PROVIDERS.get(name)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Try the chosen provider, then the other key-free hosted one, then fall
    # back to a plate.  Both free generators share the failure mode of a spent
    # quota, and they do not run out at the same time, so one covers the other.
    chain = [name]
    if name in {"hfspace", "pollinations"}:
        chain.append("pollinations" if name == "hfspace" else "hfspace")
    for candidate in chain:
        fn = PROVIDERS.get(candidate)
        if fn is None or candidate == "gradient":
            continue
        try:
            return fn(prompt, out_path, index), candidate
        except Exception as exc:  # noqa: BLE001 - one scene must not kill the render
            log.warning("image provider %s failed for scene %s: %s", candidate, index, exc)
    try:
        return _gradient(prompt, out_path, index), "plate"
    except Exception as exc:  # noqa: BLE001
        log.warning("plate compositor failed for scene %s: %s", index, exc)
        return _flat_gradient(prompt, out_path, index), "gradient"


def provider_status() -> dict[str, object]:
    name = (settings.image_provider or "pollinations").lower()
    try:
        if name == "hfspace":
            from . import hfspace

            return hfspace.status()
        if name == "pollinations":
            status, _ = _http_get_bytes("https://image.pollinations.ai/prompt/test?width=64&height=64", timeout=30)
            return {"ok": status < 400, "provider": name, "detail": f"HTTP {status}"}
        if name in {"sdwebui", "comfyui"}:
            url = settings.sdwebui_url if name == "sdwebui" else settings.comfyui_url
            path = "/sdapi/v1/options" if name == "sdwebui" else "/system_stats"
            response = httpx.get(url + path, timeout=5)
            response.raise_for_status()
            return {"ok": True, "provider": name, "detail": url}
        if name == "pexels":
            if not settings.pexels_api_key:
                return {"ok": False, "provider": name, "detail": "PEXELS_API_KEY is not set"}
            return {"ok": True, "provider": name, "detail": "key present"}
        return {"ok": True, "provider": "gradient", "detail": "offline procedural"}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "provider": name, "detail": str(exc)[:160]}
