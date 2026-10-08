"""Image generation through a public Hugging Face Space.

Free, no account, no API key, and the model behind the default Space is
FLUX.1-schnell, which is open weights and genuinely good at cinematic stills.
That combination is hard to beat for a project whose brief says to prioritise
free and open source tooling.

How the Gradio 4/5 HTTP API works, because it is not obvious:

    POST /gradio_api/call/<fn_name>   {"data": [...]}   -> {"event_id": "..."}
    GET  /gradio_api/call/<fn_name>/<event_id>          -> SSE stream
                                                           event: complete
                                                           data: [{...FileData}]

The FileData carries a `url` on the Space's own host, which is then fetched as
a normal file.  There is no websocket and no queue token to manage.

Caveats that matter in production, all handled below:

- **A free Hugging Face token is effectively required.**  Anonymous callers get
  a ZeroGPU allowance of roughly zero seconds, and the Space answers
  "You have exceeded your ZeroGPU quota (65s requested vs. 0s left)".  Signing up
  is free, needs no card, and the token goes in HF_TOKEN.  Without it a render
  will mostly fall through to the next provider.
- The quota error arrives as an SSE error event with HTTP 200, so the stream
  body has to be inspected, not just the status code.
- A Space can be asleep.  The first call then takes 30 to 90 seconds while it
  wakes, which is why the poll window is generous.
- Different Spaces expose different argument lists.  The argument order is
  configuration, not code, so pointing at another Space is an env change.
- FLUX likes to render an actual letterboxed film frame when you ask for a
  "cinematic film still", so the black bars are detected and trimmed; left in,
  they become grey bars after the Ken Burns zoom.
"""

from __future__ import annotations

import io
import json
import logging
import os
import re
import time
from pathlib import Path

import httpx
import numpy as np
from PIL import Image

from ..config import settings

log = logging.getLogger(__name__)

DEFAULT_SPACE = "black-forest-labs-flux-1-schnell"
DEFAULT_FN = "infer"

# Public Spaces share a ZeroGPU pool per IP and answer with an error event, not
# an HTTP status, once it is spent.  Each Space has its own pool, so rotating
# through a few gets a whole video rendered where one alone stalls after a
# couple of images.
MIN_GAP_SECONDS = float(os.environ.get("HF_SPACE_GAP", "4"))
_last_call = 0.0

# Each Space: its Gradio function name, its positional arguments, and the
# largest 9:16 frame it will accept.  All three differ per Space, which is why
# a single hard-coded call only ever works against one of them.
# {prompt} {seed} {width} {height} are substituted; the rest is literal.
SPACES: dict[str, dict] = {
    "black-forest-labs-flux-1-schnell": {
        "fn": "infer",
        "args": ["{prompt}", "{seed}", False, "{width}", "{height}", 4],
        "size": (768, 1344),
    },
    "stabilityai-stable-diffusion-3-5-large": {
        "fn": "infer",
        # This one caps any dimension at 1024, so the frame is 576x1024.
        "args": ["{prompt}", "", "{seed}", False, "{width}", "{height}", 4.5, 40],
        "size": (576, 1024),
    },
}

FALLBACK_SPACES = tuple(SPACES)


def _space_host() -> str:
    space = os.environ.get("HF_SPACE", DEFAULT_SPACE).strip() or DEFAULT_SPACE
    # Accept either "owner/Name" or the hyphenated host form.
    if "/" in space:
        space = space.replace("/", "-").replace(".", "-").replace("_", "-").lower()
    return space


def _headers() -> dict[str, str]:
    token = os.environ.get("HF_TOKEN", "").strip()
    return {"Authorization": f"Bearer {token}"} if token else {}


def _spec_for(space: str) -> dict:
    spec = SPACES.get(space)
    if spec is not None:
        return spec
    raw = os.environ.get("HF_SPACE_ARGS", "").strip()
    return {
        "fn": os.environ.get("HF_SPACE_FN", DEFAULT_FN).strip() or DEFAULT_FN,
        "args": json.loads(raw) if raw else SPACES[DEFAULT_SPACE]["args"],
        "size": SPACES[DEFAULT_SPACE]["size"],
    }


# Gradio number inputs on these Spaces are int32, and a seed above 2^31-1 is
# rejected outright rather than wrapped.
MAX_SEED = 2 ** 31 - 1


def _build_args_for(space: str, prompt: str, seed: int, width: int, height: int) -> list:
    template = _spec_for(space)["args"]
    seed = abs(int(seed)) % MAX_SEED
    out = []
    for item in template:
        if item == "{prompt}":
            out.append(prompt)
        elif item == "{seed}":
            out.append(seed)
        elif item == "{width}":
            out.append(width)
        elif item == "{height}":
            out.append(height)
        else:
            out.append(item)
    return out


def _trim_letterbox(img: Image.Image, threshold: int = 18) -> Image.Image:
    """Crop uniform near-black bars from the edges.

    Asked for a "cinematic film still", FLUX frequently paints the black bars
    too.  Keeping them costs a tenth of the frame and, once the Ken Burns move
    scales the still, they drift into shot as grey wedges.
    """
    arr = np.asarray(img.convert("RGB")).astype(np.int16)
    rows = arr.max(axis=(1, 2))
    cols = arr.max(axis=(0, 2))

    def first_bright(values) -> int:
        above = np.nonzero(values > threshold)[0]
        return int(above[0]) if above.size else 0

    def last_bright(values) -> int:
        above = np.nonzero(values > threshold)[0]
        return int(above[-1]) + 1 if above.size else len(values)

    top, bottom = first_bright(rows), last_bright(rows)
    left, right = first_bright(cols), last_bright(cols)
    # Refuse a crop that would eat most of the picture: that means the still is
    # legitimately dark, not letterboxed.
    if (bottom - top) < img.height * 0.5 or (right - left) < img.width * 0.5:
        return img
    if (top, left, bottom, right) == (0, 0, img.height, img.width):
        return img
    return img.crop((left, top, right, bottom))


class SpaceBusy(RuntimeError):
    """The Space's free GPU quota is spent, or it is still waking up."""


def _generate_on(host: str, prompt: str, seed: int, width: int, height: int,
                 timeout: float) -> Image.Image:
    global _last_call

    spec = _spec_for(host)
    fn = spec["fn"]
    # Each Space has its own maximum frame; asking for more is a hard error, not
    # a clamp, so take the size from the Space rather than from the caller.
    width, height = spec["size"]
    base = f"https://{host}.hf.space/gradio_api/call/{fn}"
    payload = {"data": _build_args_for(host, prompt, seed, width, height)}

    wait = MIN_GAP_SECONDS - (time.time() - _last_call)
    if wait > 0:
        time.sleep(wait)
    _last_call = time.time()

    with httpx.Client(timeout=timeout, headers=_headers(), follow_redirects=True) as client:
        start = client.post(base, json=payload)
        if start.status_code == 429:
            raise SpaceBusy("the Space is rate limited right now")
        start.raise_for_status()
        event_id = start.json().get("event_id")
        if not event_id:
            raise RuntimeError(f"no event id from {host}: {start.text[:200]}")

        # The stream can sit silent while a sleeping Space boots, so read it to
        # completion rather than giving up on the first idle second.
        body = client.get(f"{base}/{event_id}").text

    if "event: error" in body or '"error"' in body:
        detail = " ".join(body.split())[:220]
        # A bare `event: error data: null` is what ZeroGPU sends when the shared
        # pool is busy, so treat an unexplained error as busy and retry rather
        # than as a permanent failure.
        if re.search(r"quota|gpu|exceeded|limit|null", detail, re.I):
            raise SpaceBusy(f"{host} is busy: {detail}")
        raise RuntimeError(f"{host} returned an error: {detail}")

    match = re.search(r'"url":\s*"([^"]+)"', body)
    if not match:
        raise RuntimeError(f"{host} returned no image: {' '.join(body.split())[:220]}")

    with httpx.Client(timeout=timeout, headers=_headers(), follow_redirects=True) as client:
        raw = client.get(match.group(1)).content
    if len(raw) < 2000:
        raise RuntimeError(f"{host} returned {len(raw)} bytes")

    img = Image.open(io.BytesIO(raw))
    img.load()
    return _trim_letterbox(img)


def generate(prompt: str, seed: int, width: int = 768, height: int = 1344,
             timeout: float = 240.0) -> Image.Image:
    """Try the preferred Space, then the others, backing off between rounds.

    The preferred Space is tried first on every round rather than being dropped
    after one failure: its pool frees up in seconds, and it is the one whose
    output the rest of the video was matched to.
    """
    preferred = _space_host()
    hosts = [preferred] + [h for h in FALLBACK_SPACES if h != preferred]
    last: Exception | None = None

    for round_index in range(4):
        for host in hosts:
            try:
                return _generate_on(host, prompt, seed, width, height, timeout)
            except SpaceBusy as exc:
                last = exc
                log.info("space %s busy: %s", host, exc)
            except Exception as exc:  # noqa: BLE001
                last = exc
                log.warning("space %s failed: %s", host, exc)
        time.sleep(10 + 15 * round_index)
    raise RuntimeError(f"every Hugging Face Space was busy: {last}")


def status() -> dict[str, object]:
    host = _space_host()
    if not os.environ.get("HF_TOKEN", "").strip():
        return {
            "ok": False,
            "provider": "hfspace",
            "detail": (
                f"{host} reachable, but HF_TOKEN is not set. Anonymous callers get "
                "almost no GPU quota. A free huggingface.co token (no card) fixes it."
            ),
        }
    try:
        response = httpx.get(f"https://{host}.hf.space/config", timeout=10,
                             headers=_headers(), follow_redirects=True)
        ok = response.status_code == 200
        return {"ok": ok, "provider": "hfspace", "detail": f"{host} (HTTP {response.status_code})"}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "provider": "hfspace", "detail": str(exc)[:160]}
