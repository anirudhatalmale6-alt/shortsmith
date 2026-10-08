"""The Hugging Face Space image provider.

No network: the Gradio HTTP contract is exercised against a stubbed client, and
the parts that have actually bitten in practice (int32 seeds, per-Space frame
sizes, quota errors arriving as HTTP 200, letterboxed FLUX output) each get a
test.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from shortsmith.providers import hfspace


# ---------------------------------------------------------------------------
# argument building
# ---------------------------------------------------------------------------

def test_flux_arguments_are_in_the_order_the_space_expects() -> None:
    args = hfspace._build_args_for(
        "black-forest-labs-flux-1-schnell", "a street", 42, 768, 1344)
    assert args == ["a street", 42, False, 768, 1344, 4]


def test_a_seed_above_int32_is_wrapped_not_rejected() -> None:
    """Gradio number inputs are int32; a bigger seed is an error, not a wrap."""
    args = hfspace._build_args_for(
        "black-forest-labs-flux-1-schnell", "x", 2_200_108_897, 768, 1344)
    assert 0 <= args[1] <= hfspace.MAX_SEED


def test_a_negative_seed_is_made_positive() -> None:
    args = hfspace._build_args_for("black-forest-labs-flux-1-schnell", "x", -5, 768, 1344)
    assert args[1] >= 0


def test_an_unknown_space_falls_back_to_the_default_shape() -> None:
    args = hfspace._build_args_for("someone-elses-space", "a street", 7, 768, 1344)
    assert args[0] == "a street"


def test_every_known_space_declares_a_function_and_a_frame_size() -> None:
    for host, spec in hfspace.SPACES.items():
        assert spec["fn"], host
        width, height = spec["size"]
        assert width > 0 and height > 0, host
        assert "{prompt}" in spec["args"], host


def test_space_host_accepts_the_owner_slash_name_form(monkeypatch) -> None:
    monkeypatch.setenv("HF_SPACE", "black-forest-labs/FLUX.1-schnell")
    assert hfspace._space_host() == "black-forest-labs-flux-1-schnell"


def test_space_host_defaults_when_unset(monkeypatch) -> None:
    monkeypatch.delenv("HF_SPACE", raising=False)
    assert hfspace._space_host() == hfspace.DEFAULT_SPACE


# ---------------------------------------------------------------------------
# letterbox trimming
# ---------------------------------------------------------------------------

def _framed(width: int, height: int, bar: int) -> Image.Image:
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    arr[bar : height - bar, :, :] = 200
    return Image.fromarray(arr, "RGB")


def test_black_bars_are_cropped_off() -> None:
    out = hfspace._trim_letterbox(_framed(768, 1344, 100))
    assert out.height == 1344 - 200
    assert out.width == 768


def test_a_picture_with_no_bars_is_left_alone() -> None:
    arr = np.full((1344, 768, 3), 180, dtype=np.uint8)
    img = Image.fromarray(arr, "RGB")
    assert hfspace._trim_letterbox(img).size == img.size


def test_a_legitimately_dark_picture_is_not_destroyed() -> None:
    """A night shot is mostly black; cropping it to its one bright patch would
    throw the picture away, so the trim refuses to take more than half."""
    arr = np.zeros((1344, 768, 3), dtype=np.uint8)
    arr[600:650, 300:360, :] = 255        # one small lamp
    out = hfspace._trim_letterbox(Image.fromarray(arr, "RGB"))
    assert out.size == (768, 1344)


# ---------------------------------------------------------------------------
# the HTTP contract
# ---------------------------------------------------------------------------

class _Response:
    def __init__(self, *, json_body=None, text="", content=b"", status=200):
        self._json = json_body or {}
        self.text = text
        self.content = content
        self.status_code = status

    def json(self):
        return self._json

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class _Client:
    """Stands in for httpx.Client; replays a scripted POST then GETs."""

    def __init__(self, post, gets):
        self._post, self._gets = post, list(gets)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def post(self, *a, **k):
        return self._post

    def get(self, *a, **k):
        return self._gets.pop(0)


def _png_bytes() -> bytes:
    import io

    buffer = io.BytesIO()
    Image.fromarray(np.full((1344, 768, 3), 160, dtype=np.uint8), "RGB").save(buffer, "PNG")
    return buffer.getvalue()


@pytest.fixture(autouse=True)
def _no_pacing(monkeypatch):
    monkeypatch.setattr(hfspace.time, "sleep", lambda *_: None)
    monkeypatch.setattr(hfspace, "_last_call", 0.0)


def test_a_successful_call_returns_the_image(monkeypatch) -> None:
    stream = (
        'event: complete\ndata: [{"path": "/tmp/x.webp", '
        '"url": "https://host/gradio_api/file=/tmp/x.webp"}, 1]\n'
    )
    client = _Client(_Response(json_body={"event_id": "abc"}),
                     [_Response(text=stream), _Response(content=_png_bytes())])
    monkeypatch.setattr(hfspace.httpx, "Client", lambda **k: client)
    img = hfspace.generate("a street", 7)
    assert img.size == (768, 1344)


def test_a_quota_error_arrives_as_http_200_and_is_classified_as_busy(monkeypatch) -> None:
    """The Space answers 200 with an error event, so a status check alone misses it."""
    stream = ('event: error\ndata: {"error": "You have exceeded your ZeroGPU quota '
              '(65s requested vs. 0s left)."}\n')
    monkeypatch.setattr(
        hfspace.httpx, "Client",
        lambda **k: _Client(_Response(json_body={"event_id": "abc"}), [_Response(text=stream)]),
    )
    with pytest.raises(hfspace.SpaceBusy, match="busy"):
        hfspace._generate_on(hfspace.DEFAULT_SPACE, "a street", 7, 768, 1344, 10)


def test_a_bare_null_error_is_also_treated_as_busy(monkeypatch) -> None:
    monkeypatch.setattr(
        hfspace.httpx, "Client",
        lambda **k: _Client(_Response(json_body={"event_id": "abc"}),
                            [_Response(text="event: error\ndata: null\n")]),
    )
    with pytest.raises(hfspace.SpaceBusy):
        hfspace._generate_on(hfspace.DEFAULT_SPACE, "a street", 7, 768, 1344, 10)


def test_generate_moves_on_to_the_next_space_when_one_is_busy(monkeypatch) -> None:
    tried: list[str] = []
    good = Image.fromarray(np.full((1344, 768, 3), 120, dtype=np.uint8), "RGB")

    def fake(host, *a, **k):
        tried.append(host)
        if host == hfspace.DEFAULT_SPACE:
            raise hfspace.SpaceBusy("quota")
        return good

    monkeypatch.setattr(hfspace, "_generate_on", fake)
    assert hfspace.generate("a street", 7) is good
    assert tried[0] == hfspace.DEFAULT_SPACE
    assert len(tried) >= 2


def test_generate_gives_up_with_a_message_naming_the_cause(monkeypatch) -> None:
    monkeypatch.setattr(
        hfspace, "_generate_on",
        lambda *a, **k: (_ for _ in ()).throw(hfspace.SpaceBusy("quota spent")),
    )
    with pytest.raises(RuntimeError, match="quota spent"):
        hfspace.generate("a street", 7)


def test_a_truncated_download_is_rejected(monkeypatch) -> None:
    stream = 'event: complete\ndata: [{"url": "https://host/file"}, 1]\n'
    monkeypatch.setattr(
        hfspace.httpx, "Client",
        lambda **k: _Client(_Response(json_body={"event_id": "abc"}),
                            [_Response(text=stream), _Response(content=b"tiny")]),
    )
    with pytest.raises(RuntimeError, match="bytes"):
        hfspace._generate_on(hfspace.DEFAULT_SPACE, "a street", 7, 768, 1344, 10)


# ---------------------------------------------------------------------------
# status panel
# ---------------------------------------------------------------------------

def test_status_says_plainly_that_a_token_is_needed(monkeypatch) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    info = hfspace.status()
    assert info["ok"] is False
    assert "HF_TOKEN" in info["detail"]


def test_a_token_is_sent_as_a_bearer_header(monkeypatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf_example")
    assert hfspace._headers()["Authorization"] == "Bearer hf_example"


def test_no_token_means_no_auth_header(monkeypatch) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    assert hfspace._headers() == {}
