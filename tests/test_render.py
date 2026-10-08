"""End-to-end render.

Slow by design: it runs the real pipeline, with the real TTS and the real
ffmpeg, and asserts the file that comes out is actually a 1080x1920 Short with
audio. Nothing else proves the product works.

Uses the offline writer and the procedural visuals so it needs no network and
no API key.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from shortsmith.config import settings
from shortsmith.pipeline.render import (
    RenderOptions, _kenburns_filter, build_units, make_ambient_bed, probe_duration,
    render_short,
)
from shortsmith.providers.offline_writer import offline_script
from shortsmith.providers.plates import render_plate


def ffprobe(path: Path) -> dict:
    proc = subprocess.run(
        [settings.ffprobe, "-v", "error", "-show_format", "-show_streams",
         "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    )
    return json.loads(proc.stdout)


# ---------------------------------------------------------------------------
# units
# ---------------------------------------------------------------------------

def test_units_cover_hook_beats_and_closing_line() -> None:
    script = offline_script("Sad Story", 45, seed=3)
    units = build_units(script)
    assert len(units) == len(script.scenes) + 2
    assert units[0].text == script.hook
    assert units[-1].text == script.cta


def test_the_closing_line_reuses_the_last_still() -> None:
    script = offline_script("Sad Story", 45, seed=3)
    units = build_units(script)
    assert units[-1].visual_prompt == units[-2].visual_prompt


def test_the_hook_gets_its_own_prompt() -> None:
    script = offline_script("Sad Story", 45, seed=3)
    units = build_units(script)
    assert units[0].visual_prompt != units[1].visual_prompt


# ---------------------------------------------------------------------------
# Ken Burns
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("move", ["in", "out", "left", "right", "up"])
def test_kenburns_expressions_are_linear_in_the_frame_index(move: str) -> None:
    """The incremental `zoom+0.0005` form drifts; every move must use `on`."""
    chain = _kenburns_filter(move, 90)
    assert "zoompan" in chain
    assert "zoom+" not in chain
    assert "s=1080x1920" in chain
    assert "d=90" in chain


def test_kenburns_survives_a_one_frame_scene() -> None:
    # frames-1 is the divisor; a 1-frame scene must not divide by zero.
    assert _kenburns_filter("in", 1)
    assert _kenburns_filter("in", 0)


def test_kenburns_upscales_before_panning() -> None:
    chain = _kenburns_filter("right", 60)
    assert chain.startswith("scale=2160:3840")


# ---------------------------------------------------------------------------
# plates
# ---------------------------------------------------------------------------

def test_a_plate_is_rendered_at_the_video_resolution(tmp_path) -> None:
    from PIL import Image

    path = render_plate("a quiet harbour, melancholic, muted", tmp_path / "p.png", 0)
    with Image.open(path) as img:
        assert img.size == (settings.video_width, settings.video_height)


def test_plates_are_deterministic(tmp_path) -> None:
    one = render_plate("same prompt, melancholic", tmp_path / "a.png", 2).read_bytes()
    two = render_plate("same prompt, melancholic", tmp_path / "b.png", 2).read_bytes()
    assert one == two


def test_consecutive_beats_do_not_render_the_same_plate(tmp_path) -> None:
    prompt = "a room, melancholic, muted desaturated palette"
    frames = {render_plate(prompt, tmp_path / f"{i}.png", i).read_bytes() for i in range(4)}
    assert len(frames) == 4


# ---------------------------------------------------------------------------
# music bed
# ---------------------------------------------------------------------------

def test_the_built_in_music_bed_is_the_length_asked_for(tmp_path) -> None:
    path = make_ambient_bed(tmp_path / "bed.wav", 6.0)
    assert abs(probe_duration(path) - 6.0) < 0.3
    streams = ffprobe(path)["streams"]
    assert streams[0]["codec_type"] == "audio"


# ---------------------------------------------------------------------------
# the real thing
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_a_complete_short_is_rendered(tmp_path) -> None:
    script = offline_script("Sad Story", 30, seed=7)
    out = tmp_path / "short.mp4"
    result = render_short(
        script, tmp_path / "work", out,
        RenderOptions(image_provider="gradient", caption_style="bold_yellow",
                      music="auto", motion=True),
    )

    assert out.exists() and out.stat().st_size > 200_000
    info = ffprobe(out)
    video = next(s for s in info["streams"] if s["codec_type"] == "video")
    audio = next(s for s in info["streams"] if s["codec_type"] == "audio")

    assert (video["width"], video["height"]) == (1080, 1920)
    assert video["codec_name"] == "h264"
    assert audio["codec_name"] == "aac"
    assert int(audio["sample_rate"]) == 48000

    # A Short has to be under a minute, and long enough to be worth watching.
    assert 12 < result.duration < 60
    assert abs(float(info["format"]["duration"]) - result.duration) < 0.5

    assert result.thumbnail.exists()
    assert result.srt is not None and result.srt.exists()
    assert result.srt.read_text(encoding="utf-8").strip().startswith("1")

    # Captions were aligned, not guessed, and the first spoken words made it in.
    assert result.providers["captions"].startswith("whisper")
    first_word = script.hook.split()[0].strip(".,").lower()
    assert first_word in result.srt.read_text(encoding="utf-8").lower()


@pytest.mark.slow
def test_captions_can_be_turned_off(tmp_path) -> None:
    script = offline_script("Sad Story", 20, seed=11)
    out = tmp_path / "nocap.mp4"
    result = render_short(
        script, tmp_path / "work", out,
        RenderOptions(image_provider="gradient", caption_style="none", music=""),
    )
    assert out.exists()
    assert result.srt is None
