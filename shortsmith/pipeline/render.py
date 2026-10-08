"""Video assembly.

One still per story beat, a Ken Burns move on each, hard cuts on the narration
boundaries, burned-in word-by-word captions, and a ducked music bed.  All of it
is ffmpeg; there is no paid renderer and no cloud step.

The piece that matters for quality is that scene length is driven by the voice
track, not the other way round.  Each beat is spoken first, measured, and the
picture is then cut to fit it.  That is why the captions land on the word.
"""

from __future__ import annotations

import json
import logging
import math
import random
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

from ..config import settings
from ..providers import captions as cap
from ..providers import images, tts
from ..providers.script_types import ScriptResult

log = logging.getLogger(__name__)

W, H, FPS = settings.video_width, settings.video_height, settings.video_fps
GAP_SECONDS = 0.22          # breath between beats
TAIL_SECONDS = 0.45         # hold on the last frame so the CTA can land
MIN_UNIT_SECONDS = 1.2


@dataclass
class Unit:
    """One narrated beat: its text, its still, and its place on the timeline."""

    text: str
    visual_prompt: str
    audio: Path | None = None
    image: Path | None = None
    duration: float = 0.0
    start: float = 0.0
    image_provider: str = ""


@dataclass
class RenderOptions:
    voice: str = ""
    speed: float = 1.0
    caption_style: str = "bold_yellow"
    words_per_line: int = 3
    uppercase_captions: bool = True
    image_provider: str = ""
    music: str = ""              # filename inside assets/music, or "" for none
    music_gain_db: float = -22.0
    motion: bool = True
    crf: int = 20


@dataclass
class RenderResult:
    video: Path
    thumbnail: Path
    duration: float
    srt: Path | None = None
    log: list[str] = field(default_factory=list)
    providers: dict[str, str] = field(default_factory=dict)


def _run(cmd: Sequence[str], desc: str) -> None:
    log.debug("%s: %s", desc, " ".join(str(c) for c in cmd))
    proc = subprocess.run(
        [str(c) for c in cmd], capture_output=True, text=True
    )
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-12:]
        raise RuntimeError(f"{desc} failed:\n" + "\n".join(tail))


def probe_duration(path: Path) -> float:
    proc = subprocess.run(
        [settings.ffprobe, "-v", "error", "-show_entries", "format=duration",
         "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    )
    return float(json.loads(proc.stdout)["format"]["duration"])


def _threads_args() -> list[str]:
    return ["-threads", str(settings.render_threads)] if settings.render_threads else []


# ---------------------------------------------------------------------------
# Ken Burns
# ---------------------------------------------------------------------------

MOVES = ("in", "out", "left", "right", "up")


def _kenburns_filter(move: str, frames: int) -> str:
    """Build a zoompan expression for one still.

    Expressions are linear in `on` (the output frame index) rather than the
    `zoom+0.0005` form you see everywhere, because the incremental form
    accumulates rounding error and visibly stutters on long holds.
    """
    frames = max(2, frames)
    last = frames - 1
    amount = 0.16
    pan_zoom = 1.14

    if move == "in":
        z = f"1+{amount}*on/{last}"
        x, y = "iw/2-(iw/zoom/2)", "ih/2-(ih/zoom/2)"
    elif move == "out":
        z = f"{1 + amount}-{amount}*on/{last}"
        x, y = "iw/2-(iw/zoom/2)", "ih/2-(ih/zoom/2)"
    elif move == "left":
        z = str(pan_zoom)
        x, y = f"(iw-iw/zoom)*(1-on/{last})", "ih/2-(ih/zoom/2)"
    elif move == "right":
        z = str(pan_zoom)
        x, y = f"(iw-iw/zoom)*on/{last}", "ih/2-(ih/zoom/2)"
    else:  # up
        z = str(pan_zoom)
        x, y = "iw/2-(iw/zoom/2)", f"(ih-ih/zoom)*(1-on/{last})"

    # Upscale before zoompan: zoompan samples at the input resolution, so
    # feeding it a 1080-wide still makes the pan soft and steppy.
    return (
        f"scale={W * 2}:{H * 2}:flags=lanczos,setsar=1,"
        f"zoompan=z='{z}':x='{x}':y='{y}':d={frames}:s={W}x{H}:fps={FPS},"
        f"format=yuv420p"
    )


def _build_scene_clip(image: Path, out: Path, duration: float, move: str) -> None:
    frames = max(2, int(round(duration * FPS)))
    _run(
        [settings.ffmpeg, "-y", "-loglevel", "error",
         "-loop", "1", "-framerate", str(FPS), "-t", f"{duration:.3f}", "-i", str(image),
         "-vf", _kenburns_filter(move, frames),
         "-frames:v", str(frames),
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
         "-pix_fmt", "yuv420p", *_threads_args(), str(out)],
        f"scene clip {out.name}",
    )


# ---------------------------------------------------------------------------
# audio
# ---------------------------------------------------------------------------

def _silence(path: Path, seconds: float) -> None:
    _run(
        [settings.ffmpeg, "-y", "-loglevel", "error",
         "-f", "lavfi", "-i", "anullsrc=channel_layout=mono:sample_rate=48000",
         "-t", f"{seconds:.3f}", "-c:a", "pcm_s16le", str(path)],
        "silence",
    )


def _concat_audio(parts: Sequence[Path], out: Path, work: Path) -> None:
    listing = work / "audio_concat.txt"
    listing.write_text(
        "\n".join(f"file '{p.resolve().as_posix()}'" for p in parts) + "\n", encoding="utf-8"
    )
    _run(
        [settings.ffmpeg, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
         "-i", str(listing), "-c:a", "pcm_s16le", "-ar", "48000", "-ac", "1", str(out)],
        "narration concat",
    )


def make_ambient_bed(path: Path, seconds: float, key: str = "minor") -> Path:
    """Synthesize a royalty-free ambient pad with ffmpeg alone.

    Three detuned sines a minor (or major) triad apart, slow tremolo, heavy
    low-pass and a long fade.  It is deliberately plain: it sits under
    narration without competing, and it carries no licence obligations because
    nothing was sampled.
    """
    root = 110.0
    third = root * (2 ** (3 / 12)) if key == "minor" else root * (2 ** (4 / 12))
    fifth = root * (2 ** (7 / 12))
    octave = root * 2
    inputs: list[str] = []
    for freq in (root, third, fifth, octave):
        inputs += ["-f", "lavfi", "-t", f"{seconds:.3f}",
                   "-i", f"sine=frequency={freq:.2f}:sample_rate=48000"]
    filters = (
        "[0:a][1:a][2:a][3:a]amix=inputs=4:duration=longest:normalize=0,"
        "volume=0.22,tremolo=f=0.12:d=0.5,lowpass=f=900,highpass=f=60,"
        "aformat=channel_layouts=stereo,"
        f"afade=t=in:st=0:d=2.5,afade=t=out:st={max(0.0, seconds - 3):.3f}:d=3[a]"
    )
    _run(
        [settings.ffmpeg, "-y", "-loglevel", "error", *inputs,
         "-filter_complex", filters, "-map", "[a]",
         "-c:a", "pcm_s16le", "-ar", "48000", str(path)],
        "ambient bed",
    )
    return path


# ---------------------------------------------------------------------------
# main entry point
# ---------------------------------------------------------------------------

def build_units(script: ScriptResult) -> list[Unit]:
    """Hook, beats and closing line as one ordered list of narrated units.

    The hook gets its own still built from the first beat's prompt so the first
    second is not a reused frame; the closing line reuses the final still.
    """
    units: list[Unit] = []
    first_prompt = script.scenes[0].visual_prompt if script.scenes else "cinematic still, moody"
    if script.hook:
        hook_prompt = f"establishing wide shot, {first_prompt}"
        units.append(Unit(text=script.hook, visual_prompt=hook_prompt))
    for scene in script.scenes:
        units.append(Unit(text=scene.narration, visual_prompt=scene.visual_prompt))
    if script.cta:
        last = units[-1].visual_prompt if units else first_prompt
        units.append(Unit(text=script.cta, visual_prompt=last))
    return units


def render_short(
    script: ScriptResult,
    work_dir: Path,
    out_path: Path,
    options: RenderOptions | None = None,
    progress: Callable[[str, int], None] | None = None,
) -> RenderResult:
    options = options or RenderOptions()
    work_dir.mkdir(parents=True, exist_ok=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    notes: list[str] = []
    used: dict[str, str] = {}

    def step(message: str, percent: int) -> None:
        log.info("[render] %s", message)
        notes.append(message)
        if progress:
            progress(message, percent)

    units = build_units(script)
    if not units:
        raise ValueError("script produced no narration")

    # --- 1. voice -------------------------------------------------------
    step("Recording the voice over", 10)
    audio_parts: list[Path] = []
    cursor = 0.0
    for index, unit in enumerate(units):
        wav = work_dir / f"voice_{index:02d}.wav"
        tts.synthesize(unit.text, wav, voice_id=options.voice or None, speed=options.speed)
        unit.audio = wav
        unit.duration = max(MIN_UNIT_SECONDS, tts.wav_duration(wav))
        unit.start = cursor

        pad = GAP_SECONDS if index < len(units) - 1 else TAIL_SECONDS
        silence = work_dir / f"gap_{index:02d}.wav"
        _silence(silence, pad)
        audio_parts += [wav, silence]
        unit.duration += pad
        cursor += unit.duration
    used["voice"] = options.voice or settings.piper_voice

    narration = work_dir / "narration.wav"
    _concat_audio(audio_parts, narration, work_dir)
    total = probe_duration(narration)

    # --- 2. visuals -----------------------------------------------------
    step(f"Generating {len(units)} visuals", 25)
    cache: dict[str, Path] = {}
    providers_used: set[str] = set()
    for index, unit in enumerate(units):
        if unit.visual_prompt in cache:
            unit.image = cache[unit.visual_prompt]
            continue
        target = work_dir / f"scene_{index:02d}.png"
        path, provider = images.generate_image(
            unit.visual_prompt, target, index=index, provider=options.image_provider or None
        )
        unit.image = path
        unit.image_provider = provider
        providers_used.add(provider)
        cache[unit.visual_prompt] = path
        step(f"Visual {index + 1} of {len(units)} ready", 25 + int(30 * (index + 1) / len(units)))
    used["images"] = "+".join(sorted(providers_used))
    if "gradient" in providers_used and len(providers_used) > 1:
        notes.append("One or more stills fell back to the offline gradient background.")

    # --- 3. picture -----------------------------------------------------
    step("Cutting the picture to the narration", 60)
    rng = random.Random(abs(hash(script.title)) % (1 << 30))
    clips: list[Path] = []
    previous = ""
    for index, unit in enumerate(units):
        move = rng.choice([m for m in MOVES if m != previous]) if options.motion else "in"
        previous = move
        clip = work_dir / f"clip_{index:02d}.mp4"
        _build_scene_clip(unit.image, clip, unit.duration, move if options.motion else "in")
        clips.append(clip)

    listing = work_dir / "video_concat.txt"
    listing.write_text(
        "\n".join(f"file '{p.resolve().as_posix()}'" for p in clips) + "\n", encoding="utf-8"
    )
    silent_video = work_dir / "silent.mp4"
    _run(
        [settings.ffmpeg, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
         "-i", str(listing), "-c", "copy", str(silent_video)],
        "video concat",
    )

    # --- 4. captions ----------------------------------------------------
    ass_path: Path | None = None
    srt_path: Path | None = None
    if options.caption_style != "none":
        step("Aligning captions to the audio", 72)
        try:
            words = cap.transcribe_words(narration, clamp_to=total)
            used["captions"] = f"whisper {settings.whisper_model}"
        except Exception as exc:  # noqa: BLE001
            log.warning("whisper failed (%s); using estimated caption timing", exc)
            notes.append(f"Whisper unavailable ({exc}); caption timing estimated from text.")
            words = cap.words_from_text(script.narration_text, total)
            used["captions"] = "estimated"
        if words:
            ass_path = cap.build_ass(
                words,
                work_dir / "captions.ass",
                style=options.caption_style,
                words_per_line=options.words_per_line,
                uppercase=options.uppercase_captions,
            )
            srt_path = cap.write_srt(words, out_path.with_suffix(".srt"))

    # --- 5. mix and burn ------------------------------------------------
    step("Mixing audio and burning captions", 85)
    cmd: list[str] = [settings.ffmpeg, "-y", "-loglevel", "error", "-i", str(silent_video),
                      "-i", str(narration)]
    music_path: Path | None = None
    if options.music and options.music != "none":
        if options.music == "auto":
            music_path = make_ambient_bed(work_dir / "bed.wav", total)
        else:
            candidate = settings.music_dir / options.music
            if candidate.exists():
                music_path = candidate
            else:
                notes.append(f"Music track '{options.music}' not found; rendered without it.")
    if music_path:
        cmd += ["-stream_loop", "-1", "-i", str(music_path)]

    video_filter = "null"
    if ass_path:
        escaped = str(ass_path).replace("\\", "/").replace(":", r"\:")
        fonts = str(settings.fonts_dir).replace("\\", "/").replace(":", r"\:")
        video_filter = f"ass='{escaped}':fontsdir='{fonts}'"

    if music_path:
        # sidechaincompress ducks its FIRST input using its SECOND as the key,
        # so the music goes in first and a copy of the voice drives the gain.
        audio_filter = (
            "[1:a]aformat=channel_layouts=stereo,loudnorm=I=-15:TP=-1.5:LRA=11,asplit=2[v1][v2];"
            f"[2:a]aformat=channel_layouts=stereo,volume={options.music_gain_db}dB,"
            f"atrim=0:{total:.3f}[m];"
            "[m][v1]sidechaincompress=threshold=0.03:ratio=12:attack=15:release=350[mduck];"
            "[v2][mduck]amix=inputs=2:duration=first:normalize=0,alimiter=limit=0.95[aout]"
        )
        cmd += ["-filter_complex", f"[0:v]{video_filter}[vout];{audio_filter}",
                "-map", "[vout]", "-map", "[aout]"]
    else:
        cmd += ["-filter_complex",
                f"[0:v]{video_filter}[vout];"
                "[1:a]aformat=channel_layouts=stereo,loudnorm=I=-15:TP=-1.5:LRA=11,"
                "alimiter=limit=0.95[aout]",
                "-map", "[vout]", "-map", "[aout]"]

    cmd += ["-c:v", "libx264", "-preset", "medium", "-crf", str(options.crf),
            "-profile:v", "high", "-level", "4.1", "-pix_fmt", "yuv420p",
            "-g", str(FPS * 2), "-movflags", "+faststart",
            "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
            "-shortest", *_threads_args(), str(out_path)]
    _run(cmd, "final mux")

    # --- 6. thumbnail ---------------------------------------------------
    thumb = out_path.with_suffix(".jpg")
    _run(
        [settings.ffmpeg, "-y", "-loglevel", "error", "-ss", "0.6", "-i", str(out_path),
         "-frames:v", "1", "-q:v", "3", str(thumb)],
        "thumbnail",
    )

    duration = probe_duration(out_path)
    step(f"Done: {duration:.1f}s", 100)
    return RenderResult(
        video=out_path, thumbnail=thumb, duration=duration, srt=srt_path,
        log=notes, providers=used,
    )


def cleanup_work(work_dir: Path, keep: bool = False) -> None:
    if keep or not work_dir.exists():
        return
    shutil.rmtree(work_dir, ignore_errors=True)
