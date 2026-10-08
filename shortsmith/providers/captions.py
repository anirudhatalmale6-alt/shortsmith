"""Burned-in captions.

Word timings come from faster-whisper (CTranslate2 Whisper, MIT licensed, runs
on CPU).  Transcribing the voice track we just generated rather than trusting
the script text is deliberate: Piper expands "3am" and "Dr." its own way, and
only the audio knows where each word actually lands.

Output is an ASS subtitle file.  ASS, not SRT, because the look that performs on
Shorts needs per-word highlighting, a scale pop and a thick outline, and SRT
cannot express any of that.
"""

from __future__ import annotations

import logging
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from ..config import settings

log = logging.getLogger(__name__)

_model_cache: dict[str, object] = {}


@dataclass
class Word:
    text: str
    start: float
    end: float


STYLES: dict[str, dict[str, str]] = {
    # name -> ASS colours (&HAABBGGRR) and geometry
    "bold_yellow": {
        "label": "Bold yellow pop (classic Shorts)",
        "font": "Anton",
        "size": "112",
        "primary": "&H00FFFFFF",
        "highlight": "&H0000D7FF",   # amber
        "outline": "&H00000000",
        "outline_w": "7",
        "shadow": "3",
        "margin_v": "520",
    },
    "clean_white": {
        "label": "Clean white, grey inactive",
        "font": "Poppins ExtraBold",
        "size": "96",
        "primary": "&H00C8C8C8",
        "highlight": "&H00FFFFFF",
        "outline": "&H00000000",
        "outline_w": "6",
        "shadow": "2",
        "margin_v": "560",
    },
    "mint_pop": {
        "label": "Mint highlight",
        "font": "Anton",
        "size": "108",
        "primary": "&H00FFFFFF",
        "highlight": "&H009CFFB4",
        "outline": "&H00000000",
        "outline_w": "7",
        "shadow": "3",
        "margin_v": "520",
    },
    "none": {"label": "No captions"},
}


def load_audio_16k(path: Path) -> "object":
    """Decode any audio to the mono 16 kHz float32 array Whisper expects.

    faster-whisper will happily open a file path itself, but that route goes
    through PyAV, whose API changes between majors and has already broken this
    call once.  ffmpeg is a hard dependency of the project anyway, so decode
    with ffmpeg and hand Whisper a plain array: one less thing that can rot.
    """
    import numpy as np

    proc = subprocess.run(
        [settings.ffmpeg, "-nostdin", "-threads", "1", "-i", str(path),
         "-f", "f32le", "-ac", "1", "-ar", "16000", "-acodec", "pcm_f32le", "-"],
        capture_output=True,
    )
    if proc.returncode != 0:
        tail = proc.stderr.decode("utf-8", "replace").strip().splitlines()[-6:]
        raise RuntimeError("ffmpeg could not decode audio for Whisper:\n" + "\n".join(tail))
    return np.frombuffer(proc.stdout, dtype=np.float32).copy()


def _get_model():
    key = f"{settings.whisper_model}/{settings.whisper_device}/{settings.whisper_compute}"
    if key not in _model_cache:
        from faster_whisper import WhisperModel

        log.info("loading whisper model %s", key)
        _model_cache[key] = WhisperModel(
            settings.whisper_model,
            device=settings.whisper_device,
            compute_type=settings.whisper_compute,
        )
    return _model_cache[key]


def transcribe_words(audio_path: Path, language: str = "en", clamp_to: float | None = None) -> list[Word]:
    model = _get_model()
    segments, _info = model.transcribe(
        load_audio_16k(audio_path),
        language=language,
        word_timestamps=True,
        vad_filter=False,
        beam_size=5,
        condition_on_previous_text=False,
    )
    words: list[Word] = []
    for segment in segments:
        for word in segment.words or []:
            text = word.word.strip()
            if not text:
                continue
            start = float(word.start)
            end = max(float(word.end), start + 0.06)
            # Whisper occasionally emits an end before the next start; clamp so
            # the highlight never runs backwards.
            if words and start < words[-1].end:
                start = words[-1].end
                end = max(end, start + 0.06)
            words.append(Word(text=text, start=start, end=end))

    # Whisper can run its final word past the end of the file.  Left alone, the
    # last caption would still be on screen when the video has already cut, so
    # the burn-in would be trimmed mid-word.
    if clamp_to and words and words[-1].end > clamp_to:
        for word in words:
            word.start = min(word.start, clamp_to)
            word.end = min(word.end, clamp_to)
        words = [w for w in words if w.end > w.start]
    return words


def words_from_text(text: str, duration: float) -> list[Word]:
    """Even fallback timing when Whisper is unavailable.

    Weighted by word length so long words hold longer; better than a flat
    split and good enough that captions stay roughly in sync.
    """
    tokens = [t for t in re.split(r"\s+", text.strip()) if t]
    if not tokens:
        return []
    weights = [max(2, len(t)) for t in tokens]
    total = sum(weights)
    words: list[Word] = []
    cursor = 0.0
    for token, weight in zip(tokens, weights):
        span = duration * weight / total
        words.append(Word(text=token, start=cursor, end=cursor + span))
        cursor += span
    return words


def _ass_time(seconds: float) -> str:
    seconds = max(0.0, seconds)
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours}:{minutes:02d}:{secs:05.2f}"


def _escape(text: str) -> str:
    return text.replace("\\", "").replace("{", "(").replace("}", ")")


def _chunk(words: list[Word], per_line: int, max_gap: float) -> list[list[Word]]:
    """Group words into caption lines, breaking on sentence ends and on pauses."""
    chunks: list[list[Word]] = []
    current: list[Word] = []
    for word in words:
        if current:
            gap = word.start - current[-1].end
            ends_sentence = bool(re.search(r"[.!?]$", current[-1].text))
            if len(current) >= per_line or gap > max_gap or ends_sentence:
                chunks.append(current)
                current = []
        current.append(word)
    if current:
        chunks.append(current)
    return chunks


def build_ass(
    words: list[Word],
    out_path: Path,
    style: str = "bold_yellow",
    words_per_line: int = 3,
    uppercase: bool = True,
    max_gap: float = 0.55,
) -> Path | None:
    """Write an ASS file with one event per spoken word.

    The line stays on screen for its whole chunk; only the active word changes
    colour and scales up, which is what makes the text feel like it is being
    spoken rather than pasted on.
    """
    if style == "none" or not words:
        return None
    spec = STYLES.get(style, STYLES["bold_yellow"])

    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {settings.video_width}
PlayResY: {settings.video_height}
WrapStyle: 2
ScaledBorderAndShadow: yes
YCbCr Matrix: TV.709

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Pop,{spec['font']},{spec['size']},{spec['primary']},{spec['primary']},{spec['outline']},&H64000000,0,0,0,0,100,100,0,0,1,{spec['outline_w']},{spec['shadow']},2,90,90,{spec['margin_v']},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    lines: list[str] = []
    for chunk in _chunk(words, words_per_line, max_gap):
        chunk_start = chunk[0].start
        chunk_end = chunk[-1].end
        for idx, active in enumerate(chunk):
            start = active.start if idx else chunk_start
            end = chunk_end if idx == len(chunk) - 1 else chunk[idx + 1].start
            if end <= start:
                continue
            parts: list[str] = []
            for pos, word in enumerate(chunk):
                text = _escape(word.text)
                if uppercase:
                    text = text.upper()
                if pos == idx:
                    parts.append(
                        f"{{\\c{spec['highlight']}\\fscx108\\fscy108"
                        f"\\t(0,90,\\fscx118\\fscy118)\\t(90,200,\\fscx108\\fscy108)}}"
                        f"{text}{{\\r}}"
                    )
                else:
                    parts.append(text)
            body = " ".join(parts)
            # A short fade on the first word of a line stops it snapping in.
            prefix = "{\\fad(90,0)}" if idx == 0 else ""
            lines.append(
                f"Dialogue: 0,{_ass_time(start)},{_ass_time(end)},Pop,,0,0,0,,{prefix}{body}"
            )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(header + "\n".join(lines) + "\n", encoding="utf-8")
    return out_path


def write_srt(words: list[Word], out_path: Path, words_per_line: int = 7) -> Path:
    """A plain SRT alongside the ASS, for YouTube's own caption track."""
    chunks = _chunk(words, words_per_line, 0.9)
    lines: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
        def stamp(value: float) -> str:
            ms = int(round(value * 1000))
            return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"

        lines.append(str(index))
        lines.append(f"{stamp(chunk[0].start)} --> {stamp(chunk[-1].end)}")
        lines.append(" ".join(w.text for w in chunk).strip())
        lines.append("")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path
