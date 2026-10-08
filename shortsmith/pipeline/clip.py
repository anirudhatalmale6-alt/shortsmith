"""Turn one long YouTube video into several Shorts.

Pipeline: download with yt-dlp -> transcribe with Whisper -> score every
candidate window for how well it works as a standalone Short -> cut, reframe to
9:16, caption, export.

The scoring is the part that decides whether this is useful or just a chopper.
A good Short is not "any 40 seconds"; it is a window that opens on a complete
thought, contains a hook near its start, resolves, and does not end mid
sentence.  So candidates are built from sentence boundaries rather than a fixed
grid, and each one is scored on hook strength, self-containment, speech
density and length.  If a model is configured it re-ranks the shortlist, but
the heuristic alone already picks sensible cuts, which matters because this has
to work on a box with no LLM.
"""

from __future__ import annotations

import json
import logging
import math
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from ..config import settings
from ..providers import captions as cap
from ..providers.llm import _call_ollama, _call_openai, _extract_json

log = logging.getLogger(__name__)

W, H = settings.video_width, settings.video_height

HOOK_PATTERNS = (
    r"\bhere'?s (?:why|how|the)\b", r"\bthe (?:truth|problem|reason|secret|mistake)\b",
    r"\bnobody (?:tells|talks|knows)\b", r"\bmost people\b", r"\bdid you know\b",
    r"\bwhat (?:if|happens|nobody)\b", r"\bthis is (?:why|how|the)\b",
    r"\byou (?:need to|have to|should) (?:know|understand|hear)\b",
    r"\bi (?:was|had|never) \w+", r"\blet me (?:tell|show)\b",
    r"\bthe (?:craziest|weirdest|worst|best)\b", r"\bimagine\b",
    r"\bnever\b", r"\bactually\b", r"\?$",
)

NUMBER_WORDS = r"\b(one|two|three|four|five|ten|twenty|hundred|thousand|million|billion|percent|\d+)\b"

FILLER_ONLY = re.compile(r"^(?:\W|\b(?:um|uh|er|like|you know|so|and|but|okay|right)\b)+$", re.I)


@dataclass
class Sentence:
    text: str
    start: float
    end: float


@dataclass
class Candidate:
    start: float
    end: float
    text: str
    score: float = 0.0
    reasons: list[str] = field(default_factory=list)
    title: str = ""

    @property
    def duration(self) -> float:
        return self.end - self.start


def _local_path(url: str) -> Path | None:
    """A source that is already on this machine, so no download is needed.

    Accepts `file:///path`, a bare absolute path, or a path relative to the
    data directory.  Useful both for testing and for the common real case of
    dropping a long recording on the server instead of re-uploading it to
    YouTube first.
    """
    candidate = url[7:] if url.startswith("file://") else url
    if "://" in candidate:
        return None
    path = Path(candidate)
    if not path.is_absolute():
        path = settings.data_dir / candidate
    return path if path.is_file() else None


def ffprobe_info(path: Path) -> dict:
    proc = subprocess.run(
        [settings.ffprobe, "-v", "error", "-show_format", "-show_streams",
         "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    )
    data = json.loads(proc.stdout)
    return {
        "title": path.stem.replace("_", " ").replace("-", " ").strip(),
        "duration": float(data.get("format", {}).get("duration") or 0),
        "_local": True,
    }


def ytdlp_info(url: str) -> dict:
    local = _local_path(url)
    if local is not None:
        return ffprobe_info(local)

    cmd = ["yt-dlp", "--no-warnings", "--dump-single-json", "--no-playlist"]
    cmd += _auth_args()
    proc = subprocess.run(cmd + [url], capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(_explain_ytdlp_error(proc.stderr or proc.stdout))
    return json.loads(proc.stdout)


def _auth_args() -> list[str]:
    """Cookies and proxy, if the operator configured them.

    YouTube answers requests from most datacentre IP ranges with
    "Sign in to confirm you're not a bot", so a server deployment usually needs
    either a cookies.txt exported from a signed-in browser or an outbound proxy.
    Silently failing on that is the single most confusing thing this feature can
    do, hence the explicit error text below.
    """
    import os

    args: list[str] = []
    cookies = os.environ.get("YTDLP_COOKIES", "").strip()
    if cookies and Path(cookies).is_file():
        args += ["--cookies", cookies]
    browser = os.environ.get("YTDLP_COOKIES_FROM_BROWSER", "").strip()
    if browser:
        args += ["--cookies-from-browser", browser]
    proxy = os.environ.get("YTDLP_PROXY", "").strip()
    if proxy:
        args += ["--proxy", proxy]
    return args


def _explain_ytdlp_error(stderr: str) -> str:
    text = (stderr or "").strip()
    if "Sign in to confirm" in text or "not a bot" in text:
        return (
            "YouTube refused the download from this server's IP address "
            "(\"Sign in to confirm you're not a bot\"). Export cookies.txt from a "
            "browser signed in to YouTube and point YTDLP_COOKIES at it, or set "
            "YTDLP_PROXY to a residential proxy. Home and office connections "
            "normally work without either."
        )
    if "Private video" in text or "members-only" in text:
        return "That video is private or members-only, so it cannot be downloaded."
    if "Video unavailable" in text:
        return "YouTube says that video is unavailable."
    return f"yt-dlp could not read that URL: {text[-400:]}"


def download(url: str, work_dir: Path, max_height: int = 1080) -> Path:
    """Fetch the source video.

    Capped at 1080p on purpose: a Short is 1080x1920, so a 4K source costs
    bandwidth and disk for detail that the vertical crop throws away anyway.
    """
    work_dir.mkdir(parents=True, exist_ok=True)

    local = _local_path(url)
    if local is not None:
        return local

    out_template = str(work_dir / "source.%(ext)s")
    fmt = (
        f"bestvideo[height<={max_height}][ext=mp4]+bestaudio[ext=m4a]/"
        f"best[height<={max_height}][ext=mp4]/best[height<={max_height}]/best"
    )
    cmd = ["yt-dlp", "--no-warnings", "--no-playlist", "-f", fmt,
           "--merge-output-format", "mp4", "-o", out_template]
    cmd += _auth_args()
    proc = subprocess.run(cmd + [url], capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(_explain_ytdlp_error(proc.stderr or proc.stdout))
    files = sorted(work_dir.glob("source.*"))
    if not files:
        raise RuntimeError("download reported success but produced no file")
    return files[0]


# ---------------------------------------------------------------------------
# transcript -> sentences
# ---------------------------------------------------------------------------

def sentences_from_words(words: list[cap.Word], max_gap: float = 0.7) -> list[Sentence]:
    """Rebuild sentences from word timings.

    Whisper's own segments are cut on silence, not on grammar, so a segment
    boundary regularly lands mid clause.  Splitting on terminal punctuation and
    on long pauses gives boundaries a viewer would recognise as a full thought.
    """
    out: list[Sentence] = []
    buffer: list[cap.Word] = []
    for index, word in enumerate(words):
        buffer.append(word)
        ends = bool(re.search(r"[.!?]\"?$", word.text))
        gap = (words[index + 1].start - word.end) if index + 1 < len(words) else 99.0
        if ends or gap > max_gap:
            text = " ".join(w.text for w in buffer).strip()
            if text:
                out.append(Sentence(text=text, start=buffer[0].start, end=buffer[-1].end))
            buffer = []
    if buffer:
        out.append(Sentence(
            text=" ".join(w.text for w in buffer).strip(),
            start=buffer[0].start, end=buffer[-1].end,
        ))
    return [s for s in out if not FILLER_ONLY.match(s.text)]


def score_candidate(text: str, duration: float, target: float) -> tuple[float, list[str]]:
    reasons: list[str] = []
    score = 0.0
    low = text.lower()
    opening = " ".join(low.split()[:18])

    hooks = sum(1 for pattern in HOOK_PATTERNS if re.search(pattern, opening))
    if hooks:
        score += 2.2 * min(hooks, 2)
        reasons.append("opens on a hook phrase")

    if re.search(r"\?", " ".join(text.split()[:20])):
        score += 1.0
        reasons.append("opens with a question")

    numbers = len(re.findall(NUMBER_WORDS, low))
    if numbers:
        score += min(1.2, 0.4 * numbers)
        reasons.append("contains concrete numbers")

    # Self-containment: ends on a full stop, does not open on a conjunction.
    if re.search(r"[.!?]\"?$", text.strip()):
        score += 1.0
        reasons.append("ends on a complete sentence")
    if re.match(r"^(and|but|so|because|which|that|then|also)\b", low):
        score -= 1.4
        reasons.append("starts mid-thought")

    words_count = len(text.split())
    wps = words_count / max(1.0, duration)
    if 2.0 <= wps <= 3.6:
        score += 1.0
        reasons.append("natural speaking pace")
    elif wps < 1.2:
        score -= 1.2
        reasons.append("long silences")

    # Length: a bell curve round the requested target, hard floor and ceiling.
    score += 2.0 * math.exp(-((duration - target) ** 2) / (2 * (target * 0.45) ** 2))
    if duration < 12 or duration > 90:
        score -= 3.0

    if re.search(r"\b(subscribe|like and subscribe|link in (the )?description|sponsor)\b", low):
        score -= 1.0
        reasons.append("contains channel housekeeping")

    return score, reasons


def pick_segments(
    sentences: list[Sentence],
    count: int = 3,
    target: float = 40.0,
    min_seconds: float = 18.0,
    max_seconds: float = 60.0,
) -> list[Candidate]:
    """Score every sentence-aligned window, then take the best non-overlapping ones."""
    candidates: list[Candidate] = []
    for start_index in range(len(sentences)):
        for end_index in range(start_index, len(sentences)):
            start = sentences[start_index].start
            end = sentences[end_index].end
            duration = end - start
            if duration < min_seconds:
                continue
            if duration > max_seconds:
                break
            text = " ".join(s.text for s in sentences[start_index : end_index + 1])
            score, reasons = score_candidate(text, duration, target)
            candidates.append(Candidate(start=start, end=end, text=text,
                                        score=score, reasons=reasons))

    candidates.sort(key=lambda c: c.score, reverse=True)
    chosen: list[Candidate] = []
    for candidate in candidates:
        # Greedy, but reject anything that overlaps an already chosen window:
        # three cuts of the same 40 seconds is three copies of one Short.
        if any(candidate.start < c.end and c.start < candidate.end for c in chosen):
            continue
        chosen.append(candidate)
        if len(chosen) >= count:
            break
    chosen.sort(key=lambda c: c.start)
    return chosen


RERANK_PROMPT = """You are picking the clips most likely to perform as YouTube Shorts.

Below are {n} candidate clips taken from one video. For each, decide how well it
works as a standalone Short for a viewer who has no context.

Return JSON: {{"clips": [{{"index": 0, "score": 0-10, "title": "under 70 chars",
"reason": "one short sentence"}}]}}

Candidates:
{body}"""


def rerank_with_model(candidates: list[Candidate]) -> list[Candidate]:
    """Optional second opinion.  Failure here is not an error: keep the heuristic order."""
    if not candidates or settings.llm_provider not in {"ollama", "openai"}:
        return candidates
    body = "\n\n".join(
        f"[{i}] ({c.duration:.0f}s) {' '.join(c.text.split())[:700]}"
        for i, c in enumerate(candidates)
    )
    prompt = RERANK_PROMPT.format(n=len(candidates), body=body)
    try:
        raw = (_call_ollama(prompt, 300) if settings.llm_provider == "ollama"
               else _call_openai(prompt, 150))
        payload = _extract_json(raw) or {}
        for item in payload.get("clips", []):
            index = int(item.get("index", -1))
            if 0 <= index < len(candidates):
                candidates[index].score = float(item.get("score", candidates[index].score))
                title = str(item.get("title", "")).strip()
                if title:
                    candidates[index].title = title[:95]
                reason = str(item.get("reason", "")).strip()
                if reason:
                    candidates[index].reasons.insert(0, reason)
        candidates.sort(key=lambda c: c.score, reverse=True)
    except Exception as exc:  # noqa: BLE001
        log.warning("clip re-rank skipped: %s", exc)
    return candidates


# ---------------------------------------------------------------------------
# cutting
# ---------------------------------------------------------------------------

def _reframe_filter(mode: str) -> str:
    """Build the 16:9 -> 9:16 filter.

    `blur` keeps the whole frame and fills the bar with a blown-up blurred copy;
    nothing is lost, which is the safe default for talking-head and gameplay.
    `crop` fills the screen from the centre, which looks better when the subject
    is centred but will cut the sides off a wide shot.
    """
    if mode == "crop":
        return (
            f"scale={W}:-2:flags=lanczos,crop={W}:{H}:(iw-{W})/2:(ih-{H})/2,"
            f"setsar=1,format=yuv420p"
        )
    return (
        f"[0:v]split=2[bg][fg];"
        f"[bg]scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},"
        f"gblur=sigma=42,eq=brightness=-0.09:saturation=1.25[bgb];"
        f"[fg]scale={W}:-2:flags=lanczos[fgs];"
        f"[bgb][fgs]overlay=(W-w)/2:(H-h)/2,setsar=1,format=yuv420p"
    )


def cut_clip(
    source: Path,
    out_path: Path,
    start: float,
    end: float,
    reframe: str = "blur",
    ass_path: Path | None = None,
    crf: int = 20,
) -> Path:
    duration = max(1.0, end - start)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    chain = _reframe_filter(reframe)
    burn = ""
    if ass_path:
        escaped = str(ass_path).replace("\\", "/").replace(":", r"\:")
        fonts = str(settings.fonts_dir).replace("\\", "/").replace(":", r"\:")
        burn = f",ass='{escaped}':fontsdir='{fonts}'"

    if reframe == "crop":
        video_args = ["-vf", chain + burn]
    else:
        # The blur path is a graph with named pads, so it has to go through
        # filter_complex and be mapped explicitly.
        video_args = ["-filter_complex", chain + burn + "[vout]", "-map", "[vout]", "-map", "0:a?"]

    cmd = [
        settings.ffmpeg, "-y", "-loglevel", "error",
        # -ss before -i seeks fast; -copyts keeps the subtitle clock aligned to
        # the clip rather than to the source, which is why the ASS file is
        # written with clip-relative times.
        "-ss", f"{start:.3f}", "-t", f"{duration:.3f}", "-i", str(source),
        *video_args,
        "-c:v", "libx264", "-preset", "medium", "-crf", str(crf),
        "-pix_fmt", "yuv420p", "-profile:v", "high", "-level", "4.1",
        "-r", str(settings.video_fps), "-g", str(settings.video_fps * 2),
        "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2",
        "-af", "loudnorm=I=-15:TP=-1.5:LRA=11",
        "-movflags", "+faststart", str(out_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout).strip().splitlines()[-10:]
        raise RuntimeError("clip cut failed:\n" + "\n".join(tail))
    return out_path


def words_in_window(words: list[cap.Word], start: float, end: float) -> list[cap.Word]:
    """Words inside a window, rebased so t=0 is the start of the clip."""
    out: list[cap.Word] = []
    for word in words:
        if word.end <= start or word.start >= end:
            continue
        out.append(cap.Word(
            text=word.text,
            start=max(0.0, word.start - start),
            end=min(end - start, word.end - start),
        ))
    return out
