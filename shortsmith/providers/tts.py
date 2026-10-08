"""Voice over.

  piper - Piper TTS.  MIT licensed, runs on CPU, no network, no per-character
          cost.  This is the default.  Voice models live in models/piper and
          are downloaded on first use from the official Piper voice repo.
  edge  - Microsoft Edge neural voices through edge-tts.  Free and noticeably
          more expressive, but it is a network call to a service you do not
          control, so it is opt-in rather than the default.

Both return a 48 kHz mono WAV, because that is what the renderer and the
caption aligner both want.
"""

from __future__ import annotations

import asyncio
import logging
import subprocess
import wave
from pathlib import Path

import httpx

from ..config import settings

log = logging.getLogger(__name__)

PIPER_REPO = "https://huggingface.co/rhasspy/piper-voices/resolve/main"

# name -> (repo path, human label, gender hint)
PIPER_VOICES: dict[str, tuple[str, str, str]] = {
    "en_US-amy-medium": ("en/en_US/amy/medium", "Amy (US, warm narration)", "female"),
    "en_US-hfc_female-medium": ("en/en_US/hfc_female/medium", "HFC Female (US, clear)", "female"),
    "en_US-ryan-high": ("en/en_US/ryan/high", "Ryan (US, deep male)", "male"),
    "en_GB-alba-medium": ("en/en_GB/alba/medium", "Alba (UK, soft Scottish)", "female"),
    "en_US-lessac-medium": ("en/en_US/lessac/medium", "Lessac (US, neutral)", "female"),
    "en_GB-northern_english_male-medium": (
        "en/en_GB/northern_english_male/medium", "Northern English Male", "male"),
}

EDGE_VOICES: dict[str, str] = {
    "en-US-AriaNeural": "Aria (US, expressive female)",
    "en-US-GuyNeural": "Guy (US, male)",
    "en-US-JennyNeural": "Jenny (US, friendly female)",
    "en-GB-SoniaNeural": "Sonia (UK, female)",
    "en-GB-RyanNeural": "Ryan (UK, male)",
    "en-AU-NatashaNeural": "Natasha (AU, female)",
}

_voice_cache: dict[str, object] = {}


def available_voices() -> list[dict[str, str]]:
    out = [
        {"id": key, "label": label, "engine": "piper", "gender": gender}
        for key, (_, label, gender) in PIPER_VOICES.items()
    ]
    out += [
        {"id": key, "label": label, "engine": "edge", "gender": ""}
        for key, label in EDGE_VOICES.items()
    ]
    return out


def ensure_piper_voice(name: str) -> Path:
    """Return the local .onnx path, downloading the voice if it is missing."""
    target_dir = settings.models_dir / "piper"
    target_dir.mkdir(parents=True, exist_ok=True)
    onnx = target_dir / f"{name}.onnx"
    config = target_dir / f"{name}.onnx.json"
    if onnx.exists() and config.exists() and onnx.stat().st_size > 1_000_000:
        return onnx

    if name not in PIPER_VOICES:
        raise RuntimeError(f"unknown Piper voice '{name}'")
    repo_path = PIPER_VOICES[name][0]
    for url, dest in (
        (f"{PIPER_REPO}/{repo_path}/{name}.onnx", onnx),
        (f"{PIPER_REPO}/{repo_path}/{name}.onnx.json", config),
    ):
        log.info("downloading piper voice %s", url)
        with httpx.stream("GET", url, timeout=600, follow_redirects=True) as response:
            response.raise_for_status()
            tmp = dest.with_suffix(dest.suffix + ".part")
            with tmp.open("wb") as handle:
                for chunk in response.iter_bytes(1 << 16):
                    handle.write(chunk)
            tmp.replace(dest)
    return onnx


def _load_piper(name: str):
    if name in _voice_cache:
        return _voice_cache[name]
    from piper import PiperVoice

    onnx = ensure_piper_voice(name)
    voice = PiperVoice.load(str(onnx), config_path=str(onnx) + ".json")
    _voice_cache[name] = voice
    return voice


def _resample_to_48k(src: Path, dest: Path) -> None:
    subprocess.run(
        [settings.ffmpeg, "-y", "-loglevel", "error", "-i", str(src),
         "-ac", "1", "-ar", "48000", "-c:a", "pcm_s16le", str(dest)],
        check=True,
    )


def _synth_piper(text: str, out_path: Path, voice_id: str, speed: float) -> None:
    from piper import SynthesisConfig

    voice = _load_piper(voice_id)
    raw = out_path.with_suffix(".raw.wav")
    # length_scale > 1 is slower.  The UI exposes speed, so invert it here.
    config = SynthesisConfig(length_scale=max(0.5, min(2.0, 1.0 / max(0.3, speed))),
                             noise_scale=0.667, noise_w_scale=0.8, normalize_audio=True)
    with wave.open(str(raw), "wb") as handle:
        voice.synthesize_wav(text, handle, syn_config=config)
    _resample_to_48k(raw, out_path)
    raw.unlink(missing_ok=True)


def _synth_edge(text: str, out_path: Path, voice_id: str, speed: float) -> None:
    import edge_tts

    percent = int(round((speed - 1.0) * 100))
    rate = f"{percent:+d}%"
    mp3 = out_path.with_suffix(".edge.mp3")

    async def run() -> None:
        communicate = edge_tts.Communicate(text, voice_id, rate=rate)
        await communicate.save(str(mp3))

    asyncio.run(run())
    _resample_to_48k(mp3, out_path)
    mp3.unlink(missing_ok=True)


def synthesize(text: str, out_path: Path, voice_id: str | None = None, speed: float = 1.0) -> Path:
    """Render narration to a 48 kHz mono WAV.

    Falls back from edge to piper if the network call fails, because a missing
    voice track means no video at all.
    """
    text = " ".join(text.split())
    if not text:
        raise ValueError("nothing to speak")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    voice_id = voice_id or (
        settings.edge_voice if settings.tts_provider == "edge" else settings.piper_voice
    )
    engine = "edge" if voice_id in EDGE_VOICES else "piper"

    if engine == "edge":
        try:
            _synth_edge(text, out_path, voice_id, speed)
            return out_path
        except Exception as exc:  # noqa: BLE001
            log.warning("edge-tts failed (%s); falling back to Piper", exc)
            voice_id = settings.piper_voice

    _synth_piper(text, out_path, voice_id, speed)
    return out_path


def wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as handle:
        return handle.getnframes() / float(handle.getframerate())


def provider_status() -> dict[str, object]:
    name = settings.tts_provider
    if name == "edge":
        return {"ok": True, "provider": "edge", "detail": settings.edge_voice}
    onnx = settings.models_dir / "piper" / f"{settings.piper_voice}.onnx"
    if onnx.exists():
        return {"ok": True, "provider": "piper", "detail": f"{settings.piper_voice} (local)"}
    return {
        "ok": True,
        "provider": "piper",
        "detail": f"{settings.piper_voice} (will download on first render)",
    }
