"""Central configuration.

Everything is read from the environment (optionally via a .env file) so the
same code runs on a laptop and on a server without edits.  Every setting has a
working default, which means a fresh clone starts and renders a video with no
configuration at all.
"""

from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


def _load_dotenv() -> None:
    env_file = BASE_DIR / ".env"
    if not env_file.exists():
        return
    for raw in env_file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        # Real environment variables always win over the file.
        os.environ.setdefault(key, value)


_load_dotenv()


def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default).strip()


def _env_int(key: str, default: int) -> int:
    try:
        return int(_env(key, str(default)))
    except ValueError:
        return default


def _env_bool(key: str, default: bool = False) -> bool:
    val = _env(key, "1" if default else "0").lower()
    return val in {"1", "true", "yes", "on"}


class Settings:
    # --- paths -----------------------------------------------------------
    base_dir: Path = BASE_DIR
    data_dir: Path = Path(_env("DATA_DIR", str(BASE_DIR / "data")))
    models_dir: Path = Path(_env("MODELS_DIR", str(BASE_DIR / "models")))
    assets_dir: Path = Path(_env("ASSETS_DIR", str(BASE_DIR / "assets")))

    # --- web -------------------------------------------------------------
    host: str = _env("HOST", "127.0.0.1")
    port: int = _env_int("PORT", 8080)
    # Must match the redirect URI registered in Google Cloud Console.
    public_base_url: str = _env("PUBLIC_BASE_URL", "http://127.0.0.1:8080").rstrip("/")
    secret_key: str = _env("SECRET_KEY", "change-me-in-production")
    app_password: str = _env("APP_PASSWORD", "")  # empty = no login wall

    # --- script writing --------------------------------------------------
    # ollama | openai | offline        (offline always works, zero setup)
    llm_provider: str = _env("LLM_PROVIDER", "ollama")
    ollama_url: str = _env("OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")
    ollama_model: str = _env("OLLAMA_MODEL", "llama3.2:3b")
    # Any OpenAI-compatible endpoint: OpenAI, Groq, OpenRouter, LM Studio, vLLM.
    openai_base_url: str = _env("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    openai_api_key: str = _env("OPENAI_API_KEY", "")
    openai_model: str = _env("OPENAI_MODEL", "gpt-4o-mini")

    # --- visuals ---------------------------------------------------------
    # pollinations | sdwebui | comfyui | pexels | gradient
    image_provider: str = _env("IMAGE_PROVIDER", "pollinations")
    sdwebui_url: str = _env("SDWEBUI_URL", "http://127.0.0.1:7860").rstrip("/")
    sdwebui_model: str = _env("SDWEBUI_MODEL", "")
    comfyui_url: str = _env("COMFYUI_URL", "http://127.0.0.1:8188").rstrip("/")
    pexels_api_key: str = _env("PEXELS_API_KEY", "")

    # --- voice -----------------------------------------------------------
    tts_provider: str = _env("TTS_PROVIDER", "piper")  # piper | edge
    piper_voice: str = _env("PIPER_VOICE", "en_US-amy-medium")
    edge_voice: str = _env("EDGE_VOICE", "en-US-AriaNeural")

    # --- captions --------------------------------------------------------
    whisper_model: str = _env("WHISPER_MODEL", "base.en")
    whisper_device: str = _env("WHISPER_DEVICE", "cpu")
    whisper_compute: str = _env("WHISPER_COMPUTE", "int8")

    # --- video -----------------------------------------------------------
    video_width: int = _env_int("VIDEO_WIDTH", 1080)
    video_height: int = _env_int("VIDEO_HEIGHT", 1920)
    video_fps: int = _env_int("VIDEO_FPS", 30)
    ffmpeg: str = _env("FFMPEG", "ffmpeg")
    ffprobe: str = _env("FFPROBE", "ffprobe")
    render_threads: int = _env_int("RENDER_THREADS", 0)  # 0 = let ffmpeg decide

    # --- worker ----------------------------------------------------------
    worker_poll_seconds: int = _env_int("WORKER_POLL_SECONDS", 15)
    worker_enabled: bool = _env_bool("WORKER_ENABLED", True)
    max_clip_minutes: int = _env_int("MAX_CLIP_MINUTES", 90)

    # --- youtube ---------------------------------------------------------
    google_client_id: str = _env("GOOGLE_CLIENT_ID", "")
    google_client_secret: str = _env("GOOGLE_CLIENT_SECRET", "")

    @property
    def database_url(self) -> str:
        return _env("DATABASE_URL", f"sqlite:///{self.data_dir / 'shortsmith.db'}")

    @property
    def media_dir(self) -> Path:
        return self.data_dir / "media"

    @property
    def work_dir(self) -> Path:
        return self.data_dir / "work"

    @property
    def music_dir(self) -> Path:
        return self.assets_dir / "music"

    @property
    def fonts_dir(self) -> Path:
        return self.assets_dir / "fonts"

    @property
    def redirect_uri(self) -> str:
        return f"{self.public_base_url}/youtube/callback"

    def ensure_dirs(self) -> None:
        for path in (
            self.data_dir,
            self.media_dir,
            self.work_dir,
            self.models_dir / "piper",
            self.music_dir,
            self.fonts_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_dirs()
