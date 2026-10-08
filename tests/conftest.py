"""Test setup.

Every test runs against a throwaway data directory so a test run can never
touch a real queue, a real media file or a real token.  The environment is set
before `shortsmith.config` is imported, because settings are read at import.
"""

import os
import tempfile
from pathlib import Path

_TMP = Path(tempfile.mkdtemp(prefix="shortsmith-tests-"))
os.environ.setdefault("DATA_DIR", str(_TMP / "data"))
os.environ.setdefault("DATABASE_URL", f"sqlite:///{_TMP / 'test.db'}")
os.environ.setdefault("WORKER_ENABLED", "0")
os.environ.setdefault("LLM_PROVIDER", "offline")
os.environ.setdefault("IMAGE_PROVIDER", "gradient")
os.environ.setdefault("SECRET_KEY", "test-secret")
os.environ.setdefault("APP_PASSWORD", "")
# Forced, not setdefault: a developer machine may well have Google credentials
# exported for something else, and a test run must never pick them up and talk
# to a real account.
os.environ["GOOGLE_CLIENT_ID"] = ""
os.environ["GOOGLE_CLIENT_SECRET"] = ""
os.environ["OPENAI_API_KEY"] = ""
os.environ["PEXELS_API_KEY"] = ""

# Voice and Whisper models are large; reuse the checkout's copies rather than
# downloading them again into the temporary directory.
_REPO = Path(__file__).resolve().parent.parent
os.environ.setdefault("MODELS_DIR", str(_REPO / "models"))
os.environ.setdefault("ASSETS_DIR", str(_REPO / "assets"))
