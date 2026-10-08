"""Script writing.

Three interchangeable back-ends, picked with LLM_PROVIDER:

  ollama   - a local model served by Ollama.  Free, offline, no per-video cost.
             This is the default and the one the project is designed around.
  openai   - any OpenAI-compatible chat endpoint (OpenAI, Groq, OpenRouter,
             LM Studio, vLLM, llama.cpp server...).  Set OPENAI_BASE_URL.
  offline  - a built-in beat-based story writer that needs no model at all.

`write_script` never raises because of a provider problem: if the configured
model is unreachable or answers with something that is not usable JSON, it
falls back to the offline writer and records why in the returned payload.  A
missing model must never cost the user a scheduled upload slot.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

import httpx

from ..config import settings
from .offline_writer import offline_script
from .script_types import Scene, ScriptResult

log = logging.getLogger(__name__)

MIN_SCENES = 4
MAX_SCENES = 9

__all__ = ["Scene", "ScriptResult", "write_script", "offline_script", "provider_status"]


# ---------------------------------------------------------------------------
# prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a writer for viral YouTube Shorts narration.
You write spoken narration only: no scene headings, no camera directions, no
speaker labels, no stage notes, no emoji, no markdown.
Every line must be natural when read aloud by a text-to-speech voice.
You always answer with a single JSON object and nothing else."""

USER_PROMPT = """Write a {duration}-second vertical YouTube Short about: {topic}

Style: {style}
Tone: {tone}

Rules:
- The hook is one sentence, under 14 words, and must create an open loop that
  makes a viewer stop scrolling in the first second.
- Then {scene_count} short narration beats. Each beat is 1 to 2 sentences,
  under 30 words, and advances the story. No beat may restate another.
- The second to last beat must contain the turn: the thing the viewer did not
  see coming.
- The closing line is one short sentence. Do not say "subscribe" more than once
  and never say "in this video".
- Write the way a person speaks. Contractions are good. Avoid the words
  "delve", "tapestry", "testament", "realm", "unravel", "embark".
- For each beat, also write an image prompt describing one still frame that
  illustrates it: subject, setting, lighting, mood, lens. No text in the image,
  no words, no letters, no watermark. Keep characters visually consistent
  between beats by repeating their description.

Return exactly this JSON shape:
{{
  "title": "YouTube title under 70 characters, no quotes",
  "hook": "...",
  "scenes": [{{"narration": "...", "visual_prompt": "..."}}],
  "cta": "...",
  "description": "2 sentence YouTube description",
  "hashtags": ["#shorts", "..."]
}}"""


def _build_user_prompt(topic: str, duration: int, style: str, tone: str) -> str:
    scene_count = max(MIN_SCENES, min(MAX_SCENES, round(duration / 7)))
    return USER_PROMPT.format(
        topic=topic.strip(),
        duration=duration,
        style=style or "cinematic storytelling",
        tone=tone or "emotional",
        scene_count=scene_count,
    )


# ---------------------------------------------------------------------------
# JSON recovery
# ---------------------------------------------------------------------------

def _extract_json(text: str) -> dict[str, Any] | None:
    """Pull the first balanced JSON object out of a model reply.

    Small local models wrap JSON in prose or fenced blocks, and sometimes emit a
    trailing comma.  A plain json.loads on the whole reply throws away a usable
    answer, so scan for the outermost braces and repair the common faults.
    """
    if not text:
        return None
    fenced = re.search(r"```(?:json)?\s*(.+?)```", text, re.S)
    if fenced:
        text = fenced.group(1)

    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(text)):
        char = text[idx]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                blob = text[start : idx + 1]
                for candidate in (blob, re.sub(r",\s*([}\]])", r"\1", blob)):
                    try:
                        parsed = json.loads(candidate)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(parsed, dict):
                        return parsed
                return None
    return None


_BAD_LINE = re.compile(
    r"^\s*(scene|beat|shot|cut|narrator|voice\s*over|vo|v\.o\.|visual|image|caption)\s*\d*\s*[:\-]",
    re.I,
)


def _clean_line(text: Any) -> str:
    if not isinstance(text, str):
        return ""
    line = text.strip()
    line = _BAD_LINE.sub("", line).strip()
    line = re.sub(r"\s+", " ", line)
    line = line.strip("*_` ")
    # TTS reads these out loud or stumbles on them.
    line = line.replace("—", ", ").replace("–", ", ").replace("…", "...")
    line = re.sub(r"[‘’]", "'", line)
    line = re.sub(r"[“”]", '"', line)
    # Strip emoji and symbols the voice cannot pronounce.
    line = re.sub(r"[^\x00-\x7F]+", "", line)
    return line.strip()


def _coerce(payload: dict[str, Any], topic: str, provider: str) -> ScriptResult | None:
    raw_scenes = payload.get("scenes") or payload.get("beats") or []
    scenes: list[Scene] = []
    for item in raw_scenes:
        if isinstance(item, str):
            narration, visual = item, ""
        elif isinstance(item, dict):
            narration = item.get("narration") or item.get("text") or item.get("line") or ""
            visual = (
                item.get("visual_prompt")
                or item.get("image_prompt")
                or item.get("visual")
                or item.get("image")
                or ""
            )
        else:
            continue
        narration = _clean_line(narration)
        if not narration:
            continue
        visual = _clean_line(visual) or f"cinematic still illustrating: {narration}"
        scenes.append(Scene(narration=narration, visual_prompt=visual))

    if len(scenes) < 2:
        return None

    hook = _clean_line(payload.get("hook") or payload.get("opening") or "")
    if not hook:
        # Some models fold the hook into the first beat; promote it.
        hook = scenes[0].narration
        scenes = scenes[1:]
        if len(scenes) < 2:
            return None

    title = _clean_line(payload.get("title")) or topic.strip().title()
    cta = _clean_line(payload.get("cta") or payload.get("closing") or "")
    description = _clean_line(payload.get("description")) or title

    hashtags: list[str] = []
    for tag in payload.get("hashtags") or []:
        tag = _clean_line(tag).replace(" ", "")
        if not tag:
            continue
        if not tag.startswith("#"):
            tag = "#" + tag
        if tag.lower() not in {h.lower() for h in hashtags}:
            hashtags.append(tag)
    if not any(h.lower() == "#shorts" for h in hashtags):
        hashtags.insert(0, "#shorts")

    return ScriptResult(
        title=title[:95],
        hook=hook,
        scenes=scenes[:MAX_SCENES],
        cta=cta,
        description=description,
        hashtags=hashtags[:12],
        provider=provider,
    )


# ---------------------------------------------------------------------------
# back-ends
# ---------------------------------------------------------------------------

def _call_ollama(prompt: str, timeout: float) -> str:
    response = httpx.post(
        f"{settings.ollama_url}/api/chat",
        json={
            "model": settings.ollama_model,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.9, "top_p": 0.95},
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        },
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json().get("message", {}).get("content", "")


def _call_openai(prompt: str, timeout: float) -> str:
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    response = httpx.post(
        f"{settings.openai_base_url}/chat/completions",
        headers={"Authorization": f"Bearer {settings.openai_api_key}"},
        json={
            "model": settings.openai_model,
            "temperature": 0.9,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        },
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def provider_status() -> dict[str, Any]:
    """Cheap reachability check used by the dashboard."""
    provider = settings.llm_provider
    if provider == "ollama":
        try:
            response = httpx.get(f"{settings.ollama_url}/api/tags", timeout=4)
            response.raise_for_status()
            names = [m.get("name", "") for m in response.json().get("models", [])]
            if not any(n.split(":")[0] == settings.ollama_model.split(":")[0] for n in names):
                return {
                    "ok": False,
                    "provider": "ollama",
                    "detail": f"Ollama is running but '{settings.ollama_model}' is not pulled",
                }
            return {"ok": True, "provider": "ollama", "detail": settings.ollama_model}
        except Exception as exc:  # noqa: BLE001 - status panel, never fatal
            return {"ok": False, "provider": "ollama", "detail": str(exc)[:160]}
    if provider == "openai":
        if not settings.openai_api_key:
            return {"ok": False, "provider": "openai", "detail": "OPENAI_API_KEY is not set"}
        return {"ok": True, "provider": "openai", "detail": settings.openai_model}
    return {"ok": True, "provider": "offline", "detail": "built-in template writer"}


def write_script(
    topic: str,
    duration: int = 45,
    style: str = "cinematic storytelling",
    tone: str = "emotional",
    seed: int | None = None,
) -> ScriptResult:
    prompt = _build_user_prompt(topic, duration, style, tone)
    provider = settings.llm_provider
    attempts = 2 if provider in {"ollama", "openai"} else 0
    last_error = ""

    for attempt in range(attempts):
        try:
            raw = _call_ollama(prompt, 240) if provider == "ollama" else _call_openai(prompt, 120)
            payload = _extract_json(raw)
            if payload is None:
                last_error = "model did not return usable JSON"
                continue
            result = _coerce(payload, topic, provider)
            if result is None:
                last_error = "model JSON was missing the story beats"
                continue
            return result
        except Exception as exc:  # noqa: BLE001 - fall through to offline
            last_error = f"{type(exc).__name__}: {exc}"[:200]
            log.warning("script provider %s attempt %s failed: %s", provider, attempt + 1, last_error)

    result = offline_script(topic, duration, tone, seed=seed)
    if last_error:
        result.notes = f"{provider} unavailable ({last_error}); used the offline writer"
    return result
