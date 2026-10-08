"""The script writer: the offline engine, JSON recovery, and model-reply coercion."""

from __future__ import annotations

import re

import pytest

from shortsmith.providers import llm
from shortsmith.providers.offline_writer import PRONOUNS, detect_genre, offline_script

GENRES = ["Sad Story", "Sci-Fi Story", "Horror story", "Unsolved mystery",
          "Motivational discipline", "Wholesome kindness", "Karma revenge",
          "Did you know facts"]


@pytest.mark.parametrize("topic", GENRES)
def test_every_slot_is_filled(topic: str) -> None:
    """No template placeholder may survive into narration or an image prompt."""
    for seed in range(40):
        script = offline_script(topic, 45, seed=seed)
        blob = script.narration_text + " " + " ".join(s.visual_prompt for s in script.scenes)
        assert not re.findall(r"\{[a-z_]+\}", blob), f"{topic}/{seed}: {blob}"


@pytest.mark.parametrize("topic", GENRES)
def test_pronouns_do_not_contradict_the_character(topic: str) -> None:
    """One script must not call the same person he and she."""
    masculine = {"he", "him", "his", "himself"}
    feminine = {"she", "her", "hers", "herself"}
    for seed in range(60):
        script = offline_script(topic, 45, seed=seed)
        words = set(re.findall(r"\b[a-z]+\b", script.narration_text.lower()))
        # Some genres deliberately name a second character of the other gender
        # ("she paid for the man behind her"), so only the single-protagonist
        # genres can be asserted this way.
        if topic in {"Karma revenge", "Wholesome kindness"}:
            continue
        assert not (words & masculine and words & feminine), script.narration_text


def test_longer_targets_get_more_beats() -> None:
    short = offline_script("Sad Story", 30, seed=1)
    long = offline_script("Sad Story", 70, seed=1)
    assert len(short.scenes) <= len(long.scenes)
    assert len(short.scenes) >= 4


def test_same_seed_is_reproducible() -> None:
    a = offline_script("Sci-Fi Story", 45, seed=99)
    b = offline_script("Sci-Fi Story", 45, seed=99)
    assert a.narration_text == b.narration_text


def test_different_seeds_actually_differ() -> None:
    texts = {offline_script("Sad Story", 45, seed=s).narration_text for s in range(25)}
    assert len(texts) > 15, "the slot pools are not varying enough"


@pytest.mark.parametrize("topic,expected", [
    ("Sad Story", "sad"),
    ("A sci-fi story about a colony on Mars", "scifi"),
    ("creepy haunted basement at 3am", "horror"),
    ("unsolved mystery cold case", "mystery"),
    ("motivation and discipline", "motivational"),
    ("something with no keywords at all", "sad"),
])
def test_genre_detection(topic: str, expected: str) -> None:
    assert detect_genre(topic) == expected


def test_narration_text_includes_hook_and_cta() -> None:
    script = offline_script("Sad Story", 45, seed=5)
    assert script.narration_text.startswith(script.hook)
    assert script.narration_text.endswith(script.cta)


def test_pronoun_table_is_complete() -> None:
    for table in PRONOUNS.values():
        assert {"he", "him", "his", "hes", "himself"} <= set(table)


# ---------------------------------------------------------------------------
# JSON recovery
# ---------------------------------------------------------------------------

def test_extract_json_from_fenced_block() -> None:
    raw = 'Sure!\n```json\n{"title": "a", "scenes": []}\n```\nHope that helps.'
    assert llm._extract_json(raw) == {"title": "a", "scenes": []}


def test_extract_json_with_trailing_comma() -> None:
    raw = '{"title": "a", "scenes": [1, 2,], }'
    assert llm._extract_json(raw) == {"title": "a", "scenes": [1, 2]}


def test_extract_json_ignores_braces_inside_strings() -> None:
    raw = 'noise {"title": "a } b", "n": 1} tail'
    assert llm._extract_json(raw) == {"title": "a } b", "n": 1}


def test_extract_json_returns_none_when_there_is_none() -> None:
    assert llm._extract_json("no json at all") is None
    assert llm._extract_json("") is None


# ---------------------------------------------------------------------------
# coercion of a model reply
# ---------------------------------------------------------------------------

def _payload(**over):
    base = {
        "title": "A title",
        "hook": "A hook sentence.",
        "scenes": [
            {"narration": "Beat one.", "visual_prompt": "a room"},
            {"narration": "Beat two.", "visual_prompt": "a street"},
            {"narration": "Beat three.", "visual_prompt": "a window"},
        ],
        "cta": "Follow for more.",
        "description": "desc",
        "hashtags": ["shorts", "#sad"],
    }
    base.update(over)
    return base


def test_coerce_happy_path() -> None:
    result = llm._coerce(_payload(), "Sad Story", "ollama")
    assert result is not None
    assert result.hook == "A hook sentence."
    assert len(result.scenes) == 3
    assert result.hashtags[0] == "#shorts"
    assert all(tag.startswith("#") for tag in result.hashtags)


def test_coerce_promotes_first_beat_when_the_hook_is_missing() -> None:
    result = llm._coerce(_payload(hook=""), "Sad Story", "ollama")
    assert result is not None
    assert result.hook == "Beat one."
    assert len(result.scenes) == 2


def test_coerce_strips_scene_labels_and_emoji() -> None:
    payload = _payload(scenes=[
        {"narration": "Scene 1: She waited.", "visual_prompt": "x"},
        {"narration": "VO: He never came. 😢", "visual_prompt": "y"},
        {"narration": "Beat three.", "visual_prompt": "z"},
    ])
    result = llm._coerce(payload, "t", "ollama")
    assert result is not None
    assert result.scenes[0].narration == "She waited."
    assert result.scenes[1].narration == "He never came."


def test_coerce_rewrites_characters_tts_cannot_speak() -> None:
    payload = _payload(hook="He left — she stayed…")
    result = llm._coerce(payload, "t", "ollama")
    assert result is not None
    assert "—" not in result.hook and "…" not in result.hook
    assert "," in result.hook


def test_coerce_accepts_plain_string_scenes() -> None:
    payload = _payload(scenes=["One.", "Two.", "Three."])
    result = llm._coerce(payload, "t", "ollama")
    assert result is not None
    assert len(result.scenes) == 3
    assert all(scene.visual_prompt for scene in result.scenes)


def test_coerce_rejects_a_reply_with_no_story() -> None:
    assert llm._coerce({"title": "x", "scenes": []}, "t", "ollama") is None
    assert llm._coerce({"title": "x", "scenes": [{"narration": "only one"}]}, "t", "ollama") is None


def test_write_script_falls_back_when_the_model_is_unreachable(monkeypatch) -> None:
    """A dead model must cost a worse script, never a missed slot."""
    monkeypatch.setattr(llm.settings, "llm_provider", "ollama")
    monkeypatch.setattr(llm.settings, "ollama_url", "http://127.0.0.1:1")  # nothing listens
    script = llm.write_script("Sad Story", duration=40)
    assert script.provider == "offline"
    assert "unavailable" in script.notes
    assert len(script.scenes) >= 4
