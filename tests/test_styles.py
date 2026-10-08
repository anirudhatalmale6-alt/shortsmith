"""Art direction and the look lock."""

from __future__ import annotations

import pytest

from shortsmith.providers import images, styles


def test_every_preset_has_a_label_suffix_and_negative() -> None:
    for key, spec in styles.STYLE_PRESETS.items():
        label, suffix, negative = spec
        assert label and suffix and negative, key
        assert "no text" in suffix, key


def test_the_negative_prompt_always_blocks_burned_in_text() -> None:
    for key in styles.STYLE_PRESETS:
        negative = styles.style_negative(key)
        assert "watermark" in negative and "text" in negative


def test_an_unknown_style_falls_back_rather_than_raising() -> None:
    assert styles.style_suffix("no-such-style") == styles.style_suffix(styles.DEFAULT_STYLE)


def test_the_character_sheet_comes_from_the_establishing_beat() -> None:
    sheet = styles.character_sheet([
        "a father alone in a parked car, soft grey window light",
        "a woman at a window",
    ])
    assert sheet.startswith("a father")
    assert "woman" not in sheet


def test_no_character_sheet_when_nobody_is_in_shot() -> None:
    assert styles.character_sheet(["close up of a letter on a table",
                                   "an empty kitchen at dawn"]) == ""


def test_the_sheet_is_injected_only_where_a_person_appears() -> None:
    sheet = "a father in a dark coat"
    with_person = styles.apply_look("a father walking down a street", "cinematic", sheet)
    without = styles.apply_look("close up of a letter on a table", "cinematic", sheet)
    assert "the same character throughout" in with_person
    assert "the same character throughout" not in without


def test_apply_look_puts_the_subject_first_and_the_style_last() -> None:
    out = styles.apply_look("a lighthouse at dusk", "noir")
    assert out.startswith("a lighthouse at dusk")
    assert out.endswith(styles.style_suffix("noir"))


def test_apply_look_does_not_repeat_itself() -> None:
    suffix = styles.style_suffix("cinematic")
    out = styles.apply_look(f"a street, {suffix}", "cinematic")
    assert out.count("anamorphic") == 1


def test_seed_base_is_stable_for_a_title_and_differs_between_titles() -> None:
    assert styles.seed_base("A Sad Story") == styles.seed_base("A Sad Story")
    assert styles.seed_base("A Sad Story") != styles.seed_base("A Sci-Fi Story")


def test_locking_makes_beat_seeds_deterministic_and_close_together() -> None:
    images.set_look(seed_base=1_000_000, negative="")
    try:
        seeds = [images._seed_for(f"beat {i}", i) for i in range(6)]
        assert seeds == [1_000_000 + i * 101 for i in range(6)]
        assert len(set(seeds)) == 6          # same family, not the same image
        assert max(seeds) - min(seeds) < 1000
    finally:
        images.set_look(0, "")


def test_without_the_lock_seeds_come_from_the_prompt() -> None:
    images.set_look(0, "")
    first = images._seed_for("a street at night", 0)
    again = images._seed_for("a street at night", 0)
    other = images._seed_for("a beach at noon", 0)
    assert first == again
    assert first != other
