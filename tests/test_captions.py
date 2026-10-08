"""Caption building: chunking, timing, ASS output, SRT output."""

from __future__ import annotations

import re

import pytest

from shortsmith.providers import captions as cap


def words(*pairs) -> list[cap.Word]:
    return [cap.Word(text=t, start=s, end=e) for t, s, e in pairs]


def test_words_from_text_fills_the_whole_duration() -> None:
    result = cap.words_from_text("one two three four five", 10.0)
    assert len(result) == 5
    assert result[0].start == pytest.approx(0.0)
    assert result[-1].end == pytest.approx(10.0, abs=1e-6)
    # Monotonic and contiguous.
    for earlier, later in zip(result, result[1:]):
        assert earlier.end == pytest.approx(later.start)


def test_words_from_text_weights_by_length() -> None:
    result = cap.words_from_text("a extraordinarily", 10.0)
    assert (result[1].end - result[1].start) > (result[0].end - result[0].start)


def test_words_from_text_handles_empty() -> None:
    assert cap.words_from_text("   ", 10.0) == []


def test_chunking_breaks_on_sentence_end() -> None:
    chunks = cap._chunk(
        words(("Hello.", 0, .4), ("World", .4, .8), ("again", .8, 1.2)),
        per_line=5, max_gap=1.0,
    )
    assert [[w.text for w in c] for c in chunks] == [["Hello."], ["World", "again"]]


def test_chunking_breaks_on_a_pause() -> None:
    chunks = cap._chunk(
        words(("one", 0, .3), ("two", 2.0, 2.3)), per_line=5, max_gap=0.55,
    )
    assert len(chunks) == 2


def test_chunking_respects_words_per_line() -> None:
    chunks = cap._chunk(
        words(*[(f"w{i}", i * .3, i * .3 + .25) for i in range(9)]),
        per_line=3, max_gap=1.0,
    )
    assert all(len(c) <= 3 for c in chunks)


def test_build_ass_emits_one_event_per_word(tmp_path) -> None:
    source = words(("one", 0, .4), ("two", .4, .8), ("three", .8, 1.3))
    path = cap.build_ass(source, tmp_path / "c.ass", style="bold_yellow", words_per_line=3)
    assert path is not None
    text = path.read_text(encoding="utf-8")
    events = [line for line in text.splitlines() if line.startswith("Dialogue:")]
    assert len(events) == 3
    # Every event carries the whole line, with exactly one word highlighted.
    for event in events:
        assert event.count("\\c&H") == 1
        assert "ONE" in event and "TWO" in event and "THREE" in event


def test_build_ass_times_are_monotonic_and_non_empty(tmp_path) -> None:
    source = words(("a", 0, .3), ("b", .3, .7), ("c", .7, 1.0), ("d", 1.6, 2.0))
    path = cap.build_ass(source, tmp_path / "c.ass", words_per_line=2)
    assert path is not None
    stamps = re.findall(r"Dialogue: 0,([\d:.]+),([\d:.]+),", path.read_text(encoding="utf-8"))
    assert stamps

    def secs(value: str) -> float:
        hours, minutes, seconds = value.split(":")
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)

    for start, end in stamps:
        assert secs(end) > secs(start)


def test_build_ass_declares_the_video_resolution(tmp_path) -> None:
    path = cap.build_ass(words(("a", 0, .5), ("b", .5, 1)), tmp_path / "c.ass")
    assert path is not None
    text = path.read_text(encoding="utf-8")
    assert "PlayResX: 1080" in text and "PlayResY: 1920" in text


def test_build_ass_escapes_braces_that_would_become_override_tags(tmp_path) -> None:
    path = cap.build_ass(words(("{evil}", 0, .5), ("ok", .5, 1)), tmp_path / "c.ass")
    assert path is not None
    body = "\n".join(
        line.split(",,0,0,0,,", 1)[1]
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.startswith("Dialogue:")
    )
    assert "{EVIL}" not in body
    assert "(EVIL)" in body


def test_style_none_writes_nothing(tmp_path) -> None:
    assert cap.build_ass(words(("a", 0, 1)), tmp_path / "c.ass", style="none") is None
    assert cap.build_ass([], tmp_path / "c.ass") is None


def test_lowercase_option(tmp_path) -> None:
    path = cap.build_ass(words(("Hello", 0, .5), ("there", .5, 1)),
                         tmp_path / "c.ass", uppercase=False)
    assert path is not None
    assert "Hello" in path.read_text(encoding="utf-8")


def test_srt_format(tmp_path) -> None:
    path = cap.write_srt(words(("Hello", 0, .5), ("there.", .5, 1.25)), tmp_path / "c.srt")
    text = path.read_text(encoding="utf-8")
    assert text.startswith("1\n")
    assert "00:00:00,000 --> 00:00:01,250" in text
    assert "Hello there." in text
