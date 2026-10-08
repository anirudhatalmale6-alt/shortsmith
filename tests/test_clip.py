"""Clip selection: sentence rebuilding, scoring, non-overlap, window rebasing."""

from __future__ import annotations

import pytest

from shortsmith.pipeline import clip as C
from shortsmith.providers.captions import Word


def words_from(text: str, start: float = 0.0, rate: float = 0.35) -> list[Word]:
    out, cursor = [], start
    for token in text.split():
        out.append(Word(text=token, start=cursor, end=cursor + rate * 0.9))
        cursor += rate
    return out


def test_sentences_split_on_terminal_punctuation() -> None:
    sentences = C.sentences_from_words(words_from("One two. Three four. Five six."))
    assert len(sentences) == 3
    assert sentences[0].text == "One two."


def test_sentences_split_on_a_long_pause() -> None:
    first = words_from("one two three")
    second = words_from("four five six", start=first[-1].end + 3.0)
    sentences = C.sentences_from_words(first + second)
    assert len(sentences) == 2


def test_sentences_drop_filler_only_fragments() -> None:
    sentences = C.sentences_from_words(words_from("um. so. A real sentence here."))
    assert [s.text for s in sentences] == ["A real sentence here."]


def test_sentence_times_come_from_the_words() -> None:
    sentences = C.sentences_from_words(words_from("One two. Three four."))
    assert sentences[0].start == 0.0
    assert sentences[1].start > sentences[0].end - 0.01


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------

def test_a_hook_opening_beats_a_flat_one() -> None:
    hooked, _ = C.score_candidate(
        "Here's why nobody tells you this. It changed everything for me.", 40, 40)
    flat, _ = C.score_candidate(
        "The weather was fine. We walked to the shop and then we went home.", 40, 40)
    assert hooked > flat


def test_starting_mid_thought_is_penalised() -> None:
    clean, _ = C.score_candidate("They never found the car. Nobody knows why.", 40, 40)
    mid, _ = C.score_candidate("and they never found the car. Nobody knows why.", 40, 40)
    assert clean > mid


def test_ending_mid_sentence_is_penalised() -> None:
    complete, _ = C.score_candidate("They never found the car.", 40, 40)
    cut, _ = C.score_candidate("They never found the car and then", 40, 40)
    assert complete > cut


def test_length_near_the_target_scores_higher() -> None:
    text = "Here's why this matters. " * 8
    near, _ = C.score_candidate(text, 40, 40)
    far, _ = C.score_candidate(text, 85, 40)
    assert near > far


def test_absurd_lengths_are_rejected_outright() -> None:
    text = "Here's why this matters. " * 8
    ok, _ = C.score_candidate(text, 40, 40)
    tiny, _ = C.score_candidate(text, 8, 40)
    assert tiny < ok - 2


def test_channel_housekeeping_is_penalised() -> None:
    plain, _ = C.score_candidate("Here's why this matters so much to me.", 40, 40)
    spam, _ = C.score_candidate(
        "Here's why this matters so much to me. Link in the description.", 40, 40)
    assert plain > spam


def test_long_silences_are_penalised() -> None:
    dense, _ = C.score_candidate("one two three four five six " * 10, 40, 40)
    sparse, _ = C.score_candidate("one two three", 40, 40)
    assert dense > sparse


# ---------------------------------------------------------------------------
# selection
# ---------------------------------------------------------------------------

def _long_transcript() -> list[C.Sentence]:
    sentences, cursor = [], 0.0
    for index in range(40):
        duration = 6.0
        sentences.append(C.Sentence(
            text=f"Here's why point number {index} actually matters to you.",
            start=cursor, end=cursor + duration,
        ))
        cursor += duration
    return sentences


def test_picked_segments_never_overlap() -> None:
    picks = C.pick_segments(_long_transcript(), count=4, target=40)
    assert len(picks) == 4
    for earlier, later in zip(picks, picks[1:]):
        assert earlier.end <= later.start


def test_picked_segments_respect_the_length_bounds() -> None:
    picks = C.pick_segments(_long_transcript(), count=3, target=40,
                            min_seconds=18, max_seconds=60)
    assert picks
    for pick in picks:
        assert 18 <= pick.duration <= 60


def test_picks_come_back_in_source_order() -> None:
    picks = C.pick_segments(_long_transcript(), count=3, target=40)
    assert picks == sorted(picks, key=lambda c: c.start)


def test_a_transcript_too_short_to_cut_returns_nothing() -> None:
    short = [C.Sentence(text="Only this.", start=0, end=4)]
    assert C.pick_segments(short, count=3, target=40, min_seconds=18) == []


# ---------------------------------------------------------------------------
# window rebasing
# ---------------------------------------------------------------------------

def test_words_in_window_are_rebased_to_the_clip() -> None:
    source = [Word(text=f"w{i}", start=i, end=i + 0.8) for i in range(20)]
    inside = C.words_in_window(source, 5.0, 9.0)
    assert inside[0].start == pytest.approx(0.0)
    assert all(0 <= w.start <= 4.0 for w in inside)
    assert all(w.end <= 4.0 for w in inside)


def test_words_in_window_excludes_words_outside_it() -> None:
    source = [Word(text=f"w{i}", start=i, end=i + 0.8) for i in range(20)]
    inside = C.words_in_window(source, 5.0, 9.0)
    assert [w.text for w in inside] == ["w5", "w6", "w7", "w8"]


# ---------------------------------------------------------------------------
# source resolution and error translation
# ---------------------------------------------------------------------------

def test_local_path_accepts_a_real_file(tmp_path) -> None:
    target = tmp_path / "video.mp4"
    target.write_bytes(b"not really a video")
    assert C._local_path(str(target)) == target
    assert C._local_path(f"file://{target}") == target


def test_local_path_rejects_a_url_and_a_missing_file(tmp_path) -> None:
    assert C._local_path("https://youtube.com/watch?v=x") is None
    assert C._local_path(str(tmp_path / "nope.mp4")) is None


def test_bot_check_error_is_translated_into_something_actionable() -> None:
    message = C._explain_ytdlp_error(
        "ERROR: [youtube] abc: Sign in to confirm you're not a bot.")
    assert "YTDLP_COOKIES" in message
    assert "cookies.txt" in message


def test_unknown_errors_keep_the_original_text() -> None:
    assert "kaboom" in C._explain_ytdlp_error("ERROR: kaboom")


def test_reframe_filters_target_the_configured_resolution() -> None:
    assert "1080:1920" in C._reframe_filter("crop") or "1080" in C._reframe_filter("crop")
    blur = C._reframe_filter("blur")
    assert "gblur" in blur and "overlay" in blur


# ---------------------------------------------------------------------------
# titles
# ---------------------------------------------------------------------------

def test_a_title_comes_from_the_clips_own_opening_line() -> None:
    title = C.heuristic_title("Here's why nobody tells you this. It changed everything.")
    assert title.startswith("Here's why nobody tells you this")


def test_a_title_drops_leading_filler() -> None:
    assert not C.heuristic_title("So, here's the thing about compound interest.").startswith("So")


def test_a_title_strips_verbal_tics() -> None:
    title = C.heuristic_title("This is, um, the part that actually matters a lot.")
    assert "um" not in title.lower().split()


def test_a_long_title_is_cut_on_a_word_boundary() -> None:
    long_line = "This is the single most important thing that anybody ever told me about money and it changed everything."
    title = C.heuristic_title(long_line, limit=40)
    assert len(title) <= 40
    assert not title.endswith(" ")
    assert long_line.startswith(title[:10])


def test_a_title_never_ends_on_a_dangling_preposition() -> None:
    title = C.heuristic_title(
        "The thing that nobody tells you about the real cost of a mortgage is this.", limit=46)
    assert title.split()[-1].lower() not in {"of", "the", "a", "to", "and", "about"}


def test_too_short_to_be_a_title_returns_nothing() -> None:
    assert C.heuristic_title("Yeah.") == ""
    assert C.heuristic_title("") == ""


def test_a_title_is_capitalised() -> None:
    assert C.heuristic_title("here is why that matters so much").startswith("H")
