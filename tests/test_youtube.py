"""The YouTube client: request shaping, token refresh, error translation.

No network. The Google API client is replaced with a recorder so the body that
would be sent is asserted directly, which is the only part of an upload this
project controls.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from shortsmith.youtube import client as yt


class _Insert:
    def __init__(self, recorder, **kwargs):
        recorder["insert"] = kwargs
        self._done = False

    def next_chunk(self):
        self._done = True
        return SimpleNamespace(progress=lambda: 1.0), {
            "id": "VIDEO123", "status": {"privacyStatus": "private"}
        }


class _Videos:
    def __init__(self, recorder):
        self._recorder = recorder

    def insert(self, **kwargs):
        return _Insert(self._recorder, **kwargs)


class _Service:
    def __init__(self, recorder):
        self._recorder = recorder

    def videos(self):
        return _Videos(self._recorder)

    def thumbnails(self):
        raise AssertionError("no thumbnail in these tests")


@pytest.fixture
def recorder(monkeypatch):
    captured: dict = {}
    monkeypatch.setattr(yt, "service_for", lambda creds: _Service(captured))
    monkeypatch.setattr(yt, "MediaFileUpload", lambda *a, **k: object())
    return captured


def test_a_scheduled_upload_is_forced_private(recorder, tmp_path) -> None:
    """publishAt is ignored by YouTube unless privacyStatus is private."""
    slot = datetime(2030, 5, 1, 12, 0, tzinfo=timezone.utc)
    yt.upload_video(object(), tmp_path / "v.mp4", "T", "D", ["a"],
                    privacy="public", publish_at=slot)
    status = recorder["insert"]["body"]["status"]
    assert status["privacyStatus"] == "private"
    assert status["publishAt"] == "2030-05-01T12:00:00Z"


def test_a_naive_slot_is_treated_as_utc(recorder, tmp_path) -> None:
    yt.upload_video(object(), tmp_path / "v.mp4", "T", "D", [],
                    publish_at=datetime(2030, 5, 1, 12, 0))
    assert recorder["insert"]["body"]["status"]["publishAt"] == "2030-05-01T12:00:00Z"


def test_an_immediate_upload_keeps_the_requested_privacy(recorder, tmp_path) -> None:
    yt.upload_video(object(), tmp_path / "v.mp4", "T", "D", [], privacy="unlisted")
    status = recorder["insert"]["body"]["status"]
    assert status["privacyStatus"] == "unlisted"
    assert "publishAt" not in status


def test_title_and_description_are_clamped_to_youtube_limits(recorder, tmp_path) -> None:
    yt.upload_video(object(), tmp_path / "v.mp4", "x" * 300, "y" * 6000, [])
    snippet = recorder["insert"]["body"]["snippet"]
    assert len(snippet["title"]) == 100
    assert len(snippet["description"]) <= 4900


def test_hashes_are_stripped_from_tags_and_the_count_is_capped(recorder, tmp_path) -> None:
    yt.upload_video(object(), tmp_path / "v.mp4", "T", "D",
                    [f"#tag{i}" for i in range(50)])
    tags = recorder["insert"]["body"]["snippet"]["tags"]
    assert len(tags) == 30
    assert all(not tag.startswith("#") for tag in tags)


def test_made_for_kids_is_declared(recorder, tmp_path) -> None:
    yt.upload_video(object(), tmp_path / "v.mp4", "T", "D", [], made_for_kids=True)
    assert recorder["insert"]["body"]["status"]["selfDeclaredMadeForKids"] is True


# ---------------------------------------------------------------------------
# credentials
# ---------------------------------------------------------------------------

def test_credentials_round_trip() -> None:
    data = {
        "token": "t", "refresh_token": "r",
        "token_uri": "https://oauth2.googleapis.com/token",
        "client_id": "cid", "client_secret": "secret",
        "scopes": yt.SCOPES, "expiry": "2030-01-01T00:00:00",
    }
    creds = yt.credentials_from_dict(data)
    assert creds.refresh_token == "r"
    assert creds.expiry == datetime(2030, 1, 1, 0, 0)
    assert yt.credentials_to_dict(creds)["client_id"] == "cid"


def test_a_corrupt_expiry_does_not_explode() -> None:
    creds = yt.credentials_from_dict({"token": "t", "expiry": "not-a-date"})
    assert creds.expiry is None


def test_refresh_without_a_refresh_token_says_what_to_do() -> None:
    with pytest.raises(RuntimeError, match="reconnect"):
        yt.refresh_if_needed({"token": None, "refresh_token": None})


def test_valid_credentials_are_not_refreshed() -> None:
    data = {
        "token": "t", "refresh_token": "r", "client_id": "c", "client_secret": "s",
        "scopes": yt.SCOPES,
        "expiry": (datetime.utcnow() + timedelta(hours=1)).isoformat(),
    }
    _creds, out, changed = yt.refresh_if_needed(data)
    assert changed is False
    assert out is data


def test_not_configured_message_names_the_redirect_uri() -> None:
    with pytest.raises(yt.YouTubeNotConfigured, match="redirect URI"):
        yt._client_config()


# ---------------------------------------------------------------------------
# error translation
# ---------------------------------------------------------------------------

class _FakeHttpError(Exception):
    def __init__(self, reason: str, status: int = 403):
        super().__init__(reason)
        self.error_details = [{"reason": reason}]
        self.resp = SimpleNamespace(status=status)


def test_quota_error_explains_the_upload_cost() -> None:
    message = yt.describe_http_error(_FakeHttpError("quotaExceeded"))
    assert "1600" in message or "1,600" in message


def test_signup_required_error_is_plain_english() -> None:
    assert "no YouTube channel" in yt.describe_http_error(
        _FakeHttpError("youtubeSignupRequired"))


def test_an_unknown_reason_still_returns_something() -> None:
    assert yt.describe_http_error(_FakeHttpError("somethingNew"))
