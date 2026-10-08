"""The web app, the job model and the YouTube client's request shaping."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest
from fastapi.testclient import TestClient

from shortsmith import main as app_module
from shortsmith.main import _parse_local, app, to_local
from shortsmith.models import (
    STATUS_QUEUED, STATUS_READY, Channel, Job, KIND_CLIP, SessionLocal, init_db,
)


@pytest.fixture(scope="module")
def client():
    init_db()
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(autouse=True)
def _no_worker(monkeypatch):
    """The tests queue jobs; they must not actually render."""
    monkeypatch.setattr(app_module.settings, "worker_enabled", False)


# ---------------------------------------------------------------------------
# timezone round trip
# ---------------------------------------------------------------------------

def test_local_input_converts_to_utc_for_storage() -> None:
    stored = _parse_local("2026-06-01T09:30", ZoneInfo("Asia/Kolkata"))
    assert stored == datetime(2026, 6, 1, 4, 0)      # IST is UTC+5:30


def test_stored_utc_renders_back_in_the_same_local_time() -> None:
    zone = ZoneInfo("Asia/Kolkata")
    stored = _parse_local("2026-06-01T09:30", zone)
    assert to_local(stored, zone) == "01 Jun 2026, 09:30"


def test_round_trip_survives_a_dst_boundary() -> None:
    zone = ZoneInfo("Europe/London")
    # 2026-03-29 is the UK spring-forward day; 10:30 local is 09:30 UTC after it.
    stored = _parse_local("2026-03-29T10:30", zone)
    assert stored == datetime(2026, 3, 29, 9, 30)
    assert to_local(stored, zone) == "29 Mar 2026, 10:30"


def test_bad_input_is_treated_as_no_schedule() -> None:
    assert _parse_local("", ZoneInfo("UTC")) is None
    assert _parse_local("not a date", ZoneInfo("UTC")) is None


# ---------------------------------------------------------------------------
# pages
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("path", ["/", "/create", "/clip", "/channels", "/settings"])
def test_every_page_renders(client, path: str) -> None:
    response = client.get(path)
    assert response.status_code == 200
    assert "Shortsmith" in response.text


def test_healthz(client) -> None:
    assert client.get("/healthz").json() == {"ok": True}


def test_channels_page_explains_the_setup_when_oauth_is_missing(client) -> None:
    response = client.get("/channels")
    assert "youtube/callback" in response.text


# ---------------------------------------------------------------------------
# queueing
# ---------------------------------------------------------------------------

def test_creating_a_short_queues_a_job(client) -> None:
    response = client.post("/create", data={
        "topic": "Sad Story", "publish_mode": "draft", "count": "1",
        "duration": "45", "tone": "emotional", "caption_style": "bold_yellow",
        "words_per_line": "3", "speed": "1.0", "spacing_hours": "24",
        "tz_name": "UTC", "privacy": "public", "motion": "on",
    }, follow_redirects=False)
    assert response.status_code == 303
    job_id = int(response.headers["location"].rsplit("/", 1)[1])
    with SessionLocal() as session:
        job = session.get(Job, job_id)
        assert job.status == STATUS_QUEUED
        assert job.topic == "Sad Story"
        assert job.options["caption_style"] == "bold_yellow"


def test_a_batch_is_spaced_by_the_requested_interval(client) -> None:
    response = client.post("/create", data={
        "topic": "Sci-Fi Story", "publish_mode": "schedule",
        "scheduled_at": "2030-01-01T10:00", "count": "3", "spacing_hours": "6",
        "duration": "45", "tone": "emotional", "caption_style": "bold_yellow",
        "words_per_line": "3", "speed": "1.0", "tz_name": "UTC", "privacy": "public",
    }, follow_redirects=False)
    assert response.status_code == 303
    first_id = int(response.headers["location"].rsplit("/", 1)[1])
    with SessionLocal() as session:
        batch = session.query(Job).filter(Job.id >= first_id,
                                          Job.topic == "Sci-Fi Story").all()
        assert len(batch) == 3
        slots = sorted(job.scheduled_at for job in batch)
        assert slots[1] - slots[0] == timedelta(hours=6)
        assert slots[2] - slots[1] == timedelta(hours=6)


def test_an_empty_topic_is_rejected(client) -> None:
    response = client.post("/create", data={
        "topic": "   ", "publish_mode": "draft", "count": "1", "duration": "45",
        "tone": "emotional", "caption_style": "bold_yellow", "words_per_line": "3",
        "speed": "1.0", "spacing_hours": "24", "tz_name": "UTC", "privacy": "public",
    }, follow_redirects=False)
    assert response.headers["location"] == "/create"


def test_scheduling_without_a_time_is_rejected(client) -> None:
    response = client.post("/create", data={
        "topic": "Sad Story", "publish_mode": "schedule", "scheduled_at": "",
        "count": "1", "duration": "45", "tone": "emotional",
        "caption_style": "bold_yellow", "words_per_line": "3", "speed": "1.0",
        "spacing_hours": "24", "tz_name": "UTC", "privacy": "public",
    }, follow_redirects=False)
    assert response.headers["location"] == "/create"


def test_count_is_clamped(client) -> None:
    response = client.post("/create", data={
        "topic": "Clamp test", "publish_mode": "draft", "count": "500",
        "duration": "45", "tone": "emotional", "caption_style": "bold_yellow",
        "words_per_line": "3", "speed": "1.0", "spacing_hours": "24",
        "tz_name": "UTC", "privacy": "public",
    }, follow_redirects=False)
    assert response.status_code == 303
    with SessionLocal() as session:
        assert session.query(Job).filter(Job.topic == "Clamp test").count() == 30


def test_a_clip_job_rejects_a_source_that_is_neither_url_nor_file(client) -> None:
    response = client.post("/clip", data={
        "source_url": "just some text", "clip_count": "3", "clip_length": "40",
        "reframe": "blur", "caption_style": "bold_yellow", "words_per_line": "3",
        "privacy": "unlisted",
    }, follow_redirects=False)
    assert response.headers["location"] == "/clip"


def test_a_clip_job_accepts_a_url(client) -> None:
    response = client.post("/clip", data={
        "source_url": "https://www.youtube.com/watch?v=abc", "clip_count": "3",
        "clip_length": "40", "reframe": "blur", "caption_style": "bold_yellow",
        "words_per_line": "3", "privacy": "unlisted",
    }, follow_redirects=False)
    assert response.status_code == 303
    job_id = int(response.headers["location"].rsplit("/", 1)[1])
    with SessionLocal() as session:
        job = session.get(Job, job_id)
        assert job.kind == KIND_CLIP
        assert job.options["clip_count"] == 3


# ---------------------------------------------------------------------------
# job actions
# ---------------------------------------------------------------------------

def test_publishing_a_job_with_no_channel_is_refused(client) -> None:
    with SessionLocal() as session:
        job = Job(topic="x", status=STATUS_READY, video_path="/nope.mp4")
        session.add(job)
        session.commit()
        job_id = job.id
    client.post(f"/jobs/{job_id}/publish", follow_redirects=False)
    with SessionLocal() as session:
        assert session.get(Job, job_id).publish_mode != "now"


def test_regenerate_clears_the_cached_script(client) -> None:
    with SessionLocal() as session:
        job = Job(topic="x", status=STATUS_READY, script_json='{"title": "old"}')
        session.add(job)
        session.commit()
        job_id = job.id
    client.post(f"/jobs/{job_id}/regenerate", follow_redirects=False)
    with SessionLocal() as session:
        job = session.get(Job, job_id)
        assert job.script_json == ""
        assert job.status == STATUS_QUEUED


def test_the_dashboard_survives_a_job_that_has_a_channel(client) -> None:
    """Regression: job.channel is read in the template after the session closes.

    Without an eager load that raises DetachedInstanceError and 500s the whole
    queue page as soon as one Short is assigned to a channel.
    """
    with SessionLocal() as session:
        channel = Channel(youtube_channel_id="UC-detached", title="Test Channel")
        session.add(channel)
        session.flush()
        session.add(Job(topic="attached", status=STATUS_READY, channel_id=channel.id,
                        scheduled_at=datetime(2030, 1, 1, 9, 0)))
        session.commit()
    response = client.get("/")
    assert response.status_code == 200
    assert "Test Channel" in response.text


def test_api_jobs_is_json(client) -> None:
    payload = client.get("/api/jobs").json()
    assert isinstance(payload, list)
    assert {"id", "status", "progress", "message"} <= set(payload[0])


# ---------------------------------------------------------------------------
# model helpers
# ---------------------------------------------------------------------------

def test_options_property_round_trips() -> None:
    job = Job(topic="x")
    job.options = {"voice": "amy", "speed": 1.1}
    assert json.loads(job.options_json)["voice"] == "amy"
    assert job.options["speed"] == 1.1


def test_options_property_survives_corrupt_json() -> None:
    job = Job(topic="x", options_json="{not json")
    assert job.options == {}


def test_tag_list_splits_and_trims() -> None:
    assert Job(tags=" a , b ,, c ").tag_list == ["a", "b", "c"]


def test_youtube_url_is_empty_until_published() -> None:
    assert Job().youtube_url == ""
    assert Job(youtube_video_id="abc").youtube_url.endswith("/shorts/abc")


def test_channel_credentials_round_trip() -> None:
    channel = Channel(youtube_channel_id="UC1")
    channel.credentials = {"token": "t", "refresh_token": "r"}
    assert channel.credentials["refresh_token"] == "r"
