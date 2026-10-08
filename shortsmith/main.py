"""Web application.

FastAPI serving server-rendered pages.  No build step, no npm, no bundler: the
whole front end is three templates and one stylesheet, which is deliberate for
something that has to be deployed on a cheap VPS and still be editable by hand
a year from now.
"""

from __future__ import annotations

import json
import logging
import secrets
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo, available_timezones

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import joinedload
from starlette.middleware.sessions import SessionMiddleware

from . import worker
from .config import settings
from .models import (
    ACTIVE_STATUSES, STATUS_CANCELLED, STATUS_FAILED, STATUS_PUBLISHED, STATUS_QUEUED,
    STATUS_READY, Channel, Event, Job, KIND_CLIP, KIND_CLIP_CHILD, KIND_TOPIC,
    SessionLocal, init_db, log_event, utcnow,
)
from .providers import captions as cap
from .providers import images as image_provider
from .providers import llm as llm_provider
from .providers import tts as tts_provider
from .youtube import client as yt

log = logging.getLogger(__name__)

BASE = Path(__file__).resolve().parent
app = FastAPI(title="Shortsmith", docs_url=None, redoc_url=None)
app.add_middleware(SessionMiddleware, secret_key=settings.secret_key, max_age=14 * 24 * 3600)
app.mount("/static", StaticFiles(directory=str(BASE / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE / "templates"))

COMMON_TIMEZONES = [
    "UTC", "Africa/Johannesburg", "Africa/Lagos", "America/Chicago", "America/Los_Angeles",
    "America/New_York", "America/Sao_Paulo", "Asia/Dubai", "Asia/Kolkata", "Asia/Singapore",
    "Asia/Tokyo", "Australia/Sydney", "Europe/Amsterdam", "Europe/Berlin", "Europe/London",
    "Europe/Madrid", "Europe/Paris", "Pacific/Auckland",
]

TONES = ["emotional", "dark", "uplifting", "suspenseful", "calm", "energetic"]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _tz(request: Request) -> ZoneInfo:
    name = request.session.get("tz", "UTC")
    try:
        return ZoneInfo(name)
    except Exception:  # noqa: BLE001
        return ZoneInfo("UTC")


def to_local(value: datetime | None, zone: ZoneInfo) -> str:
    if value is None:
        return ""
    return value.replace(tzinfo=timezone.utc).astimezone(zone).strftime("%d %b %Y, %H:%M")


def _require_login(request: Request) -> None:
    if settings.app_password and not request.session.get("authed"):
        raise HTTPException(status_code=307, headers={"Location": "/login"})


@app.middleware("http")
async def auth_gate(request: Request, call_next):
    open_paths = ("/login", "/static", "/healthz")
    if settings.app_password and not request.url.path.startswith(open_paths):
        if not request.session.get("authed"):
            return RedirectResponse("/login", status_code=303)
    return await call_next(request)


@app.on_event("startup")
def _startup() -> None:
    init_db()
    if settings.worker_enabled:
        worker.start_background()
        log.info("background worker running inside the web process")


templates.env.filters["local"] = lambda value, zone: to_local(value, zone)


def _render(request: Request, name: str, **context):
    zone = _tz(request)
    base = {
        "request": request,
        "tz": zone,
        "tz_name": request.session.get("tz", "UTC"),
        "flash": request.session.pop("flash", None),
        "settings": settings,
        "to_local": lambda value: to_local(value, zone),
        # Stored times are naive UTC; a datetime-local input must show the
        # operator's wall clock or every reschedule drifts by the offset.
        "to_input": lambda value: (
            value.replace(tzinfo=timezone.utc).astimezone(zone).strftime("%Y-%m-%dT%H:%M")
            if value else ""
        ),
        "now_local": datetime.now(zone),
    }
    base.update(context)
    # Starlette 1.x dropped the (name, context) signature; the request is now
    # the first positional argument.
    return templates.TemplateResponse(request, name, base)


def _flash(request: Request, message: str) -> None:
    request.session["flash"] = message


def _parse_local(value: str, zone: ZoneInfo) -> datetime | None:
    """Datetime-local input -> naive UTC for storage."""
    if not value:
        return None
    try:
        naive = datetime.strptime(value, "%Y-%m-%dT%H:%M")
    except ValueError:
        try:
            naive = datetime.strptime(value, "%Y-%m-%dT%H:%M:%S")
        except ValueError:
            return None
    return naive.replace(tzinfo=zone).astimezone(timezone.utc).replace(tzinfo=None)


# ---------------------------------------------------------------------------
# auth
# ---------------------------------------------------------------------------

@app.get("/login", response_class=HTMLResponse)
def login_form(request: Request):
    if not settings.app_password:
        return RedirectResponse("/", status_code=303)
    return _render(request, "login.html", error=None)


@app.post("/login")
def login(request: Request, password: str = Form("")):
    if secrets.compare_digest(password, settings.app_password):
        request.session["authed"] = True
        return RedirectResponse("/", status_code=303)
    return _render(request, "login.html", error="Wrong password.")


@app.get("/logout")
def logout(request: Request):
    request.session.clear()
    return RedirectResponse("/login", status_code=303)


@app.get("/healthz")
def healthz():
    return {"ok": True}


# ---------------------------------------------------------------------------
# dashboard
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request):
    with SessionLocal() as session:
        # joinedload, not lazy: the template reads job.channel after this
        # session closes, and a lazy relationship there raises
        # DetachedInstanceError and 500s the whole dashboard.
        jobs = (
            session.query(Job)
            .options(joinedload(Job.channel))
            .filter(Job.kind != KIND_CLIP_CHILD)
            .order_by(Job.created_at.desc())
            .limit(40)
            .all()
        )
        children = {}
        for job in jobs:
            if job.kind == KIND_CLIP:
                children[job.id] = (
                    session.query(Job).filter(Job.parent_id == job.id)
                    .order_by(Job.id).all()
                )
        channels = session.query(Channel).filter(Channel.is_active).order_by(Channel.title).all()
        counts = {
            "queued": session.query(Job).filter(Job.status == STATUS_QUEUED).count(),
            "ready": session.query(Job).filter(Job.status == STATUS_READY).count(),
            "published": session.query(Job).filter(Job.status == STATUS_PUBLISHED).count(),
            "failed": session.query(Job).filter(Job.status == STATUS_FAILED).count(),
        }
        upcoming = (
            session.query(Job)
            .options(joinedload(Job.channel))
            .filter(Job.status.in_(ACTIVE_STATUSES), Job.scheduled_at.isnot(None))
            .order_by(Job.scheduled_at)
            .limit(8)
            .all()
        )
    return _render(
        request, "dashboard.html", jobs=jobs, children=children, channels=channels,
        counts=counts, upcoming=upcoming,
    )


# ---------------------------------------------------------------------------
# create from a topic
# ---------------------------------------------------------------------------

@app.get("/create", response_class=HTMLResponse)
def create_form(request: Request):
    with SessionLocal() as session:
        channels = session.query(Channel).filter(Channel.is_active).order_by(Channel.title).all()
    music = sorted(p.name for p in settings.music_dir.glob("*") if p.suffix.lower() in
                   {".mp3", ".wav", ".m4a", ".ogg", ".flac"})
    zone = _tz(request)
    return _render(
        request, "create.html",
        channels=channels,
        voices=tts_provider.available_voices(),
        caption_styles=cap.STYLES,
        tones=TONES,
        music_files=music,
        timezones=COMMON_TIMEZONES,
        default_slot=(datetime.now(zone) + timedelta(hours=1)).strftime("%Y-%m-%dT%H:00"),
    )


@app.post("/create")
def create(
    request: Request,
    topic: str = Form(...),
    channel_id: str = Form(""),
    scheduled_at: str = Form(""),
    publish_mode: str = Form("draft"),
    privacy: str = Form("public"),
    duration: int = Form(45),
    tone: str = Form("emotional"),
    voice: str = Form(""),
    speed: float = Form(1.0),
    caption_style: str = Form("bold_yellow"),
    words_per_line: int = Form(3),
    uppercase_captions: str = Form(""),
    image_provider_name: str = Form(""),
    music: str = Form(""),
    motion: str = Form("on"),
    made_for_kids: str = Form(""),
    count: int = Form(1),
    spacing_hours: int = Form(24),
    tz_name: str = Form("UTC"),
):
    topic = topic.strip()
    if not topic:
        _flash(request, "Give it a topic first.")
        return RedirectResponse("/create", status_code=303)

    request.session["tz"] = tz_name
    zone = _tz(request)
    first_slot = _parse_local(scheduled_at, zone)
    if publish_mode == "schedule" and first_slot is None:
        _flash(request, "Pick a date and time, or choose 'render only'.")
        return RedirectResponse("/create", status_code=303)

    options = {
        "duration": max(15, min(90, duration)),
        "tone": tone,
        "voice": voice,
        "speed": max(0.6, min(1.6, speed)),
        "caption_style": caption_style,
        "words_per_line": max(1, min(6, words_per_line)),
        "uppercase_captions": bool(uppercase_captions),
        "image_provider": image_provider_name,
        "music": music,
        "motion": bool(motion),
    }

    created: list[int] = []
    count = max(1, min(30, count))
    with SessionLocal() as session:
        for index in range(count):
            slot = None
            if first_slot is not None:
                slot = first_slot + timedelta(hours=spacing_hours * index)
            job = Job(
                kind=KIND_TOPIC,
                channel_id=int(channel_id) if channel_id else None,
                topic=topic,
                options_json=json.dumps(options),
                status=STATUS_QUEUED,
                scheduled_at=slot,
                publish_mode=publish_mode,
                privacy=privacy,
                made_for_kids=bool(made_for_kids),
                progress_message="Queued",
            )
            session.add(job)
            session.flush()
            created.append(job.id)
        session.commit()
        log_event(session, f"queued {len(created)} job(s) for '{topic}'")

    _flash(request, f"Queued {len(created)} Short{'s' if len(created) > 1 else ''}.")
    return RedirectResponse(f"/jobs/{created[0]}", status_code=303)


# ---------------------------------------------------------------------------
# create from a YouTube URL
# ---------------------------------------------------------------------------

@app.get("/clip", response_class=HTMLResponse)
def clip_form(request: Request):
    with SessionLocal() as session:
        channels = session.query(Channel).filter(Channel.is_active).order_by(Channel.title).all()
    return _render(
        request, "clip.html", channels=channels, caption_styles=cap.STYLES,
    )


@app.post("/clip")
def clip(
    request: Request,
    source_url: str = Form(...),
    channel_id: str = Form(""),
    clip_count: int = Form(3),
    clip_length: int = Form(40),
    reframe: str = Form("blur"),
    caption_style: str = Form("bold_yellow"),
    words_per_line: int = Form(3),
    privacy: str = Form("public"),
):
    from .pipeline.clip import _local_path

    source_url = source_url.strip()
    # A path is as valid a source as a URL: long recordings are often already
    # on the server, and re-uploading them to YouTube first would be silly.
    if not source_url.startswith("http") and _local_path(source_url) is None:
        _flash(request, "That is neither a video URL nor a file on this server.")
        return RedirectResponse("/clip", status_code=303)

    options = {
        "clip_count": max(1, min(10, clip_count)),
        "clip_length": max(15, min(60, clip_length)),
        "reframe": reframe,
        "caption_style": caption_style,
        "words_per_line": max(1, min(6, words_per_line)),
        "privacy": privacy,
    }
    with SessionLocal() as session:
        job = Job(
            kind=KIND_CLIP,
            channel_id=int(channel_id) if channel_id else None,
            source_url=source_url,
            topic=source_url,
            options_json=json.dumps(options),
            status=STATUS_QUEUED,
            publish_mode="draft",
            privacy=privacy,
            progress_message="Queued",
        )
        session.add(job)
        session.commit()
        job_id = job.id
    _flash(request, "Queued. Downloading and transcribing takes a few minutes.")
    return RedirectResponse(f"/jobs/{job_id}", status_code=303)


# ---------------------------------------------------------------------------
# job detail and actions
# ---------------------------------------------------------------------------

@app.get("/jobs/{job_id}", response_class=HTMLResponse)
def job_detail(request: Request, job_id: int):
    with SessionLocal() as session:
        job = session.query(Job).options(joinedload(Job.channel)).filter(
            Job.id == job_id).first()
        if job is None:
            raise HTTPException(404, "No such job")
        children = (
            session.query(Job).filter(Job.parent_id == job_id).order_by(Job.id).all()
            if job.kind == KIND_CLIP else []
        )
        channels = session.query(Channel).filter(Channel.is_active).order_by(Channel.title).all()
        events = (
            session.query(Event).filter(Event.job_id == job_id)
            .order_by(Event.created_at.desc()).limit(20).all()
        )
    return _render(
        request, "job.html", job=job, children=children, channels=channels, events=events,
        script=job.script, caption_styles=cap.STYLES,
    )


@app.post("/jobs/{job_id}/update")
def job_update(
    request: Request,
    job_id: int,
    title: str = Form(""),
    description: str = Form(""),
    tags: str = Form(""),
    channel_id: str = Form(""),
    scheduled_at: str = Form(""),
    publish_mode: str = Form("draft"),
    privacy: str = Form("public"),
):
    zone = _tz(request)
    with SessionLocal() as session:
        job = session.get(Job, job_id)
        if job is None:
            raise HTTPException(404, "No such job")
        job.title = title.strip()[:100]
        job.description = description.strip()
        job.tags = tags.strip()
        job.channel_id = int(channel_id) if channel_id else None
        job.scheduled_at = _parse_local(scheduled_at, zone)
        job.publish_mode = publish_mode
        job.privacy = privacy
        if publish_mode == "schedule" and job.scheduled_at is None:
            job.publish_mode = "draft"
        session.commit()
    _flash(request, "Saved.")
    return RedirectResponse(f"/jobs/{job_id}", status_code=303)


@app.post("/jobs/{job_id}/publish")
def job_publish_now(request: Request, job_id: int):
    with SessionLocal() as session:
        job = session.get(Job, job_id)
        if job is None:
            raise HTTPException(404, "No such job")
        if job.status != STATUS_READY:
            _flash(request, "That job is not ready to publish yet.")
        elif not job.channel_id:
            _flash(request, "Pick a channel first.")
        else:
            job.publish_mode = "now"
            job.attempts = 0
            job.error = ""
            job.progress_message = "Queued for upload"
            session.commit()
            _flash(request, "Uploading now. This page updates by itself.")
    return RedirectResponse(f"/jobs/{job_id}", status_code=303)


@app.post("/jobs/{job_id}/retry")
def job_retry(request: Request, job_id: int):
    with SessionLocal() as session:
        job = session.get(Job, job_id)
        if job is None:
            raise HTTPException(404, "No such job")
        job.status = STATUS_QUEUED
        job.attempts = 0
        job.progress = 0
        job.error = ""
        job.progress_message = "Queued"
        session.commit()
    _flash(request, "Back in the queue.")
    return RedirectResponse(f"/jobs/{job_id}", status_code=303)


@app.post("/jobs/{job_id}/regenerate")
def job_regenerate(request: Request, job_id: int):
    """Re-render from scratch, including a brand new script."""
    with SessionLocal() as session:
        job = session.get(Job, job_id)
        if job is None:
            raise HTTPException(404, "No such job")
        job.script_json = ""
        job.status = STATUS_QUEUED
        job.attempts = 0
        job.progress = 0
        job.error = ""
        job.progress_message = "Queued"
        session.commit()
    _flash(request, "Rewriting and re-rendering.")
    return RedirectResponse(f"/jobs/{job_id}", status_code=303)


@app.post("/jobs/{job_id}/cancel")
def job_cancel(request: Request, job_id: int):
    with SessionLocal() as session:
        job = session.get(Job, job_id)
        if job is None:
            raise HTTPException(404, "No such job")
        job.status = STATUS_CANCELLED
        job.progress_message = "Cancelled"
        session.commit()
    _flash(request, "Cancelled.")
    return RedirectResponse("/", status_code=303)


@app.post("/jobs/{job_id}/delete")
def job_delete(request: Request, job_id: int):
    with SessionLocal() as session:
        job = session.get(Job, job_id)
        if job is None:
            raise HTTPException(404, "No such job")
        for path in (job.video_path, job.thumb_path, job.srt_path):
            if path:
                Path(path).unlink(missing_ok=True)
        session.query(Job).filter(Job.parent_id == job_id).delete()
        session.delete(job)
        session.commit()
    _flash(request, "Deleted.")
    return RedirectResponse("/", status_code=303)


# ---------------------------------------------------------------------------
# media
# ---------------------------------------------------------------------------

@app.get("/media/{job_id}/video")
def media_video(job_id: int):
    with SessionLocal() as session:
        job = session.get(Job, job_id)
        if job is None or not job.video_path or not Path(job.video_path).exists():
            raise HTTPException(404, "No video for that job")
        path = Path(job.video_path)
        name = (job.title or f"short-{job_id}").replace("/", "-")[:80]
    return FileResponse(path, media_type="video/mp4", filename=f"{name}.mp4")


@app.get("/media/{job_id}/thumb")
def media_thumb(job_id: int):
    with SessionLocal() as session:
        job = session.get(Job, job_id)
        if job is None or not job.thumb_path or not Path(job.thumb_path).exists():
            raise HTTPException(404, "No thumbnail")
        path = Path(job.thumb_path)
    return FileResponse(path, media_type="image/jpeg")


@app.get("/media/{job_id}/srt")
def media_srt(job_id: int):
    with SessionLocal() as session:
        job = session.get(Job, job_id)
        if job is None or not job.srt_path or not Path(job.srt_path).exists():
            raise HTTPException(404, "No captions")
        path = Path(job.srt_path)
        name = (job.title or f"short-{job_id}").replace("/", "-")[:80]
    return FileResponse(path, media_type="text/plain", filename=f"{name}.srt")


# ---------------------------------------------------------------------------
# channels / OAuth
# ---------------------------------------------------------------------------

@app.get("/channels", response_class=HTMLResponse)
def channels_page(request: Request):
    with SessionLocal() as session:
        channels = session.query(Channel).order_by(Channel.title).all()
    return _render(
        request, "channels.html", channels=channels,
        configured=yt.is_configured(), redirect_uri=settings.redirect_uri,
    )


@app.get("/youtube/connect")
def youtube_connect(request: Request):
    try:
        url, state = yt.authorization_url()
    except yt.YouTubeNotConfigured as exc:
        _flash(request, str(exc))
        return RedirectResponse("/channels", status_code=303)
    request.session["oauth_state"] = state
    return RedirectResponse(url, status_code=303)


@app.get("/youtube/callback")
def youtube_callback(request: Request):
    state = request.session.pop("oauth_state", "")
    if not state:
        _flash(request, "That sign-in link expired. Start the connection again.")
        return RedirectResponse("/channels", status_code=303)
    try:
        creds = yt.credentials_from_callback(state, str(request.url))
        info = yt.fetch_channel(creds)
    except Exception as exc:  # noqa: BLE001 - surfaced to the operator verbatim
        _flash(request, f"Could not connect that channel: {exc}")
        return RedirectResponse("/channels", status_code=303)

    with SessionLocal() as session:
        channel = (
            session.query(Channel)
            .filter(Channel.youtube_channel_id == info["youtube_channel_id"])
            .first()
        )
        if channel is None:
            channel = Channel(youtube_channel_id=info["youtube_channel_id"])
            session.add(channel)
        channel.title = info["title"]
        channel.handle = info["handle"]
        channel.thumbnail_url = info["thumbnail_url"]
        channel.subscriber_count = info["subscriber_count"]
        channel.credentials = yt.credentials_to_dict(creds)
        channel.scopes = ",".join(creds.scopes or [])
        channel.is_active = True
        channel.last_error = ""
        session.commit()
        log_event(session, f"connected channel {channel.title}")
    _flash(request, f"Connected {info['title']}.")
    return RedirectResponse("/channels", status_code=303)


@app.post("/channels/{channel_id}/disconnect")
def channel_disconnect(request: Request, channel_id: int):
    with SessionLocal() as session:
        channel = session.get(Channel, channel_id)
        if channel is None:
            raise HTTPException(404, "No such channel")
        title = channel.title
        # Keep the row so published jobs still show which channel they went to;
        # only the token goes.
        channel.is_active = False
        channel.credentials_json = "{}"
        session.commit()
    _flash(request, f"Disconnected {title}.")
    return RedirectResponse("/channels", status_code=303)


# ---------------------------------------------------------------------------
# settings and status
# ---------------------------------------------------------------------------

@app.get("/settings", response_class=HTMLResponse)
def settings_page(request: Request):
    return _render(
        request, "settings.html",
        status={
            "script": llm_provider.provider_status(),
            "images": image_provider.provider_status(),
            "voice": tts_provider.provider_status(),
            "youtube": {
                "ok": yt.is_configured(),
                "provider": "google oauth",
                "detail": settings.redirect_uri if yt.is_configured()
                else "GOOGLE_CLIENT_ID / GOOGLE_CLIENT_SECRET not set",
            },
        },
        timezones=sorted(COMMON_TIMEZONES),
    )


@app.post("/settings")
def settings_save(request: Request, tz_name: str = Form("UTC")):
    if tz_name in available_timezones():
        request.session["tz"] = tz_name
        _flash(request, f"Times now shown in {tz_name}.")
    else:
        _flash(request, "Unknown timezone.")
    return RedirectResponse("/settings", status_code=303)


@app.get("/api/jobs")
def api_jobs():
    """Polled by the dashboard and the job page so progress moves without a reload."""
    with SessionLocal() as session:
        jobs = session.query(Job).order_by(Job.created_at.desc()).limit(80).all()
        return JSONResponse([
            {
                "id": job.id,
                "status": job.status,
                "progress": job.progress,
                "message": job.progress_message,
                "error": job.error,
                "duration": job.duration,
                "youtube_video_id": job.youtube_video_id,
                "title": job.title,
            }
            for job in jobs
        ])
