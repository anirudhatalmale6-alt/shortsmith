"""Background worker.

One loop, two responsibilities:

  render   - take the oldest queued job and build its video
  publish  - take any ready job whose scheduled time has arrived and upload it

It runs in a thread inside the web process by default, which keeps deployment
to a single command.  Set WORKER_ENABLED=0 and run `python -m shortsmith.worker`
to move it to its own process once there is enough volume to want that.

Jobs are claimed with a conditional UPDATE rather than a read-then-write, so two
workers cannot pick up the same job.  That matters the first time somebody runs
a second instance to get through a backlog.
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from sqlalchemy import update

from .config import settings
from .models import (
    ACTIVE_STATUSES, STATUS_FAILED, STATUS_PUBLISHED, STATUS_PUBLISHING, STATUS_QUEUED,
    STATUS_READY, STATUS_RENDERING, Channel, Job, KIND_CLIP, KIND_CLIP_CHILD, KIND_TOPIC,
    SessionLocal, init_db, log_event, utcnow,
)
from .pipeline import clip as clipper
from .pipeline.render import RenderOptions, render_short
from .providers import captions as cap
from .providers.llm import write_script
from .providers.script_types import ScriptResult

log = logging.getLogger(__name__)

MAX_ATTEMPTS = 2
_stop = threading.Event()


# ---------------------------------------------------------------------------
# claiming
# ---------------------------------------------------------------------------

def _claim(session, status_from: str, status_to: str, extra_filter=None) -> Job | None:
    query = session.query(Job).filter(Job.status == status_from)
    if extra_filter is not None:
        query = query.filter(extra_filter)
    job = query.order_by(Job.scheduled_at.is_(None), Job.scheduled_at, Job.id).first()
    if job is None:
        return None
    # Conditional update: whoever flips the status first owns the job.
    result = session.execute(
        update(Job)
        .where(Job.id == job.id, Job.status == status_from)
        .values(status=status_to, updated_at=utcnow())
    )
    session.commit()
    if result.rowcount != 1:
        return None
    session.refresh(job)
    return job


def _set_progress(job_id: int, message: str, percent: int) -> None:
    with SessionLocal() as session:
        session.execute(
            update(Job).where(Job.id == job_id)
            .values(progress=percent, progress_message=message[:255], updated_at=utcnow())
        )
        session.commit()


def _fail(job_id: int, error: str) -> None:
    with SessionLocal() as session:
        job = session.get(Job, job_id)
        if job is None:
            return
        job.attempts += 1
        job.error = error[:4000]
        # A transient provider failure deserves one more go; a bad request does not.
        job.status = STATUS_QUEUED if job.attempts < MAX_ATTEMPTS else STATUS_FAILED
        job.progress_message = "Retrying" if job.status == STATUS_QUEUED else "Failed"
        session.commit()
        log_event(session, f"job {job_id} failed: {error[:500]}", job_id=job_id, level="error")
    log.error("job %s failed: %s", job_id, error[:500])


def _options_from(job: Job) -> RenderOptions:
    opts = job.options
    return RenderOptions(
        voice=opts.get("voice", ""),
        speed=float(opts.get("speed", 1.0)),
        caption_style=opts.get("caption_style", "bold_yellow"),
        words_per_line=int(opts.get("words_per_line", 3)),
        uppercase_captions=bool(opts.get("uppercase_captions", True)),
        image_provider=opts.get("image_provider", ""),
        music=opts.get("music", ""),
        motion=bool(opts.get("motion", True)),
    )


# ---------------------------------------------------------------------------
# topic -> Short
# ---------------------------------------------------------------------------

def run_topic_job(job_id: int) -> None:
    with SessionLocal() as session:
        job = session.get(Job, job_id)
        if job is None:
            return
        topic = job.topic
        opts = job.options
        existing_script = job.script
        job.render_started_at = utcnow()
        session.commit()

    _set_progress(job_id, "Writing the script", 4)
    if existing_script:
        script = ScriptResult.from_dict(existing_script)
    else:
        script = write_script(
            topic,
            duration=int(opts.get("duration", 45)),
            style=opts.get("style", "cinematic storytelling"),
            tone=opts.get("tone", "emotional"),
        )

    with SessionLocal() as session:
        job = session.get(Job, job_id)
        job.script_json = __import__("json").dumps(script.to_dict())
        if not job.title:
            job.title = script.title
        if not job.description:
            tags = " ".join(script.hashtags)
            job.description = f"{script.description}\n\n{tags}".strip()
        if not job.tags:
            job.tags = ", ".join(h.lstrip("#") for h in script.hashtags)
        session.commit()
        options = _options_from(job)

    work_dir = settings.work_dir / f"job_{job_id}"
    out_path = settings.media_dir / f"job_{job_id}.mp4"
    result = render_short(
        script, work_dir, out_path, options,
        progress=lambda message, percent: _set_progress(job_id, message, percent),
    )

    with SessionLocal() as session:
        job = session.get(Job, job_id)
        job.video_path = str(result.video)
        job.thumb_path = str(result.thumbnail)
        job.srt_path = str(result.srt) if result.srt else ""
        job.duration = result.duration
        job.status = STATUS_READY
        job.progress = 100
        job.progress_message = "Ready"
        job.error = ""
        job.render_finished_at = utcnow()
        session.commit()
        log_event(session, f"rendered {result.duration:.1f}s video", job_id=job_id)


# ---------------------------------------------------------------------------
# URL -> several Shorts
# ---------------------------------------------------------------------------

def run_clip_job(job_id: int) -> None:
    with SessionLocal() as session:
        job = session.get(Job, job_id)
        if job is None:
            return
        url = job.source_url
        opts = job.options
        channel_id = job.channel_id
        job.render_started_at = utcnow()
        session.commit()

    want = int(opts.get("clip_count", 3))
    target = float(opts.get("clip_length", 40))
    reframe = opts.get("reframe", "blur")
    caption_style = opts.get("caption_style", "bold_yellow")
    work_dir = settings.work_dir / f"clip_{job_id}"

    _set_progress(job_id, "Reading the source video", 5)
    info = clipper.ytdlp_info(url)
    duration = float(info.get("duration") or 0)
    if duration > settings.max_clip_minutes * 60:
        raise RuntimeError(
            f"That video is {duration / 60:.0f} minutes long; the limit is "
            f"{settings.max_clip_minutes} minutes. Raise MAX_CLIP_MINUTES to allow it."
        )

    _set_progress(job_id, "Downloading", 12)
    source = clipper.download(url, work_dir)

    _set_progress(job_id, "Transcribing", 30)
    audio = work_dir / "audio.wav"
    import subprocess

    subprocess.run(
        [settings.ffmpeg, "-y", "-loglevel", "error", "-i", str(source),
         "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(audio)],
        check=True,
    )
    words = cap.transcribe_words(audio)
    if not words:
        raise RuntimeError("No speech found in that video, so there is nothing to clip on.")

    _set_progress(job_id, "Choosing the best moments", 55)
    sentences = clipper.sentences_from_words(words)
    picks = clipper.pick_segments(sentences, count=want, target=target)
    if not picks:
        raise RuntimeError(
            "No segment long enough to make a Short. Try a longer source video "
            "or reduce the clip length."
        )
    picks = clipper.rerank_with_model(picks)[:want]
    picks.sort(key=lambda c: c.start)

    source_title = info.get("title", "") or "Clip"
    created: list[int] = []
    for index, candidate in enumerate(picks, start=1):
        _set_progress(job_id, f"Cutting clip {index} of {len(picks)}",
                      55 + int(40 * index / len(picks)))
        with SessionLocal() as session:
            child = Job(
                kind=KIND_CLIP_CHILD,
                parent_id=job_id,
                channel_id=channel_id,
                topic=f"{source_title} (clip {index})",
                source_url=url,
                options_json=job_options_json(opts),
                status=STATUS_RENDERING,
                title=(candidate.title or f"{source_title} ({index})")[:95],
                description=(" ".join(candidate.text.split())[:400] + f"\n\nSource: {url}"),
                tags="shorts",
                privacy=opts.get("privacy", "public"),
                publish_mode="draft",
                progress=50,
                progress_message="Cutting",
            )
            session.add(child)
            session.commit()
            child_id = child.id

        clip_words = clipper.words_in_window(words, candidate.start, candidate.end)
        ass_path = None
        if caption_style != "none" and clip_words:
            ass_path = cap.build_ass(
                clip_words, work_dir / f"clip_{index}.ass", style=caption_style,
                words_per_line=int(opts.get("words_per_line", 3)),
                uppercase=bool(opts.get("uppercase_captions", True)),
            )
        out_path = settings.media_dir / f"job_{child_id}.mp4"
        clipper.cut_clip(source, out_path, candidate.start, candidate.end,
                         reframe=reframe, ass_path=ass_path)
        thumb = out_path.with_suffix(".jpg")
        subprocess.run(
            [settings.ffmpeg, "-y", "-loglevel", "error", "-ss", "0.5", "-i", str(out_path),
             "-frames:v", "1", "-q:v", "3", str(thumb)],
            check=False,
        )
        srt = cap.write_srt(clip_words, out_path.with_suffix(".srt")) if clip_words else None

        with SessionLocal() as session:
            child = session.get(Job, child_id)
            child.video_path = str(out_path)
            child.thumb_path = str(thumb) if thumb.exists() else ""
            child.srt_path = str(srt) if srt else ""
            child.duration = candidate.duration
            child.status = STATUS_READY
            child.progress = 100
            child.progress_message = "Ready"
            child.render_finished_at = utcnow()
            session.commit()
        created.append(child_id)

    with SessionLocal() as session:
        job = session.get(Job, job_id)
        job.status = STATUS_READY
        job.progress = 100
        job.progress_message = f"{len(created)} Shorts ready"
        job.render_finished_at = utcnow()
        session.commit()
        log_event(session, f"cut {len(created)} Shorts from {url}", job_id=job_id)


def job_options_json(opts: dict) -> str:
    import json

    return json.dumps(opts)


# ---------------------------------------------------------------------------
# publishing
# ---------------------------------------------------------------------------

def publish_job(job_id: int) -> None:
    from googleapiclient.errors import HttpError

    from .youtube import client as yt

    with SessionLocal() as session:
        job = session.get(Job, job_id)
        if job is None:
            return
        if not job.channel_id:
            raise RuntimeError("No channel selected for this job.")
        channel = session.get(Channel, job.channel_id)
        if channel is None:
            raise RuntimeError("The channel for this job has been removed.")
        video_path = Path(job.video_path)
        title = job.title or job.topic or "Short"
        description = job.description or ""
        tags = job.tag_list
        privacy = job.privacy or "public"
        made_for_kids = bool(job.made_for_kids)
        publish_mode = job.publish_mode
        scheduled_at = job.scheduled_at
        thumb = Path(job.thumb_path) if job.thumb_path else None
        srt = Path(job.srt_path) if job.srt_path else None
        creds_data = channel.credentials

    if not video_path.exists():
        raise RuntimeError(f"The rendered file is missing: {video_path}")

    creds, refreshed, changed = yt.refresh_if_needed(creds_data)
    if changed:
        with SessionLocal() as session:
            channel = session.get(Channel, job.channel_id)
            channel.credentials = refreshed
            session.commit()

    # Only pass publishAt when the slot is still in the future; a past timestamp
    # is rejected, and that is exactly what happens when a render overruns.
    publish_at = None
    if publish_mode == "schedule" and scheduled_at:
        slot = scheduled_at.replace(tzinfo=timezone.utc)
        if slot > datetime.now(timezone.utc) + timedelta(minutes=2):
            publish_at = slot

    try:
        response = yt.upload_video(
            creds, video_path, title, description, tags,
            privacy=privacy, publish_at=publish_at, made_for_kids=made_for_kids,
            thumbnail=thumb if thumb and thumb.exists() else None,
            on_progress=lambda percent: _set_progress(job_id, f"Uploading {percent}%", percent),
        )
    except HttpError as exc:
        raise RuntimeError(yt.describe_http_error(exc)) from exc

    video_id = response.get("id", "")
    if srt and srt.exists():
        try:
            yt.upload_caption(creds, video_id, srt)
        except Exception as exc:  # noqa: BLE001 - captions are a nice-to-have
            log.warning("caption upload skipped: %s", exc)

    with SessionLocal() as session:
        job = session.get(Job, job_id)
        job.youtube_video_id = video_id
        job.status = STATUS_PUBLISHED
        job.published_at = utcnow()
        job.progress = 100
        actual = (response.get("status", {}) or {}).get("privacyStatus", "")
        job.progress_message = (
            f"Scheduled on YouTube ({actual})" if publish_at else f"Published ({actual})"
        )
        job.error = ""
        session.commit()
        log_event(session, f"uploaded as {video_id}", job_id=job_id)


# ---------------------------------------------------------------------------
# loop
# ---------------------------------------------------------------------------

def tick() -> bool:
    """One pass. Returns True if it did any work."""
    did_work = False

    with SessionLocal() as session:
        job = _claim(session, STATUS_QUEUED, STATUS_RENDERING)
        job_id, kind = (job.id, job.kind) if job else (None, None)

    if job_id is not None:
        did_work = True
        try:
            if kind == KIND_CLIP:
                run_clip_job(job_id)
            else:
                run_topic_job(job_id)
        except Exception as exc:  # noqa: BLE001 - the loop must survive any job
            log.exception("render job %s blew up", job_id)
            _fail(job_id, f"{type(exc).__name__}: {exc}")

    # Anything ready, attached to a channel, and due.
    now = utcnow()
    with SessionLocal() as session:
        due = (
            session.query(Job)
            .filter(
                Job.status == STATUS_READY,
                Job.channel_id.isnot(None),
                Job.publish_mode.in_(("schedule", "now")),
                Job.kind != KIND_CLIP,
            )
            .filter((Job.publish_mode == "now") | (Job.scheduled_at <= now))
            .order_by(Job.scheduled_at)
            .first()
        )
        if due is not None:
            result = session.execute(
                update(Job).where(Job.id == due.id, Job.status == STATUS_READY)
                .values(status=STATUS_PUBLISHING, progress_message="Uploading",
                        updated_at=utcnow())
            )
            session.commit()
            publish_id = due.id if result.rowcount == 1 else None
        else:
            publish_id = None

    if publish_id is not None:
        did_work = True
        try:
            publish_job(publish_id)
        except Exception as exc:  # noqa: BLE001
            log.exception("publish job %s blew up", publish_id)
            with SessionLocal() as session:
                job = session.get(Job, publish_id)
                if job is not None:
                    job.attempts += 1
                    job.error = f"{type(exc).__name__}: {exc}"[:4000]
                    # Back to ready so it retries on the next tick, unless it has
                    # already burned its attempts; an upload failure is usually a
                    # token or quota problem that clears on its own.
                    job.status = STATUS_READY if job.attempts < MAX_ATTEMPTS else STATUS_FAILED
                    job.progress_message = (
                        "Upload failed, will retry" if job.status == STATUS_READY
                        else "Upload failed"
                    )
                    session.commit()

    return did_work


def run_forever() -> None:
    init_db()
    log.info("worker started (poll %ss)", settings.worker_poll_seconds)
    while not _stop.is_set():
        try:
            busy = tick()
        except Exception:  # noqa: BLE001
            log.exception("worker tick failed")
            busy = False
        _stop.wait(1 if busy else settings.worker_poll_seconds)


def start_background() -> threading.Thread:
    thread = threading.Thread(target=run_forever, name="shortsmith-worker", daemon=True)
    thread.start()
    return thread


def stop() -> None:
    _stop.set()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    run_forever()
