"""Database schema.

SQLite by default because this is a single-operator tool and SQLite removes a
whole class of deployment problems.  Everything goes through SQLAlchemy, so
pointing DATABASE_URL at Postgres or MySQL is a config change, not a rewrite.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import (
    Boolean, Column, DateTime, Float, ForeignKey, Integer, String, Text, create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Session, backref, relationship, sessionmaker

from .config import settings


def utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class Base(DeclarativeBase):
    pass


# --- job state machine -----------------------------------------------------
# queued -> rendering -> ready -> publishing -> published
#                   \-> failed            \-> failed
# A job with no channel stops at `ready` and is downloaded by hand.

STATUS_QUEUED = "queued"
STATUS_RENDERING = "rendering"
STATUS_READY = "ready"
STATUS_PUBLISHING = "publishing"
STATUS_PUBLISHED = "published"
STATUS_FAILED = "failed"
STATUS_CANCELLED = "cancelled"

ACTIVE_STATUSES = (STATUS_QUEUED, STATUS_RENDERING, STATUS_READY, STATUS_PUBLISHING)

KIND_TOPIC = "topic"      # topic -> generated Short
KIND_CLIP = "clip"        # a YouTube URL -> several Shorts (the parent row)
KIND_CLIP_CHILD = "clip_child"   # one Short cut out of that URL


class Channel(Base):
    __tablename__ = "channels"

    id = Column(Integer, primary_key=True)
    youtube_channel_id = Column(String(64), unique=True, nullable=False)
    title = Column(String(255), nullable=False, default="")
    handle = Column(String(255), default="")
    thumbnail_url = Column(String(512), default="")
    subscriber_count = Column(String(32), default="")
    credentials_json = Column(Text, nullable=False, default="{}")
    scopes = Column(Text, default="")
    is_active = Column(Boolean, default=True, nullable=False)
    last_error = Column(Text, default="")
    created_at = Column(DateTime, default=utcnow, nullable=False)
    updated_at = Column(DateTime, default=utcnow, onupdate=utcnow, nullable=False)

    jobs = relationship("Job", back_populates="channel")

    @property
    def credentials(self) -> dict[str, Any]:
        try:
            return json.loads(self.credentials_json or "{}")
        except json.JSONDecodeError:
            return {}

    @credentials.setter
    def credentials(self, value: dict[str, Any]) -> None:
        self.credentials_json = json.dumps(value)


class Job(Base):
    __tablename__ = "jobs"

    id = Column(Integer, primary_key=True)
    kind = Column(String(16), default=KIND_TOPIC, nullable=False)
    parent_id = Column(Integer, ForeignKey("jobs.id"), nullable=True)
    channel_id = Column(Integer, ForeignKey("channels.id"), nullable=True)

    topic = Column(String(512), default="")
    source_url = Column(String(1024), default="")
    options_json = Column(Text, default="{}")
    script_json = Column(Text, default="")

    status = Column(String(16), default=STATUS_QUEUED, nullable=False, index=True)
    progress = Column(Integer, default=0, nullable=False)
    progress_message = Column(String(255), default="")
    error = Column(Text, default="")
    attempts = Column(Integer, default=0, nullable=False)

    # Publishing.  `scheduled_at` is stored in UTC; the UI converts using the
    # timezone the user picked, because a Short posted an hour late is a
    # different Short as far as the algorithm is concerned.
    scheduled_at = Column(DateTime, nullable=True, index=True)
    publish_mode = Column(String(16), default="schedule")  # schedule | now | draft
    privacy = Column(String(16), default="public")         # public | unlisted | private
    title = Column(String(255), default="")
    description = Column(Text, default="")
    tags = Column(Text, default="")
    made_for_kids = Column(Boolean, default=False, nullable=False)

    video_path = Column(String(512), default="")
    thumb_path = Column(String(512), default="")
    srt_path = Column(String(512), default="")
    duration = Column(Float, default=0.0)
    youtube_video_id = Column(String(64), default="")
    published_at = Column(DateTime, nullable=True)

    created_at = Column(DateTime, default=utcnow, nullable=False)
    updated_at = Column(DateTime, default=utcnow, onupdate=utcnow, nullable=False)
    render_started_at = Column(DateTime, nullable=True)
    render_finished_at = Column(DateTime, nullable=True)

    channel = relationship("Channel", back_populates="jobs")
    # Self-referential: `remote_side` belongs on the MANY-TO-ONE side (the
    # parent), not on the collection, or SQLAlchemy reads both directions as
    # one-to-many and refuses to map the class at all.
    children = relationship(
        "Job", backref=backref("parent", remote_side=[id]), cascade="all"
    )

    @property
    def options(self) -> dict[str, Any]:
        try:
            return json.loads(self.options_json or "{}")
        except json.JSONDecodeError:
            return {}

    @options.setter
    def options(self, value: dict[str, Any]) -> None:
        self.options_json = json.dumps(value)

    @property
    def script(self) -> dict[str, Any] | None:
        if not self.script_json:
            return None
        try:
            return json.loads(self.script_json)
        except json.JSONDecodeError:
            return None

    @property
    def tag_list(self) -> list[str]:
        return [t.strip() for t in (self.tags or "").split(",") if t.strip()]

    @property
    def youtube_url(self) -> str:
        return f"https://www.youtube.com/shorts/{self.youtube_video_id}" if self.youtube_video_id else ""


class Setting(Base):
    """Runtime settings editable from the UI, overlaid on the .env defaults."""

    __tablename__ = "settings"

    key = Column(String(64), primary_key=True)
    value = Column(Text, default="")
    updated_at = Column(DateTime, default=utcnow, onupdate=utcnow, nullable=False)


class Event(Base):
    """Append-only activity log, so a failed overnight batch can be read back."""

    __tablename__ = "events"

    id = Column(Integer, primary_key=True)
    job_id = Column(Integer, ForeignKey("jobs.id"), nullable=True, index=True)
    level = Column(String(16), default="info")
    message = Column(Text, default="")
    created_at = Column(DateTime, default=utcnow, nullable=False, index=True)


_connect_args = {"check_same_thread": False} if settings.database_url.startswith("sqlite") else {}
engine = create_engine(settings.database_url, future=True, connect_args=_connect_args)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False, future=True)


def init_db() -> None:
    Base.metadata.create_all(engine)
    if settings.database_url.startswith("sqlite"):
        # The worker writes while the web process reads; without WAL the two
        # block each other and the dashboard stalls mid-render.
        with engine.connect() as conn:
            conn.exec_driver_sql("PRAGMA journal_mode=WAL")
            conn.exec_driver_sql("PRAGMA busy_timeout=10000")


def get_session() -> Session:
    return SessionLocal()


def log_event(session: Session, message: str, job_id: int | None = None, level: str = "info") -> None:
    session.add(Event(job_id=job_id, level=level, message=message[:4000]))
    session.commit()
