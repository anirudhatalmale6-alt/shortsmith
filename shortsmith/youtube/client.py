"""YouTube OAuth and upload.

One Google Cloud OAuth client, many connected channels.  Each channel keeps its
own refresh token, so "publish to the right channel" is just picking the right
credential rather than juggling browser logins.

Two things about the YouTube Data API that bite every project that automates
uploads, documented here because they change what the operator has to do:

1. A video can only be scheduled with `status.publishAt` if it is uploaded as
   `private`.  Setting publishAt on a public upload is silently ignored.  So a
   scheduled Short goes up private with a publish time and YouTube flips it.

2. Until the Google Cloud project passes YouTube API audit, every upload is
   locked to private and `publishAt` is ignored too.  The upload succeeds, the
   video exists, and it simply never goes public.  That is a Google-side review
   of the project, not a bug in this code, and it is why the dashboard shows
   the audit state rather than letting it look like a silent failure.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload

from ..config import settings

log = logging.getLogger(__name__)

SCOPES = [
    "https://www.googleapis.com/auth/youtube.upload",
    "https://www.googleapis.com/auth/youtube.readonly",
    "https://www.googleapis.com/auth/youtube.force-ssl",
]

UPLOAD_CHUNK = 4 * 1024 * 1024


class YouTubeNotConfigured(RuntimeError):
    pass


def is_configured() -> bool:
    return bool(settings.google_client_id and settings.google_client_secret)


def _client_config() -> dict[str, Any]:
    if not is_configured():
        raise YouTubeNotConfigured(
            "GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET are not set. "
            "Create an OAuth client (type: Web application) in Google Cloud Console "
            f"and add {settings.redirect_uri} as an authorised redirect URI."
        )
    return {
        "web": {
            "client_id": settings.google_client_id,
            "client_secret": settings.google_client_secret,
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
            "redirect_uris": [settings.redirect_uri],
        }
    }


def build_flow(state: str | None = None) -> Flow:
    flow = Flow.from_client_config(_client_config(), scopes=SCOPES, state=state)
    flow.redirect_uri = settings.redirect_uri
    return flow


def authorization_url() -> tuple[str, str]:
    """Returns (url, state).

    `access_type=offline` plus `prompt=consent` is not belt and braces: without
    the forced consent screen Google withholds the refresh token on every
    re-authorisation after the first, and the channel silently stops working a
    week later when the access token expires.
    """
    flow = build_flow()
    url, state = flow.authorization_url(
        access_type="offline",
        include_granted_scopes="true",
        prompt="consent",
    )
    return url, state


def credentials_from_callback(state: str, full_callback_url: str) -> Credentials:
    flow = build_flow(state=state)
    flow.fetch_token(authorization_response=full_callback_url)
    return flow.credentials


def credentials_to_dict(creds: Credentials) -> dict[str, Any]:
    return {
        "token": creds.token,
        "refresh_token": creds.refresh_token,
        "token_uri": creds.token_uri,
        "client_id": creds.client_id,
        "client_secret": creds.client_secret,
        "scopes": list(creds.scopes or []),
        "expiry": creds.expiry.isoformat() if creds.expiry else None,
    }


def credentials_from_dict(data: dict[str, Any]) -> Credentials:
    expiry = data.get("expiry")
    creds = Credentials(
        token=data.get("token"),
        refresh_token=data.get("refresh_token"),
        token_uri=data.get("token_uri", "https://oauth2.googleapis.com/token"),
        client_id=data.get("client_id") or settings.google_client_id,
        client_secret=data.get("client_secret") or settings.google_client_secret,
        scopes=data.get("scopes") or SCOPES,
    )
    if expiry:
        try:
            parsed = datetime.fromisoformat(expiry)
            creds.expiry = parsed.replace(tzinfo=None) if parsed.tzinfo else parsed
        except ValueError:
            pass
    return creds


def refresh_if_needed(data: dict[str, Any]) -> tuple[Credentials, dict[str, Any], bool]:
    """Return live credentials plus the dict to persist if they were refreshed."""
    creds = credentials_from_dict(data)
    if creds.valid:
        return creds, data, False
    if not creds.refresh_token:
        raise RuntimeError(
            "This channel has no refresh token. Disconnect and reconnect it; "
            "Google only issues one when the consent screen is shown."
        )
    creds.refresh(Request())
    return creds, credentials_to_dict(creds), True


def service_for(creds: Credentials):
    return build("youtube", "v3", credentials=creds, cache_discovery=False)


def fetch_channel(creds: Credentials) -> dict[str, str]:
    youtube = service_for(creds)
    response = youtube.channels().list(part="snippet,statistics,status", mine=True).execute()
    items = response.get("items") or []
    if not items:
        raise RuntimeError(
            "That Google account has no YouTube channel. Pick the Brand Account "
            "for the channel on the Google consent screen, not the personal account."
        )
    item = items[0]
    snippet = item.get("snippet", {})
    thumbs = snippet.get("thumbnails", {})
    return {
        "youtube_channel_id": item["id"],
        "title": snippet.get("title", ""),
        "handle": snippet.get("customUrl", ""),
        "thumbnail_url": (thumbs.get("medium") or thumbs.get("default") or {}).get("url", ""),
        "subscriber_count": str(item.get("statistics", {}).get("subscriberCount", "")),
    }


def upload_video(
    creds: Credentials,
    video_path: Path,
    title: str,
    description: str,
    tags: list[str],
    privacy: str = "public",
    publish_at: datetime | None = None,
    made_for_kids: bool = False,
    category_id: str = "22",
    thumbnail: Path | None = None,
    on_progress=None,
) -> dict[str, Any]:
    """Resumable upload.  Returns the created video resource."""
    youtube = service_for(creds)

    status: dict[str, Any] = {
        "selfDeclaredMadeForKids": made_for_kids,
        "privacyStatus": privacy,
    }
    if publish_at is not None:
        # Scheduling only works from `private`; see the module docstring.
        if publish_at.tzinfo is None:
            publish_at = publish_at.replace(tzinfo=timezone.utc)
        status["privacyStatus"] = "private"
        status["publishAt"] = publish_at.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")

    body = {
        "snippet": {
            "title": title[:100],
            "description": description[:4900],
            "tags": [t.lstrip("#")[:30] for t in tags][:30],
            "categoryId": category_id,
        },
        "status": status,
    }

    media = MediaFileUpload(str(video_path), chunksize=UPLOAD_CHUNK, resumable=True,
                            mimetype="video/mp4")
    request = youtube.videos().insert(part="snippet,status", body=body, media_body=media)

    response = None
    while response is None:
        progress, response = request.next_chunk()
        if progress and on_progress:
            on_progress(int(progress.progress() * 100))

    if thumbnail and thumbnail.exists():
        try:
            youtube.thumbnails().set(
                videoId=response["id"],
                media_body=MediaFileUpload(str(thumbnail), mimetype="image/jpeg"),
            ).execute()
        except HttpError as exc:
            # Custom thumbnails need a verified phone number on the channel.
            # Not having one must not fail an otherwise good upload.
            log.warning("thumbnail upload refused: %s", exc)

    return response


def upload_caption(creds: Credentials, video_id: str, srt_path: Path, language: str = "en") -> None:
    youtube = service_for(creds)
    youtube.captions().insert(
        part="snippet",
        body={"snippet": {"videoId": video_id, "language": language,
                          "name": "English", "isDraft": False}},
        media_body=MediaFileUpload(str(srt_path), mimetype="application/octet-stream"),
    ).execute()


def describe_http_error(exc: HttpError) -> str:
    """Turn Google's JSON error envelope into one line an operator can act on."""
    try:
        detail = exc.error_details  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        detail = None
    reason = ""
    if isinstance(detail, list) and detail:
        reason = detail[0].get("reason", "") or detail[0].get("message", "")
    hints = {
        "quotaExceeded": "The project's daily YouTube API quota is used up. "
                         "Uploads cost 1600 units of the default 10,000 per day, "
                         "so roughly 6 uploads a day until you request more quota.",
        "uploadLimitExceeded": "This channel has hit its own daily upload limit. Try tomorrow.",
        "youtubeSignupRequired": "That Google account has no YouTube channel yet.",
        "forbidden": "The token does not have upload permission for this channel.",
        "failedPrecondition": "YouTube rejected the request as it stands; check title and "
                              "description length and that the video file is complete.",
    }
    return f"{reason or exc.resp.status}: {hints.get(reason, str(exc)[:300])}"
