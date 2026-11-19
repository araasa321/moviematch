"""Stremio `library` events -> Trakt writes.

Stremio sends library events with a Meta ID (an IMDb id, `tt...`) and an `action`.
The protocol reference (stremio-addon-sdk docs/protocol.md) defines:

    action=libraryAdd      add to watchlist
    action=libraryRemove   remove from watchlist
    action=watched         mark watched       (optional videoId list for episodes)
    action=unwatched       mark unwatched     <- deliberately NOT implemented

Stremio fires these at every installed addon and IGNORES the response, so this
handler must never raise into the request and must be idempotent-ish: the same
event may arrive twice.

Id mapping: Stremio gives IMDb ids, Trakt's write endpoints accept
`{"ids": {"imdb": "tt..."}}` directly, so no TMDB lookup is needed for the write
itself. Episodes (videoId "tt123:1:2") are accepted but only the *show* is
recorded — per-episode history is out of scope for v1.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from .trakt import TraktClient, TraktError

log = logging.getLogger(__name__)

SUPPORTED_ACTIONS = ("libraryAdd", "libraryRemove", "watched")


@dataclass
class LibraryResult:
    ok: bool
    action: str
    detail: str = ""
    handled: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {"success": self.ok, "action": self.action, "detail": self.detail}


def _media(media_type: str, imdb_id: str) -> dict[str, Any]:
    """Trakt media object for a movie or show, keyed by IMDb id."""
    key = "movie" if media_type == "movie" else "show"
    return {key: {"ids": {"imdb": imdb_id}}}


def _base_id(meta_id: str) -> str:
    """`tt0944947:1:2` -> `tt0944947` (episode videoId -> show id)."""
    return meta_id.split(":", 1)[0]


async def handle_library_event(trakt: TraktClient, media_type: str, meta_id: str,
                               action: str, video_id: str | None = None,
                               watched_at: str = "now") -> LibraryResult:
    """Apply one Stremio library event. Never raises."""
    action = (action or "").strip()
    if action not in SUPPORTED_ACTIONS:
        # `unwatched` lands here on purpose: implementing it means deleting Trakt
        # history, which is irreversible and was explicitly excluded.
        detail = ("unwatched is not supported (history deletion is disabled)"
                  if action == "unwatched" else f"unsupported action {action!r}")
        log.info("library event ignored: %s %s/%s", detail, media_type, meta_id)
        return LibraryResult(ok=False, action=action, detail=detail, handled=False)

    if media_type not in ("movie", "series"):
        return LibraryResult(False, action, f"unsupported type {media_type!r}", False)

    imdb_id = _base_id(meta_id or "")
    if not imdb_id.startswith("tt"):
        return LibraryResult(False, action, f"not an IMDb id: {meta_id!r}", False)

    media = _media(media_type, imdb_id)
    try:
        if action == "watched":
            body = await trakt.add_history(media, watched_at=watched_at)
        elif action == "libraryAdd":
            body = await trakt.add_watchlist(media)
        else:  # libraryRemove
            body = await trakt.remove_watchlist(media)
    except TraktError as exc:
        # 403 is the common case: either no write scope or the CDN block (1010).
        log.warning("trakt %s failed for %s: %s", action, imdb_id, exc)
        return LibraryResult(False, action, str(exc)[:200])
    except Exception as exc:  # noqa: BLE001 - never 500 a Stremio event
        log.warning("trakt %s errored for %s: %s", action, imdb_id, type(exc).__name__)
        return LibraryResult(False, action, f"{type(exc).__name__}: {exc}"[:200])

    # A 2xx from Trakt does NOT mean the item was written: a bad id lands in
    # `not_found` and a repeat lands in `existing`/`updated`, both with a 2xx.
    # Reporting success without reading the body is how this silently did nothing.
    ok, note = _interpret(action, body or {})

    if action == "watched" and video_id and ok:
        note += " (show-level; per-episode history not recorded)"
    if ok:
        log.info("trakt %s ok for %s (%s): %s", action, imdb_id, media_type, note)
    else:
        log.warning("trakt %s NOT applied for %s (%s): %s", action, imdb_id,
                    media_type, note)
    return LibraryResult(ok, action, note)


def _count(body: dict[str, Any], key: str, kind: str) -> int:
    """Trakt returns {"added":{"movies":1}} / {"deleted":{...}} / {"not_found":{...}}.

    Any of those may hold a list of objects instead of a number, so this never
    assumes the value is an int (a wrong assumption crashed the first version).
    """
    section = body.get(key)
    if not isinstance(section, dict):
        return 0
    for candidate in (section.get("movies"), section.get(kind)):
        if isinstance(candidate, list):
            return len(candidate)
        if isinstance(candidate, (int, float)):
            return int(candidate)
    return 0


def _interpret(action: str, body: dict[str, Any]) -> tuple[bool, str]:
    """Decide whether Trakt actually applied the change, and describe it."""
    if not body:
        # No JSON body at all — can't confirm, so don't claim success.
        return False, "Trakt returned an empty response"

    if action == "watched":
        added = _count(body, "added", "episodes")
        if added:
            return True, f"added {added} to Trakt history"
        if _count(body, "updated", "episodes"):
            return True, "already in Trakt history (updated)"
        missing = (body.get("not_found") or {}).get("movies") or []
        if missing:
            return False, f"Trakt could not resolve this title: {missing}"
        return False, f"Trakt did not record it: {body}"

    if action == "libraryAdd":
        if _count(body, "added", "shows"):
            return True, "added to Trakt watchlist"
        if _count(body, "existing", "shows"):
            return True, "already on Trakt watchlist"
        return False, f"Trakt did not add it: {body}"

    # libraryRemove
    if _count(body, "deleted", "shows"):
        return True, "removed from Trakt watchlist"
    if _count(body, "not_found", "shows"):
        return True, "was not on the Trakt watchlist"
    return False, f"Trakt did not remove it: {body}"
