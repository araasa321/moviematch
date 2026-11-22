"""Trakt-recommendation rows: Trakt decides the order, the catalog supplies metadata.

This replaces the old ranking pipeline for the `foryou` row. Trakt's recommendation
engine is collaborative filtering over its whole user base — far more signal than
one person's watch history can give — so the addon no longer re-scores what Trakt
returns. It joins the returned ids to the local catalog (for poster/title/year/
imdb_id, which Stremio needs) and preserves Trakt's order.

What the addon still owns:
  * the Trakt -> catalog join (ids only, no matching by title),
  * filtering out anything already watched / watchlisted / hidden,
  * caching (Trakt allows 1000 GET/5min; a row is fetched at most twice per cache
    window),
  * graceful degradation — if Trakt fails, serve the last good list, then nothing.

Stremio -> Trakt WRITES are untouched: `addon/library.py` still handles
libraryAdd / libraryRemove / watched.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from .ids import point_id
from .store import Record

log = logging.getLogger(__name__)

# Trakt recommendations change slowly (they are recomputed from watch history, not
# per request). 6h is ~2 calls/hour, far inside the 1000/5min limit, and stale-serving
# means a Trakt outage is invisible to the user for hours.
RECOMMENDATIONS_TTL = 6 * 3600.0
# Hard cap on how stale a list may get before it is discarded rather than served.
RECOMMENDATIONS_MAX_STALE = 72 * 3600.0


@dataclass
class CachedList:
    """One cached Trakt response plus the bookkeeping to degrade gracefully."""
    items: list[dict[str, Any]] = field(default_factory=list)
    at: float = 0.0
    ok: bool = False


def trakt_ids(item: dict[str, Any]) -> tuple[str | None, int | None]:
    """(imdb_id, tmdb_id) from a Trakt media object, tolerating missing keys."""
    ids = item.get("ids") or {}
    imdb = ids.get("imdb") or None
    tmdb = ids.get("tmdb") or None
    try:
        tmdb = int(tmdb) if tmdb is not None else None
    except (TypeError, ValueError):
        tmdb = None
    return (str(imdb) if imdb else None), tmdb


def ordered_records(items: Iterable[dict[str, Any]], store: Any, media_type: str,
                    *, excluded: set[str] | None = None) -> list[Record]:
    """Resolve Trakt items to catalog records, PRESERVING Trakt's order.

    The catalog is keyed by `point_id(media_type, tmdb_id)` and every record stores
    `imdb_id` in its payload. Trakt returns both, so the join is: tmdb id -> point
    id. Items without a tmdb id, or not in the catalog, are skipped — the caller
    only ever sees renderable titles.

    `excluded` holds point ids that must not appear (already watched, hidden, or
    watchlisted). This is applied here as well as upstream because a cached Trakt
    list can be hours old: the user may have watched something since it was fetched.
    """
    excluded = excluded or set()
    wanted: list[str] = []
    order: list[tuple[str, dict[str, Any]]] = []
    for item in items:
        _imdb, tmdb = trakt_ids(item)
        if tmdb is None:
            continue
        pid = point_id(media_type, tmdb)
        if pid in excluded:
            continue
        wanted.append(pid)
        order.append((pid, item))

    if not wanted:
        return []
    unique = list(dict.fromkeys(wanted))
    fetched: dict[str, Record] = {}
    try:
        many = getattr(store, "get_many", None)
        if callable(many):
            fetched = dict(many(unique))
        else:
            # MemoryStore (tests, fixture serving) only exposes single lookups.
            for pid in unique:
                record = store.get(pid)
                if record is not None:
                    fetched[pid] = record
    except Exception as exc:  # noqa: BLE001 - a row must never 500 Stremio
        log.warning("catalog join failed (%s); serving an empty row",
                    type(exc).__name__)
        return []

    out: list[Record] = []
    seen: set[str] = set()
    for pid, _item in order:
        record = fetched.get(pid)
        if record is None or record.point_id in seen:
            continue
        seen.add(record.point_id)
        out.append(record)
    return out


class RecommendationCache:
    """Process-wide cache for Trakt recommendation/related lists.

    One per app (built in make_app), not per request. Mirrors the singleflight
    pattern already used for the Trakt history sync: concurrent row requests share
    one in-flight fetch instead of stampeding Trakt.
    """

    def __init__(self) -> None:
        self._lists: dict[tuple[str, str], CachedList] = {}

    def get(self, media_type: str, kind: str) -> CachedList | None:
        return self._lists.get((media_type, kind))

    def fresh(self, media_type: str, kind: str,
              ttl: float = RECOMMENDATIONS_TTL) -> list[dict[str, Any]] | None:
        """Items if the entry is inside its TTL, else None."""
        entry = self._lists.get((media_type, kind))
        if entry is None or not entry.ok:
            return None
        if time.monotonic() - entry.at > ttl:
            return None
        return entry.items

    def stale(self, media_type: str, kind: str) -> list[dict[str, Any]] | None:
        """Last good items regardless of TTL, but not older than MAX_STALE."""
        entry = self._lists.get((media_type, kind))
        if entry is None or not entry.ok:
            return None
        if time.monotonic() - entry.at > RECOMMENDATIONS_MAX_STALE:
            return None
        return entry.items

    def put(self, media_type: str, kind: str, items: Sequence[dict[str, Any]]) -> None:
        self._lists[(media_type, kind)] = CachedList(items=list(items),
                                                     at=time.monotonic(), ok=True)


# ------------------------------------------------------------------ rendering
TMDB_IMG = "https://image.tmdb.org/t/p"


def release_info(payload: dict[str, Any]) -> str:
    """`2024-2026` for a finished series, `2024-` for a running one, `2010` for a film."""
    year = payload.get("year")
    if payload.get("media_type") == "series":
        end = payload.get("end_year")
        return f"{year}-{end}" if end else f"{year}-"
    return str(year) if year else ""


def to_meta(record: Record) -> dict[str, Any]:
    """One catalog record -> Stremio meta preview (PLAN §10.4).

    Same shape as rows.to_metas produces, but takes a Record so the recommendation
    path never needs a ranked Candidate.
    """
    p = record.payload
    meta = {
        "id": p.get("imdb_id"),
        "type": p.get("media_type"),
        "name": p.get("title"),
        "poster": f"{TMDB_IMG}/w500{p['poster']}" if p.get("poster") else None,
        "posterShape": "poster",
        "background": f"{TMDB_IMG}/w780{p['backdrop']}" if p.get("backdrop") else None,
        "releaseInfo": release_info(p),
        "genres": p.get("genres") or [],
        "description": p.get("overview_short") or "",
        "imdbRating": (f"{p.get('vote_average', 0):.1f}" if p.get("vote_count") else None),
    }
    return {k: v for k, v in meta.items() if v not in (None, [])}


def page(metas: Sequence[dict[str, Any]], skip: int, limit: int) -> list[dict[str, Any]]:
    return list(metas[skip: skip + limit])
