"""Row builder — the seam every layer depends on.

`build_row(store, history, media_type, row, config)` is deliberately free of
FastAPI, httpx and Qdrant so the whole pipeline can be tested offline:

    Trakt history -> signals (joined to vectors) -> taste -> guards -> rank -> metas
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, Protocol, Sequence

import numpy as np

from .config import Config
from .ids import id_from_trakt_item, point_id
from .rank import (
    BECAUSE_ANN_LIMIT,
    BECAUSE_SEED_WEIGHT,
    BECAUSE_SIM_FLOOR,
    BECAUSE_TASTE_WEIGHT,
    Candidate,
    normalize,
    rank,
)
from .taste import (
    REWATCH,
    WATCHED,
    WATCHLIST,
    Signal,
    Taste,
    compute_taste,
    median_year,
    rating_weight,
)

MIN_SIMILARITY = 0.35
TMDB_IMG = "https://image.tmdb.org/t/p"


class Store(Protocol):
    def get(self, point_id: str) -> Any: ...
    def get_all(self, media_types: str | Iterable[str] | None = None) -> Sequence[Any]: ...


@dataclass
class RowPlan:
    """Everything the ranker needs for one row, before it is sliced by skip."""
    ranked: list[Candidate]
    taste: Taste
    positive_count: int = 0
    excluded: set[str] = field(default_factory=set)
    diagnostics: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------- guards (§9.1)
def guard_reasons(payload: dict[str, Any], *, media_type: str, langs: Iterable[str],
                  year_now: int) -> str | None:
    """Return the first failed guard, or None if the title may enter the rank pool.

    `vote_count < 50` is intentionally NOT a guard here: §9.1 waives it for titles
    from last year / this year, and rank() applies that waiver centrally.
    """
    if payload.get("media_type") != media_type:
        return "media_type"
    if not payload.get("imdb_id"):
        return "no_imdb_id"
    if not payload.get("poster"):
        return "no_poster"
    if payload.get("language") not in set(langs):
        return "language"
    return None


# ------------------------------------------------------------ Trakt -> signals
@dataclass
class UserState:
    watched: set[str] = field(default_factory=set)          # point ids
    watchlist: set[str] = field(default_factory=set)
    hidden: set[str] = field(default_factory=set)
    play_counts: dict[str, int] = field(default_factory=dict)
    signals: list[Signal] = field(default_factory=list)
    dislikes: list[Signal] = field(default_factory=list)
    last_watched: dict[str, str] = field(default_factory=dict)  # media_type -> point id
    watched_years: list[int] = field(default_factory=list)


def _at(value: Any, fallback: datetime | None = None) -> datetime:
    """Parse an ISO timestamp, or return `fallback` (or "now") when it is unusable.

    A missing timestamp must never be silently treated as "just now": that would
    invent a session signal and let stale actions dominate the recent centroid.
    """
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    return fallback or datetime.now(timezone.utc)


def build_user_state(store: Store, history: dict[str, Any], *,
                     now: datetime | None = None) -> UserState:
    """Join Trakt payloads (`movie|show.ids.tmdb`) to catalog vectors.

    Items without a tmdb id, or whose tmdb id is not in the catalog yet, are
    skipped — the nightly job force-includes Trakt items, so this self-heals (§7.3).
    """
    now = now or datetime.now(timezone.utc)
    state = UserState()
    last_seen: dict[str, datetime] = {}

    watched = history.get("watched", [])
    ratings = history.get("ratings", [])
    watchlist = history.get("watchlist", [])

    # Resolve every referenced title in ONE batch. Per-item `store.get()` meant one
    # network round trip (and one thread) per watched/rated entry, which pinned a
    # 0.1-vCPU Koyeb instance at 100% CPU.
    wanted: list[str] = []
    for item in (*watched, *ratings, *watchlist):
        key = id_from_trakt_item(item, "movie")
        if key is not None:
            wanted.append(point_id(*key))
    records: dict[str, Any] = {}
    if wanted:
        fetcher = getattr(store, "get_many", None)
        if callable(fetcher):
            records = fetcher(wanted)
        else:
            records = {pid: rec for pid in wanted if (rec := store.get(pid)) is not None}

    def join(item: dict[str, Any]) -> tuple[str, Any] | None:
        key = id_from_trakt_item(item, "movie")
        if key is None:
            return None
        media_type, tmdb = key
        rec = records.get(point_id(media_type, tmdb))
        return (media_type, rec) if rec is not None else None

    for item in watched:
        joined = join(item)
        if joined is None:
            continue
        media_type, rec = joined
        state.watched.add(rec.point_id)
        plays = int(item.get("plays") or 1)
        state.play_counts[rec.point_id] = state.play_counts.get(rec.point_id, 0) + plays
        at = _at(item.get("watched_at"), now)
        if isinstance(rec.payload.get("year"), int):
            state.watched_years.append(int(rec.payload["year"]))
        weight = WATCHED + (REWATCH - WATCHED if plays >= 2 else 0.0)
        state.signals.append(Signal(rec.vector, weight, at, "rewatch" if plays >= 2 else "watched"))
        if rec.point_id not in last_seen or at > last_seen[rec.point_id]:
            last_seen[rec.point_id] = at
            state.last_watched[media_type] = rec.point_id

    for item in ratings:
        joined = join(item)
        if joined is None:
            continue
        media_type, rec = joined
        rating = int(item.get("rating") or 0)
        at = _at(item.get("rated_at"), now)
        weight = rating_weight(rating)
        if rating <= 3:
            state.dislikes.append(Signal(rec.vector, weight, at, "rating"))
        elif weight:
            state.signals.append(Signal(rec.vector, weight, at, "rating"))

    for item in watchlist:
        joined = join(item)
        if joined is None:
            continue
        _, rec = joined
        state.watchlist.add(rec.point_id)
        state.signals.append(Signal(rec.vector, WATCHLIST, _at(item.get("listed_at")), "watchlist"))

    # Hidden items (Trakt "RecSys Hidden" list, §7.5) are exclusions only: they carry
    # no timestamp and must never enter the session/recent centroid.
    state.hidden = {str(pid) for pid in history.get("hidden", [])}

    return state


# --------------------------------------------------------------------- rows
def because_seed(store: Store, state: UserState, media_type: str) -> Any | None:
    pid = state.last_watched.get(media_type)
    return store.get(pid) if pid else None


def build_row(store: Store, history: dict[str, Any], media_type: str, row: str,
              config: Config | None = None, *, now: datetime | None = None,
              year_now: int | None = None) -> RowPlan:
    config = config or Config()
    now = now or datetime.now(timezone.utc)
    year_now = year_now or now.year
    state = build_user_state(store, history, now=now)
    taste = compute_taste(state.signals, state.dislikes, now=now)

    excluded: set[str] = set(state.watched) | set(state.hidden)
    if row != "tonight":
        excluded |= state.watchlist

    seed_title = because_seed(store, state, media_type) if row == "because" else None
    ann_limit = 200
    sim_floor = 0.35
    if row == "because":
        if seed_title is None:
            return RowPlan([], taste, count_positive_signals(state), excluded, {"reason": "no_seed"})
        # A single seed's nearest neighbours are dominated by franchise clones
        # (measured, FINDINGS-EMBEDDINGS.md), so widen the net and blend in taste.
        query = because_query(seed_title, taste)
        ann_limit = BECAUSE_ANN_LIMIT
        sim_floor = BECAUSE_SIM_FLOOR
    else:
        if not taste.ready or taste.query is None:
            return RowPlan([], taste, count_positive_signals(state), excluded, {"reason": "cold_start"})
        query = taste.query

    pool = _candidate_pool(store, query, media_type=media_type, langs=config.langs,
                           excluded=excluded, year_now=year_now, ann_limit=ann_limit)

    year_seed = (seed_title.payload.get("year") if seed_title is not None
                 else median_year(state.watched_years))
    seed_director = seed_title.payload.get("director", "") if seed_title is not None else ""
    seed_cast = seed_title.payload.get("cast", []) if seed_title is not None else ()

    ranked = rank(
        query,
        pool,
        mode=config.mode,
        year_seed=year_seed,
        seed_director=seed_director,
        seed_cast=seed_cast,
        disliked=taste.dislikes,
        langs=config.langs,
        year_now=year_now,
        limit=200,
        ann_limit=ann_limit,
        sim_floor=sim_floor,
    )
    return RowPlan(ranked, taste, count_positive_signals(state), excluded,
                   {"year_seed": year_seed, "row": row, "media_type": media_type,
                    "ann_limit": ann_limit, "pool": len(pool),
                    "pool_source": "ann" if hasattr(store, "search") else "scan"})


def _candidate_pool(store: Store, query: np.ndarray, *, media_type: str,
                    langs: Sequence[str], excluded: set[str], year_now: int,
                    ann_limit: int) -> list[Candidate]:
    """Candidates for ranking.

    Large catalogs are queried with ANN (Qdrant does the distance maths and the
    §9.1 filters); the in-memory stores used by tests scan directly. Both paths
    apply `guard_reasons` so behaviour is identical.
    """
    if hasattr(store, "search"):
        try:
            records = store.search(query, media_type=media_type, limit=ann_limit,
                                   langs=langs, exclude=excluded)
        except Exception:  # noqa: BLE001 - fall back rather than fail the row
            records = []
        if records:
            return [rec.to_candidate() for rec in records
                    if rec.point_id not in excluded
                    and guard_reasons(rec.payload, media_type=media_type, langs=langs,
                                      year_now=year_now) is None]

    return [rec.to_candidate() for rec in store.get_all(media_type)
            if rec.point_id not in excluded
            and guard_reasons(rec.payload, media_type=media_type, langs=langs,
                              year_now=year_now) is None]


def because_query(seed_title: Any, taste: Taste) -> np.ndarray:
    """Blend the seed's vector with the user's taste vector.

    `because` exists to say "more like this one", so the seed dominates (0.7);
    the taste term (0.3) pulls in titles the seed alone would never reach — the
    embedding test showed a pure seed query cannot escape its own franchise.
    Falls back to the bare seed when there is no taste to blend.
    """
    seed_vec = normalize(seed_title.vector)
    if taste.query is None:
        return seed_vec
    blended = normalize(BECAUSE_SEED_WEIGHT * seed_vec + BECAUSE_TASTE_WEIGHT * taste.query)
    return blended if np.any(blended) else seed_vec


def count_positive_signals(state: UserState) -> int:
    return sum(1 for s in state.signals if s.weight > 0)


def to_metas(ranked: Sequence[Candidate], skip: int = 0, limit: int = 15) -> list[dict[str, Any]]:
    """Stremio meta preview format (§10.4)."""
    out: list[dict[str, Any]] = []
    for cand in ranked[skip: skip + limit]:
        p = cand.payload
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
            "imdbRating": f"{p.get('vote_average', 0):.1f}" if p.get("vote_count") else None,
        }
        out.append({k: v for k, v in meta.items() if v not in (None, [])})
    return out


def release_info(payload: dict[str, Any]) -> str:
    year = payload.get("year")
    if payload.get("media_type") == "series":
        end = payload.get("end_year")
        return f"{year}-{end}" if end else f"{year}-"
    return str(year) if year else ""
