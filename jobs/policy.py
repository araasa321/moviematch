"""Catalog inclusion policy + quality scores (PLAN §5, §9.2).

Kept dependency-free so the bootstrap/nightly job and the tests can share it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

MOVIE_MIN_VOTES = 100
SERIES_MIN_VOTES = 50
RECENT_MONTHS = 24

BAYES_M = {"movie": 1000.0, "series": 300.0}
BAYES_M_GEMS = 300.0
CATALOG_MEAN_VOTE = 6.4

GENRE_NAMES = {
    28: "Action", 12: "Adventure", 16: "Animation", 35: "Comedy", 80: "Crime",
    99: "Documentary", 18: "Drama", 10751: "Family", 14: "Fantasy", 36: "History",
    27: "Horror", 10402: "Music", 9648: "Mystery", 10749: "Romance",
    878: "Sci-Fi", 10770: "TV Movie", 53: "Thriller", 10752: "War", 37: "Western",
    10759: "Action & Adventure", 10762: "Kids", 10763: "News", 10764: "Reality",
    10765: "Sci-Fi & Fantasy", 10766: "Soap", 10767: "Talk", 10768: "War & Politics",
}


@dataclass(frozen=True)
class Inclusion:
    keep: bool
    reason: str
    missing: str | None = None      # set when the title is kept only *pending* enrichment


def should_index(payload: dict[str, Any], *, year_now: int, in_trakt: bool = False) -> Inclusion:
    """PLAN §5 catalog policy.

    Movies: vote_count >= 100 OR released in the last 24 months OR in Trakt.
    Series: vote_count >= 50 OR first/last air date within 24 months OR in Trakt.
    Always: must not be adult/video; must carry an imdb_id to be *shown*.
    """
    media_type = payload.get("media_type")
    if payload.get("adult") or payload.get("video"):
        return Inclusion(False, "adult_or_video")

    votes = int(payload.get("vote_count") or 0)
    threshold = SERIES_MIN_VOTES if media_type == "series" else MOVIE_MIN_VOTES
    if votes >= threshold or in_trakt or _recent(payload, year_now):
        missing = None if payload.get("imdb_id") else "imdb_id"
        return Inclusion(True, "kept", missing)
    return Inclusion(False, "low_votes")


def _recent(payload: dict[str, Any], year_now: int) -> bool:
    """Recency clause: within the last RECENT_MONTHS (approximated on year/date)."""
    for key in ("release_date", "first_air_date", "end_year", "year"):
        value = payload.get(key)
        if not value:
            continue
        year = int(str(value)[:4])
        if year >= year_now - 1:
            return True
    return False


def bayes(vote_average: float, vote_count: float, media_type: str,
          gems: bool = False, mean: float = CATALOG_MEAN_VOTE) -> float:
    """IMDb-style Bayesian rating (§9.2). Mirrors addon.rank.bayes_score."""
    m = BAYES_M_GEMS if gems else BAYES_M.get(media_type, 500.0)
    v = float(max(0.0, vote_count))
    r = float(vote_average) if vote_count else 0.0
    return (v / (v + m)) * r + (m / (v + m)) * mean if v + m else mean


def catalog_mean_vote(payloads: Iterable[dict[str, Any]]) -> float:
    """Weekly `C` estimate: mean vote_average of sampled titles with enough votes."""
    votes = [float(p.get("vote_average") or 0) for p in payloads
             if int(p.get("vote_count") or 0) >= 50 and p.get("vote_average")]
    if len(votes) < 100:
        return CATALOG_MEAN_VOTE
    return round(sum(votes) / len(votes), 2)


def apply_quality(payload: dict[str, Any], mean: float = CATALOG_MEAN_VOTE) -> dict[str, Any]:
    """Recompute derived quality fields. Cheap, no vector work."""
    media_type = str(payload.get("media_type") or "movie")
    avg = float(payload.get("vote_average") or 0)
    count = float(payload.get("vote_count") or 0)
    payload["bayes"] = round(bayes(avg, count, media_type, False, mean), 3)
    payload["bayes_gems"] = round(bayes(avg, count, media_type, True, mean), 3)
    return payload


def normalize_payload(media_type: str, details: dict[str, Any], *, year_now: int) -> dict[str, Any]:
    """Map TMDB details (+credits/keywords/external_ids) onto PLAN §4.1 fields."""
    is_series = media_type == "series"
    genre_map = details.get("genres") or []
    genres = [g.get("name") if isinstance(g, dict) else GENRE_NAMES.get(int(g), str(g))
              for g in genre_map]
    keywords_root = details.get("keywords") or {}
    keyword_list = keywords_root.get("keywords") or keywords_root.get("results") or []
    keywords = [k.get("name") for k in keyword_list if k.get("name")]

    if is_series:
        credits = (details.get("aggregate_credits") or {})
        cast = [c.get("name") for c in (credits.get("cast") or [])][:4]
        crew = (credits.get("crew") or [])
        creators = details.get("created_by") or []
        director = (creators[0].get("name") if creators
                    else next((c.get("name") for c in crew
                               if (c.get("job") or "").lower() == "executive producer"), ""))
    else:
        credits = details.get("credits") or {}
        cast = [c.get("name") for c in (credits.get("cast") or [])][:4]
        crew = credits.get("crew") or []
        director = next((c.get("name") for c in crew if c.get("job") == "Director"), "")

    external = details.get("external_ids") or {}
    collection = details.get("belongs_to_collection") or {}
    # `.get(key, "")` only defaults when the key is ABSENT. TMDB returns explicit
    # nulls (e.g. "last_air_date": null for a series with no aired finale), which
    # crashed a live bootstrap with "'NoneType' object is not subscriptable".
    release = details.get("release_date") or details.get("first_air_date") or ""
    year = int(release[:4]) if release[:4].isdigit() else None
    end_year = None
    last_air = details.get("last_air_date") or ""
    if is_series and last_air[:4].isdigit():
        end_year = int(last_air[:4])

    payload = {
        "tmdb_id": int(details.get("id") or 0),
        "media_type": media_type,
        "imdb_id": external.get("imdb_id") or details.get("imdb_id"),
        "title": details.get("title") or details.get("name") or "",
        "original_title": details.get("original_title") or details.get("original_name") or "",
        "year": year,
        "end_year": end_year,
        "tagline": details.get("tagline") or "",
        "genres": [g for g in genres if g],
        "keywords": [k for k in keywords if k][:20],
        "director": director or "",
        "cast": [c for c in cast if c],
        "language": details.get("original_language") or "",
        "runtime": details.get("runtime") or (details.get("episode_run_time") or [None])[0],
        "vote_average": float(details.get("vote_average") or 0),
        "vote_count": int(details.get("vote_count") or 0),
        "popularity": float(details.get("popularity") or 0),
        "budget": int(details.get("budget") or 0),
        "revenue": int(details.get("revenue") or 0),
        "collection_id": collection.get("id"),
        "release_date": release,
        "poster": details.get("poster_path"),
        "backdrop": details.get("backdrop_path"),
        "overview_short": (details.get("overview") or "")[:400],
        "overview": (details.get("overview") or "")[:4000],
        "adult": bool(details.get("adult")),
        "video": bool(details.get("video")),
        "enriched": True,
    }
    return apply_quality(payload)
