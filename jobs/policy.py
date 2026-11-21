"""Catalog inclusion policy + quality scores (PLAN §5, §9.2).

Kept dependency-free so the bootstrap/nightly job and the tests can share it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

MOVIE_MIN_VOTES = 100
SERIES_MIN_VOTES = 50
# Non-English titles have far fewer votes on TMDB because votes accrue mostly on
# the en-US side. Keeping the English floor for them silently discards good
# foreign films, so a foreign title qualifies on a much lower bar plus recency.
FOREIGN_MOVIE_MIN_VOTES = 20
FOREIGN_SERIES_MIN_VOTES = 10
RECENT_MONTHS = 24

BAYES_M = {"movie": 1000.0, "series": 300.0}
BAYES_M_GEMS = 300.0
CATALOG_MEAN_VOTE = 6.4

# --- stratified bayes (#4b) ---------------------------------------------------
# One (C, m) pair for the whole catalog is wrong twice over:
#   * movies and series have different vote distributions and different m;
#   * non-English titles accumulate votes on the en-US side far more slowly, so
#     held to an English prior they look mediocre and never surface.
# So the prior is computed per (media_type, "en" | "xx") stratum.
#
# Language is bucketed rather than kept per-code on purpose: a 40k catalog has no
# statistical support for, say, a "ml" stratum, and per-language m would over-fit
# thin populations. The bias being corrected is en vs non-en.
STRATUM_MIN_SAMPLE = 100      # below this a stratum's own C is noise
SHRINK_PSEUDO_COUNT = 30       # pseudo-items pulling a thin stratum's C toward global
PRIOR_FRACTION = 0.10          # m = PRIOR_FRACTION * median(vote_count) of the stratum
GEMS_M_FRACTION = 0.25         # m_gems = max(10, GEMS_M_FRACTION * m)
M_MIN, M_MAX = 20.0, 10_000.0
# Vote floor for the C sample. The floor exists to keep a "3 votes, 9.5" title out
# of the mean — NOT to re-impose the English vote bar on foreign titles, which is
# exactly the population being corrected. policy.FOREIGN_MOVIE_MIN_VOTES is 20 and
# FOREIGN_SERIES_MIN_VOTES is 10, so a blanket >=50 silently guts the xx stratum.
MIN_VOTES_FOR_MEAN = {"en": 50, "xx": 10}


def stratum_key(payload: dict[str, Any]) -> tuple[str, str]:
    """(media_type, 'en' | 'xx') — the two axes that change the vote prior."""
    media_type = str(payload.get("media_type") or "movie")
    if media_type not in ("movie", "series"):
        media_type = "movie"
    language = str(payload.get("language") or "").strip().lower()
    return media_type, ("en" if language == "en" else "xx")


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    return (ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2.0)


def bayes_stats(payloads: Iterable[dict[str, Any]]
                ) -> dict[tuple[str, str], tuple[float, float, float]]:
    """Per-stratum (C, m, m_gems).

    C      = shrinkage mean of vote_average over items with
             vote_count >= MIN_VOTES_FOR_MEAN[bucket]:
             (n*C_s + K*C_global) / (n + K)
    m      = PRIOR_FRACTION * median(vote_count) of the stratum, clamped to
             [M_MIN, M_MAX]
    m_gems = max(10.0, GEMS_M_FRACTION * m)

    Compute this over the WHOLE catalog, never over `iter_discover(max_pages=...)`.
    That sample is the top pages by popularity.desc — overwhelmingly high-vote
    English titles — and using it would push median(vote_count) up for `xx` and
    make the non-English prior worse than today.
    """
    rows = [p for p in payloads if p]
    # Global shrinkage anchor: the mean over every item that clears its own floor.
    globals_ = [float(p.get("vote_average") or 0) for p in rows
                if float(p.get("vote_count") or 0) >= MIN_VOTES_FOR_MEAN[stratum_key(p)[1]]
                and p.get("vote_average")]
    c_global = (sum(globals_) / len(globals_)) if len(globals_) >= 100 else CATALOG_MEAN_VOTE

    by_stratum: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for p in rows:
        by_stratum.setdefault(stratum_key(p), []).append(p)

    out: dict[tuple[str, str], tuple[float, float, float]] = {}
    for key, members in by_stratum.items():
        bucket = key[1]
        floor = MIN_VOTES_FOR_MEAN.get(bucket, 50)
        votes = [float(m.get("vote_average") or 0) for m in members
                 if float(m.get("vote_count") or 0) >= floor and m.get("vote_average")]
        n = len(votes)
        if n >= STRATUM_MIN_SAMPLE:
            c = sum(votes) / n
        elif n:
            # Thin stratum: shrink its own mean toward the global one.
            c = (n * (sum(votes) / n) + SHRINK_PSEUDO_COUNT * c_global) / (
                n + SHRINK_PSEUDO_COUNT)
        else:
            c = c_global
        counts = [float(m.get("vote_count") or 0) for m in members]
        m = PRIOR_FRACTION * _median(counts) if counts else 0.0
        m = min(M_MAX, max(M_MIN, m))
        # Under a stratified m, a fixed m_gems=300 would make bayes_gems barely
        # differ from bayes for foreign titles (where m is ~40). Derive it so the
        # gems signal keeps its meaning per stratum.
        m_gems = max(10.0, GEMS_M_FRACTION * m)
        out[key] = (round(c, 4), round(m, 4), round(m_gems, 4))
    return out

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


def should_index(payload: dict[str, Any], *, year_now: int, in_trakt: bool = False,
                 keep_langs: Iterable[str] | None = None,
                 curated: bool = False) -> Inclusion:
    """PLAN §5 catalog policy.

    Movies: vote_count >= 100 OR released in the last 24 months OR in Trakt.
    Series: vote_count >= 50 OR first/last air date within 24 months OR in Trakt.
    Always: must not be adult/video; must carry an imdb_id to be *shown*.

    `curated` marks a title that came from a hand-built TMDB list. Such items skip
    the vote/recency test entirely — the human who curated the list already judged
    it worth including, and the whole reason for using a list is that regional
    titles sit far below the English vote floor. Adult/video and the imdb_id flag
    still apply: a curated list can contain mistakes, and Stremio cannot render a
    title without an imdb id.

    `keep_langs` widens the vote floor for non-English titles. When a crawl
    includes foreign languages (`--langs en ja ko`) those titles have a far
    thinner en-US vote history, so holding them to the English threshold would
    discard most of what the crawl just paid TMDB quota to fetch.

    Note `keep_langs` is the set of languages being crawled, so a title's own
    language will normally BE in it. The floor therefore keys on "is this title
    English?", not on set membership.
    """
    media_type = payload.get("media_type")
    if payload.get("adult") or payload.get("video"):
        return Inclusion(False, "adult_or_video")

    if curated:
        missing = None if payload.get("imdb_id") else "imdb_id"
        return Inclusion(True, "kept_curated", missing)

    votes = int(payload.get("vote_count") or 0)
    threshold = SERIES_MIN_VOTES if media_type == "series" else MOVIE_MIN_VOTES
    if keep_langs:
        language = str(payload.get("language") or "")
        foreign_slice = any(str(code) != "en" for code in keep_langs)
        if foreign_slice and language and language != "en":
            threshold = (FOREIGN_SERIES_MIN_VOTES if media_type == "series"
                         else FOREIGN_MOVIE_MIN_VOTES)
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


def apply_quality(payload: dict[str, Any], mean: float = CATALOG_MEAN_VOTE,
                  stats: dict[tuple[str, str], tuple[float, float, float]] | None = None
                  ) -> dict[str, Any]:
    """Recompute derived quality fields. Cheap, no vector work.

    `stats` comes from `bayes_stats(...)` over the whole catalog. When it is
    None (a single-record caller, or a first boot with no catalog yet) the module
    defaults are used, which reproduces the pre-#4b behaviour exactly.

    `bayes` and `bayes_gems` are ALWAYS stamped, even when unchanged: `refresh()`
    writes them through `set_payload_batch`, which replaces those keys wholesale,
    so a missing key would silently delete the stored value.
    """
    media_type = str(payload.get("media_type") or "movie")
    avg = float(payload.get("vote_average") or 0)
    count = float(payload.get("vote_count") or 0)
    c, m, m_gems = (stats or {}).get(stratum_key(payload), (mean, 0.0, 0.0))
    if not m:
        # No stats for this stratum: fall back to the module constants, which is
        # exactly the pre-#4b behaviour.
        m = BAYES_M.get(media_type, 500.0)
        m_gems = BAYES_M_GEMS
    v = float(max(0.0, count))
    r = avg if count else 0.0
    payload["bayes"] = round((v / (v + m)) * r + (m / (v + m)) * c if v + m else c, 3)
    payload["bayes_gems"] = round(
        (v / (v + m_gems)) * r + (m_gems / (v + m_gems)) * c if v + m_gems else c, 3)
    return payload


def normalize_payload(media_type: str, details: dict[str, Any], *, year_now: int,
                      stats: dict[tuple[str, str], tuple[float, float, float]] | None = None
                      ) -> dict[str, Any]:
    """Map TMDB details (+credits/keywords/external_ids) onto PLAN §4.1 fields.

    `stats` is threaded from the caller's `bayes_stats(...)` so a newly ingested
    title is stamped with the same stratified prior as the rest of the catalog.
    Without it, `daily()` would write module defaults and #4b would never see the
    stratification for new titles.
    """
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
        # `origin_country` is present on both movie and TV details. Kept so a
        # country-scoped crawl can be verified against what was actually stored.
        "origin_country": [c for c in (details.get("origin_country") or []) if c],
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
    return apply_quality(payload, stats=stats)
