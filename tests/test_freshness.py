"""Instant refresh: marking watched on trakt.tv must reach Stremio within ~a minute.

On trakt.tv, marking something watched changes the recommendation list immediately
(the watched title drops out, a new one appears). A plain 6-hour cache cannot
mirror that, so freshness works at two levels:

  * `/sync/last_activities` is polled at most once per ACTIVITY_TTL — a few hundred
    bytes that changes as soon as watched/rated/watchlisted/hidden does,
  * a changed fingerprint drops every cached list AND the history cache, so the
    watched title disappears immediately even before the rec list is refetched,
  * the Stremio response carries max-age=60 instead of 900 so the client cannot
    hold a stale copy for long.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from addon.recommendations import (ACTIVITY_TTL, RECOMMENDATIONS_TTL,
                                   RecommendationCache, activities_fingerprint)


BASE_ACTIVITIES = {
    "all": "2026-11-22T10:00:00.000Z",
    "movies": {"watched_at": "2026-11-22T09:00:00.000Z",
               "rated_at": "2026-11-20T09:00:00.000Z",
               "watchlisted_at": "2026-11-19T09:00:00.000Z",
               "hidden_at": "2026-11-18T09:00:00.000Z"},
    "episodes": {"watched_at": "2026-11-21T09:00:00.000Z"},
    "shows": {"rated_at": "2026-11-17T09:00:00.000Z",
              "watchlisted_at": "2026-11-16T09:00:00.000Z"},
}


# ------------------------------------------------------------- fingerprint
def test_fingerprint_is_stable_for_identical_activity():
    assert activities_fingerprint(BASE_ACTIVITIES) == activities_fingerprint(
        {**BASE_ACTIVITIES})


def test_watching_a_movie_changes_the_fingerprint():
    after = {**BASE_ACTIVITIES,
             "movies": {**BASE_ACTIVITIES["movies"],
                        "watched_at": "2026-11-22T10:30:00.000Z"}}
    assert activities_fingerprint(after) != activities_fingerprint(BASE_ACTIVITIES)


def test_watchlisting_changes_the_fingerprint():
    after = {**BASE_ACTIVITIES,
             "movies": {**BASE_ACTIVITIES["movies"],
                        "watchlisted_at": "2026-11-22T10:30:00.000Z"}}
    assert activities_fingerprint(after) != activities_fingerprint(BASE_ACTIVITIES)


def test_hiding_changes_the_fingerprint():
    after = {**BASE_ACTIVITIES,
             "movies": {**BASE_ACTIVITIES["movies"],
                        "hidden_at": "2026-11-22T10:30:00.000Z"}}
    assert activities_fingerprint(after) != activities_fingerprint(BASE_ACTIVITIES)


def test_episode_progress_counts_as_a_change():
    """Episode watches retrain the model even though they are not movie events."""
    after = {**BASE_ACTIVITIES,
             "episodes": {"watched_at": "2026-11-22T10:30:00.000Z"}}
    assert activities_fingerprint(after) != activities_fingerprint(BASE_ACTIVITIES)


def test_fingerprint_tolerates_a_missing_or_partial_response():
    assert activities_fingerprint({}) == activities_fingerprint({})
    assert isinstance(activities_fingerprint({"movies": None}), str)


# ----------------------------------------------------------------- the cache
def test_changed_activity_drops_every_cached_list():
    cache = RecommendationCache()
    cache.put("movie", "recommendations", [{"ids": {"tmdb": 1}}])
    cache.put("series", "recommendations", [{"ids": {"tmdb": 2}}])
    cache.note_activities(activities_fingerprint(BASE_ACTIVITIES))
    assert cache.fresh("movie", "recommendations") is not None

    changed = {**BASE_ACTIVITIES,
               "movies": {**BASE_ACTIVITIES["movies"],
                          "watched_at": "2026-11-22T11:00:00.000Z"}}
    dropped = cache.note_activities(activities_fingerprint(changed))

    assert dropped is True
    assert cache.fresh("movie", "recommendations") is None
    assert cache.fresh("series", "recommendations") is None


def test_unchanged_activity_keeps_the_cache():
    cache = RecommendationCache()
    cache.put("movie", "recommendations", [{"ids": {"tmdb": 1}}])
    cache.note_activities(activities_fingerprint(BASE_ACTIVITIES))
    dropped = cache.note_activities(activities_fingerprint(BASE_ACTIVITIES))
    assert dropped is False
    assert cache.fresh("movie", "recommendations") is not None


def test_first_probe_never_drops_anything():
    """With no previous fingerprint there is nothing to compare against."""
    cache = RecommendationCache()
    cache.put("movie", "recommendations", [{"ids": {"tmdb": 1}}])
    assert cache.note_activities(activities_fingerprint(BASE_ACTIVITIES)) is False
    assert cache.fresh("movie", "recommendations") is not None


def test_activity_probe_is_rate_limited():
    """One tiny GET per ACTIVITY_TTL, not one per row request."""
    cache = RecommendationCache()
    assert cache.needs_activity_check() is True
    cache.note_activities(activities_fingerprint(BASE_ACTIVITIES))
    assert cache.needs_activity_check() is False

    # Pretend ACTIVITY_TTL elapsed.
    cache._activities_at = time.monotonic() - ACTIVITY_TTL - 1
    assert cache.needs_activity_check() is True


def test_list_ttl_is_the_backstop():
    """Even with no activity change the list is refetched after its TTL."""
    cache = RecommendationCache()
    cache.put("movie", "recommendations", [{"ids": {"tmdb": 1}}])
    assert cache.fresh("movie", "recommendations", ttl=RECOMMENDATIONS_TTL) is not None
    assert cache.fresh("movie", "recommendations", ttl=0.0) is None


def test_stale_serve_survives_an_outage():
    """A Trakt failure serves the last good list rather than an empty row."""
    cache = RecommendationCache()
    cache.put("movie", "recommendations", [{"ids": {"tmdb": 1}}])
    cache._lists[("movie", "recommendations")].at = time.monotonic()
    assert cache.stale("movie", "recommendations") == [{"ids": {"tmdb": 1}}]


# --------------------------------------------------- end-to-end invalidation
def _app_with_fake_trakt(store):
    """App whose Trakt client returns a mutable, inspectable recommendation set."""
    import httpx

    from addon.config import Config
    from addon.main import make_app

    state = {"watched_at": "2026-11-22T09:00:00Z",
             "recs": [{"ids": {"tmdb": 27205}}],
             "rec_calls": 0}

    class FakeTrakt:
        class config:
            configured = True

        async def last_activities(self):
            return {"all": state["watched_at"],
                    "movies": {"watched_at": state["watched_at"]}}

        async def recommendations(self, media_type, **kw):
            state["rec_calls"] += 1
            return state["recs"]

        async def related(self, media_type, ids, **kw):
            return []

        async def fetch_user_state(self):
            return {"watched": [], "ratings": [], "watchlist": []}

    app = make_app(store=store, history_provider=None, trakt=FakeTrakt())
    blob = Config(secret="unit-test-secret", langs=("en",)).encode()

    async def fetch() -> list[str]:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as ac:
            r = await ac.get(f"/{blob}/catalog/movie/foryou.json")
        return [m["name"] for m in r.json()["metas"]]

    return app, state, fetch


def test_marking_watched_on_trakt_refreshes_the_row(store):
    """The whole point: mark watched on trakt.tv -> Stremio shows the new pick.

    Regression for a real ordering bug: the activity probe used to sit inside the
    cache-MISS path, so a 6-hour-fresh list was never revalidated and the row never
    updated.
    """
    app, state, fetch = _app_with_fake_trakt(store)

    assert asyncio.run(fetch()) == ["Inception"]
    assert asyncio.run(fetch()) == ["Inception"]
    assert state["rec_calls"] == 1, "the second request should hit the cache"

    # User marks something watched on trakt.tv: activity bumps, recs change.
    state["watched_at"] = "2026-11-22T10:00:00Z"
    state["recs"] = [{"ids": {"tmdb": 155}}]
    # ACTIVITY_TTL=60s; pretend it elapsed.
    app.state.rec_cache._activities_at = 0.0

    assert asyncio.run(fetch()) == ["The Dark Knight"], "row did not refresh"
    assert state["rec_calls"] == 2, "recommendations were not refetched"


def test_row_response_is_cacheable_for_only_a_minute(store):
    """A long max-age would freeze the client copy for longer than the server TTL."""
    import httpx

    from addon.config import Config

    app, _state, _fetch = _app_with_fake_trakt(store)
    blob = Config(secret="unit-test-secret", langs=("en",)).encode()

    async def go():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as ac:
            return await ac.get(f"/{blob}/catalog/movie/foryou.json")

    assert asyncio.run(go()).headers["cache-control"] == "max-age=60"
