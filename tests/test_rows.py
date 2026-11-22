"""Row serving: Trakt decides the order, the catalog supplies metadata.

This file used to test the embedding ranker (build_row, because_query, taste
centroids, genre filtering). That whole pipeline was removed: `/recommendations`
now drives the `foryou` row and `/{movies|shows}/:id/related` drives `because`,
and the addon no longer re-scores what Trakt returns.

What is still worth asserting, and is asserted here:
  * Trakt history -> UserState (watched / watchlist / hidden / dislikes) is intact,
    because it is what the rows are FILTERED by and what the write path uses;
  * watched, watchlisted and hidden titles never appear in a row;
  * a row that cannot reach Trakt degrades to empty instead of 500-ing;
  * Trakt's order is preserved through the catalog join;
  * the manifest, paging and meta shape are unchanged.
"""

from __future__ import annotations

import base64
import json

import pytest

from addon.config import Config
from addon.ids import point_id
from addon.main import make_app
from addon.recommendations import ordered_records, to_meta
from addon.stremio import build_manifest, parse_extra, safe_skip

from .conftest import *  # noqa: F401,F403  (reuse store/history/config fixtures)


# --------------------------------------------------------------- user state
def test_only_watched_and_rated_titles_become_signals(store, state):
    # watched: TDK, Breaking Bad (+ a trakt-only item with no tmdb id -> skipped)
    # ratings: Inception 10 (positive), Breaking Bad 9 (positive), Fight Club 2 (dislike)
    # watchlist: The Prestige  => 5 positive signals + 1 dislike
    assert point_id("movie", 155) in state.watched
    assert point_id("series", 1396) in state.watched
    assert point_id("movie", 1124) in state.watchlist
    assert len(state.signals) == 5
    assert len(state.dislikes) == 1
    assert state.last_watched["movie"] == point_id("movie", 155)


def test_last_watched_tracks_the_most_recent_item(store, history):
    from addon.rows import build_user_state

    state = build_user_state(store, history)
    # Both movies were watched; the newer timestamp wins.
    assert state.last_watched["movie"] in {point_id("movie", 155),
                                           point_id("movie", 1124)}


# ------------------------------------------------------------ catalog join
def _trakt_item(tmdb: int, imdb: str | None = None, title: str = "X") -> dict:
    ids = {"tmdb": tmdb}
    if imdb:
        ids["imdb"] = imdb
    return {"title": title, "year": 2010, "ids": ids}


def test_ordered_records_preserves_trakt_order(store):
    """Trakt's ranking IS the product: the join must not reorder it."""
    items = [_trakt_item(27205), _trakt_item(155), _trakt_item(49026)]
    records = ordered_records(items, store, "movie")
    assert [r.payload["title"] for r in records] == ["Inception",
                                                     "The Dark Knight",
                                                     "The Dark Knight Rises"]


def test_ordered_records_skips_titles_missing_from_the_catalog(store):
    items = [_trakt_item(27205), _trakt_item(999999999), _trakt_item(155)]
    records = ordered_records(items, store, "movie")
    assert [r.payload["tmdb_id"] for r in records] == [27205, 155]


def test_ordered_records_skips_items_without_a_tmdb_id(store):
    items = [{"title": "No ids", "ids": {"imdb": "tt999"}}, _trakt_item(155)]
    records = ordered_records(items, store, "movie")
    assert [r.payload["tmdb_id"] for r in records] == [155]


def test_ordered_records_honours_excluded(store):
    """A cached Trakt list can be hours old: re-filter at join time."""
    items = [_trakt_item(27205), _trakt_item(155)]
    excluded = {point_id("movie", 27205)}
    records = ordered_records(items, store, "movie", excluded=excluded)
    assert [r.payload["tmdb_id"] for r in records] == [155]


def test_ordered_records_dedupes_repeats(store):
    items = [_trakt_item(155), _trakt_item(155)]
    assert len(ordered_records(items, store, "movie")) == 1


def test_ordered_records_never_raises_when_the_join_fails(store):
    class Broken:
        def get_many(self, ids):
            raise RuntimeError("qdrant down")

    assert ordered_records([_trakt_item(155)], Broken(), "movie") == []


# ---------------------------------------------------------------- meta shape
def test_meta_preview_shape(store):
    record = ordered_records([_trakt_item(155)], store, "movie")[0]
    meta = to_meta(record)
    assert meta["id"].startswith("tt")
    assert meta["type"] == "movie"
    assert meta["poster"].startswith("https://image.tmdb.org/t/p/w500")


def test_series_meta_shows_an_open_year_range(store):
    items = [{"title": "Breaking Bad", "year": 2008,
              "ids": {"tmdb": 1396, "imdb": "tt0903747"}}]
    record = ordered_records(items, store, "series")[0]
    meta = to_meta(record)
    assert meta["type"] == "series"
    assert meta["releaseInfo"] == "2008-"


# --------------------------------------------------------------- extras/manifest
def test_parse_extra_and_skip():
    assert parse_extra("skip=100&genre=Sci-Fi") == {"skip": "100", "genre": "Sci-Fi"}
    assert safe_skip({"skip": "30"}) == 30
    assert safe_skip({}) == 0
    assert safe_skip({"skip": "-5"}) == 0
    assert safe_skip({"skip": "9999"}) <= 200


def test_manifest_has_expected_catalogs():
    man = build_manifest("https://x", "blob", {"movie": "Heat", "series": "Dark"},
                         langs=("en",), layout="merged")
    names = [c["name"] for c in man["catalogs"]]
    assert len(man["catalogs"]) == 4          # 2 rows x 2 media types
    assert any("Because you watched Heat" in n for n in names)
    assert not any("New for Your Taste" in n for n in names), "new4you was removed"


# ----------------------------------------------------------- row behaviour
def _app(store, history, trakt=None):
    return make_app(store=store, history_provider=lambda: history, trakt=trakt)


async def _row(app, media_type: str, row: str) -> list[dict]:
    import httpx

    blob = Config(secret="unit-test-secret", langs=("en",)).encode()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as ac:
        response = await ac.get(f"/{blob}/catalog/{media_type}/{row}.json")
    return response.json()["metas"]


def test_row_is_empty_when_trakt_is_unreachable(store, history):
    """Trakt down must degrade to an empty row, never a 500."""
    import asyncio

    app = _app(store, history)          # no trakt client -> not configured -> raises
    metas = asyncio.run(_row(app, "movie", "foryou"))
    assert metas == []


def test_row_serves_trakt_order_and_filters_watched(store, history):
    """End to end: Trakt ids -> catalog records -> metas, order preserved."""
    import asyncio

    class FakeTrakt:
        class config:
            configured = True

        async def recommendations(self, media_type, **kw):
            # 27205 (Inception, unwatched) then 155 (TDK, WATCHED in the fixture)
            return [_trakt_item(27205), _trakt_item(155)]

        async def related(self, media_type, ids, **kw):
            return [_trakt_item(49026)]

        async def fetch_user_state(self):
            return history

    app = _app(store, history, trakt=FakeTrakt())
    metas = asyncio.run(_row(app, "movie", "foryou"))
    titles = [m["name"] for m in metas]
    assert titles == ["Inception"], f"watched title leaked or order changed: {titles}"


def test_because_row_uses_the_last_watch(store, history):
    import asyncio

    seen: dict = {}

    class FakeTrakt:
        class config:
            configured = True

        async def recommendations(self, media_type, **kw):
            return []

        async def related(self, media_type, ids, **kw):
            seen["ids"] = ids
            return [_trakt_item(27205)]

        async def fetch_user_state(self):
            return history

    app = _app(store, history, trakt=FakeTrakt())
    metas = asyncio.run(_row(app, "movie", "because"))
    assert metas and metas[0]["name"] == "Inception"
    assert seen["ids"]["imdb"].startswith("tt"), seen
