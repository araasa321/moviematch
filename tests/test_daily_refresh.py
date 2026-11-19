"""Daily vs weekly sync split, and the request-count fixes behind it.

The previous nightly job could never finish on a ~43k-point catalog:

  * popularity was grouped on the EXACT float value, so every group held one
    title and the "batching" degenerated into ~43k requests at ~370 ms each;
  * the startup scroll pulled full payloads just to learn which ids existed;
  * the vote/bayes loop sent one request per point.

Together these exceeded the 45-minute workflow timeout, so the job was killed
before it added a single title. These tests pin the fixes and the split.
"""

from __future__ import annotations

import argparse
import asyncio

import pytest

from addon.ids import point_id
from jobs import sync_tmdb
from jobs.offline_run import FakeEmbedder, FakeQdrant, FakeTMDB


def _run(coro):
    return asyncio.run(coro)


class CountingQdrant(FakeQdrant):
    """FakeQdrant that records how many payload WRITE calls were made."""

    def __init__(self) -> None:
        super().__init__()
        self.payload_calls = 0
        self.payload_batches = 0
        self.scroll_fields: list = []

    async def set_payload(self, payload, ids, **kw):
        self.payload_calls += 1
        return await super().set_payload(payload, ids, **kw)

    async def set_payload_batch(self, updates, **kw):
        ops = list(updates)
        self.payload_batches += 1
        self.payload_calls += len(ops)
        for payload, ids in ops:
            await super().set_payload(payload, ids, **kw)
        return sum(len(ids) for _, ids in ops)

    async def iter_all_payloads(self, *, fields=None, **kw):
        self.scroll_fields.append(fields)
        async for point in super().iter_all_payloads(**kw):
            yield point


class FakeTMDBWithExport(FakeTMDB):
    """FakeTMDB plus a daily export whose popularity values are near-unique."""

    def __init__(self, export_ids: int = 50) -> None:
        super().__init__()
        self.export_ids = export_ids
        self.discover_calls: list[dict] = []
        self.list_calls: list[str] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def discover(self, media_type, page, **kwargs):
        self.discover_calls.append({"media_type": media_type, "page": page, **kwargs})
        return {"page": page, "total_pages": 1, "results": []}

    async def iter_discover(self, media_type, *, max_pages=500, **kw):
        self.discover_calls.append({"iter": True, "media_type": media_type, **kw})
        for _ in ():
            yield _

    async def details(self, media_type, tmdb_id):
        base = {"id": tmdb_id, "original_language": "en", "vote_average": 7.0,
                "vote_count": 500, "popularity": 10.0, "overview": "x",
                "tagline": "", "poster_path": "/p.jpg", "backdrop_path": "/b.jpg",
                "genres": [{"id": 18, "name": "Drama"}],
                "keywords": {"keywords": []},
                "external_ids": {"imdb_id": f"tt{1000000 + tmdb_id}"},
                "belongs_to_collection": None}
        if media_type == "series":
            base |= {"name": f"S{tmdb_id}", "first_air_date": "2020-01-01",
                     "last_air_date": "2021-01-01", "created_by": [{"name": "D"}],
                     "aggregate_credits": {"cast": [], "crew": []},
                     "keywords": {"results": []}}
        else:
            base |= {"title": f"M{tmdb_id}", "release_date": "2020-01-01",
                     "credits": {"cast": [], "crew": []}}
        return base


def _export_rows(n: int) -> list[dict]:
    # Popularity values differ in the 4th decimal, exactly like TMDB: this is what
    # defeated exact-float grouping.
    return [{"id": 1000 + i, "popularity": 20.0 + i / 10000.0} for i in range(n)]


@pytest.fixture()
def wired(monkeypatch):
    def build(store: CountingQdrant, client: FakeTMDBWithExport):
        async def fake_export(media_type, day, **kw):
            return _export_rows(client.export_ids)
        monkeypatch.setattr(sync_tmdb, "download_export", fake_export)
        monkeypatch.setattr(sync_tmdb, "TMDBClient", lambda *a, **k: client)
        monkeypatch.setattr(sync_tmdb, "QdrantClient", lambda *a, **k: store)
        monkeypatch.setattr(sync_tmdb, "Embedder", lambda *a, **k: FakeEmbedder())
    return build


def _args(**over):
    base = dict(
        mode="daily", types=["movie"], limit=None, new_cap=50,
        active_pages=1, max_pages=1, rate=1000.0, concurrency=4, batch=8, model="x",
        recreate=False, dry_run=False, stats_out=None,
        tmdb_token="t", qdrant_url="http://x", qdrant_key="k",
        decades=[2020], year_from=None, year_to=None, recent_only=False,
        no_recent_pass=True, min_votes=None, sort_by="vote_count.desc",
        langs=["en"], lists=None, recent_days=0, recent_pages=1,
        recent_langs=["en"],
    )
    base.update(over)
    return argparse.Namespace(**base)


# ------------------------------------------------- popularity grouping
def test_popularity_writes_are_batched_not_per_title(wired):
    """The regression: 200 near-unique popularity values must not mean 200 calls."""
    store, client = CountingQdrant(), FakeTMDBWithExport(export_ids=200)
    # pretend every export id is already ours
    for i in range(200):
        store.points[point_id("movie", 1000 + i)] = {
            "id": point_id("movie", 1000 + i), "payload": {"popularity": 1.0},
            "vector": []}

    wired(store, client)
    stats = _run(sync_tmdb.daily(_args(recent_days=0)))

    assert stats.payload_updates == 200
    # 200 titles, but rounded to 1dp they collapse into ~2 groups -> 1 batch call
    assert store.payload_batches == 1, f"{store.payload_batches} batch calls"
    assert store.payload_calls <= 5, (
        f"{store.payload_calls} payload ops — grouping is not collapsing values")


def test_exact_float_grouping_would_have_failed(wired):
    """Control: shows why rounding matters, using the old key directly."""
    values = [20.0 + i / 10000.0 for i in range(200)]
    exact = {v: [] for v in values}
    rounded = {}
    for v in values:
        rounded.setdefault(round(v, 1), []).append(v)
    assert len(exact) == 200          # one group per title
    assert len(rounded) == 1          # all collapse together


# ------------------------------------------------- scroll fields
def test_daily_scrolls_only_popularity(wired):
    store, client = CountingQdrant(), FakeTMDBWithExport(export_ids=5)
    wired(store, client)
    _run(sync_tmdb.daily(_args(recent_days=0)))
    assert store.scroll_fields, "collection was never scrolled"
    assert ["popularity"] in store.scroll_fields, store.scroll_fields


# ------------------------------------------------- mode dispatch
def test_nightly_is_an_alias_for_daily(monkeypatch):
    called: list[str] = []

    async def daily(args):
        called.append("daily")
        return sync_tmdb.Stats()

    async def refresh(args):
        called.append("refresh")
        return sync_tmdb.Stats()

    monkeypatch.setattr(sync_tmdb, "daily", daily)
    monkeypatch.setattr(sync_tmdb, "refresh", refresh)

    _run(sync_tmdb.run(_args(mode="nightly")))
    _run(sync_tmdb.run(_args(mode="daily")))
    _run(sync_tmdb.run(_args(mode="refresh")))
    assert called == ["daily", "daily", "refresh"], called


def test_daily_does_not_touch_votes_or_bayes(wired):
    """Daily must stay cheap: no active-set vote sweep at all."""
    store, client = CountingQdrant(), FakeTMDBWithExport(export_ids=3)
    wired(store, client)
    _run(sync_tmdb.daily(_args(recent_days=0)))
    # the active-set sweep is the only caller using popularity.desc
    sorts = [c.get("sort_by") for c in client.discover_calls]
    assert "popularity.desc" not in sorts, sorts


def test_refresh_does_not_add_new_titles(wired):
    """Refresh only maintains; adding belongs to the daily job."""
    store, client = CountingQdrant(), FakeTMDBWithExport(export_ids=3)
    wired(store, client)
    stats = _run(sync_tmdb.refresh(_args(mode="refresh")))
    assert stats.upserted == 0
    assert not store.points


# ------------------------------------------------- recent-release pass
def test_daily_runs_recent_pass_per_language_plus_all(wired):
    store, client = CountingQdrant(), FakeTMDBWithExport(export_ids=0)
    wired(store, client)
    _run(sync_tmdb.daily(_args(recent_days=3, recent_langs=["en", "ml", "hi"])))

    recent = [c for c in client.discover_calls if "release_date" in str(c)]
    langs = [c.get("with_original_language") for c in recent]
    assert "en" in langs and "ml" in langs and "hi" in langs
    assert None in langs, "the all-languages pass is missing"


def test_recent_pass_uses_the_configured_window(wired):
    store, client = CountingQdrant(), FakeTMDBWithExport(export_ids=0)
    wired(store, client)
    _run(sync_tmdb.daily(_args(recent_days=3, recent_langs=["en"])))

    recent = [c for c in client.discover_calls
              if any(k.endswith("release_date.gte") or k.endswith("air_date.gte")
                     for k in c)]
    assert recent, client.discover_calls
    # 3 days back from today; just assert it is a real ISO date
    gte = recent[0]["primary_release_date.gte"]
    assert len(gte) == 10 and gte[4] == "-" and gte[7] == "-"


def test_recent_days_zero_disables_the_pass(wired):
    store, client = CountingQdrant(), FakeTMDBWithExport(export_ids=0)
    wired(store, client)
    _run(sync_tmdb.daily(_args(recent_days=0)))
    assert not [c for c in client.discover_calls
                if any(k.endswith("release_date.gte") for k in c)]


def test_recent_langs_default_is_the_curated_set():
    assert set(sync_tmdb.DEFAULT_RECENT_LANGS) == {"en", "ml", "hi", "ta", "ko"}


# ------------------------------------------------- stats
def test_stats_line_reports_recent_found():
    s = sync_tmdb.Stats()
    s.recent_found = 42
    assert "recent=42" in s.line()
