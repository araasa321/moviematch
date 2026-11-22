"""#6 / M8 — a missing TMDB export must fail the run, loudly.

`jobs/tmdb.download_export` already tried 3 prior days and raised TMDBError. The
failure was swallowed at the CALL SITES:

  sync_tmdb.py (daily)   `except Exception ... log.warning(...); continue`
  sync_tmdb.py (refresh) the same

So a permanent 403 from files.tmdb.org produced a GREEN workflow that added
nothing. These tests pin the corrected behaviour.
"""

from __future__ import annotations

import argparse
import asyncio
import gzip
import json

import pytest

from addon.ids import point_id
from jobs import sync_tmdb
from jobs.tmdb import TMDBError, download_export

from .test_daily_refresh import CountingQdrant, FakeEmbedder, FakeTMDBWithExport


def _run(coro):
    return asyncio.run(coro)


def _args(**over) -> argparse.Namespace:
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


@pytest.fixture()
def wire(monkeypatch):
    """Wire sync_tmdb to fakes, leaving download_export to the test."""

    def build(store, client, export):
        monkeypatch.setattr(sync_tmdb, "download_export", export)
        monkeypatch.setattr(sync_tmdb, "TMDBClient", lambda *a, **k: client)
        monkeypatch.setattr(sync_tmdb, "QdrantClient", lambda *a, **k: store)
        monkeypatch.setattr(sync_tmdb, "Embedder", lambda *a, **k: FakeEmbedder())
    return build


# ------------------------------------------- daily must fail loudly
def test_daily_fails_when_export_unavailable(wire):
    """Every type is attempted, THEN the run fails — not a silent continue."""
    attempted: list[str] = []

    async def boom(media_type, day, **kw):
        attempted.append(media_type)
        raise TMDBError("403 permanent")

    store, client = CountingQdrant(), FakeTMDBWithExport(export_ids=5)
    wire(store, client, boom)
    with pytest.raises(SystemExit) as exc:
        _run(sync_tmdb.daily(_args(types=["movie", "series"])))
    assert "movie" in str(exc.value) and "series" in str(exc.value)
    # BOTH types were tried before failing: one broken export must not mask another.
    assert attempted == ["movie", "series"]


def test_daily_succeeds_when_every_type_exports(wire):
    async def ok(media_type, day, **kw):
        return [{"id": 1000 + i, "popularity": 20.0} for i in range(3)]

    store, client = CountingQdrant(), FakeTMDBWithExport(export_ids=5)
    wire(store, client, ok)
    stats = _run(sync_tmdb.daily(_args(types=["movie", "series"])))
    assert stats is not None


def test_daily_fails_when_one_type_is_missing(wire):
    """One type raising must fail the run (today's behaviour silently succeeded)."""

    async def half(media_type, day, **kw):
        if media_type == "series":
            raise TMDBError("404 for every day")
        return [{"id": 1000, "popularity": 20.0}]

    store, client = CountingQdrant(), FakeTMDBWithExport(export_ids=5)
    wire(store, client, half)
    with pytest.raises(SystemExit) as exc:
        _run(sync_tmdb.daily(_args(types=["movie", "series"])))
    assert "series" in str(exc.value)
    assert "movie" not in str(exc.value)


def test_refresh_also_fails_when_export_unavailable(wire):
    """The weekly sweep must not report success after adding nothing."""

    async def boom(media_type, day, **kw):
        raise TMDBError("500 from the CDN")

    store, client = CountingQdrant(), FakeTMDBWithExport(export_ids=5)
    wire(store, client, boom)
    with pytest.raises(SystemExit) as exc:
        _run(sync_tmdb.refresh(_args(mode="refresh", types=["movie"])))
    assert "movie" in str(exc.value)


def test_dry_run_never_reaches_daily(monkeypatch):
    """--dry-run short-circuits to the offline self-check, so M8 cannot break it."""
    args = sync_tmdb.parse_args(["--mode", "daily", "--dry-run"])
    stats = _run(sync_tmdb.run(args))
    # The offline path upserts into the fake catalog and never needs an export.
    assert stats.upserted > 0


# ------------------------------------------- download_export fallback behaviour
class _Resp:
    def __init__(self, status_code: int, body: bytes = b"") -> None:
        self.status_code = status_code
        self.content = body


class _FakeAsyncClient:
    """Feeds a scripted list of responses, one per GET."""

    def __init__(self, responses: list[_Resp]) -> None:
        self.responses = list(responses)
        self.urls: list[str] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def get(self, url):
        self.urls.append(url)
        return self.responses.pop(0)


def _gz(rows: list[dict]) -> bytes:
    return gzip.compress("\n".join(json.dumps(r) for r in rows).encode())


def _patch_client(monkeypatch, responses: list[_Resp]) -> _FakeAsyncClient:
    fake = _FakeAsyncClient(responses)
    monkeypatch.setattr("jobs.tmdb.httpx.AsyncClient", lambda *a, **k: fake)
    return fake


def test_download_export_falls_back_on_403(monkeypatch):
    """A 403 on today's file must not abort: the previous day may be fine."""
    rows = [{"id": 1, "popularity": 1.0}]
    fake = _patch_client(monkeypatch, [_Resp(403), _Resp(200, _gz(rows))])
    out = _run(download_export("movie", "11_22_2026"))
    assert out == rows
    assert len(fake.urls) == 2, fake.urls


def test_download_export_falls_back_on_500(monkeypatch):
    fake = _patch_client(monkeypatch, [_Resp(500), _Resp(200, _gz([{"id": 2}]))])
    assert _run(download_export("movie", "11_22_2026")) == [{"id": 2}]
    assert len(fake.urls) == 2


def test_download_export_raises_after_lookback(monkeypatch):
    """All 404 -> TMDBError naming the date."""
    fake = _patch_client(monkeypatch, [_Resp(404) for _ in range(3)])
    with pytest.raises(TMDBError) as exc:
        _run(download_export("movie", "11_22_2026"))
    assert "11_22_2026" in str(exc.value)
    assert len(fake.urls) == 3


def test_download_export_error_names_the_last_status_and_url(monkeypatch):
    """A permanent 403 must be diagnosable, not look like 'not published yet'."""
    fake = _patch_client(monkeypatch, [_Resp(403) for _ in range(3)])
    with pytest.raises(TMDBError) as exc:
        _run(download_export("series", "11_22_2026"))
    message = str(exc.value)
    assert "403" in message, message
    assert "tv_series_ids" in message
    assert fake.urls[-1] in message
