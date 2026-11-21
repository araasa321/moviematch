"""#2 / M1 — singleflight the Trakt sync, and cache empty row plans.

Two independent serving bugs, both proved against the pre-M1 code:

  Bug A — the Trakt stampede. `_catalog_response` -> `_cached_plan` -> the Trakt
    sync ran on every plan-cache miss. `build_manifest` advertises up to 12
    catalogs and Stremio fetches them together on a home-screen open, so a cold
    instance fired up to 12 concurrent full Trakt syncs (history x2, ratings x2,
    watchlist x2, each paginated up to 10 pages). There was no asyncio.Lock
    anywhere in addon/.

  Bug B — the empty-plan hole. `_cached_plan` cached only `if plan.ranked`, so
    any row that legitimately returns nothing recomputed in full on every
    request, forever.

A third hazard is covered here too: `current_history()` used to swallow every
Trakt error and return the bundled fixture history. Not caching that history is
necessary but not sufficient — the RowPlan built from it must not be cached
either, or a 1-second blip pins three fake titles for 15 minutes.

The rest of the suite drives coroutines with `asyncio.run()` and uses no async
pytest plugin, so these tests do the same rather than adding pytest-asyncio.
"""

from __future__ import annotations

import asyncio

import pytest

from addon import main as main_module
from addon.config import Config
from addon.main import EMPTY_PLAN_TTL, HISTORY_TTL, ROW_CACHE_SECONDS, make_app

from .conftest import *  # noqa: F401,F403

SECRET = "unit-test-secret"


# --------------------------------------------------------------- fakes
class _FakeTraktConfig:
    configured = True


class _FakeTrakt:
    """Counts syncs and can be made to hang or to fail."""

    def __init__(self) -> None:
        self.config = _FakeTraktConfig()
        self.calls = 0
        self.fail = False
        self.gate: asyncio.Event | None = None
        self.history = {
            "watched": [{"movie": {"ids": {"imdb": "tt0133093"}}, "plays": 1}],
            "ratings": [],
            "watchlist": [],
        }

    async def fetch_user_state(self):
        self.calls += 1
        if self.gate is not None:
            await self.gate.wait()
        if self.fail:
            raise RuntimeError("trakt is down")
        return self.history


def _blob() -> str:
    return Config(secret=SECRET, langs=("en",)).encode()


def _client(app):
    """Tiny sync client over the real ASGI app (see tests/test_configure.py)."""
    import httpx

    class _Sync:
        def get(self, path, params=None):
            async def run():
                transport = httpx.ASGITransport(app=app)
                async with httpx.AsyncClient(transport=transport,
                                             base_url="http://test") as ac:
                    return await ac.get(path, params=params)
            return asyncio.run(run())

    return _Sync()


def _catalog(app, row: str = "foryou", media_type: str = "movie"):
    return _client(app).get(f"/{_blob()}/catalog/{media_type}/{row}.json")


# ------------------------------------------------- 1. singleflight
def test_concurrent_requests_share_one_trakt_sync(store):
    """8 catalogs fetched together (one home-screen open) make exactly 1 sync."""
    trakt = _FakeTrakt()
    trakt.gate = asyncio.Event()          # hold all 8 inside _sync_history()
    app = make_app(store=store, trakt=trakt, history_provider=None)
    # make_app builds its own detector; that is fine, it never touches Trakt.

    paths = [f"/{_blob()}/catalog/movie/{row}.json"
             for row in ("foryou", "because", "new4you")]

    async def fire():
        import httpx

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport,
                                     base_url="http://test") as ac:
            tasks = [ac.get(paths[i % len(paths)]) for i in range(8)]
            pending = asyncio.ensure_future(asyncio.gather(*tasks))
            # Let all 8 enter current_history() and contend on the lock.
            for _ in range(50):
                await asyncio.sleep(0)
            trakt.gate.set()             # release the single in-flight sync
            return await asyncio.wait_for(pending, timeout=10)

    responses = asyncio.run(fire())
    assert all(r.status_code == 200 for r in responses), [r.status_code for r in responses]
    assert trakt.calls == 1, f"expected 1 shared sync, got {trakt.calls}"


def test_cache_hit_does_not_call_trakt(store):
    trakt = _FakeTrakt()
    app = make_app(store=store, trakt=trakt)
    first = _catalog(app)
    second = _catalog(app)
    assert first.status_code == 200 and second.status_code == 200
    assert trakt.calls == 1
    assert first.json() == second.json()


# ------------------------------------------------- 2. empty-plan cache
def test_empty_plan_is_cached_for_short_ttl(store, monkeypatch):
    """A row that legitimately returns nothing must stop re-running the pipeline."""
    calls = {"n": 0}
    real_build_row = main_module.build_row

    def counting(*args, **kwargs):
        calls["n"] += 1
        return real_build_row(*args, **kwargs)

    monkeypatch.setattr(main_module, "build_row", counting)
    # No history at all -> `because` row has no seed and ranks nothing.
    app = make_app(store=store, history_provider=lambda: {"watched": [], "ratings": [],
                                                          "watchlist": []})
    first = _catalog(app, row="because")
    assert first.status_code == 200
    assert calls["n"] == 1

    second = _catalog(app, row="because")
    assert second.status_code == 200
    assert calls["n"] == 1, "empty plan was recomputed instead of served from cache"

    # And the cached entry was given the SHORT ttl, not the 15-minute one.
    ttls = [entry[2] for entry in app.state.row_cache.values()]
    assert ttls == [EMPTY_PLAN_TTL]
    assert EMPTY_PLAN_TTL < ROW_CACHE_SECONDS


def test_empty_plan_is_recomputed_after_ttl(store, monkeypatch):
    """Freeze monotonic time and walk past EMPTY_PLAN_TTL."""
    clock = {"now": 1000.0}
    monkeypatch.setattr(main_module.time, "monotonic", lambda: clock["now"])

    apps = []
    calls = {"n": 0}
    real_build_row = main_module.build_row

    def counting(*args, **kwargs):
        calls["n"] += 1
        return real_build_row(*args, **kwargs)

    monkeypatch.setattr(main_module, "build_row", counting)
    app = make_app(store=store, history_provider=lambda: {"watched": [], "ratings": [],
                                                          "watchlist": []})
    apps.append(app)

    _catalog(app, row="because")
    assert calls["n"] == 1
    _catalog(app, row="because")
    assert calls["n"] == 1

    clock["now"] += EMPTY_PLAN_TTL + 1.0
    _catalog(app, row="because")
    assert calls["n"] == 2, "expired empty plan was still served from cache"


def test_populated_plan_keeps_the_long_ttl(store, history):
    """The empty-plan ttl must not shorten a real plan's cache lifetime."""
    # `history` is the bundled fixture history; it keys on TMDB ids, so it does
    # produce a non-empty `foryou` row (a hand-written imdb-id history resolves
    # to nothing and reports `cold_start`).
    app = make_app(store=store, history_provider=lambda: history)
    body = _catalog(app, row="foryou").json()
    assert body["metas"], "fixture history should produce a non-empty row"
    ttls = [entry[2] for entry in app.state.row_cache.values()]
    assert ttls and all(t == ROW_CACHE_SECONDS for t in ttls), ttls


def _history():
    """A history that actually resolves against the fixture catalog.

    The fixture catalog keys on TMDB ids (see tests/conftest.py `history`), so a
    hand-written imdb-id history resolves to nothing and silently reports
    `cold_start` — which would make the provider tests below pass vacuously.
    """
    from addon.fixtures import trakt_history

    return trakt_history()


# ------------------------------------- 3. fallback is never cached (the §4 fix)
def test_fixture_fallback_is_not_cached(store):
    """A failed sync serves the fixture history but must not store it."""
    trakt = _FakeTrakt()
    trakt.fail = True
    app = make_app(store=store, trakt=trakt)
    response = _catalog(app, row="foryou")
    assert response.status_code == 200, "a Trakt outage must not 500 Stremio"
    assert app.state.history_cache == {}, "fixture history was cached"
    assert trakt.calls == 1


def test_fallback_plan_is_not_written_to_row_cache(store):
    """The RowPlan built from fixture history must not be cached either.

    Not caching the history is necessary but not sufficient: _cached_plan would
    still store the plan built from it for ROW_CACHE_SECONDS, showing the user
    three fake titles for 15 minutes.
    """
    trakt = _FakeTrakt()
    trakt.fail = True
    app = make_app(store=store, trakt=trakt)

    first = _catalog(app, row="foryou")
    assert first.status_code == 200
    assert app.state.row_cache == {}, (
        f"fallback plan was cached: {sorted(app.state.row_cache)}")

    # Trakt recovers: the very next request must build a REAL plan.
    trakt.fail = False
    second = _catalog(app, row="foryou")
    assert second.status_code == 200
    assert trakt.calls == 2, "the recovering request did not re-sync"
    assert app.state.row_cache, "a real plan was not cached after recovery"


def test_fallback_rows_do_not_reuse_the_stale_fixture_plan(store):
    """Two consecutive failed-sync requests must both hit the real pipeline."""
    trakt = _FakeTrakt()
    trakt.fail = True
    app = make_app(store=store, trakt=trakt)
    _catalog(app, row="foryou")
    assert app.state.row_cache == {}
    _catalog(app, row="foryou")
    # The second request re-entered the pipeline (the lock let it through
    # because nothing was cached) rather than serving a memoised fake.
    assert app.state.row_cache == {}


# ------------------------------------------------- 4. provider still wins
def test_history_provider_still_wins(store):
    """The 5 tests that inject `history_provider` must keep working, untouched."""
    trakt = _FakeTrakt()
    called = {"n": 0}

    def provider():
        called["n"] += 1
        return _history()

    app = make_app(store=store, trakt=trakt, history_provider=provider)
    response = _catalog(app, row="foryou")
    assert response.status_code == 200
    # Guard against a vacuous pass: the provider history must produce a real row,
    # otherwise "the provider was used" says nothing about the fallback flag.
    assert response.json()["metas"], "provider history produced an empty row"
    assert called["n"] >= 1
    assert trakt.calls == 0, "an injected provider must stop the Trakt call"
    assert app.state.history_cache == {}, "provider history must not be cached"


def test_provider_path_reports_not_fallback():
    """/health must not claim fallback when a provider is supplying real data."""
    import httpx

    app = make_app(history_provider=lambda: _history())
    async def run():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport,
                                     base_url="http://test") as ac:
            return await ac.get("/health")
    body = asyncio.run(run()).json()
    assert body["history_is_fallback"] is False
    assert body["history_cached"] is False


# ------------------------------------------------- 5. /health surface
def test_health_exposes_history_state(store):
    import httpx

    trakt = _FakeTrakt()
    app = make_app(store=store, trakt=trakt)

    async def get_health():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport,
                                     base_url="http://test") as ac:
            return await ac.get("/health")

    before = asyncio.run(get_health()).json()
    assert before["history_cached"] is False
    _catalog(app, row="foryou")
    after = asyncio.run(get_health()).json()
    assert after["history_cached"] is True
    assert after["history_is_fallback"] is False


def test_health_reports_fallback_after_a_failed_sync(store):
    import httpx

    trakt = _FakeTrakt()
    trakt.fail = True
    app = make_app(store=store, trakt=trakt)
    _catalog(app, row="foryou")

    async def get_health():
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport,
                                     base_url="http://test") as ac:
            return await ac.get("/health")

    body = asyncio.run(get_health()).json()
    assert body["history_is_fallback"] is True
    assert body["history_cached"] is False


def test_history_ttl_is_shorter_than_the_row_cache():
    """A cached history must not outlive one row-cache cycle by much."""
    assert HISTORY_TTL <= ROW_CACHE_SECONDS
