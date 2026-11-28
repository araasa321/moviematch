"""Automatic Trakt token refresh.

Context: Trakt access tokens now live 7 days (they were 90 days before Mar 2025,
then 24h). The addon pasted a static token into an env var and never refreshed it,
so after 7 days EVERY Trakt call returned 401 — recommendations froze on stale
cache and Stremio -> Trakt writes silently stopped. It went unnoticed for days
because the probe failure logged at DEBUG and /health said `trakt_auth: configured`.

The risky part is rotation: Trakt refresh tokens are SINGLE-USE. Each exchange
returns a new one and invalidates the old, so a lost write or a concurrent refresh
permanently logs the user out. These tests pin exactly that.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from addon.trakt import TraktClient, TraktConfig, TraktUnauthorized
from addon.trakt_token import (REFRESH_AT_FRACTION, TokenPair, TokenStore,
                               TraktAuthError, parse_token_response)


DAY = 86400.0


# ------------------------------------------------------------------ TokenPair
def test_token_pair_without_expiry_is_treated_as_expired():
    """Unknown lifetime -> refresh. One wasted refresh beats a live 401."""
    assert TokenPair(access_token="a").expired_soon() is True


def test_token_pair_fresh_token_is_not_expired():
    now = time.time()
    pair = TokenPair(access_token="a", expires_at=now + 6 * DAY, issued_at=now)
    assert pair.expired_soon(now=now) is False


def test_token_pair_refreshes_at_eighty_percent():
    issued = time.time()
    life = 7 * DAY
    pair = TokenPair(access_token="a", expires_at=issued + life, issued_at=issued)
    # Just under 80 % elapsed -> still fine.
    assert pair.expired_soon(now=issued + 0.79 * life) is False
    # Just over 80 % elapsed -> refresh now, a day and a half before death.
    assert pair.expired_soon(now=issued + 0.81 * life) is True


def test_token_pair_expired_is_expired():
    now = time.time()
    pair = TokenPair(access_token="a", expires_at=now - 10, issued_at=now - DAY)
    assert pair.expired_soon(now=now) is True


def test_no_hardcoded_lifetime():
    """The refresh point must derive from the token, not a constant 7 days."""
    now = time.time()
    # A 24-hour token must be refreshed at ~19 h, not at 5.6 days.
    pair = TokenPair(access_token="a", expires_at=now + 0.2 * DAY, issued_at=now - 0.8 * DAY)
    assert pair.lifetime == pytest.approx(DAY)
    assert pair.expired_soon(now=now) is True
    # ...and the same point on a 7-day token is 1.4 days, i.e. NOT yet expired at 24h.
    week = TokenPair(access_token="a", expires_at=now + 6 * DAY, issued_at=now - DAY)
    assert week.lifetime == pytest.approx(7 * DAY)
    assert week.expired_soon(now=now) is False


def test_parse_token_response_uses_expires_in():
    body = {"access_token": "new", "refresh_token": "newr", "expires_in": 7 * 86400,
            "created_at": 1000, "token_type": "bearer", "scope": "public"}
    pair = parse_token_response(body)
    assert pair.access_token == "new" and pair.refresh_token == "newr"
    assert pair.expires_at == pytest.approx(1000 + 7 * 86400)
    assert pair.lifetime == pytest.approx(7 * 86400)


def test_parse_token_response_tolerates_a_short_lifetime():
    pair = parse_token_response({"access_token": "x", "expires_in": 3600,
                                 "created_at": 500})
    assert pair.lifetime == pytest.approx(3600)


# ---------------------------------------------------------------- TokenStore
class _FakeQdrant:
    def __init__(self) -> None:
        self.saved: dict | None = None
        self.fail_times = 0

    async def get_payload(self, point_id, collection=None):
        return self.saved

    async def save_payload(self, payload, point_id, collection=None):
        if self.fail_times > 0:
            self.fail_times -= 1
            raise RuntimeError("qdrant write failed")
        self.saved = dict(payload)


def test_store_roundtrip():
    q = _FakeQdrant()
    store = TokenStore(q)
    pair = TokenPair(access_token="a", refresh_token="r",
                     expires_at=1000.0, issued_at=900.0)
    asyncio.run(store.save(pair))
    loaded = asyncio.run(TokenStore(q).load())
    assert loaded.access_token == "a" and loaded.refresh_token == "r"
    assert loaded.expires_at == 1000.0


def test_store_retries_a_failed_save(monkeypatch):
    """Losing the rotated token forces a re-login: retry hard."""
    import addon.trakt_token as mod

    monkeypatch.setattr(mod, "SAVE_BACKOFF", 0.0)
    q = _FakeQdrant()
    q.fail_times = 2                      # fail twice, succeed on the third
    asyncio.run(TokenStore(q).save(TokenPair(access_token="a", refresh_token="r")))
    assert q.saved is not None and q.saved["refresh_token"] == "r"


def test_store_without_qdrant_does_not_raise():
    asyncio.run(TokenStore(None).save(TokenPair(access_token="a", refresh_token="r")))
    assert asyncio.run(TokenStore(None).load()) is None


# ------------------------------------------------------- client refresh flow
def _client(**over):
    cfg = TraktConfig(client_id="cid", access_token="old", client_secret="sec",
                      redirect_uri="https://x/ingest/trakt/callback",
                      refresh_token="refresh-1")
    for k, v in over.items():
        cfg = TraktConfig(**{**cfg.__dict__, k: v})
    return TraktClient(cfg)


def _patch_exchange(monkeypatch, calls: list, *, fail: Exception | None = None):
    import addon.trakt as mod

    async def fake_exchange(**kw):
        calls.append(kw)
        # Yield control so siblings actually get scheduled. Without a suspension
        # point the whole coroutine runs to completion before the next task starts,
        # the lock is never contended, and a concurrency test passes vacuously
        # while measuring nothing. A real exchange is a network round trip, so this
        # models it faithfully.
        await asyncio.sleep(0)
        if fail is not None:
            raise fail
        now = time.time()
        return TokenPair(access_token=f"access-{len(calls)}",
                         refresh_token=f"refresh-{len(calls) + 1}",
                         expires_at=now + 7 * DAY, issued_at=now)

    monkeypatch.setattr(mod, "exchange_refresh_token", fake_exchange)


def test_refresh_persists_both_rotated_tokens(monkeypatch):
    """The single-use refresh token MUST be stored, or the chain breaks."""
    q = _FakeQdrant()
    client = _client()
    client.store = TokenStore(q)
    _patch_exchange(monkeypatch, [])

    asyncio.run(client.refresh())

    assert q.saved is not None
    assert q.saved["access_token"] == "access-1"
    assert q.saved["refresh_token"] == "refresh-2", "rotated refresh token lost!"
    assert client.current_token == "access-1"


def test_concurrent_refresh_happens_exactly_once(monkeypatch):
    """Single-use tokens: two refreshes would invalidate each other."""
    calls: list = []
    client = _client()
    client.store = TokenStore(_FakeQdrant())
    _patch_exchange(monkeypatch, calls)

    async def burst():
        # Start all five as real tasks, then release them together. Launching them
        # without a rendezvous lets the first task run to completion before the
        # others are even scheduled, which proves nothing about the lock.
        ready = asyncio.Event()
        holders = {"n": 0}
        tasks = []
        for _ in range(5):
            async def one():
                holders["n"] += 1
                if holders["n"] == 5:
                    ready.set()
                await ready.wait()
                await client.refresh()
            tasks.append(asyncio.create_task(one()))
        await asyncio.gather(*tasks)

    asyncio.run(burst())
    assert len(calls) == 1, f"{len(calls)} refreshes burned the token chain"


def test_refresh_adopts_a_pair_another_worker_rotated(monkeypatch):
    """Second worker must NOT refresh again — it must adopt the stored token."""
    calls: list = []
    q = _FakeQdrant()
    q.saved = {"access_token": "already-rotated", "refresh_token": "refresh-9",
               "expires_at": time.time() + 7 * DAY, "issued_at": time.time()}
    client = _client()
    client.store = TokenStore(q)
    _patch_exchange(monkeypatch, calls)

    asyncio.run(client.refresh())
    assert calls == [], "refreshed despite another worker having rotated"
    assert client.current_token == "already-rotated"


def test_refresh_requires_the_three_identity_fields():
    client = _client(redirect_uri="")
    with pytest.raises(TraktAuthError):
        asyncio.run(client.refresh())


def test_invalid_grant_is_actionable(monkeypatch):
    """Legacy tokens cannot be exchanged (Trakt's Nov 2026 migration).

    Exercises the REAL `exchange_refresh_token` against a mocked 400 so the
    "re-authorize" wording is the code's, not a string the test itself supplied.
    """
    import httpx

    def handler(request):
        return httpx.Response(400, json={"error": "invalid_grant",
                                         "error_description": "session not found"})

    real_client = httpx.AsyncClient

    def fake_client(*args, **kw):
        kw["transport"] = httpx.MockTransport(handler)
        kw.pop("base_url", None)
        return real_client(*args, **kw)

    import addon.trakt_token as tok
    monkeypatch.setattr(tok.httpx, "AsyncClient", fake_client)

    client = _client()
    client.store = TokenStore(_FakeQdrant())
    with pytest.raises(TraktAuthError) as exc:
        asyncio.run(client.refresh())

    message = str(exc.value)
    assert "invalid_grant" in message
    assert "re-authorize" in message, "the operator must be told how to recover"
    status = asyncio.run(client.auth_status())
    assert status["trakt_auth"] == "refresh_failed"
    assert status["trakt_refresh_failures"] == 1


def test_401_triggers_refresh_and_retries_once(monkeypatch):
    """The end-to-end behaviour: a dead token self-heals on the next request."""
    calls: list = []
    client = _client()
    client.store = TokenStore(_FakeQdrant())
    _patch_exchange(monkeypatch, calls)

    attempts = {"n": 0}

    async def flaky_call():
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise TraktUnauthorized("401")
        return "ok"

    assert asyncio.run(client._with_token(flaky_call)) == "ok"
    assert attempts["n"] == 2
    assert len(calls) == 1


def test_proactive_refresh_avoids_the_401(monkeypatch):
    """A token past 80 % of life is refreshed BEFORE the call is attempted."""
    calls: list = []
    client = _client()
    client.store = TokenStore(_FakeQdrant())
    _patch_exchange(monkeypatch, calls)
    issued = time.time()
    # 90 % of a 1-day lifetime elapsed: 2.4 h left of 24 h.
    client._pair = TokenPair(access_token="old", refresh_token="r1",
                             expires_at=issued + 0.1 * DAY, issued_at=issued - 0.9 * DAY)

    async def call():
        return "ok"

    assert asyncio.run(client._with_token(call)) == "ok"
    assert len(calls) == 1, "did not refresh a nearly-expired token proactively"


def test_no_refresh_without_a_store_or_refresh_token():
    """Behaviour must be identical to before when refresh is not configured."""
    client = TraktClient(TraktConfig(client_id="cid", access_token="old"))
    async def call():
        return "ok"
    assert asyncio.run(client._with_token(call)) == "ok"


# ----------------------------------------------------------------- /health
def test_auth_status_reports_expiry(monkeypatch):
    client = _client()
    client.store = TokenStore(_FakeQdrant())
    now = time.time()
    client._pair = TokenPair(access_token="a", refresh_token="r",
                             expires_at=now + 5 * DAY, issued_at=now)
    status = asyncio.run(client.auth_status())
    assert status["trakt_auth"] == "ok"
    assert 119 < status["trakt_token_expires_in_h"] < 121
    assert status["trakt_refresh_enabled"] is True


def test_auth_status_reports_expired():
    client = _client()
    client.store = TokenStore(_FakeQdrant())
    now = time.time()
    client._pair = TokenPair(access_token="a", refresh_token="r",
                             expires_at=now - 100, issued_at=now - DAY)
    assert asyncio.run(client.auth_status())["trakt_auth"] == "expired"
