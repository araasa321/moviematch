"""The exact JSON sent to Trakt must match their documented schema.

A live deployment sent `{"movies":[{"movie":{"ids":{...}}}]}` instead of
`{"movies":[{"ids":{...}}]}`. Trakt answered 201 with the item in `not_found`,
so nothing was recorded — invisible until the response body was inspected.

These tests intercept the outgoing HTTP request, so a wrong nesting fails here.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from addon.trakt import TraktClient, TraktConfig, TraktError


class Captured:
    def __init__(self, status: int = 201, body: dict | None = None) -> None:
        self.status = status
        self.body = body or {"added": {"movies": 1}, "not_found": {"movies": []}}
        self.calls: list[tuple[str, dict]] = []
        self.headers: dict[str, str] = {}


@pytest.fixture()
def captured(monkeypatch):
    state = Captured()

    class FakeResponse:
        def __init__(self, status: int, payload: dict) -> None:
            self.status_code = status
            self._payload = payload
            self.text = json.dumps(payload)
            self.content = b"x"

        def json(self):
            return self._payload

    class FakeClient:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def post(self, url, headers=None, json=None, **kw):
            state.calls.append((url, json))
            state.headers = headers or {}
            return FakeResponse(state.status, state.body)

        async def get(self, url, headers=None, params=None, **kw):
            state.calls.append((url, params or {}))
            state.headers = headers or {}
            return FakeResponse(state.status, state.body)

    monkeypatch.setattr("addon.trakt.httpx.AsyncClient", FakeClient)
    return state


def _client() -> TraktClient:
    return TraktClient(TraktConfig(client_id="cid", access_token="tok"))


def test_history_sends_ids_at_item_level(captured):
    """The regression: ids must NOT be nested under a 'movie' key."""
    client = _client()
    media = {"movie": {"ids": {"imdb": "tt0993840"}}}
    asyncio.run(client.add_history(media, watched_at="now"))

    url, body = captured.calls[-1]
    assert url.endswith("/sync/history")
    assert body == {"movies": [{"ids": {"imdb": "tt0993840"}, "watched_at": "now"}]}
    assert "movie" not in body["movies"][0], "ids must not be nested under 'movie'"


def test_history_sends_show_item_level(captured):
    client = _client()
    asyncio.run(client.add_history({"show": {"ids": {"imdb": "tt0903747"}}},
                                   watched_at="unknown"))
    _, body = captured.calls[-1]
    assert body == {"shows": [{"ids": {"imdb": "tt0903747"}, "watched_at": "unknown"}]}


def test_watchlist_add_shape(captured):
    client = _client()
    asyncio.run(client.add_watchlist({"movie": {"ids": {"imdb": "tt1"}}}))
    url, body = captured.calls[-1]
    assert url.endswith("/sync/watchlist")
    assert body == {"movies": [{"movie": {"ids": {"imdb": "tt1"}}}]}


def test_watchlist_remove_shape(captured):
    client = _client()
    asyncio.run(client.remove_watchlist({"show": {"ids": {"imdb": "tt2"}}}))
    url, body = captured.calls[-1]
    assert url.endswith("/sync/watchlist/remove"), "remove is a POST to /remove"
    assert body == {"shows": [{"show": {"ids": {"imdb": "tt2"}}}]}


def test_identifying_user_agent_is_sent(captured):
    """Trakt's CDN answers 403 'error code: 1010' to bot-like agents."""
    client = _client()
    asyncio.run(client.add_history({"movie": {"ids": {"imdb": "tt1"}}}))
    ua = captured.headers.get("User-Agent", "")
    assert ua.startswith("MovieMatch/"), ua
    assert "eval" not in ua.lower()


def test_no_client_side_mutation_of_caller_media(captured):
    """add_history must not stamp watched_at onto the caller's dict."""
    media = {"movie": {"ids": {"imdb": "tt1"}}}
    asyncio.run(_client().add_history(media))
    assert media == {"movie": {"ids": {"imdb": "tt1"}}}


def test_1010_is_reported_as_cdn_block_not_permissions(monkeypatch):
    state = Captured(status=403, body={"error": "blocked"})
    state.body = {"error": "blocked"}
    monkeypatch.setattr("addon.trakt.httpx.AsyncClient", _client_for(state, 403,
                                                                     "error code: 1010"))
    with pytest.raises(TraktError, match="1010"):
        asyncio.run(_client().add_history({"movie": {"ids": {"imdb": "tt1"}}}))


def test_403_without_1010_mentions_permissions(monkeypatch):
    state = Captured()
    monkeypatch.setattr("addon.trakt.httpx.AsyncClient",
                        _client_for(state, 403, "permission denied"))
    with pytest.raises(TraktError, match="write permissions"):
        asyncio.run(_client().add_history({"movie": {"ids": {"imdb": "tt1"}}}))


def _client_for(state, status: int, text: str):
    class FakeResponse:
        status_code = status
        content = b"x"

        def __init__(self):
            self.text = text

        def json(self):
            return {}

    class FakeClient:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def post(self, url, **kw):
            state.calls.append((url, kw.get("json")))
            return FakeResponse()

    return FakeClient
