"""Stremio library events -> Trakt writes.

Covers the three implemented actions and asserts `unwatched` is refused WITHOUT
any Trakt call (history deletion is deliberately out of scope).
"""

from __future__ import annotations

import asyncio
import json

import pytest

from addon.library import SUPPORTED_ACTIONS, handle_library_event
from addon.trakt import TraktConfig, TraktError


class FakeTrakt:
    """Records write calls; returns Trakt-shaped bodies. Never touches the network."""

    HISTORY_ADDED = {"added": {"movies": 1, "episodes": 0}, "updated": {"movies": 0},
                     "not_found": {"movies": []}}
    HISTORY_NOT_FOUND = {"added": {"movies": 0}, "updated": {"movies": 0},
                         "not_found": {"movies": [{"ids": {"imdb": "ttX"}}]}}
    WATCHLIST_ADDED = {"added": {"movies": 1, "shows": 0}, "existing": {"movies": 0},
                       "not_found": {"movies": []}}
    WATCHLIST_DELETED = {"deleted": {"movies": 1, "shows": 0}, "not_found": {"movies": []}}

    def __init__(self, fail: Exception | None = None, body: dict | None = None) -> None:
        self.calls: list[tuple[str, dict]] = []
        self.fail = fail
        self.body = body

    async def add_history(self, media, watched_at="now"):
        return self._record("history", media, self.HISTORY_ADDED)

    async def add_watchlist(self, media):
        return self._record("watchlist_add", media, self.WATCHLIST_ADDED)

    async def remove_watchlist(self, media):
        return self._record("watchlist_remove", media, self.WATCHLIST_DELETED)

    def _record(self, name, media, default):
        if self.fail:
            raise self.fail
        self.calls.append((name, media))
        return self.body if self.body is not None else default


def run(coro):
    return asyncio.run(coro)


def test_actions_are_the_three_we_support():
    assert SUPPORTED_ACTIONS == ("libraryAdd", "libraryRemove", "watched")
    assert "unwatched" not in SUPPORTED_ACTIONS


def test_watched_movie_maps_to_history():
    trakt = FakeTrakt()
    result = run(handle_library_event(trakt, "movie", "tt0468569", "watched"))
    assert result.ok
    assert trakt.calls == [("history", {"movie": {"ids": {"imdb": "tt0468569"}}})]


def test_library_add_goes_to_watchlist():
    trakt = FakeTrakt()
    result = run(handle_library_event(trakt, "movie", "tt1375666", "libraryAdd"))
    assert result.ok
    assert trakt.calls == [("watchlist_add", {"movie": {"ids": {"imdb": "tt1375666"}}})]


def test_library_remove_posts_to_remove_endpoint():
    trakt = FakeTrakt()
    result = run(handle_library_event(trakt, "series", "tt0903747", "libraryRemove"))
    assert result.ok
    assert trakt.calls == [("watchlist_remove", {"show": {"ids": {"imdb": "tt0903747"}}})]


def test_unwatched_is_refused_without_calling_trakt():
    """The one action that could delete history must never reach Trakt."""
    trakt = FakeTrakt()
    result = run(handle_library_event(trakt, "movie", "tt0468569", "unwatched"))
    assert not result.ok and not result.handled
    assert "not supported" in result.detail
    assert trakt.calls == []


def test_unknown_action_is_ignored():
    trakt = FakeTrakt()
    result = run(handle_library_event(trakt, "movie", "tt1", "bogus"))
    assert not result.handled and trakt.calls == []


def test_non_imdb_id_is_rejected():
    trakt = FakeTrakt()
    result = run(handle_library_event(trakt, "movie", "12345", "watched"))
    assert not result.ok and trakt.calls == []


def test_episode_video_id_is_reduced_to_the_show():
    trakt = FakeTrakt()
    result = run(handle_library_event(trakt, "series", "tt0944947:1:1", "watched",
                                      video_id="tt0944947:1:1"))
    assert result.ok
    assert trakt.calls == [("history", {"show": {"ids": {"imdb": "tt0944947"}}})]
    assert "show-level" in result.detail


def test_trakt_403_is_reported_not_raised():
    """A token without write scope must produce a clear message, not a 500."""
    trakt = FakeTrakt(fail=TraktError("Trakt refused the write (403): needs write permissions"))
    result = run(handle_library_event(trakt, "movie", "tt1", "watched"))
    assert not result.ok
    assert "403" in result.detail


def test_unexpected_exception_is_contained():
    class Boom(FakeTrakt):
        async def add_history(self, media, watched_at="now"):
            raise RuntimeError("connection reset")

    result = run(handle_library_event(Boom(), "movie", "tt1", "watched"))
    assert not result.ok and "RuntimeError" in result.detail


# ------------------------------------------- success must reflect Trakt's body
def test_trakt_not_found_is_reported_as_failure():
    """A 2xx with not_found means nothing was written — never report success."""
    trakt = FakeTrakt(body=FakeTrakt.HISTORY_NOT_FOUND)
    result = run(handle_library_event(trakt, "movie", "ttX", "watched"))
    assert not result.ok
    assert "could not resolve" in result.detail


def test_already_watched_counts_as_success():
    trakt = FakeTrakt(body={"added": {"movies": 0}, "updated": {"movies": 1},
                            "not_found": {"movies": []}})
    result = run(handle_library_event(trakt, "movie", "tt1", "watched"))
    assert result.ok and "already" in result.detail


def test_empty_response_is_not_success():
    trakt = FakeTrakt(body={})
    result = run(handle_library_event(trakt, "movie", "tt1", "watched"))
    assert not result.ok and "empty" in result.detail


def test_watchlist_add_reports_existing():
    trakt = FakeTrakt(body={"added": {"movies": 0}, "existing": {"movies": 1},
                            "not_found": {"movies": []}})
    result = run(handle_library_event(trakt, "movie", "tt1", "libraryAdd"))
    assert result.ok and "already" in result.detail


def test_watchlist_remove_of_absent_item_is_success():
    trakt = FakeTrakt(body={"deleted": {"movies": 0},
                            "not_found": {"movies": [{"ids": {"imdb": "tt1"}}]}})
    result = run(handle_library_event(trakt, "movie", "tt1", "libraryRemove"))
    assert result.ok and "was not on" in result.detail


def test_cdn_block_message_is_not_blamed_on_permissions():
    """403 error 1010 is a Cloudflare block, not a scope problem."""
    trakt = FakeTrakt(fail=TraktError("Trakt's CDN blocked this request (403 error code: 1010)"))
    result = run(handle_library_event(trakt, "movie", "tt1", "watched"))
    assert not result.ok and "1010" in result.detail


# --------------------------------------------------------------- HTTP layer
@pytest.fixture()
def client(store, history, secret):
    import socket
    import threading
    import time

    import httpx
    import uvicorn

    from addon.main import make_app

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    trakt = FakeTrakt()
    app = make_app(store=store, history_provider=lambda: history, trakt=trakt)
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port,
                                           log_level="error"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.05)
    with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=10.0) as http:
        yield http, trakt
    server.should_exit = True
    thread.join(timeout=5)


def _blob(secret: str) -> str:
    from addon.config import Config

    return Config(secret=secret).encode()


def test_manifest_declares_library_resource():
    from addon.stremio import build_manifest

    manifest = build_manifest("https://x.test", "blob")
    resources = manifest["resources"]
    assert "catalog" in resources
    lib = [r for r in resources if isinstance(r, dict) and r.get("name") == "library"]
    assert lib, "manifest must declare the library resource or Stremio sends no events"
    assert lib[0]["idPrefixes"] == ["tt"]


def test_library_endpoint_post(client, secret):
    http, trakt = client
    response = http.post(f"/{_blob(secret)}/library/movie/tt0468569/action=watched.json")
    assert response.status_code == 200
    assert response.json()["success"] is True
    assert trakt.calls[0][0] == "history"


def test_library_endpoint_get_also_works(client, secret):
    """Stremio may GET the event; both verbs are registered."""
    http, trakt = client
    response = http.get(f"/{_blob(secret)}/library/movie/tt1375666/action=libraryAdd.json")
    assert response.status_code == 200
    assert trakt.calls[0][0] == "watchlist_add"


def test_library_endpoint_requires_the_secret(client):
    http, trakt = client
    response = http.post(f"/{_blob('wrong')}/library/movie/tt1/action=watched.json")
    assert response.status_code == 403
    assert trakt.calls == []


def test_library_endpoint_unwatched_makes_no_write(client, secret):
    http, trakt = client
    response = http.post(f"/{_blob(secret)}/library/movie/tt0468569/action=unwatched.json")
    assert response.status_code == 200
    assert response.json()["success"] is False
    assert trakt.calls == []


def test_library_endpoint_rejects_unknown_type(client, secret):
    http, _ = client
    response = http.post(f"/{_blob(secret)}/library/channel/tt1/action=watched.json")
    assert response.status_code == 404
