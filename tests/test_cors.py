"""Stremio Web fetches the addon CROSS-ORIGIN.

Live symptom: DevTools showed the manifest request with "Provisional headers are
shown" and no response headers, and the client said "Failed to get addon manifest"
while the server logged a clean 200. Cause: no CORS headers at all, so the browser
discarded the response. The desktop app is not a browser, which is why this only
appeared on web.stremio.com.
"""

from __future__ import annotations

import pytest

from addon.main import make_app

from .conftest import *  # noqa: F401,F403

ORIGIN = "https://web.stremio.com"


@pytest.fixture()
def client(store, history):
    """Real socket + uvicorn: CORS is applied by middleware, not by the handler."""
    import socket
    import threading
    import time

    import httpx
    import uvicorn

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    app = make_app(store=store, history_provider=lambda: history)
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.05)
    with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=10.0) as http:
        yield http
    server.should_exit = True
    thread.join(timeout=5)


def test_manifest_sends_allow_origin(client, secret):
    blob = make_blob(secret)
    response = client.get(f"/{blob}/manifest.json", headers={"Origin": ORIGIN})
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") in ("*", ORIGIN)


def test_catalog_sends_allow_origin(client, secret):
    blob = make_blob(secret)
    response = client.get(f"/{blob}/catalog/movie/foryou.json", headers={"Origin": ORIGIN})
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") in ("*", ORIGIN)


def test_preflight_is_answered(client, secret):
    """Browsers preflight before a cross-origin GET with custom headers."""
    blob = make_blob(secret)
    response = client.options(
        f"/{blob}/manifest.json",
        headers={"Origin": ORIGIN,
                 "Access-Control-Request-Method": "GET",
                 "Access-Control-Request-Headers": "authorization"},
    )
    assert response.status_code in (200, 204), response.status_code
    assert "GET" in response.headers.get("access-control-allow-methods", "")
    assert response.headers.get("access-control-allow-origin") in ("*", ORIGIN)


def test_cors_applies_to_errors_too(client, secret):
    """A 403 without CORS headers shows up in the browser as an opaque failure."""
    response = client.get(f"/{make_blob('wrong')}/manifest.json", headers={"Origin": ORIGIN})
    assert response.status_code == 403
    assert response.headers.get("access-control-allow-origin") in ("*", ORIGIN)


def test_health_and_logo_are_cross_origin_friendly(client):
    for path in ("/health", "/static/logo.png"):
        response = client.get(path, headers={"Origin": ORIGIN})
        assert response.status_code == 200, path
        assert response.headers.get("access-control-allow-origin") in ("*", ORIGIN), path


def make_blob(secret: str) -> str:
    from addon.config import Config

    return Config(secret=secret).encode()
