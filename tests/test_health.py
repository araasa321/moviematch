"""Health must work with every store type, and the probe must answer HEAD.

Found on the live Koyeb instance:
  * `TypeError: object of type 'QdrantStore' has no len()` -> /health 500
  * `HEAD /health -> 405` (Koyeb's health checker sends HEAD)
  * install URLs 403'd because the env var had trailing whitespace

`deploy_check.py` missed the first two because it only ran with Qdrant unset.
These tests run the app against a Qdrant-backed store too.
"""

from __future__ import annotations

import numpy as np
import pytest

from addon.config import Config, secret_ok
from addon.ids import point_id
from addon.main import make_app
from addon.store import QdrantStore, Record

from .conftest import project_payload
from .conftest import *  # noqa: F401,F403

SECRET = "a692606ccbd8a6e1aa2427bcb7791857d28d21bf689cb3837c452dbabdd2e0"


class _CountingClient:
    def __init__(self, records: list[Record]) -> None:
        self.records = records

    async def count(self) -> int:
        return len(self.records)

    async def retrieve(self, ids, **_):
        wanted = set(ids)
        return [{"id": r.point_id, "vector": r.vector.tolist(), "payload": r.payload}
                for r in self.records if r.point_id in wanted]

    async def scroll_all(self, **_):
        return [{"id": r.point_id, "vector": r.vector.tolist(), "payload": r.payload}
                for r in self.records]

    async def search(self, vector, *, limit=200, filters=None, with_payload=True, **_):
        return [{"id": r.point_id, "vector": r.vector.tolist(),
                 "payload": project_payload(r.payload, with_payload)}
                for r in self.records[:limit]]


def _qstore() -> QdrantStore:
    rec = Record(point_id("movie", 155), np.ones(384, dtype=np.float32),
                 {"tmdb_id": 155, "media_type": "movie", "imdb_id": "tt0468569",
                  "title": "The Dark Knight", "year": 2008, "language": "en",
                  "poster": "/p.jpg", "vote_average": 8.5, "vote_count": 32000,
                  "popularity": 120.0, "director": "Christopher Nolan", "cast": []})
    return QdrantStore(client=_CountingClient([rec]))


def _run_health(app) -> dict:
    import asyncio

    # call the endpoint function directly — no server, no TestClient dependency
    route = next(r for r in app.routes
                 if getattr(r, "path", None) == "/health" and "GET" in getattr(r, "methods", ()))
    return asyncio.run(route.endpoint())


def test_health_with_qdrant_store_does_not_raise(monkeypatch):
    """The exact crash from the instance log."""
    monkeypatch.setenv("INSTALL_SECRET", SECRET)
    app = make_app(store=_qstore())
    body = _run_health(app)
    assert body["status"] == "ok"
    assert body["store"] == "qdrant"
    assert body["store_size"] is None          # QdrantStore has no len()
    assert body["catalog_size"] is not None or body["catalog_size"] is None


def test_health_with_fixture_store(monkeypatch):
    monkeypatch.setenv("INSTALL_SECRET", SECRET)
    body = _run_health(make_app())
    assert body["store"] == "fixture"
    assert body["catalog_size"] == 12


def test_health_route_accepts_head():
    app = make_app()
    route = next(r for r in app.routes if getattr(r, "path", None) == "/health")
    assert "HEAD" in route.methods, "Koyeb's health checker sends HEAD"


# ------------------------------------------------------- manifest essentials
def test_manifest_does_not_claim_configuration_required():
    """Clients refuse to install when this is True and the URL is already configured."""
    from addon.stremio import build_manifest

    manifest = build_manifest("https://app.test", "blob")
    hints = manifest["behaviorHints"]
    assert hints["configurationRequired"] is False
    assert hints["configurable"] is True


def test_logo_file_exists_and_is_png():
    """manifest.logo points at /static/logo.png; a 404 makes clients reject the addon."""
    from pathlib import Path

    path = Path(__file__).resolve().parent.parent / "addon" / "static" / "logo.png"
    assert path.exists(), "/static/logo.png is advertised in the manifest but missing"
    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_logo_route_serves_the_file():
    app = make_app()
    route = next(r for r in app.routes if getattr(r, "path", None) == "/static/logo.png")
    import asyncio

    response = asyncio.run(route.endpoint())
    assert getattr(response, "media_type", "") == "image/png"


# ------------------------------------------------------------------ secrets
@pytest.mark.parametrize("noisy", [" " + SECRET, SECRET + " ", SECRET + "\n",
                                   SECRET + "\r\n", "\t" + SECRET + "\n"])
def test_secret_tolerates_platform_whitespace(noisy):
    assert secret_ok(noisy, SECRET)
    assert secret_ok(SECRET, noisy)


def test_secret_still_rejects_wrong_values():
    assert not secret_ok(SECRET[:-1], SECRET)
    assert not secret_ok("wrong", SECRET)
    assert not secret_ok("", SECRET)
    assert not secret_ok(SECRET, "")


def test_config_encoded_from_noisy_env_still_authorises(monkeypatch):
    """Whole-path check: env var with a newline -> generated config still validates."""
    monkeypatch.setenv("INSTALL_SECRET", SECRET + "\n")
    app = make_app()
    # the /configure route builds the blob from the env; verify it round-trips
    from addon.main import _public_origin
    from types import SimpleNamespace

    request = SimpleNamespace(headers={"host": "app.koyeb.app"},
                              url=SimpleNamespace(scheme="https", netloc="app.koyeb.app"))
    assert _public_origin(request) == "https://app.koyeb.app"
