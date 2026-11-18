"""Deploy readiness check — what Koyeb will actually see.

Verifies, without any cloud account:
  * the app boots with only the runtime requirements (no ML imports)
  * /health answers 200 and degrades cleanly with no Qdrant configured
  * /configure produces a working install URL from forwarded headers
  * the secret gate rejects a wrong secret
"""

from __future__ import annotations

import json
import os
import socket
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import httpx  # noqa: E402
import uvicorn  # noqa: E402

SECRET = "deploy-check-secret"


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def main() -> int:
    os.environ["INSTALL_SECRET"] = SECRET
    os.environ.pop("QDRANT_URL", None)          # simulate "before M1 has run"

    from addon.config import Config
    from addon.main import make_app

    port = free_port()
    server = uvicorn.Server(uvicorn.Config(make_app(), host="127.0.0.1", port=port,
                                           log_level="error"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.05)
    if not server.started:
        print("FAIL: uvicorn did not start")
        return 1

    base = f"http://127.0.0.1:{port}"
    failures: list[str] = []
    try:
        with httpx.Client(timeout=10.0) as client:
            health = client.get(f"{base}/health")
            body = health.json()
            print(f"/health          {health.status_code}")
            print(json.dumps(body, indent=1)[:700])
            if health.status_code != 200:
                failures.append("/health must return 200 for the Koyeb probe")
            if body.get("status") != "ok":
                failures.append("/health status must be 'ok'")
            if not body.get("secret_configured"):
                failures.append("INSTALL_SECRET was not picked up from the environment")
            if body["qdrant"].get("mode") is None:
                failures.append("expected qdrant fallback mode when QDRANT_URL is unset")

            redirect = client.get(f"{base}/manifest.json", follow_redirects=False)
            print(f"/manifest.json   {redirect.status_code} -> {redirect.headers.get('location')}")
            if redirect.status_code != 302:
                failures.append("bare /manifest.json should redirect to /configure")

            # Koyeb terminates TLS and forwards the original scheme/host
            configure = client.get(f"{base}/configure",
                                   headers={"x-forwarded-proto": "https",
                                            "x-forwarded-host": "moviematch-koyeb.app"})
            page = configure.text
            print(f"/configure       {configure.status_code}")
            if "https://moviematch-koyeb.app/" not in page:
                failures.append("/configure ignored X-Forwarded-* headers (install URL "
                                "would point at http://127.0.0.1 on Koyeb)")

            blob = Config(secret=SECRET).encode()
            good = client.get(f"{base}/{blob}/catalog/movie/because.json")
            print(f"catalog (good)   {good.status_code} metas={len(good.json().get('metas', []))}")
            if good.status_code != 200:
                failures.append("valid config should return 200")

            bad = client.get(f"{base}/{Config(secret='wrong').encode()}/manifest.json")
            print(f"manifest (bad)   {bad.status_code}")
            if bad.status_code != 403:
                failures.append("wrong secret must be 403")
    finally:
        server.should_exit = True
        thread.join(timeout=5)

    from addon.trakt import TraktConfig
    cfg = TraktConfig.from_env({})
    print(f"trakt (unset)    configured={cfg.configured} "
          f"(fixture history is used until credentials exist)")
    if cfg.configured:
        failures.append("empty env must leave Trakt unconfigured")

    # --- with a Qdrant-backed store (the case that 500'd on Koyeb) -------------
    print("\n--- with Qdrant store attached ---")
    failures += _check_qdrant_store()


def _check_qdrant_store() -> list[str]:
    """Coerce: /health must not assume len() on the store."""
    import asyncio

    import numpy as np

    from addon.ids import point_id
    from addon.main import make_app as _make_app
    from addon.store import QdrantStore, Record

    class _Client:
        async def count(self):
            return 1

        async def retrieve(self, ids, **_):
            return []

        async def scroll_all(self, **_):
            return []

        async def search(self, *a, **k):
            return []

    rec = Record(point_id("movie", 155), np.ones(384, dtype=np.float32),
                 {"tmdb_id": 155, "media_type": "movie", "imdb_id": "tt0468569",
                  "title": "The Dark Knight", "year": 2008, "language": "en",
                  "poster": "/p.jpg", "vote_average": 8.5, "vote_count": 32000,
                  "popularity": 120.0, "director": "Christopher Nolan", "cast": []})
    app = _make_app(store=QdrantStore(client=_Client()))
    route = next(r for r in app.routes
                 if getattr(r, "path", None) == "/health" and "GET" in r.methods)
    try:
        body = asyncio.run(route.endpoint())
    except Exception as exc:  # noqa: BLE001
        print(f"/health (qdrant)  RAISED {type(exc).__name__}: {exc}")
        return [f"/health crashed with a Qdrant store: {exc}"]
    print(f"/health (qdrant)  ok  store={body['store']} catalog={body['catalog_size']}")
    problems = []
    if body["store"] != "qdrant":
        problems.append("/health should report store=qdrant")
    if "HEAD" not in route.methods:
        problems.append("/health must accept HEAD (Koyeb health check)")
    _ = rec
    return problems

    if failures:
        print("\nFAILED:")
        for item in failures:
            print("  -", item)
        return 1
    print("\nDEPLOY CHECK PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
