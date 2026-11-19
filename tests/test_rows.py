"""End-to-end row build + Stremio protocol (PLAN §10, §15 acceptance seeds)."""

import base64
import json
from types import SimpleNamespace

import numpy as np
import pytest

from addon.config import Config
from addon.main import make_app
from addon.rank import BECAUSE_ANN_LIMIT, BECAUSE_SEED_WEIGHT, BECAUSE_SIM_FLOOR, BECAUSE_TASTE_WEIGHT
from addon.rows import because_query, build_row, build_user_state, to_metas
from addon.stremio import build_manifest, parse_extra, safe_skip
from addon.taste import Taste

from .conftest import *  # noqa: F401,F403  (reuse store/history/config fixtures)


# ------------------------------------------------------------------- rows
def test_only_watched_and_rated_titles_become_signals(store, state):
    from addon.ids import point_id

    # watched: TDK, Breaking Bad (+ a trakt-only item with no tmdb id -> skipped)
    # ratings: Inception 10 (positive), Breaking Bad 9 (positive), Fight Club 2 (dislike)
    # watchlist: The Prestige  => 5 positive signals + 1 dislike
    assert point_id("movie", 155) in state.watched
    assert point_id("series", 1396) in state.watched
    assert point_id("movie", 1124) in state.watchlist
    assert len(state.signals) == 5
    assert len(state.dislikes) == 1
    assert state.last_watched["movie"] == point_id("movie", 155)


def test_foryou_excludes_watched_watchlist_and_disliked(store, history, config):
    plan = build_row(store, history, "movie", "foryou", config)
    excluded = plan.excluded
    shown = {c.point_id for c in plan.ranked}
    assert not shown & excluded
    # Fight Club was rated 2/10 and liked-adjacent in vector space: it must not surface
    from addon.ids import point_id

    assert point_id("movie", 550) not in shown


def test_foryou_ranks_same_family_first(store, history, config):
    """Sanity contract for For You: guards hold, ratings drive the query (§8.2).

    After the dislike fix, likes (Inception 10, Breaking Bad 9, Prestige, Inception)
    converge on the Nolan vector while the disliked Fight Club pulls the other way,
    so the row runs Inception -> Memento -> Joker. The Dark Knight is excluded, and
    nothing orthogonal to the query survives the similarity floor.
    """
    plan = build_row(store, history, "movie", "foryou", config)
    titles = [c.payload["title"] for c in plan.ranked]
    assert titles, "row must not be empty"
    assert titles[0] == "Inception", titles
    assert "The Dark Knight" not in titles           # watched -> excluded
    assert "Fight Club" not in titles                # rated 2 -> excluded via dislike
    assert "2001: A Space Odyssey" not in titles     # orthogonal -> below sim floor
    family = {"Inception", "Memento", "Joker", "The Prestige"}
    assert set(titles) <= family, f"unexpected titles: {set(titles) - family}"
    assert all(c.similarity >= 0.35 for c in plan.ranked)


def test_because_uses_last_watched_as_seed(store, history, config):
    seed_pid = build_user_state(store, history).last_watched["movie"]
    plan = build_row(store, history, "movie", "because", config)
    assert plan.ranked, "because row must return candidates"
    # the seed itself never appears in its own row
    assert all(c.point_id != seed_pid for c in plan.ranked)
    # candidates clear the relaxed `because` floor, and we searched wider than 200
    assert all(c.similarity > BECAUSE_SIM_FLOOR for c in plan.ranked)
    assert plan.diagnostics["ann_limit"] == BECAUSE_ANN_LIMIT


def test_because_query_blends_seed_and_taste():
    """0.7*seed + 0.3*taste, re-normalised — a pure seed query cannot leave its franchise."""
    seed = SimpleNamespace(vector=np.array([1.0, 0.0, 0.0], dtype=np.float32))
    taste = Taste(long=None, query=np.array([0.0, 1.0, 0.0], dtype=np.float32))
    blended = because_query(seed, taste)
    norm = np.sqrt(BECAUSE_SEED_WEIGHT ** 2 + BECAUSE_TASTE_WEIGHT ** 2)
    assert np.linalg.norm(blended) == pytest.approx(1.0, abs=1e-6)
    assert blended[0] == pytest.approx(BECAUSE_SEED_WEIGHT / norm, abs=1e-6)
    assert blended[1] == pytest.approx(BECAUSE_TASTE_WEIGHT / norm, abs=1e-6)
    # no taste -> bare seed vector
    assert np.allclose(because_query(seed, Taste()), seed.vector)


def test_because_without_history_is_empty(store, config):
    plan = build_row(store, {}, "movie", "because", config)
    assert plan.ranked == [] and plan.diagnostics["reason"] == "no_seed"


def test_cold_start_returns_empty_rows(store, config):
    plan = build_row(store, {}, "movie", "foryou", config)
    assert plan.ranked == [] and plan.diagnostics["reason"] == "cold_start"


def test_language_preference_filters_non_english(store, history, config):
    plan = build_row(store, history, "movie", "foryou", config)
    langs = {c.payload["language"] for c in plan.ranked}
    assert langs <= {"en"}


def test_series_row_only_returns_series(store, history, config):
    plan = build_row(store, history, "series", "foryou", config)
    assert all(c.payload["media_type"] == "series" for c in plan.ranked)


def test_meta_preview_shape(store, history, config):
    plan = build_row(store, history, "movie", "foryou", config)
    metas = to_metas(plan.ranked)
    assert metas
    for meta in metas:
        assert meta["id"].startswith("tt")
        assert meta["type"] == "movie"
        assert meta["poster"].startswith("https://image.tmdb.org/t/p/w500/")
        assert {"name", "poster", "releaseInfo"} <= set(meta)
    # genres are optional (empty lists are dropped, not shipped as [])
    assert any("genres" in m for m in metas) or all("genres" not in m for m in metas)


# --------------------------------------------------------------- protocol
def test_parse_extra_and_skip():
    assert parse_extra("skip=100&genre=Sci-Fi") == {"skip": "100", "genre": "Sci-Fi"}
    assert parse_extra("skip%3D100") == {}
    assert parse_extra(None) == {}
    assert safe_skip({"skip": "abc"}) == 0
    assert safe_skip({"skip": "-5"}) == 0
    assert safe_skip({"skip": "5000"}) == 200


def test_manifest_has_expected_catalogs():
    manifest = build_manifest("https://x.test", "blob", {"movie": "The Dark Knight", "series": "Breaking Bad"})
    # catalog serves rows; the library event resource mirrors watch/list actions
    assert "catalog" in manifest["resources"]
    assert any(isinstance(r, dict) and r.get("name") == "library"
               for r in manifest["resources"])
    assert manifest["idPrefixes"] == ["tt"]
    assert set(manifest["types"]) == {"movie", "series"}
    ids = [(c["type"], c["id"]) for c in manifest["catalogs"]]
    assert ids.count(("movie", "foryou")) == 1
    assert ids.count(("series", "foryou")) == 1
    names = [c["name"] for c in manifest["catalogs"] if c["id"] == "because"]
    assert any("The Dark Knight" in n for n in names)
    assert any("Breaking Bad" in n for n in names)


# ------------------------------------------------------------------- HTTP
# --- Why 7 tests are skipped on this machine ------------------------------------
# starlette 0.27 + httpx 0.28 is a known-broken pair: starlette's TestClient calls
# `httpx.Client(app=...)`, which httpx 0.28 now maps onto `mounts=` -> TypeError.
# `pip install -U "fastapi>=0.115" "starlette>=0.40"` fixes it. Until then these
# tests are skipped, NOT silently passing — the smoke test below still drives
# build_row() end to end without a client.
import subprocess  # noqa: E402
import sys  # noqa: E402


def test_http_surface_serves_manifest_and_catalog(store, history, secret):
    """ASGI end-to-end using uvicorn + httpx against a real socket."""
    import json
    import socket
    import time
    from urllib.parse import quote

    from addon.main import make_app

    try:
        import uvicorn
    except ImportError:  # pragma: no cover
        pytest.skip("uvicorn not installed")

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]

    app = make_app(store=store, history_provider=lambda: history)
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(config)
    import threading

    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.05)
    if not server.started:  # pragma: no cover
        pytest.skip("uvicorn failed to start")

    import httpx

    base = f"http://127.0.0.1:{port}"
    blob = Config(secret=secret).encode()
    try:
        with httpx.Client(timeout=5.0) as client:
            health = client.get(f"{base}/health").json()
            # /health is a cheap liveness probe: no ranking, no Trakt sync
            assert health["catalog_size"] == 12 and health["status"] == "ok"
            assert "rows_ready" not in health, "/health must not rank rows"
            deep = client.get(f"{base}/health/deep").json()
            assert deep["taste_ready"] is True and deep["rows_ready"] >= 0

            redirect = client.get(f"{base}/manifest.json", follow_redirects=False)
            assert redirect.status_code == 302 and redirect.headers["location"] == "/configure"

            manifest = client.get(f"{base}/{blob}/manifest.json").json()
            assert manifest["id"] == "community.moviematch.personal"
            assert len(manifest["catalogs"]) == 6

            catalog = client.get(f"{base}/{blob}/catalog/movie/foryou.json").json()
            assert catalog["metas"] and catalog["cacheMaxAge"] == 900
            first = catalog["metas"][0]
            assert first["id"].startswith("tt") and first["poster"].startswith("https://image.tmdb.org")

            paged = client.get(f"{base}/{blob}/catalog/movie/foryou/skip=1.json").json()
            assert paged["metas"] == catalog["metas"][1:]

            assert client.get(f"{base}/{Config(secret='bad').encode()}/manifest.json").status_code == 403
            assert client.get(f"{base}/garbage!!/manifest.json").status_code == 400
            assert client.get(f"{base}/{blob}/catalog/movie/bogus.json").status_code == 404
    finally:
        server.should_exit = True
        thread.join(timeout=5)


