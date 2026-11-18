"""FastAPI addon server (PLAN §10.1).

    GET /{config}/manifest.json
    GET /{config}/catalog/{type}/{id}.json
    GET /{config}/catalog/{type}/{id}/{extra}.json     extra = "skip=100&genre=Sci-Fi"
    GET /health
    GET /configure                                     (M4 builds the real page)

Run locally against the fixture catalog:
    uvicorn addon.main:app --reload --port 8000
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any, Callable

import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

from . import __version__
from .config import Config, ConfigError, secret_ok
from .fixtures import catalog as fixture_catalog
from .fixtures import trakt_history
from .memory_store import MemoryStore, load_records_from_json
from .qdrant import QdrantClient, QdrantConfig
from .rows import build_row, build_user_state, to_metas
from .stremio import PAGE_SIZE, build_manifest, catalog_response, parse_extra, safe_skip
from .trakt import TraktClient, TraktConfig

log = logging.getLogger(__name__)

VALID_ROWS = {"foryou", "because", "new4you"}
VALID_TYPES = {"movie", "series"}
DEV_SECRET = "dev-secret-please-set-INSTALL_SECRET"


def _public_origin(request: Any) -> str:
    """Public origin behind a TLS-terminating proxy (Koyeb).

    Koyeb forwards `X-Forwarded-Proto: https` and the public host, but the header
    carrying the host has varied (`X-Forwarded-Host` vs plain `Host`), and a wrong
    guess yields an `http://<internal>:8000/...` install URL that Stremio refuses.
    So: prefer forwarded headers, fall back to Host, and never emit plain http.
    """
    headers = getattr(request, "headers", {}) or {}
    proto = (headers.get("x-forwarded-proto") or "https").split(",")[0].strip().lower()
    host = (headers.get("x-forwarded-host")
            or headers.get("host")
            or request.url.netloc).split(",")[0].strip()
    if not proto.startswith("http"):
        proto = "https"
    return f"{proto}://{host}"


def make_app(store: MemoryStore | None = None,
             history_provider: Callable[[], dict[str, Any]] | None = None,
             dim: int = 384,
             trakt: TraktClient | None = None) -> FastAPI:
    """Factory so tests and smoke runs inject data instead of calling TMDB/Trakt."""
    store = store if store is not None else _fixture_store(dim)
    trakt = trakt or TraktClient(TraktConfig.from_env(os.environ))
    started = datetime.now(timezone.utc)

    app = FastAPI(title="MovieMatch", version=__version__)
    app.state.store = store
    app.state.trakt = trakt

    async def current_history() -> dict[str, Any]:
        """Trakt when configured; the bundled fixture otherwise. Never raises."""
        if trakt.config.configured:
            try:
                return await trakt.fetch_user_state()
            except Exception as exc:  # noqa: BLE001 - fall back, don't 500 Stremio
                log.warning("trakt sync failed (%s); serving fixture history",
                            type(exc).__name__)
        return history_provider() if history_provider else trakt_history()

    app.state.history_provider = current_history

    def current_secret() -> str:
        return os.environ.get("INSTALL_SECRET") or DEV_SECRET

    def load_config(blob: str) -> tuple[Config | None, JSONResponse | None]:
        try:
            config = Config.decode(blob)
        except ConfigError:
            return None, JSONResponse({"err": "bad config"}, status_code=400)
        if not secret_ok(config.secret, current_secret()):
            return None, JSONResponse({"err": "forbidden"}, status_code=403)
        return config, None

    def base_url(request: Request) -> str:
        return _public_origin(request)

    @app.get("/health")
    async def health() -> dict[str, Any]:
        """Koyeb probe (§11.3). Never fails the request: degraded info is in the body."""
        history = await current_history()
        plan = build_row(store, history, "movie", "foryou", Config())
        qdrant_cfg = QdrantConfig.from_env(os.environ)
        qdrant_status: dict[str, Any] = {"configured": qdrant_cfg.configured,
                                         "reachable": False}
        if qdrant_cfg.configured:
            try:
                qdrant_status["reachable"] = True
                qdrant_status["catalog_size"] = await QdrantClient(qdrant_cfg).count()
            except Exception as exc:  # noqa: BLE001 - health must not 500
                qdrant_status["error"] = str(exc)[:160]
                qdrant_status["reachable"] = False
        else:
            qdrant_status["mode"] = "fixture (QDRANT_URL unset)"

        return {
            "status": "ok",
            "version": __version__,
            "qdrant": qdrant_status,
            "trakt": "fixture",
            "trakt_auth": "fixture",
            "catalog_size": len(store),
            "positive_signals": plan.positive_count,
            "taste_ready": plan.taste.ready,
            "rows_ready": len(plan.ranked),
            "started_at": started.isoformat(),
            "secret_configured": current_secret() != DEV_SECRET,
        }

    @app.get("/manifest.json")
    async def bare_manifest() -> RedirectResponse:
        return RedirectResponse("/configure", status_code=302)

    @app.get("/configure")
    async def configure(request: Request) -> HTMLResponse:
        blob = Config(secret=current_secret(),
                      mode=os.environ.get("MODE_DEFAULT", "blockbuster")).encode()
        # Self-check: if this round-trips, a decoded config failure is transport corruption.
        try:
            round_trip_ok = Config.decode(blob).secret == current_secret()
        except ConfigError:
            round_trip_ok = False
        origin = base_url(request)
        https_url = f"{origin}/{blob}/manifest.json"
        https_url = https_url if https_url.startswith("https://") else https_url.replace(
            "http://", "https://", 1)
        stremio_url = https_url[len("https://"):]
        return HTMLResponse(
            "<h1>MovieMatch</h1>"
            "<p>Minimal /configure placeholder — the real page lands in M4.</p>"
            f'<p><a href="stremio://{stremio_url}"><b>Install in Stremio</b></a></p>'
            f'<p>Or open: <a href="{https_url}">{https_url}</a></p>'
            f"<p style='font-size:12px'>config round-trip: "
            f"{'OK' if round_trip_ok else 'FAILED'} · origin seen by app: {origin}</p>"
            '<p style="font-size:12px">This product uses the TMDB API but is not '
            "endorsed or certified by TMDB.</p>"
        )

    @app.get("/{config_blob}/manifest.json")
    async def manifest(request: Request, config_blob: str) -> JSONResponse:
        config, error = load_config(config_blob)
        if error is not None:
            return error
        history = await current_history()
        seeds = _seed_names(store, history, config)
        payload = build_manifest(base_url(request), config_blob, seeds)
        return JSONResponse(payload, headers={"Cache-Control": "max-age=300"})

    @app.get("/{config_blob}/catalog/{media_type}/{row}/{extra}.json")
    async def catalog_extra(config_blob: str, media_type: str, row: str,
                            extra: str) -> JSONResponse:
        return await _catalog_response(config_blob, media_type, row, parse_extra(extra))

    @app.get("/{config_blob}/catalog/{media_type}/{row}.json")
    async def catalog_plain(config_blob: str, media_type: str, row: str) -> JSONResponse:
        return await _catalog_response(config_blob, media_type, row, {})

    async def _catalog_response(config_blob: str, media_type: str, row: str,
                                extra: dict[str, str]) -> JSONResponse:
        config, error = load_config(config_blob)
        if error is not None:
            return error
        if media_type not in VALID_TYPES or row not in VALID_ROWS:
            return JSONResponse({"metas": []}, status_code=404)
        plan = build_row(store, await current_history(), media_type, row, config)
        metas = to_metas(plan.ranked, skip=safe_skip(extra), limit=PAGE_SIZE)
        return JSONResponse(catalog_response(metas), headers={"Cache-Control": "max-age=900"})

    return app


def _seed_names(store: MemoryStore, history: dict[str, Any], config: Config) -> dict[str, str]:
    """Per-type title of the most recent watch, for the dynamic `because` row name.

    Manifest default is the last watched title of each type; when the user has no
    history at all the row is named "Because you watched your last watch" until
    /configure onboarding (§8.4) picks seeds.
    """
    state = build_user_state(store, history)
    seeds: dict[str, str] = {}
    for media_type in VALID_TYPES:
        pid = state.last_watched.get(media_type)
        record = store.get(pid) if pid else None
        seeds[media_type] = str(record.payload.get("title", ""))[:40] if record else "your last watch"
    return seeds


def _fixture_store(dim: int) -> MemoryStore:
    store = MemoryStore()
    load_records_from_json(store, fixture_catalog(dim=dim))
    return store


app = make_app()


