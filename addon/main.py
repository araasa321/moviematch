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
import time
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Callable

import numpy as np
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse

from . import __version__
from .config import Config, ConfigError, secret_ok
from .fixtures import catalog as fixture_catalog
from .fixtures import trakt_history
from .library import handle_library_event
from .memory_store import MemoryStore, load_records_from_json
from .qdrant import QdrantClient, QdrantConfig
from .rows import RowPlan, build_row, build_user_state, to_metas
from .store import QdrantStore
from .stremio import PAGE_SIZE, build_manifest, catalog_response, parse_extra, safe_skip
from .trakt import TraktClient, TraktConfig

VALID_ROWS = {"foryou", "because", "new4you"}
VALID_TYPES = {"movie", "series"}
DEV_SECRET = "dev-secret-please-set-INSTALL_SECRET"
ROW_CACHE_SECONDS = 900          # §11.3 catalogs: 900 s

log = logging.getLogger(__name__)


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


def make_app(store: Any | None = None,
             history_provider: Callable[[], dict[str, Any]] | None = None,
             dim: int = 384,
             trakt: TraktClient | None = None) -> FastAPI:
    """Factory so tests and smoke runs inject data instead of calling TMDB/Trakt.

    When `store` is omitted and QDRANT_URL is set, serving reads the real catalog
    from Qdrant; otherwise it falls back to the bundled fixture catalog.
    """
    qdrant_store = QdrantStore.from_env(os.environ) if store is None else None
    store = store if store is not None else (qdrant_store or _fixture_store(dim))
    log.info("serving store: %s", "qdrant" if qdrant_store else "fixture")
    trakt = trakt or TraktClient(TraktConfig.from_env(os.environ))
    started = datetime.now(timezone.utc)
    row_cache: dict[tuple, tuple[float, RowPlan]] = {}

    app = FastAPI(title="MovieMatch", version=__version__)
    # Stremio Web (web.stremio.com) fetches manifests/catalogs CROSS-ORIGIN.
    # Without Access-Control-Allow-Origin the browser blocks the response and the
    # client reports "Failed to get addon manifest" while the server logs a clean
    # 200 — the exact symptom that made this look like a URL problem.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET", "HEAD", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["*"],
        max_age=86400,
    )
    app.state.store = store
    app.state.trakt = trakt
    app.state.row_cache = row_cache

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
    @app.head("/health")
    async def health() -> dict[str, Any]:
        """Cheap liveness probe (§11.3). Never fails; degraded info is in the body.

        This must NOT rank rows or fetch Trakt. The original version called
        build_row() to report `rows_ready`, which meant every probe did a full
        Trakt sync + Qdrant ANN + ranking — 8.6 s per probe on a 0.1-vCPU instance
        and a direct contributor to pegged CPU. Deep checks live on /health/deep.
        """
        qdrant_cfg = QdrantConfig.from_env(os.environ)
        qdrant_status: dict[str, Any] = {"configured": qdrant_cfg.configured,
                                         "reachable": False}
        total: int | None = None
        if qdrant_cfg.configured:
            try:
                total = await QdrantClient(qdrant_cfg).count()
                qdrant_status["reachable"] = True
                qdrant_status["catalog_size"] = total
            except Exception as exc:  # noqa: BLE001 - health must not 500
                qdrant_status["error"] = str(exc)[:160]
                qdrant_status["reachable"] = False
        else:
            qdrant_status["mode"] = "fixture (QDRANT_URL unset)"

        # len() only exists on the in-memory fixture store; never assume it
        store_size = len(store) if hasattr(store, "__len__") else None
        if total is None:
            total = store_size
        return {
            "status": "ok",
            "version": __version__,
            "store": "qdrant" if isinstance(store, QdrantStore) else "fixture",
            "qdrant": qdrant_status,
            "trakt": "trakt" if trakt.config.configured else "fixture",
            "trakt_auth": "configured" if trakt.config.configured else "fixture",
            "catalog_size": total,
            "store_size": store_size,
            "row_cache_size": len(row_cache),
            "row_cache_keys": sorted("/".join(k) for k in row_cache),
            "started_at": started.isoformat(),
            "secret_configured": current_secret() != DEV_SECRET,
        }

    @app.get("/health/deep")
    async def health_deep() -> dict[str, Any]:
        """Expensive diagnostics: Trakt sync + ranking. Never polled by Koyeb."""
        history = await current_history()
        started_at = time.monotonic()
        plan = build_row(store, history, "movie", "foryou", Config())
        return {
            "positive_signals": plan.positive_count,
            "taste_ready": plan.taste.ready,
            "rows_ready": len(plan.ranked),
            "pool_source": plan.diagnostics.get("pool_source"),
            "pool": plan.diagnostics.get("pool"),
            "trakt_items": sum(len(history.get(k) or []) for k in
                               ("watched", "ratings", "watchlist")),
            "seconds": round(time.monotonic() - started_at, 2),
        }

    @app.get("/manifest.json")
    async def bare_manifest() -> RedirectResponse:
        return RedirectResponse("/configure", status_code=302)

    _static_dir = Path(__file__).parent / "static"

    @app.get("/static/logo.png")
    async def logo() -> FileResponse:
        """The manifest advertises this URL; a 404 here makes clients reject the addon."""
        path = _static_dir / "logo.png"
        if not path.exists():
            return JSONResponse({"err": "logo missing"}, status_code=404)
        return FileResponse(path, media_type="image/png",
                            headers={"Cache-Control": "public, max-age=86400"})

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
        plan = await _cached_plan(config, media_type, row)
        metas = to_metas(plan.ranked, skip=safe_skip(extra), limit=PAGE_SIZE)
        return JSONResponse(catalog_response(metas), headers={"Cache-Control": "max-age=900"})

    async def _cached_plan(config: Config, media_type: str, row: str) -> RowPlan:
        """Rank once per (row, type, mode) and reuse for ~15 min (§11.3).

        Ranking 70k titles on every Stremio request would blow the client's ~10 s
        timeout on a cold Koyeb instance; the cache makes only the FIRST request
        pay the cost.
        """
        key = (row, media_type, config.mode)
        now = time.monotonic()
        cached = row_cache.get(key)
        if cached and now - cached[0] < ROW_CACHE_SECONDS:
            return cached[1]
        history = await current_history()
        plan = build_row(store, history, media_type, row, config)
        # a transient failure must not be cached for 15 minutes
        if plan.ranked:
            row_cache[key] = (now, plan)
        return plan

    # ------------------------------------------------- library events (writes)
    async def _library(config_blob: str, media_type: str, meta_id: str,
                       extra: dict[str, str]) -> JSONResponse:
        config, error = load_config(config_blob)
        if error is not None:
            return error
        if media_type not in VALID_TYPES:
            return JSONResponse({"success": False, "detail": "unsupported type"},
                                status_code=404)
        result = await handle_library_event(
            trakt, media_type, meta_id, extra.get("action", ""),
            video_id=extra.get("videoId") or None)
        # Stremio ignores the body; 200 keeps its logs clean.
        return JSONResponse(result.as_dict(), status_code=200)

    @app.get("/{config_blob}/library/{media_type}/{meta_id}/{extra}.json")
    @app.post("/{config_blob}/library/{media_type}/{meta_id}/{extra}.json")
    async def library_extra(config_blob: str, media_type: str, meta_id: str,
                            extra: str) -> JSONResponse:
        return await _library(config_blob, media_type, meta_id, parse_extra(extra))

    @app.get("/{config_blob}/library/{media_type}/{meta_id}.json")
    @app.post("/{config_blob}/library/{media_type}/{meta_id}.json")
    async def library_plain(config_blob: str, media_type: str, meta_id: str) -> JSONResponse:
        return await _library(config_blob, media_type, meta_id, {})

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



