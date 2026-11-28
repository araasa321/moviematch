"""FastAPI addon server (PLAN §10.1).

    GET /{config}/manifest.json
    GET /{config}/catalog/{type}/{id}.json
    GET /{config}/catalog/{type}/{id}/{extra}.json     extra = "skip=100&genre=Sci-Fi"
    GET /health
    GET /configure                                     custom config page (M4)
    GET /configure.json                                install-link builder for that page

Run locally against the fixture catalog:
    uvicorn addon.main:app --reload --port 8000
"""

from __future__ import annotations

import asyncio
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
from fastapi.templating import Jinja2Templates

from . import __version__
from .config import (DEFAULT_LANGS, DEFAULT_LAYOUT, LANG_NAMES, SUPPORTED_LANGS,
                     VALID_LAYOUTS, VALID_MODES, Config, ConfigError, secret_ok,
                     _LANG_CODE_RE)
from .languages import THIN_CATALOG_TITLES, LanguageDetector
from .languages import build_language_list as build_language_list
from .fixtures import catalog as fixture_catalog
from .fixtures import trakt_history
from .library import handle_library_event
from .memory_store import MemoryStore, load_records_from_json
from .qdrant import QdrantClient, QdrantConfig
from .recommendations import (ACTIVITY_TTL, RECOMMENDATIONS_MAX_STALE,
                              RECOMMENDATIONS_TTL, RecommendationCache,
                              activities_fingerprint, ordered_records, to_meta)
from .rows import RowPlan, build_user_state
from .store import QdrantStore
from .stremio import (PAGE_SIZE, build_manifest, catalog_response, parse_extra,
                      safe_skip, split_row_id)
from .trakt import TraktClient, TraktConfig
from .trakt_token import TokenStore

# Trakt decides what is in these rows now (#4b follow-up):
#   foryou  -> /recommendations/{movies|shows}, Trakt's order preserved
#   because -> /{movies|shows}/:id/related for the most recent watch
# `new4you` was removed with the ranker: it had no implementation and folded into
# the recommendation row.
VALID_ROWS = {"foryou", "because"}
VALID_TYPES = {"movie", "series"}
DEV_SECRET = "dev-secret-please-set-INSTALL_SECRET"
# Short by design: a cache entry must not outlive one sync cycle, so a user who
# fixes a bad Trakt token sees the effect within minutes.
HISTORY_TTL = 600.0
_TEMPLATE_DIR = Path(__file__).parent / "templates"

log = logging.getLogger(__name__)


def _display_lang(code: str) -> str:
    """Human name for a language code, falling back to the uppercase code.

    Not every code TMDB emits is in LANG_NAMES, and the page must still be able
    to name a language it discovered in the catalog.
    """
    from .languages import _display_name

    return _display_name(code)


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
             trakt: TraktClient | None = None,
             languages: Any | None = None) -> FastAPI:
    """Factory so tests and smoke runs inject data instead of calling TMDB/Trakt.

    When `store` is omitted and QDRANT_URL is set, serving reads the real catalog
    from Qdrant; otherwise it falls back to the bundled fixture catalog.

    `languages` overrides the LanguageDetector /configure uses. When omitted it
    faceted Qdrant if configured; with the fixture store it reports "detection
    unavailable" and the page falls back to the defaults.
    """
    qdrant_store = QdrantStore.from_env(os.environ) if store is None else None
    store = store if store is not None else (qdrant_store or _fixture_store(dim))
    log.info("serving store: %s", "qdrant" if qdrant_store else "fixture")
    trakt = trakt or TraktClient(TraktConfig.from_env(os.environ))
    # Give the client somewhere to persist rotated OAuth tokens. Without this the
    # token pair lives only in memory and every restart falls back to the (now
    # expired) env token — which is exactly how the addon sat on a 401 for days.
    if getattr(trakt, "store", None) is None:
        token_qdrant_cfg = QdrantConfig.from_env(os.environ)
        trakt.store = TokenStore(QdrantClient(token_qdrant_cfg)
                                 if token_qdrant_cfg.configured else None)
    if languages is None:
        qdrant_cfg = QdrantConfig.from_env(os.environ)
        languages = LanguageDetector(QdrantClient(qdrant_cfg)
                                     if qdrant_cfg.configured else None)
    started = datetime.now(timezone.utc)
    # Trakt-response cache and its per-(type,row) locks, process-wide.
    rec_cache = RecommendationCache()
    rec_locks: dict[tuple[str, str], asyncio.Lock] = {}
    # (saved_at, plan, ttl) — retained for the /health/deep diagnostic path only;
    # the serving rows no longer go through the ranker.
    row_cache: dict[tuple, tuple[float, RowPlan, float]] = {}
    # Process-wide, not per-request. `trakt.fetch_user_state()` does up to
    # history x2 + ratings x2 + watchlist x2, each paginated 100/page up to 10
    # pages. build_manifest advertises up to 12 catalogs and Stremio fetches them
    # together on a cold home-screen open, so without this lock a cold instance
    # fires up to 12 concurrent full Trakt syncs.
    history_lock = asyncio.Lock()
    history_cache: dict[str, Any] = {}
    history_is_fallback = False

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
    app.state.languages = languages
    language_detector = languages

    async def _sync_history() -> dict[str, Any]:
        """Real Trakt sync. Raises on failure — the caller decides what to do."""
        return await trakt.fetch_user_state()

    async def current_history() -> tuple[dict[str, Any], bool]:
        """Trakt history, singleflighted. Never raises.

        Returns `(history, is_fallback)`.

        `is_fallback` is True when the returned history is NOT the user's real
        Trakt state — either the Trakt sync failed and we served the bundled
        fixture, or there is no history provider at all. Callers that cache
        anything derived from history MUST honour it: a single transient Trakt
        error must not pin fake recommendations on the user's home screen.

        Concurrency: concurrent callers share ONE in-flight sync. The lock is held
        across exactly one `await _sync_history()` with no nested awaits, so it
        cannot deadlock. Callers are re-checked against the cache after acquiring
        the lock (double-checked locking): request 2..12 wake up, see the entry
        request 1 just wrote, and return immediately.

        The fixture fallback deliberately does NOT write `history_cache` — a bad
        token would otherwise pin a fake 3-title history for HISTORY_TTL.
        """
        nonlocal history_is_fallback
        if history_provider is not None:           # tests inject a plain callable
            return history_provider(), False
        if not trakt.config.configured:
            return trakt_history(), True
        async with history_lock:
            fresh = history_cache.get("value")
            if fresh is not None and time.monotonic() - history_cache["at"] < HISTORY_TTL:
                return fresh, history_is_fallback
            try:
                data = await _sync_history()
            except Exception as exc:                # noqa: BLE001 - never 500 Stremio
                log.warning("trakt sync failed (%s); serving fixture history",
                            type(exc).__name__)
                history_is_fallback = True
                return trakt_history(), True
            history_cache.update(value=data, at=time.monotonic())
            history_is_fallback = False
            return data, False

    app.state.history_provider = current_history
    app.state.history_cache = history_cache
    # Exposed for /health/deep and for tests that need to age the cache.
    app.state.rec_cache = rec_cache

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
        # Real auth state, not just "an env var exists". This key used to read
        # "configured" while every Trakt call was 401ing, which is why a dead token
        # went unnoticed for days. No network call: it reports what the client
        # already knows.
        try:
            auth = await trakt.auth_status()
        except Exception as exc:  # noqa: BLE001 - health must never 500
            auth = {"trakt_auth": "unknown", "trakt_auth_error": type(exc).__name__}
        return {
            "status": "ok",
            "version": __version__,
            "store": "qdrant" if isinstance(store, QdrantStore) else "fixture",
            "qdrant": qdrant_status,
            "trakt": "trakt" if trakt.config.configured else "fixture",
            **auth,
            "catalog_size": total,
            "store_size": store_size,
            "row_cache_size": len(row_cache),
            "row_cache_keys": sorted("/".join(k) for k in row_cache),
            "history_cached": bool(history_cache),
            # True while serving the bundled fixture because the Trakt sync is
            # failing or unconfigured — a stuck/blind sync is visible here without
            # a deep probe. No new I/O.
            "history_is_fallback": history_is_fallback,
            "started_at": started.isoformat(),
            "secret_configured": current_secret() != DEV_SECRET,
        }

    @app.get("/health/deep")
    async def health_deep() -> dict[str, Any]:
        """Expensive diagnostics: Trakt sync + how much of each row survives the join.

        `in_catalog` is the number that matters now: Trakt recommends titles the
        45k crawl may never have fetched, and those are skipped. A persistently low
        ratio means the crawl is missing what Trakt thinks you should watch.
        """
        history, _ = await current_history()
        started_at = time.monotonic()
        rows: dict[str, Any] = {}
        for media_type in ("movie", "series"):
            for media_type_, kind in ((media_type, "recommendations"),
                                      (media_type, "because")):
                items = await _trakt_items(media_type_, kind)
                state = build_user_state(store, history)
                excluded = set(state.watched) | set(state.watchlist) | set(state.hidden)
                records = ordered_records(items, store, media_type_, excluded=excluded)
                rows[f"{media_type_}/{kind}"] = {
                    "trakt_returned": len(items),
                    "in_catalog": len(records),
                    "hit_rate": round(len(records) / len(items), 3) if items else None,
                }
        return {
            "positive_signals": sum(1 for s in build_user_state(store, history).signals),
            "trakt_items": sum(len(history.get(k) or []) for k in
                               ("watched", "ratings", "watchlist")),
            "rows": rows,
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

    # ------------------------------------------------------------- /configure
    # PLAN §10.1/M4: the real configuration page. Per the Stremio docs
    # (stremio-addon-sdk/docs/advanced.md, "Creating Addon Configuration Pages")
    # a custom page must embed the user's choices in the addon repository URL and
    # offer a `stremio://host/<blob>/manifest.json` install link. That is exactly
    # what /configure and /configure.json below produce.
    def _templates() -> Jinja2Templates:
        return Jinja2Templates(directory=str(_TEMPLATE_DIR))

    def _https_manifest_url(request: Request, blob: str) -> tuple[str, str]:
        """Return (https manifest URL, stremio deep link) for a config blob."""
        origin = base_url(request)
        https_url = f"{origin}/{blob}/manifest.json"
        # Stremio refuses plain http; Koyeb terminates TLS and forwards the
        # public scheme, but a local run still says http://, so upgrade it.
        if not https_url.startswith("https://"):
            https_url = https_url.replace("http://", "https://", 1)
        return https_url, "stremio://" + https_url[len("https://"):]

    def _self_endpoint(request: Request, path: str) -> str:
        """Path the page should call back on — deliberately RELATIVE.

        Getting this wrong has now broken the page twice in opposite directions:

          * https-upgrading it broke every local run (fetch to a TLS port that is
            not serving TLS -> ERR_SSL_PROTOCOL_ERROR);
          * trusting `request.base_url` broke production, because behind a TLS
            proxy (Koyeb) the container sees `http://` while the browser is on
            `https://`, so the fetch was blocked as Mixed Content.

        A relative path sidesteps the whole question: the browser resolves it
        against the page's own scheme and origin, so it can never mismatch. There
        is nothing left for the server to guess.
        """
        return path

    @app.get("/configure")
    async def configure(request: Request) -> HTMLResponse:
        blob = Config(secret=current_secret(),
                      mode=os.environ.get("MODE_DEFAULT", "blockbuster")).encode()
        # Self-check: if this round-trips, a decoded config failure is transport corruption.
        try:
            round_trip_ok = Config.decode(blob).secret == current_secret()
        except ConfigError:
            round_trip_ok = False
        https_url, _ = _https_manifest_url(request, blob)
        # The catalog decides which languages are offered (see addon/languages.py).
        # Detection never raises: if it fails the page falls back to the default.
        detected = await language_detector.languages()
        offered_codes = {lang.code for lang in detected}
        selected = [code for code in DEFAULT_LANGS if code in offered_codes] or list(
            DEFAULT_LANGS)
        # starlette >= 0.36 takes `request` first, with a context dict that does
        # NOT need a "request" key. This is the signature `requirements.txt`
        # pins for (`starlette>=0.37,<1.0`): 0.36/0.37 still accept the old
        # (name, {"request": ...}) form as a deprecated overload, and 1.0.0
        # removed it outright. Under starlette 0.27 this call raised
        # `ValueError: context must include a "request" key`, which is why the
        # suite needs the pin. Do not "fix" this back to the old shape.
        return _templates().TemplateResponse(request, "configure.html", {
            "langs": detected,
            "detected": language_detector.available,
            "lang_count": len(detected),
            "selected_langs": set(selected),
            "layouts": [
                ("merged", "Mixed",
                 "One For You row pooling every selected language. The ranking "
                 "decides what is good, so a strong foreign title can beat a weak "
                 "English one."),
                ("separate", "Separate rows",
                 "Each row repeated per language. One language behaves like Mixed."),
            ],
            "selected_layout": DEFAULT_LAYOUT,
            "modes": [
                ("blockbuster", "Blockbusters", "Favour widely known, high-vote titles."),
                ("gems", "Hidden gems", "Favour well-rated titles with fewer votes."),
            ],
            "selected_mode": os.environ.get("MODE_DEFAULT", "blockbuster"),
            "update_url": _self_endpoint(request, "/configure.json"),
            "https_url_base": https_url,
            "origin": base_url(request),
            "round_trip_ok": round_trip_ok,
        })

    @app.get("/configure.json")
    async def configure_json(request: Request) -> JSONResponse:
        """Build the install link for the choices in the query string.

        The page calls this on every change rather than encoding the config in
        JavaScript: one encoder (addon.config.Config) means the preview can never
        drift from what a real install produces.
        """
        params = request.query_params
        chosen = [code.strip().lower() for code in params.getlist("langs") if code.strip()]
        if not chosen:
            return JSONResponse({"ok": False,
                                 "error": "Select at least one language."})
        # Validate shape only; which codes are "real" is the catalog's business.
        bad = [code for code in chosen if not _LANG_CODE_RE.fullmatch(code)]
        if bad:
            return JSONResponse({"ok": False,
                                 "error": f"Invalid language code: {', '.join(bad)}"})
        layout = params.get("layout") or DEFAULT_LAYOUT
        mode = params.get("mode") or os.environ.get("MODE_DEFAULT", "blockbuster")
        if layout not in VALID_LAYOUTS or mode not in VALID_MODES:
            return JSONResponse({"ok": False, "error": "Unknown option."})

        # Warn (do not block) about languages that were never crawled: the user may
        # be configuring before the ingest run finishes, and a hard rejection here
        # would make the page unusable during a bootstrap.
        facets = await language_detector.facets()
        detected = build_language_list(facets)
        known = {lang.code: lang for lang in detected}
        try:
            config = Config.from_dict({"secret": current_secret(), "mode": mode,
                                       "langs": chosen, "layout": layout})
        except ConfigError as exc:
            return JSONResponse({"ok": False, "error": str(exc)})

        blob = config.encode()
        https_url, stremio_url = _https_manifest_url(request, blob)
        names = ", ".join(known[c].name if c in known else _display_lang(c)
                          for c in config.langs)
        layout_label = "Mixed" if config.layout == "merged" else "one row per language"
        mode_label = "Hidden gems" if config.mode == "gems" else "Blockbusters"
        summary = (f"<b>{len(config.langs)}</b> language"
                   f"{'' if len(config.langs) == 1 else 's'} ({names}) · "
                   f"<b>{layout_label}</b> · <b>{mode_label}</b>")

        warnings: list[str] = []
        foreign = [c for c in config.langs if c != "en"]
        if facets is not None:
            # Only claim "nothing crawled" when we actually read the catalog.
            # With detection unavailable every count is 0 and the claim would be
            # both false and alarming.
            missing = [c for c in config.langs if c not in known or not known[c].count]
            if missing:
                labels = ", ".join(_display_lang(c) for c in missing)
                warnings.append(
                    f"No titles crawled yet for {labels}. Run the bootstrap workflow "
                    f"with --langs '{' '.join(config.langs)}' to fill them; until then "
                    "those rows will be empty.")
            thin = [c for c in config.langs
                    if c in known and 0 < known[c].count < THIN_CATALOG_TITLES]
            if thin:
                labels = ", ".join(f"{_display_lang(c)} ({known[c].count:,})" for c in thin)
                warnings.append(f"Thin catalog: {labels}. Rows may run out quickly.")
        elif foreign:
            # Detection unavailable AND a foreign language was chosen: we cannot
            # verify it exists, and staying silent would let a user install
            # foreign rows that may never fill.
            labels = ", ".join(_display_lang(c) for c in foreign)
            warnings.append(
                f"Could not read the catalog to confirm {labels} was crawled. "
                f"If those rows look empty, run the bootstrap workflow with "
                f"--langs '{' '.join(config.langs)}'.")
        return JSONResponse({"ok": True, "https_url": https_url,
                             "stremio_url": stremio_url, "blob": blob,
                             "summary": summary, "warning": " ".join(warnings)})

    @app.get("/{config_blob}/manifest.json")
    async def manifest(request: Request, config_blob: str) -> JSONResponse:
        config, error = load_config(config_blob)
        if error is not None:
            return error
        history, _ = await current_history()
        seeds = _seed_names(store, history, config)
        payload = build_manifest(base_url(request), config_blob, seeds,
                                 langs=config.langs, layout=config.layout)
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
        base_row, lang = split_row_id(row)
        if media_type not in VALID_TYPES or base_row not in VALID_ROWS:
            return JSONResponse({"metas": []}, status_code=404)
        metas = await _trakt_row(media_type, base_row)
        page_ = metas[safe_skip(extra): safe_skip(extra) + PAGE_SIZE]
        # 60s, not the old 900: the row is expected to change the moment something
        # is watched on trakt.tv, and the server-side activity probe already keeps
        # the Trakt call count tiny. A long max-age would freeze the client copy.
        return JSONResponse(catalog_response(page_),
                            headers={"Cache-Control": "max-age=60"})

    async def _probe_activities() -> bool:
        """Ask Trakt whether anything changed. True when cached lists were dropped.

        This is what makes "mark watched on trakt.tv -> Stremio updates in ~a
        minute" work. It is a tiny GET compared with a full recommendation fetch,
        and it also re-filters the history cache, so the watched title disappears
        from rows even before the recommendation list is refetched.

        A failure here is deliberately silent: the recommendation fetch has its own
        error handling, and a probe must never be the reason a row fails.
        """
        nonlocal history_is_fallback
        if not trakt.config.configured:
            return False
        if not rec_cache.needs_activity_check():
            return False
        try:
            activities = await trakt.last_activities()
        except Exception as exc:  # noqa: BLE001 - never 500 Stremio
            # WARNING, not DEBUG: a failing probe is what silently froze this addon
            # for days. It is rate-limited to once a minute, so the log stays cheap.
            log.warning("last_activities probe failed (%s: %s); keeping cache",
                        type(exc).__name__, str(exc)[:160])
            return False
        changed = rec_cache.note_activities(activities_fingerprint(activities))
        # The history cache is what rows are filtered by; drop it so a title just
        # marked watched stops appearing immediately, without waiting for HISTORY_TTL.
        history_cache.clear()
        history_is_fallback = False
        return changed

    async def _trakt_items(media_type: str, kind: str) -> list[dict[str, Any]]:
        """Trakt ids for a row, from cache when fresh.

        Singleflighted per (media_type, kind) so a home-screen open that fires
        several rows does not turn into several Trakt calls. On failure the last
        good list is served (up to RECOMMENDATIONS_MAX_STALE); with no good list at
        all the row is simply empty — the addon never 500s because Trakt is down.

        The activity probe runs BEFORE the fresh-cache check, not inside the miss
        path: a 6-hour-fresh list must still be invalidated the moment something is
        watched on trakt.tv. It is rate-limited to one tiny GET per ACTIVITY_TTL, so
        the common case (nothing changed) still costs zero recommendation fetches.
        """
        # Probe first: a 6-hour-fresh list must still be invalidated the moment
        # something is watched on trakt.tv. Rate-limited internally, so the common
        # case (nothing changed) still costs zero recommendation fetches.
        await _probe_activities()
        cached = rec_cache.fresh(media_type, kind)
        if cached is not None:
            return cached
        lock = rec_locks.setdefault((media_type, kind), asyncio.Lock())
        async with lock:
            cached = rec_cache.fresh(media_type, kind)
            if cached is not None:
                return cached
            try:
                if kind == "because":
                    items = await _because_items(media_type)
                else:
                    items = await trakt.recommendations(media_type)
            except Exception as exc:  # noqa: BLE001 - never 500 Stremio
                log.warning("trakt %s fetch failed (%s); serving stale if any",
                            kind, type(exc).__name__)
                return rec_cache.stale(media_type, kind) or []
            rec_cache.put(media_type, kind, items)
            return items

    async def _because_items(media_type: str) -> list[dict[str, Any]]:
        """Titles related to the most recent watch, via Trakt /related."""
        history, _ = await current_history()
        state = build_user_state(store, history)
        pid = state.last_watched.get(media_type)
        record = store.get(pid) if pid else None
        if record is None:
            return []
        imdb = str(record.payload.get("imdb_id") or "")
        if not imdb:
            return []
        return await trakt.related(media_type, {"imdb": imdb})

    async def _trakt_row(media_type: str, row: str) -> list[dict[str, Any]]:
        """Trakt's order, joined to the catalog, never re-ranked."""
        kind = "recommendations" if row == "foryou" else row
        items = await _trakt_items(media_type, kind)
        history, _ = await current_history()
        state = build_user_state(store, history)
        excluded = set(state.watched) | set(state.watchlist) | set(state.hidden)
        records = ordered_records(items, store, media_type, excluded=excluded)
        return [to_meta(r) for r in records]


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



