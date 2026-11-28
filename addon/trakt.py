"""Trakt client (PLAN §7) — serving-side sync.

Read-only by default: the addon needs the user's watched / watchlist / ratings /
hidden state to build taste and exclusions. Writes (hidden list) land in M2/M6.

Auth model:
  * `TRAKT_CLIENT_ID` is public-ish app identity, sent on every call.
  * The **access token** can bootstrap from the `TRAKT_ACCESS_TOKEN` env var, so a
    personal single-user deployment can go live without running the OAuth device
    flow yet (PLAN §7.1's flow is still the M2 deliverable and the durable path).

Tokens are never logged.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from .trakt_token import (TokenPair, TraktAuthError, exchange_refresh_token)

log = logging.getLogger(__name__)

API = "https://api.trakt.tv"
PAGE_LIMIT = 100
MAX_PAGES = 10
# /recommendations/* returns at most 100 items and is NOT paginated.
RECOMMENDATIONS_LIMIT = 100
# The `because` row shows one page; ask for a little more so filtering leaves enough.
RELATED_LIMIT = 40
# Identifying UA — required by Trakt (their CDN answers 403 "error code: 1010"
# to unknown / bot-like agents).
USER_AGENT = "MovieMatch/1.0 (Stremio addon; personal use)"


@dataclass(frozen=True)
class TraktConfig:
    client_id: str = ""
    access_token: str = ""
    username: str = ""
    client_secret: str = ""
    redirect_uri: str = ""
    refresh_token: str = ""

    @property
    def configured(self) -> bool:
        return bool(self.client_id and self.access_token)

    @property
    def can_refresh(self) -> bool:
        """Refresh needs a stored refresh_token plus the app's identity fields."""
        return bool(self.refresh_token and self.client_id and self.redirect_uri)

    @classmethod
    def from_env(cls, env: dict[str, str]) -> "TraktConfig":
        return cls(
            client_id=env.get("TRAKT_CLIENT_ID", "").strip(),
            access_token=env.get("TRAKT_ACCESS_TOKEN", "").strip(),
            username=env.get("TRAKT_USERNAME", "").strip(),
            client_secret=env.get("TRAKT_CLIENT_SECRET", "").strip(),
            redirect_uri=env.get("TRAKT_REDIRECT_URI", "").strip(),
            refresh_token=env.get("TRAKT_REFRESH_TOKEN", "").strip(),
        )


class TraktError(RuntimeError):
    pass


class TraktUnauthorized(TraktError):
    """HTTP 401: the access token is missing, expired or revoked.

    Distinct from other TraktErrors so `_with_token` can refresh-and-retry exactly
    once rather than guessing from the message.
    """


@dataclass
class TraktClient:
    config: TraktConfig
    timeout: float = 15.0
    calls: int = field(default=0, init=False, repr=False)
    # Set by make_app / tests. When present the token pair lives there and is
    # refreshed automatically; without it the client behaves exactly as before.
    store: Any = field(default=None, repr=False)
    _pair: Any = field(default=None, init=False, repr=False)
    _refresh_lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False,
                                        repr=False)
    _refresh_failures: int = field(default=0, init=False, repr=False)
    _refresh_count: int = field(default=0, init=False, repr=False)
    _last_error: str = field(default="", init=False, repr=False)

    # ------------------------------------------------------------------ token
    @property
    def current_token(self) -> str:
        if self._pair is not None and getattr(self._pair, "usable", False):
            return str(self._pair.access_token)
        return self.config.access_token

    def _headers(self) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "trakt-api-version": "2",
            "trakt-api-key": self.config.client_id,
            "Authorization": f"Bearer {self.current_token}",
            # Trakt's Cloudflare layer blocks clients with a missing or bot-like
            # User-Agent (error 1010). Their maintainers ask for an identifying
            # "App/version" string (trakt-api discussion #655).
            "User-Agent": USER_AGENT,
        }

    def _plain_headers(self) -> dict[str, str]:
        """Headers without Authorization, for the OAuth token endpoint."""
        headers = self._headers()
        headers.pop("Authorization", None)
        return headers

    async def _load_pair(self) -> None:
        """Load the stored pair, seeding it from env on first run."""
        if self._pair is not None:
            return
        if self.store is not None:
            stored = await self.store.load()
            if stored is not None:
                self._pair = stored
                return
        if self.config.refresh_token:
            # Seed the store once so later rotations have somewhere to live.
            pair = TokenPair(access_token=self.config.access_token,
                             refresh_token=self.config.refresh_token)
            self._pair = pair
            if self.store is not None:
                await self.store.save(pair)

    async def _ensure_token(self) -> None:
        """Refresh proactively when the stored token is past ~80 % of its life.

        Only for a token whose lifetime we actually KNOW (`expires_at` set from a
        real `expires_in`). A seeded env token has no expiry information, and
        refreshing on every call just to guess would burn the single-use refresh
        chain for nothing — in that case the 401 path in `_with_token` is the
        trigger, which costs one failed request and nothing else.
        """
        await self._load_pair()
        if self._pair is None:
            return
        if not self.config.can_refresh:
            return
        if not getattr(self._pair, "expires_at", 0):
            return
        if self._pair.expired_soon():
            await self.refresh()

    async def refresh(self) -> None:
        """Exchange the refresh token. Serialised; the pair rotates exactly once.

        Refresh tokens are single-use. Two concurrent refreshes would make one of
        them present an already-invalidated token and permanently break the chain,
        so the lock is mandatory. Inside the lock the STORE is re-read (not the
        in-memory pair, which every waiter loaded before the lock and which is
        therefore stale for all of them): if the stored token differs from the one
        we were about to exchange, another worker already rotated and we adopt it.
        """
        # Generation counter, NOT token values or object identity.
        #
        # Both of the obvious guards are wrong here:
        #  * comparing the stored token to the one we hold fails because
        #    `_load_pair()` re-reads the store, so a late waiter arrives already
        #    holding the token the winner wrote — the values match and it exchanges
        #    the fresh token, spending the chain once per waiter;
        #  * comparing `_pair` identity fails for the same reason: `_load_pair()`
        #    rebuilds the object on every call, so identity never survives.
        # Only a counter bumped exclusively on a successful rotation answers
        # "did anybody rotate while I waited?", which is exactly the question.
        generation = self._refresh_count
        await self._load_pair()
        # What we WOULD exchange if we turn out to be first. Taken from config, not
        # from `_pair`, precisely because `_load_pair()` has already replaced
        # `_pair` with whatever is stored — in which case this caller already sees
        # the other worker's token and there is nothing to adopt. Comparing the
        # store against `_pair` would therefore always match and adopt never fires.
        pending = str(self.config.refresh_token or "")
        async with self._refresh_lock:
            if self._refresh_count != generation:
                # Another caller rotated while we queued. Our token is spent; adopt
                # theirs instead of exchanging (which would 400 invalid_grant and,
                # before that, burn a token the winner still needs).
                await self._load_pair()
                return
            stored = (await self.store.load(fresh=True)
                      if self.store is not None else None)
            if stored is not None and stored.usable:
                stored_refresh = str(getattr(stored, "refresh_token", "") or "")
                if stored_refresh and stored_refresh != pending:
                    # Another WORKER (separate process) rotated: adopt, do not exchange.
                    self._pair = stored
                    return
            if not self.config.can_refresh:
                raise TraktAuthError(
                    "cannot refresh: TRAKT_REFRESH_TOKEN / TRAKT_CLIENT_ID / "
                    "TRAKT_REDIRECT_URI are not all set")
            try:
                pair = await exchange_refresh_token(
                    refresh_token=pending,
                    client_id=self.config.client_id,
                    client_secret=self.config.client_secret,
                    redirect_uri=self.config.redirect_uri,
                )
            except Exception as exc:
                self._refresh_failures += 1
                self._last_error = f"{type(exc).__name__}: {exc}"[:300]
                raise
            self._pair = pair
            self._refresh_failures = 0
            self._refresh_count += 1
            self._last_error = ""
            if self.store is not None:
                await self.store.save(pair)
            log.info("trakt token refreshed; valid for %.1f days",
                     (pair.lifetime or 0) / 86400)

    async def auth_status(self) -> dict[str, Any]:
        """Compact auth health for /health. Never performs a network call."""
        await self._load_pair()
        pair = self._pair
        if pair is None:
            return {"trakt_auth": "fixture" if not self.config.configured else "ok",
                    "trakt_refresh_failures": self._refresh_failures}
        remaining = (pair.expires_at - time.time()) if pair.expires_at else 0.0
        status = "ok"
        if self._refresh_failures:
            status = "refresh_failed"
        elif pair.expires_at and remaining <= 0:
            status = "expired"
        return {
            "trakt_auth": status,
            "trakt_token_expires_in_h": round(max(remaining, 0.0) / 3600, 1),
            "trakt_refresh_failures": self._refresh_failures,
            "trakt_refresh_enabled": self.config.can_refresh,
            "trakt_auth_error": self._last_error or None,
        }

    async def _with_token(self, call):
        """Run `call()`, refreshing at most ONCE across the whole call.

        Exactly one refresh per call is the invariant. Refresh tokens are
        single-use: a second exchange in the same call would present an
        already-spent token and permanently break the chain. So the retry is only
        allowed to refresh when the proactive step did NOT already rotate.
        If a 401 survives the retry, the token is genuinely bad and the caller sees
        the error instead of the client burning a second refresh.
        """
        did_refresh = False
        refreshes_before = self._refresh_count
        try:
            await self._ensure_token()
            # A rotation is only counted when an exchange actually happened;
            # seeding the pair from env on first use is not one.
            did_refresh = self._refresh_count > refreshes_before
        except TraktAuthError:
            raise
        except Exception as exc:  # noqa: BLE001 - a refresh blip must not 500
            log.warning("proactive refresh failed (%s); trying the call anyway",
                        type(exc).__name__)
        try:
            return await call()
        except TraktUnauthorized:
            if not self.config.can_refresh:
                raise
            if did_refresh:
                # We already rotated for this call; a second exchange would spend
                # an invalidated token. Surface the 401 instead.
                raise
            log.info("trakt 401; refreshing the token and retrying once")
            await self.refresh()
            return await call()

    async def _get(self, path: str, **params: Any) -> list[dict[str, Any]]:
        if not self.config.configured:
            raise TraktError("Trakt is not configured (TRAKT_CLIENT_ID/TRAKT_ACCESS_TOKEN)")
        return await self._with_token(lambda: self._get_raw(path, **params))

    async def _get_raw(self, path: str, **params: Any) -> list[dict[str, Any]]:
        url = f"{API}{path}"
        pages: list[dict[str, Any]] = []
        for page in range(1, MAX_PAGES + 1):
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(url, headers=self._headers(),
                                            params={**params, "page": page,
                                                    "limit": PAGE_LIMIT})
            self.calls += 1
            if response.status_code == 401:
                raise TraktUnauthorized("Trakt rejected the access token (401)")
            if response.status_code == 420:
                raise TraktError("Trakt rate limit hit (420)")
            if response.status_code != 200:
                raise TraktError(f"{path} -> {response.status_code}")
            chunk = response.json() or []
            pages.extend(chunk)
            total_pages = int(response.headers.get("x-pagination-page-count") or 1)
            if page >= total_pages or len(chunk) < PAGE_LIMIT:
                break
        return pages

    async def history(self, media_type: str = "movies") -> list[dict[str, Any]]:
        return await self._get(f"/sync/history/{media_type}", limit=PAGE_LIMIT)

    async def ratings(self, media_type: str = "movies") -> list[dict[str, Any]]:
        return await self._get(f"/sync/ratings/{media_type}")

    async def watchlist(self, media_type: str = "movies") -> list[dict[str, Any]]:
        return await self._get(f"/sync/watchlist/{media_type}")

    # ------------------------------------------------------- recommendations
    async def _get_once(self, path: str, **params: Any) -> list[dict[str, Any]]:
        """ONE request, no pagination loop. For endpoints with a hard item cap.

        `/recommendations/*` returns at most `limit` (max 100) items and is not
        paginated, so the `_get` loop would issue a pointless second request that
        is guaranteed empty. Returning non-list JSON as `[]` keeps callers simple.
        """
        if not self.config.configured:
            raise TraktError("Trakt is not configured (TRAKT_CLIENT_ID/TRAKT_ACCESS_TOKEN)")
        return await self._with_token(lambda: self._get_once_raw(path, **params))

    async def _get_once_raw(self, path: str, **params: Any) -> list[dict[str, Any]]:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(f"{API}{path}", headers=self._headers(),
                                        params=params)
        self.calls += 1
        if response.status_code == 401:
            raise TraktUnauthorized("Trakt rejected the access token (401)")
        if response.status_code == 420:
            raise TraktError("Trakt rate limit hit (420)")
        if response.status_code == 429:
            raise TraktError("Trakt rate limit exceeded (429)")
        if response.status_code != 200:
            raise TraktError(f"{path} -> {response.status_code}")
        data = response.json()
        return data if isinstance(data, list) else []

    async def recommendations(self, media_type: str = "movie",
                              limit: int = RECOMMENDATIONS_LIMIT,
                              *, ignore_watchlisted: bool = True,
                              ignore_collected: bool = True) -> list[dict[str, Any]]:
        """Personalized Trakt recommendations, in Trakt's own order.

        `media_type` here is the ADDON's vocabulary ("movie"/"series"); Trakt calls
        the endpoint `movies`/`shows`.

        `ignore_watchlisted` / `ignore_collected` drop titles the user already has
        queued or collected. Watched titles are excluded by Trakt itself — but the
        caller must STILL filter against current history at serve time, because a
        cached response can be hours old and the user may have watched something
        in the meantime.

        Each item is a movie/show object: {title, year, ids:{trakt,tmdb,imdb,slug}}
        plus `recommended_by` (why it was suggested) and `favorited_by` when the
        response is extended. Order is meaningful: earlier = stronger suggestion.
        """
        segment = "movies" if media_type in ("movie", "movies") else "shows"
        return await self._get_once(
            f"/recommendations/{segment}",
            limit=max(1, min(int(limit), RECOMMENDATIONS_LIMIT)),
            ignore_watchlisted="true" if ignore_watchlisted else "false",
            ignore_collected="true" if ignore_collected else "false",
        )

    async def related(self, media_type: str, ids: dict[str, Any],
                      limit: int = RELATED_LIMIT) -> list[dict[str, Any]]:
        """Titles related to ONE seed, for the `because` row.

        Trakt accepts a trakt id, slug or IMDb id in the path. We prefer the IMDb
        id because it is what the catalog already stores, and a slug would need an
        extra lookup. `limit` is applied client-side: /related has no limit param
        and returns a fixed list.
        """
        imdb = str((ids or {}).get("imdb") or "")
        trakt_id = (ids or {}).get("trakt")
        key = imdb if imdb.startswith("tt") else str(trakt_id or "")
        if not key:
            raise TraktError("related() needs an imdb or trakt id")
        segment = "movies" if media_type in ("movie", "movies") else "shows"
        items = await self._get_once(f"/{segment}/{key}/related")
        return items[:limit]

    async def last_activities(self) -> dict[str, Any]:
        """EVERYTHING that can invalidate a cached recommendation list, in one call.

        Marking something watched on trakt.tv immediately changes /recommendations
        (the watched title drops out, a new one appears). Without this, a 6-hour
        cache would leave Stremio stale for hours. This endpoint is tiny and is the
        cheapest possible "did anything change?" probe.

        Watched, rated, watchlisted and hidden are the four that matter: they change
        which titles are eligible AND retrain Trakt's model.
        """
        if not self.config.configured:
            raise TraktError("Trakt is not configured")
        return await self._with_token(self._last_activities_raw)

    async def _last_activities_raw(self) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(f"{API}/sync/last_activities",
                                        headers=self._headers())
        self.calls += 1
        if response.status_code == 401:
            raise TraktUnauthorized("Trakt rejected the access token (401)")
        if response.status_code != 200:
            raise TraktError(f"/sync/last_activities -> {response.status_code}")
        data = response.json()
        return data if isinstance(data, dict) else {}

    async def fetch_user_state(self) -> dict[str, Any]:
        """Everything `addon.rows.build_user_state` consumes, in one Trakt-shaped dict."""
        if not self.config.configured:
            raise TraktError("Trakt is not configured")
        watched = await self.history("movies") + await self.history("shows")
        rated = await self.ratings("movies") + await self.ratings("shows")
        listed = await self.watchlist("movies") + await self.watchlist("shows")
        return {"watched": watched, "ratings": rated, "watchlist": listed}

    # ------------------------------------------------------------- writes
    # NOTE: there is deliberately NO history-remove / unwatched support.
    # Deleting watch history is irreversible and a wrong removal corrupts the
    # user's taste profile, so that action is not implemented (by request).
    async def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        if not self.config.configured:
            raise TraktError("Trakt is not configured")
        return await self._with_token(lambda: self._post_raw(path, payload))

    async def _post_raw(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(f"{API}{path}", headers=self._headers(), json=payload)
        self.calls += 1
        if response.status_code == 401:
            raise TraktUnauthorized("Trakt rejected the access token (401)")
        if response.status_code == 403:
            body = response.text[:200]
            if "1010" in body:
                # Cloudflare blocked the client itself — NOT a permission problem.
                raise TraktError(
                    f"Trakt's CDN blocked this request (403 error code: 1010): {body}")
            raise TraktError(
                "Trakt refused the write (403): the app needs write permissions — "
                f"re-authorize with a Trakt app that has write access. {body}")
        if response.status_code == 420:
            raise TraktError("Trakt limit reached (420)")
        if response.status_code >= 300:
            raise TraktError(f"POST {path} -> {response.status_code}: {response.text[:160]}")
        return response.json() if response.content else {}

    async def add_history(self, media: dict[str, Any],
                          watched_at: str = "now") -> dict[str, Any]:
        """POST /sync/history — mark watched.

        `media` is the {"movie"/"show": {...}} wrapper also used by the watchlist
        calls; Trakt's history body wants the INNER object plus `watched_at`:
            {"movies": [{"ids": {"imdb": "tt..."}, "watched_at": "now"}]}
        Wrapping it as {"movies": [{"movie": {...}}]} puts the id one level too
        deep, Trakt answers 201 with the item in `not_found`, and nothing is
        recorded — the exact failure seen on the live deployment.

        Trakt does NOT de-duplicate: duplicate item+watched_at pairs create
        duplicate plays, so callers should send a stable value when replaying.
        """
        key = "movie" if "movie" in media else "show"
        inner = dict(media[key])
        inner["watched_at"] = watched_at
        return await self._post("/sync/history", {f"{key}s": [inner]})

    async def add_watchlist(self, media: dict[str, Any]) -> dict[str, Any]:
        """POST /sync/watchlist — add to watchlist (Trakt answers 201).

        Unwrap `media` exactly like `add_history` does. The watchlist body wants
        the INNER object:
            {"movies": [{"ids": {"imdb": "tt..."}}]}
        Passing the wrapper instead sends {"movies": [{"movie": {...}}]}, which
        Trakt accepts with a 2xx while reporting the item in `not_found` — the id
        sits one level too deep, so nothing is written and nothing errors. That
        was the live "libraryAdd NOT applied" failure.
        """
        key = "movie" if "movie" in media else "show"
        return await self._post("/sync/watchlist", {f"{key}s": [dict(media[key])]})

    async def remove_watchlist(self, media: dict[str, Any]) -> dict[str, Any]:
        """POST /sync/watchlist/remove — note: a POST to /remove, not a DELETE.

        Same unwrapping rule as `add_watchlist`; see the comment there.
        """
        key = "movie" if "movie" in media else "show"
        return await self._post("/sync/watchlist/remove", {f"{key}s": [dict(media[key])]})

    async def health(self) -> dict[str, Any]:
        if not self.config.configured:
            return {"configured": False, "ok": False, "mode": "unconfigured"}
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(f"{API}/users/settings", headers=self._headers())
            if response.status_code == 401:
                return {"configured": True, "ok": False, "trakt_auth": "invalid_token"}
            return {"configured": True, "ok": response.status_code == 200,
                    "trakt_auth": "ok" if response.status_code == 200 else "error"}
        except Exception as exc:  # noqa: BLE001 - health must not raise
            return {"configured": True, "ok": False, "error": type(exc).__name__}
