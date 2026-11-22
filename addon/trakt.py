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

import logging
from dataclasses import dataclass, field
from typing import Any

import httpx

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

    @property
    def configured(self) -> bool:
        return bool(self.client_id and self.access_token)

    @classmethod
    def from_env(cls, env: dict[str, str]) -> "TraktConfig":
        return cls(client_id=env.get("TRAKT_CLIENT_ID", "").strip(),
                   access_token=env.get("TRAKT_ACCESS_TOKEN", "").strip(),
                   username=env.get("TRAKT_USERNAME", "").strip())


class TraktError(RuntimeError):
    pass


@dataclass
class TraktClient:
    config: TraktConfig
    timeout: float = 15.0
    calls: int = field(default=0, init=False, repr=False)

    def _headers(self) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "trakt-api-version": "2",
            "trakt-api-key": self.config.client_id,
            "Authorization": f"Bearer {self.config.access_token}",
            # Trakt's Cloudflare layer blocks clients with a missing or bot-like
            # User-Agent (error 1010). Their maintainers ask for an identifying
            # "App/version" string (trakt-api discussion #655).
            "User-Agent": USER_AGENT,
        }

    async def _get(self, path: str, **params: Any) -> list[dict[str, Any]]:
        if not self.config.configured:
            raise TraktError("Trakt is not configured (TRAKT_CLIENT_ID/TRAKT_ACCESS_TOKEN)")
        url = f"{API}{path}"
        pages: list[dict[str, Any]] = []
        for page in range(1, MAX_PAGES + 1):
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(url, headers=self._headers(),
                                            params={**params, "page": page,
                                                    "limit": PAGE_LIMIT})
            self.calls += 1
            if response.status_code == 401:
                raise TraktError("Trakt rejected the access token (401)")
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
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(f"{API}{path}", headers=self._headers(),
                                        params=params)
        self.calls += 1
        if response.status_code == 401:
            raise TraktError("Trakt rejected the access token (401)")
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
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(f"{API}/sync/last_activities",
                                        headers=self._headers())
        self.calls += 1
        if response.status_code == 401:
            raise TraktError("Trakt rejected the access token (401)")
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
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(f"{API}{path}", headers=self._headers(), json=payload)
        self.calls += 1
        if response.status_code == 401:
            raise TraktError("Trakt rejected the access token (401)")
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
