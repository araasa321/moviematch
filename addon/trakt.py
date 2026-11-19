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
        """POST /sync/watchlist — add to watchlist (Trakt answers 201)."""
        return await self._post("/sync/watchlist",
                                {"movies": [media]} if "movie" in media else {"shows": [media]})

    async def remove_watchlist(self, media: dict[str, Any]) -> dict[str, Any]:
        """POST /sync/watchlist/remove — note: a POST to /remove, not a DELETE."""
        return await self._post("/sync/watchlist/remove",
                                {"movies": [media]} if "movie" in media else {"shows": [media]})

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
