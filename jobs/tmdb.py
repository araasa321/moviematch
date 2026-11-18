"""TMDB client for the ingest job (PLAN §2 pacing, §6.2 bootstrap, §6.3 nightly).

Only httpx is needed. The client is deliberately thin: pacing, retry/backoff,
and the handful of endpoints the job actually uses. Everything that turns TMDB
JSON into catalog records lives in `policy.normalize_payload`.
"""

from __future__ import annotations

import asyncio
import gzip
import io
import json
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

import httpx

API = "https://api.themoviedb.org/3"
EXPORTS = "http://files.tmdb.org/p/exports"

DEFAULT_RATE = 15.0          # req/s (PLAN §20 TMDB_PACING)
MAX_RETRIES = 5
DETAILS_APPEND = {
    "movie": "credits,keywords,external_ids",
    "series": "aggregate_credits,keywords,external_ids",
}


class TMDBError(RuntimeError):
    pass


@dataclass
class TMDBClient:
    token: str
    rate: float = DEFAULT_RATE
    timeout: float = 30.0
    max_retries: int = MAX_RETRIES
    _client: httpx.AsyncClient | None = field(default=None, repr=False)
    _interval: float = field(init=False, repr=False)
    _next_at: float = field(default=0.0, init=False, repr=False)
    calls: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self._interval = 1.0 / max(self.rate, 0.1)

    async def __aenter__(self) -> "TMDBClient":
        self._client = httpx.AsyncClient(
            base_url=API,
            timeout=self.timeout,
            headers={"Authorization": f"Bearer {self.token}", "Accept": "application/json"},
        )
        return self

    async def __aexit__(self, *exc: object) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def _pace(self) -> None:
        now = time.monotonic()
        if now < self._next_at:
            await asyncio.sleep(self._next_at - now)
        self._next_at = max(now, self._next_at) + self._interval

    async def get(self, path: str, **params: Any) -> dict[str, Any]:
        """GET with pacing, retry on 429/5xx and Retry-After support."""
        if self._client is None:
            raise TMDBError("client not opened; use `async with TMDBClient(...)`")
        clean = {k: v for k, v in params.items() if v is not None}
        delay = 1.0
        last: Exception | None = None
        for attempt in range(self.max_retries):
            await self._pace()
            try:
                self.calls += 1
                response = await self._client.get(path, params=clean)
            except httpx.HTTPError as exc:
                last = exc
                await asyncio.sleep(delay)
                delay = min(delay * 2, 30.0)
                continue
            if response.status_code == 200:
                return response.json()
            if response.status_code == 429 or response.status_code >= 500:
                retry_after = response.headers.get("retry-after")
                wait = float(retry_after) if retry_after else delay
                await asyncio.sleep(min(wait, 60.0))
                delay = min(delay * 2, 30.0)
                last = TMDBError(f"{response.status_code} on {path}")
                continue
            if response.status_code == 404:
                raise TMDBError(f"404 for {path}")
            raise TMDBError(f"{response.status_code} for {path}: {response.text[:200]}")
        raise TMDBError(f"giving up on {path} after {self.max_retries} tries: {last}")

    # -------------------------------------------------------------- endpoints
    async def details(self, media_type: str, tmdb_id: int) -> dict[str, Any]:
        segment = "tv" if media_type == "series" else "movie"
        # external_ids gives the imdb_id, which Stremio requires (§5)
        data = await self.get(f"/{segment}/{tmdb_id}",
                              append_to_response=DETAILS_APPEND[media_type])
        data.setdefault("media_type", media_type)
        return data

    async def discover(self, media_type: str, page: int, *, sort_by: str = "popularity.desc",
                       vote_count_gte: int | None = None, **extra: Any) -> dict[str, Any]:
        segment = "tv" if media_type == "series" else "movie"
        params: dict[str, Any] = {"page": page, "sort_by": sort_by,
                                  "vote_count.gte": vote_count_gte}
        params.update(extra)
        return await self.get(f"/discover/{segment}", **params)

    async def iter_discover(self, media_type: str, *, max_pages: int = 500,
                            **kwargs: Any) -> AsyncIterator[dict[str, Any]]:
        """Yield result objects page by page until TMDB runs out of pages."""
        for page in range(1, max_pages + 1):
            data = await self.discover(media_type, page, **kwargs)
            results = data.get("results") or []
            if not results:
                return
            for item in results:
                yield item
            if page >= int(data.get("total_pages") or 1):
                return

    async def trending(self, media_type: str, window: str = "week") -> list[dict[str, Any]]:
        segment = "tv" if media_type == "series" else "movie"
        data = await self.get(f"/trending/{segment}/{window}")
        return data.get("results") or []


async def download_export(media_type: str, day: str, *, lookback_days: int = 3) -> list[dict]:
    """TMDB daily ID export (PLAN §6.3 step 1), with yesterday-fallback.

    Args:
        day: `MM_DD_YYYY` of the preferred file.
        lookback_days: how many prior days to try before giving up.
    """
    prefix = "movie_ids" if media_type == "movie" else "tv_series_ids"
    from datetime import datetime, timedelta

    base = datetime.strptime(day, "%m_%d_%Y")
    async with httpx.AsyncClient(timeout=120.0) as client:
        for back in range(lookback_days):
            stamp = (base - timedelta(days=back)).strftime("%m_%d_%Y")
            url = f"{EXPORTS}/{prefix}_{stamp}.json.gz"
            response = await client.get(url)
            if response.status_code == 200:
                payload = gzip.decompress(io.BytesIO(response.content).read()).decode("utf-8")
                return [json.loads(line) for line in payload.splitlines() if line.strip()]
            if response.status_code != 404:
                raise TMDBError(f"{response.status_code} downloading {url}")
    raise TMDBError(f"no {prefix} export found within {lookback_days} days of {day}")
