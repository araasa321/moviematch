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

DEFAULT_RATE = 30.0          # req/s; TMDB's documented ceiling is ~40 req/s
MAX_RETRIES = 5
LIST_PAGE_SIZE = 20          # TMDB's v3 list endpoint returns 20 items per page
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
    concurrency: int = 8          # in-flight requests; keeps rate at `rate` but hides latency
    _client: httpx.AsyncClient | None = field(default=None, repr=False)
    _interval: float = field(init=False, repr=False)
    _next_at: float = field(default=0.0, init=False, repr=False)
    _lock: asyncio.Lock | None = field(default=None, init=False, repr=False)
    calls: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self._interval = 1.0 / max(self.rate, 0.1)

    async def __aenter__(self) -> "TMDBClient":
        self._client = httpx.AsyncClient(
            base_url=API,
            timeout=self.timeout,
            limits=httpx.Limits(max_connections=self.concurrency + 2,
                                max_keepalive_connections=self.concurrency + 2),
            headers={"Authorization": f"Bearer {self.token}", "Accept": "application/json"},
        )
        self._lock = asyncio.Lock()
        return self

    async def __aexit__(self, *exc: object) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def _pace(self) -> None:
        """Reserve this call's slot at `rate` req/s.

        Lock-protected because several requests are now in flight at once: without
        it, concurrent callers would all observe the same `_next_at` and burst
        past the rate limit. Each caller takes its own slot, then sleeps until
        that slot arrives — so the *average* rate is respected while latency
        overlaps instead of serialising every call.
        """
        lock = self._lock or asyncio.Lock()
        async with lock:
            now = time.monotonic()
            slot = max(now, self._next_at)
            self._next_at = slot + self._interval
        delay = slot - time.monotonic()
        if delay > 0:
            await asyncio.sleep(delay)

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
                            page_concurrency: int = 6,
                            **kwargs: Any) -> AsyncIterator[dict[str, Any]]:
        """Yield result objects page by page until TMDB runs out of pages.

        Pages are fetched in small concurrent windows instead of strictly
        one-by-one. `_pace()` still books every request a slot at `rate` req/s and
        the client retries 429 with `Retry-After`, so the *request rate* is
        unchanged — the win is that ~200 ms of latency per page stops adding up
        on a 500-page crawl.

        Page 1 is always fetched alone: it is the only page that reports
        `total_pages`, and probing it first means a 3-page catalogue costs 3
        requests instead of a full window of wasted ones.
        """
        first = await self.discover(media_type, 1, **kwargs)
        total_pages = min(max_pages, max(1, int(first.get("total_pages") or 1)))
        results = first.get("results") or []
        if not results:
            return
        for item in results:
            yield item

        page = 2
        while page <= total_pages:
            window = list(range(page, min(page + page_concurrency, total_pages + 1)))
            batch = await asyncio.gather(
                *(self.discover(media_type, p, **kwargs) for p in window),
                return_exceptions=True)
            for data in batch:
                if isinstance(data, BaseException):
                    # A page that failed after all retries aborts the crawl rather
                    # than silently skipping a chunk of the catalog (a hole here
                    # would never be re-fetched, because the titles are missing
                    # rather than merely un-enriched).
                    raise data
                results = data.get("results") or []
                if not results:
                    return
                for item in results:
                    yield item
            page += len(window)

    # ------------------------------------------------------------------ lists
    async def list_page(self, list_id: str, page: int) -> dict[str, Any]:
        """One page of a public TMDB list. 20 items per page, plus `item_count`."""
        return await self.get(f"/list/{list_id}", page=page)

    async def fetch_list(self, list_id: str, *, page_concurrency: int = 6,
                         max_pages: int = 500) -> list[dict[str, Any]]:
        """Every item of a public list, deduped by (media_type, id).

        Lists mix movies and series and each item carries its own `media_type`,
        so nothing is filtered here — the caller splits them.

        Page 1 is fetched alone to learn `item_count`; a 3,101-item list is 156
        pages, so the rest are fetched in concurrent windows exactly like
        `iter_discover`. Without that the request latency alone would dominate.
        """
        first = await self.list_page(list_id, 1)
        total = int(first.get("item_count") or 0)
        items = list(first.get("items") or [])
        if not total or not items:
            return []

        # item_count is authoritative; trust it over "keep going until empty",
        # which would cost an extra empty request per list. 65 items is 4 pages,
        # not 5.
        total_pages = min(max_pages, max(1, (total + LIST_PAGE_SIZE - 1) // LIST_PAGE_SIZE))
        page = 2
        while page <= total_pages and len(items) < total:
            window = list(range(page, min(page + page_concurrency, total_pages + 1)))
            batch = await asyncio.gather(
                *(self.list_page(list_id, p) for p in window),
                return_exceptions=True)
            for data in batch:
                if isinstance(data, BaseException):
                    # A page lost to an error would silently truncate the list;
                    # better to fail the run and retry than to call it done.
                    raise data
                for item in data.get("items") or []:
                    items.append(item)
            page += len(window)

        # The same title can appear twice on a hand-curated list.
        seen: set[tuple[str, int]] = set()
        unique: list[dict[str, Any]] = []
        for item in items:
            tmdb_id = int(item.get("id") or 0)
            media_type = str(item.get("media_type") or "")
            if not tmdb_id or not media_type:
                continue
            key = (media_type, tmdb_id)
            if key in seen:
                continue
            seen.add(key)
            unique.append(item)
        return unique[:total] if total else unique

    async def trending(self, media_type: str, window: str = "week") -> list[dict[str, Any]]:
        segment = "tv" if media_type == "series" else "movie"
        data = await self.get(f"/trending/{segment}/{window}")
        return data.get("results") or []


async def download_export(media_type: str, day: str, *, lookback_days: int = 3) -> list[dict]:
    """TMDB daily ID export (PLAN §6.3 step 1), with previous-day fallback.

    Every day is tried, not just 404s: the export CDN is redundant and a 403/5xx
    on one dated file does not mean the previous day's file is also gone. The
    date is not the only axis along which the fetch can fail.

    Raises TMDBError naming the LAST status and URL it saw, so a permanent
    failure (e.g. a revoked token returning 403 forever) is diagnosable from the
    job log instead of looking like "the export was not published yet".

    Args:
        day: `MM_DD_YYYY` of the preferred file.
        lookback_days: how many prior days to try before giving up.
    """
    prefix = "movie_ids" if media_type == "movie" else "tv_series_ids"
    from datetime import datetime, timedelta

    base = datetime.strptime(day, "%m_%d_%Y")
    last_status: int | None = None
    last_url = ""
    async with httpx.AsyncClient(timeout=120.0) as client:
        for back in range(lookback_days):
            stamp = (base - timedelta(days=back)).strftime("%m_%d_%Y")
            url = f"{EXPORTS}/{prefix}_{stamp}.json.gz"
            response = await client.get(url)
            last_status, last_url = response.status_code, url
            if response.status_code == 200:
                payload = gzip.decompress(io.BytesIO(response.content).read()).decode("utf-8")
                return [json.loads(line) for line in payload.splitlines() if line.strip()]
    raise TMDBError(
        f"no {prefix} export within {lookback_days} days of {day} "
        f"(last: HTTP {last_status} from {last_url})")
