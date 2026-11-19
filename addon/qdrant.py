"""Minimal Qdrant REST client (PLAN §4).

The addon only READS from Qdrant and writes `user_state` (taste cache, tokens,
last_good_rows). Ingestion writes the `titles` collection via `jobs/`.

Optional: if `QDRANT_URL` is unset the addon falls back to the in-memory fixture
catalog, which is what makes /health green on Koyeb before M1 has run.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import httpx

log = logging.getLogger(__name__)

VECTOR_DIM = 384
TITLES = "titles"
USER_STATE = "user_state"


def payload_filter(*, media_type: str | None = None, langs: Sequence[str] | None = None,
                   min_votes: int | None = None, exclude: Iterable[str] | None = None,
                   has_imdb: bool = True) -> dict[str, Any]:
    """Qdrant filter for the §9.1 guards. Empty clauses are omitted entirely.

    `imdb_id` is REQUIRED for Stremio (§4.1); Qdrant treats null/missing as empty.
    """
    must: list[dict[str, Any]] = []
    must_not: list[dict[str, Any]] = []

    if media_type:
        must.append({"key": "media_type", "match": {"value": media_type}})
    if langs:
        must.append({"key": "language", "match": {"any": list(langs)}})
    if min_votes is not None:
        must.append({"key": "vote_count", "range": {"gte": min_votes}})
    if has_imdb:
        must_not.append({"is_empty": {"key": "imdb_id"}})
    if exclude:
        ids = list(exclude)
        if ids:
            must_not.append({"has_id": ids})

    clause: dict[str, Any] = {}
    if must:
        clause["must"] = must
    if must_not:
        clause["must_not"] = must_not
    return clause


@dataclass(frozen=True)
class QdrantConfig:
    url: str = ""
    api_key: str = ""
    timeout: float = 10.0

    @property
    def configured(self) -> bool:
        return bool(self.url)

    @classmethod
    def from_env(cls, env: dict[str, str]) -> "QdrantConfig":
        return cls(url=env.get("QDRANT_URL", "").strip(),
                   api_key=env.get("QDRANT_API_KEY", "").strip())


class QdrantError(RuntimeError):
    pass


class QdrantClient:
    """Thin async wrapper; used by /health and (M2+) by the row builder."""

    def __init__(self, config: QdrantConfig) -> None:
        self.config = config
        self._headers = {"Content-Type": "application/json"}
        if config.api_key:
            self._headers["api-key"] = config.api_key

    async def _request(self, method: str, path: str, payload: Any | None = None,
                       params: dict[str, Any] | None = None) -> dict[str, Any]:
        if not self.config.configured:
            raise QdrantError("QDRANT_URL is not set")
        async with httpx.AsyncClient(timeout=self.config.timeout) as client:
            response = await client.request(method, f"{self.config.url.rstrip('/')}{path}",
                                            json=payload, params=params, headers=self._headers)
        if response.status_code >= 300:
            raise QdrantError(f"{method} {path} -> {response.status_code}: "
                              f"{response.text[:200]}")
        return response.json() if response.content else {}

    async def count(self, collection: str = TITLES) -> int:
        data = await self._request("POST", f"/collections/{collection}/points/count",
                                   {"exact": True})
        return int(data.get("result", {}).get("count", 0))

    async def facet(self, key: str, *, collection: str = TITLES, limit: int = 100,
                    exact: bool = False) -> list[tuple[str, int]]:
        """Distinct values of one payload key with counts, descending by count.

        Qdrant's facet API (v1.12+) counts unique values of an INDEXED keyword
        field in one request, so this is how the addon learns which languages
        actually exist in the catalog instead of trusting a hardcoded list.
        `exact=False` uses the HNSW-style approximation; counts are used for
        ordering and rough magnitude, not for correctness decisions.
        """
        data = await self._request("POST", f"/collections/{collection}/facet",
                                   {"key": key, "limit": limit, "exact": exact})
        hits = ((data.get("result") or {}).get("hits") or [])
        out: list[tuple[str, int]] = []
        for hit in hits:
            value = hit.get("value")
            if isinstance(value, str) and value:
                out.append((value, int(hit.get("count") or 0)))
        return out

    async def get_payload(self, point_id: int | str,
                          collection: str = USER_STATE) -> dict[str, Any] | None:
        data = await self._request("POST", f"/collections/{collection}/points",
                                   {"ids": [point_id], "with_payload": True,
                                    "with_vector": False})
        points = data.get("result") or []
        return (points[0].get("payload") or {}) if points else None

    async def save_payload(self, payload: dict[str, Any], point_id: int | str = 1,
                           collection: str = USER_STATE) -> None:
        """Upsert a single user_state point; the vector is a zero placeholder."""
        await self._request("PUT", f"/collections/{collection}/points",
                            {"points": [{"id": point_id,
                                         "vector": [0.0] * VECTOR_DIM,
                                         "payload": payload}]},
                            params={"wait": "true"})

    async def search(self, vector: Sequence[float], *, collection: str = TITLES,
                     limit: int = 200, filters: dict[str, Any] | None = None,
                     with_payload: bool = True) -> list[dict[str, Any]]:
        body: dict[str, Any] = {"vector": list(vector), "limit": limit,
                                "with_payload": with_payload, "with_vector": True}
        if filters:
            body["filter"] = filters
        data = await self._request("POST", f"/collections/{collection}/points/search", body)
        return data.get("result") or []

    async def retrieve(self, ids: Sequence[str], *, collection: str = TITLES,
                       with_payload: bool = True,
                       with_vector: bool = True) -> list[dict[str, Any]]:
        data = await self._request("POST", f"/collections/{collection}/points",
                                   {"ids": list(ids), "with_payload": with_payload,
                                    "with_vector": with_vector})
        result = data.get("result")
        return result if isinstance(result, list) else []

    async def scroll_all(self, *, collection: str = TITLES, limit: int = 1000,
                         with_payload: bool = True, pages: int = 100) -> list[dict[str, Any]]:
        """Page through the collection; capped by `pages` so a request can't run away."""
        out: list[dict[str, Any]] = []
        offset = None
        for _ in range(pages):
            body: dict[str, Any] = {"limit": limit, "with_payload": with_payload,
                                    "with_vector": True}
            if offset is not None:
                body["offset"] = offset
            data = await self._request("POST", f"/collections/{collection}/points/scroll",
                                       body, params={"wait": "true"})
            result = data.get("result")
            if not isinstance(result, dict):
                break
            chunk = result.get("points") or []
            out.extend(chunk)
            offset = result.get("next_page_offset")
            if offset is None or not chunk:
                break
        return out
