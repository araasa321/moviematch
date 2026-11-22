"""Qdrant **write/ingest** client for the jobs (PLAN §4).

Distinct from `addon/qdrant.py`, which is a small read-only client for serving.
Kept as its own module so the runtime image never gains write helpers it does
not use, and so a signature change in one cannot silently break the other
(that exact bug broke the first bootstrap run).
"""

from __future__ import annotations

import asyncio
from typing import Any, Iterable, Sequence

import httpx

VECTOR_DIM = 384
TITLES = "titles"
USER_STATE = "user_state"

INDEXES: tuple[tuple[str, str], ...] = (
    ("media_type", "keyword"),
    ("imdb_id", "keyword"),
    ("year", "integer"),
    ("language", "keyword"),
    ("vote_count", "integer"),
    ("popularity", "float"),
    ("collection_id", "keyword"),
    ("genres", "keyword"),
    ("release_date", "keyword"),
    ("enriched", "bool"),
)


class QdrantError(RuntimeError):
    pass


def titles_collection_spec() -> dict[str, Any]:
    """PLAN §4.1: 384-d cosine, HNSW m=16 ef_construct=100, int8 on disk."""
    return {
        "vectors": {
            "size": VECTOR_DIM,
            "distance": "Cosine",
            "on_disk": True,
            "hnsw_config": {"m": 16, "ef_construct": 100},
        },
        "quantization_config": {"scalar": {"type": "int8", "always_ram": True}},
        "optimizers_config": {"default_segment_number": 2},
        "on_disk_payload": True,
    }


class IngestQdrant:
    """Minimal REST client: create collection, upsert, scroll, set_payload."""

    def __init__(self, url: str, api_key: str = "", *, timeout: float = 60.0,
                 concurrency: int = 4) -> None:
        if not url:
            raise QdrantError("QDRANT_URL is empty")
        self.url = url.rstrip("/")
        self._headers = {"Content-Type": "application/json"}
        if api_key:
            self._headers["api-key"] = api_key
        self._timeout = timeout
        self._sem = asyncio.Semaphore(concurrency)

    async def _request(self, method: str, path: str, payload: Any | None = None,
                       params: dict[str, Any] | None = None) -> dict[str, Any]:
        async with self._sem:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.request(method, f"{self.url}{path}", json=payload,
                                                params=params, headers=self._headers)
        if response.status_code >= 300:
            hint = ""
            if response.status_code == 403:
                hint = " (check QDRANT_API_KEY)"
            elif response.status_code == 404:
                hint = " (is QDRANT_URL the cluster URL, e.g. https://xxx.cloud.qdrant.io?)"
            raise QdrantError(f"{method} {path} -> {response.status_code}: "
                              f"{response.text[:200]}{hint}")
        return response.json() if response.content else {}

    # ---------------------------------------------------------- collections
    async def collection_exists(self, name: str = TITLES) -> bool:
        try:
            await self._request("GET", f"/collections/{name}")
            return True
        except QdrantError as exc:
            if "-> 404" in str(exc):
                return False
            raise

    async def create_titles_collection(self, *, recreate: bool = False,
                                       name: str = TITLES) -> bool:
        exists = await self.collection_exists(name)
        if exists and not recreate:
            return False
        if exists:
            await self._request("DELETE", f"/collections/{name}")
        await self._request("PUT", f"/collections/{name}", titles_collection_spec())
        for field_name, schema in INDEXES:
            await self._request("PUT", f"/collections/{name}/index",
                                {"field_name": field_name, "field_schema": schema},
                                params={"wait": "true"})
        return True

    async def ensure_user_state(self, name: str = USER_STATE) -> None:
        if await self.collection_exists(name):
            return
        await self._request("PUT", f"/collections/{name}",
                            {"vectors": {"size": VECTOR_DIM, "distance": "Cosine"}})

    async def count(self, name: str = TITLES) -> int:
        data = await self._request("POST", f"/collections/{name}/points/count",
                                   {"exact": True})
        result = data.get("result")
        return int(result.get("count", 0)) if isinstance(result, dict) else 0

    # --------------------------------------------------------------- points
    async def upsert(self, points: Sequence[dict[str, Any]], *, wait: bool = False,
                     name: str = TITLES) -> None:
        """Create-or-REPLACE points.

        DANGER: this is SET semantics. `PUT /points` fully replaces the stored
        payload with whatever is in the request. Upserting a point whose payload
        holds only a few fields ERASES every other field on that point — for a
        maintenance script over the whole catalog, that is a full outage. To
        change only the vector, use `update_vectors`; to change a few payload
        fields and keep the rest, use `set_payload` / `set_payload_batch`
        (both merge keys).
        """
        if not points:
            return
        await self._request("PUT", f"/collections/{name}/points",
                            {"points": list(points)},
                            params={"wait": "true" if wait else "false"})

    async def update_vectors(self, points: Sequence[tuple[str, Sequence[float]]], *,
                             wait: bool = False, name: str = TITLES) -> None:
        """Replace ONLY the vectors of existing points; the payload is untouched.

        Uses PUT /collections/{c}/points/vectors (Qdrant >= 1.2.0), which updates
        the specified vectors in place and leaves everything else, including the
        payload, alone. Unlike `upsert` it does not transmit — and therefore
        cannot overwrite — a payload. This is what the #3 re-embed driver uses.

        The method is PUT, not POST: POST on this path returns
        `405 Method Not Allowed` (observed against Qdrant Cloud).
        """
        if not points:
            return
        await self._request("PUT", f"/collections/{name}/points/vectors",
                            {"points": [{"id": pid, "vector": list(vec)}
                                        for pid, vec in points]},
                            params={"wait": "true" if wait else "false"})

    async def get_point(self, point_id: str, *, name: str = TITLES,
                        with_payload: bool = True,
                        with_vector: bool = False) -> dict[str, Any] | None:
        """Fetch ONE point, used by the re-embed preflight/verification guard."""
        data = await self._request("POST", f"/collections/{name}/points",
                                   {"ids": [point_id], "with_payload": with_payload,
                                    "with_vector": with_vector})
        points = data.get("result") or []
        return points[0] if points else None

    async def scroll(self, *, name: str = TITLES, limit: int = 1000, offset: Any = None,
                     with_payload: bool | list[str] = True,
                     ) -> tuple[list[dict[str, Any]], Any]:
        body: dict[str, Any] = {"limit": limit, "with_payload": with_payload,
                                "with_vector": False}
        if offset is not None:
            body["offset"] = offset
        data = await self._request("POST", f"/collections/{name}/points/scroll", body,
                                   params={"wait": "true"})
        result = data.get("result")
        if not isinstance(result, dict):        # defensive: never assume a shape
            return [], None
        return result.get("points") or [], result.get("next_page_offset")

    async def iter_all_payloads(self, *, name: str = TITLES, batch: int = 1000,
                                fields: list[str] | None = None) -> Any:
        offset = None
        while True:
            points, offset = await self.scroll(name=name, limit=batch, offset=offset,
                                               with_payload=fields or True)
            for point in points:
                yield point
            if offset is None:
                return

    async def set_payload(self, payload: dict[str, Any], ids: Iterable[str], *,
                          name: str = TITLES) -> None:
        ids_list = list(ids)
        if not ids_list:
            return
        await self._request("POST", f"/collections/{name}/points/payload",
                            {"payload": payload, "points": ids_list},
                            params={"wait": "false"})

    async def set_payload_batch(self, updates: Iterable[tuple[dict[str, Any], list[str]]],
                                *, name: str = TITLES, chunk: int = 500) -> int:
        """Apply DIFFERENT payloads to DIFFERENT points, chunked.

        `set_payload` sends one payload to many ids, which only helps when the
        values are identical. Score refreshes are per-title (each bayes value is
        unique), so the previous one-request-per-point loop dominated the whole
        nightly run: ~43k round trips at ~370 ms each.

        Qdrant's batch endpoint takes a list of operations, so many distinct
        (payload, ids) pairs go in one HTTP request. Returns the number of points
        touched.
        """
        pending: list[dict[str, Any]] = []
        touched = 0
        for payload, ids in updates:
            ids_list = list(ids)
            if not ids_list:
                continue
            pending.append({"set_payload": {"payload": payload, "points": ids_list}})
            touched += len(ids_list)
            if len(pending) >= chunk:
                await self._request("POST", f"/collections/{name}/points/batch",
                                    {"operations": pending}, params={"wait": "false"})
                pending = []
        if pending:
            await self._request("POST", f"/collections/{name}/points/batch",
                                {"operations": pending}, params={"wait": "false"})
        return touched


# Backwards-compatible alias: existing job code constructs `QdrantClient(url, key)`.
QdrantClient = IngestQdrant
