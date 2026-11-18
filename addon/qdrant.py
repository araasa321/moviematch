"""Minimal Qdrant REST client (PLAN §4).

The addon only READS from Qdrant and writes `user_state` (taste cache, tokens,
last_good_rows). Ingestion writes the `titles` collection via `jobs/`.

Optional: if `QDRANT_URL` is unset the addon falls back to the in-memory fixture
catalog, which is what makes /health green on Koyeb before M1 has run.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Sequence

import httpx

log = logging.getLogger(__name__)

VECTOR_DIM = 384
TITLES = "titles"
USER_STATE = "user_state"


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
        return data.get("result") or []
