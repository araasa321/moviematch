"""In-memory vector store — the Qdrant stand-in for tests and local smoke runs.

`store.py` (real Qdrant client) only has to satisfy the same small surface:
    get_all(kind) -> {point_id: Record}
    get(point_id) -> Record | None
so rows can be exercised end-to-end without a cloud cluster or an API key.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from .rank import Candidate, normalize


@dataclass
class Record:
    point_id: str
    vector: np.ndarray
    payload: dict[str, Any]

    @property
    def media_type(self) -> str:
        return str(self.payload.get("media_type", "movie"))

    def to_candidate(self) -> Candidate:
        return Candidate(self.point_id, str(self.payload.get("imdb_id", "")), self.payload, self.vector)


class MemoryStore:
    def __init__(self) -> None:
        self._records: dict[str, Record] = {}

    def __len__(self) -> int:
        return len(self._records)

    def upsert(self, point_id: str, vector: Iterable[float], payload: dict[str, Any]) -> None:
        self._records[point_id] = Record(point_id, normalize(np.asarray(list(vector), dtype=np.float32)), payload)

    def get(self, point_id: str) -> Record | None:
        return self._records.get(point_id)

    def get_all(self, media_types: str | Iterable[str] | None = None) -> list[Record]:
        if media_types is None:
            return list(self._records.values())
        wanted = {media_types} if isinstance(media_types, str) else set(media_types)
        return [r for r in self._records.values() if r.media_type in wanted]

    def ids(self) -> set[str]:
        return set(self._records)


def load_records_from_json(store: MemoryStore, items: Iterable[dict[str, Any]]) -> None:
    """Fixture loader: [{"point_id":..., "vector": [...], "payload": {...}}, ...]."""
    for item in items:
        store.upsert(item["point_id"], item["vector"], item["payload"])
