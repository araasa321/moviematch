"""Qdrant-backed store for serving rows (PLAN §9.1).

The fixture/live difference: with ~70k titles we cannot pull the whole catalog
into memory per request (that was fine for 12 fixture titles, not 80k rows of
payload). So this store implements the same `Store` protocol the ranker uses,
plus an optional `search()` capability that `rows.build_row` prefers:

    query vector -> Qdrant ANN top-N (server-side) -> Candidates -> rank()

`get_all()` still works but is only for small collections / debugging: it
scrolls the whole collection and refuses past `get_all_limit`.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np

from .qdrant import TITLES, QdrantClient, QdrantConfig

log = logging.getLogger(__name__)


@dataclass
class Record:
    """Same shape as `addon.memory_store.Record` so `to_candidate()` is identical."""
    point_id: str
    vector: np.ndarray
    payload: dict[str, Any]

    @property
    def media_type(self) -> str:
        return str(self.payload.get("media_type", "movie"))

    def to_candidate(self) -> Any:
        from .rank import Candidate
        return Candidate(self.point_id, str(self.payload.get("imdb_id", "")),
                         self.payload, self.vector)


def _record(raw: dict[str, Any]) -> Record | None:
    vector = raw.get("vector")
    if isinstance(vector, dict):              # named vectors
        vector = next(iter(vector.values()), None)
    if not vector:
        return None
    return Record(str(raw.get("id")), np.asarray(vector, dtype=np.float32),
                  raw.get("payload") or {})


@dataclass
class QdrantStore:
    """Read-only store over the `titles` collection."""

    client: QdrantClient
    collection: str = TITLES
    get_all_limit: int = 20_000          # refuse whole-collection scrolls above this
    _cache: dict[str, Record] = field(default_factory=dict, repr=False)
    _cache_ttl: float = 900.0
    _cache_at: float = field(default=0.0, init=False, repr=False)

    @classmethod
    def from_env(cls, env: dict[str, str]) -> "QdrantStore | None":
        config = QdrantConfig.from_env(env)
        return cls(QdrantClient(config)) if config.configured else None

    # ------------------------------------------------------------ protocol
    def get(self, point_id: str) -> Record | None:
        """Point lookup used by Trakt joins (watched/rated titles)."""
        import asyncio

        if point_id in self._cache:
            return self._cache[point_id]
        try:
            rows = _run(self.client.retrieve([point_id], with_payload=True, with_vector=True))
        except Exception as exc:  # noqa: BLE001
            log.warning("qdrant retrieve failed: %s", type(exc).__name__)
            return None
        record = _record(rows[0]) if rows else None
        if record is not None:
            self._cache[point_id] = record
        return record

    def get_all(self, media_types: str | Iterable[str] | None = None) -> Sequence[Record]:
        """Whole-collection scan. Small collections only; cached for `_cache_ttl`."""
        now = time.monotonic()
        if self._cache and (now - self._cache_at) < self._cache_ttl:
            wanted = {media_types} if isinstance(media_types, str) else set(media_types or [])
            rows = list(self._cache.values())
            return rows if not wanted else [r for r in rows if r.media_type in wanted]

        total = _run(self.client.count())
        if total > self.get_all_limit:
            raise RuntimeError(
                f"catalog has {total} points; refusing a full scan "
                f"(limit {self.get_all_limit}). Use search() instead."
            )
        records: list[Record] = []
        for raw in _run(self.client.scroll_all(limit=1000)):
            record = _record(raw)
            if record is not None:
                records.append(record)
        self._cache = {r.point_id: r for r in records}
        self._cache_at = now
        wanted = {media_types} if isinstance(media_types, str) else set(media_types or [])
        return [r for r in records if not wanted or r.media_type in wanted]

    # ------------------------------------------------------------- search
    def search(self, vector: Sequence[float] | np.ndarray, *, media_type: str,
               limit: int, langs: Sequence[str] = (), min_votes: int | None = None,
               exclude: Iterable[str] = ()) -> list[Record]:
        """ANN query with the §9.1 guards pushed down into Qdrant."""
        from .qdrant import payload_filter

        filters = payload_filter(media_type=media_type, langs=langs,
                                 min_votes=min_votes, exclude=exclude)
        try:
            rows = _run(self.client.search(np.asarray(vector, dtype=np.float32).tolist(),
                                           collection=self.collection, limit=limit,
                                           filters=filters, with_payload=True))
        except Exception as exc:  # noqa: BLE001 - never 500 a Stremio request
            log.warning("qdrant search failed: %s", type(exc).__name__)
            return []
        records = [r for r in (_record(raw) for raw in rows) if r is not None]
        for record in records:
            self._cache.setdefault(record.point_id, record)
        return records


def _run(coro: Any) -> Any:
    """Run a coroutine from sync code (Stremio handlers are sync by nature)."""
    import asyncio

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    # Already inside a loop (FastAPI async handler): run in a worker thread's own loop.
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()
