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

import concurrent.futures
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
    def get_many(self, point_ids: Sequence[str]) -> dict[str, Record]:
        """Batch lookup — ONE round trip for the whole Trakt history.

        `build_user_state` needs every watched/rated/watchlist title. Doing that
        with per-id `get()` meant one thread + event loop + HTTP client per item,
        which pinned the instance at 100% CPU on the live deployment.
        """
        ids = [pid for pid in dict.fromkeys(point_ids) if pid]
        if not ids:
            return {}
        found = {pid: rec for pid in ids if (rec := self._cache.get(pid)) is not None}
        missing = [pid for pid in ids if pid not in found]
        if not missing:
            return found
        try:
            rows = _run(self.client.retrieve(missing, with_payload=True, with_vector=True))
        except Exception as exc:  # noqa: BLE001 - serves empty rather than 500
            log.warning("qdrant retrieve failed: %s", type(exc).__name__)
            return found
        for raw in rows:
            record = _record(raw)
            if record is not None:
                found[record.point_id] = record
                self._cache[record.point_id] = record
        return found

    def get(self, point_id: str) -> Record | None:
        """Point lookup used by Trakt joins (watched/rated titles)."""
        return self.get_many([point_id]).get(point_id)

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
        from .qdrant import SEARCH_PAYLOAD_FIELDS, payload_filter

        filters = payload_filter(media_type=media_type, langs=langs,
                                 min_votes=min_votes, exclude=exclude)
        try:
            # Vectors stay ON — rank.mmr() needs them (`vecs @ vecs.T`). Only the
            # PAYLOAD is projected to the fields serving actually reads: the full
            # record is ~5.3 KB/point and 200 candidates measured 1,037.7 KB
            # versus 148.6 KB projected.
            rows = _run(self.client.search(np.asarray(vector, dtype=np.float32).tolist(),
                                           collection=self.collection, limit=limit,
                                           filters=filters,
                                           with_payload=list(SEARCH_PAYLOAD_FIELDS)))
        except Exception as exc:  # noqa: BLE001 - never 500 a Stremio request
            log.warning("qdrant search failed: %s", type(exc).__name__)
            return []
        records = [r for r in (_record(raw) for raw in rows) if r is not None]
        for record in records:
            self._cache.setdefault(record.point_id, record)
        return records

    def search_many(self, vectors: Sequence[Sequence[float] | np.ndarray], *,
                    media_type: str, limit: int, langs: Sequence[str] = (),
                    min_votes: int | None = None,
                    exclude: Iterable[str] = ()) -> list[Record]:
        """ANN query with SEVERAL query vectors, deduped by point id.

        This is what multi-intent retrieval needs: one request covering every taste
        cluster instead of k sequential round trips. On a 0.1-vCPU instance the
        round trips dominated, and Qdrant's query planner shares the segment pass
        across searches that carry the same filter.

        Falls back to sequential `search()` when the client has no batched
        endpoint (the in-memory fakes), so behaviour is identical either way.
        """
        from .qdrant import SEARCH_PAYLOAD_FIELDS, payload_filter

        vectors = [v for v in vectors if v is not None]
        if not vectors:
            return []
        if len(vectors) == 1:
            return self.search(vectors[0], media_type=media_type, limit=limit,
                               langs=langs, min_votes=min_votes, exclude=exclude)

        filters = payload_filter(media_type=media_type, langs=langs,
                                 min_votes=min_votes, exclude=exclude)
        batched = getattr(self.client, "search_batch", None)
        try:
            if callable(batched) and not _requires_sequential(self.client):
                raw = _run(batched([np.asarray(v, dtype=np.float32).tolist()
                                    for v in vectors],
                                   collection=self.collection, limit=limit,
                                   filters=filters,
                                   with_payload=list(SEARCH_PAYLOAD_FIELDS)))
            else:
                raw = []
                for vec in vectors:
                    raw.extend(_run(self.client.search(
                        np.asarray(vec, dtype=np.float32).tolist(),
                        collection=self.collection, limit=limit, filters=filters,
                        with_payload=list(SEARCH_PAYLOAD_FIELDS))))
        except Exception as exc:  # noqa: BLE001 - never 500 a Stremio request
            log.warning("qdrant batch search failed: %s", type(exc).__name__)
            return []

        seen: set[str] = set()
        records: list[Record] = []
        for hit in raw:
            record = _record(hit)
            if record is None or record.point_id in seen:
                continue
            seen.add(record.point_id)
            records.append(record)
        for record in records:
            self._cache.setdefault(record.point_id, record)
        return records


def _requires_sequential(client: Any) -> bool:
    """True for fake clients whose search_batch is not a real Qdrant batcher.

    Explicitly marks test doubles apart from `QdrantClient`. Without this a fake
    defining `search_batch` with different semantics would silently change what
    every row test exercises.
    """
    return getattr(client, "IS_FAKE", False)


_EXECUTOR: concurrent.futures.ThreadPoolExecutor | None = None


def _run(coro: Any) -> Any:
    """Run a coroutine from sync code (Stremio handlers are sync by nature).

    Reuses ONE worker thread for the process. Creating a thread + event loop per
    call (the original approach) burned CPU under load; callers should also use
    `get_many` so a request makes one round trip, not one per item.
    """
    import asyncio

    global _EXECUTOR
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    if _EXECUTOR is None:
        _EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=2,
                                                           thread_name_prefix="qdrant")
    return _EXECUTOR.submit(asyncio.run, coro).result()
