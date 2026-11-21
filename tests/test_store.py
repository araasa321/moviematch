"""Qdrant-backed serving: ANN pool, guards, full-scan refusal, row cache."""

from __future__ import annotations

import asyncio

import numpy as np
import pytest

from addon.ids import point_id
from addon.rows import build_row
from addon.store import QdrantStore, Record

from .conftest import project_payload


def _vec(*values: float, dim: int = 8) -> np.ndarray:
    out = np.zeros(dim, dtype=np.float32)
    out[: len(values)] = values
    return out


def _payload(tmdb: int, title: str, media_type: str = "movie", **kw) -> dict:
    base = {"tmdb_id": tmdb, "media_type": media_type, "imdb_id": f"tt{tmdb}",
            "title": title, "year": 2010, "language": "en", "poster": "/p.jpg",
            "vote_average": 8.0, "vote_count": 5000, "popularity": 50.0,
            "director": "Christopher Nolan", "cast": ["Christian Bale"]}
    base.update(kw)
    return base


class FakeReadClient:
    """Mimics the read-only Qdrant client used by serving."""

    def __init__(self, records: list[Record], total: int | None = None) -> None:
        self.records = records
        self.total = total if total is not None else len(records)
        self.searched = 0
        self.scrolled = False

    async def count(self) -> int:
        return self.total

    async def retrieve(self, ids, **kwargs):
        wanted = set(ids)
        return [{"id": r.point_id, "vector": r.vector.tolist(), "payload": r.payload}
                for r in self.records if r.point_id in wanted]

    async def scroll_all(self, **kwargs):
        self.scrolled = True
        return [{"id": r.point_id, "vector": r.vector.tolist(), "payload": r.payload}
                for r in self.records]

    async def search(self, vector, *, limit=200, filters=None, with_payload=True, **kwargs):
        self.searched += 1
        q = np.asarray(vector, dtype=np.float32)
        q = q / (np.linalg.norm(q) or 1.0)
        scored = []
        for r in self.records:
            v = r.vector / (np.linalg.norm(r.vector) or 1.0)
            media_ok = filters is None or all(
                c.get("key") != "media_type" or r.payload.get("media_type") == c["match"]["value"]
                for c in (filters.get("must") or []))
            if not media_ok:
                continue
            scored.append((float(v @ q), r))
        scored.sort(key=lambda pair: -pair[0])
        return [{"id": r.point_id, "vector": r.vector.tolist(),
                 "payload": project_payload(r.payload, with_payload)}
                for _, r in scored[:limit]]


def _store(client: FakeReadClient, **kw) -> QdrantStore:
    return QdrantStore(client=client, **kw)


def test_search_returns_records_with_vectors():
    """8-dim stand-ins are enough to check plumbing (the math is dim-agnostic).

    ANN returns the nearest `limit` items regardless of absolute similarity —
    the relevance floor lives in rank(), not in the store.
    """
    a = Record(point_id("movie", 1), _vec(1, 0), _payload(1, "The Dark Knight"))
    b = Record(point_id("movie", 2), _vec(0, 1), _payload(2, "Spaced Out"))
    store = _store(FakeReadClient([a, b]))
    got = store.search(_vec(1, 0), media_type="movie", limit=10)
    assert [r.payload["title"] for r in got] == ["The Dark Knight", "Spaced Out"]
    assert got[0].vector.shape == (8,)
    # type guard is pushed into the filter
    assert store.search(_vec(1, 0), media_type="series", limit=10) == []


def test_get_all_refuses_large_catalog():
    """A 70k catalog must never be scanned into memory per request."""
    store = _store(FakeReadClient([], total=70_000), get_all_limit=20_000)
    with pytest.raises(RuntimeError, match="refusing a full scan"):
        store.get_all()


def test_get_all_works_on_small_catalog_and_caches():
    r = Record(point_id("movie", 1), _vec(1, 0), _payload(1, "X"))
    client = FakeReadClient([r])
    store = _store(client)
    assert len(store.get_all()) == 1
    first = client.scrolled
    store.get_all()
    assert client.scrolled is first, "second call should serve from cache"


def test_get_uses_retrieve_for_trakt_joins():
    r = Record(point_id("movie", 155), _vec(1, 0), _payload(155, "The Dark Knight"))
    store = _store(FakeReadClient([r]))
    got = store.get(point_id("movie", 155))
    assert got is not None and got.payload["title"] == "The Dark Knight"
    assert store.get(point_id("movie", 999)) is None


def test_build_row_uses_search_not_scan():
    seed = Record(point_id("movie", 155), _vec(1, 0), _payload(155, "The Dark Knight"))
    other = Record(point_id("movie", 9), _vec(0.9, 0.1), _payload(9, "Joker"))
    client = FakeReadClient([seed, other], total=70_000)
    store = _store(client)
    history = {"watched": [{"movie": {"ids": {"tmdb": 155}}, "plays": 1,
                            "watched_at": "2026-11-18T10:00:00+00:00"}]}
    plan = build_row(store, history, "movie", "because")
    assert plan.diagnostics["pool_source"] == "ann"
    assert client.searched >= 1
    assert client.scrolled is False, "must not scan the whole catalog"
    assert [c.payload["title"] for c in plan.ranked] == ["Joker"]


def test_build_row_marks_scan_source_for_memory_store():
    from addon.fixtures import catalog as fixture_catalog
    from addon.memory_store import MemoryStore, load_records_from_json

    store = MemoryStore()
    load_records_from_json(store, fixture_catalog(dim=384))
    history = {"watched": [{"movie": {"ids": {"tmdb": 155}}, "plays": 1,
                            "watched_at": "2026-11-18T10:00:00+00:00"}]}
    plan = build_row(store, history, "movie", "because")
    assert plan.diagnostics["pool_source"] == "scan"


def test_store_falls_back_when_search_raises():
    class Broken(FakeReadClient):
        async def search(self, *a, **k):
            raise RuntimeError("qdrant down")

    r = Record(point_id("movie", 1), _vec(1, 0), _payload(1, "X"))
    store = _store(Broken([r]))
    assert store.search(_vec(1, 0), media_type="movie", limit=5) == []
