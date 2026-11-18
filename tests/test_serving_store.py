"""The serving app must use Qdrant when it is configured.

The original bug: `/health` reported the Qdrant size while rows came from the
12-title fixture, because `store = _fixture_store(dim)` ran unconditionally.
"""

from __future__ import annotations

import numpy as np

from addon.ids import point_id
from addon.main import make_app
from addon.store import QdrantStore, Record


class _Client:
    """Just enough Qdrant for make_app + QdrantStore."""

    def __init__(self, records: list[Record]) -> None:
        self.records = records

    async def count(self) -> int:
        return len(self.records)

    async def retrieve(self, ids, **_):
        wanted = set(ids)
        return [{"id": r.point_id, "vector": r.vector.tolist(), "payload": r.payload}
                for r in self.records if r.point_id in wanted]

    async def scroll_all(self, **_):
        return [{"id": r.point_id, "vector": r.vector.tolist(), "payload": r.payload}
                for r in self.records]

    async def search(self, vector, *, limit=200, filters=None, **_):
        q = np.asarray(vector, dtype=np.float32)
        q = q / (np.linalg.norm(q) or 1.0)
        scored = [(float((r.vector / np.linalg.norm(r.vector)) @ q), r) for r in self.records]
        scored.sort(key=lambda pair: -pair[0])
        return [{"id": r.point_id, "vector": r.vector.tolist(), "payload": r.payload}
                for _, r in scored[:limit]]


def _rec(tmdb: int, title: str, vec: list[float]) -> Record:
    return Record(point_id("movie", tmdb), np.asarray(vec, dtype=np.float32),
                  {"tmdb_id": tmdb, "media_type": "movie", "imdb_id": f"tt{tmdb}",
                   "title": title, "year": 2010, "language": "en", "poster": "/p.jpg",
                   "vote_average": 8.0, "vote_count": 5000, "popularity": 50.0,
                   "director": "Christopher Nolan", "cast": ["Christian Bale"]})


def test_app_uses_qdrant_store_when_configured(monkeypatch):
    monkeypatch.setenv("QDRANT_URL", "https://example.cloud.qdrant.io")
    monkeypatch.setenv("QDRANT_API_KEY", "k")
    records = [_rec(155, "The Dark Knight", [1, 0, 0]), _rec(9, "Joker", [0.9, 0.1, 0])]
    monkeypatch.setattr(
        "addon.main.QdrantStore.from_env",
        classmethod(lambda cls, env: QdrantStore(client=_Client(records))))
    app = make_app()
    assert isinstance(app.state.store, QdrantStore)


def test_app_falls_back_to_fixture_without_qdrant(monkeypatch):
    monkeypatch.delenv("QDRANT_URL", raising=False)
    app = make_app()
    assert not isinstance(app.state.store, QdrantStore)
    assert len(app.state.store) == 12


def test_injected_store_wins_over_env(monkeypatch):
    from addon.memory_store import MemoryStore

    monkeypatch.setenv("QDRANT_URL", "https://example.cloud.qdrant.io")
    mine = MemoryStore()
    assert make_app(store=mine).state.store is mine


def test_row_cache_starts_empty(monkeypatch):
    monkeypatch.delenv("QDRANT_URL", raising=False)
    app = make_app()
    assert app.state.row_cache == {}
