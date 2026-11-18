"""A request must not make one network call per Trakt item.

Live symptom: 100% CPU and multi-second responses on a 0.1-vCPU Koyeb instance,
because `build_user_state` resolved each watched/rated/watchlist title with its
own `retrieve` round trip (plus a thread + event loop per call).
"""

from __future__ import annotations

import numpy as np

from addon.ids import point_id
from addon.rows import build_row
from addon.store import QdrantStore, Record


class CountingClient:
    def __init__(self, tmdb_ids: list[int]) -> None:
        self.records = [
            Record(point_id("movie", tmdb), np.eye(8, dtype=np.float32)[i % 8],
                   {"tmdb_id": tmdb, "media_type": "movie", "imdb_id": f"tt{tmdb}",
                    "title": f"Title {tmdb}", "year": 2010, "language": "en",
                    "poster": "/p.jpg", "vote_average": 8.0, "vote_count": 5000,
                    "popularity": 50.0, "director": "D", "cast": []})
            for i, tmdb in enumerate(tmdb_ids)
        ]
        self.retrieve_calls = 0
        self.search_calls = 0
        self.scroll_calls = 0

    async def count(self) -> int:
        return len(self.records)

    async def retrieve(self, ids, **_):
        self.retrieve_calls += 1
        wanted = set(ids)
        return [{"id": r.point_id, "vector": r.vector.tolist(), "payload": r.payload}
                for r in self.records if r.point_id in wanted]

    async def scroll_all(self, **_):
        self.scroll_calls += 1
        return []

    async def search(self, vector, *, limit=200, filters=None, **_):
        self.search_calls += 1
        return [{"id": r.point_id, "vector": r.vector.tolist(), "payload": r.payload}
                for r in self.records[:limit]]


def _history(n: int) -> dict:
    return {
        "watched": [{"movie": {"ids": {"tmdb": 1000 + i}}, "plays": 1,
                     "watched_at": "2026-11-18T10:00:00+00:00"} for i in range(n)],
        "ratings": [{"movie": {"ids": {"tmdb": 2000 + i}}, "rating": 9,
                     "rated_at": "2026-11-18T10:00:00+00:00"} for i in range(n)],
        "watchlist": [{"movie": {"ids": {"tmdb": 3000 + i}},
                       "listed_at": "2026-11-18T10:00:00+00:00"} for i in range(n)],
    }


def test_one_retrieve_for_whole_history():
    ids = [1000 + i for i in range(50)] + [2000 + i for i in range(50)] + \
          [3000 + i for i in range(50)]
    client = CountingClient(ids)
    store = QdrantStore(client=client)
    plan = build_row(store, _history(50), "movie", "foryou")

    assert client.retrieve_calls == 1, (
        f"expected 1 batched retrieve for 150 items, got {client.retrieve_calls}")
    assert client.scroll_calls == 0, "must never scan the catalog"
    assert plan.diagnostics["pool_source"] == "ann"


def test_repeated_rows_still_cheap():
    client = CountingClient([1000, 2000, 3000])
    store = QdrantStore(client=client)
    for _ in range(3):
        build_row(store, _history(1), "movie", "because")
    # 1 retrieve for the batch, plus 1 to resolve the because-seed
    assert client.retrieve_calls <= 2 * 3


def test_get_many_dedupes_and_caches():
    client = CountingClient([1000, 1001])
    store = QdrantStore(client=client)
    first = store.get_many([point_id("movie", 1000), point_id("movie", 1000)])
    assert len(first) == 1 and client.retrieve_calls == 1
    store.get_many([point_id("movie", 1000)])       # cached, no new call
    assert client.retrieve_calls == 1


def test_get_many_survives_backend_error():
    class Broken(CountingClient):
        async def retrieve(self, ids, **_):
            raise RuntimeError("qdrant down")

    store = QdrantStore(client=Broken([1000]))
    assert store.get_many([point_id("movie", 1000)]) == {}
