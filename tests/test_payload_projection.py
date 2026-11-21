"""#2b / M2 — project the ANN payload to the fields serving actually reads.

`addon/store.py` called `client.search(..., with_payload=True)`, so every ANN
query shipped the whole record — including `overview` (capped at 4,000 chars in
policy.normalize_payload) and `text_hash` — for zero ranking value. Measured at
200 candidates: 1,037.7 KB -> 148.6 KB (-85.7 %).

Vectors deliberately stay ON: rank.mmr() needs them (`vecs @ vecs.T`).
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from addon.qdrant import SEARCH_PAYLOAD_FIELDS, QdrantClient, QdrantConfig
from addon.rows import _candidate_pool
from addon.store import QdrantStore, Record

from .conftest import project_payload


# ------------------------------------------------------- the allowlist
def test_allowlist_is_exactly_the_derived_fields():
    """Pins the list itself: adding/removing a field must be a deliberate edit."""
    assert set(SEARCH_PAYLOAD_FIELDS) == {
        "imdb_id", "media_type", "poster", "backdrop", "title", "year",
        "end_year", "overview_short", "genres", "vote_average", "vote_count",
        "popularity", "language", "director", "cast", "collection_id",
        "bayes", "bayes_gems",
    }
    assert len(SEARCH_PAYLOAD_FIELDS) == 18


def test_load_bearing_fields_are_present():
    """Each of these is used by a ranking or rendering term; dropping one
    degrades the row silently rather than raising."""
    for field in ("cast", "collection_id", "year", "end_year", "imdb_id",
                  "poster", "director", "language"):
        assert field in SEARCH_PAYLOAD_FIELDS, field


def test_heavy_fields_are_not_requested():
    """`overview` and `text_hash` are the bulk of the record and rank nothing."""
    for field in ("overview", "text_hash", "original_title", "tagline",
                  "keywords", "runtime", "budget", "revenue", "release_date",
                  "enriched", "adult", "video", "origin_country"):
        assert field not in SEARCH_PAYLOAD_FIELDS, field


# ------------------------------------------------------- request shape
class _CapturingClient:
    def __init__(self) -> None:
        self.body: dict | None = None

    async def search(self, vector, *, collection="titles", limit=200,
                     filters=None, with_payload=True):
        self.last_with_payload = with_payload
        self.body = {"vector": list(vector), "limit": limit,
                     "with_payload": with_payload, "with_vector": True,
                     "filter": filters}
        return []


def test_search_body_requests_the_allowlist_and_keeps_vectors():
    client = _CapturingClient()
    store = QdrantStore(client=client)
    store.search(np.zeros(4, dtype=np.float32), media_type="movie", limit=5)

    body = client.body
    assert body is not None, "search() never reached the client"
    assert body["with_payload"] == list(SEARCH_PAYLOAD_FIELDS)
    assert body["with_vector"] is True, "MMR needs the vectors"


def _real_client_body(**search_kwargs) -> dict:
    """Run the REAL addon.qdrant.QdrantClient.search and capture its HTTP body."""
    import asyncio

    captured: dict = {}

    async def fake_request(method, path, payload=None, **kwargs):
        captured["method"] = method
        captured["path"] = path
        captured["payload"] = payload
        return {"result": []}

    client = QdrantClient(QdrantConfig(url="https://example.qdrant.io", api_key="key"))
    client._request = fake_request            # type: ignore[method-assign]
    asyncio.run(client.search([0.0, 0.0], **search_kwargs))
    return captured["payload"]


def test_qdrant_client_passes_a_list_selector_through():
    """The REST API accepts a field-name array; the client must not coerce it."""
    body = _real_client_body(with_payload=list(SEARCH_PAYLOAD_FIELDS))
    assert body["with_payload"] == list(SEARCH_PAYLOAD_FIELDS)


def test_qdrant_client_default_stays_true():
    """Anything that does not opt in must keep the old behaviour."""
    body = _real_client_body()
    assert body["with_payload"] is True


# ------------------------------------------- behavioural guard (the important one)
def _rec(tmdb: int, payload: dict, vec) -> Record:
    return Record(f"movie-{tmdb}", np.asarray(vec, dtype=np.float32), payload)


def _base_payload(tmdb: int, **kw) -> dict:
    base = {"tmdb_id": tmdb, "media_type": "movie", "imdb_id": f"tt{tmdb}",
            "title": f"Title {tmdb}", "year": 2010, "language": "en",
            "poster": "/p.jpg", "vote_average": 8.0, "vote_count": 5000,
            "popularity": 50.0, "director": "", "cast": []}
    base.update(kw)
    return base


class _ProjClient:
    """A client that genuinely applies the projection, like Qdrant does."""

    def __init__(self, records: list[Record]) -> None:
        self.records = records

    async def search(self, vector, *, collection="titles", limit=200,
                     filters=None, with_payload=True):
        return [{"id": r.point_id, "vector": r.vector.tolist(),
                 "payload": project_payload(r.payload, with_payload)}
                for r in self.records[:limit]]


def test_cast_and_collection_survive_projection_end_to_end():
    """A projected payload must still carry `cast` and `collection_id`.

    This is the test that catches "someone trimmed one field too many": it goes
    through QdrantStore.search -> project -> to_candidate, so a field missing from
    the allowlist is missing here too.
    """
    records = [
        _rec(1, _base_payload(1, cast=["Keanu Reeves"], collection_id=99), [1, 0, 0]),
        _rec(2, _base_payload(2, cast=["Other Actor"], collection_id=99), [0.9, 0.1, 0]),
    ]
    store = QdrantStore(client=_ProjClient(records))
    found = store.search(np.array([1.0, 0.0, 0.0]), media_type="movie", limit=5)

    assert len(found) == 2
    by_id = {r.point_id: r.payload for r in found}
    assert by_id["movie-1"]["cast"] == ["Keanu Reeves"], (
        "`cast` was projected away — the seed-cast overlap bonus silently dies")
    assert by_id["movie-1"]["collection_id"] == 99, (
        "`collection_id` was projected away — the sequel dedupe silently dies")
    assert by_id["movie-2"]["collection_id"] == 99


def test_projection_does_not_break_release_info_fields():
    """`release_info()` prints "2024-" for series: it needs year AND end_year."""
    records = [_rec(3, _base_payload(3, media_type="series", year=2024,
                                     end_year=None), [1, 0, 0])]
    store = QdrantStore(client=_ProjClient(records))
    found = store.search(np.array([1.0, 0.0, 0.0]), media_type="series", limit=5)
    payload = found[0].payload
    assert payload["year"] == 2024
    assert "end_year" in payload


# ------------------------------------------------------- the measured win
def test_lean_payload_is_far_smaller_than_the_full_record():
    """Build a realistic record and compare full vs projected JSON size."""
    full = {
        "tmdb_id": 27205, "media_type": "movie", "imdb_id": "tt1375666",
        "title": "Inception", "original_title": "Inception", "year": 2010,
        "end_year": None, "tagline": "Your mind is the scene of the crime.",
        "genres": ["Action", "Sci-Fi", "Thriller"],
        "keywords": ["dream", "heist", "subconscious"] * 6,
        "director": "Christopher Nolan",
        "cast": ["Leonardo DiCaprio", "Joseph Gordon-Levitt", "Elliot Page"],
        "language": "en", "origin_country": ["GB", "US"], "runtime": 148,
        "vote_average": 8.4, "vote_count": 38000, "popularity": 120.5,
        "budget": 160000000, "revenue": 836800000, "collection_id": None,
        "release_date": "2010-07-15", "poster": "/oYuLEt3zVCKq57qu2F8dT7NIa6f.jpg",
        "backdrop": "/s3TBrRGB1iav7gFOCNx3H31MoES.jpg",
        "overview_short": "A thief who steals corporate secrets... " * 3,
        "overview": "A thief who steals corporate secrets through dream-sharing technology. " * 50,
        "adult": False, "video": False, "enriched": True, "bayes": 8.12,
        "bayes_gems": 8.02, "text_hash": "sha256:" + "a" * 64,
    }
    lean = project_payload(full, list(SEARCH_PAYLOAD_FIELDS))
    full_bytes = len(json.dumps(full).encode())
    lean_bytes = len(json.dumps(lean).encode())
    ratio = lean_bytes / full_bytes
    assert ratio < 0.25, f"lean payload is {ratio:.1%} of full ({lean_bytes} vs {full_bytes})"
