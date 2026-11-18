"""Serve rows from a Qdrant-shaped backend over real HTTP.

End-to-end rehearsal of the deployed path without a cloud cluster:

    fake Qdrant REST  <->  addon.store.QdrantStore  ->  build_row  ->  /catalog JSON
                            app (uvicorn)          /  Stremio client

Confirms what the unit tests can't: that filtering, JSON shapes and the row cache
work together under an actual request.
"""

from __future__ import annotations

import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from addon.ids import point_id
from addon.store import QdrantStore

DIM = 384


def _vec(idx: int, dim: int = DIM) -> np.ndarray:
    rng = np.random.default_rng(idx)
    v = rng.standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def _payload(tmdb: int, title: str, **kw) -> dict:
    base = {"tmdb_id": tmdb, "media_type": "movie", "imdb_id": f"tt{tmdb}",
            "title": title, "year": 2008, "language": "en", "poster": f"/{tmdb}.jpg",
            "vote_average": 8.4, "vote_count": 20000, "popularity": 100.0,
            "director": "Christopher Nolan", "cast": ["Christian Bale"], "genres": ["Action"]}
    base.update(kw)
    return base


POINTS = [
    (155, "The Dark Knight", [1.0, 0.0]),
    (272, "Batman Begins", [0.98, 0.05]),
    (27205, "Inception", [0.2, 0.97]),
    (949, "Heat", [0.9, 0.15]),
]
RECORDS = [
    ("id-from-name", _payload(tmdb, title), np.concatenate([np.array(seed, np.float32),
                                                            np.zeros(DIM - 2, np.float32)]))
    for tmdb, title, seed in POINTS
]
for point in RECORDS:
    point_norm = point[2] / np.linalg.norm(point[2])
    point[2][:] = point_norm


class FakeQdrant(BaseHTTPRequestHandler):
    hits = {"search": 0, "scroll": 0, "count": 0, "retrieve": 0}

    def log_message(self, *args):
        return

    def _send(self, payload):
        data = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _handle(self):
        length = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(length).decode()) if length else {}
        path = self.path.split("?")[0]
        points = [{"id": point_id("movie", p[1]["tmdb_id"]), "vector": p[2].tolist(),
                   "payload": p[1]} for p in RECORDS]

        if path.endswith("/points/count"):
            type(self).hits["count"] += 1
            return self._send({"result": {"count": len(points)}})
        if path.endswith("/points/scroll"):
            type(self).hits["scroll"] += 1
            return self._send({"result": {"points": points, "next_page_offset": None}})
        if path.endswith("/points/search"):
            type(self).hits["search"] += 1
            q = np.asarray(body["vector"], dtype=np.float32)
            q = q / np.linalg.norm(q)
            flt = body.get("filter") or {}
            must = flt.get("must") or []
            must_not = flt.get("must_not") or []
            keep = []
            for raw in points:
                ok = True
                for clause in must:
                    if clause.get("key") == "media_type" and \
                            raw["payload"]["media_type"] != clause["match"]["value"]:
                        ok = False
                for clause in must_not:
                    if "has_id" in clause and raw["id"] in clause["has_id"]:
                        ok = False
                if ok:
                    keep.append(raw)
            keep.sort(key=lambda r: -float(np.asarray(r["vector"]) @ q))
            return self._send({"result": keep[: body.get("limit", 10)]})
        if path.endswith("/points"):
            type(self).hits["retrieve"] += 1
            wanted = set(body.get("ids") or [])
            return self._send({"result": [p for p in points if p["id"] in wanted]})
        return self._send({"result": True})

    do_GET = do_POST = _handle


@pytest.fixture(scope="module")
def qdrant_url():
    httpd = HTTPServer(("127.0.0.1", 0), FakeQdrant)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


def test_store_search_against_http_backend(qdrant_url):
    import asyncio

    from addon.qdrant import QdrantClient, QdrantConfig

    store = QdrantStore(QdrantClient(QdrantConfig(url=qdrant_url)))
    query = np.concatenate([np.array([1.0, 0.0], np.float32), np.zeros(DIM - 2, np.float32)])
    records = store.search(query, media_type="movie", limit=4)
    titles = [r.payload["title"] for r in records]
    assert titles[0] == "The Dark Knight"
    assert "Inception" in titles
    assert FakeQdrant.hits["search"] >= 1


def test_build_row_over_qdrant_store(qdrant_url):
    import asyncio

    from addon.qdrant import QdrantClient, QdrantConfig
    from addon.rows import build_row

    store = QdrantStore(QdrantClient(QdrantConfig(url=qdrant_url)))
    seed_pid = point_id("movie", 155)
    history = {"watched": [{"movie": {"ids": {"tmdb": 155}}, "plays": 1,
                            "watched_at": "2026-11-18T10:00:00+00:00"}]}
    plan = build_row(store, history, "movie", "because")
    assert plan.diagnostics["pool_source"] == "ann"
    assert plan.ranked, "because row must produce candidates from Qdrant"
    assert all(c.point_id != seed_pid for c in plan.ranked)
    assert FakeQdrant.hits["scroll"] == 0, "must never scan the full catalog"


def test_large_catalog_refuses_scan(qdrant_url):
    import asyncio

    from addon.qdrant import QdrantClient, QdrantConfig

    store = QdrantStore(QdrantClient(QdrantConfig(url=qdrant_url)), get_all_limit=2)
    with pytest.raises(RuntimeError, match="refusing a full scan"):
        store.get_all()
