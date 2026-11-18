"""The ingest job must be able to construct and drive the real Qdrant client.

The first live bootstrap run died with `TypeError: QdrantClient.__init__() takes 2
positional arguments but 3 were given` because every job test used a fake. These
tests run the REAL client against a tiny local HTTP server that speaks enough
Qdrant REST to matter, so a signature or method rename fails here instead of in CI.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from jobs import sync_tmdb
from jobs.qdrant_client import IngestQdrant, QdrantClient, QdrantError


class Recorder(BaseHTTPRequestHandler):
    """Records requests and answers like Qdrant for the endpoints we use."""

    requests: list[tuple[str, str, dict]] = []
    collections: set[str] = set()          # real Qdrant 404s unknown collections

    def log_message(self, *args):  # silence
        return

    def _send(self, code: int, payload: dict | None = None) -> None:
        data = json.dumps(payload or {}).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _handle(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length).decode() if length else ""
        payload = json.loads(raw) if raw else {}
        type(self).requests.append(
            (self.command, self.path, payload, self.headers.get("api-key") or "")
        )

        if self.path.endswith("/points/scroll"):
            return self._send(200, {"result": {"points": [], "next_page_offset": None}})
        if "points/count" in self.path:
            return self._send(200, {"result": {"count": 7}})

        name = self.path.split("/collections/", 1)[-1].split("/")[0] if "/collections/" in self.path else ""
        if self.command == "GET" and self.path.rstrip("/").endswith(f"/collections/{name}"):
            if name in type(self).collections:
                return self._send(200, {"result": {"status": "green"}})
            return self._send(404, {"status": {"error": "Not found"}})
        if self.command == "PUT" and self.path.rstrip("/").endswith(f"/collections/{name}"):
            type(self).collections.add(name)
            return self._send(200, {"result": True})
        return self._send(200, {"result": True})

    do_GET = _handle
    do_PUT = _handle
    do_POST = _handle
    do_DELETE = _handle


@pytest.fixture()
def server():
    Recorder.requests = []
    Recorder.collections = set()
    httpd = HTTPServer(("127.0.0.1", 0), Recorder)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


def test_job_client_takes_url_and_key(server):
    """The exact call the job makes — this is what regressed."""
    client = sync_tmdb.QdrantClient(server, "test-key")
    assert isinstance(client, IngestQdrant)


def test_alias_and_empty_url():
    assert QdrantClient is IngestQdrant
    with pytest.raises(QdrantError):
        IngestQdrant("")


def test_create_upsert_scroll_payload(server):
    import asyncio

    client = sync_tmdb.QdrantClient(server, "k")

    async def run():
        created = await client.create_titles_collection()
        assert created is True
        # second call is a no-op, not a destructive recreate
        assert await client.create_titles_collection() is False
        await client.ensure_user_state()
        await client.upsert([{"id": "abc", "vector": [0.0] * 384, "payload": {"a": 1}}])
        await client.set_payload({"popularity": 1.0}, ["abc"])
        assert [p async for p in client.iter_all_payloads()] == []
        assert await client.count() == 7

    asyncio.run(run())
    paths = [path.split("?")[0] for _, path, _, _ in Recorder.requests]
    assert "/collections/titles" in paths
    assert "/collections/titles/points" in paths
    assert "/collections/titles/points/payload" in paths
    assert "/collections/titles/points/count" in paths
    assert any(p == "/collections/titles/index" for p in paths)


def test_api_key_is_sent_as_header(server):
    import asyncio

    client = sync_tmdb.QdrantClient(server, "secret-key")
    asyncio.run(client.count())
    assert Recorder.requests[-1][3] == "secret-key"


def test_forbidden_raises_helpful_error():
    import asyncio

    class Forbidden(BaseHTTPRequestHandler):
        def log_message(self, *args):
            return

        def _handle(self):
            self.send_response(403)
            self.send_header("Content-Length", "0")
            self.end_headers()

        do_GET = do_PUT = do_POST = do_DELETE = _handle

    httpd = HTTPServer(("127.0.0.1", 0), Forbidden)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        client = sync_tmdb.QdrantClient(f"http://127.0.0.1:{httpd.server_address[1]}", "bad")
        with pytest.raises(QdrantError, match="QDRANT_API_KEY"):
            asyncio.run(client.count())
    finally:
        httpd.shutdown()
