"""Offline harness for the ingest job.

No TMDB key and no Qdrant cluster are needed: a fake TMDB serves realistic
`/discover` and `/{type}/{id}` payloads for a small title list, and the embedder
is replaced by a deterministic hashing vectoriser. That exercises the real code
path — pacing, policy, text recipe, point building, batching, resume — end to end.

    python jobs/offline_run.py            # bootstrap + nightly on fakes
    python jobs/offline_run.py --dry-run  # skip even the fake writes
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from addon.ids import point_id                     # noqa: E402
from jobs import sync_tmdb                         # noqa: E402
from jobs.embed import build_text                  # noqa: E402

log = logging.getLogger("offline")

TITLES = [
    # id, type, title, year, votes, avg, pop, director, cast, lang
    (155, "movie", "The Dark Knight", "2008-07-16", 32000, 8.5, 120.0, "Christopher Nolan",
     ["Christian Bale", "Heath Ledger"], "en"),
    (272, "movie", "Batman Begins", "2005-06-15", 21000, 7.7, 90.0, "Christopher Nolan",
     ["Christian Bale", "Liam Neeson"], "en"),
    (27205, "movie", "Inception", "2010-07-16", 36000, 8.4, 160.0, "Christopher Nolan",
     ["Leonardo DiCaprio", "Tom Hardy"], "en"),
    (949, "movie", "Heat", "1995-12-15", 7000, 7.7, 30.0, "Michael Mann",
     ["Al Pacino", "Robert De Niro"], "en"),
    (475557, "movie", "Joker", "2019-10-04", 26000, 8.2, 150.0, "Todd Phillips",
     ["Joaquin Phoenix"], "en"),
    (1000, "movie", "Obscure Indie", "2019-01-01", 3, 6.0, 1.0, "Someone",
     ["Nobody"], "en"),                       # dropped: low votes, too old
    (2000, "movie", "Brand New Release", "2026-10-01", 0, 0.0, 2.0, "New Director",
     ["New Star"], "en"),                     # kept: recency clause
    (3000, "movie", "No Imdb Title", "2015-01-01", 500, 7.0, 10.0, "X", ["Y"], "en"),
    (4000, "series", "Breaking Bad", "2008-01-20", 14000, 8.9, 95.0, "Vince Gilligan",
     ["Bryan Cranston", "Aaron Paul"], "en"),
    (5000, "movie", "Foreign Gem", "2018-01-01", 900, 8.0, 15.0, "Z", ["W"], "fr"),
    (6000, "movie", "Adult Feature", "2020-01-01", 900, 8.0, 15.0, "Z", ["W"], "en"),
]

ADULT_IDS = {6000}
NO_IMDB_IDS = {3000}


def fake_vector(text: str, dim: int = 384) -> list[float]:
    """Deterministic bag-of-words hashing — good enough to prove the plumbing."""
    vec = np.zeros(dim, dtype=np.float32)
    for token in text.lower().split():
        digest = hashlib.sha256(token.encode()).digest()
        vec[int.from_bytes(digest[:4], "little") % dim] += 1.0
    norm = float(np.linalg.norm(vec))
    return (vec / norm if norm else vec).tolist()


class FakeEmbedder:
    def __init__(self) -> None:
        self.encoded = 0
        self.batch_size = 8

    def encode(self, texts: list[str]) -> list[list[float]]:
        self.encoded += len(texts)
        return [fake_vector(t) for t in texts]


def detail_payload(row: tuple) -> dict[str, Any]:
    tmdb_id, media_type, title, date, votes, avg, pop, director, cast, lang = row
    base: dict[str, Any] = {
        "id": tmdb_id,
        "original_language": lang,
        "vote_average": avg,
        "vote_count": votes,
        "popularity": pop,
        "overview": f"{title} is a film about {title.lower()} things happening to people.",
        "tagline": "",
        "poster_path": f"/{tmdb_id}.jpg",
        "backdrop_path": f"/{tmdb_id}_b.jpg",
        "genres": [{"id": 18, "name": "Drama"}],
        "keywords": {"keywords": [{"name": "test"}]},
        "external_ids": {"imdb_id": None if tmdb_id in NO_IMDB_IDS else f"tt{1000000 + tmdb_id}"},
        "belongs_to_collection": None,
    }
    if media_type == "series":
        base |= {"name": title, "first_air_date": date, "last_air_date": "2013-09-29",
                 "created_by": [{"name": director}],
                 "aggregate_credits": {"cast": [{"name": c} for c in cast], "crew": []},
                 "keywords": {"results": [{"name": "test"}]}}
    else:
        base |= {"title": title, "release_date": date, "budget": 1, "revenue": 2,
                 "credits": {"cast": [{"name": c} for c in cast],
                             "crew": [{"job": "Director", "name": director}]}}
    if tmdb_id in ADULT_IDS:
        base["adult"] = True
    return base


class FakeTMDB:
    def __init__(self) -> None:
        self.calls = 0

    async def details(self, media_type: str, tmdb_id: int) -> dict[str, Any]:
        self.calls += 1
        for row in TITLES:
            if row[0] == tmdb_id and row[1] == media_type:
                return detail_payload(row)
        raise RuntimeError(f"404 {media_type}/{tmdb_id}")

    async def discover(self, media_type: str, page: int, **_: Any) -> dict[str, Any]:
        self.calls += 1
        rows = [r for r in TITLES if r[1] == media_type]
        if page > 1:
            rows = []
        return {"page": page, "total_pages": 1,
                "results": [{"id": r[0], "title" if media_type == "movie" else "name": r[2],
                             "vote_count": r[4], "vote_average": r[5], "popularity": r[6],
                             **({"release_date": r[3]} if media_type == "movie"
                                else {"first_air_date": r[3]})} for r in rows]}

    async def iter_discover(self, media_type: str, *, max_pages: int = 500, **kw: Any):
        data = await self.discover(media_type, 1, **kw)
        for item in data["results"]:
            yield item


class FakeQdrant:
    """Minimal in-memory stand-in for the operations sync_tmdb uses."""

    def __init__(self) -> None:
        self.points: dict[str, dict[str, Any]] = {}
        self.payload_writes = 0
        self.created = False

    async def create_titles_collection(self, recreate: bool = False) -> bool:
        if self.created and not recreate:
            return False
        self.created = True
        if recreate:
            self.points.clear()
        return True

    async def ensure_user_state(self) -> None:
        return None

    async def upsert(self, points: list[dict[str, Any]], **_: Any) -> None:
        # SET semantics, like the real PUT /points: the payload is REPLACED, not
        # merged. Keep this faithful — the #3 data-loss guard depends on it.
        for point in points:
            self.points[point["id"]] = {"id": point["id"], "payload": point["payload"],
                                        "vector": point["vector"]}

    async def update_vectors(self, points: list[tuple[str, Any]], **_: Any) -> None:
        """Replace the VECTOR only. Payload must come through untouched."""
        for pid, vec in points:
            if pid in self.points:
                self.points[pid]["vector"] = list(vec)
            else:
                raise KeyError(f"update_vectors: unknown point {pid}")

    async def get_point(self, point_id: str, *, with_payload: bool = True,
                        with_vector: bool = False, **_: Any) -> dict[str, Any] | None:
        point = self.points.get(point_id)
        if point is None:
            return None
        out: dict[str, Any] = {"id": point_id}
        if with_payload:
            out["payload"] = dict(point["payload"])
        if with_vector:
            out["vector"] = point["vector"]
        return out

    async def scroll(self, *, limit: int = 1000, offset: Any = None,
                     with_payload: Any = True, **_: Any) -> tuple[list[dict[str, Any]], Any]:
        keys = sorted(self.points)
        start = keys.index(offset) if offset in self.points else 0
        window = keys[start:start + limit]
        out = []
        for pid in window:
            payload = dict(self.points[pid]["payload"])
            if isinstance(with_payload, (list, tuple)):
                payload = {k: v for k, v in payload.items() if k in with_payload}
            out.append({"id": pid, "payload": payload})
        nxt = keys[start + limit] if start + limit < len(keys) else None
        return out, nxt

    async def count(self, **_: Any) -> int:
        return len(self.points)

    async def iter_all_payloads(self, **_: Any):
        for point in list(self.points.values()):
            yield point

    async def set_payload(self, payload: dict[str, Any], ids: list[str], **_: Any) -> None:
        for pid in ids:
            if pid in self.points:
                self.points[pid]["payload"].update(payload)
                self.payload_writes += 1

    async def set_payload_batch(self, updates: Any, **_: Any) -> int:
        touched = 0
        for payload, ids in updates:
            ids_list = list(ids)
            await self.set_payload(payload, ids_list)
            touched += len(ids_list)
        return touched


def fake_export(media_type: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return rows


async def main(dry_run: bool) -> None:
    fake_tmdb = FakeTMDB()
    fake_qdrant = FakeQdrant()
    embedder = FakeEmbedder()
    stats = sync_tmdb.Stats()
    year_now = 2026

    # ---- bootstrap path: discover page 1 -> details -> policy -> points
    items = [item async for item in fake_tmdb.iter_discover("movie")]
    ready = await sync_tmdb.fetch_and_build(fake_tmdb, "movie", items, year_now=year_now,
                                            stats=stats, dry_run=dry_run)
    kept = [(r["payload"]["title"], r["payload"]["imdb_id"], r["payload"]["year"]) for r in ready]
    print("\n=== bootstrap: movie page 1 ===")
    for title, imdb, year in kept:
        print(f"  kept  {title:<22} {str(imdb):<12} {year}")
    print(f"  stats {stats.line()}")
    assert not any(t == "Obscure Indie" for t, _, _ in kept), "policy must drop old low-vote"
    assert any(t == "Brand New Release" for t, _, _ in kept), "recency clause must keep 0-vote"
    assert not any(t == "Adult Feature" for t, _, _ in kept), "adult must be dropped"
    assert any(t == "No Imdb Title" for t, _, _ in kept), "kept but flagged missing imdb"
    assert stats.missing_imdb == 1

    await sync_tmdb.upsert_ready(fake_qdrant, embedder, ready, stats, dry_run=dry_run)
    print(f"  expected {len(ready)} points; qdrant has {len(fake_qdrant.points)}")

    # ---- resume: same input must not re-embed anything
    before = embedder.encoded
    again = await sync_tmdb.fetch_and_build(fake_tmdb, "movie", items, year_now=year_now,
                                            stats=stats, dry_run=dry_run)
    await sync_tmdb.upsert_ready(fake_qdrant, embedder, again, stats,
                                 existing=set(fake_qdrant.points), dry_run=dry_run)
    print(f"\n=== resume ===\n  skipped_existing={stats.skipped_existing} "
          f"new embeddings={embedder.encoded - before} (must be 0)")
    assert embedder.encoded == before, "resume must not re-embed"

    # ---- nightly: popularity + votes are payload-only (no embedding)
    print("\n=== nightly: payload refresh with zero embeddings ===")
    for point in fake_qdrant.points.values():
        point["payload"]["popularity"] = 0.0
    export = [{"id": r[0], "popularity": r[6] * 2, "adult": False, "video": False}
              for r in TITLES if r[1] == "movie"]
    payloads = {pid: p["payload"] for pid, p in fake_qdrant.points.items()}
    updates = 0
    for row in export:
        pid = point_id("movie", row["id"])
        if pid in payloads:
            await fake_qdrant.set_payload({"popularity": row["popularity"]}, [pid])
            updates += 1
    before = embedder.encoded
    print(f"  popularity updated on {updates} points; embeddings used: "
          f"{embedder.encoded - before}")
    assert updates == len(fake_qdrant.points) and embedder.encoded == before

    sample = await fake_tmdb.discover("movie", 1)
    from jobs import policy
    mean = policy.catalog_mean_vote(sample["results"])
    print(f"  catalog mean vote estimate: {mean} (log line only)")
    # #4b: the stratified prior is computed over the whole catalog payloads and
    # threaded into apply_quality, so the offline path exercises the real thing
    # rather than the module defaults.
    stats_by_stratum = policy.bayes_stats(payloads.values())
    print("  stratified bayes: " + ", ".join(
        f"{k[0]}/{k[1]} C={v[0]} m={v[1]} m_gems={v[2]}"
        for k, v in sorted(stats_by_stratum.items())))
    pid = point_id("movie", 155)
    merged = {**payloads[pid], **{"vote_average": 8.6, "vote_count": 33000}}
    policy.apply_quality(merged, mean, stats=stats_by_stratum)
    assert merged["bayes"] > 0 and merged["bayes_gems"] > 0, "bayes must never be unset"
    print(f"  bayes for The Dark Knight: {merged['bayes']} (gems {merged['bayes_gems']})")

    # ---- text recipe: the title must lead, because truncation keeps the HEAD
    sample_text = build_text(payloads[pid])
    print(f"\n=== embedding text (first 200 chars) ===\n  {sample_text[:200]}...")
    head = " ".join(sample_text.split()[:6])
    assert "The Dark Knight" in head, f"title must lead the text: {head!r}"
    long_text = build_text({**payloads[pid], "overview": "x " * 2000})
    assert len(long_text.split()) < 200, f"overview not bounded: {len(long_text.split())}"
    print(f"  title in first 6 words; 4,000-char overview -> {len(long_text.split())} words")

    # ---- re-embed guard: vectors + text_hash only, payload retained (#3/M7)
    print("\n=== re-embed guard ===")
    from jobs.reembed import reembed
    import jobs.reembed as reembed_mod
    fake_qdrant.points[pid]["payload"]["text_hash"] = "sha256:stale"
    before_payload = dict(fake_qdrant.points[pid]["payload"])

    class _Shim:
        def __init__(self, *a, **k):
            pass

        def __getattr__(self, name):
            return getattr(fake_qdrant, name)

    _orig = reembed_mod.QdrantClient
    reembed_mod.QdrantClient = _Shim
    try:
        stats_reembed = await reembed(embedder=embedder)
    finally:
        reembed_mod.QdrantClient = _orig
    after_payload = fake_qdrant.points[pid]["payload"]
    for field in ("imdb_id", "poster", "vote_count", "language"):
        if field in before_payload:
            assert after_payload.get(field) == before_payload[field], (
                f"re-embed lost payload field {field}")
    assert after_payload["text_hash"] != "sha256:stale", "text_hash was not rewritten"
    print(f"  re-embedded {stats_reembed['reembedded']} point(s); payload fields intact")
    print("\nALL OFFLINE CHECKS PASSED")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    opts = parser.parse_args()
    asyncio.run(main(opts.dry_run))
