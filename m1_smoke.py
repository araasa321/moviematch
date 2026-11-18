"""M1 offline smoke: ingest fixtures -> ranker -> Stremio metas.

Proves the two halves fit before any credentials exist. The job's fixture catalog
goes through the *real* ingest functions (`fetch_and_build` -> policy -> recipe),
the resulting points are loaded into a store, and the *real* row builder ranks
them into Stremio metas.

    python m1_smoke.py
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from addon.config import Config
from addon.ids import point_id
from addon.memory_store import MemoryStore, load_records_from_json
from addon.rows import build_row, to_metas
from jobs import sync_tmdb
from jobs.offline_run import FakeEmbedder, FakeTMDB


async def ingest() -> MemoryStore:
    """Run the real ingest path over the fake TMDB and collect the points."""
    client = FakeTMDB()
    embedder = FakeEmbedder()
    stats = sync_tmdb.Stats()
    store = MemoryStore()
    records = []
    for media_type in ("movie", "series"):
        items = [item async for item in client.iter_discover(media_type)]
        ready = await sync_tmdb.fetch_and_build(client, media_type, items,
                                                year_now=datetime.now(timezone.utc).year,
                                                stats=stats)
        for start in range(0, len(ready), embedder.batch_size):
            chunk = ready[start:start + embedder.batch_size]
            for row, vector in zip(chunk, embedder.encode([r["text"] for r in chunk])):
                payload = row["payload"]
                records.append({
                    "point_id": point_id(payload["media_type"], payload["tmdb_id"]),
                    "vector": vector,
                    "payload": payload,
                })
    load_records_from_json(store, records)
    print(f"ingest: {stats.line()}")
    print(f"store : {len(store)} points")
    return store


def main() -> None:
    store = asyncio.run(ingest())
    now = datetime.now(timezone.utc)

    # A user who watched The Dark Knight (trakt-shaped history, tmdb ids only)
    history = {"watched": [{"movie": {"ids": {"tmdb": 155}}, "plays": 2,
                            "watched_at": now.isoformat()}]}
    config = Config(secret="dev", mode="blockbuster", langs=("en",))

    for row in ("because", "foryou"):
        plan = build_row(store, history, "movie", row, config, now=now)
        print(f"\n{row}: {len(plan.ranked)} ranked"
              f" ({plan.diagnostics.get('reason') or 'ok'})")
        for meta in to_metas(plan.ranked, limit=6):
            print(f"   {meta['id']:<12} {meta['name']:<22} {meta.get('releaseInfo', ''):<6} "
                  f"rating={meta.get('imdbRating', '-')}")
        if row == "because":
            assert plan.ranked, "because row must produce candidates"
            assert all(m["id"].startswith("tt") for m in to_metas(plan.ranked))
    print("\nM1 SMOKE PASSED — ingest output is rankable and renders as tt-id metas")


if __name__ == "__main__":
    main()
