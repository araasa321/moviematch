"""#3 / M7 — re-embed existing titles after an embedding-recipe change.

Any reordering in `jobs.embed.build_text` changes `text_hash` for EVERY title, so
an M7 deploy is not a no-op: the stored vectors keep describing the OLD text until
a re-embed runs.

DATA-LOSS GUARD (the reason this module exists as its own step)
---------------------------------------------------------------
`QdrantClient.upsert` is SET semantics: `PUT /points` replaces the stored payload
wholesale. Re-upserting points built from the 8 recipe fields would therefore
erase `imdb_id`, `poster`, `bayes`, `language`, `collection_id`, `vote_count`,
`enriched`, … across the whole catalog — a full outage caused by maintenance.
This driver therefore writes exactly two things, neither of which is a point
upsert:

  1. the vector      -> POST /collections/titles/points/vectors  (payload untouched)
  2. the text_hash   -> POST /collections/titles/points/batch    (set_payload MERGES keys)

and it verifies (1) actually happened without disturbing the payload, on a single
point, before touching the rest of the catalog.

Run order (see --dry-run):
    python jobs/reembed.py --dry-run --limit 5     # counts, writes nothing
    python jobs/reembed.py --limit 5               # one small real batch
    python jobs/reembed.py                         # the full sweep
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path
from typing import Any, Sequence

# Same bootstrap as sync_tmdb.py: `python jobs/reembed.py` must work from the repo
# root (which is where the workflows run), so the package root goes on sys.path.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jobs.embed import build_text, text_hash                              # noqa: E402
from jobs.qdrant_client import QdrantClient, QdrantError                  # noqa: E402

log = logging.getLogger(__name__)

# Exactly the fields build_text reads. Different code path from serving's #2b
# allowlist: this scroll must ask for them explicitly.
RECIPE_FIELDS = ["title", "tagline", "genres", "keywords", "director", "cast",
                 "overview", "overview_short"]
# Present on every real record and NOT part of the recipe: if any of these
# changes, something overwrote the payload and the run must stop immediately.
CANARY_FIELDS = ("imdb_id", "poster", "vote_count", "language")


def _plan(record: dict[str, Any]) -> tuple[str, str] | None:
    """(new_text, new_hash) for one scrolled record, or None if already current."""
    payload = record.get("payload") or {}
    text = build_text(payload)
    new_hash = text_hash(text)
    if payload.get("text_hash") == new_hash:
        return None
    return text, new_hash


async def _collect(qdrant: QdrantClient, *, limit: int | None,
                   ) -> list[tuple[str, str]]:
    """Scroll the catalog and return [(point_id, new_hash)] needing a re-embed."""
    todo: list[tuple[str, str]] = []
    offset = None
    scanned = 0
    while True:
        points, offset = await qdrant.scroll(limit=1000, offset=offset,
                                             with_payload=[*RECIPE_FIELDS, "text_hash"])
        for point in points:
            scanned += 1
            planned = _plan(point)
            if planned is not None:
                todo.append((str(point["id"]), planned[1]))
                if limit is not None and len(todo) >= limit:
                    log.info("scanned %s points; stopping at --limit %s", scanned, limit)
                    return todo
        if offset is None:
            break
    log.info("scanned %s points", scanned)
    return todo


async def reembed(*, dry_run: bool = False, limit: int | None = None,
                  batch: int = 128, qdrant_url: str = "", qdrant_key: str = "",
                  embedder: Any | None = None) -> dict[str, int]:
    """Re-embed titles whose text_hash no longer matches the current recipe."""
    stats = {"scanned": 0, "changed": 0, "reembedded": 0, "unchanged": 0}
    qdrant = QdrantClient(qdrant_url, qdrant_key)

    total = await qdrant.count()
    todo = await _collect(qdrant, limit=limit)
    stats["scanned"] = total
    stats["changed"] = len(todo)
    stats["unchanged"] = max(0, total - len(todo))
    log.info("%s/%s points need re-embedding", len(todo), total)
    if dry_run:
        log.info("--dry-run: no writes performed")
        return stats
    if not todo:
        return stats

    if embedder is None:
        from jobs.sync_tmdb import Embedder          # lazy: heavy import
        embedder = Embedder()
    # ---- PREFLIGHT: prove update_vectors works and spares the payload ---------
    probe_id = todo[0][0]
    before = await qdrant.get_point(probe_id, with_payload=True, with_vector=True)
    if before is None:
        raise QdrantError(f"preflight: point {probe_id} vanished between scroll and fetch")

    # Rebuild the text for this one point from the FULL record so both the vector
    # and the hash we write belong together.
    full = before.get("payload") or {}
    probe_text = build_text(full)
    probe_vec = embedder.encode([probe_text])[0]
    try:
        await qdrant.update_vectors([(probe_id, probe_vec)], wait=True)
    except QdrantError as exc:
        # Fail BEFORE any payload write. The text_hash write below never runs, so
        # a rejected vector update leaves the point exactly as it was.
        raise QdrantError(
            f"preflight: update_vectors rejected ({exc}). NOTHING was written — "
            f"the catalog is unchanged. Common cause: an unsupported endpoint or "
            f"HTTP method for your Qdrant version (this needs Qdrant >= 1.2.0 and "
            f"PUT /collections/titles/points/vectors).") from exc
    await qdrant.set_payload_batch([({"text_hash": text_hash(probe_text)}, [probe_id])])

    after = await qdrant.get_point(probe_id, with_payload=True, with_vector=True)
    if after is None:
        raise QdrantError(f"preflight: point {probe_id} disappeared after update_vectors")
    lost = [f for f in CANARY_FIELDS
            if f in (before.get("payload") or {})
            and (after.get("payload") or {}).get(f) != (before.get("payload") or {}).get(f)]
    if lost:
        raise QdrantError(
            f"PREFLIGHT FAILED: re-embedding {probe_id} changed or removed {lost}. "
            f"Aborting BEFORE touching the catalog. Restore from the snapshot taken "
            f"before this run.")
    if (after.get("payload") or {}).get("text_hash") != text_hash(probe_text):
        raise QdrantError(f"preflight: text_hash was not written for {probe_id}")
    log.info("preflight OK on %s: vector replaced, %s payload fields intact",
             probe_id, len(after.get("payload") or {}))
    stats["reembedded"] += 1
    todo = todo[1:]

    # ---- the sweep -----------------------------------------------------------
    for start in range(0, len(todo), batch):
        chunk = todo[start:start + batch]
        # Re-fetch each point's FULL payload once, purely to rebuild the source
        # TEXT. The write below is still vectors + text_hash: nothing here goes
        # through upsert(), so no payload field can be replaced.
        pids: list[str] = []
        texts: list[str] = []
        hashes: list[tuple[dict[str, Any], list[str]]] = []
        for pid, _new_hash in chunk:
            record = await qdrant.get_point(pid, with_payload=True)
            if record is None:
                log.warning("point %s vanished mid-run; skipping", pid)
                continue
            text = build_text(record.get("payload") or {})
            pids.append(pid)
            texts.append(text)
            hashes.append(({"text_hash": text_hash(text)}, [pid]))
        if not pids:
            continue
        encoded = embedder.encode(texts)
        await qdrant.update_vectors([(pid, vec) for pid, vec in zip(pids, encoded)],
                                    wait=False)
        await qdrant.set_payload_batch(hashes)
        stats["reembedded"] += len(pids)
        log.info("re-embedded %s/%s", stats["reembedded"], stats["changed"])
    return stats


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Re-embed the titles catalog (#3/M7)")
    parser.add_argument("--dry-run", action="store_true",
                        help="report how many titles would be re-embedded; write nothing")
    parser.add_argument("--limit", type=int, default=None,
                        help="stop after N changed points (smoke run)")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    import os

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(message)s")
    args = parse_args(argv)
    url = os.environ.get("QDRANT_URL", "").strip()
    key = os.environ.get("QDRANT_API_KEY", "").strip()
    if not url:
        raise SystemExit("QDRANT_URL is required")
    embedder = None
    if not args.dry_run:
        from jobs.sync_tmdb import Embedder          # lazy: heavy import
        embedder = Embedder(args.model, args.batch)
    stats = asyncio.run(reembed(dry_run=args.dry_run, limit=args.limit, batch=args.batch,
                                qdrant_url=url, qdrant_key=key, embedder=embedder))
    log.info("done: %s", stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
