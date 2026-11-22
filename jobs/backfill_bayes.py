"""Full-catalog bayes backfill (#4b).

`refresh()` recomputes `bayes`/`bayes_gems` only for `active_sample` — 25 pages of
popularity-sorted discover, ~1,000 titles. On a 45k catalog that is ~2 % of the
points, so after a deploy most of the catalog still carries the OLD single-mean
values. Observed live: "vote/bayes refresh applied to 915 points" out of 45,177.

This driver does what refresh() never did: scroll the WHOLE catalog, compute the
stratified prior once over it, and recompute `bayes`/`bayes_gems` for every point.

Why this is cheap and safe:
  * payload-only — no embeddings, no vectors touched;
  * writes via `set_payload_batch`, whose `set_payload` operations MERGE keys, so
    no other payload field can be replaced (this is not an upsert);
  * idempotent — recomputing from unchanged inputs yields unchanged values.

Run order:
    python jobs/backfill_bayes.py --dry-run     # stats + counts, writes nothing
    python jobs/backfill_bayes.py               # full catalog (~1-2 min per 10k)
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jobs import policy                                                  # noqa: E402
from jobs.qdrant_client import QdrantClient, QdrantError                  # noqa: E402

log = logging.getLogger(__name__)

# The fields the stratified prior and apply_quality actually read, PLUS the two we
# write. `bayes`/`bayes_gems` must be in the scroll or the idempotency check
# compares against None and re-writes every point on every run.
# Requesting only these keeps the scroll cheap on a 45k catalog (the full payload
# carries a 4,000-char overview per title).
BACKFILL_FIELDS = ["media_type", "language", "vote_average", "vote_count",
                   "bayes", "bayes_gems"]


async def backfill(*, dry_run: bool = False, limit: int | None = None,
                   chunk: int = 500, qdrant_url: str = "", qdrant_key: str = "",
                   ) -> dict[str, int]:
    """Recompute bayes/bayes_gems for every point from a catalog-wide prior."""
    stats = {"scanned": 0, "updated": 0, "skipped": 0}
    qdrant = QdrantClient(qdrant_url, qdrant_key)

    # ---- pass 1: collect the catalog-wide prior ------------------------------
    # The prior MUST come from the whole catalog, not from a popularity sample:
    # see policy.bayes_stats' docstring.
    payloads: dict[str, dict[str, Any]] = {}
    offset = None
    while True:
        points, offset = await qdrant.scroll(limit=1000, offset=offset,
                                             with_payload=BACKFILL_FIELDS)
        for point in points:
            payloads[str(point["id"])] = point.get("payload") or {}
            if limit is not None and len(payloads) >= limit:
                break
        if offset is None or (limit is not None and len(payloads) >= limit):
            break
    stats["scanned"] = len(payloads)
    log.info("scanned %s points", len(payloads))

    stats_by_stratum = policy.bayes_stats(payloads.values())
    for key, (c, m, m_gems) in sorted(stats_by_stratum.items()):
        log.info("  %s/%s: C=%.4f m=%.1f m_gems=%.1f", key[0], key[1], c, m, m_gems)
    if not stats_by_stratum:
        log.warning("catalog is empty or has no usable vote data; nothing to do")
        return stats

    # ---- pass 2: recompute and write -----------------------------------------
    updates: list[tuple[dict[str, Any], list[str]]] = []
    for pid, payload in payloads.items():
        before = (payload.get("bayes"), payload.get("bayes_gems"))
        merged = dict(payload)
        policy.apply_quality(merged, stats=stats_by_stratum)
        new = (merged["bayes"], merged["bayes_gems"])
        if new == before:
            stats["skipped"] += 1
            continue
        updates.append(({"bayes": new[0], "bayes_gems": new[1]}, [pid]))
        if len(updates) >= chunk:
            if not dry_run:
                stats["updated"] += await qdrant.set_payload_batch(updates)
            else:
                stats["updated"] += sum(len(ids) for _p, ids in updates)
            updates = []
            log.info("  %s/%s points updated", stats["updated"], stats["scanned"])
    if updates:
        if not dry_run:
            stats["updated"] += await qdrant.set_payload_batch(updates)
        else:
            stats["updated"] += sum(len(ids) for _p, ids in updates)

    log.info("%s %s points; %s already correct",
             "would update" if dry_run else "updated",
             stats["updated"], stats["skipped"])
    if dry_run:
        log.info("--dry-run: no writes performed")
    return stats


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Full-catalog bayes backfill (#4b)")
    parser.add_argument("--dry-run", action="store_true",
                        help="compute and report; write nothing")
    parser.add_argument("--limit", type=int, default=None,
                        help="only the first N points (smoke run)")
    parser.add_argument("--chunk", type=int, default=500,
                        help="points per set_payload_batch request")
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
    stats = asyncio.run(backfill(dry_run=args.dry_run, limit=args.limit,
                                 chunk=args.chunk, qdrant_url=url, qdrant_key=key))
    log.info("done: %s", stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
