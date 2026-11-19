"""Ingest pipeline — bootstrap + nightly (PLAN §6.2, §6.3).

    python jobs/sync_tmdb.py --mode bootstrap   # one-time, resumable, ~60-90 min
    python jobs/sync_tmdb.py --mode nightly     # daily, idempotent, <15 min
    python jobs/sync_tmdb.py --mode nightly --dry-run --limit 50

Design rules taken straight from the plan:
  * embeddings are created ONLY here (the addon never loads a model)
  * popularity/votes are payload updates — never a re-embed (§0.1)
  * every step is idempotent; a failed night self-heals the next one
  * `--limit` and `--dry-run` exist so the pipeline can be exercised on a small
    slice before committing to a 70k-point bootstrap
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from addon.ids import point_id                                    # noqa: E402
from jobs import policy                                            # noqa: E402
from jobs.embed import build_text, text_hash                       # noqa: E402
from jobs.qdrant_client import QdrantClient                        # noqa: E402
from jobs.tmdb import TMDBClient, download_export                  # noqa: E402

log = logging.getLogger("sync")

DEFAULT_LANGS = ("en",)
YEAR_WINDOWS = 3          # bootstrap crawls the catalog in slices to stay resumable


# ----------------------------------------------------------------- embedder
class Embedder:
    """Lazy sentence-transformers wrapper; only the job ever imports this."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 batch_size: int = 64) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self._model: Any = None

    def _load(self) -> Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # heavy, jobs only
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._load().encode(list(texts), batch_size=self.batch_size,
                                      normalize_embeddings=True, show_progress_bar=False)
        return [v.tolist() for v in vectors]


# ------------------------------------------------------------------ stats
@dataclass
class Stats:
    fetched: int = 0
    kept: int = 0
    dropped: int = 0
    missing_imdb: int = 0
    upserted: int = 0
    skipped_existing: int = 0
    payload_updates: int = 0
    reembedded: int = 0
    errors: int = 0
    calls: int = 0
    started: float = field(default_factory=time.monotonic)

    def as_dict(self) -> dict[str, Any]:
        return {**self.__dict__, "started": None,
                "seconds": round(time.monotonic() - self.started, 1)}

    def line(self) -> str:
        return (f"fetched={self.fetched} kept={self.kept} dropped={self.dropped} "
                f"no_imdb={self.missing_imdb} errors={self.errors} "
                f"upserted={self.upserted} "
                f"skipped={self.skipped_existing} payload={self.payload_updates} "
                f"reembed={self.reembedded} calls={self.calls} "
                f"{time.monotonic() - self.started:.1f}s")


# ---------------------------------------------------------------- helpers
def point_for(payload: dict[str, Any], vector: Sequence[float]) -> dict[str, Any]:
    return {
        "id": point_id(payload["media_type"], payload["tmdb_id"]),
        "vector": list(vector),
        "payload": payload,
    }


def enrichable(details: dict[str, Any]) -> bool:
    """Does this TMDB object carry the fields the text recipe needs?"""
    return bool(details.get("id")) and bool(details.get("title") or details.get("name"))


async def fetch_and_build(client: TMDBClient, media_type: str,
                          candidates: Iterable[dict[str, Any]], *,
                          year_now: int, stats: Stats, limit: int | None = None,
                          trakt_ids: set[int] | None = None,
                          dry_run: bool = False,
                          known: set[str] | None = None) -> list[dict[str, Any]]:
    """Details -> normalize -> policy -> text. Embeddings happen in `upsert_ready`.

    `known` is the set of point ids already in the collection. They are filtered
    out BEFORE the per-title `/details` call: re-running over an already-filled
    decade used to re-fetch every title just to discard it as a duplicate, which
    made repeat runs slow and burned TMDB quota for nothing.
    """
    trakt_ids = trakt_ids or set()
    known = known or set()
    ready: list[dict[str, Any]] = []
    seen: set[int] = set()
    skipped = 0

    # Dedupe + skip known BEFORE any network work.
    pending: list[int] = []
    for item in candidates:
        tmdb_id = int(item.get("id") or 0)
        if not tmdb_id or tmdb_id in seen:
            continue
        seen.add(tmdb_id)
        if point_id(media_type, tmdb_id) in known:
            skipped += 1
            continue
        pending.append(tmdb_id)
        if limit is not None and len(pending) >= limit:
            break

    async def one(tmdb_id: int) -> dict[str, Any] | None:
        """Fetch one title. Returns a ready row, or None (dropped / errored)."""
        stats.fetched += 1
        try:
            details = await client.details(media_type, tmdb_id)
            if not enrichable(details):
                stats.dropped += 1
                return None
            payload = policy.normalize_payload(media_type, details, year_now=year_now)
            verdict = policy.should_index(payload, year_now=year_now,
                                          in_trakt=tmdb_id in trakt_ids)
        except Exception as exc:  # noqa: BLE001 - one bad title must not kill the run
            # Covers the network call AND payload normalisation: TMDB occasionally
            # returns explicit nulls for optional fields, and one malformed record
            # used to abort an entire 90-minute bootstrap.
            stats.errors += 1
            log.warning("skipping %s:%s (%s: %s)", media_type, tmdb_id,
                        type(exc).__name__, exc)
            return None
        if not verdict.keep:
            stats.dropped += 1
            return None
        stats.kept += 1
        if verdict.missing == "imdb_id":
            # kept in the catalog but Stremio cannot render it (§5)
            stats.missing_imdb += 1
        text = build_text(payload)
        payload["text_hash"] = text_hash(text)
        return {"payload": payload, "text": text}

    # Bounded fan-out: the client's pacer keeps the average rate at `rate` req/s
    # while up to `concurrency` requests overlap, so latency stops serialising
    # the run. Results are appended in completion order (order is irrelevant
    # here: every row carries its own payload).
    workers = max(1, int(getattr(client, "concurrency", 1) or 1))
    for start in range(0, len(pending), workers):
        chunk = pending[start:start + workers]
        for row in await asyncio.gather(*(one(tid) for tid in chunk)):
            if row is not None:
                ready.append(row)

    if skipped:
        stats.skipped_existing += skipped
        log.info("%s: skipped %s already-indexed titles before fetching details",
                 media_type, skipped)
    stats.calls = client.calls
    return ready


async def upsert_ready(qdrant: QdrantClient, embedder: Embedder,
                       ready: list[dict[str, Any]], stats: Stats,
                       *, existing: set[str] | None = None,
                       dry_run: bool = False) -> None:
    """Embed + upsert a list of ready rows, skipping ids already enriched."""
    if not ready:
        return
    rows = []
    for row in ready:
        point = point_id(row["payload"]["media_type"], row["payload"]["tmdb_id"])
        if existing and point in existing:
            stats.skipped_existing += 1
            continue
        rows.append(row)
    if dry_run or not rows:
        stats.upserted += len(rows)
        return

    batch = embedder.batch_size
    written = 0
    for start in range(0, len(rows), batch):
        chunk = rows[start:start + batch]
        vectors = embedder.encode([r["text"] for r in chunk])
        points = [point_for(r["payload"], v) for r, v in zip(chunk, vectors)]
        await qdrant.upsert(points, wait=False)
        stats.upserted += len(points)
        written += len(points)
        log.info("upserted %s/%s this pass", written, len(rows))


async def existing_ids(qdrant: QdrantClient, *, enriched_only: bool = False) -> set[str]:
    ids: set[str] = set()
    fields = ["enriched"] if enriched_only else []
    async for point in qdrant.iter_all_payloads(fields=fields or None):
        if enriched_only and not (point.get("payload") or {}).get("enriched"):
            continue
        ids.add(str(point["id"]))
    return ids


async def existing_payloads(qdrant: QdrantClient) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    async for point in qdrant.iter_all_payloads():
        out[str(point["id"])] = point.get("payload") or {}
    return out


# --------------------------------------------------------------- bootstrap
async def bootstrap(args: argparse.Namespace) -> Stats:
    stats = Stats()
    embedder = Embedder(args.model, args.batch)
    qdrant = QdrantClient(args.qdrant_url, args.qdrant_key)
    if not args.dry_run:
        created = await qdrant.create_titles_collection(recreate=args.recreate)
        log.info("titles collection %s", "created" if created else "already present")
        await qdrant.ensure_user_state()
        done = await existing_ids(qdrant, enriched_only=True)
        log.info("resume: %s points already enriched", len(done))
    else:
        done = set()

    year_now = datetime.now(timezone.utc).year
    years = bootstrap_years(args)
    log.info("bootstrap years: %s", _describe_years(years))
    async with TMDBClient(args.tmdb_token, rate=args.rate,
                                concurrency=args.concurrency) as client:
        for media_type in args.types:
            threshold = policy.MOVIE_MIN_VOTES if media_type == "movie" else \
                policy.SERIES_MIN_VOTES
            if args.min_votes is not None:
                threshold = args.min_votes
            # --limit is a PER-TYPE budget. It used to be global, so "movie series"
            # with a limit filled movies and silently never reached series.
            type_target = args.limit
            type_written = 0
            for decade in years:
                if type_target is not None and type_written >= type_target:
                    break
                span_start, span_end = decade, min(decade + 9, year_now)
                window = {
                    "vote_count.gte": threshold,
                    "primary_release_date.gte": f"{span_start}-01-01",
                    "primary_release_date.lte": f"{span_end}-12-31",
                }
                if media_type == "series":
                    window = {
                        "vote_count.gte": threshold,
                        "first_air_date.gte": f"{span_start}-01-01",
                        "first_air_date.lte": f"{span_end}-12-31",
                    }
                # Within a decade, walk by vote count instead of current popularity:
                # popularity is *today's* buzz, so it buries a 1995 classic under a
                # 2024 release. Inside a fixed year window, votes ≈ "known film".
                log.info("bootstrap %s %s-%s (votes>=%s, sort=%s)", media_type, span_start,
                         span_end, threshold, args.sort_by or "popularity.desc")
                items = [item async for item in client.iter_discover(
                    media_type, max_pages=args.max_pages,
                    # sort_by is a named parameter of discover(); it must NOT go
                    # through **window or it collides ("multiple values").
                    sort_by=args.sort_by or "popularity.desc", **window)]
                remaining = (None if type_target is None
                             else max(0, type_target - type_written))
                ready = await fetch_and_build(client, media_type, items,
                                              year_now=year_now, stats=stats,
                                              limit=remaining, known=done,
                                              dry_run=args.dry_run)
                before = stats.upserted
                await upsert_ready(qdrant, embedder, ready, stats,
                                   existing=done, dry_run=args.dry_run)
                done.update(point_id(r["payload"]["media_type"], r["payload"]["tmdb_id"])
                            for r in ready)
                type_written += stats.upserted - before
                log.info("after %s %s-%s: %s", media_type, span_start, span_end, stats.line())
                if args.recent_only:
                    break
            if type_target is not None:
                log.info("%s: wrote %s of %s requested this run", media_type,
                         type_written, type_target)

        # PLAN §5 recency clause: the newest titles have no vote threshold yet.
        # Skipped only when the caller limited the run (that's a probe, not a fill).
        if not args.no_recent_pass:
            for media_type in args.types:
                date_field = ("primary_release_date" if media_type == "movie"
                              else "first_air_date")
                recent = await client.discover(
                    media_type, 1, sort_by=f"{date_field}.desc",
                    **{f"{date_field}.gte": f"{year_now - 1}-01-01"})
                ready = await fetch_and_build(client, media_type,
                                              recent.get("results") or [],
                                              year_now=year_now, stats=stats,
                                              known=done, dry_run=args.dry_run)
                await upsert_ready(qdrant, embedder, ready, stats, existing=done,
                                   dry_run=args.dry_run)
                done.update(point_id(r["payload"]["media_type"], r["payload"]["tmdb_id"])
                            for r in ready)
    return stats


def bootstrap_years(args: argparse.Namespace) -> list[int]:
    """Decade starts to crawl, newest first.

    The original crawl went 1960 -> 2020 in order, so a partial run filled the
    catalog with 1960s/70s films and the user's For You row was all classic movies.
    Newest-first means even an interrupted run leaves a *modern* catalog.
    """
    year_now = datetime.now(timezone.utc).year
    if args.decades:
        starts = []
        for token in args.decades:
            start = int(token)
            starts.append(start - start % 10)
        starts = sorted(set(starts), reverse=True)
    elif args.year_from or args.year_to:
        low = args.year_from if args.year_from is not None else 1900
        high = args.year_to if args.year_to is not None else year_now
        starts = sorted(set(range(low - low % 10, high + 1, 10)), reverse=True)
    else:
        starts = list(range(1960, year_now, 10))
        starts.sort(reverse=True)
    return [s for s in starts if s <= year_now]


def _describe_years(starts: list[int]) -> str:
    return ", ".join(f"{s}-{s + 9}" for s in starts) if starts else "(none)"


# ----------------------------------------------------------------- nightly
async def nightly(args: argparse.Namespace) -> Stats:
    stats = Stats()
    embedder = Embedder(args.model, args.batch)
    qdrant = QdrantClient(args.qdrant_url, args.qdrant_key)
    if not args.dry_run:
        await qdrant.create_titles_collection()
    year_now = datetime.now(timezone.utc).year
    today = datetime.now(timezone.utc).strftime("%m_%d_%Y")

    async with TMDBClient(args.tmdb_token, rate=args.rate,
                                concurrency=args.concurrency) as client:
        payloads = await existing_payloads(qdrant) if not args.dry_run else {}

        # 1+2. daily exports -> new ids and popularity refresh
        for media_type in args.types:
            try:
                export = await download_export(media_type, today)
            except Exception as exc:  # noqa: BLE001
                log.warning("export unavailable for %s: %s", media_type, exc)
                continue
            log.info("%s export: %s ids", media_type, len(export))

            by_id = {int(row["id"]): row for row in export if row.get("id")}
            # 4. popularity refresh for ALL known points — payload only, no re-embed
            updates: dict[float, list[str]] = {}
            for tmdb_id, row in by_id.items():
                pid = point_id(media_type, tmdb_id)
                if pid in payloads and "popularity" in row:
                    updates.setdefault(float(row["popularity"]), []).append(pid)
            for popularity, ids in updates.items():
                if args.dry_run:
                    stats.payload_updates += len(ids)
                    continue
                await qdrant.set_payload({"popularity": popularity}, ids)
                stats.payload_updates += len(ids)

            # 3. new ids -> details -> policy -> embed -> upsert (capped per day)
            fresh = [(tmdb_id, row) for tmdb_id, row in by_id.items()
                     if point_id(media_type, tmdb_id) not in payloads]
            fresh.sort(key=lambda pair: -float(pair[1].get("popularity") or 0))
            cap = args.new_cap
            fresh = fresh[:cap]
            log.info("%s new candidates: %s (cap %s)", media_type, len(fresh), cap)
            ready = await fetch_and_build(client, media_type,
                                          [row | {"id": tmdb_id} for tmdb_id, row in fresh],
                                          year_now=year_now, stats=stats,
                                          limit=cap, dry_run=args.dry_run)
            await upsert_ready(qdrant, embedder, ready, stats,
                               existing=set(payloads), dry_run=args.dry_run)

        # 5+6. votes/ratings for the active set, then recompute bayes
        updates_by_id: dict[str, dict[str, Any]] = {}
        active_sample: list[dict[str, Any]] = []
        for media_type in args.types:
            async for item in client.iter_discover(media_type, max_pages=args.active_pages,
                                                   sort_by="popularity.desc"):
                active_sample.append(item)
        mean = policy.catalog_mean_vote(active_sample)
        log.info("catalog mean vote estimate: %s (from %s titles)",
                 mean, len(active_sample))
        for item in active_sample:
            media_type = "series" if "first_air_date" in item else "movie"
            pid = point_id(media_type, int(item.get("id") or 0))
            if pid not in payloads:
                continue
            merged = {**(payloads.get(pid) or {}), **item, "media_type": media_type}
            policy.apply_quality(merged, mean)
            updates_by_id[pid] = {
                "vote_average": merged.get("vote_average", 0),
                "vote_count": merged.get("vote_count", 0),
                "bayes": merged["bayes"],
                "bayes_gems": merged["bayes_gems"],
            }
        for pid, patch in updates_by_id.items():
            if not args.dry_run:
                await qdrant.set_payload(patch, [pid])
            stats.payload_updates += 1
        log.info("vote/bayes refresh applied to %s points", len(updates_by_id))

    return stats


# -------------------------------------------------------------------- cli
def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MovieMatch TMDB ingest")
    parser.add_argument("--mode", choices=("bootstrap", "nightly"), default="nightly")
    parser.add_argument("--types", nargs="+", default=["movie", "series"],
                        choices=["movie", "series"])
    parser.add_argument("--limit", type=int, default=None,
                        help="stop after N upserted points (smoke runs)")
    parser.add_argument("--new-cap", type=int, default=3000,
                        help="max new titles added per night (PLAN §6.3)")
    parser.add_argument("--active-pages", type=int, default=25,
                        help="discover pages for the votes refresh")
    parser.add_argument("--max-pages", type=int, default=500)
    parser.add_argument("--sort-by", default="vote_count.desc",
                        help="discover sort inside each decade slice. Default walks "
                             "biggest titles first; 'popularity.desc' = current buzz")
    # --- which decades to crawl (bootstrap_years reads all of these) -----------
    parser.add_argument("--decades", nargs="+", default=None, metavar="YEAR",
                        help="only these decades, e.g. --decades 1990 2000 2010 2020")
    parser.add_argument("--year-from", type=int, default=None,
                        help="crawl decades starting at this year (e.g. 1990)")
    parser.add_argument("--year-to", type=int, default=None,
                        help="crawl decades up to this year (e.g. 2026)")
    parser.add_argument("--recent-only", action="store_true",
                        help="crawl just the newest decade pass, then stop")
    parser.add_argument("--no-recent-pass", action="store_true",
                        help="skip the 0-vote recency pass (probes only)")
    parser.add_argument("--min-votes", type=int, default=None,
                        help="override the vote_count threshold for every slice")
    parser.add_argument("--rate", type=float, default=float(os.environ.get("TMDB_RATE", "15")))
    parser.add_argument("--concurrency", type=int,
                        default=int(os.environ.get("TMDB_CONCURRENCY", "8")),
                        help="in-flight TMDB requests; the pacer still caps the rate")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--recreate", action="store_true", help="drop + recreate collection")
    parser.add_argument("--dry-run", action="store_true",
                        help="offline self-check: exercise recipe/policy/points against "
                             "the built-in fixture catalog; no TMDB, Qdrant or model access")
    parser.add_argument("--stats-out", default=None, help="write run stats as JSON")
    args = parser.parse_args(argv)
    args.tmdb_token = os.environ.get("TMDB_TOKEN") or os.environ.get("TMDB_API_KEY") or ""
    args.qdrant_url = os.environ.get("QDRANT_URL", "")
    args.qdrant_key = os.environ.get("QDRANT_API_KEY", "")
    if not args.dry_run and not args.tmdb_token:
        parser.error("TMDB_TOKEN is required (or pass --dry-run)")
    if not args.dry_run and not args.qdrant_url:
        parser.error("QDRANT_URL is required (or pass --dry-run)")
    return args


async def run(args: argparse.Namespace) -> Stats:
    if args.dry_run:
        return await dry_run()
    return await (bootstrap(args) if args.mode == "bootstrap" else nightly(args))


async def dry_run() -> Stats:
    """Offline self-check against the built-in fixture catalog (no network, no model).

    This is the smallest thing that can fail before a real run: the text recipe,
    the policy, point construction and batching.
    """
    from jobs.offline_run import FakeEmbedder, FakeTMDB, FakeQdrant

    log.info("dry-run: offline fixture catalog; no TMDB/Qdrant/model access")
    fake_qdrant = FakeQdrant()
    await fake_qdrant.create_titles_collection()
    fake_client = FakeTMDB()
    stats = Stats()
    year_now = datetime.now(timezone.utc).year
    for media_type in ("movie", "series"):
        items = [item async for item in fake_client.iter_discover(media_type)]
        ready = await fetch_and_build(fake_client, media_type, items,
                                      year_now=year_now, stats=stats)
        await upsert_ready(fake_qdrant, FakeEmbedder(), ready, stats)
    log.info("dry-run produced %s points", len(fake_qdrant.points))
    return stats


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(message)s")
    args = parse_args(argv)
    log.info("mode=%s types=%s dry_run=%s", args.mode, args.types, args.dry_run)
    stats = asyncio.run(run(args))
    log.info("done: %s", stats.line())
    if args.stats_out:
        Path(args.stats_out).write_text(json.dumps(stats.as_dict(), indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

