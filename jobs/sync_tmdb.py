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
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
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

# Default crawl languages. This constant used to be defined and never read, so
# the pipeline was English-only by accident rather than by design. It is now the
# real default for `--langs`, and any extra code enables a full extra crawl pass.
DEFAULT_LANGS = ("en",)

# Languages scanned for recent releases by the daily job. Mirrors the curated
# lists this catalog is built from, so a same-week Malayalam/Tamil/Hindi/Korean
# release is caught even before it has votes. `with_original_language` accepts a
# single value only (comma/pipe lists return nothing), hence one call each.
DEFAULT_RECENT_LANGS = ("en", "ml", "hi", "ta", "ko")

# Matches a TMDB watchlist URL and captures the numeric id.
_LIST_URL_RE = re.compile(r"/list/(\d+)")
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
    list_items: int = 0          # raw items read across every --lists source
    list_future: int = 0         # dropped because the release month is in the future
    list_movies: int = 0
    list_series: int = 0
    lists_used: int = 0
    recent_found: int = 0        # candidates from the daily recent-release passes
    started: float = field(default_factory=time.monotonic)

    def as_dict(self) -> dict[str, Any]:
        return {**self.__dict__, "started": None,
                "seconds": round(time.monotonic() - self.started, 1)}

    def line(self) -> str:
        base = (f"fetched={self.fetched} kept={self.kept} dropped={self.dropped} "
                f"no_imdb={self.missing_imdb} errors={self.errors} "
                f"upserted={self.upserted} "
                f"skipped={self.skipped_existing} payload={self.payload_updates} "
                f"reembed={self.reembedded} calls={self.calls} "
                f"{time.monotonic() - self.started:.1f}s")
        if self.lists_used:
            base += (f" lists={self.lists_used} list_items={self.list_items} "
                     f"list_future={self.list_future} list_movies={self.list_movies} "
                     f"list_series={self.list_series}")
        if self.recent_found:
            base += f" recent={self.recent_found}"
        return base


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
                          known: set[str] | None = None,
                          keep_langs: tuple[str, ...] | None = None,
                          curated: bool = False,
                          extra: dict[str, Any] | None = None,
                          bayes_stats: dict[tuple[str, str], tuple[float, float, float]]
                          | None = None) -> list[dict[str, Any]]:
    """Details -> normalize -> policy -> text. Embeddings happen in `upsert_ready`.

    `known` is the set of point ids already in the collection. They are filtered
    out BEFORE the per-title `/details` call: re-running over an already-filled
    decade used to re-fetch every title just to discard it as a duplicate, which
    made repeat runs slow and burned TMDB quota for nothing.

    `bayes_stats` is the stratified prior from `policy.bayes_stats(...)`; without
    it, `normalize_payload` writes the module defaults and #4b's stratification
    never reaches newly ingested titles.
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
            payload = policy.normalize_payload(media_type, details, year_now=year_now,
                                               stats=bayes_stats)
            verdict = policy.should_index(payload, year_now=year_now,
                                          in_trakt=tmdb_id in trakt_ids,
                                          keep_langs=keep_langs, curated=curated)
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
        row = {"payload": payload, "text": text}
        if extra:
            row.update(extra)          # e.g. the language that produced this row
        return row

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


async def existing_payloads(qdrant: QdrantClient, *,
                            fields: Sequence[str] | None = None,
                            ) -> dict[str, dict[str, Any]]:
    """Point id -> payload for the whole collection.

    `fields` restricts the scroll to the keys actually needed. On a 43k-point
    catalog a full-payload scroll was the slowest part of the nightly run (43
    batches at ~1 s each) even though the caller only wanted popularity.
    """
    out: dict[str, dict[str, Any]] = {}
    async for point in qdrant.iter_all_payloads(fields=list(fields) if fields else None):
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
    # Language plan: ("en",) keeps the original single-crawl behaviour. Any extra
    # language adds one crawl per language. `--langs en ja ko` therefore crawls
    # English FIRST and then ja/ko, so an interrupted run still leaves a complete
    # English catalog behind (same reasoning as newest-decade-first).
    languages = tuple(args.langs or DEFAULT_LANGS)
    log.info("bootstrap languages: %s", ", ".join(languages))
    async with TMDBClient(args.tmdb_token, rate=args.rate,
                                concurrency=args.concurrency) as client:
        for lang in languages:
            # Non-English slices get a lower vote floor: those titles have far
            # fewer en-US votes and the English bar would discard them.
            keep_langs = languages if len(languages) > 1 or lang != "en" else None
            for media_type in args.types:
                threshold = _vote_threshold(media_type, lang, len(languages) > 1)
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
                        "with_original_language": lang,
                        "primary_release_date.gte": f"{span_start}-01-01",
                        "primary_release_date.lte": f"{span_end}-12-31",
                    }
                    if media_type == "series":
                        window = {
                            "vote_count.gte": threshold,
                            "with_original_language": lang,
                            "first_air_date.gte": f"{span_start}-01-01",
                            "first_air_date.lte": f"{span_end}-12-31",
                        }
                    # Within a decade, walk by vote count instead of current popularity:
                    # popularity is *today's* buzz, so it buries a 1995 classic under a
                    # 2024 release. Inside a fixed year window, votes ≈ "known film".
                    log.info("bootstrap %s[%s] %s-%s (votes>=%s, sort=%s)", media_type, lang,
                             span_start, span_end, threshold, args.sort_by or "popularity.desc")
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
                                                  keep_langs=keep_langs,
                                                  dry_run=args.dry_run)
                    before = stats.upserted
                    await upsert_ready(qdrant, embedder, ready, stats,
                                       existing=done, dry_run=args.dry_run)
                    done.update(point_id(r["payload"]["media_type"], r["payload"]["tmdb_id"])
                                for r in ready)
                    type_written += stats.upserted - before
                    log.info("after %s[%s] %s-%s: %s", media_type, lang, span_start,
                             span_end, stats.line())
                    if args.recent_only:
                        break
                if type_target is not None:
                    log.info("%s[%s]: wrote %s of %s requested this run", media_type, lang,
                             type_written, type_target)

            # PLAN §5 recency clause: the newest titles have no vote threshold yet.
            # Skipped only when the caller limited the run (that's a probe, not a fill).
            if not args.no_recent_pass:
                for media_type in args.types:
                    date_field = ("primary_release_date" if media_type == "movie"
                                  else "first_air_date")
                    recent = await client.discover(
                        media_type, 1, sort_by=f"{date_field}.desc",
                        with_original_language=lang,
                        **{f"{date_field}.gte": f"{year_now - 1}-01-01"})
                    ready = await fetch_and_build(client, media_type,
                                                  recent.get("results") or [],
                                                  year_now=year_now, stats=stats,
                                                  known=done, keep_langs=keep_langs,
                                                  dry_run=args.dry_run)
                    await upsert_ready(qdrant, embedder, ready, stats, existing=done,
                                       dry_run=args.dry_run)
                    done.update(point_id(r["payload"]["media_type"], r["payload"]["tmdb_id"])
                                for r in ready)

        # ---- curated TMDB lists (supplement, not replace) --------------------
        # Runs after the discover crawl so the crawl's newest-first benefit is
        # kept, and so `done` already holds everything the crawl stored: any list
        # item already present costs zero details calls.
        await _ingest_lists(client, qdrant, embedder, args, stats, done, year_now)
    return stats


async def _ingest_lists(client: TMDBClient, qdrant: QdrantClient, embedder: Embedder,
                        args: argparse.Namespace, stats: Stats, done: set[str],
                        year_now: int) -> None:
    """Ingest every `--lists` source, splitting each list by media type.

    A list mixes movies and series and each item declares its own `media_type`
    (`"tv"` on TMDB's side, `"series"` here), so lists are the one input that does
    NOT obey `--types`.

    List items bypass the vote floor (`curated=True`): the curator already judged
    them worth having, and the point of a list is to reach regional titles that
    sit far below the English vote threshold.
    """
    list_ids: list[str] = getattr(args, "lists", None) or []
    if not list_ids:
        return

    now = datetime.now(timezone.utc)
    today = (now.year, now.month)
    for list_id in list_ids:
        try:
            items = await client.fetch_list(list_id)
        except Exception as exc:  # noqa: BLE001 - one bad list must not kill the run
            stats.errors += 1
            log.warning("list %s could not be read: %s: %s", list_id,
                        type(exc).__name__, exc)
            continue
        stats.lists_used += 1
        stats.list_items += len(items)

        usable: list[dict[str, Any]] = []
        for item in items:
            if is_not_future(item, today):
                usable.append(item)
            else:
                stats.list_future += 1

        by_type: dict[str, list[dict[str, Any]]] = {"movie": [], "series": []}
        for item in usable:
            media_type = _normalise_media_type(item.get("media_type"))
            if media_type is not None:
                by_type[media_type].append(item)
        stats.list_movies += len(by_type["movie"])
        stats.list_series += len(by_type["series"])

        log.info("list %s: %s items (%s movie, %s series, %s future-dated skipped)",
                 list_id, len(items), len(by_type["movie"]), len(by_type["series"]),
                 len(items) - len(usable))

        for media_type in ("movie", "series"):
            candidates = by_type[media_type]
            if not candidates:
                continue
            ready = await fetch_and_build(client, media_type, candidates,
                                          year_now=year_now, stats=stats,
                                          known=done, curated=True,
                                          dry_run=args.dry_run)
            await upsert_ready(qdrant, embedder, ready, stats, existing=done,
                               dry_run=args.dry_run)
            done.update(point_id(r["payload"]["media_type"], r["payload"]["tmdb_id"])
                        for r in ready)
            log.info("after list %s [%s]: %s", list_id, media_type, stats.line())


def _normalise_media_type(value: Any) -> str | None:
    """TMDB list items say `movie` or `tv`; this codebase says `series`."""
    text = str(value or "").strip().lower()
    if text == "movie":
        return "movie"
    if text in ("tv", "series"):
        return "series"
    return None


def _vote_threshold(media_type: str, lang: str, multi_lang: bool) -> int:
    """Vote floor for one (type, language) slice.

    English-only runs keep the original PLAN §5 thresholds so existing behaviour
    is untouched. A non-English crawl, or any multi-language crawl, uses the
    lower foreign floor — otherwise the ja/ko slices mostly fail the vote test
    and the crawl spends quota to import almost nothing.
    """
    foreign = lang != "en" or multi_lang
    if media_type == "series":
        return policy.FOREIGN_SERIES_MIN_VOTES if foreign else policy.SERIES_MIN_VOTES
    return policy.FOREIGN_MOVIE_MIN_VOTES if foreign else policy.MOVIE_MIN_VOTES


def parse_list_ids(values: Sequence[str]) -> list[str]:
    """Turn `--lists` input into list ids.

    Accepts anything a user is likely to paste, because making them strip URLs by
    hand is a needless way to get this wrong:

        8717522
        https://www.themoviedb.org/list/8717522-malayalam
        8717528,8717534

    Raises ValueError naming the offending token rather than silently ignoring it
    (a typo'd list would otherwise look like "the list was empty").
    """
    ids: list[str] = []
    bad: list[str] = []
    for raw in values:
        for token in str(raw).replace(",", " ").split():
            token = token.strip().rstrip("/")
            match = _LIST_URL_RE.search(token)
            candidate = match.group(1) if match else token
            if candidate.isdigit():
                if candidate not in ids:
                    ids.append(candidate)
            else:
                bad.append(token)
    if bad:
        raise ValueError(
            "could not read these as TMDB list ids or urls: " + ", ".join(bad))
    return ids


def parse_month(value: Any) -> tuple[int, int] | None:
    """`"2025-12-05"` -> `(2025, 12)`; None when the date is missing or unusable."""
    text = str(value or "")[:7]
    if len(text) < 7 or text[4] != "-":
        return None
    year, _, month = text.partition("-")
    if not (year.isdigit() and month.isdigit()):
        return None
    month_num = int(month)
    if not 1 <= month_num <= 12:
        return None
    return int(year), month_num


def is_not_future(item: dict[str, Any], today: tuple[int, int]) -> bool:
    """Has this list item already been released?

    TMDB lists are full of announced titles (`release_date` years out) and
    zero-vote placeholders would otherwise flood the catalog. Comparison is by
    (year, month), so anything dated this month or earlier counts as released.

    A missing date is KEPT: TMDB often has no date for very old or obscure
    entries, and silently dropping those would defeat the point of using a
    hand-curated list.
    """
    date = item.get("release_date") or item.get("first_air_date")
    parsed = parse_month(date)
    if parsed is None:
        return True
    return parsed <= today


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
async def daily(args: argparse.Namespace) -> Stats:
    """Every-day sync: add new titles, catch recent releases, light popularity.

    Deliberately does NOT sweep popularity across the whole catalog or recompute
    every score — that is `refresh()`'s job on Sundays. On a 43k-point catalog the
    full sweep used to exceed the 45-minute workflow timeout, so the job was
    killed before it ever added anything. Keeping the daily path small is what
    makes it finish.
    """
    stats = Stats()
    embedder = Embedder(args.model, args.batch)
    qdrant = QdrantClient(args.qdrant_url, args.qdrant_key)
    if not args.dry_run:
        await qdrant.create_titles_collection()
    now = datetime.now(timezone.utc)
    year_now = now.year
    today = now.strftime("%m_%d_%Y")

    async with TMDBClient(args.tmdb_token, rate=args.rate,
                                concurrency=args.concurrency) as client:
        # `popularity` drives the light refresh; the other four feed
        # policy.bayes_stats so newly ingested titles get the same stratified
        # `bayes` as the rest of the catalog (#4b). Without media_type/language/
        # vote_average/vote_count here, daily() would write module defaults and
        # the stratification would never reach new titles.
        stats_fields = ["popularity", "media_type", "language", "vote_average",
                        "vote_count"]
        payloads = (await existing_payloads(qdrant, fields=stats_fields)
                    if not args.dry_run else {})
        # One pass over the catalog; None on a first boot / --dry-run, in which
        # case apply_quality falls back to the module defaults.
        # `media_type` is a stored payload field (point ids are opaque uuid5s, so
        # it cannot be recovered from the id).
        bayes_stats = policy.bayes_stats(payloads.values()) if payloads else None

        for media_type in args.types:
            try:
                export = await download_export(media_type, today)
            except Exception as exc:  # noqa: BLE001
                log.warning("export unavailable for %s: %s", media_type, exc)
                continue
            log.info("%s export: %s ids", media_type, len(export))

            by_id = {int(row["id"]): row for row in export if row.get("id")}

            # Light popularity refresh: only ids we already hold, and the values
            # are already in memory, so this costs no extra requests.
            updates: dict[float, list[str]] = {}
            for tmdb_id, row in by_id.items():
                pid = point_id(media_type, tmdb_id)
                if pid in payloads and "popularity" in row:
                    # Round to one decimal: TMDB popularity is a near-unique
                    # float, so grouping on the exact value produced one group
                    # per title and the "batching" degenerated into ~43k separate
                    # requests, each ~370 ms. One decimal place is far more
                    # precision than a ranking signal needs.
                    updates.setdefault(round(float(row["popularity"]), 1),
                                       []).append(pid)
            if updates:
                if args.dry_run:
                    stats.payload_updates += sum(len(v) for v in updates.values())
                else:
                    stats.payload_updates += await qdrant.set_payload_batch(
                        (({"popularity": pop}, ids) for pop, ids in updates.items()))

            # New ids -> details -> policy -> embed -> upsert (capped per day)
            fresh = [(tmdb_id, row) for tmdb_id, row in by_id.items()
                     if point_id(media_type, tmdb_id) not in payloads]
            fresh.sort(key=lambda pair: -float(pair[1].get("popularity") or 0))
            cap = args.new_cap
            fresh = fresh[:cap]
            log.info("%s new candidates: %s (cap %s)", media_type, len(fresh), cap)
            ready = await fetch_and_build(client, media_type,
                                          [row | {"id": tmdb_id} for tmdb_id, row in fresh],
                                          year_now=year_now, stats=stats,
                                          limit=cap, dry_run=args.dry_run,
                                          keep_langs=getattr(args, "langs", None),
                                          bayes_stats=bayes_stats)
            await upsert_ready(qdrant, embedder, ready, stats,
                               existing=set(payloads), dry_run=args.dry_run)
            payloads.update({point_id(media_type, r["payload"]["tmdb_id"]): r["payload"]
                             for r in ready})

        # Recently released titles, in two passes:
        #   (a) the user's own languages, one discover call each
        #   (b) all languages, so a foreign debut is not missed entirely
        # A title found by both is fetched once: `payloads` is updated between
        # passes, so the second pass dedupes for free.
        if args.recent_days > 0:
            since = (now - timedelta(days=args.recent_days)).strftime("%Y-%m-%d")
            for media_type in args.types:
                date_field = ("primary_release_date" if media_type == "movie"
                              else "first_air_date")
                windows: list[dict[str, Any]] = [
                    {"with_original_language": lang} for lang in args.recent_langs]
                windows.append({})          # no language filter
                for window in windows:
                    label = window.get("with_original_language") or "all"
                    try:
                        found = await _recent_candidates(
                            client, media_type, date_field, since, window,
                            max_pages=args.recent_pages)
                    except Exception as exc:  # noqa: BLE001 - never kill the run
                        stats.errors += 1
                        log.warning("recent pass %s[%s] failed: %s", media_type, label, exc)
                        continue
                    stats.recent_found += len(found)
                    ready = await fetch_and_build(client, media_type, found,
                                                  year_now=year_now, stats=stats,
                                                  known=set(payloads),
                                                  keep_langs=getattr(args, "langs", None),
                                                  dry_run=args.dry_run)
                    await upsert_ready(qdrant, embedder, ready, stats,
                                       existing=set(payloads), dry_run=args.dry_run)
                    for row in ready:
                        payloads[point_id(media_type, row["payload"]["tmdb_id"])] = (
                            row["payload"])
                    log.info("recent %s[%s] since %s: %s", media_type, label, since,
                             stats.line())
    return stats


async def _recent_candidates(client: TMDBClient, media_type: str, date_field: str,
                             since: str, window: dict[str, Any], *,
                             max_pages: int = 2) -> list[dict[str, Any]]:
    """Titles released in the last few days, newest first."""
    return [item async for item in client.iter_discover(
        media_type, max_pages=max_pages, sort_by=f"{date_field}.desc",
        **{f"{date_field}.gte": since}, **window)]


async def refresh(args: argparse.Namespace) -> Stats:
    """Weekly maintenance: popularity across everything, then votes/bayes.

    Split out of the daily job because this is the expensive half. With the
    popularity grouping fixed it is ~200 requests instead of ~43k, but it still
    has no business running every day.
    """
    stats = Stats()
    qdrant = QdrantClient(args.qdrant_url, args.qdrant_key)
    if not args.dry_run:
        await qdrant.create_titles_collection()
    now = datetime.now(timezone.utc)
    year_now = now.year
    today = now.strftime("%m_%d_%Y")

    async with TMDBClient(args.tmdb_token, rate=args.rate,
                                concurrency=args.concurrency) as client:
        payloads = await existing_payloads(qdrant) if not args.dry_run else {}

        # 1. popularity refresh for ALL known points — payload only, no re-embed
        for media_type in args.types:
            try:
                export = await download_export(media_type, today)
            except Exception as exc:  # noqa: BLE001
                log.warning("export unavailable for %s: %s", media_type, exc)
                continue
            by_id = {int(row["id"]): row for row in export if row.get("id")}
            updates: dict[float, list[str]] = {}
            for tmdb_id, row in by_id.items():
                pid = point_id(media_type, tmdb_id)
                if pid in payloads and "popularity" in row:
                    updates.setdefault(round(float(row["popularity"]), 1),
                                       []).append(pid)
            if args.dry_run:
                stats.payload_updates += sum(len(v) for v in updates.values())
            else:
                stats.payload_updates += await qdrant.set_payload_batch(
                    (({"popularity": pop}, ids) for pop, ids in updates.items()))
            log.info("%s popularity: %s points in %s requests", media_type,
                     sum(len(v) for v in updates.values()), len(updates))

        # 2+3. votes/ratings for the active set, then recompute bayes
        updates_by_id: dict[str, dict[str, Any]] = {}
        active_sample: list[dict[str, Any]] = []
        for media_type in args.types:
            async for item in client.iter_discover(media_type, max_pages=args.active_pages,
                                                   sort_by="popularity.desc"):
                active_sample.append(item)
        # The stratified prior is computed over the FULL scrolled catalog, not over
        # `active_sample`. active_sample is the top pages by popularity.desc —
        # overwhelmingly high-vote English titles — so using it would push
        # median(vote_count) up for the xx stratum and make the non-English prior
        # worse than today.
        stats_by_stratum = policy.bayes_stats(payloads.values())
        mean = policy.catalog_mean_vote(active_sample)
        # Kept as a log line only; it must no longer feed apply_quality.
        log.info("catalog mean vote estimate: %s (from %s titles, log only)",
                 mean, len(active_sample))
        log.info("stratified bayes: %s", {
            f"{k[0]}/{k[1]}": f"C={v[0]} m={v[1]} m_gems={v[2]}"
            for k, v in sorted(stats_by_stratum.items())})
        for item in active_sample:
            media_type = "series" if "first_air_date" in item else "movie"
            pid = point_id(media_type, int(item.get("id") or 0))
            if pid not in payloads:
                continue
            merged = {**(payloads.get(pid) or {}), **item, "media_type": media_type}
            policy.apply_quality(merged, mean, stats=stats_by_stratum)
            updates_by_id[pid] = {
                "vote_average": merged.get("vote_average", 0),
                "vote_count": merged.get("vote_count", 0),
                "bayes": merged["bayes"],
                "bayes_gems": merged["bayes_gems"],
            }
        if updates_by_id:
            if not args.dry_run:
                # One request per point made this the second half of the timeout
                # problem; the batch endpoint takes distinct payloads per point.
                stats.payload_updates += await qdrant.set_payload_batch(
                    ((patch, [pid]) for pid, patch in updates_by_id.items()))
            else:
                stats.payload_updates += len(updates_by_id)
        log.info("vote/bayes refresh applied to %s points", len(updates_by_id))

    return stats


# -------------------------------------------------------------------- cli
def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MovieMatch TMDB ingest")
    parser.add_argument("--mode", choices=("bootstrap", "daily", "refresh", "nightly"),
                        default="daily",
                        help="bootstrap = fill the catalog; daily = add new titles + "
                             "recent releases (every day); refresh = weekly popularity "
                             "sweep + votes/bayes. 'nightly' is a legacy alias for "
                             "'daily'.")
    parser.add_argument("--types", nargs="+", default=["movie", "series"],
                        choices=["movie", "series"])
    parser.add_argument("--langs", nargs="+", default=list(DEFAULT_LANGS), metavar="ISO639_1",
                        help="original-language slices to crawl, e.g. --langs en ja ko. "
                             "English is crawled first; every extra language adds a "
                             "full crawl pass and uses the lower foreign vote floor.")
    parser.add_argument("--lists", nargs="+", default=None, metavar="ID_OR_URL",
                        help="TMDB list ids or full URLs to ingest in addition to the "
                             "crawl, e.g. --lists 8717522 "
                             "https://www.themoviedb.org/list/8717528-korean . "
                             "Items bypass the vote floor and ignore --types (each "
                             "list item declares its own movie/series type).")
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
    # --- daily-mode recent-release pass -------------------------------------
    parser.add_argument("--recent-days", type=int, default=3,
                        help="daily mode: also pick up titles released in the last "
                             "N days (0 disables). Catches same-week releases that "
                             "have no votes yet and so miss the export ranking.")
    parser.add_argument("--recent-pages", type=int, default=2,
                        help="daily mode: discover pages per recent pass")
    parser.add_argument("--recent-langs", nargs="+", default=None, metavar="ISO639_1",
                        help="daily mode: scan these languages for recent releases, "
                             "one discover call each. Defaults to "
                             f"{' '.join(DEFAULT_RECENT_LANGS)}.")
    parser.add_argument("--rate", type=float,
                        default=float(os.environ.get("TMDB_RATE", "30")),
                        help="average requests/second. TMDB's documented ceiling is "
                             "~40 req/s (https://developer.themoviedb.org/docs/"
                             "rate-limiting); 30 leaves headroom and 429 back-off "
                             "handles the rest.")
    parser.add_argument("--concurrency", type=int,
                        default=int(os.environ.get("TMDB_CONCURRENCY", "16")),
                        help="in-flight TMDB requests; the pacer still caps the rate")
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--recreate", action="store_true", help="drop + recreate collection")
    parser.add_argument("--dry-run", action="store_true",
                        help="offline self-check: exercise recipe/policy/points against "
                             "the built-in fixture catalog; no TMDB, Qdrant or model access")
    parser.add_argument("--stats-out", default=None, help="write run stats as JSON")
    args = parser.parse_args(argv)
    # Daily-mode recent pass always scans the user's languages plus "all
    # languages" (an empty window), so a foreign debut is never missed entirely.
    if not args.recent_langs:
        args.recent_langs = list(DEFAULT_RECENT_LANGS)
    # Normalise here so a typo'd list id fails before any network work begins.
    if args.lists:
        try:
            args.lists = parse_list_ids(args.lists)
        except ValueError as exc:
            parser.error(str(exc))
        if not args.lists:
            parser.error("--lists was given but contained no list ids")
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
    if args.mode == "bootstrap":
        return await bootstrap(args)
    if args.mode == "refresh":
        return await refresh(args)
    # "nightly" is the legacy name for what is now the daily job; keeping it as
    # an alias means an existing workflow or cron that passes --mode nightly does
    # not silently start running the weekly sweep instead.
    return await daily(args)


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

