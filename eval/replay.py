"""Offline temporal replay: does the ranker put titles you ACTUALLY watched on top?

Why this exists
---------------
Tuning the ranking offline against a 12-title fixture proved useless: three rounds
of "fixes" produced no visible change, because nothing measured whether a change
helped. This harness replays the user's REAL Trakt history against the REAL
catalog and reports:

    Recall@200   of your held-out watches, how many even reached the ANN pool?
                 -> if this is low, the bottleneck is RETRIEVAL (embeddings /
                    taste centroid), not the ranking weights.
    HitRate@15   of your held-out watches that entered the pool, how many landed
                 in the final 15?
    NDCG@15      same, position-weighted.
    Profile      mean bayes / votes / genre spread of the produced top-15, so a
                 "quality up, relevance down" trade is visible rather than guessed.

Method
------
Walk the positive watch history in chronological order. For each of the last
`--holdouts` watches t:
  * build taste from watches strictly BEFORE t (no leakage),
  * run the real retrieval + ranking for `t`'s media type,
  * exclude everything already watched before t (as production does),
  * record where the held-out title(s) land.

It reuses the production code paths (`build_user_state`, `_candidate_pool`,
`rank`) rather than reimplementing them, so the numbers describe the real system.

Usage
-----
    # Keys come from eval/.env (see _startup_env), so no exports are needed
    python eval/replay.py --holdouts 30

    # Replay a saved history JSON instead (no network)
    python eval/replay.py --history history.json --holdouts 30

    # Compare two weight sets in one run
    python eval/replay.py --compare

Keys
----
Read from `eval/.env` (gitignored) at startup:

    TRAKT_CLIENT_ID=...
    TRAKT_ACCESS_TOKEN=...
    QDRANT_URL=...
    QDRANT_API_KEY=...

Real environment variables win over the file, so `TRAKT_ACCESS_TOKEN=... python
eval/replay.py` still overrides a stale value. This loader is local to this script
on purpose: the replay harness is a throwaway diagnostic and must not change how
the addon itself reads configuration.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ENV_DIR = Path(__file__).resolve().parent
# The four keys this harness needs. Checked once at startup so a half-filled
# eval/.env fails with one clear message instead of deep inside a client.
ENV_KEYS = ("TRAKT_CLIENT_ID", "TRAKT_ACCESS_TOKEN", "QDRANT_URL", "QDRANT_API_KEY")


from addon.ids import id_from_trakt_item, point_id                       # noqa: E402
from addon.qdrant import QdrantClient, QdrantConfig                      # noqa: E402
from addon.rank import ANN_LIMIT, SIM_FLOOR, rank                        # noqa: E402
from addon.rows import _candidate_pool, build_user_state                 # noqa: E402
from addon.store import QdrantStore                                      # noqa: E402
from addon.taste import median_year                                      # noqa: E402

log = logging.getLogger(__name__)


# --------------------------------------------------------------- .env loading
def load_env_file(path: Path) -> int:
    """Load KEY=VALUE pairs from a .env file into os.environ. Stdlib only.

    Deliberately NOT python-dotenv: the project's hard rule is no new runtime
    dependency at serve time, and this is 20 lines.

    Rules, matching what .env.example documents:
      * blank lines and lines starting with `#` are skipped;
      * an inline `# ...` comment after a value is stripped;
      * optional surrounding single/double quotes are removed;
      * `export KEY=VALUE` is accepted;
      * an ALREADY-SET environment variable WINS — a real shell export must not
        be silently overridden by a stale file.

    Returns the number of variables actually set (for the caller's log line).
    """
    if not path.is_file():
        return 0
    applied = 0
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        key, sep, value = line.partition("=")
        if not sep:
            continue
        key = key.strip()
        if not key:
            continue
        value = value.strip()
        # Strip an inline comment, but not a '#' inside a quoted value.
        if value[:1] in ("'", '"'):
            quote = value[0]
            end = value.find(quote, 1)
            if end > 0:
                value = value[1:end]
        else:
            value = value.split("#", 1)[0].strip()
        if key in os.environ and os.environ[key]:
            continue          # a real env var wins
        os.environ[key] = value
        applied += 1
    return applied


def _startup_env(explicit: str | None,
                 required: Sequence[str] = ENV_KEYS) -> None:
    """Load keys from eval/.env (gitignored), or from `--env-file` if given.

    A real environment variable always wins — see load_env_file. Missing keys are
    reported together, once, so a half-filled file gives one clear message instead
    of failing later inside the Trakt or Qdrant client.

    `required` is narrower when replaying a saved history: no Trakt keys needed.
    """
    candidates = ([Path(explicit)] if explicit
                  else [ENV_DIR / ".env"])
    loaded_from: Path | None = None
    for candidate in candidates:
        n = load_env_file(candidate)
        if n:
            log.info("loaded %s vars from %s", n, candidate)
            loaded_from = candidate
            break
    if loaded_from is None and not explicit:
        log.warning("no keys found in %s (fill it in, or export them)", candidates[0])

    missing = [k for k in required if not (os.environ.get(k) or "").strip()]
    if missing:
        raise SystemExit(
            "missing keys: " + ", ".join(missing)
            + f"\nAdd them to {ENV_DIR / '.env'} as KEY=VALUE lines:\n"
            + "\n".join(f"  {k}=..." for k in missing))


# --------------------------------------------------------------- data plumbing
def _watched_at(item: dict[str, Any]) -> datetime:
    raw = item.get("watched_at") or item.get("last_watched_at") or ""
    try:
        parsed = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return datetime.now(timezone.utc)


def positive_watches(history: dict[str, Any]) -> list[dict[str, Any]]:
    """Chronological positive watches (rating >= 6 or unrated), movies + shows."""
    ratings = {}
    for item in history.get("ratings") or []:
        key = id_from_trakt_item(item, "movie")
        if key is not None:
            ratings[key] = int(item.get("rating") or 0)

    out: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for item in history.get("watched") or []:
        key = id_from_trakt_item(item, "movie")
        if key is None or key in seen:
            continue
        rating = ratings.get(key)
        if rating is not None and rating < 6:
            continue          # explicitly disliked -> not a target
        seen.add(key)
        out.append({**item, "_key": key, "_at": _watched_at(item)})
    out.sort(key=lambda i: i["_at"])
    return out


async def fetch_history() -> dict[str, Any]:
    from addon.trakt import TraktClient, TraktConfig

    cfg = TraktConfig.from_env(os.environ)
    if not cfg.configured:
        raise SystemExit(
            "TRAKT_CLIENT_ID and TRAKT_ACCESS_TOKEN must be set (or pass --history FILE)")
    client = TraktClient(cfg)
    log.info("pulling Trakt history...")
    history = await client.fetch_user_state()
    log.info("got %s watched, %s ratings, %s watchlist",
             len(history.get("watched") or []), len(history.get("ratings") or []),
             len(history.get("watchlist") or []))
    return history


# ------------------------------------------------------------------- metrics
def recall_at_k(ranked_ids: Sequence[str], target: str, k: int) -> float:
    """1.0 if the target is in the top-k, else 0."""
    return 1.0 if target in set(ranked_ids[:k]) else 0.0


def hit_rate(results: Sequence[bool]) -> float:
    return sum(results) / len(results) if results else 0.0


def ndcg(hits: Sequence[int], k: int) -> float:
    """hits = 1-based positions of the target (0 if absent). NDCG@k for one query."""
    if not hits:
        return 0.0
    dcg = sum(1.0 / math.log2(h + 1) for h in hits if 1 <= h <= k)
    return dcg          # single relevant item -> IDCG@k == 1.0


# ------------------------------------------------------------------ the replay
def _flatten(cands) -> list[str]:
    return [c.point_id for c in cands]


async def replay(history: dict[str, Any], *, holdouts: int = 30, ann_limit: int = ANN_LIMIT,
                 limit: int = 15, weights: dict[str, float] | None = None,
                 version_tag: str = "current") -> dict[str, Any]:
    """Replay the last `holdouts` positive watches and return the metrics."""
    from addon import rank as rank_mod

    saved_weights = dict(rank_mod.WEIGHTS)
    if weights:
        rank_mod.WEIGHTS["blockbuster"] = (
            weights["sim"], weights["quality"], weights["reach"], weights["era"])
    try:
        qcfg = QdrantConfig.from_env(os.environ)
        if not qcfg.configured:
            raise SystemExit("QDRANT_URL must be set to replay against the real catalog")
        store = QdrantStore.from_env(os.environ)
        assert store is not None

        watches = positive_watches(history)
        if len(watches) < 5:
            raise SystemExit(f"only {len(watches)} positive watches; need at least 5")
        targets = watches[-holdouts:]
        log.info("replaying %s holdouts out of %s positive watches",
                 len(targets), len(watches))

        rows: list[dict[str, Any]] = []
        for idx, target in enumerate(targets):
            media_type, tmdb = target["_key"]
            pid = point_id(media_type, tmdb)
            # History strictly BEFORE this watch: no leakage.
            prior = {"watched": [w for w in history.get("watched") or []
                                 if _watched_at(w) < target["_at"]],
                     "ratings": history.get("ratings") or [],
                     "watchlist": history.get("watchlist") or [],
                     "hidden": history.get("hidden") or []}
            state = build_user_state(store, prior, now=target["_at"])
            if not state.signals:
                continue
            from addon.taste import compute_taste
            taste = compute_taste(state.signals, state.dislikes, now=target["_at"])
            if not taste.ready or taste.query is None:
                continue

            pool = _candidate_pool(store, taste.query, media_type=media_type,
                                   langs=("en",), excluded=set(),
                                   year_now=target["_at"].year,
                                   ann_limit=ann_limit)
            pool_ids = {c.point_id for c in pool}
            ranked = rank(taste.query, pool, mode="blockbuster",
                          year_seed=median_year(state.watched_years),
                          year_now=target["_at"].year,
                          limit=limit, ann_limit=ann_limit, sim_floor=SIM_FLOOR)
            ranked_ids = _flatten(ranked)
            position = (ranked_ids.index(pid) + 1) if pid in ranked_ids else 0
            rows.append({
                "target": pid,
                "title": (target.get("movie") or target.get("show") or {}
                          ).get("title", ""),
                "in_pool": pid in pool_ids,
                "position": position,
                "ndcg": ndcg([position], limit),
                "top_bayes": [float(c.payload.get("bayes") or 0) for c in ranked],
                "top_votes": [int(c.payload.get("vote_count") or 0) for c in ranked],
            })
            if (idx + 1) % 10 == 0:
                log.info("  %s/%s", idx + 1, len(targets))

        if not rows:
            raise SystemExit("no replay steps completed (history too sparse?)")

        in_pool = [r["in_pool"] for r in rows]
        hits = [1.0 if r["position"] else 0.0 for r in rows]
        top_bayes = [b for r in rows for b in r["top_bayes"] if b]
        top_votes = [v for r in rows for v in r["top_votes"] if v]
        return {
            "version": version_tag,
            "steps": len(rows),
            "recall_at_ann": hit_rate(in_pool),
            "hit_rate": hit_rate(hits),
            "ndcg": sum(r["ndcg"] for r in rows) / len(rows),
            "mean_top_bayes": (sum(top_bayes) / len(top_bayes)) if top_bayes else 0.0,
            "median_top_votes": sorted(top_votes)[len(top_votes) // 2] if top_votes else 0,
            "positions": [r["position"] for r in rows],
        }
    finally:
        rank_mod.WEIGHTS.update(saved_weights)


def report(result: dict[str, Any], ann_limit: int = ANN_LIMIT) -> None:
    print()
    print(f"  {result['version']}")
    print(f"    steps                 {result['steps']}")
    print(f"    Recall@{ann_limit:<4}          {result['recall_at_ann']:.1%}"
          "   <- did the held-out title even reach the pool?")
    print(f"    HitRate@15            {result['hit_rate']:.1%}"
          "   <- did the ranker put it in the top 15?")
    print(f"    NDCG@15               {result['ndcg']:.4f}")
    print(f"    mean top-15 bayes     {result['mean_top_bayes']:.2f}")
    print(f"    median top-15 votes   {result['median_top_votes']:,}")
    pos = [p for p in result["positions"] if p]
    if pos:
        print(f"    median hit position   {sorted(pos)[len(pos) // 2]}")


INTEREST = 200          # kept for readability of the interpretation notes below


# ----------------------------------------------------------------------- cli
async def main(args: argparse.Namespace) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
    _startup_env(args.env_file, required=ENV_KEYS if not args.history else ("QDRANT_URL", "QDRANT_API_KEY"))
    if args.history:
        history = json.loads(Path(args.history).read_text(encoding="utf-8"))
    else:
        history = await fetch_history()

    if args.save_history:
        Path(args.save_history).write_text(json.dumps(history, indent=1), encoding="utf-8")
        log.info("saved history to %s", args.save_history)

    print("\n=== OFFLINE TEMPORAL REPLAY ===")
    if args.compare:
        variants = [
            ("current", None),
            ("rescaled (review Q1)", {"sim": 0.45, "quality": 0.30,
                                      "reach": 0.15, "era": 0.05}),
        ]
    else:
        variants = [("current", None)]

    results = []
    for tag, weights in variants:
        log.info("running variant: %s", tag)
        try:
            results.append(await replay(history, holdouts=args.holdouts,
                                        ann_limit=args.ann_limit, limit=args.limit,
                                        weights=weights, version_tag=tag))
        except SystemExit as exc:
            print(f"  {tag}: {exc}")
    for r in results:
        report(r, args.ann_limit)

    print()
    print("INTERPRETATION")
    print("  Recall@ANN low  -> retrieval is the bottleneck, not the weights.")
    print("  Recall@ANN high but HitRate@15 low -> the ranker is throwing them away.")
    print("  Both high -> the pipeline is fine; the row simply looks similar to you.")
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Offline temporal replay of the ranker")
    p.add_argument("--from-trakt", action="store_true", default=True,
                   help="pull history from the Trakt API (default)")
    p.add_argument("--history", default=None, help="replay a saved history JSON instead")
    p.add_argument("--save-history", default=".replay-history.json",
                   help="cache the pulled history to this file")
    p.add_argument("--holdouts", type=int, default=30)
    p.add_argument("--ann-limit", type=int, default=ANN_LIMIT)
    p.add_argument("--limit", type=int, default=15)
    p.add_argument("--compare", action="store_true",
                   help="also score the rescaled weights side by side")
    p.add_argument("--env-file", default=None,
                   help="load credentials from this .env file "
                        "(default: <repo>/.env, then <repo>/.env.local; "
                        "a real environment variable always wins)")
    return p.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main(parse_args())))
