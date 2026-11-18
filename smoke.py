"""Local smoke run — no network, no Qdrant, no API keys.

    python smoke.py

Pipeline under test:
    Trakt history -> signals -> taste vector -> guards -> score -> rank -> metas
then a scale check: ANN + rank latency at the p99 personal catalog size (70k).
"""

from __future__ import annotations

import time

import numpy as np

from addon.config import Config
from addon.fixtures import catalog, trakt_history
from addon.memory_store import MemoryStore, Record, load_records_from_json
from addon.rank import Candidate, normalize, rank
from addon.rows import build_row, build_user_state, to_metas
from addon.taste import compute_taste

CONFIG = Config(secret="dev", mode="blockbuster", langs=("en",))


def banner(text: str) -> None:
    print(f"\n{'=' * 74}\n{text}\n{'=' * 74}")


def show(store, history, media_type: str, row: str) -> None:
    started = time.perf_counter()
    plan = build_row(store, history, media_type, row, CONFIG)
    elapsed = (time.perf_counter() - started) * 1000
    banner(f"{media_type} / {row}   {elapsed:.1f} ms · {len(plan.ranked)} ranked · "
           f"excluded {len(plan.excluded)}")
    if not plan.ranked:
        print(f"  (empty — {plan.diagnostics})")
        return
    print(f"  {'#':>2}  {'title':<26} {'sim':>6} {'bayes':>6} {'reach':>6} {'score':>6}  imdb")
    for i, cand in enumerate(plan.ranked[:12], 1):
        p = cand.payload
        print(f"  {i:>2}  {p['title'][:26]:<26} {cand.similarity:>6.3f} "
              f"{cand.extras.get('quality', 0):>6.2f} {cand.extras.get('reach', 0):>6.2f} "
              f"{cand.score:>6.3f}  {p['imdb_id']}")


def scale_check(n: int = 70_000, dim: int = 384, queries: int = 20) -> None:
    """PLAN §2 sizing / §11.2 latency: does the funnel fit the free tier?"""
    rng = np.random.default_rng(0)
    store = MemoryStore()
    for i in range(n):
        store._records[str(i)] = Record(str(i), rng.standard_normal(dim).astype(np.float32), {})

    def make_candidate(idx: int) -> Candidate:
        return Candidate(str(idx), f"tt{1_000_000 + idx:07d}",
                         {"media_type": "movie", "imdb_id": f"tt{idx}", "poster": "/p.jpg",
                          "vote_average": float(rng.uniform(4, 9)),
                          "vote_count": int(rng.integers(50, 40000)),
                          "popularity": float(rng.uniform(1, 400)),
                          "year": int(rng.integers(1960, 2026)), "language": "en"},
                         store._records[str(idx)].vector)

    t0 = time.perf_counter()
    for _ in range(queries):
        q = normalize(rng.standard_normal(dim).astype(np.float32))
        sims = np.vstack([r.vector for r in store._records.values()]) @ q
        top = np.argpartition(sims, -200)[-200:]
    ann_ms = (time.perf_counter() - t0) * 1000 / queries

    t0 = time.perf_counter()
    for _ in range(queries):
        q = normalize(rng.standard_normal(dim).astype(np.float32))
        rank(q, [make_candidate(int(i)) for i in top], year_seed=2010, langs=("en",))
    rank_ms = (time.perf_counter() - t0) * 1000 / queries

    print(f"  vectors            : {n:,} x {dim}  = {n * dim * 4 / 1e6:.0f} MB float32 "
          f"({n * dim / 1e6:.0f} MB int8) — PLAN budget 27 MB int8")
    print(f"  ANN top-200 (numpy): {ann_ms:7.2f} ms/query   (Qdrant HNSW: ~1-3 ms)")
    print(f"  rank 200 candidates: {rank_ms:7.2f} ms/query")
    print(f"  total              : {ann_ms + rank_ms:7.2f} ms/query  — Stremio allows ~10 s")


def main() -> None:
    store = MemoryStore()
    load_records_from_json(store, catalog(dim=384))
    history = trakt_history()

    banner(f"fixture catalog: {len(store)} titles · vectors {len(store.get_all()[0].vector)}-d")
    state = build_user_state(store, history)
    taste = compute_taste(state.signals, state.dislikes)
    print(f"  watched={len(state.watched)} watchlist={len(state.watchlist)} hidden={len(state.hidden)}")
    print(f"  positive signals={sum(1 for s in state.signals if s.weight > 0)} "
          f"dislikes={len(state.dislikes)} ready={taste.ready}")
    for name, vec in (("taste_long", taste.long), ("taste_session", taste.session),
                      ("query", taste.query)):
        if vec is not None:
            print(f"  {name:14s}-> [{vec[0]:+.3f} {vec[1]:+.3f} {vec[2]:+.3f} ...]")

    show(store, history, "movie", "foryou")
    show(store, history, "movie", "because")

    banner("series rows — fixture has exactly 1 series and it is already watched")
    show(store, history, "series", "foryou")

    banner("Stremio meta previews (movie / foryou)")
    for meta in to_metas(build_row(store, history, "movie", "foryou", CONFIG).ranked)[:3]:
        print(f"  {meta}")

    banner("p99 catalog scale: 70k titles")
    scale_check()

    banner("cold start — no Trakt history")
    cold = build_row(store, {}, "movie", "foryou", CONFIG)
    print(f"  rows={len(cold.ranked)} reason={cold.diagnostics.get('reason')} "
          f"(§8.4: fall back to Trending — not built yet)")


if __name__ == "__main__":
    main()
