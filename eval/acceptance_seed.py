"""§15 acceptance seed, for real — through the project's own ranker.

The whole point: does all-MiniLM-L6-v2 + the §9 ranker put The Dark Knight's
neighbourhood where a person would expect it?

  seed    : The Dark Knight (watched)
  expect  : Inception, The Prestige, The Dark Knight Rises, Batman Begins, Joker,
            Heat, The Batman  (§15 wants >= 3 of these in the top 15)
  guard   : collection_id caps the Batman trilogy to ONE item
  mode    : blockbuster weights vs gems weights

Vector space comes from the local numpy MiniLM (verified cos=0.999999 vs the
official ONNX graph). Payload vote counts are synthetic because Wikipedia does
not carry TMDB vote data; they are drawn per title so quality terms are exercised,
with real-world-ish values for the targets.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))            # eval/
sys.path.insert(0, str(Path(__file__).parent.parent))     # project root
import minilm  # noqa: E402

from addon.ids import point_id  # noqa: E402
from addon.memory_store import MemoryStore, load_records_from_json  # noqa: E402
from addon.rank import rank  # noqa: E402
from addon.rows import build_user_state, to_metas  # noqa: E402

DATA = Path(__file__).parent / "data"
TARGETS = ["Inception", "The Prestige", "The Dark Knight Rises", "Batman Begins",
           "Joker", "Heat", "The Batman"]
COLLECTION_BATMAN_NOLAN = 263

# rough TMDB-ish numbers for the known titles; everything else is drawn from a prior
KNOWN = {
    "The Dark Knight": (8.5, 32000, 120.0, 2008),
    "Batman Begins": (7.7, 21000, 90.0, 2005),
    "The Dark Knight Rises": (7.8, 18000, 100.0, 2012),
    "Joker": (8.2, 26000, 150.0, 2019),
    "The Batman": (7.7, 12000, 200.0, 2022),
    "Inception": (8.4, 36000, 160.0, 2010),
    "The Prestige": (8.2, 15000, 70.0, 2006),
    "Heat": (7.7, 7000, 30.0, 1995),
    "Memento": (8.2, 14000, 60.0, 2000),
    "Interstellar": (8.4, 33000, 140.0, 2014),
    "Tenet": (6.8, 12000, 110.0, 2020),
    "Oppenheimer": (8.0, 11000, 300.0, 2023),
}


def clean_name(name: str) -> str:
    return re.sub(r"\s*\([^)]*\)\s*$", "", name)


def build_store(vectors: np.ndarray, titles: list[dict]) -> tuple[MemoryStore, list[int]]:
    store = MemoryStore()
    rng = np.random.default_rng(7)
    rows, ids = [], []
    for i, (entry, vec) in enumerate(zip(titles, vectors)):
        name = clean_name(entry["name"])
        avg, votes, pop, year = KNOWN.get(
            name, (float(np.clip(rng.normal(6.8, 0.9), 3, 9)), int(rng.integers(200, 30000)),
                   float(rng.uniform(5, 250)), entry["year"]))
        coll = COLLECTION_BATMAN_NOLAN if name in (
            "Batman Begins", "The Dark Knight", "The Dark Knight Rises") else None
        payload = {
            "tmdb_id": 1000 + i, "media_type": "movie", "imdb_id": f"tt{1000000 + i:07d}",
            "title": name, "year": year, "end_year": None,
            "genres": [entry["franchise"]] if entry["franchise"] else [],
            "keywords": [], "director": entry["director"], "cast": entry["cast"],
            "language": "en", "runtime": 120, "vote_average": avg, "vote_count": votes,
            "popularity": pop, "collection_id": coll, "poster": f"/p{i}.jpg",
            "backdrop": f"/b{i}.jpg", "overview_short": entry["overview"][:200],
            "enriched": True,
        }
        rows.append({"point_id": point_id("movie", payload["tmdb_id"]),
                     "vector": vec.tolist(), "payload": payload})
        ids.append(i)
    load_records_from_json(store, rows)
    return store, ids


def main() -> None:
    titles = json.loads((DATA / "titles.json").read_text(encoding="utf-8"))
    vectors = np.load(DATA / "st_vectors_clean.npz")["vectors"]
    store, _ = build_store(vectors, titles)

    seed_tmdb = next(1000 + i for i, t in enumerate(titles)
                     if clean_name(t["name"]) == "The Dark Knight")
    history = {"watched": [{"movie": {"ids": {"tmdb": seed_tmdb}}, "plays": 1,
                            "watched_at": "2026-11-10T20:00:00+00:00"}]}
    state = build_user_state(store, history)
    seed_pid = state.last_watched["movie"]
    seed_vec = store.get(seed_pid).vector

    print(f"catalog: {len(store)} titles | seed: The Dark Knight "
          f"({seed_pid[:8]}...)\n")

    for mode in ("blockbuster", "gems"):
        pool = [rec.to_candidate() for rec in store.get_all("movie")
                if rec.point_id != seed_pid]
        ranked = rank(seed_vec, pool, mode=mode, year_seed=2008,
                      seed_director="Christopher Nolan",
                      seed_cast=store.get(seed_pid).payload["cast"],
                      langs=("en",), limit=15, year_now=2026)
        top15 = [c.payload["title"] for c in ranked]
        hits = [t for t in TARGETS if t in top15]
        print(f"=== {mode.upper()} — Because you watched The Dark Knight ===")
        for i, cand in enumerate(ranked, 1):
            p = cand.payload
            mark = "*" if p["title"] in TARGETS else " "
            print(f" {i:>2}{mark} {p['title'][:34]:<34} sim={cand.similarity:.3f} "
                  f"bayes={cand.extras.get('quality', 0):.2f} score={cand.score:.3f}")
        colls = [c.payload.get("collection_id") for c in ranked]
        print(f"\n  §15 hits: {len(hits)}/7 of {TARGETS[:3] if False else 'targets'} "
              f"-> {hits}   (acceptance: >= 3)")
        print(f"  batman-collection items in top15: {colls.count(COLLECTION_BATMAN_NOLAN)} "
              f"(acceptance: <= 1)")
        print(f"  min vote_count shown: {min(c.payload['vote_count'] for c in ranked)}")
        metas = to_metas(ranked, limit=3)
        print(f"  sample metas: {[m['name'] for m in metas]}\n")

    # what the raw ANN-only top-15 would have shown (no ranker) — the "before" picture
    pool = [rec.to_candidate() for rec in store.get_all("movie") if rec.point_id != seed_pid]
    for c in pool:
        c.similarity = float(c.vector @ seed_vec)
    raw = sorted(pool, key=lambda c: -c.similarity)[:15]
    print("=== RAW cosine top-15 (no ranker, for comparison) ===")
    print("   ", ", ".join(c.payload["title"] for c in raw))
    print(f"    §15 hits: {[t for t in TARGETS if t in [c.payload['title'] for c in raw]]}")


if __name__ == "__main__":
    main()
