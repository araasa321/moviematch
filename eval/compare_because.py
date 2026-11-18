"""Does the FIXED `because` design actually escape the franchise shelf?

The earlier acceptance run showed raw single-seed ranking returned 8 Batman
titles ending at Madame Web/Morbius. This re-runs the same seed through the
*current* code path with the changes made after that finding:

  * wider candidate pool (600 instead of 200)
  * relaxed similarity floor (0.30)
  * query = normalize(0.7*seed + 0.3*taste)

Uses the real MiniLM vectors (numpy engine, ONNX-verified) over the 97-title
Wikipedia corpus, through addon/rank.py.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

EVAL = Path(__file__).parent
sys.path.insert(0, str(EVAL))
sys.path.insert(0, str(EVAL.parent))

from addon import rank as rank_mod  # noqa: E402
from addon.ids import point_id  # noqa: E402
from addon.memory_store import MemoryStore, load_records_from_json  # noqa: E402
from addon.rank import BECAUSE_ANN_LIMIT, rank  # noqa: E402
from addon.taste import Signal, compute_taste  # noqa: E402
from addon.rows import because_query  # noqa: E402
from datetime import datetime, timedelta, timezone  # noqa: E402

TARGETS = ["Inception", "The Prestige", "The Dark Knight Rises", "Batman Begins",
           "Joker", "Heat", "The Batman"]
NOLAN = 263


def clean(name: str) -> str:
    return re.sub(r"\s*\([^)]*\)\s*$", "", name)


def main() -> None:
    titles = json.loads((EVAL / "data" / "titles.json").read_text(encoding="utf-8"))
    vectors = np.load(EVAL / "data" / "st_vectors_clean.npz")["vectors"]

    rows = []
    for i, (entry, vec) in enumerate(zip(titles, vectors)):
        name = clean(entry["name"])
        rows.append({
            "point_id": point_id("movie", 1000 + i),
            "vector": vec.tolist(),
            "payload": {
                "tmdb_id": 1000 + i, "media_type": "movie", "imdb_id": f"tt{1000000 + i}",
                "title": name, "year": entry["year"], "language": "en",
                "vote_average": 7.5, "vote_count": 5000, "popularity": 50.0,
                "director": entry["director"], "cast": entry["cast"],
                "collection_id": NOLAN if name in ("Batman Begins", "The Dark Knight",
                                                   "The Dark Knight Rises") else None,
                "poster": "/p.jpg",
            },
        })
    store = MemoryStore()
    load_records_from_json(store, rows)

    seed_idx = next(i for i, t in enumerate(titles) if clean(t["name"]) == "The Dark Knight")
    seed = store.get(point_id("movie", 1000 + seed_idx))
    now = datetime.now(timezone.utc)

    # taste = two Nolan films the user rated 9 (so the blend has somewhere to go)
    taste_signals = []
    for name in ("Inception", "Memento"):
        idx = next(i for i, t in enumerate(titles) if clean(t["name"]) == name)
        taste_signals.append(Signal(store.get(point_id("movie", 1000 + idx)).vector, 1.5,
                                    now - timedelta(days=10), "rating"))
    taste = compute_taste(taste_signals, now=now)

    pool = [rec.to_candidate() for rec in store.get_all("movie") if rec.point_id != seed.point_id]
    seed_vec = seed.vector

    variants = {
        "OLD: seed only, 200 cap, floor .35": (seed_vec, rank_mod.ANN_LIMIT, rank_mod.SIM_FLOOR),
        "NEW: seed+taste, 600 cap, floor .30": (because_query(seed, taste),
                                                BECAUSE_ANN_LIMIT, rank_mod.BECAUSE_SIM_FLOOR),
    }
    for label, (query, ann, floor) in variants.items():
        import copy
        cands = copy.deepcopy(pool)
        ranked = rank(query, cands, year_seed=2008, seed_director="Christopher Nolan",
                      seed_cast=seed.payload["cast"], langs=("en",), limit=15,
                      ann_limit=ann, sim_floor=floor, year_now=2026)
        names = [c.payload["title"] for c in ranked]
        batman = sum(1 for c in ranked if c.payload.get("collection_id") == NOLAN
                     or c.payload["title"] in ("Batman", "Batman Returns", "The Batman",
                                               "Joker", "Joker: Folie à Deux"))
        hits = [t for t in TARGETS if t in names]
        print(f"\n=== {label} ===")
        for i, name in enumerate(names, 1):
            mark = "*" if name in TARGETS else " "
            print(f" {i:>2}{mark} {name}")
        print(f"  §15 hits {len(hits)}/7: {hits}")
        print(f"  batman-family titles: {batman}")

    print("\nNote: corpus is 97 titles, so 600 candidates = the whole catalog;")
    print("this shows direction of travel, not behaviour at 70k.")


if __name__ == "__main__":
    main()
