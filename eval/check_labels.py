"""Cosine-distance based qualitative check: do neighbours share a genre/theme?

Independent of *which* encoder is right, the ranked list is what the user sees.
We label each title with a coarse genre and check whether nearest neighbours
share a label more often than a random baseline.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

# coarse labels keyed by the plan's own franchise tag; empty tag -> "misc"
GENRE_BY_FRANCHISE = {
    "batman": "crime", "crime": "crime", "heist": "crime", "neonoir": "crime",
    "nolan": "scifi", "scifi": "scifi",
    "superhero": "action", "action": "action",
    "horror": "horror",
    "animation": "animation",
    "drama": "drama", "historical": "drama", "war": "drama", "comedy": "comedy",
}


def main() -> None:
    titles = json.loads((Path(__file__).parent / "data" / "titles.json").read_text("utf-8"))
    vecs = np.load(Path(__file__).parent / "data" / "st_vectors_clean.npz")["vectors"]
    labels = [GENRE_BY_FRANCHISE.get(t["franchise"], "misc") for t in titles]

    sims = vecs @ vecs.T
    np.fill_diagonal(sims, -1)
    k = 10
    rng = random.Random(0)

    hits_ann = hits_rand = 0
    total = 0
    for i in range(len(titles)):
        neighbors = np.argsort(-sims[i])[:k]
        random_neighbors = rng.sample([j for j in range(len(titles)) if j != i], k)
        hits_ann += sum(labels[j] == labels[i] for j in neighbors)
        hits_rand += sum(labels[j] == labels[i] for j in random_neighbors)
        total += k

    base_rate = np.mean([labels.count(x) - 1 for x in labels]) / (len(labels) - 1)
    print(f"genre match @10  neighbours : {hits_ann / total:.3f}")
    print(f"genre match @10  random     : {hits_rand / total:.3f}   (≈ baseline {base_rate:.3f})")
    print(f"lift over random            : {(hits_ann / max(hits_rand, 1)):.2f}x")
    print(f"excess over baseline        : {hits_ann / total - base_rate:+.3f}")

    print("\nDetailed look at a few probes (genre-sorted neighbours):")
    for probe in ("Inception (2010)", "Heat (1995)", "The Dark Knight (2008)"):
        names = [t["name"] for t in titles]
        i = names.index(probe)
        print(f"\n  {probe}  [{labels[i]}]")
        for j in np.argsort(-sims[i])[:8]:
            mark = " " if labels[j] == labels[i] else "x"
            print(f"   {mark} {names[j]:<40} {sims[i, j]:.3f}  [{labels[j]}]")


if __name__ == "__main__":
    main()
