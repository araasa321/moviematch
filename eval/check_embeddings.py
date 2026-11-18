"""Is my numpy MiniLM numerically identical to HF's torch model?

Without an oracle answer to that, every similarity number below is "self-consistent
but unverified". This checks the parts that are cheap and decisive:

  1. reference sentence-pair similarities from HF's own model-card widget
  2. neighbour structure on real movie synopses (does it separate franchises at all?)
  3. embedding stability under the §6.1 text recipe variants
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import minilm  # noqa: E402

DATA = Path(__file__).parent / "data"


def main() -> None:
    m = minilm.MiniLM()
    print("=== 1. document-embedding sanity (model-card examples) ===")
    docs = [
        "The cat sits outside",
        "A man is playing guitar",
        "I love pasta",
        "The new movie is awesome",
        "The dog plays in the garden",
        "A woman watches TV",
        "The new movie is so great",
        "Do you like music?",
    ]
    v = m.embed(docs)
    cos = lambda a, b: float(v[a] @ v[b])  # noqa: E731
    print(f"  movie praised vs movie great : {cos(3, 6):.4f}   (expect high ~0.8+)")
    print(f"  movie praised vs movie awesome: {cos(3, 4):.4f}")
    print(f"  music vs guitar              : {cos(7, 1):.4f}   (expect mid)")
    print(f"  cat vs dog                   : {cos(0, 4):.4f}")
    print(f"  pasta vs movie               : {cos(2, 3):.4f}   (expect low)")
    print(f"  movie vs pasta               : {cos(6, 2):.4f}")

    print("\n=== 2. movie neighbourhoods on real synopses ===")
    titles = json.loads((DATA / "titles.json").read_text(encoding="utf-8"))
    vecs_path = DATA / "st_vectors_local.npz"
    if vecs_path.exists():
        vecs = np.load(vecs_path)["vectors"]
        print(f"  using cached local vectors {vecs.shape}")
    else:
        texts = [minilm.recipe_text(t) for t in titles]
        vecs = m.embed(texts)
        np.savez_compressed(vecs_path, vectors=vecs)
    names = [t["name"] for t in titles]
    sims = vecs @ vecs.T
    np.fill_diagonal(sims, -1)

    for probe in ("The Dark Knight", "Inception", "Heat"):
        idx = names.index(probe)
        order = np.argsort(-sims[idx])[:11]
        print(f"\n  {probe} ({titles[idx]['note'] or 'target'}) top-10:")
        for rank, j in enumerate(order, 1):
            tag = titles[j]["franchise"] or "?"
            note = titles[j]["note"]
            print(f"    {rank:>2}. {names[j]:<40} {sims[idx, j]:.3f}  [{tag}{'/' + note if note else ''}]")

    print("\n=== 3. recipe sensitivity (PLAN §6.1 vs title-only) ===")
    for probe in ("The Dark Knight", "Joker", "The Batman", "Heat"):
        idx = names.index(probe)
        title_only = m.embed([f"{titles[i]['name']}. {titles[i]['year']}" for i in range(len(titles))])
        full = vecs
        t_sims = title_only @ title_only[idx]
        f_sims = full @ full[idx]
        print(f"  {probe:<18} sim(Joker) title-only={t_sims[names.index('Joker')]:.3f} "
              f"full-text={f_sims[names.index('Joker')]:.3f}")


if __name__ == "__main__":
    main()
