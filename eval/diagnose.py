"""Are the (possibly off) MiniLM vectors wrong-uninformative, or just a bit different?

Compares three spaces on the same 97 titles:
    A. filename-token TF-IDF    (offline floor — pure lexical)
    B. local numpy MiniLM       (384-d, our engine)
    C. random 384-d gaussian    (noise floor)

For each seed we report the mean similarity of friends vs unrelated titles and a
separation ratio. If C separates roughly as well, something is badly wrong with
the encoder. If B is clearly above C, the geometry carries real signal even though
my reproduction may differ slightly from torch.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
DEFAULT_CATEGORIES = ("action", "animation", "comedy", "crime", "drama", "historical",
                      "horror", "scifi", "superhero", "war", "batman", "nolan", "tarantino")


def clean(titles: list[dict]) -> list[str]:
    """Strip the disambiguation suffix wikipedia uses: "Heat (1995 film)" -> "Heat"."""
    out = []
    for t in titles:
        name = re.sub(r"\s*\([^)]*\)\s*$", "", t["name"])
        year = t["year"]
        out.append(f"{name} ({year})")
    return out


def tfidf(names: list[str]) -> np.ndarray:
    tokens = [re.findall(r"[a-z0-9]+", n.lower()) for n in names]
    vocab = sorted({t for doc in tokens for t in doc})
    index = {t: i for i, t in enumerate(vocab)}
    docs = len(names)
    df = np.zeros(len(vocab))
    for doc in tokens:
        for t in set(doc):
            df[index[t]] += 1
    idf = np.log((1 + docs) / (1 + df)) + 1.0
    mat = np.zeros((docs, len(vocab)), dtype=np.float64)
    for row, doc in enumerate(tokens):
        for t in doc:
            mat[row, index[t]] += 1.0
    mat *= idf
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    return mat / np.maximum(norms, 1e-12)


def local_vectors(titles: list[dict]) -> np.ndarray:
    import minilm

    cache = Path(__file__).parent / "data" / "st_vectors_clean.npz"
    if cache.exists():
        return np.load(cache)["vectors"]
    model = minilm.MiniLM()
    texts = [minilm.recipe_text({**t, "name": n}) for t, n in zip(titles, clean(titles))]
    vecs = model.embed(texts)
    np.savez_compressed(cache, vectors=vecs,
                        names=[t["name"] for t in titles])
    return vecs


def separation(sims: np.ndarray, tags: list[str]) -> float:
    """Mean same-cluster similarity minus mean different-cluster similarity."""
    same, diff = [], []
    n = len(tags)
    for i in range(n):
        for j in range(i + 1, n):
            if tags[i] == tags[j] and tags[i] not in ("drama", ""):
                same.append(sims[i, j])
            elif tags[i] != tags[j]:
                diff.append(sims[i, j])
    if not same:
        return 0.0
    return float(np.mean(same) - np.mean(diff))


def main() -> None:
    titles = json.loads((Path(__file__).parent / "data" / "titles.json").read_text("utf-8"))
    names = clean(titles)
    tags = [t["franchise"] for t in titles]

    rng = np.random.default_rng(0)
    random_vecs = rng.standard_normal((len(titles), 384)).astype(np.float32)
    random_vecs /= np.linalg.norm(random_vecs, axis=1, keepdims=True)

    spaces = {
        "tf-idf (lexical floor)": tfidf(names),
        "MiniLM numpy (ours)": local_vectors(titles),
        "random noise (floor)": random_vecs,
    }
    print(f"{len(titles)} titles\n")
    print(f"{'space':<26} {'sep ratio':>10}  {'Dark Knight top-3':<52}")
    for label, vecs in spaces.items():
        sims = vecs @ vecs.T
        np.fill_diagonal(sims, -1)
        ratio = separation(sims, tags)
        idx = next(i for i, t in enumerate(titles) if t["name"] == "The Dark Knight")
        top = np.argsort(-sims[idx])[:3]
        top_names = ", ".join(f"{names[j].split(' (')[0]}({sims[idx, j]:.2f})" for j in top)
        print(f"{label:<26} {ratio:>10.3f}  {top_names:<52}")

    print("\nMiniLM same-cluster vs different-cluster detail:")
    vecs = spaces["MiniLM numpy (ours)"]
    sims = vecs @ vecs.T
    np.fill_diagonal(sims, np.nan)
    for tag in DEFAULT_CATEGORIES:
        rows = [i for i, t in enumerate(tags) if t == tag]
        cols = [j for j, t in enumerate(tags) if t != tag]
        if len(rows) < 2 or not cols:
            continue
        same = np.nanmean(sims[np.ix_(rows, rows)])
        diff = np.nanmean(sims[np.ix_(rows, cols)])
        print(f"  {tag:<12} n={len(rows):>2}  same={same:.3f}  other={diff:.3f}  Δ={same - diff:+.3f}")


if __name__ == "__main__":
    main()
