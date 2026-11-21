"""Ranking engine (PLAN §9) — pure functions over numpy arrays.

Nothing here touches the network or any database, so every stage of the funnel
can be unit-tested against hand-computed fixtures.

Funnel (§9.1):  query vector -> ANN top-200 -> score -> collection dedupe
                -> MMR diversity -> exploration slots -> top N
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Iterable, Sequence

import numpy as np

# --- constants (PLAN §20) ---------------------------------------------------
MODEL_DIM = 384
ANN_LIMIT = 200

BAYES_M = {"movie": 1000.0, "series": 300.0}   # §9.2
BAYES_M_GEMS = 300.0
CATALOG_MEAN_VOTE = 6.4                        # recomputed weekly in the nightly job

WEIGHTS = {                                    # §9.4 sim / quality / reach / era
    "blockbuster": (0.45, 0.25, 0.20, 0.10),
    "gems":        (0.60, 0.35, 0.00, 0.05),
}
DIRECTOR_BONUS = 0.15
CAST_BONUS = 0.05
CAST_BONUS_MAX = 2
DISLIKE_SIM = 0.8
DISLIKE_PENALTY = 0.30
LANG_PENALTY = 0.20

MMR_LAMBDA = 0.7
MMR_DUP_SIM = 0.90
EXPLORATION_SLOTS = 2
SIM_FLOOR = 0.35        # below: below relevance threshold, drop from the pool
MIN_VOTE_COUNT = 50     # §9.1 guard; waived for titles from the current/previous year

# `because` rows (seeded from ONE watched title) need a wider net than `foryou`.
# Measured on real MiniLM vectors: the raw nearest 15 of The Dark Knight were 8
# Batman titles ending at "Madame Web"/"Morbius". 200 candidates is not enough
# distance from the seed for the ranker to find anything off-franchise.
BECAUSE_ANN_LIMIT = 600
BECAUSE_SIM_FLOOR = 0.30
BECAUSE_SEED_WEIGHT = 0.7      # query = normalize(0.7*seed + 0.3*taste)
BECAUSE_TASTE_WEIGHT = 0.3

# normalisation anchors — refreshed from the nightly job (§20: norms from catalog p99)
VOTE_P99 = 20000.0
POPULARITY_P99 = 400.0


def normalize(vec: Sequence[float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    return arr if norm == 0.0 else arr / norm


def bayes_score(vote_average: float, vote_count: float, media_type: str = "movie",
                gems: bool = False, mean: float = CATALOG_MEAN_VOTE) -> float:
    """IMDb-style Bayesian rating (§9.2). Kills "8.9 from 12 people"."""
    m = BAYES_M_GEMS if gems else BAYES_M.get(media_type, 500.0)
    v = float(max(0.0, vote_count))
    r = float(vote_average) if vote_count else 0.0
    if v + m == 0.0:
        return mean
    return (v / (v + m)) * r + (m / (v + m)) * mean


def bayes_from_payload(payload: dict[str, Any], mode: str = "blockbuster") -> float:
    """Quality score: the ingest job's stored value when present, else recompute.

    The weekly refresh computes a stratified prior per (media_type, en|non-en) and
    stores the result in `bayes` / `bayes_gems` (#4b). Serving reads it instead of
    recomputing from module constants, which used to make the weekly `C` estimate
    dead on arrival: it was written to the payload and never read.

    The fallback keeps behaviour identical for rows ingested before #4b, or before
    the first post-#4b `refresh` backfill, so this is safe to deploy first.

    `bayes` is stored rounded to 3 dp at ingest, so a comparison against
    `bayes_score(...)` needs approx tolerance rather than equality.
    """
    key = "bayes_gems" if mode == "gems" else "bayes"
    value = payload.get(key)
    if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0:
        return float(value)
    return bayes_score(payload.get("vote_average", 0.0), payload.get("vote_count", 0),
                       payload.get("media_type", "movie"), gems=(mode == "gems"))


def reach_score(vote_count: float, popularity: float,
                vote_p99: float = VOTE_P99, pop_p99: float = POPULARITY_P99) -> float:
    """0.6*log1p(votes) + 0.4*log1p(popularity), each squashed against catalog p99."""
    def squash(value: float, p99: float) -> float:
        return min(1.0, math.log1p(max(0.0, value)) / math.log1p(max(1.0, p99)))

    return 0.6 * squash(vote_count, vote_p99) + 0.4 * squash(popularity, pop_p99)


def era_score(year: Any, year_seed: Any, half_life: float = 15.0) -> float:
    """exp(-|year - seed| / 15). Unknown year gets a neutral 0.5."""
    try:
        y = int(year)
        s = int(year_seed)
    except (TypeError, ValueError):
        return 0.5
    return math.exp(-abs(y - s) / half_life)


def dislike_penalty(vecs: np.ndarray, dislike_vecs: np.ndarray) -> np.ndarray:
    """Per-candidate cosine vs every dislike vector; penalty if the max exceeds 0.8."""
    if dislike_vecs.size == 0:
        return np.zeros(vecs.shape[0], dtype=np.float32)
    sims = vecs @ dislike_vecs.T
    return np.where(sims.max(axis=1) > DISLIKE_SIM, DISLIKE_PENALTY, 0.0).astype(np.float32)


@dataclass
class Candidate:
    """One catalog point as the ranker sees it."""
    point_id: str
    imdb_id: str
    payload: dict[str, Any]
    vector: np.ndarray
    similarity: float = 0.0
    score: float = 0.0
    extras: dict[str, float] = field(default_factory=dict)


def rank(
    query_vec: Sequence[float],
    candidates: Sequence[Candidate],
    *,
    mode: str = "blockbuster",
    year_seed: Any = None,
    seed_director: str = "",
    seed_cast: Iterable[str] = (),
    disliked: np.ndarray | None = None,
    langs: Sequence[str] = ("en",),
    year_now: int | None = None,
    exploration: Sequence[Candidate] | None = None,
    limit: int = 15,
    ann_limit: int = ANN_LIMIT,
    sim_floor: float = SIM_FLOOR,
) -> list[Candidate]:
    """Apply guards -> score -> collection dedupe -> MMR -> exploration -> top N.

    Args:
        ann_limit: how many candidates to consider before ranking. `because` rows
            pass a larger value: a single seed's nearest 200 are almost all
            franchise clones (measured — see FINDINGS-EMBEDDINGS.md), so the
            ranker needs a wider net to find anything off-franchise.
        sim_floor: relevance cutoff. `because` rows relax it slightly for the
            same reason.
    """
    w_sim, w_q, w_r, w_e = WEIGHTS.get(mode, WEIGHTS["blockbuster"])
    q = normalize(query_vec)
    seed_cast = {c for c in seed_cast if c}
    year_now = year_now if year_now is not None else date.today().year

    pool: list[Candidate] = []
    for cand in candidates:
        payload = cand.payload
        if not payload.get("imdb_id"):
            continue
        if cand.vector.shape[0] != q.shape[0]:
            continue
        if len(pool) >= ann_limit:
            break
        # §9.1 guard, kept here too so rank() is safe when called directly
        if int(payload.get("vote_count") or 0) < MIN_VOTE_COUNT:
            year = payload.get("year")
            if not isinstance(year, int) or year < year_now - 1:
                continue
        cand.similarity = float(np.dot(normalize(cand.vector), q))
        if cand.similarity < sim_floor:
            continue
        # §9.4 score terms
        q_score = bayes_from_payload(payload, mode)
        reach = reach_score(payload.get("vote_count", 0), payload.get("popularity", 0))
        era = era_score(payload.get("year"), year_seed)
        director = 1.0 if seed_director and payload.get("director") == seed_director else 0.0
        cast_overlap = len(seed_cast.intersection(payload.get("cast") or ()))
        lang_miss = 1.0 if payload.get("language") not in set(langs) else 0.0

        cand.extras = {"similarity": cand.similarity, "quality": q_score, "reach": reach}
        cand.score = (
            w_sim * cand.similarity
            + w_q * (q_score / 10.0)
            + w_r * reach
            + w_e * era
            + DIRECTOR_BONUS * director
            + CAST_BONUS * min(CAST_BONUS_MAX, cast_overlap)
            - LANG_PENALTY * lang_miss
        )
        pool.append(cand)

    if disliked is not None and pool:
        vecs = np.vstack([normalize(c.vector) for c in pool])
        for cand, pen in zip(pool, dislike_penalty(vecs, disliked)):
            cand.score -= float(pen)

    pool.sort(key=lambda c: c.score, reverse=True)

    # §9.5 collections: at most one title per TMDB collection (sequel flooding)
    seen_collections: set[str] = set()
    collected: list[Candidate] = []
    for cand in pool:
        cid = cand.payload.get("collection_id")
        if cid not in (None, 0, ""):
            key = str(cid)
            if key in seen_collections:
                continue
            seen_collections.add(key)
        collected.append(cand)

    picked = mmr(collected, limit=limit)

    # §9.5 anti-bubble: swap the two weakest slots for trending/new items not already picked
    if exploration:
        chosen = {c.point_id for c in picked}
        free = [c for c in exploration if c.point_id not in chosen][:EXPLORATION_SLOTS]
        if free and len(picked) >= len(free):
            picked = picked[: len(picked) - len(free)] + free
    return picked[:limit]


def mmr(candidates: Sequence[Candidate], limit: int = 15,
        lam: float = MMR_LAMBDA, dup_sim: float = MMR_DUP_SIM) -> list[Candidate]:
    """Greedy MMR over already-scored candidates with a hard near-duplicate cutoff."""
    if not candidates:
        return []
    vecs = np.vstack([normalize(c.vector) for c in candidates])
    scores = np.array([c.score for c in candidates], dtype=np.float32)
    sims = vecs @ vecs.T

    target = min(limit, len(candidates))
    chosen: list[int] = [int(np.argmax(scores))]
    while len(chosen) < target:
        best_idx, best_val = -1, -math.inf
        for idx in range(len(candidates)):
            if idx in chosen:
                continue
            redundancy = float(sims[idx, chosen].max())
            if redundancy > dup_sim:
                continue
            val = lam * float(scores[idx]) - (1.0 - lam) * redundancy
            if val > best_val:
                best_idx, best_val = idx, val
        if best_idx < 0:          # everything left is a near-duplicate
            break
        chosen.append(best_idx)
    return [candidates[i] for i in chosen]
