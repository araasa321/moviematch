"""Multi-intent retrieval: taste clustering + best-intent scoring.

Motivation, measured on a real 45k catalog with real Trakt history:

    Recall@200 = 10 %   (only 3 of 30 next-watches were even in the candidate pool)
    HitRate@15 = 3.3 %
    median top-15 vote_count = 10,104

A single averaged taste centroid returns the neighbours of "the average of
everything you watch", which is a generic mainstream film. Re-ranking cannot
recover a title that was never retrieved, which is why three rounds of weight and
bayes tuning produced no visible change.

These tests pin the two halves of the fix:
  * taste_queries() splits a signal set into distinct interests;
  * rank() scores each candidate against its BEST-matching intent.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from addon.rank import Candidate, rank
from addon.taste import (CLUSTER_MAX, CLUSTER_MIN_SIGNALS, Signal, compute_taste,
                         taste_queries)

DIM = 16
NOW = datetime(2026, 11, 22, tzinfo=timezone.utc)


def _unit(*idx: int) -> np.ndarray:
    v = np.zeros(DIM, dtype=np.float32)
    for i in idx:
        v[i] = 1.0
    return v


def _signals(groups: dict[tuple[int, ...], int]) -> list[Signal]:
    """`{(vector_indices): count}` -> one Signal per item."""
    out: list[Signal] = []
    for idx, n in groups.items():
        for _ in range(n):
            out.append(Signal(_unit(*idx), 0.7, NOW - timedelta(days=10), "watched"))
    return out


# ------------------------------------------------------------ clustering
def test_too_little_history_returns_one_query():
    """Below CLUSTER_MIN_SIGNALS the single centroid is all the evidence there is."""
    signals = _signals({(0,): 3, (1,): 2})
    assert len(signals) < CLUSTER_MIN_SIGNALS
    taste = compute_taste(signals, now=NOW, dim=DIM)
    assert len(taste_queries(signals, taste)) == 1


def test_two_distinct_tastes_produce_two_queries():
    """The core claim: sci-fi viewers who ALSO watch drama get both intents."""
    signals = _signals({(0,): 20, (7,): 20})
    taste = compute_taste(signals, now=NOW, dim=DIM)
    queries = taste_queries(signals, taste)
    assert len(queries) >= 2, f"expected 2 intents, got {len(queries)}"


def test_each_cluster_points_at_a_real_interest():
    """At least one query per watched direction must exist.

    The blended taste query is deliberately also returned (it lands BETWEEN
    clusters, at cos ~0.71 here), so the assertion is that both directions are
    covered — not that every query is a pure direction.
    """
    signals = _signals({(0,): 20, (7,): 20})
    taste = compute_taste(signals, now=NOW, dim=DIM)
    queries = taste_queries(signals, taste)
    for intent in (_unit(0), _unit(7)):
        best = max(float(q @ intent) for q in queries)
        assert best > 0.9, f"interest {intent.argmax()} is not covered (best {best:.2f})"


def test_one_taste_does_not_fragment():
    """A single interest must not be split into duplicate centroids."""
    signals = _signals({(0,): 40})
    taste = compute_taste(signals, now=NOW, dim=DIM)
    queries = taste_queries(signals, taste)
    assert len(queries) == 1, f"one taste became {len(queries)} queries"


def test_query_count_is_capped():
    signals = _signals({(0,): 15, (1,): 15, (2,): 15, (3,): 15, (4,): 15, (5,): 15})
    taste = compute_taste(signals, now=NOW, dim=DIM)
    queries = taste_queries(signals, taste)
    assert len(queries) <= CLUSTER_MAX + 1


def test_clustering_is_deterministic():
    """Same history must give the same row twice: no random init."""
    signals = _signals({(0,): 20, (7,): 20, (3,): 20})
    taste = compute_taste(signals, now=NOW, dim=DIM)
    a = taste_queries(signals, taste)
    b = taste_queries(signals, taste)
    assert len(a) == len(b)
    for x, y in zip(a, b):
        assert np.allclose(x, y)


def test_cold_start_returns_empty_or_single():
    taste = compute_taste([], now=NOW, dim=DIM)
    assert taste_queries([], taste) == []


# ------------------------------------------------- best-intent scoring in rank()
def _cand(pid: str, vec: np.ndarray, **payload) -> Candidate:
    base = {"media_type": "movie", "imdb_id": f"tt{pid}", "vote_count": 5000,
            "vote_average": 7.5, "popularity": 50.0, "year": 2010,
            "language": "en", "poster": "/p.jpg", "director": "", "cast": []}
    base.update(payload)
    return Candidate(pid, base["imdb_id"], base, np.asarray(vec, dtype=np.float32))


def test_candidate_is_scored_against_its_best_intent():
    """A drama must score high on the drama intent, not be judged by sci-fi."""
    sci = _unit(0)
    drama = _unit(7)
    drama_cand = _cand("d", _unit(7))
    rank([sci, drama], [drama_cand], sim_floor=0.0)
    assert drama_cand.similarity == pytest.approx(1.0, abs=1e-5), (
        "candidate was scored against the wrong intent")


def test_best_intent_uses_max_not_mean():
    """max over intents, so one strong match is enough."""
    both = _cand("x", _unit(0, 7))
    rank([_unit(0), _unit(7)], [both], sim_floor=0.0)
    # cos to either unit is 1/sqrt(2); the mean-of-queries answer would be lower.
    assert both.similarity == pytest.approx(1 / np.sqrt(2), abs=1e-5)


def test_single_vector_path_is_unchanged():
    """Passing one vector must behave exactly as before."""
    cand = _cand("a", _unit(0))
    rank(_unit(0), [cand], sim_floor=0.0)
    assert cand.similarity == pytest.approx(1.0, abs=1e-5)


def test_bare_list_of_floats_is_one_vector_not_many():
    """`[1, 0, 0]` is a single 3-d query. Regression for a shape-detection bug."""
    cand = Candidate("a", "tta",
                     {"media_type": "movie", "imdb_id": "tta", "vote_count": 5000,
                      "vote_average": 7.5, "popularity": 50.0, "year": 2010,
                      "language": "en"},
                     np.array([1.0, 0.0, 0.0], dtype=np.float32))
    rank([1.0, 0.0, 0.0], [cand], sim_floor=0.0)
    assert cand.similarity == pytest.approx(1.0, abs=1e-5)


def test_stacked_array_is_many_vectors():
    cand = _cand("a", _unit(7))
    queries = np.vstack([_unit(0), _unit(7)])
    rank(queries, [cand], sim_floor=0.0)
    assert cand.similarity == pytest.approx(1.0, abs=1e-5)


def test_multi_intent_lets_a_second_interest_through_the_floor():
    """The practical payoff: a drama survives a SIM_FLOOR that sci-fi alone fails."""
    drama_cand = _cand("d", _unit(7))
    # With only the sci-fi intent this candidate is orthogonal -> filtered out.
    assert rank([_unit(0)], [drama_cand], sim_floor=0.35) == []
    # With the drama intent included it survives.
    assert [c.point_id for c in rank([_unit(0), _unit(7)], [drama_cand],
                                     sim_floor=0.35)] == ["d"]
