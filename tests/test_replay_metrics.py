"""The replay harness's metrics and history parsing.

The harness itself needs a live Trakt token + Qdrant, so what is testable offline
is the part that can be silently wrong: the metric maths and the positive-watch
extraction. A wrong NDCG would make every comparison meaningless while looking
perfectly plausible.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

import pytest

from eval.replay import hit_rate, ndcg, positive_watches, recall_at_k


# --------------------------------------------------------------- metrics
def test_recall_at_k():
    ranked = ["a", "b", "c"]
    assert recall_at_k(ranked, "a", 3) == 1.0
    assert recall_at_k(ranked, "c", 3) == 1.0
    assert recall_at_k(ranked, "c", 2) == 0.0
    assert recall_at_k(ranked, "zzz", 3) == 0.0


def test_hit_rate():
    assert hit_rate([True, False, True, True]) == 0.75
    assert hit_rate([]) == 0.0


def test_ndcg_single_relevant_item():
    """One relevant item: DCG = 1/log2(pos+1), IDCG = 1."""
    assert ndcg([1], 15) == pytest.approx(1.0)
    assert ndcg([2], 15) == pytest.approx(1 / math.log2(3))
    assert ndcg([15], 15) == pytest.approx(1 / math.log2(16))
    # Outside the cutoff contributes nothing.
    assert ndcg([16], 15) == 0.0
    # Absent entirely.
    assert ndcg([0], 15) == 0.0


def test_ndcg_penalises_being_lower():
    assert ndcg([1], 15) > ndcg([3], 15) > ndcg([10], 15)


# ------------------------------------------------------- history parsing
def _watch(tmdb: int, when: str, kind="movie") -> dict:
    return {kind: {"ids": {"tmdb": tmdb}, "title": f"T{tmdb}"}, "watched_at": when}


def test_positive_watches_are_chronological():
    hist = {"watched": [_watch(3, "2026-03-01T00:00:00Z"),
                        _watch(1, "2026-01-01T00:00:00Z"),
                        _watch(2, "2026-02-01T00:00:00Z")]}
    out = positive_watches(hist)
    assert [w["_key"][1] for w in out] == [1, 2, 3]


def test_positive_watches_drop_explicit_dislikes():
    """rating < 6 means the user rejected it: not a target to predict."""
    hist = {
        "watched": [_watch(1, "2026-01-01T00:00:00Z"),
                    _watch(2, "2026-01-02T00:00:00Z")],
        "ratings": [{"movie": {"ids": {"tmdb": 2}}, "rating": 3}],
    }
    out = positive_watches(hist)
    assert [w["_key"][1] for w in out] == [1]


def test_positive_watches_keep_high_ratings():
    hist = {
        "watched": [_watch(1, "2026-01-01T00:00:00Z")],
        "ratings": [{"movie": {"ids": {"tmdb": 1}}, "rating": 9}],
    }
    assert len(positive_watches(hist)) == 1


def test_positive_watches_dedupe_repeats():
    """A rewatch is one target, not two."""
    hist = {"watched": [_watch(1, "2026-01-01T00:00:00Z"),
                        _watch(1, "2026-02-01T00:00:00Z")]}
    assert len(positive_watches(hist)) == 1


def test_positive_watches_skip_items_without_a_tmdb_id():
    hist = {"watched": [{"movie": {"ids": {"imdb": "tt123"}},
                         "watched_at": "2026-01-01T00:00:00Z"}]}
    assert positive_watches(hist) == []


def test_shows_and_movies_both_count():
    hist = {"watched": [_watch(1, "2026-01-01T00:00:00Z", kind="movie"),
                        _watch(2, "2026-01-02T00:00:00Z", kind="show")]}
    out = positive_watches(hist)
    assert {w["_key"][0] for w in out} == {"movie", "series"}
