"""Ranker math (PLAN §9, §15) against hand-computed fixtures."""

import math

import numpy as np
import pytest

from addon.ids import id_from_trakt_item, point_id
from addon.rank import (
    Candidate,
    bayes_score,
    dislike_penalty,
    era_score,
    mmr,
    normalize,
    rank,
    reach_score,
)


# ------------------------------------------------------------------ quality
def test_bayes_matches_imdb_formula():
    # (v/(v+m))*R + (m/(v+m))*C with m=1000, C=6.4
    expected = (32000 / 33000) * 8.5 + (1000 / 33000) * 6.4
    assert bayes_score(8.5, 32000, "movie") == pytest.approx(expected, rel=1e-6)


def test_bayes_kills_low_vote_high_average():
    blockbuster = bayes_score(8.5, 32000, "movie")
    obscure = bayes_score(8.9, 12, "movie")
    assert blockbuster > obscure


def test_bayes_series_uses_smaller_prior():
    assert bayes_score(7.0, 300, "series") > bayes_score(7.0, 300, "movie")


def test_bayes_no_votes_returns_catalog_mean():
    assert bayes_score(0, 0, "movie") == pytest.approx(6.4)


def test_gems_mode_m_300():
    expected = (50 / 350) * 7.5 + (300 / 350) * 6.4
    assert bayes_score(7.5, 50, "movie", gems=True) == pytest.approx(expected)


# --------------------------------------------------------------------- reach
def test_reach_is_monotonic_and_bounded():
    low = reach_score(100, 5)
    high = reach_score(30000, 200)
    assert 0.0 <= low < high <= 1.0


# ----------------------------------------------------------------------- era
def test_era_decays_exponentially():
    assert era_score(2008, 2008) == pytest.approx(1.0)
    assert era_score(2023, 2008) == pytest.approx(math.exp(-1.0))
    assert era_score(None, 2008) == pytest.approx(0.5)


# ------------------------------------------------------------------ dislikes
def test_dislike_penalty_threshold():
    dislike = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    close = np.array([[0.9, math.sqrt(1 - 0.81), 0.0]], dtype=np.float32)   # cos = 0.9
    far = np.array([[0.5, math.sqrt(0.75), 0.0]], dtype=np.float32)         # cos = 0.5
    penalties = dislike_penalty(np.vstack([close, far]), dislike)
    assert penalties[0] == pytest.approx(0.30)
    assert penalties[1] == pytest.approx(0.0)


# ------------------------------------------------------------- guards / rank
def _cand(pid, vec, **payload):
    pid = str(pid)
    base = {"media_type": "movie", "imdb_id": f"tt{pid}", "vote_count": 5000,
            "vote_average": 8.0, "popularity": 50.0, "year": 2010, "language": "en",
            "poster": "/p.jpg"}
    base.update(payload)
    return Candidate(pid, base["imdb_id"], base, np.asarray(vec, dtype=np.float32))


def test_rank_drops_low_vote_below_guard_unless_recent():
    old = _cand(1, [1, 0, 0], vote_count=10, year=1999)
    recent = _cand(2, [1, 0, 0], vote_count=10, year=2025)
    out = rank([1, 0, 0], [old, recent], year_seed=2010, year_now=2025)
    ids = {c.point_id for c in out}
    assert "2" in ids and "1" not in ids


def test_rank_drops_missing_imdb_and_low_similarity():
    missing = _cand(1, [1, 0, 0], imdb_id="")
    orthogonal = _cand(2, [0, 1, 0])
    good = _cand(3, [1, 0, 0])
    out = rank([1, 0, 0], [missing, orthogonal, good])
    assert [c.point_id for c in out] == ["3"]


def test_rank_ignores_language_because_guards_filter_it():
    """Deliberate replacement for test_rank_language_penalty_orders_english_first.

    The old test called rank() directly with an unfiltered list — a shape
    production never produces, because `guard_reasons` drops every candidate
    outside the row's languages BEFORE ranking. So it was asserting behaviour that
    could not occur, and the 0.20 LANG_PENALTY it protected was unreachable (and
    larger than DIRECTOR_BONUS, i.e. a landmine if the guard were ever relaxed).

    rank() is now language-agnostic by design; the test that language is enforced
    in one place lives in test_language_guard.py.
    """
    en = _cand(1, [1, 0, 0], language="en")
    foreign = _cand(2, [0, 1, 0], language="ja")
    out = rank([1, 0, 0], [foreign, en], sim_floor=0.0)
    # Identical quality/reach/era and no language term: both survive, and neither
    # is pushed down by a penalty. (Orthogonal vectors keep MMR's near-duplicate
    # cutoff out of the picture.)
    assert {c.point_id for c in out} == {"1", "2"}


def test_rank_collection_caps_at_one_per_collection():
    a = _cand(1, [1, 0, 0], collection_id=263, popularity=100)
    b = _cand(2, [1, 0, 0], collection_id=263, popularity=90)
    c = _cand(3, [1, 0, 0], collection_id=None, popularity=10)
    out = rank([1, 0, 0], [a, b, c])
    assert sum(1 for cand in out if cand.payload.get("collection_id") == 263) == 1


def test_mmr_breaks_near_duplicates():
    far = [1, 0, 0, 0]
    near_dup = [1, 0.01, 0, 0]     # cos ~0.99995
    distinct = [0.8, 0.6, 0, 0]    # cos 0.8
    cands = [
        Candidate("a", "tt1", {}, np.array(far, np.float32), score=1.0),
        Candidate("b", "tt2", {}, np.array(near_dup, np.float32), score=0.99),
        Candidate("c", "tt3", {}, np.array(distinct, np.float32), score=0.5),
    ]
    out = [c.point_id for c in mmr(cands, limit=3)]
    assert out[0] == "a"
    assert "b" not in out          # > 0.90 similarity to an already-picked item
    assert "c" in out


def test_mmr_lambda_trades_score_for_diversity():
    """With a redundancy cutoff wide open, low lambda must pick the diverse item next."""
    v0 = np.array([1.0, 0.0, 0.0], np.float32)
    v_dup = np.array([0.9, 0.4359, 0.0], np.float32)     # cos ~0.9 to v0
    v_div = np.array([0.0, 1.0, 0.0], np.float32)        # cos 0 to v0
    v_dup /= np.linalg.norm(v_dup)
    cands = [Candidate("a", "tt1", {}, v0, score=1.0),
             Candidate("dup", "tt2", {}, v_dup, score=0.95),
             Candidate("div", "tt3", {}, v_div, score=0.80)]
    greedy = [c.point_id for c in mmr(cands, limit=2, lam=1.0, dup_sim=1.1)]
    diverse = [c.point_id for c in mmr(cands, limit=2, lam=0.1, dup_sim=1.1)]
    assert greedy == ["a", "dup"]
    assert diverse == ["a", "div"]


# ---------------------------------------------------------------- ids / parse
def test_point_id_is_uuid5_and_deterministic():
    import uuid

    assert point_id("movie", 155) == str(uuid.uuid5(uuid.NAMESPACE_URL, "movie:155"))
    assert point_id("movie", 155) == point_id("movie", 155)
    assert point_id("movie", 155) != point_id("series", 155)


def test_trakt_item_lookup():
    assert id_from_trakt_item({"movie": {"ids": {"tmdb": 155}}}, "movie") == ("movie", 155)
    assert id_from_trakt_item({"show": {"ids": {"tmdb": 1396}}}, "movie") == ("series", 1396)
    assert id_from_trakt_item({"movie": {"ids": {"trakt": 1}}}, "movie") is None


def test_normalize_handles_zero_vector():
    assert not np.any(normalize([0, 0, 0]))
    assert np.linalg.norm(normalize([3, 4, 0])) == pytest.approx(1.0)
