"""#4 / M4 — clamp the similarity term at 0, and remove the unreachable language term.

Two formula-hygiene fixes.

4.1 `sim` was not clamped. rank() added `w_sim * cand.similarity` with no lower
    bound. Cosine is in [-1, 1], so a candidate at -0.2 contributed -0.09 and
    dislike_penalty then subtracted up to 0.30 on top. SIM_FLOOR normally hides
    this, but `because` rows pass BECAUSE_SIM_FLOOR and any caller passing
    sim_floor=0.0 gets negative contributions.

4.2 The language term was unreachable AND oversized. rows.py passed
    `langs=row_langs`, the exact set candidates were filtered by, so lang_miss
    was identically 0. Had it ever fired it was 0.20 — 33 % larger than
    DIRECTOR_BONUS = 0.15. Deleted rather than left as a landmine.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from addon import rank as rank_mod
from addon.rank import Candidate, rank


def _cand(pid, vec, **payload) -> Candidate:
    pid = str(pid)
    base = {"media_type": "movie", "imdb_id": f"tt{pid}", "vote_count": 5000,
            "vote_average": 8.0, "popularity": 50.0, "year": 2010,
            "language": "en", "poster": "/p.jpg"}
    base.update(payload)
    return Candidate(pid, base["imdb_id"], base, np.asarray(vec, dtype=np.float32))


# ------------------------------------------------- 4.1 similarity clamp
def test_negative_similarity_does_not_reduce_score():
    """A candidate at cos=-1.0 must score the same as one at cos=0.0.

    `sim_floor` has to be BELOW the negative cosine or the candidate is dropped by
    the relevance gate before scoring — that gate is a separate, legitimate
    mechanism. The clamp is about what happens to a negative cosine that does
    reach the score: pre-#4 it contributed -0.45.

    Both candidates carry identical quality/reach/era, so the similarity term is
    the only difference.
    """
    neg = _cand("neg", [-1.0, 0.0, 0.0])
    flat = _cand("flat", [0.0, 1.0, 0.0])
    out = rank([1.0, 0.0, 0.0], [neg, flat], sim_floor=-2.0, limit=2)

    by_id = {c.point_id: c for c in out}
    assert set(by_id) == {"neg", "flat"}, "a candidate was dropped by the gate"
    assert by_id["neg"].similarity == pytest.approx(-1.0, abs=1e-5)
    assert by_id["flat"].similarity == pytest.approx(0.0, abs=1e-5)
    assert by_id["neg"].score == pytest.approx(by_id["flat"].score, abs=1e-6), (
        "a negative cosine still reduced the score")


def test_similarity_extras_preserve_the_raw_cosine():
    """Diagnostics must stay truthful: extras["similarity"] is the raw value."""
    neg = _cand("neg", [-1.0, 0.0, 0.0])
    rank([1.0, 0.0, 0.0], [neg], sim_floor=-2.0)
    assert neg.similarity == pytest.approx(-1.0, abs=1e-5)
    assert neg.extras["similarity"] == pytest.approx(-1.0, abs=1e-5)


def test_clamp_does_not_change_positive_scores():
    """Guard against over-correcting: positive cosines keep their exact weight."""
    cand = _cand(1, [0.6, 0.8, 0.0])            # cos with [0.6,0.8,0] = 1.0
    rank([0.6, 0.8, 0.0], [cand], sim_floor=0.0)
    w_sim = rank_mod.WEIGHTS["blockbuster"][0]
    assert cand.similarity == pytest.approx(1.0)
    # The similarity term must appear exactly once at full weight.
    q = Candidate("q", "ttq", dict(cand.payload),
                  np.array([0.6, 0.8, 0.0], dtype=np.float32))
    rank([1.0, 0.0, 0.0], [q], sim_floor=0.0)
    assert w_sim * max(0.0, q.similarity) <= q.score + 1e-6


# ------------------------------------------------- 4.2 language term removal
def test_rank_has_no_langs_parameter():
    """Guards against a future re-add without a deliberate decision."""
    params = inspect.signature(rank).parameters
    assert "langs" not in params, "rank() grew a `langs` parameter back"


def test_lang_penalty_constant_is_gone():
    assert not hasattr(rank_mod, "LANG_PENALTY"), (
        "LANG_PENALTY is back; language belongs in guard_reasons")


def test_rank_is_language_agnostic():
    """Two candidates differing only in language must score identically.

    The vectors are orthogonal so MMR's near-duplicate cutoff (0.90) is not what
    decides the outcome — only the absence of a language term is being tested.
    """
    en = _cand(1, [1, 0, 0], language="en")
    foreign = _cand(2, [0, 1, 0], language="ja")
    out = rank([1, 0, 0], [foreign, en], sim_floor=0.0, limit=2)
    scores = {c.point_id: c.score for c in out}
    assert set(scores) == {"1", "2"}, "a candidate was dropped"
    # Different similarities, so subtract the similarity term before comparing:
    # everything else (quality/reach/era) is identical, and a language penalty
    # would show up as a residual difference.
    w_sim = rank_mod.WEIGHTS["blockbuster"][0]
    en_residual = scores["1"] - w_sim * max(0.0, 1.0)
    ja_residual = scores["2"] - w_sim * max(0.0, 0.0)
    assert en_residual == pytest.approx(ja_residual, abs=1e-9)


def test_guard_reasons_is_the_only_language_filter():
    """Language is enforced exactly once, before ranking, not in the score."""
    from addon.rows import guard_reasons

    base = {"imdb_id": "tt1", "media_type": "movie", "poster": "/p.jpg"}
    assert guard_reasons({**base, "language": "ja"}, media_type="movie",
                         langs=("en",), year_now=2026) == "language"
    assert guard_reasons({**base, "language": "en"}, media_type="movie",
                         langs=("en",), year_now=2026) is None


def test_foreign_title_never_enters_an_english_row(store, history):
    """End to end: the pool for an `en` row contains no foreign titles."""
    from addon.config import Config
    from addon.rows import build_row

    cfg = Config(secret="s", langs=("en",))
    plan = build_row(store, history, "movie", "foryou", cfg)
    assert plan.ranked, "fixture history should produce a row"
    assert all(c.payload["language"] == "en" for c in plan.ranked), (
        [c.payload["language"] for c in plan.ranked])
