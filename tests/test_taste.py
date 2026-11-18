"""Taste engine (PLAN §8, §15): weight mapping, decay, centroid, session blend."""

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from addon.taste import (
    REWATCH,
    WATCHED,
    WATCHLIST,
    Signal,
    compute_taste,
    decay,
    rating_weight,
)

NOW = datetime(2026, 11, 18, tzinfo=timezone.utc)
DIM = 4


def vec(*values):
    out = np.zeros(DIM, dtype=np.float32)
    out[: len(values)] = values
    return out


def sig(vector, weight, kind="watched", age_days=0.0):
    return Signal(vector, weight, NOW - timedelta(days=age_days), kind)


# ------------------------------------------------------------------- weights
@pytest.mark.parametrize("rating,expected", [(10, 2.0), (8, 1.0), (6, 0.0), (4, -1.0), (2, -2.0)])
def test_rating_weight_matches_plan_table(rating, expected):
    assert rating_weight(rating) == pytest.approx(expected)


def test_weight_constants_match_plan():
    assert (WATCHED, WATCHLIST, REWATCH) == (0.7, 0.8, 2.5)


# --------------------------------------------------------------------- decay
def test_decay_halves_at_half_life():
    assert decay(0) == pytest.approx(1.0)
    assert decay(365) == pytest.approx(0.5)
    assert decay(730) == pytest.approx(0.25)


# ------------------------------------------------------------------ centroid
def test_single_signal_gives_unit_taste_vector():
    taste = compute_taste([sig(vec(1, 0, 0, 0), 1.0, age_days=0)], now=NOW, dim=DIM)
    assert taste.long is not None
    assert taste.long[0] == pytest.approx(1.0)
    assert taste.session is not None
    # long and session agree, so the blend is the same direction
    assert taste.query[0] == pytest.approx(1.0)


def test_two_signal_centroid_is_weighted_mean():
    a = sig(vec(1, 0, 0, 0), 2.0, age_days=0)     # stable & recent weight 2
    b = sig(vec(0, 1, 0, 0), 1.0, age_days=0)     # weight 1
    taste = compute_taste([a, b], now=NOW, dim=DIM)
    # stable = normalize(2a + 1b) -> [2,1,0,0]/sqrt(5)
    expected = np.array([2, 1, 0, 0]) / np.sqrt(5)
    assert np.allclose(taste.long, expected, atol=1e-6)


def test_recent_decay_favours_new_signals_in_long_vector():
    old = sig(vec(1, 0, 0, 0), 1.0, age_days=365)   # stable decay .5, recent decay .5**(365/60) ~0
    new = sig(vec(0, 1, 0, 0), 1.0, age_days=0)
    taste = compute_taste([old, new], now=NOW, dim=DIM)
    # recent is dominated by `new`, so taste_long must lean towards e_y
    assert taste.long[1] > taste.long[0]


def test_session_blend_uses_last_actions():
    long_days_ago = [sig(vec(1, 0, 0, 0), 2.0, age_days=200)]
    fresh = [sig(vec(0, 0, 1, 0), 2.0, age_days=0)]        # inside 72h window
    taste = compute_taste(long_days_ago + fresh, now=NOW, dim=DIM)
    assert taste.session is not None and taste.session[2] == pytest.approx(1.0)
    # query = normalize(0.7*long + 0.3*session) -> both components present
    assert taste.query[0] > 0 and taste.query[2] > 0


def test_actions_older_than_72h_do_not_form_session():
    taste = compute_taste([sig(vec(1, 0, 0, 0), 1.0, age_days=10)], now=NOW, dim=DIM)
    assert taste.session is None
    assert np.allclose(taste.query, taste.long)


def test_negative_ratings_never_enter_stable_centroid():
    pos = sig(vec(1, 0, 0, 0), 1.0, kind="rating", age_days=0)
    neg = sig(vec(0, 1, 0, 0), -2.0, kind="rating", age_days=0)
    taste = compute_taste([pos, neg], dislike_signals=[neg], now=NOW, dim=DIM)
    # stable centroid is driven by the positive only...
    assert np.allclose(taste.long, [1, 0, 0, 0], atol=1e-6)
    # ...while the dislike list is exposed for the rank penalty
    assert taste.dislikes is not None and taste.dislikes.shape[0] == 1


def test_ready_flag_needs_three_positive_signals():
    one = compute_taste([sig(vec(1, 0, 0, 0), 1, age_days=0)], now=NOW, dim=DIM)
    three = compute_taste([sig(vec(1, 0, 0, 0), 1, age_days=i) for i in range(3)], now=NOW, dim=DIM)
    assert one.ready is False
    assert three.ready is True
    assert three.positive_count == 3


def test_empty_input_is_not_ready():
    taste = compute_taste([], now=NOW, dim=DIM)
    assert taste.ready is False and taste.query is None


def test_unknown_zero_vectors_are_ignored():
    taste = compute_taste([sig(np.zeros(DIM), 1.0, age_days=0)], now=NOW, dim=DIM)
    assert taste.long is None
