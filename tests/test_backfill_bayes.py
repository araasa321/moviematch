"""Fixes for two follow-ups found on the live 45k catalog.

1. `m` collapsed to M_MIN (20) for three of four strata.
   `bayes_stats` took median(vote_count) over EVERY member of a stratum, while
   `catalog_mean_vote` right above it filtered at >= 50. The catalog deliberately
   keeps a large zero/low-vote tail (the recent-release crawl ingests unreleased
   titles via the recency clause), so that tail dragged the median down. At m=20 a
   10-vote title keeps 33 % of its raw rating, i.e. the Bayesian smoothing that
   "kills 8.9 from 12 people" stops working.

2. `refresh()` only ever backfilled ~2 % of the catalog.
   It recomputes bayes for `active_sample` (25 discover pages by popularity,
   ~1,000 titles). Observed live: "vote/bayes refresh applied to 915 points" out
   of 45,177, so 98 % of titles kept their old single-mean bayes and the #4b
   change was effectively undeployed.
"""

from __future__ import annotations

import asyncio

import numpy as np

from jobs import policy
from jobs.backfill_bayes import backfill
from jobs.offline_run import FakeQdrant


# --------------------------------------------------------------- the m floor
def _catalog_with_zero_tail(n_real: int = 4000, n_zero: int = 6000) -> list[dict]:
    rows = [{"media_type": "movie", "language": "en", "vote_average": 6.8,
             "vote_count": 100 + (i % 5800)} for i in range(n_real)]
    rows += [{"media_type": "movie", "language": "en", "vote_average": 0.0,
              "vote_count": 0} for _ in range(n_zero)]
    return rows


def test_zero_vote_tail_no_longer_drags_m_to_the_floor():
    """The reported symptom: m == M_MIN for almost every stratum."""
    stats = policy.bayes_stats(_catalog_with_zero_tail())
    _c, m, _mg = stats[("movie", "en")]
    assert m > policy.M_MIN, f"m={m} still clamped to the floor"
    assert m > 100, f"m={m} is still far too small for a real catalog"


def test_m_median_ignores_titles_below_the_vote_floor():
    """m is now computed over the SAME population as C."""
    cat = _catalog_with_zero_tail()
    stats = policy.bayes_stats(cat)
    floor = policy.MIN_VOTES_FOR_MEAN["en"]
    expected_median = policy._median(
        [float(r["vote_count"]) for r in cat if r["vote_count"] >= floor])
    expected_m = min(policy.M_MAX,
                     max(policy.M_MIN, policy.PRIOR_FRACTION * expected_median))
    assert stats[("movie", "en")][1] == round(expected_m, 4)


def test_low_vote_titles_are_smoothed_again():
    """At the fixed m a 10-vote 9.0 must NOT keep a third of its raw rating."""
    stats = policy.bayes_stats(_catalog_with_zero_tail())
    c, m, _mg = stats[("movie", "en")]
    out = policy.apply_quality({"media_type": "movie", "language": "en",
                                "vote_average": 9.0, "vote_count": 10}, stats=stats)
    weight_on_raw = 10 / (10 + m)
    assert weight_on_raw < 0.10, f"raw rating still carries {weight_on_raw:.0%}"
    # And the result sits close to the stratum mean, not near 9.0.
    assert abs(out["bayes"] - c) < 0.5, out["bayes"]


def test_xx_floor_is_still_lower_than_en():
    """The fix must not re-impose the English vote bar on foreign titles."""
    assert policy.MIN_VOTES_FOR_MEAN["xx"] < policy.MIN_VOTES_FOR_MEAN["en"]
    rows = [{"media_type": "movie", "language": "ml", "vote_average": 8.5,
             "vote_count": 20}] * 500
    stats = policy.bayes_stats(rows)
    # 500 titles at 20 votes each clear the xx floor, so they define C.
    assert stats[("movie", "xx")][0] > 8.0, stats


# ------------------------------------------------------- full-catalog backfill
def _seeded(n: int = 40) -> FakeQdrant:
    q = FakeQdrant()
    for i in range(n):
        q.points[f"p{i}"] = {
            "id": f"p{i}", "vector": np.zeros(8, dtype=np.float32).tolist(),
            "payload": {"media_type": "movie", "language": "en",
                        "vote_average": 7.0 + (i % 20) / 10,
                        "vote_count": 100 + i * 37,
                        # deliberately stale values, as on the live catalog
                        "bayes": 1.0, "bayes_gems": 1.0,
                        "imdb_id": f"tt{i}", "poster": "/p.jpg"}}
    return q


def _run_backfill(q: FakeQdrant, **kw):
    import jobs.backfill_bayes as mod

    class _Shim:
        def __init__(self, *a, **k):
            pass

        def __getattr__(self, name):
            return getattr(q, name)

    orig = mod.QdrantClient
    mod.QdrantClient = _Shim
    try:
        return asyncio.run(mod.backfill(**kw))
    finally:
        mod.QdrantClient = orig


def test_backfill_updates_the_whole_catalog():
    """The 2 % problem: every point must be recomputed, not just a sample."""
    q = _seeded(40)
    stats = _run_backfill(q)
    assert stats["updated"] == 40, f"only {stats['updated']} of 40 updated"
    for pid, point in q.points.items():
        assert point["payload"]["bayes"] != 1.0, f"{pid} kept its stale bayes"


def test_backfill_preserves_other_payload_fields():
    """set_payload merges keys; nothing else may be touched."""
    q = _seeded(10)
    before = {pid: dict(p["payload"]) for pid, p in q.points.items()}
    _run_backfill(q)
    for pid, point in q.points.items():
        after = point["payload"]
        assert after["imdb_id"] == before[pid]["imdb_id"]
        assert after["poster"] == before[pid]["poster"]
        assert after["vote_count"] == before[pid]["vote_count"]
        assert after["language"] == before[pid]["language"]


def test_backfill_does_not_touch_vectors():
    q = _seeded(10)
    before = {pid: list(p["vector"]) for pid, p in q.points.items()}
    _run_backfill(q)
    for pid, point in q.points.items():
        assert point["vector"] == before[pid]


def test_backfill_dry_run_writes_nothing():
    q = _seeded(10)
    before = {pid: dict(p["payload"]) for pid, p in q.points.items()}
    stats = _run_backfill(q, dry_run=True)
    assert stats["updated"] == 10
    for pid, point in q.points.items():
        assert point["payload"] == before[pid], f"{pid} was modified by --dry-run"


def test_backfill_is_idempotent():
    q = _seeded(10)
    _run_backfill(q)
    first = {pid: dict(p["payload"]) for pid, p in q.points.items()}
    second = _run_backfill(q)
    assert second["updated"] == 0, "a second run rewrote already-correct points"
    assert second["skipped"] == 10
    for pid, point in q.points.items():
        assert point["payload"] == first[pid]


def test_backfill_uses_a_catalog_wide_prior():
    """The prior must come from all points, not from a popularity sample."""
    q = _seeded(40)
    _run_backfill(q)
    expected = policy.bayes_stats(
        [p["payload"] for p in _seeded(40).points.values()])
    for pid, point in q.points.items():
        payload = dict(point["payload"])
        policy.apply_quality(payload, stats=expected)
        assert point["payload"]["bayes"] == payload["bayes"], pid


def test_backfill_on_an_empty_catalog_is_a_noop():
    stats = _run_backfill(FakeQdrant())
    assert stats["scanned"] == 0 and stats["updated"] == 0
