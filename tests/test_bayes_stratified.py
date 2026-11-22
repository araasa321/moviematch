"""#4b / M3 — stratified bayes, computed at ingest and served from the payload.

Two bugs, one fix:

  1. Drift / dead weekly computation. `rank.py` called `bayes_score(...)` with
     module constants and IGNORED the `bayes` / `bayes_gems` fields the weekly
     refresh wrote, so the recomputed `C` never reached serving.
  2. Mixed-population `C`. `refresh()` accumulated `active_sample` across BOTH
     movie and series and took one mean. Movies and series have different vote
     distributions and different m; one C cannot be right for both.

The fix computes a stratified (C, m, m_gems) per (media_type, "en"|"xx") ONCE in
jobs/policy.py and has rank.py read the stored value, falling back to the old
formula only when it is absent.
"""

from __future__ import annotations

import pytest

from addon.rank import bayes_from_payload, bayes_score
from jobs import policy


# ------------------------------------------------------------- rank reads payload
def test_bayes_from_payload_prefers_the_stored_value():
    """A stored bayes of 7.4 must beat what the old formula would compute."""
    payload = {"bayes": 7.4, "vote_average": 9.9, "vote_count": 3,
               "media_type": "movie"}
    recomputed = bayes_score(9.9, 3, "movie")
    assert recomputed != pytest.approx(7.4)
    assert bayes_from_payload(payload, "blockbuster") == pytest.approx(7.4)


def test_bayes_from_payload_falls_back_when_absent():
    """Pre-#4b rows keep today's behaviour exactly."""
    payload = {"vote_average": 8.4, "vote_count": 60, "media_type": "movie"}
    assert bayes_from_payload(payload, "blockbuster") == pytest.approx(
        bayes_score(8.4, 60, "movie"))


def test_bayes_from_payload_uses_the_gems_field_in_gems_mode():
    payload = {"bayes": 7.0, "bayes_gems": 6.5, "vote_average": 8.0,
               "vote_count": 100, "media_type": "movie"}
    assert bayes_from_payload(payload, "gems") == pytest.approx(6.5)
    assert bayes_from_payload(payload, "blockbuster") == pytest.approx(7.0)


def test_bayes_from_payload_ignores_a_zero_or_negative_value():
    """0 / negative means "not computed", not "quality 0"."""
    payload = {"bayes": 0.0, "vote_average": 8.0, "vote_count": 1000,
               "media_type": "movie"}
    assert bayes_from_payload(payload) == pytest.approx(bayes_score(8.0, 1000, "movie"))
    payload = {"bayes": -1.0, "vote_average": 8.0, "vote_count": 1000,
               "media_type": "movie"}
    assert bayes_from_payload(payload) == pytest.approx(bayes_score(8.0, 1000, "movie"))


def test_bayes_from_payload_ignores_a_bool():
    """JSON round-trips sometimes turn numbers into bools; do not treat True as 1.0."""
    payload = {"bayes": True, "vote_average": 8.0, "vote_count": 1000,
               "media_type": "movie"}
    assert bayes_from_payload(payload) == pytest.approx(bayes_score(8.0, 1000, "movie"))


# ------------------------------------------------------------- stratification
def _catalog(movie_en=0, movie_xx=0, series_en=0, **kw) -> list[dict]:
    """A synthetic catalog shaped like the live one.

    Median vote_count matters here: the live 45k catalog measured medians of
    381 (movie/en), 133 (movie/xx), 154 (series/en) and 73 (series/xx). A flat
    vote_count is NOT representative — it produces m at the clamp and hides the
    very behaviour these tests check for.
    """
    rows: list[dict] = []

    def fill(n, media_type, language, mean_vote, median_votes):
        for i in range(n):
            # Spread around the median so median(vote_count) ~= median_votes.
            v = median_votes + (i - n // 2) * 3
            rows.append({"media_type": media_type, "language": language,
                         "vote_average": mean_vote, "vote_count": max(60, v)})

    fill(movie_en, "movie", "en", 7.2, 381)
    fill(movie_xx, "movie", "ml", 8.0, 133)
    fill(series_en, "series", "en", 7.5, 154)
    return rows


def test_bayes_stats_separates_movie_and_series():
    """Two populations with different means get different C."""
    stats = policy.bayes_stats(_catalog(movie_en=400, series_en=400))
    movie_c = stats[("movie", "en")][0]
    series_c = stats[("series", "en")][0]
    # movie_en mean 7.2, series... same rows but media_type differs, so means are
    # equal here; use distinct means to be meaningful.
    stats2 = policy.bayes_stats(
        [{"media_type": "movie", "language": "en", "vote_average": 7.0,
          "vote_count": 5000}] * 300
        + [{"media_type": "series", "language": "en", "vote_average": 8.5,
            "vote_count": 300}] * 300)
    assert stats2[("movie", "en")][0] == pytest.approx(7.0, abs=0.01)
    assert stats2[("series", "en")][0] == pytest.approx(8.5, abs=0.01)
    assert stats2[("movie", "en")] != stats2[("series", "en")]
    assert movie_c is not None and series_c is not None


def test_bayes_stats_non_english_gets_its_own_prior():
    """The xx stratum's C tracks its own population, not the English one."""
    stats = policy.bayes_stats(_catalog(movie_en=400, movie_xx=400))
    assert stats[("movie", "xx")][0] == pytest.approx(8.0, abs=0.05)
    assert stats[("movie", "en")][0] == pytest.approx(7.2, abs=0.05)


def test_thin_stratum_shrinks_toward_global():
    """A 5-item stratum's C sits between its own mean and the global mean."""
    rows = [{"media_type": "movie", "language": "en", "vote_average": 7.0,
             "vote_count": 5000}] * 300
    rows += [{"media_type": "movie", "language": "ko", "vote_average": 9.8,
              "vote_count": 60}] * 5
    stats = policy.bayes_stats(rows)
    own_mean = 9.8
    global_mean = stats[("movie", "en")][0]
    c_xx = stats[("movie", "xx")][0]
    assert min(own_mean, global_mean) < c_xx < max(own_mean, global_mean), (
        f"thin stratum C={c_xx} did not shrink between {own_mean} and {global_mean}")


def test_foreign_stratum_uses_a_lower_vote_floor():
    """vote_count >= 50 would discard most foreign titles — the population being
    corrected. The xx floor must be 10, so a 20-vote foreign title counts."""
    rows = [{"media_type": "movie", "language": "en", "vote_average": 7.0,
             "vote_count": 5000}] * 300
    # 200 foreign titles with only 20 votes each: below the `en` floor of 50.
    rows += [{"media_type": "movie", "language": "ml", "vote_average": 9.0,
              "vote_count": 20}] * 200
    stats = policy.bayes_stats(rows)
    assert stats[("movie", "xx")][0] == pytest.approx(9.0, abs=0.01), (
        "the xx stratum's C ignored 20-vote titles (floor is still 50)")


def test_m_gems_is_derived_not_fixed():
    """m_gems must scale with the stratum's m, or it loses meaning on `xx`."""
    stats = policy.bayes_stats(_catalog(movie_en=400, movie_xx=400))
    for (media_type, bucket), (c, m, m_gems) in stats.items():
        assert m_gems == pytest.approx(max(10.0, policy.GEMS_M_FRACTION * m)), (
            f"{media_type}/{bucket}: m_gems={m_gems} not derived from m={m}")
    # The xx stratum has a much smaller m than the en one.
    assert stats[("movie", "xx")][1] < stats[("movie", "en")][1]


def test_m_is_clamped():
    stats = policy.bayes_stats(
        [{"media_type": "movie", "language": "en", "vote_average": 7.0,
          "vote_count": 10_000_000}] * 5)
    assert stats[("movie", "en")][1] <= policy.M_MAX


def test_bayes_stats_on_an_empty_catalog_is_empty():
    assert policy.bayes_stats([]) == {}


# ------------------------------------------------------------- headline
def test_smoothing_suppresses_a_low_vote_title():
    """The property this whole mechanism exists for.

    An earlier version of this file asserted that a foreign classic (R=8.4, v=60)
    beats a mediocre English blockbuster (R=7.2, v=5000) on `bayes`. It passed —
    but only because m had collapsed to its floor, which switched smoothing off.
    With a correctly sized m the ordering does NOT reverse, and that is correct:
    60 votes is not evidence that a film is good.

    What is asserted instead is what the Bayesian term is actually for.
    """
    stats = policy.bayes_stats(_catalog(movie_en=400, movie_xx=400))
    low_vote = policy.apply_quality(
        {"media_type": "movie", "language": "en", "vote_average": 9.5,
         "vote_count": 12}, stats=stats)
    blockbuster = policy.apply_quality(
        {"media_type": "movie", "language": "en", "vote_average": 7.2,
         "vote_count": 5000}, stats=stats)
    # The raw 9.5 must be pulled far down toward the stratum mean. With m in the
    # hundreds a 12-vote title keeps ~1-2 % of its rating, so it lands essentially
    # ON the mean (7.2) and cannot beat a genuinely well-rated film.
    assert low_vote["bayes"] < 7.6, (
        f"low-vote title was not suppressed: {low_vote['bayes']}")
    assert low_vote["bayes"] <= blockbuster["bayes"] + 0.1, (
        f"'9.5 from 12 people' outranked a 5000-vote film: "
        f"{low_vote['bayes']} vs {blockbuster['bayes']}")
    # And the raw rating really was mostly discarded.
    assert low_vote["bayes"] < 8.0, low_vote["bayes"]


def test_m_is_sized_for_a_real_catalog():
    """m must be big enough that low-vote titles are actually shrunk.

    Regression for the live bug: PRIOR_FRACTION 0.10 against a real catalog's
    medians (73-381) produced m=20, i.e. a 10-vote title kept 33 % of its rating.
    """
    stats = policy.bayes_stats(_catalog(movie_en=400, movie_xx=400))
    for key, (_c, m, _mg) in stats.items():
        assert m > policy.M_MIN, f"{key}: m={m} still pinned to the floor"
        weight_on_raw = 10 / (10 + m)
        assert weight_on_raw < 0.10, (
            f"{key}: a 10-vote title still keeps {weight_on_raw:.0%} of its rating")


def test_strata_are_allowed_to_differ_in_m():
    """Stratification's real job: each stratum shrunk against its OWN votes.

    A title with the same vote_count lands at a different `bayes` depending on how
    vote-heavy its stratum is — that is the correction, and it does not require
    the foreign stratum to be rated lower on average.
    """
    stats = policy.bayes_stats(_catalog(movie_en=400, movie_xx=400))
    en_m = stats[("movie", "en")][1]
    xx_m = stats[("movie", "xx")][1]
    assert en_m != xx_m, "strata collapsed to one m"

    same = {"media_type": "movie", "vote_average": 8.0, "vote_count": 100}
    en = policy.apply_quality({**same, "language": "en"}, stats=stats)
    xx = policy.apply_quality({**same, "language": "ml"}, stats=stats)
    # Same votes, different stratum -> different shrinkage. That is the point.
    assert en["bayes"] != pytest.approx(xx["bayes"], abs=1e-3)


def test_gems_mode_is_where_low_vote_titles_get_through():
    """gems mode deliberately uses a much weaker prior (m_gems = 0.25 * m).

    This is the legitimate place for a well-rated low-vote title to surface —
    not the blockbuster row, which is what a broken shared m used to do.
    """
    stats = policy.bayes_stats(_catalog(movie_en=400, movie_xx=400))
    out = policy.apply_quality({"media_type": "movie", "language": "en",
                                "vote_average": 8.4, "vote_count": 60}, stats=stats)
    assert out["bayes_gems"] > out["bayes"], (
        "gems prior is not weaker than the blockbuster prior")
    _c, m, m_gems = stats[("movie", "en")]
    assert m_gems < m
    # 60 votes is meaningful weight under the gems prior (~24 %), where under the
    # blockbuster prior it is ~7 %.
    assert 60 / (60 + m_gems) > 0.2
    assert 60 / (60 + m_gems) > 2 * (60 / (60 + m))


# ------------------------------------------------------------- ingest stamping
def test_apply_quality_always_stamps_both_fields():
    """set_payload_batch writes these keys wholesale: never leave one unset."""
    payload = policy.apply_quality({"media_type": "movie", "vote_average": 8.0,
                                    "vote_count": 100})
    assert "bayes" in payload and "bayes_gems" in payload
    assert payload["bayes"] > 0 and payload["bayes_gems"] > 0


def test_apply_quality_without_stats_matches_the_old_formula():
    """Signature-compatible default: a single-record caller keeps working."""
    payload = policy.apply_quality({"media_type": "movie", "vote_average": 8.0,
                                    "vote_count": 100, "language": "en"})
    assert payload["bayes"] == pytest.approx(policy.bayes(8.0, 100, "movie"), abs=1e-3)
    assert payload["bayes_gems"] == pytest.approx(
        policy.bayes(8.0, 100, "movie", gems=True), abs=1e-3)


def test_normalize_payload_threads_stats():
    """daily() -> fetch_and_build -> normalize_payload must carry the stats."""
    details = {
        "id": 1, "title": "X", "name": "X", "original_language": "ml",
        "vote_average": 8.4, "vote_count": 60, "genres": [], "credits": {},
        "aggregate_credits": {}, "external_ids": {"imdb_id": "tt1"},
        "created_by": [], "belongs_to_collection": None, "release_date": "2015-01-01",
        "overview": "", "keywords": {},
    }
    stats = policy.bayes_stats(_catalog(movie_en=400, movie_xx=400))
    payload = policy.normalize_payload("movie", details, year_now=2026, stats=stats)
    alone = policy.apply_quality(
        {"media_type": "movie", "language": "ml", "vote_average": 8.4,
         "vote_count": 60}, stats=stats)
    assert payload["bayes"] == pytest.approx(alone["bayes"], abs=1e-3)
    # And it differs from what the un-stratified default would have written.
    default = policy.apply_quality(
        {"media_type": "movie", "language": "ml", "vote_average": 8.4,
         "vote_count": 60})
    assert payload["bayes"] != pytest.approx(default["bayes"], abs=1e-3)


def test_rank_uses_the_stored_bayes_end_to_end():
    """rank() must score with the payload value, not recompute from constants."""
    import numpy as np

    from addon.rank import Candidate, rank

    def make(pid, payload, vec):
        return Candidate(pid, payload["imdb_id"], payload,
                         np.array(vec, dtype=np.float32))

    base = {"media_type": "movie", "vote_average": 7.0, "vote_count": 100,
            "year": 2010, "language": "en", "popularity": 50.0, "director": "",
            "cast": []}
    # Orthogonal directions: a redundancy above MMR_DUP_SIM (0.90) would make MMR
    # drop the second candidate regardless of its score, which would test nothing
    # about bayes.
    stored_high = make("a", {**base, "imdb_id": "tt1", "bayes": 9.9}, [1.0, 0.0, 0.0])
    stored_low = make("b", {**base, "imdb_id": "tt2", "bayes": 1.0}, [0.0, 1.0, 0.0])
    out = rank([1, 0, 0], [stored_low, stored_high], year_seed=2010,
               sim_floor=0.0, limit=2)
    assert [c.point_id for c in out] == ["a", "b"]
    assert stored_high.extras["quality"] == pytest.approx(9.9)
