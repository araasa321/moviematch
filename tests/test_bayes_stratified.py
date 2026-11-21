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
    rows: list[dict] = []
    for _ in range(movie_en):
        rows.append({"media_type": "movie", "language": "en", "vote_average": 7.2,
                     "vote_count": 5000})
    for _ in range(movie_xx):
        rows.append({"media_type": "movie", "language": "ml", "vote_average": 8.0,
                     "vote_count": 60})
    for _ in range(series_en):
        rows.append({"media_type": "series", "language": "en", "vote_average": 7.5,
                     "vote_count": 300})
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
def test_foreign_classic_beats_mediocre_english():
    """The headline assertion.

    R=8.4, v=60, non-English vs R=7.2, v=5000, English.
    Under today's constants: 6.513 vs 7.067 — English wins.
    Stratified, the foreign classic must win on `bayes` alone.
    """
    # Today's behaviour, pinned so the premise of the test cannot rot.
    today_foreign = bayes_score(8.4, 60, "movie")
    today_english = bayes_score(7.2, 5000, "movie")
    assert today_foreign == pytest.approx(6.513, abs=1e-3)
    assert today_english == pytest.approx(7.067, abs=1e-3)
    assert today_english > today_foreign

    stats = policy.bayes_stats(_catalog(movie_en=400, movie_xx=400))
    foreign = policy.apply_quality(
        {"media_type": "movie", "language": "ml", "vote_average": 8.4,
         "vote_count": 60}, stats=stats)
    english = policy.apply_quality(
        {"media_type": "movie", "language": "en", "vote_average": 7.2,
         "vote_count": 5000}, stats=stats)
    assert foreign["bayes"] > english["bayes"], (
        f"stratification did not reverse the ordering: "
        f"foreign={foreign['bayes']} english={english['bayes']}")


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
