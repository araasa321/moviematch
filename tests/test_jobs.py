"""Ingest-job tests (PLAN §5 policy, §6 recipe, §6.3 nightly behaviour)."""

from __future__ import annotations

import pytest

from jobs import policy
from jobs.embed import build_text, needs_reembed, text_hash


# ------------------------------------------------------------------- policy
def _movie(**kw):
    base = {"media_type": "movie", "vote_count": 500, "release_date": "2015-01-01",
            "imdb_id": "tt1", "adult": False, "video": False}
    base.update(kw)
    return base


def test_movies_need_100_votes():
    assert policy.should_index(_movie(vote_count=100), year_now=2026).keep
    assert not policy.should_index(_movie(vote_count=99), year_now=2026).keep


def test_series_need_only_50_votes():
    assert policy.should_index({"media_type": "series", "vote_count": 50,
                                "first_air_date": "2010-01-01", "imdb_id": "tt2"},
                               year_now=2026).keep


def test_recency_clause_keeps_zero_vote_releases():
    """The fix for the old TF-IDF repo's cold start (PLAN §5)."""
    fresh = _movie(vote_count=0, release_date="2026-10-01")
    assert policy.should_index(fresh, year_now=2026).keep


def test_trakt_presence_overrides_thresholds():
    old = _movie(vote_count=2, release_date="1999-01-01")
    assert not policy.should_index(old, year_now=2026).keep
    assert policy.should_index(old, year_now=2026, in_trakt=True).keep


def test_adult_and_video_are_always_dropped():
    assert not policy.should_index(_movie(adult=True), year_now=2026).keep
    assert not policy.should_index(_movie(video=True), year_now=2026).keep


def test_missing_imdb_is_kept_but_flagged():
    verdict = policy.should_index(_movie(imdb_id=None), year_now=2026)
    assert verdict.keep and verdict.missing == "imdb_id"


# ------------------------------------------------------------------ quality
def test_apply_quality_writes_bayes_and_gems():
    payload = {"media_type": "movie", "vote_average": 8.5, "vote_count": 32000}
    policy.apply_quality(payload, mean=6.4)
    assert payload["bayes"] == pytest.approx(8.44, abs=0.02)
    # gems uses m=300, so a low-vote gem scores higher than under m=1000
    assert payload["bayes_gems"] > payload["bayes"] - 0.05


def test_catalog_mean_ignores_tiny_vote_counts():
    sample = [{"vote_average": 9.0, "vote_count": 5}] * 500
    assert policy.catalog_mean_vote(sample) == policy.CATALOG_MEAN_VOTE


# ------------------------------------------------------------------- recipe
def test_recipe_puts_title_last_so_truncation_keeps_it():
    text = build_text({"title": "The Dark Knight", "director": "Christopher Nolan",
                       "genres": ["Action"], "cast": ["Christian Bale"],
                       "overview": "x " * 900})
    assert text.endswith("The Dark Knight.")
    assert "Directed by Christopher Nolan Directed by Christopher Nolan" in text


def test_recipe_is_deterministic_and_hashable():
    payload = {"title": "Heat", "director": "Michael Mann", "overview": "L.A. crime."}
    assert build_text(payload) == build_text(payload)
    assert text_hash(build_text(payload)).startswith("sha256:")


def test_popularity_change_does_not_trigger_reembed():
    old = {"title": "Heat", "overview": "L.A. crime.", "popularity": 5.0, "vote_count": 10}
    new = {**old, "popularity": 999.0, "vote_count": 5000, "vote_average": 7.9}
    assert not needs_reembed(old, new)


def test_text_change_triggers_reembed():
    old = {"title": "Heat", "overview": "L.A. crime."}
    new = {"title": "Heat", "overview": "A different synopsis entirely."}
    assert needs_reembed(old, new)


# ------------------------------------------------------- TMDB normalisation
def test_normalize_movie_maps_plan_fields():
    details = {
        "id": 155, "title": "The Dark Knight", "release_date": "2008-07-16",
        "genres": [{"id": 28, "name": "Action"}],
        "keywords": {"keywords": [{"name": "vigilante"}]},
        "credits": {"cast": [{"name": "Christian Bale"}, {"name": "Heath Ledger"}],
                    "crew": [{"job": "Director", "name": "Christopher Nolan"}]},
        "external_ids": {"imdb_id": "tt0468569"},
        "belongs_to_collection": {"id": 263},
        "vote_average": 8.5, "vote_count": 32000, "popularity": 120.0,
        "original_language": "en", "overview": "Batman raises the stakes.",
    }
    payload = policy.normalize_payload("movie", details, year_now=2026)
    assert payload["tmdb_id"] == 155
    assert payload["imdb_id"] == "tt0468569"
    assert payload["year"] == 2008
    assert payload["director"] == "Christopher Nolan"
    assert payload["cast"][:2] == ["Christian Bale", "Heath Ledger"]
    assert payload["collection_id"] == 263
    assert payload["keywords"] == ["vigilante"]
    assert payload["enriched"] is True and "bayes" in payload


def test_normalize_series_uses_creator_and_aggregate_credits():
    details = {
        "id": 1396, "name": "Breaking Bad", "first_air_date": "2008-01-20",
        "last_air_date": "2013-09-29",
        "created_by": [{"name": "Vince Gilligan"}],
        "aggregate_credits": {"cast": [{"name": "Bryan Cranston"}], "crew": []},
        "keywords": {"results": [{"name": "drugs"}]},
        "external_ids": {"imdb_id": "tt0903747"},
        "vote_average": 8.9, "vote_count": 14000, "original_language": "en",
    }
    payload = policy.normalize_payload("series", details, year_now=2026)
    assert payload["director"] == "Vince Gilligan"
    assert payload["end_year"] == 2013
    assert payload["cast"] == ["Bryan Cranston"]
