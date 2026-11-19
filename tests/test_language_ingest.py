"""Language-aware ingest: `--langs`, the foreign vote floor, and pacing.

All of this is exercised against fakes — no TMDB key, no network. The point is to
pin the decisions that are easy to get silently wrong:

  * `--langs en` must behave EXACTLY like the old English-only crawl;
  * a ja/ko crawl must actually ask TMDB for that language;
  * non-English titles must clear a lower vote bar, or the crawl imports nothing;
  * paged discover must stay in order and stop at `total_pages`.
"""

from __future__ import annotations

import asyncio

import pytest

from jobs import policy
from jobs import sync_tmdb
from jobs.tmdb import TMDBClient


# ------------------------------------------------------------------ policy
def _payload(**over) -> dict:
    base = {"media_type": "movie", "vote_count": 0, "vote_average": 7.0,
            "language": "en", "imdb_id": "tt1", "adult": False, "video": False,
            "release_date": "1995-01-01"}
    base.update(over)
    return base


def test_english_title_keeps_english_threshold():
    """A 60-vote English film is still dropped — unchanged from before."""
    verdict = policy.should_index(_payload(vote_count=60, language="en"),
                                  year_now=2026, keep_langs=("en", "ja"))
    assert verdict.keep is False


def test_foreign_title_uses_lower_floor():
    """The same 60 votes passes for a Japanese title once languages are enabled."""
    verdict = policy.should_index(_payload(vote_count=60, language="ja"),
                                  year_now=2026, keep_langs=("en", "ja"))
    assert verdict.keep is True


def test_foreign_title_below_foreign_floor_still_dropped():
    verdict = policy.should_index(_payload(vote_count=2, language="ja"),
                                  year_now=2026, keep_langs=("en", "ja"))
    assert verdict.keep is False


def test_no_keep_langs_preserves_original_behaviour():
    """Without `keep_langs` every language is held to the English bar."""
    ja = policy.should_index(_payload(vote_count=60, language="ja"), year_now=2026)
    en = policy.should_index(_payload(vote_count=60, language="en"), year_now=2026)
    assert ja.keep == en.keep == False


def test_recent_foreign_title_kept_regardless_of_votes():
    verdict = policy.should_index(
        _payload(vote_count=0, language="ko", release_date="2026-01-01"),
        year_now=2026, keep_langs=("en", "ko"))
    assert verdict.keep is True


def test_adult_foreign_title_still_dropped():
    verdict = policy.should_index(_payload(vote_count=500, language="ja", adult=True),
                                  year_now=2026, keep_langs=("en", "ja"))
    assert verdict.keep is False


def test_normalize_records_origin_country():
    details = {"id": 1, "title": "X", "release_date": "2000-01-01",
               "original_language": "ja", "origin_country": ["JP", "US"],
               "credits": {"cast": [], "crew": []}, "keywords": {"keywords": []},
               "external_ids": {"imdb_id": "tt1"}}
    payload = policy.normalize_payload("movie", details, year_now=2026)
    assert payload["origin_country"] == ["JP", "US"]
    assert payload["language"] == "ja"


# ------------------------------------------------------------------ thresholds
def test_vote_threshold_english_only_is_unchanged():
    assert sync_tmdb._vote_threshold("movie", "en", False) == policy.MOVIE_MIN_VOTES
    assert sync_tmdb._vote_threshold("series", "en", False) == policy.SERIES_MIN_VOTES


def test_vote_threshold_foreign_slices_are_lower():
    assert sync_tmdb._vote_threshold("movie", "ja", True) == policy.FOREIGN_MOVIE_MIN_VOTES
    assert sync_tmdb._vote_threshold("series", "ko", True) == policy.FOREIGN_SERIES_MIN_VOTES
    # an explicit single non-English language is also a foreign slice
    assert sync_tmdb._vote_threshold("movie", "ja", False) == policy.FOREIGN_MOVIE_MIN_VOTES


# ------------------------------------------------------- discover pagination
class _PagedFake(TMDBClient):
    """Serves `total_pages` pages of 2 results each and records the requests.

    Subclasses the real client so `iter_discover` under test is the shipped code,
    not a reimplementation; only the network-facing `discover` is faked.
    """

    def __init__(self, total_pages: int) -> None:
        super().__init__(token="fake")
        self.total_pages = total_pages
        self.requested: list[int] = []

    async def discover(self, media_type: str, page: int, **kwargs) -> dict:
        self.requested.append(page)
        if page > self.total_pages:
            return {"page": page, "total_pages": self.total_pages, "results": []}
        return {"page": page, "total_pages": self.total_pages,
                "results": [{"id": page * 10}, {"id": page * 10 + 1}]}


def _collect(fake, **kw) -> list[int]:
    async def run() -> list[int]:
        return [item["id"] async for item in fake.iter_discover("movie", **kw)]
    return asyncio.run(run())


def test_iter_discover_walks_pages_in_order_and_stops():
    fake = _PagedFake(total_pages=5)
    ids = _collect(fake, max_pages=500, page_concurrency=3)
    assert ids == [10, 11, 20, 21, 30, 31, 40, 41, 50, 51]
    # never requests a page beyond total_pages
    assert max(fake.requested) == 5


def test_iter_discover_respects_max_pages():
    fake = _PagedFake(total_pages=50)
    ids = _collect(fake, max_pages=2, page_concurrency=4)
    assert ids == [10, 11, 20, 21]
    assert max(fake.requested) == 2


def test_iter_discover_empty_page_ends_crawl():
    fake = _PagedFake(total_pages=0)
    assert _collect(fake, max_pages=10, page_concurrency=2) == []


def test_iter_discover_propagates_page_failure():
    """A permanently failing page must abort, not silently skip catalog pages."""
    from jobs.tmdb import TMDBError

    class _Broken(_PagedFake):
        async def discover(self, media_type, page, **kwargs):
            if page == 2:
                raise TMDBError("500 for /discover/movie")
            return await super().discover(media_type, page, **kwargs)

    with pytest.raises(TMDBError):
        _collect(_Broken(total_pages=5), max_pages=10, page_concurrency=3)
