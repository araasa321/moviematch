"""Curated TMDB lists as a bootstrap input (`--lists`).

A discover crawl cannot reach regional cinema: those titles sit far below the
vote floor, so `--langs ml` found ~370 Malayalam films while a hand-built list
carried 3,101. Lists supplement the crawl and bypass the vote floor.

What these tests pin:
  * list ids/URLs parse, and a typo fails loudly rather than looking empty;
  * lists page correctly and stop at `item_count`;
  * `tv` from TMDB maps to `series` here;
  * future-dated items are dropped by (year, month);
  * curated items skip the vote floor but still lose adult entries.
"""

from __future__ import annotations

import asyncio

import pytest

from jobs import policy, sync_tmdb
from jobs.tmdb import LIST_PAGE_SIZE, TMDBClient, TMDBError


# ------------------------------------------------------------------ parsing
def test_parse_bare_ids():
    assert sync_tmdb.parse_list_ids(["8717522", "8717528"]) == ["8717522", "8717528"]


def test_parse_full_urls():
    urls = ["https://www.themoviedb.org/list/8717522-malayalam",
            "https://www.themoviedb.org/list/8717528-korean/"]
    assert sync_tmdb.parse_list_ids(urls) == ["8717522", "8717528"]


def test_parse_mixed_ids_and_urls():
    values = ["8717522", "https://www.themoviedb.org/list/8717534-hindi"]
    assert sync_tmdb.parse_list_ids(values) == ["8717522", "8717534"]


def test_parse_comma_separated_in_one_token():
    assert sync_tmdb.parse_list_ids(["8717522,8717528"]) == ["8717522", "8717528"]


def test_parse_dedupes_preserving_order():
    values = ["8717522", "8717522", "https://www.themoviedb.org/list/8717522-x"]
    assert sync_tmdb.parse_list_ids(values) == ["8717522"]


def test_parse_rejects_garbage_and_names_it():
    with pytest.raises(ValueError) as exc:
        sync_tmdb.parse_list_ids(["not-a-list"])
    assert "not-a-list" in str(exc.value)


def test_parse_empty_input_is_empty():
    assert sync_tmdb.parse_list_ids([]) == []


# ------------------------------------------------------------------ dates
def test_parse_month_variants():
    assert sync_tmdb.parse_month("2025-12-05") == (2025, 12)
    assert sync_tmdb.parse_month("2028-10-17") == (2028, 10)
    assert sync_tmdb.parse_month("") is None
    assert sync_tmdb.parse_month(None) is None
    assert sync_tmdb.parse_month("2025") is None
    assert sync_tmdb.parse_month("2025-13-01") is None


def test_future_month_is_skipped():
    today = (2026, 11)
    assert not sync_tmdb.is_not_future({"release_date": "2026-12-24"}, today)
    assert not sync_tmdb.is_not_future({"release_date": "2028-10-17"}, today)


def test_current_and_past_months_are_kept():
    today = (2026, 11)
    assert sync_tmdb.is_not_future({"release_date": "2026-11-01"}, today)
    assert sync_tmdb.is_not_future({"release_date": "2025-12-05"}, today)
    assert sync_tmdb.is_not_future({"release_date": "1993-12-25"}, today)


def test_series_uses_first_air_date():
    today = (2026, 11)
    assert sync_tmdb.is_not_future({"first_air_date": "2019-01-25"}, today)
    assert not sync_tmdb.is_not_future({"first_air_date": "2027-01-25"}, today)


def test_missing_date_is_kept():
    """Obscure entries often have no date; dropping them defeats the list."""
    assert sync_tmdb.is_not_future({}, (2026, 11))
    assert sync_tmdb.is_not_future({"release_date": None}, (2026, 11))


# ------------------------------------------------------------------ media type
def test_normalise_media_type():
    assert sync_tmdb._normalise_media_type("movie") == "movie"
    assert sync_tmdb._normalise_media_type("tv") == "series"
    assert sync_tmdb._normalise_media_type("series") == "series"
    assert sync_tmdb._normalise_media_type("person") is None
    assert sync_tmdb._normalise_media_type(None) is None


# ------------------------------------------------------------------ policy
def _payload(**over):
    base = {"media_type": "movie", "vote_count": 0, "vote_average": 7.0,
            "language": "ml", "imdb_id": "tt1", "adult": False, "video": False,
            "release_date": "1995-01-01"}
    base.update(over)
    return base


def test_curated_item_passes_with_zero_votes():
    """The whole point: a curated Malayalam classic with 3 votes is kept."""
    verdict = policy.should_index(_payload(vote_count=3), year_now=2026, curated=True)
    assert verdict.keep is True
    assert verdict.reason == "kept_curated"


def test_curated_old_title_passes_without_recency():
    verdict = policy.should_index(_payload(vote_count=0, release_date="1975-01-01"),
                                  year_now=2026, curated=True)
    assert verdict.keep is True


def test_uncurated_zero_vote_title_is_still_dropped():
    """Control: the bypass must not leak into the normal crawl."""
    verdict = policy.should_index(_payload(vote_count=3, release_date="1995-01-01"),
                                  year_now=2026)
    assert verdict.keep is False


def test_curated_adult_item_is_still_rejected():
    """A curated list can contain mistakes; adult content stays out."""
    verdict = policy.should_index(_payload(vote_count=500, adult=True),
                                  year_now=2026, curated=True)
    assert verdict.keep is False


def test_curated_item_without_imdb_is_kept_but_flagged():
    verdict = policy.should_index(_payload(vote_count=500, imdb_id=None),
                                  year_now=2026, curated=True)
    assert verdict.keep is True
    assert verdict.missing == "imdb_id"


# ------------------------------------------------------------------ paging
class _ListFake(TMDBClient):
    def __init__(self, total: int, *, fail_on: int | None = None,
                 duplicate_first: bool = False) -> None:
        super().__init__(token="fake")
        self.total = total
        self.fail_on = fail_on
        self.duplicate_first = duplicate_first
        self.requested: list[int] = []

    async def list_page(self, list_id: str, page: int) -> dict:
        self.requested.append(page)
        if self.fail_on is not None and page == self.fail_on:
            raise TMDBError(f"500 for /list/{list_id}")
        start = (page - 1) * LIST_PAGE_SIZE
        count = max(0, min(LIST_PAGE_SIZE, self.total - start))
        items = [{"id": 1000 + start + i,
                  "media_type": "movie" if i % 2 == 0 else "tv",
                  "release_date": "2020-01-01"} for i in range(count)]
        if self.duplicate_first and page == 2 and items:
            items.append(dict(items[0]))       # same title listed twice
        return {"id": list_id, "item_count": self.total, "items": items}


def _fetch(fake, **kw):
    return asyncio.run(fake.fetch_list("8717522", **kw))


def test_fetch_list_pages_through_everything():
    fake = _ListFake(total=65)
    items = _fetch(fake)
    assert len(items) == 65
    # 65 items at 20/page -> 4 pages
    assert max(fake.requested) == 4


def test_fetch_list_empty():
    assert _fetch(_ListFake(total=0)) == []


def test_fetch_list_collapses_duplicates():
    fake = _ListFake(total=25, duplicate_first=True)
    items = _fetch(fake)
    ids = [(i["media_type"], i["id"]) for i in items]
    assert len(ids) == len(set(ids)), "duplicate (media_type, id) survived"


def test_fetch_list_raises_instead_of_truncating():
    """A lost page must fail the run, not silently return a partial list."""
    with pytest.raises(TMDBError):
        _fetch(_ListFake(total=200, fail_on=3))


def test_fetch_list_maps_tv_items():
    items = _fetch(_ListFake(total=4))
    assert {sync_tmdb._normalise_media_type(i["media_type"]) for i in items} == {
        "movie", "series"}
