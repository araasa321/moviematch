"""Edge cases for concurrent page crawling in `TMDBClient.iter_discover`.

These pin the failure modes that a naive "gather all pages" rewrite gets wrong:
walking past the end of the catalogue, and silently skipping a page that failed.
"""

from __future__ import annotations

import asyncio

import pytest

from jobs.tmdb import TMDBClient, TMDBError


class _Client(TMDBClient):
    def __init__(self, pages: int, *, fail_on: int | None = None,
                 empty_on: int | None = None) -> None:
        super().__init__(token="fake")
        self.pages = pages
        self.fail_on = fail_on
        self.empty_on = empty_on
        self.requested: list[int] = []

    async def discover(self, media_type: str, page: int, **kwargs) -> dict:
        self.requested.append(page)
        if self.fail_on is not None and page == self.fail_on:
            raise TMDBError(f"500 for /discover/{media_type}")
        if self.empty_on is not None and page == self.empty_on:
            return {"page": page, "total_pages": self.pages, "results": []}
        if page > self.pages:
            return {"page": page, "total_pages": self.pages, "results": []}
        return {"page": page, "total_pages": self.pages,
                "results": [{"id": page * 100}, {"id": page * 100 + 1}]}


def _ids(client: _Client, **kwargs) -> list[int]:
    async def run() -> list[int]:
        return [item["id"] async for item in client.iter_discover("movie", **kwargs)]
    return asyncio.run(run())


def test_never_requests_past_total_pages():
    """page 1 says there are 3 pages; a window larger than that must not overshoot."""
    client = _Client(pages=3)
    _ids(client, max_pages=500, page_concurrency=10)
    assert max(client.requested) == 3, client.requested


def test_short_last_window_is_partial():
    """7 pages with a window of 4 must fetch 1-4 then 5-7, not 5-8."""
    client = _Client(pages=7)
    ids = _ids(client, max_pages=500, page_concurrency=4)
    assert max(client.requested) == 7
    assert len(ids) == 14


def test_shrinks_to_real_total_pages_on_first_window():
    """A window fetched alongside page 1 must still respect the real total.

    With page_concurrency=10 over a 2-page catalogue the first window asks for
    1-10 BEFORE `total_pages` is known. Those extra pages come back empty rather
    than 404, so the crawl must survive them and report only real results.
    """
    client = _Client(pages=2)
    ids = _ids(client, max_pages=500, page_concurrency=10)
    assert ids == [100, 101, 200, 201]
    assert max(client.requested) <= 10


def test_empty_page_stops_the_result_stream():
    client = _Client(pages=10, empty_on=3)
    ids = _ids(client, max_pages=500, page_concurrency=2)
    assert ids == [100, 101, 200, 201]


def test_failed_page_raises_instead_of_skipping():
    client = _Client(pages=10, fail_on=3)
    with pytest.raises(TMDBError):
        _ids(client, max_pages=500, page_concurrency=2)


def test_page_concurrency_one_matches_serial_behaviour():
    client = _Client(pages=4)
    ids = _ids(client, max_pages=500, page_concurrency=1)
    assert client.requested == [1, 2, 3, 4]
    assert ids == [100, 101, 200, 201, 300, 301, 400, 401]


def test_results_are_in_page_order():
    client = _Client(pages=6)
    ids = _ids(client, max_pages=500, page_concurrency=3)
    assert ids == sorted(ids), "pages must be yielded in order, not completion order"
