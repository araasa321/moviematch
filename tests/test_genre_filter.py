"""Genre filter on catalog rows (Stremio `genre=` extra).

The manifest advertises a genre dropdown on For You, Stremio sends
`genre=Comedy` in the URL, and the server used to parse it and then drop it on
the floor: `_catalog_response` only ever read `skip`. These tests pin the fixed
behaviour so the value cannot silently start being ignored again.
"""

from __future__ import annotations

import pytest

from addon.rows import build_row


def _genres(plan) -> set[str]:
    return {g for cand in plan.ranked for g in (cand.payload.get("genres") or [])}


def _titles(plan) -> set[str]:
    return {cand.payload.get("title") for cand in plan.ranked}


def test_genre_filters_ranked_pool(store, history, config):
    """Every ranked candidate must actually carry the requested genre.

    Titles are multi-genre, so the assertion is membership, not equality: an
    Action/Thriller film legitimately satisfies an "Action" request.
    """
    plan = build_row(store, history, "movie", "foryou", config, genre="Action")
    assert plan.ranked, "genre filter must not empty an otherwise-populated row"
    missing = [c.payload["title"] for c in plan.ranked
               if "Action" not in (c.payload.get("genres") or [])]
    assert not missing, f"leaked non-Action titles: {missing}"


def test_genre_filter_excludes_titles_without_it(store, history, config):
    """A title that does not carry the genre must not appear."""
    plan = build_row(store, history, "movie", "foryou", config, genre="Action")
    titles = _titles(plan)
    assert "Inception" in titles
    assert "Spirited Away" not in titles, "animation title leaked into Action"
    assert "2001: A Space Odyssey" not in titles


def test_genre_is_case_insensitive(store, history, config):
    """Stremio options may differ in case from the stored payload strings."""
    lower = build_row(store, history, "movie", "foryou", config, genre="action")
    proper = build_row(store, history, "movie", "foryou", config, genre="Action")
    assert _titles(lower) == _titles(proper)


def test_different_genres_give_different_rows(store, history, config):
    """The regression: genre A and genre B used to return an identical list."""
    a = build_row(store, history, "movie", "foryou", config, genre="Action")
    b = build_row(store, history, "movie", "foryou", config, genre="Drama")
    assert _titles(a) != _titles(b), "two genres returned the same titles"


def test_unknown_genre_returns_empty_not_everything(store, history, config):
    """An unmatched genre must yield an empty row, never the unfiltered pool."""
    plan = build_row(store, history, "movie", "foryou", config, genre="Nonexistent Genre")
    assert plan.ranked == []
    assert plan.diagnostics.get("reason") == "empty_genre"


def test_no_genre_keeps_full_pool(store, history, config):
    """Absent genre must behave exactly as before the change."""
    unfiltered = build_row(store, history, "movie", "foryou", config)
    empty = build_row(store, history, "movie", "foryou", config, genre="")
    assert _titles(unfiltered) == _titles(empty)
    assert unfiltered.ranked
