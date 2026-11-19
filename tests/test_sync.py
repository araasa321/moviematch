"""End-to-end ingest test using the offline fakes (no network, no model).

Proves the two promises that matter for the nightly job:
  * replaying the same input never re-embeds (idempotent / resumable)
  * a popularity/votes refresh costs zero embeddings
"""

from __future__ import annotations

import asyncio

import pytest

from addon.ids import point_id
from jobs import policy, sync_tmdb
from jobs.embed import build_text
from jobs.offline_run import FakeEmbedder, FakeQdrant, FakeTMDB


def _run(coro):
    return asyncio.run(coro)


@pytest.fixture()
def fakes():
    return FakeTMDB(), FakeQdrant(), FakeEmbedder()


def test_bootstrap_applies_catalog_policy(fakes):
    client, qdrant, embedder = fakes
    stats = sync_tmdb.Stats()
    items = _run(_collect(client, "movie"))
    ready = _run(sync_tmdb.fetch_and_build(client, "movie", items, year_now=2026, stats=stats))
    titles = {row["payload"]["title"] for row in ready}

    assert "The Dark Knight" in titles
    assert "Brand New Release" in titles          # recency clause
    assert "Obscure Indie" not in titles          # low votes + old
    assert "Adult Feature" not in titles          # adult
    assert stats.missing_imdb == 1                # kept but unrenderable


async def _collect(client, media_type):
    return [item async for item in client.iter_discover(media_type)]


def test_points_are_uuid5_and_upserted(fakes):
    client, qdrant, embedder = fakes
    stats = sync_tmdb.Stats()
    ready = _run(sync_tmdb.fetch_and_build(client, "movie", _run(_collect(client, "movie")),
                                           year_now=2026, stats=stats))
    _run(sync_tmdb.upsert_ready(qdrant, embedder, ready, stats))
    assert point_id("movie", 155) in qdrant.points
    payload = qdrant.points[point_id("movie", 155)]["payload"]
    assert payload["text_hash"].startswith("sha256:")
    assert len(qdrant.points[point_id("movie", 155)]["vector"]) == 384


def test_resume_never_reembeds(fakes):
    client, qdrant, embedder = fakes
    stats = sync_tmdb.Stats()
    items = _run(_collect(client, "movie"))
    ready = _run(sync_tmdb.fetch_and_build(client, "movie", items, year_now=2026, stats=stats))
    _run(sync_tmdb.upsert_ready(qdrant, embedder, ready, stats))

    before = embedder.encoded
    again = _run(sync_tmdb.fetch_and_build(client, "movie", items, year_now=2026, stats=stats))
    _run(sync_tmdb.upsert_ready(qdrant, embedder, again, stats,
                                existing=set(qdrant.points)))
    assert embedder.encoded == before, "second pass must not embed anything"
    assert stats.skipped_existing == len(qdrant.points)


def test_nightly_payload_refresh_costs_zero_embeddings(fakes):
    client, qdrant, embedder = fakes
    stats = sync_tmdb.Stats()
    ready = _run(sync_tmdb.fetch_and_build(client, "movie", _run(_collect(client, "movie")),
                                           year_now=2026, stats=stats))
    _run(sync_tmdb.upsert_ready(qdrant, embedder, ready, stats))

    before = embedder.encoded
    sample = _run(client.discover("movie", 1))["results"]
    mean = policy.catalog_mean_vote(sample)
    payloads = {pid: point["payload"] for pid, point in qdrant.points.items()}
    touched = 0
    for item in sample:
        pid = point_id("movie", item["id"])
        if pid not in payloads:          # not in the catalog (policy dropped it)
            continue
        merged = {**payloads[pid], **item}
        policy.apply_quality(merged, mean)
        _run(qdrant.set_payload({"bayes": merged["bayes"],
                                 "vote_count": merged["vote_count"]}, [pid]))
        touched += 1
    assert embedder.encoded == before, "payload refresh must not embed"
    # every indexed title got a payload write; dropped titles are simply absent
    assert touched == len(qdrant.points)


def test_recipe_survives_long_overview():
    """The truncation fix: a 4000-char synopsis must not push the title out."""
    payload = {"title": "Heat", "director": "Michael Mann",
               "overview": "crime " * 700,
               "genres": ["Crime", "Thriller"], "cast": ["Al Pacino"]}
    text = build_text(payload)
    assert text.endswith("Heat.")


def test_cli_dry_run_is_offline():
    args = sync_tmdb.parse_args(["--mode", "nightly", "--dry-run"])
    stats = _run(sync_tmdb.run(args))
    assert stats.upserted > 0


def test_cli_requires_credentials_without_dry_run(monkeypatch):
    monkeypatch.delenv("TMDB_TOKEN", raising=False)
    monkeypatch.delenv("TMDB_API_KEY", raising=False)
    with pytest.raises(SystemExit):
        sync_tmdb.parse_args(["--mode", "nightly"])


# ------------------------------------------------ decade ordering and sorting
def test_bootstrap_crawls_newest_decades_first():
    """A partial run must leave a MODERN catalog, not a 1960s one."""
    args = sync_tmdb.parse_args(["--mode", "bootstrap", "--dry-run"])
    years = sync_tmdb.bootstrap_years(args)
    assert years == sorted(years, reverse=True), years
    assert years[0] >= 2020


def test_bootstrap_respects_explicit_year_range():
    args = sync_tmdb.parse_args(["--mode", "bootstrap", "--dry-run",
                                 "--year-from", "1990", "--year-to", "2026"])
    years = sync_tmdb.bootstrap_years(args)
    assert years[0] == 2020 and years[-1] == 1990
    assert all(1990 <= y <= 2026 for y in years)


def test_default_sort_is_vote_count_not_popularity():
    """Popularity is today's buzz; inside a decade that buries 1995 classics."""
    args = sync_tmdb.parse_args(["--mode", "bootstrap", "--dry-run"])
    assert args.sort_by == "vote_count.desc"


def test_sort_by_can_be_overridden():
    args = sync_tmdb.parse_args(["--mode", "bootstrap", "--dry-run",
                                 "--sort-by", "popularity.desc"])
    assert args.sort_by == "popularity.desc"


# ------------------------------------------------- dedup + per-type --limit
def test_known_ids_skip_before_fetching_details(fakes):
    """Re-running over a filled decade must not re-fetch every title."""
    client, _qdrant, _embedder = fakes
    items = _run(_collect(client, "movie"))
    known = {point_id("movie", 155), point_id("movie", 272), point_id("movie", 27205)}
    before = client.calls
    ready = _run(sync_tmdb.fetch_and_build(client, "movie", items, year_now=2026,
                                           stats=sync_tmdb.Stats(), known=known))
    titles = {r["payload"]["title"] for r in ready}
    assert "The Dark Knight" not in titles
    assert "Inception" not in titles
    # one call per *unknown* title only
    assert client.calls - before <= len(items) - len(known)


def test_per_type_limit_lets_both_types_run(monkeypatch):
    """The old global limit filled movies and never reached series.

    Drives the real bootstrap() with fake TMDB/Qdrant and asserts both media
    types are visited and each type gets its own budget.
    """
    import argparse

    visited: list[str] = []

    class RecordingTMDB(FakeTMDB):
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def details(self, media_type, tmdb_id):
            visited.append(media_type)
            return await super().details(media_type, tmdb_id)

    class TinyQdrant(FakeQdrant):
        pass

    args = argparse.Namespace(
        mode="bootstrap", types=["movie", "series"], limit=1, new_cap=50,
        active_pages=1, max_pages=1, rate=50.0, concurrency=8, batch=8, model="x",
        recreate=False, dry_run=False, stats_out=None,
        tmdb_token="t", qdrant_url="http://x", qdrant_key="k",
        decades=[2020], year_from=None, year_to=None, recent_only=True,
        no_recent_pass=True, min_votes=None, sort_by="vote_count.desc",
        langs=["en"],
    )

    real_fetch = sync_tmdb.fetch_and_build

    async def fake_fetch(client, media_type, candidates, **kw):
        return await real_fetch(client, media_type, list(candidates), **kw)

    monkeypatch.setattr(sync_tmdb, "TMDBClient", lambda *a, **k: RecordingTMDB())
    monkeypatch.setattr(sync_tmdb, "QdrantClient", lambda *a, **k: TinyQdrant())
    monkeypatch.setattr(sync_tmdb, "Embedder", lambda *a, **k: FakeEmbedder())
    monkeypatch.setattr(sync_tmdb, "fetch_and_build", fake_fetch)

    _run(sync_tmdb.bootstrap(args))
    # with a per-type limit of 1, at least one title of EACH type must be fetched
    assert "movie" in visited, visited
    assert "series" in visited, visited


def test_bootstrap_crawls_every_requested_language(monkeypatch):
    """`--langs en ja` must drive one distinct discover pass per language.

    Guards the actual feature: without this, adding a language to the workflow
    input could silently still crawl English only.
    """
    import argparse

    discover_calls: list[dict] = []
    details_langs: list[str] = []

    class RecordingTMDB(FakeTMDB):
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def discover(self, media_type, page, **kwargs):
            discover_calls.append({"media_type": media_type, "page": page, **kwargs})
            return await super().discover(media_type, page, **kwargs)

        async def details(self, media_type, tmdb_id):
            data = await super().details(media_type, tmdb_id)
            details_langs.append(data.get("original_language"))
            return data

    args = argparse.Namespace(
        mode="bootstrap", types=["movie"], limit=None, new_cap=50,
        active_pages=1, max_pages=1, rate=50.0, concurrency=8, batch=8, model="x",
        recreate=False, dry_run=False, stats_out=None,
        tmdb_token="t", qdrant_url="http://x", qdrant_key="k",
        decades=[2020], year_from=None, year_to=None, recent_only=False,
        no_recent_pass=True, min_votes=None, sort_by="vote_count.desc",
        langs=["en", "ja"],
    )

    monkeypatch.setattr(sync_tmdb, "TMDBClient", lambda *a, **k: RecordingTMDB())
    monkeypatch.setattr(sync_tmdb, "QdrantClient", lambda *a, **k: FakeQdrant())
    monkeypatch.setattr(sync_tmdb, "Embedder", lambda *a, **k: FakeEmbedder())

    _run(sync_tmdb.bootstrap(args))

    requested = [c.get("with_original_language") for c in discover_calls
                 if c["media_type"] == "movie"]
    assert "en" in requested, f"English pass missing: {discover_calls}"
    assert "ja" in requested, f"Japanese pass missing: {discover_calls}"


def test_bootstrap_english_only_never_filters_language(monkeypatch):
    """A plain `--langs en` run must behave like the original English crawl."""
    import argparse

    discover_calls: list[dict] = []

    class RecordingTMDB(FakeTMDB):
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def discover(self, media_type, page, **kwargs):
            discover_calls.append({"media_type": media_type, "page": page, **kwargs})
            return await super().discover(media_type, page, **kwargs)

    args = argparse.Namespace(
        mode="bootstrap", types=["movie"], limit=None, new_cap=50,
        active_pages=1, max_pages=1, rate=50.0, concurrency=8, batch=8, model="x",
        recreate=False, dry_run=False, stats_out=None,
        tmdb_token="t", qdrant_url="http://x", qdrant_key="k",
        decades=[2020], year_from=None, year_to=None, recent_only=False,
        no_recent_pass=True, min_votes=None, sort_by="vote_count.desc",
        langs=["en"],
    )

    monkeypatch.setattr(sync_tmdb, "TMDBClient", lambda *a, **k: RecordingTMDB())
    monkeypatch.setattr(sync_tmdb, "QdrantClient", lambda *a, **k: FakeQdrant())
    monkeypatch.setattr(sync_tmdb, "Embedder", lambda *a, **k: FakeEmbedder())

    _run(sync_tmdb.bootstrap(args))

    langs = {c.get("with_original_language") for c in discover_calls}
    assert langs == {"en"}, f"English-only crawl sent {langs}"


def test_recency_pass_also_uses_known_set():
    import inspect

    source = inspect.getsource(sync_tmdb.bootstrap)
    recent_section = source.split("no_recent_pass")[1]
    assert "known=done" in recent_section
