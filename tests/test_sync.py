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
