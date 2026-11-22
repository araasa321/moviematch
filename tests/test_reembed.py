"""#3 / M7 — embedding recipe + the re-embed driver (and its data-loss guard).

Why every title needs re-embedding after this change: `text_hash` is derived from
`build_text`, so any reordering changes the hash for EVERY record. That makes M7
a job run, not an instant deploy.

The dangerous part is the WRITE. `QdrantClient.upsert` is SET semantics
(`PUT /points` replaces the stored payload wholesale), so a maintenance script
that re-upserts recipe-only points would erase imdb_id/poster/bayes/language/…
across the whole catalog. The driver writes vectors via `update_vectors` and the
hash via `set_payload_batch` — neither of which can replace a payload — and
verifies that on one point before touching the rest.
"""

from __future__ import annotations

import asyncio

import pytest

from jobs.embed import OVERVIEW_WORDS, build_text, text_hash
from jobs.offline_run import FakeQdrant, FakeEmbedder
from jobs.reembed import CANARY_FIELDS, _plan, reembed


# ------------------------------------------------------------ recipe shape
_PAYLOAD = {"title": "The Matrix", "tagline": "Free your mind.",
            "genres": ["Action", "Sci-Fi"], "keywords": ["time loop", "whodunit"],
            "director": "Lana Wachowski", "cast": ["Keanu", "Carrie"],
            "overview": "A hacker learns."}


def test_title_appears_in_first_six_words():
    text = build_text(_PAYLOAD)
    assert "The Matrix" in " ".join(text.split()[:6]), text[:80]


def test_title_survives_a_4000_char_overview():
    """The actual bug: the title must come BEFORE the overview, not after it."""
    overview = ("A thief steals secrets through dream-sharing technology. " * 64)[:4000]
    text = build_text({**_PAYLOAD, "overview": overview})
    words = text.split()
    title_positions = [i for i, w in enumerate(words) if "Matrix" in w]
    overview_start = words.index("A") if "A" in words else len(words)
    assert title_positions, "title vanished entirely"
    assert min(title_positions) < 6, f"title only at word {min(title_positions)}"
    # A repeat immediately precedes the prose.
    assert any(p < overview_start for p in title_positions)


def test_overview_is_bounded():
    """A 4,000-char overview must not blow past the token budget."""
    overview = ("x " * 2000).strip()
    text = build_text({**_PAYLOAD, "overview": overview})
    # ~120 words of overview + ~25 words of structured signal.
    assert len(text.split()) <= 180, len(text.split())
    # And a short overview is left intact rather than padded/truncated.
    short = build_text({**_PAYLOAD, "overview": "Short plot."})
    assert short.endswith("Short plot.")


def test_overview_is_truncated_at_the_word_budget():
    text = build_text({**_PAYLOAD, "overview": "w " * 500}, overview_words=10)
    assert text.split()[-10:] == ["w"] * 10


def test_build_text_is_deterministic():
    assert build_text(_PAYLOAD) == build_text(_PAYLOAD)
    assert text_hash(build_text(_PAYLOAD)) == text_hash(build_text(_PAYLOAD))


def test_reordering_changes_hash():
    """Documents that M7 REQUIRES a re-embed of existing titles."""
    old_style = ("The Matrix. Free your mind. Genres: Action, Sci-Fi. "
                 "Keywords: time loop, whodunit. Directed by Lana Wachowski. "
                 "Directed by Lana Wachowski. Starring Keanu, Carrie. "
                 "A hacker learns. The Matrix. The Matrix.")
    new = build_text(_PAYLOAD)
    assert text_hash(old_style) != text_hash(new), (
        "recipe change did not change the hash — check build_text")


def test_tagline_gets_a_period():
    """Without one, the tagline ran into the genre block as prose."""
    text = build_text({**_PAYLOAD, "tagline": "Free your mind"})
    assert "Free your mind. Genres:" in text
    # An already-terminated tagline is not double-punctuated.
    assert "Free your mind.. " not in build_text(_PAYLOAD)


# ------------------------------------------------ re-embed: payload retention
def _point(pid: str, *, text_hash_value: str = "sha256:stale") -> dict:
    return {"id": pid, "vector": [0.0] * 8,
            "payload": {"imdb_id": "tt0133093", "poster": "/matrix.jpg",
                        "vote_count": 24000, "language": "en", "enriched": True,
                        "media_type": "movie", "title": "The Matrix",
                        "director": "Lana Wachowski", "overview": "A hacker learns.",
                        "text_hash": text_hash_value}}


def _seed(n: int = 3) -> FakeQdrant:
    q = FakeQdrant()
    for i in range(n):
        q.points[f"p{i}"] = _point(f"p{i}")
    return q


def _reembed_against(q: FakeQdrant, **kw):
    """Run the driver with FakeQdrant and FakeEmbedder injected."""
    import jobs.reembed as reembed_mod

    class _Q:
        """Shim exposing FakeQdrant through the QdrantClient constructor."""

        def __init__(self, *a, **k) -> None:
            pass

        def __getattr__(self, name):
            return getattr(q, name)

    original = reembed_mod.QdrantClient
    reembed_mod.QdrantClient = _Q
    try:
        return asyncio.run(reembed_mod.reembed(embedder=kw.pop("embedder", FakeEmbedder()),
                                               **kw))
    finally:
        reembed_mod.QdrantClient = original


def test_reembed_retains_untouched_payload_keys():
    """MANDATORY: only the vector and text_hash may move.

    This is the test that fails loudly if anyone replaces the driver with a naive
    upsert. FakeQdrant.upsert keeps real SET semantics, so a regression here is
    detected rather than hidden.
    """
    q = _seed(3)
    before = {pid: dict(p["payload"]) for pid, p in q.points.items()}
    _reembed_against(q)
    for pid, point in q.points.items():
        after = point["payload"]
        for field in ("imdb_id", "poster", "enriched", "vote_count", "language"):
            assert after.get(field) == before[pid].get(field), (
                f"{pid}: field {field!r} was lost/changed by the re-embed")
        # text_hash moved to the current recipe's value
        assert after["text_hash"] == text_hash(build_text(after))
        assert after["text_hash"] != before[pid]["text_hash"]


def test_reembed_updates_the_vector():
    q = _seed(2)
    old_vectors = {pid: list(p["vector"]) for pid, p in q.points.items()}
    embedder = FakeEmbedder()
    _reembed_against(q, embedder=embedder)
    assert embedder.encoded > 0, "no embedding was computed"
    for pid, point in q.points.items():
        assert point["vector"] != old_vectors[pid], f"{pid}: vector unchanged"


def test_reembed_never_calls_upsert(monkeypatch):
    """A regression guard at the API level: the driver must not use upsert()."""
    q = _seed(2)
    calls = {"upsert": 0}
    real_upsert = q.upsert

    async def counting_upsert(points, **kw):
        calls["upsert"] += 1
        return await real_upsert(points, **kw)

    q.upsert = counting_upsert
    _reembed_against(q)
    assert calls["upsert"] == 0, "the re-embed driver called upsert() — payloads are at risk"


def test_update_vectors_uses_put_and_sends_no_payload():
    """Each point in the body is exactly {id, vector}, and the method is PUT.

    POST on this path returns 405 Method Not Allowed (hit against Qdrant Cloud on
    the first real --limit 5 run), so the verb is pinned here.
    """
    from jobs.qdrant_client import QdrantClient

    captured: dict = {}

    async def fake_request(method, path, payload=None, **kwargs):
        captured["method"], captured["path"], captured["payload"] = method, path, payload
        return {"result": {}}

    client = QdrantClient("https://x.qdrant.io", "k")
    client._request = fake_request            # type: ignore[method-assign]
    asyncio.run(client.update_vectors([("p1", [0.1, 0.2])], wait=True))

    assert captured["method"] == "PUT", (
        f"update_vectors must use PUT; {captured['method']} returns 405")
    assert captured["path"].endswith("/points/vectors"), captured["path"]
    points = captured["payload"]["points"]
    assert points == [{"id": "p1", "vector": [0.1, 0.2]}]
    assert "payload" not in points[0]
    # And it is not the upsert endpoint.
    assert not captured["path"].endswith("/points")


def test_dry_run_writes_nothing():
    q = _seed(3)
    snapshot = {pid: (dict(p["payload"]), list(p["vector"])) for pid, p in q.points.items()}
    stats = _reembed_against(q, dry_run=True)
    assert stats["changed"] == 3
    for pid, point in q.points.items():
        assert point["payload"] == snapshot[pid][0]
        assert point["vector"] == snapshot[pid][1]


def test_already_current_records_are_skipped():
    q = FakeQdrant()
    fresh = _point("p0")
    fresh["payload"]["text_hash"] = text_hash(build_text(fresh["payload"]))
    q.points["p0"] = fresh
    stats = _reembed_against(q, dry_run=True)
    assert stats["changed"] == 0


def test_plan_reports_none_for_an_unchanged_record():
    payload = dict(_point("p0")["payload"])
    payload["text_hash"] = text_hash(build_text(payload))
    assert _plan({"id": "p0", "payload": payload}) is None


def test_preflight_aborts_if_payload_would_change():
    """If a write path ever clobbered the payload, the run must stop BEFORE the sweep."""
    q = _seed(3)

    async def lossy_update_vectors(points, **kw):
        for pid, vec in points:
            if pid in q.points:
                # a regression that also nukes the payload
                q.points[pid] = {"id": pid, "vector": list(vec), "payload": {}}

    q.update_vectors = lossy_update_vectors
    import jobs.reembed as reembed_mod
    from jobs.qdrant_client import QdrantError

    with pytest.raises(QdrantError) as exc:
        _reembed_against(q)
    assert "PREFLIGHT FAILED" in str(exc.value)
    # Only the probe point may have been touched; the rest are untouched.
    intact = [p for p in q.points.values() if p["payload"].get("imdb_id")]
    assert len(intact) >= 2, "the sweep continued after a failed preflight"


def test_canary_fields_are_real_recipe_external_fields():
    assert "imdb_id" in CANARY_FIELDS and "text_hash" not in CANARY_FIELDS


def test_rejected_update_vectors_writes_nothing():
    """A 405/4xx from update_vectors must leave every point exactly as it was.

    This is the scenario that actually happened on the first live run: the
    endpoint was called with POST and Qdrant answered 405. Because update_vectors
    threw BEFORE set_payload ran, no point — including the probe — was modified.
    """
    q = _seed(3)
    snapshot = {pid: (dict(p["payload"]), list(p["vector"])) for pid, p in q.points.items()}

    async def reject(points, **kw):
        from jobs.qdrant_client import QdrantError
        raise QdrantError("PUT /collections/titles/points/vectors -> 405: ")

    q.update_vectors = reject
    from jobs.qdrant_client import QdrantError

    with pytest.raises(QdrantError) as exc:
        _reembed_against(q)
    assert "NOTHING was written" in str(exc.value)
    # Every point is byte-identical: no vector moved, no hash rewritten.
    for pid, point in q.points.items():
        assert point["payload"] == snapshot[pid][0], f"{pid}: payload changed"
        assert point["vector"] == snapshot[pid][1], f"{pid}: vector changed"


def test_sweep_does_not_fetch_each_point():
    """Throughput guard: the sweep must not do one round trip per title.

    Measured on Qdrant Cloud: each POST /points costs ~0.43 s, so fetching 64
    points per batch was ~80 % of the run's wall time (6.9 h projected for 45k
    titles). `_collect` already carries the built text out of the scroll, so the
    sweep needs ZERO per-point reads.
    """
    q = _seed(3)
    reads = {"n": 0}
    real_get = q.get_point

    async def counting_get(pid, **kw):
        reads["n"] += 1
        return await real_get(pid, **kw)

    q.get_point = counting_get
    _reembed_against(q, batch=1)
    # Only the preflight probe may read: once BEFORE and once AFTER the write, to
    # prove the payload survived. The sweep itself reads nothing.
    assert reads["n"] == 2, f"the sweep made {reads['n'] - 2} per-point fetches"


def test_overview_words_constant_is_sane():
    # ~120 words is ~170 word pieces for MiniLM: headroom under 256 tokens.
    assert 80 <= OVERVIEW_WORDS <= 160
