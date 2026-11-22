"""#5 / M6 — cap titles per director, per row.

rank.py capped collections (§9.5) but not directors. Demonstrated, not assumed:
a live rank() call with 8 same-director candidates returned 8/8 Christopher
Nolan. In the fixture catalog Nolan owns 6 of 12 titles, so a heavy-Nolan history
floods a 15-item row.

The cap is per ROW, not per page: _cached_plan builds the full ~200-item ranked
list in one pass and to_metas slices 15 at a time by `skip`, so a director can
appear at most DIRECTOR_CAP times across the whole row, every page.
"""

from __future__ import annotations

import numpy as np

from addon.rank import DIRECTOR_CAP, Candidate, rank


def _cand(pid: int, director: str, *, collection_id=None, vec_index: int | None = None,
          dim: int = 16) -> Candidate:
    """A candidate with a distinct direction so MMR does not collapse it."""
    payload = {"media_type": "movie", "imdb_id": f"tt{pid}", "vote_count": 5000,
               "vote_average": 8.0, "popularity": 50.0, "year": 2010,
               "language": "en", "poster": "/p.jpg", "director": director,
               "cast": [], "collection_id": collection_id}
    v = np.zeros(dim, dtype=np.float32)
    v[(vec_index if vec_index is not None else pid) % dim] = 1.0
    v[((vec_index if vec_index is not None else pid) + 3) % dim] = 0.35
    return Candidate(str(pid), payload["imdb_id"], payload, v)


def _query(dim: int = 16) -> list[float]:
    q = [0.0] * dim
    q[0] = 1.0
    return q


def test_director_cap_limits_duplicates():
    """5 Nolan candidates must yield at most DIRECTOR_CAP in the output."""
    cands = [_cand(i, "Christopher Nolan") for i in range(5)]
    out = rank(_query(), cands, sim_floor=-2.0, limit=5)
    nolan = [c for c in out if c.payload["director"] == "Christopher Nolan"]
    assert len(nolan) <= DIRECTOR_CAP, (
        f"{len(nolan)} Nolan titles survived a cap of {DIRECTOR_CAP}")
    assert DIRECTOR_CAP == 2


def test_director_cap_reproduces_the_reported_flood():
    """The exact shape that produced 8/8 Nolan before the cap."""
    cands = [_cand(i, "Christopher Nolan") for i in range(8)]
    out = rank(_query(), cands, sim_floor=-2.0, limit=8)
    assert len(out) == DIRECTOR_CAP


def test_director_cap_exempts_unknown_director():
    """5 candidates with director="" must all survive, not collapse into one bucket."""
    cands = [_cand(i, "") for i in range(5)]
    out = rank(_query(), cands, sim_floor=-2.0, limit=5)
    assert len(out) == 5, f"unknown-director titles were bucketed: {len(out)} survived"


def test_director_cap_ignores_whitespace_only_director():
    """A director of "   " is the same as unknown: exempt."""
    cands = [_cand(i, "   ") for i in range(4)]
    out = rank(_query(), cands, sim_floor=-2.0, limit=4)
    assert len(out) == 4


def test_director_cap_does_not_starve_small_pools():
    """3 candidates with 3 distinct directors must return all 3."""
    cands = [_cand(1, "A"), _cand(2, "B"), _cand(3, "C")]
    out = rank(_query(), cands, sim_floor=-2.0, limit=3)
    assert len(out) == 3


def test_director_cap_allows_two_from_one_director():
    """The cap is 2, not 1: a director's second-best title still gets through."""
    cands = [_cand(1, "A"), _cand(2, "A"), _cand(3, "A"), _cand(4, "B")]
    out = rank(_query(), cands, sim_floor=-2.0, limit=4)
    a = [c for c in out if c.payload["director"] == "A"]
    assert len(a) == DIRECTOR_CAP


def test_director_cap_lets_other_directors_through():
    """De-flooding must not just drop the pool: the next director fills the row."""
    cands = [_cand(i, "Nolan") for i in range(6)] + [_cand(100 + i, f"Dir{i}")
                                                      for i in range(4)]
    out = rank(_query(), cands, sim_floor=-2.0, limit=6)
    directors = [c.payload["director"] for c in out]
    assert directors.count("Nolan") <= DIRECTOR_CAP
    assert len(out) == 6, f"row was starved to {len(out)} items"
    assert any(d != "Nolan" for d in directors)


def test_collection_cap_still_applies():
    """Regression: max 1 per collection_id, now alongside the director cap."""
    a = _cand(1, "A", collection_id=263)
    b = _cand(2, "B", collection_id=263)
    c = _cand(3, "C")
    out = rank(_query(), [a, b, c], sim_floor=-2.0, limit=3)
    assert sum(1 for c2 in out if c2.payload.get("collection_id") == 263) == 1


def test_collection_and_director_caps_compose():
    """Both caps hold at once: same collection AND same director."""
    cands = [_cand(i, "Nolan", collection_id=263) for i in range(4)]
    cands += [_cand(100 + i, "Nolan") for i in range(4)]
    out = rank(_query(), cands, sim_floor=-2.0, limit=8)
    assert sum(1 for c in out if c.payload.get("collection_id") == 263) == 1
    assert sum(1 for c in out if c.payload["director"] == "Nolan") == DIRECTOR_CAP
