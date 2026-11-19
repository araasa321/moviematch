"""Pacing must actually hold the configured request rate.

The bootstrap speed-up raised the default rate from 15 to 30 req/s and page
fetches now overlap. That is only safe if `_pace()` still books every request a
slot at the configured rate — otherwise "faster" quietly becomes "429s and
retry storms", or worse, disrespecting TMDB's ~40 req/s ceiling.

These tests measure real elapsed time, so they are deliberately small.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from jobs.tmdb import TMDBClient


def _elapsed_for(n: int, rate: float) -> float:
    async def run() -> float:
        client = TMDBClient(token="fake", rate=rate)
        client._lock = asyncio.Lock()
        start = time.monotonic()
        await asyncio.gather(*(client._pace() for _ in range(n)))
        return time.monotonic() - start

    return asyncio.run(run())


def test_pace_holds_rate_under_concurrency():
    """20 slots at 50/s must take at least ~19 intervals (~0.38 s)."""
    n, rate = 20, 50.0
    elapsed = _elapsed_for(n, rate)
    expected = (n - 1) / rate
    assert elapsed >= expected * 0.7, (
        f"paced {n} requests at {rate}/s in {elapsed:.3f}s; "
        f"expected at least ~{expected:.3f}s — the limiter is leaking")


def test_pace_slots_are_not_shared_between_callers():
    """Concurrent callers must each get their own slot, not the same one."""
    n, rate = 10, 25.0
    elapsed = _elapsed_for(n, rate)
    # Sharing one slot would finish almost instantly (~0 s); a correct limiter
    # needs (n-1)/rate seconds minimum.
    assert elapsed >= (n - 1) / rate * 0.7


def test_default_rate_stays_under_tmdb_ceiling():
    """TMDB documents a ~40 req/s ceiling; the shipped default must respect it."""
    assert TMDBClient(token="x").rate <= 40.0


def test_interval_matches_rate():
    client = TMDBClient(token="x", rate=30.0)
    assert client._interval == pytest.approx(1 / 30.0)


def test_rate_floor_prevents_division_by_zero():
    client = TMDBClient(token="x", rate=0.0)
    assert client._interval > 0
