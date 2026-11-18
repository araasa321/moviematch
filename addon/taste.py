"""Personalization engine (PLAN §8) — numpy only, O(#signals).

Signal weights (§8.1, Trakt 10-scale):
    rating r   -> (r - 6) / 2      (10 -> +2, 8 -> +1, 6 -> 0, 4 -> -1, 2 -> -2)
    watched    -> +0.7             watchlist -> +0.8     rewatch (plays>=2) -> +2.5
    rating<=3 / hidden -> dislike list (penalty only, never subtracted from centroid)

Decay (§8.2):
    taste_stable  = mean(w * 0.5**(age/365))     ratings use a 730-day half-life
    taste_recent  = mean(w * 0.5**(age/60))
    taste_long    = normalize(0.6*stable + 0.4*recent)
    taste_session = mean of actions in the last 72h, then
    query_vec     = normalize(0.7*taste_long + 0.3*taste_session)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Iterable, Sequence

import numpy as np

from .rank import normalize

WATCHED = 0.7
WATCHLIST = 0.8
REWATCH = 2.5
DISLIKE_MAX_RATING = 3
STABLE_HALFLIFE_DAYS = 365.0
RATING_HALFLIFE_DAYS = 730.0
RECENT_HALFLIFE_DAYS = 60.0
SESSION_WINDOW_HOURS = 72.0
SESSION_MAX_ACTIONS = 3
LONG_STABLE_MIX = 0.6
QUERY_MIX = 0.7
MIN_POSITIVE_SIGNALS = 3      # below this -> cold start (§8.4)
MIN_SESSION_NORM = 0.4        # reject near-zero session vectors (unknown vectors)


def rating_weight(rating: int) -> float:
    """(r - 6) / 2 on Trakt's 1-10 scale."""
    return (float(rating) - 6.0) / 2.0


def decay(age_days: float, half_life: float = STABLE_HALFLIFE_DAYS) -> float:
    return 0.5 ** (max(0.0, age_days) / half_life)


@dataclass
class Signal:
    """One Trakt action joined to its catalog vector."""
    vector: np.ndarray
    weight: float
    at: datetime
    kind: str = "watched"         # watched | rating | watchlist | rewatch | seed


@dataclass
class Taste:
    long: np.ndarray | None = None
    session: np.ndarray | None = None
    query: np.ndarray | None = None
    dislikes: np.ndarray | None = None
    positive_count: int = 0
    ready: bool = False
    extra: dict = field(default_factory=dict)


def _utcnow(now: datetime | None) -> datetime:
    if now is None:
        return datetime.now(timezone.utc)
    return now if now.tzinfo else now.replace(tzinfo=timezone.utc)


def compute_taste(signals: Sequence[Signal], dislike_signals: Sequence[Signal] = (),
                  now: datetime | None = None, dim: int = 384) -> Taste:
    now = _utcnow(now)
    dim = _infer_dim(signals, dislike_signals, fallback=dim)
    stable_num = np.zeros(dim, dtype=np.float64)
    stable_den = 0.0
    recent_num = np.zeros(dim, dtype=np.float64)
    recent_den = 0.0
    session_num = np.zeros(dim, dtype=np.float64)
    session_den = 0.0
    session_cut = now.timestamp() - SESSION_WINDOW_HOURS * 3600
    session_actions: list[tuple[float, np.ndarray, float]] = []
    positives = 0
    for sig in signals:
        if sig.weight == 0.0:
            continue
        vec = normalize(sig.vector)
        if not np.any(vec):
            continue
        age_days = (now - sig.at).total_seconds() / 86400.0
        half_life = RATING_HALFLIFE_DAYS if sig.kind == "rating" else STABLE_HALFLIFE_DAYS
        if sig.weight > 0:
            positives += 1
            ds = sig.weight * decay(age_days, half_life)
            dr = sig.weight * decay(age_days, RECENT_HALFLIFE_DAYS)
            stable_num += ds * vec
            stable_den += ds
            recent_num += dr * vec
            recent_den += dr
            if sig.at.timestamp() >= session_cut:
                session_actions.append((sig.at.timestamp(), vec, sig.weight))
        # Negative signals (rating <= 3) are deliberately absent from both centroids:
        # §8.1 says dislikes live in the penalty list only. They arrive via
        # `dislike_signals` and are used by rank.dislike_penalty().

    taste = Taste(positive_count=positives)
    taste.dislikes = _stack(dislike_signals, dim)
    if stable_den <= 0.0:
        return taste

    stable = stable_num / stable_den
    recent = recent_num / recent_den if recent_den > 0 else np.zeros(dim, dtype=np.float64)
    long_vec = normalize(LONG_STABLE_MIX * stable + (1.0 - LONG_STABLE_MIX) * recent)
    if not np.any(long_vec):
        return taste
    taste.long = long_vec

    session_actions.sort(key=lambda t: t[0], reverse=True)
    for _, vec, weight in session_actions[:SESSION_MAX_ACTIONS]:
        session_num += max(weight, 0.1) * vec
        session_den += max(weight, 0.1)
    if session_den > 0:
        taste.session = normalize(session_num / session_den)

    query = long_vec
    if taste.session is not None:
        blend = normalize(QUERY_MIX * long_vec + (1.0 - QUERY_MIX) * taste.session)
        if np.any(blend):
            query = blend
    taste.query = query
    taste.ready = positives >= MIN_POSITIVE_SIGNALS
    return taste


def _infer_dim(signals: Sequence[Signal], dislikes: Sequence[Signal],
               fallback: int = 384) -> int:
    """Derive vector width from the data.

    The production model is 384-d, but hardcoding it meant any other store
    (tests, a future bge swap) crashed with a broadcast error deep in the maths.
    """
    for group in (signals, dislikes):
        for sig in group:
            width = int(np.asarray(sig.vector).shape[-1]) if np.ndim(sig.vector) else 0
            if width:
                return width
    return fallback


def _stack(signals: Iterable[Signal], dim: int) -> np.ndarray | None:
    rows = [normalize(s.vector) for s in signals if np.any(s.vector)]
    return np.vstack(rows) if rows else None


def median_year(years: Sequence[int | None]) -> int | None:
    """era seed for For You (§9.3): median release year of the user's positive items."""
    vals = sorted(int(y) for y in years if isinstance(y, int) and y > 1900)
    if not vals:
        return None
    return int(np.median(vals))
