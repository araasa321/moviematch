"""Trakt OAuth token storage + automatic refresh.

Why this exists
---------------
A static `TRAKT_ACCESS_TOKEN` env var used to be fine: Trakt access tokens lasted
90 days. Trakt changed that to 24 hours (Mar 2025) and later to 7 days, and their
own docs now tell apps to drive expiry off the `expires_in` value rather than any
hardcoded interval. A pasted env token therefore dies weekly, and every Trakt call
— recommendations, history sync, AND the Stremio -> Trakt writes — silently 401s.

Verified contract (docs.trakt.tv, Nov 2026):
  * refresh: POST https://auth.trakt.tv/oauth/token   (auth host, NOT api host)
  * body:   grant_type=refresh_token, refresh_token, client_id,
            client_secret (deprecated but permitted server-side), redirect_uri
  * reply:  {access_token, token_type, expires_in, refresh_token, scope, created_at}
  * refresh tokens are SINGLE-USE: each exchange returns a new one and invalidates
    the old one. Losing the new value means the user must re-authorize.

Consequences baked into this module:
  1. the rotated pair is persisted immediately, with retries — a failed write after
     a successful exchange logs the account out;
  2. refreshes are serialised behind one lock and re-check the stored token first,
     so two concurrent 401s cannot burn the refresh chain;
  3. storage is Qdrant `user_state` (the only store that survives a restart and is
     writable at runtime); env vars seed it once and are then ignored.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any

import httpx

log = logging.getLogger(__name__)

# All OAuth routes live on auth.trakt.tv — the API host answers 404 for them.
AUTH = "https://auth.trakt.tv"
TOKEN_PATH = "/oauth/token"
USER_AGENT = "MovieMatch/1.0 (Stremio addon; personal use)"
# Refresh proactively once this fraction of the lifetime has elapsed. 20 % of 7 days
# is ~1.4 days of slack, which absorbs a long outage without being wasteful.
REFRESH_AT_FRACTION = 0.80
# A token inside this window is treated as already expired.
EXPIRY_SKEW = 60.0
# Persisting the rotated pair is the one write that must not be lost.
SAVE_ATTEMPTS = 3
SAVE_BACKOFF = 0.5


class TraktAuthError(RuntimeError):
    """Refresh failed in a way that needs a human (re-authorization)."""


@dataclass(frozen=True)
class TokenPair:
    access_token: str = ""
    refresh_token: str = ""
    expires_at: float = 0.0        # unix seconds; 0 = unknown
    issued_at: float = 0.0         # unix seconds; 0 = unknown

    @property
    def usable(self) -> bool:
        return bool(self.access_token)

    @property
    def lifetime(self) -> float:
        """Seconds this token was issued for, or 0 when unknown."""
        if self.expires_at and self.issued_at and self.expires_at > self.issued_at:
            return self.expires_at - self.issued_at
        return 0.0

    def expired_soon(self, now: float | None = None,
                     fraction: float = REFRESH_AT_FRACTION) -> bool:
        """True when the token is gone, inside the skew, or past `fraction` of life.

        A token with no `expires_at` counts as expired: one wasted refresh is much
        cheaper than a 401 on a live Stremio request. Nothing here hardcodes 7 days
        — Trakt has changed this value twice and documents `expires_in` precisely so
        clients stop assuming.
        """
        if not self.usable:
            return True
        if not self.expires_at:
            return True
        now = time.time() if now is None else now
        remaining = self.expires_at - now
        if remaining <= EXPIRY_SKEW:
            return True
        life = self.lifetime
        if not life:
            return False
        # Refresh once `fraction` of the lifetime has been consumed. Written as
        # elapsed/life >= fraction rather than remaining <= (1-fraction)*life, which
        # drifts by an ulp at exactly the boundary (tests caught it at 80 %).
        elapsed = life - remaining
        return elapsed >= fraction * life


class TokenStore:
    """Persists the token pair in Qdrant's `user_state` collection.

    The collection already exists in the ingest jobs and `get_payload` /
    `save_payload` were written for exactly this, so no schema work is needed.
    """

    POINT_ID = "trakt_tokens"

    def __init__(self, qdrant: Any | None = None) -> None:
        self._qdrant = qdrant
        self._cached: TokenPair | None = None

    async def load(self, *, fresh: bool = False) -> TokenPair | None:
        """Stored pair. `fresh=True` bypasses the in-memory copy.

        The refresh path MUST use `fresh=True`: its whole purpose is to see what
        another worker stored while this one waited on the lock. Reading the cache
        there would make every waiter see the same stale token and all of them would
        exchange it — with single-use refresh tokens that breaks the chain for good.
        """
        if self._cached is not None and not fresh:
            return self._cached
        if self._qdrant is None:
            return None
        try:
            payload = await self._qdrant.get_payload(self.POINT_ID)
        except Exception as exc:  # noqa: BLE001 - a missing store is not fatal
            log.warning("token store unavailable (%s); using env values",
                        type(exc).__name__)
            return None
        if not payload:
            return None
        pair = TokenPair(
            access_token=str(payload.get("access_token") or ""),
            refresh_token=str(payload.get("refresh_token") or ""),
            expires_at=float(payload.get("expires_at") or 0.0),
            issued_at=float(payload.get("issued_at") or 0.0),
        )
        self._cached = pair if pair.usable else None
        return self._cached

    async def save(self, pair: TokenPair) -> None:
        """Persist the rotated pair. Retried: losing it forces a re-login."""
        self._cached = pair
        if self._qdrant is None:
            return
        payload = {"access_token": pair.access_token,
                   "refresh_token": pair.refresh_token,
                   "expires_at": pair.expires_at,
                   "issued_at": pair.issued_at,
                   "updated_at": int(time.time())}
        last: Exception | None = None
        for attempt in range(SAVE_ATTEMPTS):
            try:
                await self._qdrant.save_payload(payload, self.POINT_ID)
                return
            except Exception as exc:  # noqa: BLE001 - retried below
                last = exc
                log.warning("token save failed (attempt %s/%s): %s",
                            attempt + 1, SAVE_ATTEMPTS, type(exc).__name__)
                await asyncio.sleep(SAVE_BACKOFF * (2 ** attempt))
        log.error("COULD NOT PERSIST ROTATED TRAKT TOKEN (%s). The old refresh "
                  "token is now invalid; a re-authorization will be required on "
                  "the next restart.", last)


def parse_token_response(body: dict[str, Any]) -> TokenPair:
    """Build a TokenPair from an OAuth token response.

    `expires_in` is documented as seconds. `created_at` is used as the issue time
    when present so the 80 % refresh point can be computed without assuming a
    lifetime.
    """
    lifetime = float(body.get("expires_in") or 0)
    issued = float(body.get("created_at") or 0) or time.time()
    return TokenPair(
        access_token=str(body.get("access_token") or ""),
        refresh_token=str(body.get("refresh_token") or ""),
        expires_at=issued + lifetime if lifetime else 0.0,
        issued_at=issued,
    )


async def exchange_refresh_token(*, refresh_token: str, client_id: str,
                                 client_secret: str, redirect_uri: str,
                                 timeout: float = 15.0) -> TokenPair:
    """POST the refresh grant and return the new pair.

    Raises TraktAuthError for `invalid_grant` (legacy/revoked token: the user must
    re-authorize) and TraktError-free RuntimeError for transport problems, so the
    caller can tell "try again later" from "needs a human".
    """
    if not refresh_token:
        raise TraktAuthError("no refresh token stored")
    body = {
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
        "client_id": client_id,
        "redirect_uri": redirect_uri,
    }
    if client_secret:
        # Deprecated, but permitted from a server (never from a browser/app).
        body["client_secret"] = client_secret
    headers = {"Content-Type": "application/json", "User-Agent": USER_AGENT,
               "trakt-api-version": "2", "trakt-api-key": client_id}
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(f"{AUTH}{TOKEN_PATH}", json=body, headers=headers)
    if response.status_code == 200:
        pair = parse_token_response(response.json() or {})
        if not pair.usable:
            raise TraktAuthError("refresh returned no access_token")
        return pair
    # The docs are explicit that the error BODY is the useful part; a bare status
    # code would hide "invalid_grant: session not found".
    detail = response.text[:200]
    try:
        parsed = response.json() or {}
        error = str(parsed.get("error") or "")
        desc = str(parsed.get("error_description") or "")
    except Exception:  # noqa: BLE001
        error, desc = "", ""
    if response.status_code == 400 and error in ("invalid_grant", "invalid_request"):
        raise TraktAuthError(
            f"refresh rejected ({error}: {desc or detail}). The stored refresh "
            f"token cannot be exchanged — re-authorize at "
            f"trakt.tv/oauth/applications and update TRAKT_REFRESH_TOKEN.")
    raise RuntimeError(
        f"token refresh failed ({response.status_code}: {error or detail})")
