"""Verify a Trakt access token before you deploy it.

    python trakt_check.py --client-id <id> --token <access_token>

Checks the token against /users/settings, then counts what the addon will
actually read: watched, ratings, watchlist. If the counts are 0, the token works
but there is nothing to personalise with yet.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request

API = "https://api.trakt.tv"


def get(path: str, client_id: str, token: str) -> tuple[int, object]:
    request = urllib.request.Request(
        f"{API}{path}",
        headers={"User-Agent": "MovieMatch-check/1.0", "Content-Type": "application/json",
                 "trakt-api-version": "2", "trakt-api-key": client_id,
                 "Authorization": f"Bearer {token}"},
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            return response.status, json.loads(response.read().decode() or "null")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode()[:300]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--client-id", required=True)
    parser.add_argument("--token", required=True)
    args = parser.parse_args()

    status, settings = get("/users/settings", args.client_id, args.token)
    if status == 401:
        print("FAIL: token rejected (401). It expired, was revoked, or is mistyped.")
        print("      Get a fresh one:  python trakt_login.py --client-id ... --client-secret ...")
        return 1
    if status != 200:
        print(f"FAIL: /users/settings -> {status}: {str(settings)[:200]}")
        return 1

    user = (settings or {}).get("user", {}) if isinstance(settings, dict) else {}
    print(f"OK  token valid for @{user.get('username', '?')} "
          f"({'VIP' if user.get('vip') else 'free'})")

    totals = {}
    for label, path in (("watched movies", "/sync/history/movies?limit=1"),
                        ("watched shows", "/sync/history/shows?limit=1"),
                        ("rated movies", "/sync/ratings/movies"),
                        ("watchlist movies", "/sync/watchlist/movies")):
        code, data = get(path, args.client_id, args.token)
        if isinstance(data, list):
            totals[label] = len(data)
        else:
            totals[label] = f"error {code}"
    print("counts:", json.dumps(totals))
    if all(v == 0 for v in totals.values() if isinstance(v, int)):
        print("NOTE: no history yet — rows will fall back to cold start.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
