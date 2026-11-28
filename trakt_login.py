"""Get a Trakt access token via the OAuth device flow — one command, no copy-paste.

    python trakt_login.py --client-id <id> --client-secret <secret>

Prints the user_code and verification URL, waits while you approve in a browser,
then prints ONLY the access token on its final line (so you can copy it straight
into Koyeb). Nothing is written to disk — your secret never lands in a file.

Trakt's device-endpoint answers with 400 + a JSON body whose `error` field is the
real status ("authorization_pending", "slow_down", ...). Treating 400 as failure —
as the naive Invoke-RestMethod snippet does — is the usual reason this step appears
broken, so this script reads the body instead.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

API = "https://api.trakt.tv"
UA = {"User-Agent": "MovieMatch-setup/1.0", "Content-Type": "application/json"}

PENDING = {"authorization_pending", "slow_down", "expired", "invalid"}
FATAL = {
    "access_denied": "you clicked 'no' / closed the page — run the script again",
    "invalid_client": "client_id or client_secret is wrong (check the Trakt app page)",
    "unsupported_grant_type": "device flow is not enabled for this app",
}


def post(path: str, payload: dict[str, str], client_id: str) -> tuple[int, dict]:
    body = json.dumps(payload).encode()
    request = urllib.request.Request(
        f"{API}{path}", data=body, method="POST",
        headers={**UA, "trakt-api-version": "2", "trakt-api-key": client_id},
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            return response.status, json.loads(response.read().decode() or "{}")
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode() or "{}"
        try:
            return exc.code, json.loads(raw)
        except json.JSONDecodeError:
            return exc.code, {"error": raw[:200]}


def main() -> int:
    parser = argparse.ArgumentParser(description="Trakt device-flow login")
    parser.add_argument("--client-id", required=True)
    parser.add_argument("--client-secret", required=True)
    parser.add_argument("--timeout", type=int, default=300,
                        help="seconds to wait for approval (default 300)")
    args = parser.parse_args()

    print("1. Requesting a device code ...", flush=True)
    status, data = post("/oauth/device/code", {"client_id": args.client_id}, args.client_id)
    if status != 200 or "device_code" not in data:
        print(f"   FAILED ({status}): {json.dumps(data)[:300]}")
        return 1

    code = data["device_code"]
    user_code = data["user_code"]
    verify_url = data.get("verification_url") or "https://trakt.tv/activate"
    interval = int(data.get("interval") or 5)
    expires_in = int(data.get("expires_in") or 600)

    print()
    print("=" * 62)
    print(f"   CODE TO ENTER : {user_code}")
    print(f"   OPEN THIS URL : {verify_url}")
    print("=" * 62)
    print("   -> log in to Trakt if asked, enter the code, click Approve.")
    print("   -> leave this terminal running; it finishes on its own.")
    print()

    deadline = time.monotonic() + min(args.timeout, expires_in)
    attempt = 0
    while time.monotonic() < deadline:
        time.sleep(interval)
        attempt += 1
        status, data = post(
            "/oauth/device/token",
            {"code": code, "client_id": args.client_id, "client_secret": args.client_secret},
            args.client_id,
        )
        if status == 200 and "access_token" in data:
            print("\n   APPROVED.")
            print(f"   expires_in ~{data.get('expires_in', 0) // 3600} hours")
            print(f"   scope: {data.get('scope', '')}")
            if data.get("refresh_token"):
                print("   (the refresh token below is SINGLE-USE and rotates on "
                      "every refresh; keep it private)")
            print("\n--- TRAKT_ACCESS_TOKEN ---")
            print(data["access_token"])
            print("--- END ---")
            if data.get("refresh_token"):
                print("\n--- TRAKT_REFRESH_TOKEN ---")
                print(data["refresh_token"])
                print("--- END ---")
            print("\nSet all three on the service, or the token dies in ~7 days:")
            print("   TRAKT_ACCESS_TOKEN=<above>")
            if data.get("refresh_token"):
                print("   TRAKT_REFRESH_TOKEN=<above>")
            else:
                print("   TRAKT_REFRESH_TOKEN=<none returned: refresh will not work>")
            print("   TRAKT_REDIRECT_URI=<the URI registered on your Trakt app>")
            print("   TRAKT_CLIENT_SECRET=<your app's secret>")
            print("Refreshing needs the redirect URI to match the app EXACTLY, "
                  "and TRAKT_REFRESH_TOKEN is required for auto-refresh.")
            if not data.get("refresh_token"):
                print("\nWARNING: no refresh_token in the response. Trakt only "
                      "returns one when your app has the right scopes; the "
                      "addon will stop working when the access token expires.")
            print(json.dumps({"expires_in": data.get("expires_in"),
                              "token_type": data.get("token_type")}, indent=1))
            return 0

        error = str(data.get("error", "")).strip()
        if status == 400 and error == "authorization_pending":
            print(f"   waiting for approval ... ({attempt * interval}s)", flush=True)
            continue
        if status == 400 and error == "slow_down":
            interval += 1
            print(f"   slowing down to {interval}s", flush=True)
            continue
        if status == 400 and error == "expired_token":
            print("   the code expired before approval; run the script again")
            return 1
        if status == 429:
            print("   rate limited by Trakt; waiting 10s")
            time.sleep(10)
            continue
        print(f"   FAILED ({status}): {json.dumps(data)[:300]}")
        if error in FATAL:
            print(f"   -> {FATAL[error]}")
        return 1

    print("   timed out waiting for approval; run the script again")
    return 1


if __name__ == "__main__":
    sys.exit(main())
