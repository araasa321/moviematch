"""Fetch the live manifest exactly as a Stremio client would.

Differences from live_check.py that matter to a client:
  * follows redirects? (Stremio does not always)
  * no custom User-Agent (some servers behave differently)
  * checks the response is valid JSON *before* reporting success
  * also tries the /manifest.json redirect path and trailing-slash variants
"""

from __future__ import annotations

import json
import re
import socket
import urllib.error
import urllib.request

BASE = "https://icy-ema-gyutfvu-d23fba6f.koyeb.app"
UA_STREMIO = "Stremio/4.4.168 (WindowsNT10.0;x64)"


def fetch(url: str, ua: str | None, follow: bool = True) -> tuple[int | str, dict, float]:
    import time

    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *a, **k):
            return None

    handlers = [] if follow else [NoRedirect()]
    opener = urllib.request.build_opener(*handlers)
    req = urllib.request.Request(url, headers={"User-Agent": ua} if ua else {})
    started = time.monotonic()
    try:
        with opener.open(req, timeout=60) as resp:
            return resp.status, dict(resp.headers), time.monotonic() - started
    except urllib.error.HTTPError as exc:
        return exc.code, dict(exc.headers), time.monotonic() - started
    except Exception as exc:  # noqa: BLE001
        return f"{type(exc).__name__}: {exc}", {}, time.monotonic() - started


def get_blob() -> str:
    with urllib.request.urlopen(f"{BASE}/configure", timeout=60) as resp:
        page = resp.read().decode()
    return re.search(r"/(eyJ[A-Za-z0-9_\-]+)/manifest\.json", page).group(1)


def main() -> int:
    blob = get_blob()
    https_url = f"{BASE}/{blob}/manifest.json"
    print(f"https url length: {len(https_url)}")
    print(f"https url        : {https_url}\n")

    print("--- real fetch, Stremio UA, JSON validated ---")
    for label, ua in (("stremio UA", UA_STREMIO), ("no UA", None)):
        status, headers, took = fetch(https_url, ua)
        print(f"{label:<12} status={status}  {took:.2f}s  "
              f"ctype={headers.get('Content-Type') or headers.get('content-type')}")
        if isinstance(status, int) and status == 200:
            req = urllib.request.Request(https_url, headers={"User-Agent": ua} if ua else {})
            with urllib.request.urlopen(req, timeout=60) as resp:
                raw = resp.read()
            try:
                manifest = json.loads(raw)
                print(f"             JSON ok: id={manifest['id']} "
                      f"catalogs={len(manifest['catalogs'])}")
            except Exception as exc:  # noqa: BLE001
                print(f"             JSON INVALID: {exc}; first bytes {raw[:40]!r}")

    print("\n--- redirect behaviour ---")
    for path, label in ((f"/{blob}/manifest.json", "config manifest"),
                        ("/manifest.json", "bare manifest")):
        status, headers, _ = fetch(f"{BASE}{path}", UA_STREMIO, follow=False)
        loc = headers.get("Location") or headers.get("location")
        print(f"{label:<16} status={status}  location={loc}")

    print("\n--- trailing-slash + case variants a client may try ---")
    for suffix in ("", "/"):
        status, _, _ = fetch(f"{BASE}/{blob}/manifest.json{suffix}", UA_STREMIO)
        print(f"manifest.json{suffix!r:<3} -> {status}")

    print("\n--- DNS/TLS sanity from this machine ---")
    try:
        host = BASE.split("//")[1]
        print("resolves to:", socket.gethostbyname(host))
    except Exception as exc:  # noqa: BLE001
        print("DNS failed:", exc)

    print("\n--- what the stremio:// form looks like ---")
    print(f"stremio://{https_url[len('https://'):]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
