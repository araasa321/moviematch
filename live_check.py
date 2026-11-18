"""Hit the live Koyeb deployment the way Stremio would.

`Invoke-WebRequest` cannot complete TLS to this host on this machine, so this uses
Python's urllib. Prints status + a short body so a failing manifest is obvious.
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request

BASE = "https://icy-ema-gyutfvu-d23fba6f.koyeb.app"
# clean (uncorrupted) config for the secret reported by /health
BLOB = ("eyJzZWNyZXQiOiJhNjkyNjA2Y2NiZDhhNmUxYWEyNDI3YmNiNzc5MTg1N2QyOGQy"
        "MWJmNjg5Y2IzODM3YzQ1MmRiYWJkZDJlMCIsIm1vZGUiOiJibG9ja2J1c3RlciIs"
        "ImxhbmdzIjpbImVuIl19")


def get(path: str, timeout: float = 60.0) -> tuple[int, str, bytes, float]:
    import time

    req = urllib.request.Request(f"{BASE}{path}", headers={"User-Agent": "moviematch-check"})
    started = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.headers.get("content-type", ""), resp.read(), time.monotonic() - started
    except urllib.error.HTTPError as exc:
        return exc.code, exc.headers.get("content-type", ""), exc.read(), time.monotonic() - started
    except Exception as exc:  # noqa: BLE001
        return -1, "", f"{type(exc).__name__}: {exc}".encode(), time.monotonic() - started


def main() -> int:
    print(f"target: {BASE}\n")

    status, _, body, took = get("/health")
    print(f"GET /health                          -> {status}  {took:6.2f}s")
    if status == 200:
        data = json.loads(body)
        print(f"    store={data.get('store')} catalog={data.get('catalog_size')} "
              f"trakt={data.get('trakt')} secret_configured={data.get('secret_configured')}")

    status, _, body, took = get(f"/{BLOB}/manifest.json")
    print(f"GET /{{clean blob}}/manifest.json      -> {status}  {took:6.2f}s")
    if status == 200:
        manifest = json.loads(body)
        print(f"    id={manifest.get('id')} catalogs={len(manifest.get('catalogs', []))}")
        for cat in manifest.get("catalogs", []):
            print(f"      {cat['type']:<7} {cat['id']:<9} {cat['name']}")
    else:
        print(f"    body: {body[:200]!r}")

    for row in ("foryou", "because"):
        # first call pays ranking cost; second should hit the row cache
        status, _, body, took = get(f"/{BLOB}/catalog/movie/{row}.json")
        print(f"GET catalog/movie/{row:<8}          -> {status}  {took:6.2f}s", end="")
        if status == 200:
            metas = json.loads(body).get("metas", [])
            print(f"  {len(metas)} metas", end="")
            if metas:
                print(f"  first: {metas[0].get('name')!r}")
            else:
                print()
            status2, _, body2, took2 = get(f"/{BLOB}/catalog/movie/{row}.json")
            cached = len(json.loads(body2).get("metas", [])) if status2 == 200 else "?"
            print(f"    repeat -> {status2}  {took2:6.2f}s  {cached} metas "
                  f"(cached: {'yes' if took2 < took / 2 else 'NO'})")
        else:
            print(f"  body: {body[:160]!r}")

    status, _, body, took = get(f"/{BLOB}/catalog/series/foryou.json")
    print(f"GET catalog/series/foryou           -> {status}  {took:6.2f}s", end="")
    if status == 200:
        print(f"  {len(json.loads(body).get('metas', []))} metas")
    else:
        print(f"  body: {body[:160]!r}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
