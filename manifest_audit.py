"""Validate the live manifest the way a Stremio client will read it.

Stremio rejects manifests with a 200 status for reasons that never show up in the
server log: missing required fields, a `logo` URL that 404s, catalog rows whose
`extra` shapes are wrong, or a `behaviorHints.configurationRequired` that sends the
client back to a configure page it cannot open.

    python manifest_audit.py
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request

BASE = "https://icy-ema-gyutfvu-d23fba6f.koyeb.app"


def fetch(path: str, timeout: float = 60.0) -> tuple[int, bytes]:
    req = urllib.request.Request(f"{BASE}{path}", headers={"User-Agent": "moviematch-audit"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def discover_blob() -> str:
    import re

    status, body = fetch("/configure")
    match = re.search(r"/(eyJ[A-Za-z0-9_\-]+)/manifest\.json", body.decode())
    if status != 200 or not match:
        raise SystemExit(f"/configure unusable ({status})")
    return match.group(1)


REQUIRED = ("id", "version", "name", "resources", "types", "catalogs")


def main() -> int:
    blob = discover_blob()
    status, body = fetch(f"/{blob}/manifest.json")
    print(f"manifest status: {status}\n")
    if status != 200:
        print(body[:300])
        return 1
    manifest = json.loads(body)
    problems: list[str] = []

    for key in REQUIRED:
        if key not in manifest:
            problems.append(f"missing required key: {key}")
    print(f"id       : {manifest.get('id')}")
    print(f"version  : {manifest.get('version')}")
    print(f"name     : {manifest.get('name')}")
    print(f"resources: {manifest.get('resources')}")
    print(f"types    : {manifest.get('types')}")
    print(f"idPrefixes: {manifest.get('idPrefixes')}")
    print(f"behaviourHints: {manifest.get('behaviorHints')}")

    hints = manifest.get("behaviorHints") or {}
    if hints.get("configurationRequired"):
        problems.append(
            "behaviorHints.configurationRequired=true tells some clients to open the "
            "configure page instead of installing; the URL-based install may be refused")

    logo = manifest.get("logo")
    if logo:
        lstatus, _ = fetch(logo.replace(BASE, ""), timeout=20)
        print(f"logo     : {logo} -> {lstatus}")
        if lstatus != 200:
            problems.append(f"logo returns {lstatus}; some clients refuse a manifest "
                            f"whose logo cannot load")
    else:
        print("logo     : (none)")

    for resource in manifest.get("resources") or []:
        if isinstance(resource, dict):
            print(f"  resource object: {resource}")

    print("\ncatalogs:")
    for cat in manifest.get("catalogs") or []:
        extra = cat.get("extra") or []
        shape = ", ".join((e.get("name") or "?") for e in extra) or "-"
        print(f"  {cat.get('type'):<7} {cat.get('id'):<9} extras[{shape}]  {cat.get('name')}")
        for key in ("type", "id", "name"):
            if not cat.get(key):
                problems.append(f"catalog missing {key}: {cat}")
        if not isinstance(extra, list):
            problems.append(f"catalog {cat.get('type')}/{cat.get('id')} extra is not a list")
        for item in extra:
            if not isinstance(item, dict) or "name" not in item:
                problems.append(f"catalog {cat.get('type')}/{cat.get('id')} bad extra: {item}")
            elif item.get("name") == "genre" and not item.get("options"):
                problems.append("genre extra without options")

    # a real catalog request must return the documented envelope
    print("\ncatalog probe (movie/foryou):")
    status, body = fetch(f"/{blob}/catalog/movie/foryou.json")
    print(f"  status {status}")
    if status == 200:
        payload = json.loads(body)
        metas = payload.get("metas") or []
        print(f"  metas {len(metas)}  cacheMaxAge {payload.get('cacheMaxAge')}")
        bad = [m for m in metas if not str(m.get("id", "")).startswith("tt")]
        if bad:
            problems.append(f"{len(bad)} metas without tt ids")
        noposter = [m for m in metas if not m.get("poster")]
        if noposter:
            problems.append(f"{len(noposter)} metas without poster")

    print()
    if problems:
        print("PROBLEMS:")
        for item in problems:
            print("  -", item)
        return 1
    print("manifest audit passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
