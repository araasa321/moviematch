"""Stremio protocol helpers (PLAN §10): manifest, catalog ids, extra-string parsing."""

from __future__ import annotations

import json
import re
from typing import Any
from urllib.parse import unquote_plus

ADDON_ID = "community.moviematch.personal"
VERSION = "0.1.0"
PAGE_SIZE = 15
MAX_RANKED = 200

# catalog id -> (display name, dynamic "because" name?)
CATALOGS: dict[str, tuple[str, bool]] = {
    "foryou": ("🎯 For You", False),
    "because": ("🎬 Because you watched {seed}", True),
    "new4you": ("🆕 New for Your Taste", False),
}
# M6 adds tonight / trending / gems (PLAN §10.3)

GENRE_OPTIONS = ["Action", "Comedy", "Drama", "Sci-Fi", "Thriller", "Horror", "Romance", "Animation"]

_EXTRA_RE = re.compile(r"^(?P<key>[a-zA-Z0-9_.]+)=(?P<value>.*)$")


def parse_extra(raw: str | None) -> dict[str, str]:
    """`skip=100&genre=Sci-Fi` -> {"skip": "100", "genre": "Sci-Fi"}.

    Stremio sends this in the URL path (e.g. `foryou/skip=100&genre=Sci-Fi.json`),
    so values may be percent-encoded and `+` may mean a space.
    """
    if not raw:
        return {}
    out: dict[str, str] = {}
    for part in raw.strip("/").split("&"):
        match = _EXTRA_RE.match(part)
        if not match:
            continue
        out[match.group("key")] = unquote_plus(match.group("value")).strip()
    return out


def safe_skip(extra: dict[str, str]) -> int:
    try:
        return max(0, min(MAX_RANKED, int(extra.get("skip", 0))))
    except (TypeError, ValueError):
        return 0


def build_manifest(base_url: str, config_blob: str, seeds: dict[str, str] | None = None) -> dict[str, Any]:
    """Manifest with rows for both types; `because` names are filled from Trakt (§10.2)."""
    seeds = seeds or {}
    catalogs: list[dict[str, Any]] = []
    for catalog_id, (label, dynamic) in CATALOGS.items():
        for media_type in ("movie", "series"):
            name = label.format(seed=seeds.get(media_type, "—")) if dynamic else label
            if media_type == "series" and not dynamic:
                name = f"{name} (Series)"
            entry: dict[str, Any] = {"type": media_type, "id": catalog_id, "name": name}
            extra: list[dict[str, Any]] = [{"name": "skip"}]
            if catalog_id == "foryou" and media_type == "movie":
                extra.append({"name": "genre", "options": GENRE_OPTIONS, "isRequired": False})
            entry["extra"] = extra
            catalogs.append(entry)

    return {
        "id": ADDON_ID,
        "version": VERSION,
        "name": "MovieMatch — For You",
        "description": "Trakt-personalized recommendations. Uses TMDB.",
        "logo": f"{base_url}/static/logo.png",
        "resources": ["catalog"],
        "types": ["movie", "series"],
        "idPrefixes": ["tt"],
        "behaviorHints": {"configurable": True, "configurationRequired": True},
        "catalogs": catalogs,
    }


def catalog_response(metas: list[dict[str, Any]], cache_max_age: int = 900,
                     stale_revalidate: int = 3600) -> dict[str, Any]:
    return {"metas": metas, "cacheMaxAge": cache_max_age, "staleRevalidate": stale_revalidate}


def dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
