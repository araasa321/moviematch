"""Stremio protocol helpers (PLAN §10): manifest, catalog ids, extra-string parsing."""

from __future__ import annotations

import json
import re
from typing import Any
from urllib.parse import unquote_plus

from .config import LANG_NAMES

ADDON_ID = "community.moviematch.personal"
VERSION = "0.1.0"
PAGE_SIZE = 15
MAX_RANKED = 200

# catalog id -> (display name, dynamic "because" name?)
#
# Trakt drives these rows now: `foryou` is /recommendations/{movies|shows} and
# `because` is /{movies|shows}/:id/related for the most recent watch. Trakt's
# order is preserved, so there is nothing left to rank.
#
# `new4you` was removed with the ranker — it never had an implementation and its
# intent (fresh titles for your taste) is what /recommendations already does.
CATALOGS: dict[str, tuple[str, bool]] = {
    "foryou": ("🎯 For You", False),
    "because": ("🎬 Because you watched {seed}", True),
}

GENRE_OPTIONS = ["Action", "Comedy", "Drama", "Sci-Fi", "Thriller", "Horror", "Romance", "Animation"]

# A per-language catalog id looks like "foryou@ja". Keeping the base id as a
# prefix means the router and the row builder can split on the last "@" and
# still find the real row, and ids stay stable for a given language.
LANG_SEPARATOR = "@"


def split_row_id(catalog_id: str) -> tuple[str, str | None]:
    """`"foryou@ja"` -> `("foryou", "ja")`; `"foryou"` -> `("foryou", None)`."""
    base, sep, lang = catalog_id.partition(LANG_SEPARATOR)
    return base, (lang or None) if sep else None

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


def build_manifest(base_url: str, config_blob: str, seeds: dict[str, str] | None = None,
                   *, langs: Sequence[str] = ("en",),
                   layout: str = "merged") -> dict[str, Any]:
    """Manifest with rows for both types; `because` names are filled from Trakt (§10.2).

    `layout` decides how the chosen languages are presented:

    * ``merged``   — one row per catalog id, pooling every language. The ranking
      decides what wins, so a great Korean film can outrank a weak English one.
    * ``separate`` — the same rows repeated once per language (``foryou@ja``),
      each row filtered to that language.

    A single language always renders as ``merged``: one language's "separate" row
    set would be byte-for-byte the merged set under a uglier name.
    """
    seeds = seeds or {}
    lang_list = tuple(langs or ("en",))
    per_lang = layout == "separate" and len(lang_list) > 1
    catalogs: list[dict[str, Any]] = []

    def add(catalog_id: str, label: str, dynamic: bool, lang: str | None) -> None:
        for media_type in ("movie", "series"):
            name = label.format(seed=seeds.get(media_type, "—")) if dynamic else label
            if media_type == "series" and not dynamic:
                name = f"{name} (Series)"
            if lang and per_lang:
                tag = LANG_NAMES.get(lang, lang.upper())
                name = f"{name} · {tag}"
            entry: dict[str, Any] = {"type": media_type, "id": catalog_id, "name": name}
            extra: list[dict[str, Any]] = [{"name": "skip"}]
            # Genre filtering applies to every ranked row (For You / Because /
            # New) — the handler honours `genre=` regardless of catalog id, so
            # only offering it on `foryou`+`movie` hid a working feature.
            extra.append({"name": "genre", "options": GENRE_OPTIONS, "isRequired": False})
            entry["extra"] = extra
            catalogs.append(entry)

    for catalog_id, (label, dynamic) in CATALOGS.items():
        if per_lang:
            for lang in lang_list:
                add(f"{catalog_id}{LANG_SEPARATOR}{lang}", label, dynamic, lang)
        else:
            add(catalog_id, label, dynamic, None)

    return {
        "id": ADDON_ID,
        "version": VERSION,
        "name": "MovieMatch — For You",
        "description": "Trakt-personalized recommendations. Uses TMDB.",
        "logo": f"{base_url}/static/logo.png",
        # `catalog` serves rows; `library` is an EVENT resource — Stremio posts the
        # user's mark-watched / add-to-list actions here so we can mirror them to
        # Trakt, which its own integration does not do (stremio-features#1305).
        "resources": [
            "catalog",
            {"name": "library", "types": ["movie", "series"], "idPrefixes": ["tt"]},
        ],
        "types": ["movie", "series"],
        "idPrefixes": ["tt"],
        # `configurationRequired` must stay False: the config is already baked into
        # this URL. Setting it True made clients treat the addon as unconfigured and
        # refuse to install. `configurable: True` still offers the setup page.
        "behaviorHints": {"configurable": True, "configurationRequired": False},
        "catalogs": catalogs,
    }


def catalog_response(metas: list[dict[str, Any]], cache_max_age: int = 900,
                     stale_revalidate: int = 3600) -> dict[str, Any]:
    return {"metas": metas, "cacheMaxAge": cache_max_age, "staleRevalidate": stale_revalidate}


def dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
