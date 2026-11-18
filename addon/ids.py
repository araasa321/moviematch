"""Point id scheme (PLAN §4.1).

Qdrant accepts uint64 or UUID point ids ONLY — never strings like "movie:155".
Every title is addressed by a deterministic uuid5 so ingestion, Trakt joins and
tests all agree on the same id without any lookup table.
"""

from __future__ import annotations

import uuid
from typing import Any

NAMESPACE = uuid.NAMESPACE_URL


def point_id(media_type: str, tmdb_id: int) -> str:
    """`uuid5(NAMESPACE_URL, f"{media_type}:{tmdb_id}")` rendered as a plain string."""
    return str(uuid.uuid5(NAMESPACE, f"{media_type}:{int(tmdb_id)}"))


def id_from_trakt_item(item: dict[str, Any], fallback_type: str) -> tuple[str, int] | None:
    """Trakt sync items wrap the object in `movie` / `show`; return (media_type, tmdb_id).

    Returns None when Trakt has no tmdb id (the nightly job self-heals those, PLAN §7.3).
    """
    for key in ("movie", "show"):
        obj = item.get(key)
        if isinstance(obj, dict):
            ids = obj.get("ids") or {}
            tmdb = ids.get("tmdb")
            if tmdb:
                return ("series" if key == "show" else "movie"), int(tmdb)
            return None
    ids = item.get("ids") or {}
    tmdb = ids.get("tmdb")
    return (fallback_type, int(tmdb)) if tmdb else None
