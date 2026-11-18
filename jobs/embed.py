"""Embedding text recipe + text hashing (PLAN §6.1), with the truncation fix.

The plan's recipe concatenates title, tagline, genres, keywords, "Directed by X"
(twice), cast and the **full overview**. The embedding test showed two problems
with that ordering:

  1. `all-MiniLM-L6-v2` truncates at 256 word pieces. A long synopsis plus a
     credits block can push the *title* out of the window, so the vector stops
     describing the title at all.
  2. Titles, not plot sentences, are what carry franchise identity (Batman/
     Nolan clustered far more tightly than any genre did).

So the recipe is unchanged in *content* but re-ordered: title (+ director) is
repeated at the END of the string. Whatever gets truncated, the title survives.
`build_text` is deterministic; `text_hash` lets the nightly job re-embed only
titles whose source text actually changed.
"""

from __future__ import annotations

import hashlib

MAX_TOKENS = 256
TITLE_REPEAT = 2          # PLAN §6.1 repeats the director twice; we repeat the title too


def build_text(payload: dict, *, title_repeats: int | None = None) -> str:
    """Deterministic embedding text for one catalog record.

    Args:
        payload: TMDB-derived record following PLAN §4.1 field names.
        title_repeats: override for how many times the title is appended.
    """
    repeats = max(1, title_repeats if title_repeats is not None else TITLE_REPEAT)
    title = str(payload.get("title") or "").strip()
    tagline = str(payload.get("tagline") or "").strip()
    genres = ", ".join(payload.get("genres") or [])
    keywords = ", ".join((payload.get("keywords") or [])[:12])
    director = str(payload.get("director") or "").strip()
    cast = ", ".join((payload.get("cast") or [])[:4])
    overview = str(payload.get("overview") or payload.get("overview_short") or "").strip()

    head = f"{title}."
    if tagline:
        head += f" {tagline}"
    if genres:
        head += f" Genres: {genres}."
    if keywords:
        head += f" Keywords: {keywords}."
    if director:
        head += f" Directed by {director}"
        head += f" Directed by {director}."
    if cast:
        head += f" Starring {cast}."
    if overview:
        head += f" {overview}"
    tail = f" {title}." * repeats
    return " ".join((head + tail).split())


def text_hash(text: str) -> str:
    """`sha256:...` marker stored in the payload (PLAN §4.1)."""
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def needs_reembed(old_payload: dict, new_payload: dict) -> bool:
    """True when the *content* fields that feed the text changed.

    Popularity/votes/ratings deliberately do NOT trigger re-embedding (PLAN §0.1:
    facts belong in the payload, not the vector).
    """
    return text_hash(build_text(old_payload)) != text_hash(build_text(new_payload))

