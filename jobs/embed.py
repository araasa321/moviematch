"""Embedding text recipe + text hashing (PLAN §6.1).

Recipe order is SIGNAL FIRST, PROSE LAST, BOUNDED:

    {title}. {title}. {tagline}. Genres: {genres}. Keywords: {keywords}.
    Directed by {director}. Directed by {director}. Starring {cast}.
    {title}. {first ~120 words of overview}

Why this order
--------------
`all-MiniLM-L6-v2` truncates to `max_seq_length` (256) word pieces, and it keeps
the FIRST tokens, discarding the tail — the sentence-transformers docs are
explicit: "Longer texts will be truncated to the first model.max_seq_length
tokens". `overview` is stored up to 4,000 chars in `policy.normalize_payload`, so
a realistic synopsis overflows the window on its own.

The previous recipe appended the title repeats at the END, so the tail — the one
thing it was trying to protect — was the FIRST thing destroyed whenever the
overview was long. Its docstring claimed the opposite. Measured on a trivial
record the bug is invisible; it only bites on real, long overviews.

So the title now leads the string AND is repeated immediately before the
overview, surviving either way. The overview is bounded to ~120 words (~170 word
pieces), leaving headroom under 256 so a truncation event no longer discards any
structured signal.

`build_text` is deterministic — no timestamps, no dict ordering, no locale — so
`text_hash` is stable and the job re-embeds only records whose source text
actually changed.
"""

from __future__ import annotations

import hashlib

MAX_TOKENS = 256
TITLE_REPEAT = 2          # PLAN §6.1 repeats the director twice; we repeat the title too
# ~120 words is ~170 word pieces for MiniLM, leaving ~85 pieces of headroom under
# MAX_TOKENS so a longer-than-expected render still cannot truncate the structured
# block above the overview.
OVERVIEW_WORDS = 120


def _overview_excerpt(overview: str, limit: int = OVERVIEW_WORDS) -> str:
    """First `limit` words of the overview, whitespace-normalised."""
    words = overview.split()
    if len(words) <= limit:
        return " ".join(words)
    return " ".join(words[:limit])


def build_text(payload: dict, *, title_repeats: int | None = None,
               overview_words: int = OVERVIEW_WORDS) -> str:
    """Deterministic embedding text for one catalog record.

    Args:
        payload: TMDB-derived record following PLAN §4.1 field names.
        title_repeats: how many times the title is repeated (before the overview).
        overview_words: word budget for the trailing synopsis.
    """
    repeats = max(1, title_repeats if title_repeats is not None else TITLE_REPEAT)
    title = str(payload.get("title") or "").strip()
    tagline = str(payload.get("tagline") or "").strip()
    genres = ", ".join(payload.get("genres") or [])
    keywords = ", ".join((payload.get("keywords") or [])[:12])
    director = str(payload.get("director") or "").strip()
    cast = ", ".join((payload.get("cast") or [])[:4])
    overview = str(payload.get("overview") or payload.get("overview_short") or "").strip()

    parts: list[str] = []
    # 1. Title leads: it is the strongest franchise signal and must never be
    #    the thing truncation reaches first.
    if title:
        parts.append(f"{title}.")
        if repeats > 1:
            parts.append(" ".join([f"{title}."] * (repeats - 1)))
    if tagline:
        # Trailing period: without it the tagline ran into "Genres:" as
        # "...Free your mind Genres:" and the genre block read as prose.
        parts.append(f"{tagline}." if not tagline.endswith(".") else tagline)
    if genres:
        parts.append(f"Genres: {genres}.")
    if keywords:
        parts.append(f"Keywords: {keywords}.")
    if director:
        # Kept twice: PLAN §6.1 specifies the auteur repeat.
        parts.append(f"Directed by {director}. Directed by {director}.")
    if cast:
        parts.append(f"Starring {cast}.")
    # 2. Title again immediately BEFORE the prose, so it survives even if the
    #    overview overflows the window.
    if overview and title:
        parts.append(f"{title}.")
    if overview:
        parts.append(_overview_excerpt(overview, overview_words))
    return " ".join(" ".join(parts).split())


def text_hash(text: str) -> str:
    """`sha256:...` marker stored in the payload (PLAN §4.1)."""
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def needs_reembed(old_payload: dict, new_payload: dict) -> bool:
    """True when the *content* fields that feed the text changed.

    Popularity/votes/ratings deliberately do NOT trigger re-embedding (PLAN §0.1:
    facts belong in the payload, not the vector).
    """
    return text_hash(build_text(old_payload)) != text_hash(build_text(new_payload))
