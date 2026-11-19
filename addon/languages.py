"""Which languages does the catalog actually contain?

The /configure page originally offered a hardcoded list of 21 languages. That was
wrong in both directions: it showed languages nobody had crawled (so rows looked
empty for no visible reason) and hid any language that WAS crawled but happened
not to be in the list.

So the catalog is the source of truth. `DetectedLanguage` is one entry the page
renders, and it carries the title count so a thin language advertises itself as
thin instead of silently producing a one-item row.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

from .config import DEFAULT_LANGS, LANG_NAMES

log = logging.getLogger(__name__)

# How long a facet result is reused. The facet is a collection-wide count, so it
# must not run on every page load; the catalog only changes on a nightly ingest.
FACET_TTL_SECONDS = 600

# A language with only a handful of titles cannot fill a row, so offering it
# would promise something the catalog cannot deliver.
MIN_TITLES_TO_OFFER = 25

# Below this, a language is still offered (if observed) but the user is warned
# that its rows will run short.
THIN_CATALOG_TITLES = 50

# Faculty data reports ISO-639-1 mostly, but TMDB occasionally carries 3-letter
# or region-tagged codes. Anything not in the table is still listed (so nothing
# is hidden from the user), just shown as its uppercase code.
KNOWN_NAMES = dict(LANG_NAMES)


@dataclass(frozen=True)
class DetectedLanguage:
    code: str
    name: str
    count: int
    crawled: bool = True

    @property
    def label(self) -> str:
        return f"{self.name} ({self.count:,})" if self.count else self.name


def _display_name(code: str) -> str:
    return KNOWN_NAMES.get(code) or (code.upper() if len(code) <= 3 else code)


def _sort_key(lang: DetectedLanguage) -> tuple[int, int, str]:
    # English first (it is the default and the one every install relies on),
    # then by how much content exists, then alphabetically for stability.
    return (0 if lang.code in DEFAULT_LANGS else 1, -lang.count, lang.code)


def build_language_list(facets: list[tuple[str, int]] | None,
                        *, min_titles: int = MIN_TITLES_TO_OFFER) -> list[DetectedLanguage]:
    """Turn raw facet hits into the list /configure renders.

    `facets is None` means detection was impossible (no Qdrant, or the facet call
    failed). The page then falls back to a known-but-unverified list rather than
    showing an empty page — a broken detector must not make the addon
    unconfigurable.

    Languages with fewer than `min_titles` titles are dropped by default: a
    language with 2 titles cannot fill a row, so offering it would be a promise
    the catalog cannot keep.
    """
    if facets is None:
        return [DetectedLanguage(code=code, name=_display_name(code), count=0,
                                 crawled=False) for code in DEFAULT_LANGS]

    seen: set[str] = set()
    out: list[DetectedLanguage] = []
    for code, count in facets:
        code = str(code or "").strip().lower()
        if not code or code in seen:
            continue
        seen.add(code)
        if count < min_titles:
            continue
        out.append(DetectedLanguage(code=code, name=_display_name(code), count=count))

    # English is always offered: an install that selected a foreign language and
    # then unticked English must still be able to get back to the default.
    for default in DEFAULT_LANGS:
        if default not in seen:
            out.append(DetectedLanguage(code=default, name=_display_name(default),
                                        count=0, crawled=False))

    out.sort(key=_sort_key)
    return out


class LanguageDetector:
    """Facet the `language` payload key, with a short cache.

    Wrapped in a tiny class so the app can hold ONE instance across requests and
    so tests can substitute a fake without touching Qdrant.
    """

    def __init__(self, client: Any | None) -> None:
        self._client = client
        self._cached_at = 0.0
        self._cached: list[tuple[str, int]] | None = None

    @property
    def available(self) -> bool:
        return self._client is not None

    async def facets(self) -> list[tuple[str, int]] | None:
        if self._client is None:
            return None
        now = time.monotonic()
        if self._cached is not None and now - self._cached_at < FACET_TTL_SECONDS:
            return self._cached
        try:
            facets = await self._client.facet("language", limit=200)
        except Exception as exc:  # noqa: BLE001 - /configure must never 500
            # Older Qdrant (<1.12) has no facet endpoint, and a network blip must
            # not take the config page down with it.
            log.warning("language facet failed: %s", exc)
            return None
        self._cached = facets
        self._cached_at = now
        log.info("detected %s languages in catalog", len(facets))
        return facets

    async def languages(self) -> list[DetectedLanguage]:
        return build_language_list(await self.facets())
