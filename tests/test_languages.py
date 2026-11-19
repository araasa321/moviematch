"""Catalog-driven language detection for /configure.

The page used to offer a hardcoded list of languages, which was wrong in both
directions: it hid languages that had been crawled and offered ones that never
were. Detection comes from a Qdrant facet on the `language` payload key, so the
catalog is the source of truth.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest

from addon.config import Config, ConfigError
from addon.languages import (DetectedLanguage, LanguageDetector,
                             build_language_list)
from addon.main import make_app

from .conftest import *  # noqa: F401,F403


# ------------------------------------------------------------------ unit
def test_facets_become_languages_sorted_by_size():
    langs = build_language_list([("en", 29346), ("ja", 812), ("ko", 40)])
    assert [lang.code for lang in langs] == ["en", "ja", "ko"]
    assert langs[0].count == 29346


def test_english_is_pinned_first_regardless_of_count():
    langs = build_language_list([("ja", 5000), ("en", 30)])
    assert langs[0].code == "en"


def test_unknown_codes_are_still_listed():
    """Nothing crawled should be hidden just because we lack a pretty name."""
    langs = build_language_list([("en", 900), ("xyz", 60)])
    codes = [lang.code for lang in langs]
    assert "xyz" in codes


def test_unnameable_code_is_shown_as_its_uppercase_code():
    langs = build_language_list([("xyz", 60)])
    assert next(lang for lang in langs if lang.code == "xyz").name == "XYZ"


def test_tiny_languages_are_dropped_by_default():
    langs = build_language_list([("en", 100), ("is", 2)])
    assert "is" not in [lang.code for lang in langs]


def test_min_titles_zero_keeps_everything():
    langs = build_language_list([("en", 100), ("is", 2)], min_titles=0)
    assert "is" in [lang.code for lang in langs]


def test_english_always_present_even_if_absent_from_catalog():
    """An install must always be able to return to the default."""
    langs = build_language_list([("ja", 500)])
    assert "en" in [lang.code for lang in langs]
    en = next(lang for lang in langs if lang.code == "en")
    assert en.count == 0 and en.crawled is False


def test_no_detection_falls_back_to_defaults():
    langs = build_language_list(None)
    assert [lang.code for lang in langs] == ["en"]
    assert langs[0].crawled is False


def test_duplicate_facet_values_collapse():
    langs = build_language_list([("en", 900), ("en", 900), ("ja", 40)])
    assert [lang.code for lang in langs] == ["en", "ja"]


def test_label_includes_formatted_count():
    lang = DetectedLanguage(code="en", name="English", count=29346)
    assert lang.label == "English (29,346)"


# ------------------------------------------------------------------ detector
class _FakeFacetClient:
    def __init__(self, hits, *, raises: bool = False) -> None:
        self.hits = hits
        self.raises = raises
        self.calls = 0

    async def facet(self, key, *, limit=100, exact=False):
        self.calls += 1
        if self.raises:
            raise RuntimeError("facet not supported")
        return self.hits


def _run(coro):
    return asyncio.run(coro)


def test_detector_reports_unavailable_without_client():
    detector = LanguageDetector(None)
    assert detector.available is False
    assert [lang.code for lang in _run(detector.languages())] == ["en"]


def test_detector_caches_facet_result():
    client = _FakeFacetClient([("en", 100), ("ja", 5)])
    detector = LanguageDetector(client)
    _run(detector.languages())
    _run(detector.languages())
    _run(detector.languages())
    assert client.calls == 1, "facet is a collection-wide count; it must be cached"


def test_detector_survives_facet_failure():
    """An older Qdrant (no facet endpoint) must not take /configure down."""
    detector = LanguageDetector(_FakeFacetClient([], raises=True))
    langs = _run(detector.languages())
    assert [lang.code for lang in langs] == ["en"]


# ------------------------------------------------------------------ config
def test_config_accepts_any_language_code():
    cfg = Config.from_dict({"secret": "s", "langs": ["en", "ml"]})
    assert cfg.langs == ("en", "ml")


def test_config_rejects_malformed_language_codes():
    with pytest.raises(ConfigError):
        Config.from_dict({"secret": "s", "langs": ["english"]})


def test_config_rejects_empty_language_string():
    with pytest.raises(ConfigError):
        Config.from_dict({"secret": "s", "langs": ["  "]})


# ------------------------------------------------------------------ HTTP
@pytest.fixture()
def make_client(store, history):
    def build(facets=None, *, raises=False):
        detector = LanguageDetector(
            _FakeFacetClient(facets, raises=raises) if facets is not None else None)
        app = make_app(store=store, history_provider=lambda: history,
                       languages=detector)

        class _Sync:
            def get(self, path, params=None):
                async def run():
                    tr = httpx.ASGITransport(app=app)
                    async with httpx.AsyncClient(transport=tr,
                                                 base_url="http://test") as ac:
                        return await ac.get(path, params=params)
                return asyncio.run(run())

        return _Sync()
    return build


def test_page_lists_detected_languages_with_counts(make_client):
    client = make_client([("en", 29346), ("ja", 812), ("ko", 40)])
    body = client.get("/configure").text
    assert "Japanese" in body and "Korean" in body
    assert "29,346" in body or "812" in body


def test_page_hides_languages_absent_from_catalog(make_client):
    client = make_client([("en", 1000), ("ja", 50)])
    body = client.get("/configure").text
    assert "Japanese" in body
    assert "Turkish" not in body, "uncrawled language must not be offered"


def test_page_survives_detection_failure(make_client):
    client = make_client([], raises=True)
    page = client.get("/configure")
    assert page.status_code == 200
    assert "English" in page.text


def test_install_link_accepts_a_detected_language(make_client):
    client = make_client([("en", 1000), ("ml", 300)])
    data = client.get("/configure.json", params=[("langs", "en"), ("langs", "ml")]).json()
    assert data["ok"] is True
    assert Config.decode(data["blob"]).langs == ("en", "ml")


def test_install_link_rejects_malformed_code(make_client):
    client = make_client([("en", 1000)])
    data = client.get("/configure.json", params=[("langs", "english")]).json()
    assert data["ok"] is False


def test_warning_names_languages_with_no_titles(make_client):
    client = make_client([("en", 1000)])
    data = client.get("/configure.json", params=[("langs", "en"), ("langs", "ja")]).json()
    assert data["ok"] is True
    assert "Japanese" in data["warning"]
    assert "No titles crawled" in data["warning"]


def test_warning_flags_a_thin_catalogue(make_client):
    """Offered above MIN_TITLES_TO_OFFER but still small enough to warn about."""
    client = make_client([("en", 5000), ("ja", 30)])
    data = client.get("/configure.json", params=[("langs", "en"), ("langs", "ja")]).json()
    assert "Thin catalog" in data["warning"]


def test_no_warning_for_a_healthy_english_only_install(make_client):
    client = make_client([("en", 29000)])
    data = client.get("/configure.json", params=[("langs", "en")]).json()
    assert not data["warning"]
