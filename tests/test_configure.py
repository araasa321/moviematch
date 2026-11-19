"""The /configure page, its install-link builder, and per-language rows.

Option B of the language work: a custom configuration page that embeds the
user's choices in the addon repository URL and offers a `stremio://` install
link (stremio-addon-sdk docs, "Creating Addon Configuration Pages").

The critical property is that the page and the server agree: the install link is
produced by the SAME encoder that decodes on the serving side, so what a user
installs is exactly what the manifest/rows interpret.
"""

from __future__ import annotations

import httpx
import pytest

from addon.config import DEFAULT_LAYOUT, Config, ConfigError
from addon.main import make_app
from addon.stremio import build_manifest, split_row_id

from .conftest import *  # noqa: F401,F403


# ------------------------------------------------------------------ config model
def test_config_defaults_to_merged_english(secret):
    cfg = Config(secret=secret)
    assert cfg.langs == ("en",)
    assert cfg.layout == DEFAULT_LAYOUT == "merged"


def test_config_roundtrips_layout():
    cfg = Config(secret="s", langs=("en", "ja"), layout="separate")
    assert Config.decode(cfg.encode()).layout == "separate"


def test_config_omits_default_layout_from_blob():
    """A default config must keep the exact blob existing installs were given."""
    blob = Config(secret="s", langs=("en",), layout="merged").encode()
    assert "layout" not in Config.decode(blob).raw


def test_config_rejects_unknown_layout():
    with pytest.raises(ConfigError):
        Config.from_dict({"secret": "s", "langs": ["en"], "layout": "grid"})


def test_config_langs_are_normalised_and_sorted():
    cfg = Config.from_dict({"secret": "s", "langs": ["JA", " en ", "ja"]})
    assert cfg.langs == ("en", "ja")


# ------------------------------------------------------------------ row ids
def test_split_row_id_plain():
    assert split_row_id("foryou") == ("foryou", None)


def test_split_row_id_language():
    assert split_row_id("foryou@ja") == ("foryou", "ja")


def test_split_row_id_empty_language_is_none():
    assert split_row_id("foryou@") == ("foryou", None)


# ------------------------------------------------------------------ manifest
def test_merged_manifest_has_one_row_per_catalog_and_type():
    man = build_manifest("https://x", "blob", langs=("en", "ja"), layout="merged")
    ids = [c["id"] for c in man["catalogs"]]
    assert len(ids) == 6
    assert all("@" not in i for i in ids)


def test_separate_manifest_repeats_each_row_per_language():
    man = build_manifest("https://x", "blob", langs=("en", "ja"), layout="separate")
    ids = [c["id"] for c in man["catalogs"]]
    assert len(ids) == 12
    assert "foryou@en" in ids and "foryou@ja" in ids


def test_separate_with_single_language_falls_back_to_merged():
    """One language's "separate" set is identical to merged under a worse name."""
    man = build_manifest("https://x", "blob", langs=("en",), layout="separate")
    assert all("@" not in c["id"] for c in man["catalogs"])
    assert len(man["catalogs"]) == 6


def test_separate_rows_are_labelled_by_language():
    man = build_manifest("https://x", "blob", langs=("en", "ja"), layout="separate")
    names = {c["id"]: c["name"] for c in man["catalogs"]}
    assert "Japanese" in names["foryou@ja"]
    assert "English" in names["foryou@en"]


def test_every_catalog_still_offers_genre():
    man = build_manifest("https://x", "blob", langs=("en", "ja"), layout="separate")
    for cat in man["catalogs"]:
        assert any(e["name"] == "genre" for e in cat["extra"]), cat["id"]


# ------------------------------------------------------------------ rows
def test_per_language_row_only_returns_that_language(store, history):
    from addon.config import Config as C
    from addon.rows import build_row

    cfg = C(secret="s", langs=("en", "ja"))
    plan = build_row(store, history, "movie", "foryou", cfg, lang="ja")
    for cand in plan.ranked:
        assert cand.payload["language"] == "ja", cand.payload["title"]


def test_per_language_row_refuses_a_disabled_language(store, history):
    """A stale `foryou@ko` link must render empty, not leak titles."""
    cfg = Config(secret="s", langs=("en",))
    from addon.rows import build_row

    plan = build_row(store, history, "movie", "foryou", cfg, lang="ko")
    assert plan.ranked == []
    assert plan.diagnostics["reason"] == "lang_not_enabled"


def test_merged_row_spans_languages_when_both_enabled(store, history):
    """Guard against a per-language filter leaking into the merged row."""
    from addon.rows import build_row

    cfg = Config(secret="s", langs=("en", "ja"))
    plan = build_row(store, history, "movie", "foryou", cfg)
    assert plan.diagnostics.get("reason") != "lang_not_enabled"


# ------------------------------------------------------------------ /configure
@pytest.fixture()
def client(store, history):
    """Sync HTTP client over the real ASGI app.

    `httpx.Client` needs an async-capable transport driven by a loop, so a tiny
    event-loop runner wraps `ASGITransport`: this exercises the true FastAPI
    routing and jinja2 rendering instead of calling handlers directly.
    """
    import asyncio

    app = make_app(store=store, history_provider=lambda: history)

    class _Sync:
        def get(self, path: str, params=None):
            async def run():
                transport = httpx.ASGITransport(app=app)
                async with httpx.AsyncClient(transport=transport,
                                             base_url="http://test") as ac:
                    return await ac.get(path, params=params)
            return asyncio.run(run())

    return _Sync()


def test_configure_page_renders_real_controls(client):
    page = client.get("/configure")
    assert page.status_code == 200
    body = page.text
    assert "placeholder" not in body.lower()
    assert "Languages" in body and "Row layout" in body
    assert "English" in body and "Separate rows" in body


def test_configure_page_lists_every_detected_language(client):
    """Which languages appear is decided by the catalog, not a hardcoded list.

    This page is rendered from the fixture catalog with detection unavailable, so
    the fallback set must still name its languages rather than showing raw codes.
    """
    body = client.get("/configure").text
    assert "English" in body
    assert 'value="en"' in body


def test_configure_json_builds_a_working_install_link(client):
    data = client.get("/configure.json", params=[("langs", "en"), ("langs", "ja")]).json()
    assert data["ok"] is True
    assert data["stremio_url"].startswith("stremio://")

    # the produced blob must decode to exactly what was asked for
    cfg = Config.decode(data["blob"])
    assert cfg.langs == ("en", "ja")


def test_configure_json_respects_layout_choice(client):
    data = client.get("/configure.json",
                      params=[("langs", "en"), ("langs", "ja"),
                              ("layout", "separate")]).json()
    assert data["ok"] is True
    assert Config.decode(data["blob"]).layout == "separate"


def test_install_link_manifest_matches_the_choices(client):
    """End-to-end: the link the page hands out must serve the chosen rows."""
    data = client.get("/configure.json",
                      params=[("langs", "en"), ("langs", "ja"),
                              ("layout", "separate")]).json()
    manifest = client.get(f"/{data['blob']}/manifest.json").json()
    ids = [c["id"] for c in manifest["catalogs"]]
    assert "foryou@ja" in ids and "foryou@en" in ids


def test_configure_json_rejects_malformed_language_code(client):
    """Shape is validated even though the catalog decides what exists."""
    data = client.get("/configure.json", params=[("langs", "xx!!")]).json()
    assert data["ok"] is False


def test_configure_json_requires_a_language(client):
    data = client.get("/configure.json").json()
    assert data["ok"] is False
    assert "at least one" in data["error"]


def test_configure_json_rejects_bad_layout(client):
    data = client.get("/configure.json",
                      params=[("langs", "en"), ("layout", "grid")]).json()
    assert data["ok"] is False


def test_configure_warns_when_foreign_languages_chosen(client):
    """With detection unavailable, a foreign pick must still warn."""
    data = client.get("/configure.json", params=[("langs", "en"), ("langs", "ja")]).json()
    assert data["warning"], "user must be told foreign titles need a crawl"


def test_configure_english_only_has_no_warning(client):
    data = client.get("/configure.json", params=[("langs", "en")]).json()
    assert not data.get("warning")


# ------------------------------------------------- regressions found in a browser
# Both of these passed the server-side tests and only failed in a real page,
# so they are pinned here at the level that would have caught them.
def test_page_fetches_itself_over_the_served_scheme(client):
    """The page must call back on the scheme it was served with.

    The install URL is deliberately https-upgraded (Stremio rejects http), but
    that upgrade was also applied to the in-page fetch endpoint, so every request
    died with ERR_SSL_PROTOCOL_ERROR whenever TLS terminated in front of the app.
    """
    body = client.get("/configure").text
    assert "ERR_SSL" not in body
    # served over http:// in tests, so the endpoint must not be https://
    assert '"http://test/configure.json"' in body, body[-800:]


def test_install_link_is_still_https_upgraded(client):
    """The upgrade must survive for the install link itself."""
    data = client.get("/configure.json", params=[("langs", "en")]).json()
    assert data["https_url"].startswith("https://")
    assert data["stremio_url"].startswith("stremio://")


def test_summary_uses_friendly_mode_name(client):
    """`blockbuster` is an internal key; the page must not leak it."""
    data = client.get("/configure.json",
                      params=[("langs", "en"), ("mode", "blockbuster")]).json()
    assert "Blockbusters" in data["summary"]
    assert "blockbuster" not in data["summary"]


def test_gems_mode_summary(client):
    data = client.get("/configure.json",
                      params=[("langs", "en"), ("mode", "gems")]).json()
    assert "Hidden gems" in data["summary"]
