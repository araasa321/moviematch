"""Config encoding + install-secret auth (PLAN §10.1, §13)."""

import base64

import pytest

from addon.config import Config, ConfigError, secret_ok

from .conftest import *  # noqa: F401,F403


def test_config_roundtrip():
    blob = Config(secret="s", mode="gems", langs=("EN", "ja")).encode()
    decoded = Config.decode(blob)
    assert decoded.secret == "s"
    assert decoded.mode == "gems"
    assert decoded.langs == ("en", "ja")


def test_config_requires_a_secret_match_for_auth():
    assert secret_ok("abc", "abc") is True
    assert secret_ok("abc", "abd") is False
    # an unset install secret must never authorise anyone
    assert secret_ok("anything", "") is False
    assert secret_ok("", "") is False


def test_config_rejects_unknown_mode():
    with pytest.raises(ConfigError):
        Config.from_dict({"mode": "chaos"})


def test_config_rejects_bad_langs():
    with pytest.raises(ConfigError):
        Config.from_dict({"langs": "en"})


def test_decode_rejects_garbage():
    with pytest.raises(ConfigError):
        Config.decode(base64.urlsafe_b64encode(b"not json").decode().rstrip("="))
    with pytest.raises(ConfigError):
        Config.decode("!!!not base64!!!")


def test_gems_ratio_is_clamped():
    assert Config.from_dict({"gems": 5}).gems_ratio == 1.0
    assert Config.from_dict({"gems": -1}).gems_ratio == 0.0
