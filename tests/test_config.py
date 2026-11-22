"""Config encoding + install-secret auth (PLAN §10.1, §13)."""

import base64
import json

import pytest

from addon.config import Config, ConfigError, secret_ok

from .conftest import *  # noqa: F401,F403


def _decode_b64(blob: str) -> str:
    padded = blob + "=" * (-len(blob) % 4)
    return base64.urlsafe_b64decode(padded.encode()).decode()


def _encode_raw(data: dict) -> str:
    return base64.urlsafe_b64encode(
        json.dumps(data, separators=(",", ":")).encode()).decode().rstrip("=")


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


def test_legacy_gems_key_is_ignored():
    """#4c: `gems_ratio` was read by nothing, so it is gone.

    A blob that still carries `gems` must decode fine (no 400 for an install link
    issued before the change) and must not re-encode the key.
    """
    assert not hasattr(Config(secret="s"), "gems_ratio")
    cfg = Config.from_dict({"secret": "s", "gems": 0.5})
    assert cfg.secret == "s"
    # The key survives in `raw` (it is the decoded blob, kept verbatim) but is
    # never written back by encode().
    assert "gems" not in json.loads(_decode_b64(cfg.encode()))


def test_existing_blob_with_gems_still_decodes():
    """Hand-build a pre-#4c blob and prove an old install URL keeps working."""
    legacy = _encode_raw({"secret": "s", "mode": "gems", "langs": ["en"], "gems": 0.5})
    cfg = Config.decode(legacy)          # must not raise ConfigError
    assert cfg.secret == "s"
    assert cfg.mode == "gems"
    assert not hasattr(cfg, "gems_ratio")
    # Re-encoding drops the key but keeps everything that matters.
    reencoded = Config.decode(cfg.encode())
    assert reencoded.mode == "gems" and reencoded.langs == ("en",)


def test_gems_mode_still_selects_gems_weights():
    """Regression: mode="gems" is the real switch and is unchanged by #4c."""
    from addon.rank import WEIGHTS

    assert Config.from_dict({"mode": "gems"}).mode == "gems"
    assert WEIGHTS["gems"] != WEIGHTS["blockbuster"]
