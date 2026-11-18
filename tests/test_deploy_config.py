"""Deploy-time failure modes seen on the real Koyeb instance.

1. A config blob corrupted in transit ("Failed to get addon manifest") must be
   rejected cleanly, never accepted or crash.
2. Behind Koyeb's TLS proxy the install URL must be https://<public-host>.
"""

from __future__ import annotations

import base64
from types import SimpleNamespace

import pytest

from addon.config import Config, ConfigError, secret_ok
from addon.main import _public_origin

from .conftest import *  # noqa: F401,F403

# The bytes a user actually pasted from a broken install link
# (0x0E / 0x96 / ')' injected mid-JSON).
CORRUPT_DECODED = (b'{"secret":"a692606ccbd8a6e\x0e1aa24H7bcb7791857d28d21bf689c'
                   b'\x0eb3837c452dbabdd2e0\x96,"mode)z\x96blockbuster\x96,"langs":["en"]s')


def _b64(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def _request(headers, netloc="10.0.0.1:8000"):
    return SimpleNamespace(headers=headers,
                           url=SimpleNamespace(scheme="http", netloc=netloc))


def test_corrupted_blob_is_rejected():
    with pytest.raises(ConfigError):
        Config.decode(_b64(CORRUPT_DECODED))


@pytest.mark.parametrize("bad", ["", "!!!!", "eyJ", _b64(b'{"secret":"x"}\x00')])
def test_garbage_blobs_raise(bad):
    with pytest.raises(ConfigError):
        Config.decode(bad)


def test_real_secret_round_trips():
    secret = "a692606ccbd8a6e1aa2427bcb7791857d28d21bf689cb3837c452dbabdd2e0"
    decoded = Config.decode(Config(secret=secret, langs=("en",)).encode())
    assert decoded.secret == secret
    assert secret_ok(decoded.secret, secret)


def test_blob_is_url_safe():
    blob = Config(secret="abc123+/=", mode="gems").encode()
    assert all(c.isalnum() or c in "-_" for c in blob), blob


# ------------------------------------------------------------- install URL
def test_forwarded_headers_win():
    origin = _public_origin(_request({"x-forwarded-proto": "https",
                                      "x-forwarded-host": "icy-ema.koyeb.app"}))
    assert origin == "https://icy-ema.koyeb.app"


def test_falls_back_to_host_header():
    origin = _public_origin(_request({"x-forwarded-proto": "https",
                                      "host": "icy-ema.koyeb.app"}))
    assert origin == "https://icy-ema.koyeb.app"


def test_never_emits_plain_http():
    """No forwarded headers at all -> still https, so Stremio accepts the link."""
    origin = _public_origin(_request({}, netloc="icy-ema.koyeb.app"))
    assert origin.startswith("https://")


def test_first_value_of_comma_lists():
    origin = _public_origin(_request({"x-forwarded-proto": "https, http",
                                      "x-forwarded-host": "a.koyeb.app, internal"}))
    assert origin == "https://a.koyeb.app"
