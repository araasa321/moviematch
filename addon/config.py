"""Per-request config (PLAN §10.1).

The Stremio manifest URL embeds a base64url(JSON) config so the addon stays
stateless. Parameters are agent-chosen but kept meaningful:

  secret  install secret — must equal INSTALL_SECRET, else 403
  mode    "blockbuster" | "gems"                       (default: MODE_DEFAULT)
  langs   list of ISO-639-1 codes to allow             (default: USER_LANGS)
  gems    float 0..1 — fraction of gems-mode slots (M6; accepted, unused in v1)
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
from dataclasses import dataclass, field
from typing import Any

VALID_MODES = ("blockbuster", "gems")
DEFAULT_LANGS = ("en",)


class ConfigError(ValueError):
    """Raised for malformed or unauthorised config blobs."""


def _b64url_encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _b64url_decode(text: str) -> bytes:
    padded = text + "=" * (-len(text) % 4)
    return base64.urlsafe_b64decode(padded.encode("ascii"))


@dataclass(frozen=True)
class Config:
    secret: str = ""
    mode: str = "blockbuster"
    langs: tuple[str, ...] = DEFAULT_LANGS
    gems_ratio: float = 0.0
    raw: dict[str, Any] = field(default_factory=dict, compare=False)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Config":
        if not isinstance(data, dict):
            raise ConfigError("config must be a JSON object")

        mode = str(data.get("mode") or "blockbuster").strip().lower()
        if mode not in VALID_MODES:
            raise ConfigError(f"unknown mode {mode!r}")

        langs = data.get("langs") or list(DEFAULT_LANGS)
        if not isinstance(langs, list) or not all(isinstance(x, str) and x for x in langs):
            raise ConfigError("langs must be a non-empty list of strings")

        try:
            gems_ratio = min(1.0, max(0.0, float(data.get("gems") or 0.0)))
        except (TypeError, ValueError) as exc:
            raise ConfigError("gems must be a number between 0 and 1") from exc

        norm_langs = tuple(sorted({x.strip().lower() for x in langs}))
        return cls(
            secret=str(data.get("secret") or ""),
            mode=mode,
            langs=norm_langs,
            gems_ratio=gems_ratio,
            raw=dict(data),
        )

    def encode(self) -> str:
        payload = {"secret": self.secret, "mode": self.mode, "langs": list(self.langs)}
        if self.gems_ratio:
            payload["gems"] = self.gems_ratio
        return _b64url_encode(json.dumps(payload, separators=(",", ":")).encode())

    @classmethod
    def decode(cls, blob: str) -> "Config":
        try:
            data = json.loads(_b64url_decode(blob))
        except (binascii.Error, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ConfigError("config is not valid base64url JSON") from exc
        return cls.from_dict(data)


def secret_ok(given: str, expected: str) -> bool:
    """Constant-time comparison; an unset INSTALL_SECRET must never authorise.

    Surrounding whitespace is trimmed on BOTH sides: platform env-var UIs happily
    store a trailing newline pasted from a terminal, which silently turned every
    install URL into a 403 (`secret_configured: true` yet rejected).
    """
    given = (given or "").strip()
    expected = (expected or "").strip()
    if not expected or not given:
        return False
    return hmac.compare_digest(
        hashlib.sha256(given.encode()).digest(),
        hashlib.sha256(expected.encode()).digest(),
    )
