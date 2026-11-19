"""Per-request config (PLAN §10.1).

The Stremio manifest URL embeds a base64url(JSON) config so the addon stays
stateless. Parameters are agent-chosen but kept meaningful:

  secret  install secret — must equal INSTALL_SECRET, else 403
  mode    "blockbuster" | "gems"                       (default: MODE_DEFAULT)
  langs   list of ISO-639-1 codes to allow             (default: USER_LANGS)
  layout  "merged" | "separate" — one row per language, or one mixed row
  gems    float 0..1 — fraction of gems-mode slots (M6; accepted, unused in v1)
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import json
import re
from dataclasses import dataclass, field
from typing import Any

_LANG_CODE_RE = re.compile(r"[a-z]{2,3}(-[a-z0-9]{2,8})?")

VALID_MODES = ("blockbuster", "gems")
DEFAULT_LANGS = ("en",)

# Display names for languages we can prettify. NOT a whitelist: the catalog is
# the source of truth for which languages exist (see addon/languages.py), because
# a fixed list both hid crawled languages and offered uncrawled ones. An unknown
# code is accepted and displayed as its uppercase form.
SUPPORTED_LANGS: tuple[tuple[str, str], ...] = (
    ("en", "English"),
    ("ja", "Japanese"),
    ("ko", "Korean"),
    ("zh", "Chinese"),
    ("hi", "Hindi"),
    ("ta", "Tamil"),
    ("te", "Telugu"),
    ("ml", "Malayalam"),
    ("bn", "Bengali"),
    ("mr", "Marathi"),
    ("pa", "Punjabi"),
    ("ur", "Urdu"),
    ("es", "Spanish"),
    ("fr", "French"),
    ("de", "German"),
    ("it", "Italian"),
    ("pt", "Portuguese"),
    ("ru", "Russian"),
    ("uk", "Ukrainian"),
    ("tr", "Turkish"),
    ("ar", "Arabic"),
    ("fa", "Persian"),
    ("he", "Hebrew"),
    ("th", "Thai"),
    ("vi", "Vietnamese"),
    ("id", "Indonesian"),
    ("ms", "Malay"),
    ("tl", "Filipino"),
    ("sv", "Swedish"),
    ("no", "Norwegian"),
    ("da", "Danish"),
    ("fi", "Finnish"),
    ("nl", "Dutch"),
    ("pl", "Polish"),
    ("cs", "Czech"),
    ("hu", "Hungarian"),
    ("ro", "Romanian"),
    ("el", "Greek"),
)
SUPPORTED_LANG_CODES = tuple(code for code, _ in SUPPORTED_LANGS)
LANG_NAMES = dict(SUPPORTED_LANGS)

# Row presentation. "merged" = one For You row pooling every language (the
# ranking decides what is good, regardless of language). "separate" = the same
# rows repeated per language, so each gets its own row in Stremio.
VALID_LAYOUTS = ("merged", "separate")
DEFAULT_LAYOUT = "merged"


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
    layout: str = DEFAULT_LAYOUT
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
        # Accept any well-formed ISO-639-1-ish code rather than a fixed whitelist:
        # the catalog decides what exists. Reject anything that is not a plausible
        # language tag so a corrupted URL still fails loudly.
        cleaned = {x.strip().lower() for x in langs}
        bad = [x for x in cleaned if not _LANG_CODE_RE.fullmatch(x)]
        if bad:
            raise ConfigError(f"invalid language code(s): {', '.join(sorted(bad))}")

        layout = str(data.get("layout") or DEFAULT_LAYOUT).strip().lower()
        if layout not in VALID_LAYOUTS:
            raise ConfigError(f"unknown layout {layout!r}")

        try:
            gems_ratio = min(1.0, max(0.0, float(data.get("gems") or 0.0)))
        except (TypeError, ValueError) as exc:
            raise ConfigError("gems must be a number between 0 and 1") from exc

        norm_langs = tuple(sorted(cleaned))
        return cls(
            secret=str(data.get("secret") or ""),
            mode=mode,
            langs=norm_langs,
            layout=layout,
            gems_ratio=gems_ratio,
            raw=dict(data),
        )

    def encode(self) -> str:
        payload = {"secret": self.secret, "mode": self.mode, "langs": list(self.langs)}
        if self.layout != DEFAULT_LAYOUT:
            # Omitted when default so existing install URLs keep the exact blob
            # they were issued with (a changed blob is a different addon entry).
            payload["layout"] = self.layout
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
