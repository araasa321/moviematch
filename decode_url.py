"""Decode the manifest URL captured from the failing Stremio Web request."""

from __future__ import annotations

import base64

# transcribed from the DevTools screenshot, concatenating wrapped lines
RAW = ("eyJzZWNyZXQiOiJhNjkyNjA2Y2NiZDhhNmU0MWFhMjRlN2JjYjc3OTE4NTdkMjhkMjFiZjY4OWM0"
       "YjM4MzdjNDUyZGJhYmRkMmUwIiwibW9kZSI6ImJsb2NrYnVzdGVyIiwibGFuZ3MiOlsiZW4iXX0")


def main() -> None:
    print("length:", len(RAW))
    bad = sorted({c for c in RAW if not (c.isalnum() or c in "-_")})
    print("non-base64url chars:", bad or "none")
    padded = RAW + "=" * (-len(RAW) % 4)
    try:
        raw = base64.urlsafe_b64decode(padded)
        print("decoded:", raw.decode())
    except Exception as exc:  # noqa: BLE001
        print("DECODE FAILED:", type(exc).__name__, exc)


if __name__ == "__main__":
    main()
