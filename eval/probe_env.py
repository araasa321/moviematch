"""Connectivity + environment probe for the real-embedding eval.

Checks, in order of what matters:
  1. Is a sentence-transformers model reachable (HF or mirror)?
  2. Is TMDB reachable with a token (env TMDB_TOKEN / TMDB_API_KEY)?
  3. Is Wikipedia reachable (metadata fallback)?
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request

UA = {"User-Agent": "MovieMatch-eval/0.1 (local test; contact: none)"}


def head(url: str, timeout: int = 15) -> tuple[int, str]:
    req = urllib.request.Request(url, headers=UA)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.geturl()
    except urllib.error.HTTPError as exc:
        return exc.code, str(exc)
    except Exception as exc:  # noqa: BLE001 - probe wants the reason
        return -1, f"{type(exc).__name__}: {exc}"


def get_json(url: str, headers: dict[str, str] | None = None, timeout: int = 15):
    req = urllib.request.Request(url, headers={**UA, **(headers or {})})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def main() -> None:
    print("=== model hosting ===")
    for host in ("https://huggingface.co", "https://hf-mirror.com", "https://pypi.org/simple/"):
        status, detail = head(host + "/")
        print(f"  {host:32s} {status} {detail[:80]}")

    model = "sentence-transformers/all-MiniLM-L6-v2"
    for base in ("https://huggingface.co", "https://hf-mirror.com"):
        status, detail = head(f"{base}/api/models/{model}")
        print(f"  model meta {base:26s} {status} {detail[:80]}")
    status, detail = head(
        "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/onnx/model.onnx")
    print(f"  onnx file head {status} (content-length: {detail[:40]})")

    print("=== TMDB ===")
    token = os.environ.get("TMDB_TOKEN") or os.environ.get("TMDB_API_KEY")
    if not token:
        print("  no TMDB_TOKEN / TMDB_API_KEY in env")
    else:
        try:
            data = get_json("https://api.themoviedb.org/3/configuration",
                            {"Authorization": f"Bearer {token}"})
            print("  TMDB v4 bearer OK:", list(data)[:3])
        except Exception as exc:  # noqa: BLE001
            print("  TMDB v4 bearer failed:", str(exc)[:100])
            try:
                data = get_json(
                    f"https://api.themoviedb.org/3/configuration?api_key={urllib.parse.quote(token)}")
                print("  TMDB v3 api_key OK")
            except Exception as exc2:  # noqa: BLE001
                print("  TMDB v3 failed:", str(exc2)[:100])

    print("=== Wikipedia (metadata fallback) ===")
    try:
        q = urllib.parse.quote("The Dark Knight")
        data = get_json("https://en.wikipedia.org/w/api.php?action=query&format=json"
                        f"&list=search&srsearch={q}&srlimit=3")
        print("  search OK:", [x["title"] for x in data["query"]["search"]])
        page = urllib.parse.quote("The_Dark_Knight_(film)", safe="")
        summ = get_json(f"https://en.wikipedia.org/api/rest_v1/page/summary/{page}")
        print("  summary OK:", summ.get("title"), len(summ.get("extract", "")), "chars")
    except Exception as exc:  # noqa: BLE001
        print("  wiki failed:", str(exc)[:160])


if __name__ == "__main__":
    main()
