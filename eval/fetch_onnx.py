"""Download a model file quietly, with size verification."""

import sys
import urllib.request
from pathlib import Path

REPO = "sentence-transformers/all-MiniLM-L6-v2"


def get(rel: str, expect_min: int, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    for base in ("https://huggingface.co", "https://hf-mirror.com"):
        url = f"{base}/{REPO}/resolve/main/{rel}"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "MovieMatch-eval/0.1"})
            with urllib.request.urlopen(req, timeout=300) as resp, open(dest, "wb") as out:
                declared = int(resp.headers.get("Content-Length") or 0)
                total = 0
                while chunk := resp.read(1 << 20):
                    out.write(chunk)
                    total += len(chunk)
            ok = total >= expect_min and (declared == 0 or total == declared)
            print(f"{'ok ' if ok else 'BAD'} {rel}: {total:,} bytes (declared {declared:,})")
            if ok:
                return
            dest.unlink(missing_ok=True)
        except Exception as exc:  # noqa: BLE001
            print(f"failed {base}: {str(exc)[:120]}")
    sys.exit(f"could not fetch {rel}")


if __name__ == "__main__":
    out = Path(__file__).parent / "model" / "all-MiniLM-L6-v2" / "onnx" / "model.onnx"
    get("onnx/model.onnx", expect_min=80_000_000, dest=out)
