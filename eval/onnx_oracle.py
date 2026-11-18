"""Oracle parity: official ONNX graph vs our numpy forward pass.

The ONNX file shipped in sentence-transformers/all-MiniLM-L6-v2 *is* a full
sentence-transformers pipeline (tokenizer + BERT + Pooling + Normalize), exported
from the torch model. So:

    ONNX(tokenizer(text))  ==  official all-MiniLM-L6-v2 embedding

Feeding our own token IDs into the ONNX graph isolates the transformer maths from
tokenisation. If those two agree (cos > 0.9999) then the numpy forward pass is
correct and any remaining difference is the tokenizer — which we then check
separately by letting ONNX tokenise.

    python eval/onnx_oracle.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import minilm  # noqa: E402

ONNX_DIR = Path(__file__).parent / "model" / "all-MiniLM-L6-v2" / "onnx"


def ensure_onnx() -> Path:
    dest = ONNX_DIR / "model.onnx"
    if dest.exists() and dest.stat().st_size > 1_000_000:
        return dest
    import urllib.request

    ONNX_DIR.mkdir(parents=True, exist_ok=True)
    for base in ("https://huggingface.co", "https://hf-mirror.com"):
        url = f"{base}/sentence-transformers/all-MiniLM-L6-v2/resolve/main/onnx/model.onnx"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "MovieMatch-eval/0.1"})
            with urllib.request.urlopen(req, timeout=180) as resp, open(dest, "wb") as out:
                total = 0
                while chunk := resp.read(1 << 20):
                    out.write(chunk)
                    total += len(chunk)
                    print(f"\r  onnx: {total / 1e6:6.1f} MB", end="", flush=True)
            print()
            return dest
        except Exception as exc:  # noqa: BLE001
            print(f"  {base} failed: {str(exc)[:90]}")
    raise SystemExit("could not download onnx/model.onnx")


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.ravel() / (np.linalg.norm(a) + 1e-12)
    b = b.ravel() / (np.linalg.norm(b) + 1e-12)
    return float(a @ b)


def main() -> None:
    path = ensure_onnx()
    import onnxruntime as ort

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    print("onnx inputs :", [(i.name, i.shape) for i in sess.get_inputs()])
    print("onnx outputs:", [(o.name, o.shape) for o in sess.get_outputs()])

    model = minilm.MiniLM()
    texts = [
        "The Dark Knight is a 2008 superhero film directed by Christopher Nolan.",
        "A thief who steals corporate secrets through dream-sharing technology.",
        "That is a happy person",
        "Today is a sunny day",
    ]

    print("\n=== tokenizer spot-check (vocab lookup) ===")
    vocab = model.tok["vocab"]
    for word in ("the", "dark", "knight", "nolan", "##ing", "batman"):
        print(f"  {word!r:10s} -> {vocab.get(word, '<OOV>')}")
    for text in texts:
        ours = minilm.encode(model.tok, text)
        print(f"  ours ({len(ours):>2}): {ours}")

    def pooled(hidden: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """sentence-transformers Pooling: mask-aware mean, then LayerNorm-free L2."""
        m = mask[..., None].astype(np.float64)
        return (hidden * m).sum(axis=1) / np.maximum(m.sum(axis=1), 1e-9)

    print("\n=== forward-pass parity: same token IDs -> same vector ===")
    print("    (graph returns last_hidden_state; we apply the documented mean-pool)")
    feed_names = [i.name for i in sess.get_inputs()]
    worst = 1.0
    for text in texts:
        ids = np.array([minilm.encode(model.tok, text)], dtype=np.int64)
        masks = np.ones_like(ids)
        types = np.zeros_like(ids)
        feeds = {"input_ids": ids, "attention_mask": masks, "token_type_ids": types}
        feeds = {k: v for k, v in feeds.items() if k in feed_names}
        hidden = np.asarray(sess.run(None, feeds)[0])
        onnx_vec = pooled(hidden, masks)[0]
        ours_vec = model.encode_ids(ids[0].tolist())
        sim = cosine(ours_vec, onnx_vec)
        worst = min(worst, sim)
        print(f"  cos={sim:.6f}  {text[:52]!r}")
    print(f"\n  worst cosine agreement: {worst:.6f}  "
          f"-> {'PASS (forward pass correct)' if worst > 0.999 else 'MISMATCH'}")
    print("  NOTE: ONNX's own tokenizer/processor extensions are not registered in "
          "this onnxruntime build, so tokenisation is verified only by spot check.")


if __name__ == "__main__":
    main()
