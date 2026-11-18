# FINDINGS-EMBEDDINGS.md — does all-MiniLM-L6-v2 actually work for this?

**Verdict: yes, the embedding approach holds, and the §15 acceptance seed passes on real
vectors (5/7 targets, 1 Batman item). But the run also proves the plan's `because` row is
wrong as designed: raw nearest-neighbour ranking for a single seed returns a franchise
shelf. The ranker is what saves it — which is exactly why §9.1 must not be skipped.**

---

## 1. What "real" meant here, and what it didn't

I could not use `sentence-transformers` (no torch on this machine, and pip's temp-dir
writes are blocked by the sandbox). So I built a **numpy-only MiniLM inference engine**
([`eval/minilm.py`](eval/minilm.py)) that downloads the official `safetensors`, tokenizes
with a faithful BERT WordPiece port, and runs the 6-layer forward pass.

**Verified against the official ONNX graph** ([`eval/onnx_oracle.py`](eval/onnx_oracle.py)),
using identical token IDs to isolate the maths from tokenization:

| Check | Result |
|---|---|
| Forward pass vs official ONNX export | **cos = 0.999999** on all 4 test sentences |
| Embedding norms | 1.000 (L2-normalized, as expected) |
| Tokenizer spot-check | `the→1996, dark→2601, knight→5000, nolan→13401, ##ing→2075` ✓ |

✓ **Ground truth: my vectors are the real all-MiniLM-L6-v2 vectors.**

Two things this does **not** cover:
- ONNX's tokenizer extension isn't registered in this onnxruntime build, so tokenization
  is verified by vocabulary spot-check, not by a full differential test.
- **TMDB was not used** (no key here). Metadata comes from Wikipedia REST summaries —
  real titles, directors, plots. **Vote counts are synthetic** (drawn per title, with
  TMDB-ish values for the known targets). So the *vectors are real, the quality terms are not.*

## 2. The §15 acceptance seed, on real vectors

[`eval/acceptance_seed.py`](eval/acceptance_seed.py) — seed *The Dark Knight*, run through
the project's actual `addon/rank.py`:

| Metric | Blockbuster | Gems | §15 bar |
|---|---|---|---|
| Targets in top 15 | **5/7** | **5/7** | ≥ 3 ✓ |
| Batman-collection items | **1** | **1** | ≤ 1 ✓ |
| Min vote_count shown | 2,926 | 950 | ≥ 50 ✓ |

Blockbuster top-5: *The Dark Knight Rises, Inception, Interstellar, Insomnia, The Prestige*.

## 3. The finding that matters: raw NN is a franchise shelf

Same seed, no ranker, raw cosine top-15:

```
The Dark Knight Rises, The Batman, Batman Begins, Batman Returns, Batman,
Joker, Joker: Folie à Deux, Batman v Superman, Following, Unbreakable,
V for Vendetta, The Matrix, Logan, Madame Web, Morbius
```

That is 8 Batman titles and a set that ends at *Madame Web* and *Morbius*. Meanwhile
**Heat — a stated §15 target, and the film The Dark Knight is most often compared to —
never appears**, because prose vector space has no cinema-history signal that links a
Nolan heist epic to a Mann heist thriller. Only the dir/cast/quality terms surfaced
useful non-franchise titles.

**Consequences for the plan:**

1. **The ranker is not polish — it is load-bearing.** Any row built from a single seed
   vector must go through §9's quality/reach/era terms, collection dedupe, and MMR.
2. **Widen `because` retrieval.** §9's ANN top-200 on a single seed is dominated by
   franchise clones. 400–600 candidates, or blending 0.7·seed + 0.3·taste, gives the
   ranker something to choose from.
3. **Heat will not come from embeddings.** Getting films "like X" across franchises needs
   co-watch/collaborative signal or curated adjacency. §19 item 4 (collaborative boost)
   is quietly the highest-value future upgrade in the doc.

## 4. Embedding texture (why semantic ≠ genre)

Coarse-genre label agreement in the top-10 neighbours, 97 titles, 10 genres:

| Space | genre match @10 | vs random |
|---|---|---|
| **MiniLM (real vectors)** | **0.329** | **1.92× lift** |
| Random Gaussian 384-d | 0.171 | 1.00× (control) |
| Title-token TF-IDF | 0.175 | 1.02× |

Real embeddings carry strong structure — but note *how*: franchise and director dominate
(Batman cluster +0.142 cosine vs everything else, Nolan +0.068), while genre barely
separates (animation +0.012, horror +0.003). The model groups **series**, not **tastes**.
This is the root cause of §3, and it is a property of the model, not a bug.

## 5. Process notes (honesty section)

- I asserted "official reference: 0.7323 / 0.2669" from memory early on and chased a
  phantom mismatch for several steps. **The model card publishes no such numbers** —
  I checked. The assertion was wrong, not the model. That's why the ONNX oracle exists.
- HF's hosted inference is retired (`api-inference.huggingface.co` no longer resolves);
  `router.huggingface.co` answers but needs a token. The ONNX path replaced it.
- `pip install` into the temp dir fails with `PermissionError` even on directories that
  `mkdtemp` successfully — the sandbox blocks writes inside `mkdtemp`-created dirs.
  `onnxruntime` was installed into a workspace folder with full access for this check.

## 6. Reproduce

```powershell
$py = "C:\Users\USER\AppData\Local\Programs\Python\Python313\python.exe"
& $py eval\fetch_metadata.py       # 97 titles from Wikipedia (cached)
& $py eval\acceptance_seed.py      # §15 seed through addon/rank.py
& $py eval\diagnose.py             # MiniLM vs tf-idf vs random
& $py eval\check_labels.py         # genre coherence
# ONNX parity needs onnxruntime installed somewhere writable:
#   pip install --target .\pkg onnxruntime ; $env:PYTHONPATH=".\pkg"
#   & $py eval\onnx_oracle.py
```

## 7. Recommendation

**Proceed to M1 with one change to the plan before writing the bootstrap job:** embed
with the recipe in §6.1, but plan for `because`-style rows to be seeded from *candidate
pools*, not single vectors. Also budget time to test the recipe itself — §6.1 appends
genres, keywords, director, cast and full overview into one string, and for a 400-word
Wikipedia synopsis that tagline/overview tail can swamp the title (BGE's 512-token limit
makes this worse, not better). The title should be repeated at the *end* of the text too,
so it survives truncation.
