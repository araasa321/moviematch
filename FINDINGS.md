# FINDINGS.md — minimal slice of PLAN.md, built and tested

**Verdict: yes, the plan proceeds.** The core loop works end-to-end on synthetic
data with zero network calls. Building it surfaced five real problems in the plan
(§Bugs found) and three assumptions worth re-checking (§Risks). None are fatal;
two need a decision from you before M1.

**How to reproduce**

```powershell
$py = "C:\Users\USER\AppData\Local\Programs\Python\Python313\python.exe"   # has fastapi, numpy, pytest
& $py -m pytest tests -q          # 51 passed
& $py smoke.py                    # prints rows + 70k-title latency
```

---

## A. What was built (the minimum that proves the loop)

| File | Role | Plan section |
|---|---|---|
| `addon/ids.py` | `uuid5(NAMESPACE_URL, "{type}:{tmdb_id}")` point ids | §4.1 |
| `addon/config.py` | base64url config + `INSTALL_SECRET` constant-time check | §10.1, §13 |
| `addon/taste.py` | weights, decay, stable/recent/session centroids, blend | §8.1–8.2 |
| `addon/rank.py` | Bayes rating, reach, era, score, collections, MMR, exploration | §9 |
| `addon/rows.py` | Trakt → signals → guards → rank → Stremio metas | §7.3, §9.1, §10.4 |
| `addon/stremio.py` | manifest builder, `extra` parsing, catalog envelope | §10.2–10.4 |
| `addon/main.py` | FastAPI routes (`/health`, `/manifest.json`, `/{config}/catalog/...`) | §10.1 |
| `addon/memory_store.py` | in-memory stand-in for Qdrant (12-line surface) | §4 |
| `addon/fixtures.py` | 12-title synthetic catalog + Trakt-shaped history | — |
| `tests/` | 51 tests: math fixtures, taste vectors, guards, protocol, real HTTP | §15 |
| `smoke.py` | prints actual rows + 70k-title latency | §2, §15 |

Deliberately **not** built: TMDB client, Qdrant client, Trakt OAuth, embeddings,
Dockerfile, `/configure` UI. Those are mechanical once the seam holds.

## B. Verified by running

- **Ranking engine** — Bayes kills "8.9★ from 12 people" (8.31 beats 8.24), MMR
  trades score for diversity as λ drops, collection rule allows one Batman.
- **Taste vector** — single signal → unit vector; two signals → exact weighted
  mean `[2,1,0,0]/√5`; decay halves at 365d; negative ratings stay out of the centroid.
- **Exclusions** — watched ∪ watchlist ∪ hidden never resurface; language filter drops `ja`/`id`.
- **Protocol** — manifest has 6 catalogs, bare `/manifest.json` → 302 `/configure`,
  bad secret → 403, garbage config → 400, unknown row → 404, `skip=` paging matches.
- **Latency at 70k titles** — ANN 67 ms (exact numpy; Qdrant HNSW will be 1–3 ms),
  rank 1.96 ms. Total well inside Stremio's ~10 s. **Ranking math is not a bottleneck.**

## C. Bugs found (fixed here — each would have shipped)

1. **Dislikes poisoned the taste vector.** `compute_taste` folded *every* signal,
   including the 2/10 rating, into the recent centroid. So rating a film 10 while
   hating another produced a blend, not a preference. §8.1 says dislikes are
   "penalty list only"; the code now honours that.
2. **Watchlist entries entered the session window.** Missing `listed_at` fell back
   to `now`, which fabricated a "just watched" session signal. Missing timestamps
   no longer imply "now".
3. **Hidden items had no timestamp at all** and were being injected as
   session-weight signals. Hidden is now an exclusion set only.
4. **Vote-count guard was implemented twice**, and the copy inside `rank()` had no
   recency waiver → a fresh 0-vote release could never pass even though §9.1 waives
   the guard for `year ≥ current−1`. Guard now lives only in `rank()`, once.
5. **Toolchain drift.** `starlette 0.27 + httpx 0.28` (this machine) cannot build a
   `TestClient` at all: 7 protocol tests silently *skipped*. Replaced with a real
   uvicorn-on-a-socket test, so the HTTP surface is genuinely exercised.

## D. Plan-level concerns

1. **ANN top-200 is too shallow for `because` rows.** The seed is one title; a
   single vector's 200 nearest neighbours are its most literal clones (sequels,
   franchise, same director's entire filmography). The "Because you watched X"
   row will look like a franchise shelf. Suggest widening to 400–600 for `because`
   and blending the seed vector with the taste vector (e.g. 0.7 seed / 0.3 taste).
2. **Watchlist-only users get an almost empty For You.** 1–2 watchlist items pass
   `ready` but a thin vector plus a 0.35 similarity floor yields ~1 row. §8.4
   handles *zero* signals, not *thin* ones. Needs a "thin profile → mix trending"
   path, which is real work, not a constant.
3. **Cold start has no fallback yet.** `build_row` correctly refuses to fabricate
   rows, but §8.4's Trending fallback is unbuilt — so a brand-new install shows
   nothing. Fine for v1 sequencing, but it is the first thing a fresh Trakt
   history will hit.
4. **Series taste is separate from movie taste.** The current design builds one
   taste vector per media type, so 200 watched movies do nothing for the series
   rows. The plan does not say whether the signal should cross over. Worth deciding.

## E. Free-tier assumptions to verify before M0 sign-off

- **Koyeb free** — 512 MB RAM with 108 MB of float32 vectors *in RAM* leaves little
  headroom. Int8 quantization (27 MB) is not optional for the app; it is required
  for it to boot. Confirm RSS with quantization against the real Qdrant client.
- **Qdrant Cloud free** — 4 GB disk should hold 70k × 384 with int8 payload indexes,
  but snapshot behaviour counts against that quota. Check the storage page.
- **Trakt refresh tokens are single-use.** §7.1's rotation handling is correct and
  is the single highest-risk piece of plumbing in the plan — test it with a
  deliberate race before deploying, not after.

## F. Suggested next step

M1 (`jobs/`) is the right next slice: everything above was proven with fake
vectors, so the only unknown left in the critical path is whether
`all-MiniLM-L6-v2` puts *The Dark Knight* near *Joker* / *Inception* / *Heat* in
384-d space. Embed 200 real TMDB titles on your machine and check the
§15 acceptance seed by eye **before** committing to a 70k bootstrap job.
