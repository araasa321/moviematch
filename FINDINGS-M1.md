# FINDINGS-M1.md — ingest job built, plus the embedding-test changes

**M1 exists and runs offline. It has not touched live TMDB or Qdrant yet** — that
needs your credentials. Everything that could be verified without them is verified.

`73 tests pass` · `python jobs/offline_run.py` · `python jobs/sync_tmdb.py --dry-run`
· `python m1_smoke.py`

---

## 1. What M1 added

| File | Role | Plan section |
|---|---|---|
| [`jobs/sync_tmdb.py`](jobs/sync_tmdb.py) | bootstrap + nightly orchestration, resumable/idempotent | §6.2, §6.3 |
| [`jobs/tmdb.py`](jobs/tmdb.py) | httpx client: pacing, 429/Retry-After backoff, detail/discover/export | §2, §6 |
| [`jobs/policy.py`](jobs/policy.py) | catalog inclusion rules, Bayes/gems, TMDB→payload mapping | §5, §9.2 |
| [`jobs/embed.py`](jobs/embed.py) | text recipe + `text_hash` re-embed detection | §6.1 |
| [`addon/qdrant.py`](addon/qdrant.py) | Qdrant REST client (create, upsert, scroll, set_payload, search) | §4 |
| [`jobs/offline_run.py`](jobs/offline_run.py) | fake TMDB + fake Qdrant harness | — |
| [`.github/workflows/`](.github/workflows) | nightly cron + manual bootstrap with `--limit` | §6.4 |
| [`m1_smoke.py`](m1_smoke.py) | ingest output → ranker → Stremio metas | — |

## 2. Changes forced by the embedding test

The MiniLM findings changed **four things in the code**. `PLAN.md` is untouched,
as you asked — these are implementation decisions, recorded here so they're not
mistaken for plan drift.

**a. `because` rows retrieve far wider and blend in taste.** A single seed's
nearest neighbours are franchise clones, so `rank()` now takes `ann_limit` and
`sim_floor`, and `because` uses 600 candidates at a 0.30 floor with
`query = normalize(0.7·seed + 0.3·taste)` ([`addon/rows.py`](addon/rows.py)).

**b. The text recipe repeats the title at the end.** §6.1's ordering put the full
overview last, so a long synopsis could truncate the title out of the 256-token
window entirely. Title now appears first *and* last.

**c. Popularity/votes never re-embed.** `needs_reembed()` compares only
content fields; a nightly popularity refresh is a pure `set_payload` (tested:
zero embeddings on replay).

**d. Two run-safety knobs that don't exist in the plan.** `--limit N` and
`--dry-run` (a truly offline check). Given the bootstrap has never run live,
spending 90 GitHub Actions minutes before a 200-title slice works would be silly.

## 3. A correction: the "fix" did not clearly help

I ran the new `because` design against the old one on the 97-title corpus
([`eval/compare_because.py`](eval/compare_because.py)):

| Variant | §15 hits /7 | Batman-family titles |
|---|---|---|
| Old: seed only, 200 cap, floor .35 | **4** | 2 |
| New: seed+taste, 600 cap, floor .30 | 3 | 1 |

The new variant is cleaner (one fewer franchise item) but lost a target. **The
corpus is 97 titles, so a 600-candidate pool is the entire catalog** — this cannot
show what happens at 70k, where the wider pool is the whole point. I'm keeping the
change because the reasoning holds (you cannot rank what you never retrieved), but
**I have not proven it is better.** Treat it as unvalidated until the catalog is real.

## 4. Untested against the live services

These are the specific things that can still break on the first real run:

1. **TMDB auth.** The client sends `Authorization: Bearer $TMDB_TOKEN`. A v3 *api
   key* needs `?api_key=` instead. If you use a v3 key, set `TMDB_TOKEN` to the v4
   bearer or patch `TMDBClient`.
2. **Discover params.** The bootstrap slices by `primary_release_date.gte/lte`
   (movies) and `first_air_date.*` (series). TMDB silently ignores unknown params
   rather than erroring, so a typo shows up as suspiciously few results — check
   `fetched` in `sync-stats.json` on the first slice.
3. **Qdrant quantization config.** `titles_collection_spec()` uses
   `scalar: {type: int8, always_ram: true}` with `on_disk` vectors. If the free
   tier rejects it, rows still work (vectors are just unquantized) but memory
   doubles — worth checking RSS against the 512 MB Koyeb limit.
4. **Rate limits.** Pacing is 15 req/s with 429 backoff; the plan allows ~50.
   The 200-title slice will tell you the real ceiling cheaply.

## 5. Recommended first real run

```powershell
$py jobs\sync_tmdb.py --mode bootstrap --types movie --limit 200 --stats-out b200.json
```

Then inspect `b200.json`: `fetched` should be ~200+, `dropped` low, `no_imdb`
small. Point the addon at that collection and open `/configure` — if rows render
with real posters, the remaining work is volume, not correctness.
