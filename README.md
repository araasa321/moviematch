# MovieMatch

Personal Stremio recommendation addon — Trakt-driven, TMDB-powered, $0/month.
Design lives in [PLAN.md](PLAN.md); findings from building/testing it live in
[FINDINGS.md](FINDINGS.md) and [FINDINGS-EMBEDDINGS.md](FINDINGS-EMBEDDINGS.md).

## Status

| Part | State |
|---|---|
| Taste engine + ranker (`addon/`) | working, unit-tested |
| Stremio protocol (`manifest`, `catalog`, `health`) | working, tested over a real socket |
| Ingest job (`jobs/`) — bootstrap + nightly | working offline against fakes |
| Dockerfile + `/health` for Koyeb | ready — `deploy_check.py` passes |
| Trakt read path (`addon/trakt.py`) | working; token refresh is not (M2) |
| Real TMDB / Qdrant runs | **not yet executed** (needs credentials, see DEPLOY.md) |
| Qdrant-backed rows, `/configure` UI | not started (M3/M4) |

## Quick start (no credentials needed)

```powershell
$py = "C:\Users\USER\AppData\Local\Programs\Python\Python313\python.exe"

& $py -m pytest tests -q          # 73 tests
& $py smoke.py                    # ranked rows on the fixture catalog
& $py jobs\offline_run.py         # full ingest path against fake TMDB + Qdrant
& $py jobs\sync_tmdb.py --dry-run # same, through the real CLI
```

Run the addon locally against the fixture catalog:

```powershell
& $py -m uvicorn addon.main:app --port 8000
# then: http://127.0.0.1:8000/configure  -> gives you an install URL
```

## Deploying

**→ [DEPLOY.md](DEPLOY.md)** — GitHub push, Koyeb service + env vars, Qdrant/TMDB
bootstrap, Trakt, and a pre-deploy checklist.

Deploy-critical files: [`Dockerfile`](Dockerfile), [`.dockerignore`](.dockerignore),
[`.env.example`](.env.example), [`deploy_check.py`](deploy_check.py), and the two
workflows in [`.github/workflows/`](.github/workflows).

```powershell
& $py deploy_check.py   # simulates what Koyeb will see, no cloud account needed
```

## Real ingest (M1)

Set these, then run the bootstrap. The **first run should be a small slice** — the
pipeline has never been exercised against live TMDB, so prove it cheaply:

```powershell
$env:TMDB_TOKEN    = "<v3 api key or v4 bearer>"
$env:QDRANT_URL    = "https://xxxx.cloud.qdrant.io"
$env:QDRANT_API_KEY = "<key>"
$py -m pip install -r jobs/requirements.txt

# 1) smoke: 200 movies only
& $py jobs\sync_tmdb.py --mode bootstrap --types movie --limit 200

# 2) sanity-check what landed, then the full bootstrap (~70k, 60-90 min)
& $py jobs\sync_tmdb.py --mode bootstrap            # resumable: safe to re-run
& $py jobs\sync_tmdb.py --mode nightly              # daily thereafter
```

`--recreate` drops and rebuilds the `titles` collection. `--stats-out run.json`
writes per-run counters for the GitHub Actions artifact.

### GitHub Actions

`.github/workflows/bootstrap.yml` (manual, 180 min) and `nightly.yml`
(cron `15 9 * * *` UTC) need three repo secrets: `TMDB_TOKEN`, `QDRANT_URL`,
`QDRANT_API_KEY`.

## Layout

```
addon/            serving: config, taste, rank, rows, stremio, main, qdrant client
jobs/             ingest: sync_tmdb, tmdb, policy, embed, offline_run
eval/             embedding validation (MiniLM numpy engine, ONNX parity, acceptance seed)
tests/            pytest
smoke.py          local row demo on the fixture catalog
```

## Attribution & license

This product uses the TMDB API but is not endorsed or certified by TMDB.
Posters/backdrops are hotlinked from `image.tmdb.org` (no redistribution).
Trakt data is used under Trakt's API terms, for personal use only.
