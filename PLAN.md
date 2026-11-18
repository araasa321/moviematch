# PLAN.md — MovieMatch 🎬
### Personal AI recommendation addon for Stremio · Trakt-driven · TMDB-powered · $0/month

> **One-liner:** A Stremio catalog addon that adds personalized rows ("For You",
> "Because you watched X", "New for Your Taste", "Tonight from Watchlist",
> "Trending", "Hidden Gems") powered by embeddings + vector search + quality
> ranking. Your Trakt account (already connected to Stremio) is the single
> source of truth for watched / watchlist / ratings. Hosted on Koyeb free tier.

---

## 0. Core principles (do not violate)

1. **Meaning → vector. Facts → payload. Judgment → ranker.**
   - Embedding text: title, tagline, overview, genres, keywords, director, top cast.
   - Payload (never embedded): popularity, vote_average, vote_count, year,
     budget/revenue, language, runtime, collection, providers. Refreshed cheaply
     without re-embedding (`set_payload`).
2. **No ML at runtime.** The Koyeb app never loads an embedding model. All
   vectors are precomputed by the nightly job. Taste vector = arithmetic over
   stored vectors. Runtime deps: FastAPI + httpx + numpy only.
3. **Trakt is the user database.** We do not build our own watched/watchlist
   storage. Stremio scrobbles to Trakt → we read Trakt → rows update.
4. **Stateless app server.** Koyeb free tier has no persistent disk. All state
   lives in Qdrant Cloud (catalog + user_state) and Trakt (user actions).
5. **Similar ≠ worth-watching.** Always retrieve wide (top 200), then rank by
   quality + reach + era (league = v2). Never show raw cosine top-15.
6. **Single user, personal use.** No multi-tenant auth complexity. One install
   secret protects the instance.

---

## 1. Goals / Non-goals

### Goals
- Stremio addon (movie + series) with 6 personalized rows × 2 types.
- Personalization from Trakt: watched, ratings (1–10), watchlist, rewatches,
  series completion, hidden items. Updates within ~15 min of any Trakt action.
- Latest movies/series always recommendable (nightly TMDB sync, no retraining).
- Quality-aware ranking: no old/unpopular/failed mockbusters when you watched
  a blockbuster. "Hidden Gems" mode as an opt-in dial.
- 100% free hosting: Koyeb (app) + Qdrant Cloud free (vectors) + GitHub
  Actions (nightly ingestion) + TMDB/Trakt free APIs.

### Non-goals (v1)
- Multi-user / public distribution.
- Stream providers (Torrentio etc. already handle streams; we only add catalogs).
- Full meta pages (Cinemeta handles detail pages via IMDB ids).
- LLM chat mode (future upgrade, §19).
- "Not interested" button inside Stremio (protocol has no UI hook — handled
  via Trakt list, see §7.5).

---

## 2. Free-tier constraints & ⚠️ VERIFY checklist (do at M0)

| Service | Assumed free tier | ⚠️ Verify at signup | Fallback if wrong |
|---|---|---|---|
| **Koyeb** | 1 free web Service, ~512 MB RAM / 0.1 vCPU, scale-to-zero on idle, ephemeral disk, HTTPS on `*.koyeb.app` | Exact RAM/CPU, idle timeout, whether hours are metered, available regions | App is stateless & ~150 MB RSS → also fits Render/Railway/Fly free alternatives; keep Dockerfile portable |
| **Qdrant Cloud** | 1 forever-free cluster (~0.5 vCPU / 1 GB RAM / 4 GB disk), API key auth, no card | Cluster pause-on-idle policy, current size limits | Supabase/Neon free Postgres + pgvector; or in-process hnswlib + SQLite snapshot in object storage |
| **GitHub Actions** | 2,000 min/month private repos, unlimited public | Current quota | Make repo public, or run job locally (same script) |
| **TMDB API** | Free, key/token; ~50 req/s (older docs: 40/10s) | Current rate limit in docs | Job uses adaptive pacing + 429 backoff (default 15 req/s) |
| **Trakt API** | Free; GET ~1,000 calls/5 min; OAuth access token ~90 days; refresh tokens SINGLE-USE (rotate) | Token `expires_in`, rate headers, VIP-gated endpoints | Personal sync uses <50 calls/day; auto-refresh handles expiry |
| **Stremio clients** | Addon fetch timeout ~10s | Test cold-start load on your apps/devices | Serve cached rows instantly (§11.3) |
| **Qdrant snapshots** | May count against 4 GB disk | Check cluster storage page after first snapshot | Delete old snapshots after download |

**Sizing budget (personal catalog ~70k titles):**
- Vectors: 70k × 384d × 4B ≈ **108 MB** (int8 quantized in-RAM ≈ 27 MB)
- HNSW links + payload + indexes ≈ **150–250 MB** → comfortably inside 1 GB / 4 GB.
- Koyeb app RSS target: **< 250 MB** (no torch, no transformers, no pandas at runtime).

---

## 3. Architecture

```
┌──────────────────────── NIGHTLY (GitHub Actions, ~5–15 min) ───────────────────────┐
│ TMDB daily ID exports (movie + tv, incl. popularity)                               │
│   → diff vs Qdrant IDs → catalog policy → new/changed items                        │
│   → /details?append_to_response=credits,keywords,external_ids                      │
│   → build text → embed (all-MiniLM-L6-v2, on GH runner) → upsert vector+payload    │
│   → /discover bulk pages → refresh votes/ratings for active set (no per-id calls)  │
│   → set_payload popularity for ALL from export; recompute bayes/bayes_gems        │
└─────────────────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
                        ┌─────────────────────┐
                        │  Qdrant Cloud (free)│
                        │  • titles  (70k pts)│
                        │  • user_state (1 pt)│
                        └─────────────────────┘
                          ▲                 ▲
        Trakt REST sync   │                 │  ANN + filters + payload reads
   (history/ratings/      │                 │
    watchlist/hidden)     │                 │
┌──────────────┐   ┌──────────────────────────────────────┐   ┌──────────────────┐
│ Trakt account│◄──│  Koyeb free web Service (stateless)  │◄──│ Stremio clients  │
│ (scrobbled   │   │  FastAPI addon server                │   │ manifest/catalog │
│  by Stremio) │   │  taste engine + ranker + row cache   │   │ requests (HTTPS) │
└──────────────┘   └──────────────────────────────────────┘   └──────────────────┘
```

**Component responsibilities**
- `jobs/` (GH Actions): owns catalog freshness. Only place embeddings are created.
- `addon/` (Koyeb): owns serving. Reads Qdrant + Trakt, computes taste, ranks, caches.
- Qdrant: durable state (catalog, tokens encrypted, cached taste, exclusion lists).
- Trakt: durable user actions. Never duplicated as source of truth.

---

## 4. Data model (Qdrant)

### 4.1 Collection `titles` (movies + series unified)
- Vector: **384-d, cosine**, HNSW (m=16, ef_construct=100), scalar int8 quantization
  (`always_ram: true`), vectors on disk.
- **Point ID scheme (UUID):** `uuid5(NAMESPACE_URL, f"{media_type}:{tmdb_id}")`.
  Qdrant accepts uint64 or UUID ONLY — not strings like "movie:155".
  Keep media_type + tmdb_id in payload for debugging.

**Payload schema**
```jsonc
{
  "tmdb_id": 155, "media_type": "movie",        // "movie" | "series"
  "imdb_id": "tt0468569",                        // REQUIRED for Stremio; skip if absent
  "title": "The Dark Knight", "year": 2008, "end_year": null,
  "genres": ["Action","Crime","Drama"],
  "keywords": ["dc comics","vigilante","moral dilemma"],
  "director": "Christopher Nolan", "cast": ["Christian Bale","Heath Ledger","Aaron Eckhart"],
  "language": "en", "runtime": 152,
  "vote_average": 8.5, "vote_count": 32000, "popularity": 120.4,
  "bayes": 8.31,                                 // m=1000 movie / 300 series
  "bayes_gems": 7.4,                             // m=300, gems-mode scoring
  "budget": 185000000, "revenue": 1004558444,    // 0 = unknown
  "collection_id": 263,                          // belongs_to_collection (Batman)
  "release_date": "2008-07-16",
  "poster": "/qJ2tW6WMUDux911r6m7haRef0WH.jpg",  // TMDB path; serve w500 URL
  "backdrop": "/nMKdUUepR0i5zn0y1T4CsSB5chy.jpg",
  "overview_short": "Batman raises the stakes...",
  "text_hash": "sha256:...",                     // re-embed only if text changed
  "enriched": true                               // credits/keywords fetched
}
```
**Payload indexes:** `media_type`(keyword), `imdb_id`(keyword), `year`(integer),
`language`(keyword), `vote_count`(integer), `popularity`(float),
`collection_id`(keyword), `genres`(keyword), `release_date`(keyword).

### 4.2 Collection `user_state` (1 point, id=1)
Vector: 384-d zero placeholder. Payload:
```jsonc
{
  "trakt_tokens": "<Fernet-encrypted {current:{access,refresh}, previous:{access,refresh}, expires_at}>",
  "trakt_user": "yourname",
  "last_sync": {"history": "...", "ratings": "...", "watchlist": "...", "hidden": "..."},
  "watched_ids": [155, 1000001396],              // point ids, for must_not
  "watchlist_ids": [...], "hidden_ids": [...],
  "dislike_vecs": [[...]],                       // from ratings<=3 & hidden; penalty only
  "taste_long": [...], "taste_session": [...],
  "clusters": [{"vec": [...], "weight": 0.6}, ...],
  "last_watched": {"point_id": 27205, "title": "Inception", "at": "..."},
  "seed_picks": [],                              // onboarding favorites (cold start)
  "prefs": {"mode": "blockbuster", "langs": ["en"], "gems_ratio": 0.0},
  "last_good_rows": {},                          // last successful metas per row (§11.3)
}
```
> Tokens are encrypted with a key derived from `INSTALL_SECRET` (Fernet/HKDF).
> Koyeb env cannot be written at runtime, so Qdrant is the writable secret store.

---

## 5. Catalog policy (what gets indexed)

Keep the index personal-sized, not all 1.5M TMDB rows:
- **Movies:** `vote_count ≥ 100` **OR** released in last **24 months** (any votes)
  **OR** present in user's Trakt history/watchlist. Exclude `adult`/`video`.
- **Series:** `vote_count ≥ 50` **OR** first/last air date within 24 months **OR** in Trakt.
- **Must have `imdb_id`** to appear in rows (Stremio/Cinemeta requirement).
- Expected size: ~50–60k movies + ~15–20k series ≈ **70k points**.
- New releases with 0 votes are **included** (recency clause) — this is the fix
  for the old TF-IDF repo's vote-filter cold start.
- If cluster size approaches limits: raise thresholds (100→200) before anything else.

---

## 6. Ingestion pipeline (`jobs/`)

### 6.1 Embedding recipe (static text, embedded once)
```
{title}. {tagline}. Genres: {genres}. Keywords: {keywords}.
Directed by {director}. Starring {top 4 cast}. {overview}
```
- Model: `sentence-transformers/all-MiniLM-L6-v2` (384-d, fast on CPU runners).
  Upgrade path: `BAAI/bge-small-en-v1.5` (better retrieval; re-embed all once, §19).
- Director name repeated ×2 in text (auteur signal). Full overview (not 50 words).
- `text_hash` stored; re-embed only when hash changes (TMDB `/changes` or detected diff).
- Series: same recipe with `aggregate_credits` (top 4 cast, creator as director).

### 6.2 Bootstrap (one-time, resumable)
1. Crawl `/discover/movie` & `/discover/tv` with `vote_count.gte` + year windows
   (`primary_release_date.gte/lte`), `sort_by=popularity.desc`, ≤500 pages/window.
   Discover returns full objects (votes, popularity, dates, overview, poster) —
   this doubles as the bulk metadata source.
2. For each kept title: `/movie/{id}?append_to_response=credits,keywords,external_ids`
   (TV: `aggregate_credits,keywords,external_ids`) → text → embed → upsert.
3. Adaptive pacing (default 15 req/s, honor 429 `Retry-After`). ~70k detail calls
   ≈ 60–90 min. Run as manual `workflow_dispatch` (timeout 180 min) or locally.
4. Resumable: skip point ids already `enriched=true`. Safe to re-run any time.

### 6.3 Nightly job (cron `15 9 * * *` UTC — after TMDB exports land ~8 AM)
1. Download `movie_ids_{MM_DD_YYYY}.json.gz` + `tv_series_ids_...` from
   `files.tmdb.org/p/exports/` (fallback: yesterday's file). Lines carry
   `id, popularity, adult, video`.
2. Scroll Qdrant point ids → diff: **new ids**, **removed ids**.
3. New ids: sort by popularity desc, cap 3,000/day → fetch details → policy check
   → embed → upsert. (New release with low popularity still enters via Trakt
   presence or next-day popularity rise; cap prevents junk flood.)
4. **Popularity refresh for ALL:** batch `set_payload` from export (1,000 pts/call).
   Zero embedding calls. This is why popularity is payload, never vector.
5. **Votes/ratings refresh (active set):** `/discover/movie?sort_by=popularity.desc`
   500 pages (10k items) + `sort_by=release_date.desc` recent window (~2k) +
   user's Trakt items. Discover objects carry fresh votes → `set_payload`.
   Same for TV. ~600–800 calls ≈ 2–4 min. No per-id calls for numbers.
6. Recompute `bayes` and `bayes_gems` for refreshed items (§9) → batch `set_payload`.
7. Text changes: for active-set items, compare overview/credits hash if fetched;
   re-embed only changed (usually <100/day).
8. Weekly (Sun): prune ids absent from export; log catalog size & cluster stats.
9. Job is **idempotent**; failure → next night self-heals. GH failure email on.

### 6.4 GitHub Actions (`.github/workflows/nightly.yml`)
```yaml
name: nightly-sync
on:
  schedule: [{cron: "15 9 * * *"}]
jobs:
  sync:
    runs-on: ubuntu-latest
    timeout-minutes: 45
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: {python-version: "3.12"}
      - uses: actions/cache@v4
        with: {path: ~/.cache/huggingface, key: hf-minilm}
      - run: pip install -r jobs/requirements.txt
      - run: python jobs/sync_tmdb.py --mode nightly
        env:
          TMDB_TOKEN: ${{ secrets.TMDB_TOKEN }}
          QDRANT_URL: ${{ secrets.QDRANT_URL }}
          QDRANT_API_KEY: ${{ secrets.QDRANT_API_KEY }}
```

`.github/workflows/bootstrap.yml` — same steps, `on: workflow_dispatch`,
`timeout-minutes: 180`, `python jobs/sync_tmdb.py --mode bootstrap`.
`jobs/requirements.txt`: sentence-transformers, httpx, numpy, tenacity.
Monthly minutes: bootstrap ~90 + nightly ~10×30 ≈ **400 min** (inside 2,000).

---

## 7. Trakt integration (`addon/trakt.py`)

### 7.1 Auth (device flow, done once on /configure)
1. `POST /oauth/device/code` → show `user_code` + `verification_url` (trakt.tv/activate).
2. Poll `POST /oauth/device/token` until authorized.
3. Store {current:{access,refresh}, previous:{...}, expires_at} encrypted in user_state.
4. Refresh when <24h left. Refresh tokens are SINGLE-USE:
   a. Re-read user_state from Qdrant FIRST (another process may have refreshed).
   b. On success: persist current+previous in ONE atomic upsert, then ack.
   c. On failure: never discard tokens; backoff, re-read, retry (old token may
      still be valid if the request never reached Trakt).
   d. Both tokens rejected → trakt_auth="expired": Trending fallback, RED banner
      on /configure, /health exposes it.
   e. Lost-response window is irreducible → backstop = device-flow re-auth (~30s).
   Never log tokens.

Headers: `trakt-api-version: 2`, `trakt-api-key: {client_id}`, Bearer token.

### 7.2 Read endpoints (all paginated via `X-Pagination-*` headers)
| Data | Endpoint | Used for |
|---|---|---|
| Watched history | `GET /sync/history/movies` + `/sync/history/shows` (`start_at` incremental) | watched signal, rewatches (plays), last_watched, exclusion |
| Show completion | `GET /sync/watched/shows` (seasons/episodes nested) | completion fraction × TMDB `number_of_episodes` |
| Ratings | `GET /sync/ratings/movies` + `/shows` (1–10) | explicit weights, dislike list |
| Watchlist | `GET /sync/watchlist/movies` + `/shows` | watchlist signal + "Tonight" row |
| Hidden | `GET /users/{u}/lists/recsys-hidden/items` | exclusion + penalty vectors |
| Identity | `GET /users/settings` | username for config |

### 7.3 ID mapping (the join that makes everything work)
Trakt items include `ids.{trakt, imdb, tmdb, tvdb}`.
- `tmdb` (+ type) → Qdrant point id → vector/payload.
- `imdb` → Stremio meta id (`tt...`).
- Missing tmdb id → lookup via `/search/{id_type}?id_type=imdb` (Trakt search) once, cache.
- Item not in catalog (e.g., obscure watched film) → fetch TMDB details, embed
  **on the fly is NOT allowed at runtime** → enqueue to nightly via `user_state.missing_ids`;
  meanwhile it contributes nothing (acceptable) — but nightly job always includes
  Trakt items, so it self-heals within 24h.

### 7.4 Sync strategy
- On cold start: load cached `user_state` (instant rows), then background refresh.
- Incremental: `start_at=last_sync` per endpoint; full refresh nightly.
- In-memory TTL cache 15 min for Trakt responses & computed taste.
- Rate budget: <50 GETs/sync vs ~1,000/5 min limit. Trivial.

### 7.5 Hidden / "not interested"
- Auto-create Trakt personal list **"RecSys Hidden"** (slug `recsys-hidden`) at setup.
- Add from Trakt web/mobile app, or from /configure page (TMDB search + hide button
  → `POST /users/{u}/lists/{id}/items`).
- Stremio itself has no addon UI hook for a dislike button — documented limitation.
- Ratings ≤ 3 also feed the dislike penalty list automatically.

---

## 8. Personalization engine (`addon/taste.py`)

### 8.1 Signal weights (Trakt 10-scale)
| Signal | Weight |
|---|---|
| Rating r (1–10) | `(r − 6) / 2` → 10→+2, 8→+1, 6→0, 4→−1, 2→−2 |
| Watched, no rating | +0.7 |
| Watchlist | +0.8 |
| Rewatch (plays ≥ 2) | +2.5 |
| Hidden / rating ≤ 3 | −2 → **penalty list only, never subtracted from centroid** |
| Series | × (watched episodes / aired episodes), min 0.1 if any watched |

### 8.2 Taste vector
```
decay_stable = 0.5 ** (age_days / 365)        # ratings use /730
decay_recent = 0.5 ** (age_days / 60)
taste_stable = Σ(w⁺·decay_stable·v) / Σ(w⁺·decay_stable)
taste_recent = Σ(w⁺·decay_recent·v) / Σ(w⁺·decay_recent)
taste_long   = normalize(0.6·taste_stable + 0.4·taste_recent)
taste_session = weighted mean of last 3 actions within 72h
query_vec = normalize(0.7·taste_long + 0.3·taste_session)   # if session exists
```
- Computed with numpy over retrieved item vectors (batch `retrieve` with vectors).
- Cached in `user_state` + memory; recomputed on sync (O(#items), milliseconds).
- **Dislikes:** candidate penalized if max cosine to any `dislike_vecs` > 0.8.

### 8.3 Multi-taste clusters
v1: single taste vector. Clusters deferred to §19 (enable at 100+ positive signals).

### 8.4 Cold start
- < 3 positive signals → rows fall back to Trending/Popular; /configure offers
  "pick 5 favorites" (TMDB search) → `seed_picks` act as +2 ratings instantly.

---

## 9. Ranking engine (`addon/rank.py`)

### 9.1 Funnel (never skip stages)
```
query vector → ANN top-200 (Qdrant, with guards) → score → group collections
→ MMR diversity → exploration slots → slice by skip → top 100/response
```
**Guards (Qdrant filter):** `media_type` = row type; `vote_count ≥ 50`
(waived if year ≥ current−1); `language ∈ prefs.langs ∪ {seed.language}`;
`must_not has_id(watched ∪ hidden ∪ watchlist*)` (*watchlist excluded everywhere
except Tonight row); `imdb_id` exists.

### 9.2 Quality — Bayesian rating (IMDb formula), precomputed nightly
```
bayes = (v/(v+m))·R + (m/(v+m))·C
movies: m=1000, series: m=300, C = catalog mean vote_average (~6.4, recomputed weekly)
bayes_gems = same formula, m=300 for ALL titles. Both fields precomputed nightly.
```
Kills "8.9★ from 12 people" beating "8.1★ from 30,000".

### 9.3 Reach & era (league/tier = v2; precomputed where possible)
```
reach = 0.6·norm(log1p(vote_count)) + 0.4·norm(log1p(popularity))   # norms from catalog p99
era   = exp(−|year_c − year_seed| / 15)        # For You: year_seed = median year of user's positive items
# league/tier = v2 (§19): major-studio flag + budget-when-known + votes percentile.
```

### 9.4 Final score & modes
```
score = W_sim·similarity + W_q·(bayes/10) + W_r·reach + W_e·era
      + 0.15·(director == seed.director) + 0.05·min(2, cast_overlap)
      − 0.30·dislike_penalty − 0.20·language_mismatch
# gems mode scores with bayes_gems
```
| Mode | W_sim | W_q | W_r | W_e |
|---|---|---|---|---|
| blockbuster | .45 | .25 | .20 | .10 |
| gems        | .60 | .35 | .00 | .05 |

### 9.5 Post-processing
- **Collections:** max 1 per `collection_id` (best-scoring); sequels don't flood.
- **MMR:** λ=0.7; skip candidate if cosine to any already-picked > 0.90.
- **Exploration:** 2 of 15 slots = trending/new items outside profile (anti-bubble).
- **Series rows:** same engine, `media_type=series`.

---

## 10. Stremio addon (`addon/stremio.py`)

### 10.1 Protocol (catalog-only addon; HTTPS required — Koyeb provides it)
```
GET /{config}/manifest.json
GET /{config}/catalog/{type}/{id}.json
GET /{config}/catalog/{type}/{id}/{extra}.json     # extra = "skip=100&genre=Action"
GET /health                                         # Koyeb probe, no config
GET /configure                                      # HTML setup page
```
- `config` = base64url(JSON `{secret, mode, langs, ...}`); validate
  `secret == INSTALL_SECRET` else 403. Bare `/manifest.json` → redirect to /configure.
- `resources: ["catalog"]`, `idPrefixes: ["tt"]`, `types: ["movie","series"]`.
- We provide **meta previews only**; clicking opens Cinemeta; streams come from
  the user's existing addons. No `/stream`, no `/meta` handlers in v1.

### 10.2 Manifest (generated per request; "Because" name is dynamic)
```jsonc
{
  "id": "community.moviematch.personal",
  "version": "1.0.0",
  "name": "MovieMatch — For You",
  "description": "Trakt-personalized recommendations. Uses TMDB.",
  "logo": "https://<app>.koyeb.app/static/logo.png",
  "resources": ["catalog"],
  "types": ["movie", "series"],
  "idPrefixes": ["tt"],
  "behaviorHints": {"configurable": true, "configurationRequired": true},
  "catalogs": [
    {"type": "movie",  "id": "foryou",   "name": "🎯 For You",                    "extra": [{"name":"skip"},{"name":"genre","options":["Action","Comedy","Drama","Sci-Fi","Thriller","Horror","Romance","Animation"],"isRequired":false}]},
    {"type": "series", "id": "foryou",   "name": "🎯 For You (Series)",           "extra": [{"name":"skip"}]},
    {"type": "movie",  "id": "because",  "name": "🎬 Because you watched Inception","extra": [{"name":"skip"}]},
    {"type": "series", "id": "because",  "name": "🎬 Because you watched Breaking Bad","extra":[{"name":"skip"}]},
    {"type": "movie",  "id": "new4you",  "name": "🆕 New for Your Taste",         "extra": [{"name":"skip"}]},
    {"type": "series", "id": "new4you",  "name": "🆕 New for Your Taste (Series)","extra": [{"name":"skip"}]}
  ]
}
// M6: tonight, trending, gems ×2 types (trending stays internal fallback meanwhile)
```
> "Because you watched {title}" names come from Trakt `last_watched` at manifest
> time (per type). Manifest cache 300s so the name stays fresh.
> Rows appear in **Discover** (catalog dropdown) and on the **Board** feed;
> exact Board placement varies by Stremio client version — ⚠️ verify on your clients.

### 10.3 Row → query mapping
| Catalog id | Query vector | Filters / source | Notes |
|---|---|---|---|
| `foryou` | taste (clusters, session blend) | exclude watched+watchlist+hidden; optional `genre` payload filter | exploration 2 slots |
| `because` | last watched item's vector (per type) | exclude watched; era seed = that item's year | (league seed = that item → v2) |
| `new4you` | taste | `year ≥ current−1` OR last 18 months; recency boost ×1.2 | the "latest movies" row |
| `tonight` (M6) | taste | **restrict to watchlist point ids** (`has_id`), rank by score; runtime fits evening | pageSize small (5–10) |
| `trending` (M6) | none | TMDB `/trending/{movie|tv}/week` → map to catalog → tt ids | cache 1h |
| `gems` (M6) | taste | gems-mode weights; `vote_count` 50–2000 band; `vote_average ≥ 7.0 OR bayes_gems ≥ 6.5` | opt-in discovery |

### 10.4 Meta preview format (catalog response)
```jsonc
{"metas": [{
  "id": "tt1375666", "type": "movie", "name": "Inception",
  "poster": "https://image.tmdb.org/t/p/w500/<path>.jpg",
  "posterShape": "poster",
  "background": "https://image.tmdb.org/t/p/w780/<path>.jpg",
  "releaseInfo": "2010", "imdbRating": "8.4",
  "genres": ["Action","Sci-Fi"], "description": "A thief who steals..."
}], "cacheMaxAge": 900, "staleRevalidate": 3600}
```
- Series `releaseInfo`: "2019-" or "2019-2024".
- Pagination: compute ranked top-200 once per (row, config), cache 15 min,
  slice by `skip`; return `{"metas": []}` beyond 200.
- Skip items without `imdb_id` or poster (Stremio renders poorly otherwise).

### 10.5 /configure page
- Trakt device-code connect (status indicator), mode toggle (blockbuster/gems),
  language multi-select, "pick 5 favorites" cold-start search, hidden-list manager,
  **Install** button → `stremio://<host>/<config>/manifest.json` + HTTPS fallback link.
- Footer: *"This product uses the TMDB API but is not endorsed or certified by TMDB."* + TMDB logo.

---

## 11. Koyeb deployment

### 11.1 Dockerfile
```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY addon/ addon/
ENV PYTHONUNBUFFERED=1
CMD ["sh","-c","uvicorn addon.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
```
`requirements.txt` (runtime — **hard rule: no ML libs**):
`fastapi`, `uvicorn`, `httpx`, `numpy`, `jinja2`, `cryptography`, `tenacity`.

### 11.2 Environment variables (Koyeb service settings)
| Var | Purpose |
|---|---|
| `QDRANT_URL`, `QDRANT_API_KEY` | vector DB |
| `TMDB_TOKEN` | trending/search/details at serve time |
| `TRAKT_CLIENT_ID`, `TRAKT_CLIENT_SECRET` | OAuth (tokens themselves live encrypted in Qdrant) |
| `INSTALL_SECRET` | config-URL auth + token encryption key derivation |
| `USER_LANGS` | default `en` |
| `MODE_DEFAULT` | `blockbuster` |
| `LOG_LEVEL` | `info` (never log tokens) |
| `PORT` | injected by Koyeb |

### 11.3 Free-tier behavior
- Deploy via GitHub integration (auto-deploy on push) or ghcr.io image.
- Health check path `/health`; reports `{qdrant, trakt, trakt_auth, tmdb, catalog_size, taste_ready, last_sync}`.
- **Scale-to-zero:** cold start serves user_state.last_good_rows from Qdrant
  INSTANTLY; refresh taste+rows in background, persist new last_good_rows.
  Stremio won't auto-refetch → fresh personalization on next open. Never rely
  on in-memory cache (dies with the instance). Keep-warm: cron-job.org → /health
  every 10 min; /health also pings Qdrant (one cron warms both). ⚠️ if hours unmetered.
- Memory: 1 worker, async httpx, LRU caches (~2 MB), numpy taste math. Target <250 MB RSS.
- Region: pick closest to you (⚠️ verify free regions).

---

## 12. Repository structure
```
moviematch/
├── PLAN.md                  ← this file
├── README.md                ← install guide + TMDB/Trakt attribution
├── Dockerfile
├── requirements.txt         ← runtime (no ML)
├── addon/
│   ├── main.py              ← FastAPI app, routes, config parsing, caches
│   ├── stremio.py           ← manifest builder, catalog handlers, meta previews
│   ├── trakt.py             ← OAuth device flow, sync, id mapping, hidden list
│   ├── taste.py             ← weights, decay, centroid, clusters, session
│   ├── rank.py              ← guards, scoring, modes, MMR, collections
│   ├── qdrant.py            ← client, retrieve/search/set_payload helpers
│   ├── state.py             ← user_state load/save, token encryption
│   └── templates/configure.html
├── jobs/
│   ├── sync_tmdb.py         ← bootstrap + nightly (idempotent, resumable)
│   ├── policy.py            ← catalog inclusion rules
│   ├── embed.py             ← text recipe + model
│   └── requirements.txt     ← sentence-transformers, httpx, numpy, tenacity
├── .github/workflows/nightly.yml
├── .github/workflows/bootstrap.yml
└── tests/                   ← pytest (rank math, weights, id mapping, protocol) + eval_holdout.py
```

---

## 13. Security & privacy
- Single user; `INSTALL_SECRET` in config URL prevents strangers using the instance.
- Trakt tokens encrypted at rest (Fernet, key = HKDF(INSTALL_SECRET)); never in
  URLs, never logged. Qdrant API key + TMDB token in Koyeb/GH secrets only.
- HTTPS everywhere (Koyeb terminates TLS).
- No analytics, no third-party calls beyond TMDB/Trakt/Qdrant.
- Personal data (watch history) stays in your Trakt + your Qdrant cluster.

---

## 14. Failure modes & mitigations
| Failure | Behavior | Mitigation |
|---|---|---|
| Koyeb cold start | first load slow | cache headers; optional keep-warm; stateless fast boot |
| Qdrant unreachable | rows empty | serve Trending live from TMDB; /health red; last_good_rows in Qdrant user_state |
| Trakt token expired | personalization stale | rotation-safe refresh (§7.1); trakt_auth in /health + red banner; Trending fallback |
| TMDB export late/missing | sync skips | fallback to yesterday's file; idempotent next-night heal |
| TMDB 429 | job slows | adaptive pacing + Retry-After backoff |
| GH minutes exhausted | sync stops | repo public / run locally; catalog stays usable meanwhile |
| Item missing imdb_id | invisible in Stremio | filtered at serve; nightly external_ids enrichment |
| Trakt item not in catalog | no taste contribution | `missing_ids` queue → nightly force-include (self-heals <24h) |
| Free tier terms change | hosting risk | Dockerfile portable; Qdrant→pgvector adapter isolated in qdrant.py |

---

## 15. Testing & acceptance criteria
**Unit (pytest):** weight mapping (10-scale), decay math, Bayesian formula,
MMR, collection grouping, config secret validation, extra-string
parsing (`skip=100&genre=Sci-Fi`), Trakt→point-id mapping.

### 15.1 Offline eval (run BEFORE M3 sign-off)
tests/eval_holdout.py: hold out last 20 rated≥7 items (temporal split).
Build taste ONLY from pre-cutoff actions (no leakage).
Metrics: recall@50, MRR, median percentile-rank of held-out titles.
Baselines: trending AND raw cosine. Output a report — NOT a CI assert.
Single-user data → directional, not statistically robust.

**Integration / manual acceptance:**
- [ ] Seed *The Dark Knight* → top-15 contains ≥3 of {Inception, The Prestige,
      The Dark Knight Rises, Batman Begins, Joker, Heat, The Batman}; zero items
      with vote_count<50; ≤1 Batman-collection item.
- [ ] Rate a movie 10/10 in Trakt → For You visibly shifts within 15 min (cache TTL).
- [ ] Mark watched in Stremio → scrobbles to Trakt → item excluded from rows.
- [ ] Tonight row ⊆ Trakt watchlist, ordered by taste score. (M6)
- [ ] New release (0 votes, this month) appears in New-for-Your-Taste.
- [ ] Gems mode surfaces vote_count 50–2000 titles; blockbuster mode doesn't. (M6)
- [ ] Every meta has tt-id + poster; clicking opens Cinemeta; streams play via
      existing addons.
- [ ] Cold start < 10s; RSS < 300 MB; nightly job < 15 min; bootstrap resumable.
- [ ] Series: finishing a show boosts related series/movies (completion weight).

---

## 16. Milestones
| # | Deliverable | Acceptance |
|---|---|---|
| **M0** | Accounts + ⚠️VERIFY checklist (§2) signed off; secrets created; Qdrant collections + indexes created | free tiers confirmed; /health skeleton deployed on Koyeb |
| **M1** | `jobs/` bootstrap + nightly; GH workflow | ~70k enriched points; nightly <15 min; popularity payload fresh daily |
| **M2** | Trakt OAuth + sync + taste engine | device flow works; taste vector matches hand-computed fixture; incremental sync |
| **M3** | Ranker + all 6 rows (local FastAPI) | §15 seed tests pass; eval report beats both baselines (directional); rows correct against Trakt state |
| **M4** | Stremio protocol + /configure + install | 3 rows visible in Stremio Discover/Board on your clients; install URL works |
| **M5** | Koyeb production deploy | HTTPS install; cold start OK; health green; keep-warm decision made |
| **M6** | Polish: gems toggle, exploration, diversity, series completion, hidden manager + tonight/trending/gems rows + gems mode | full §15 checklist green |

---

## 17. Cost summary
| Item | Cost |
|---|---|
| Koyeb free web Service | $0 |
| Qdrant Cloud free cluster | $0 |
| GitHub Actions (≤~400 min/mo) | $0 |
| TMDB API / Trakt API | $0 |
| Embeddings (local on GH runner) | $0 |
| Domain (`*.koyeb.app`) | $0 |
| **Total** | **$0/month** |

---

## 18. Attribution & licenses
- README + /configure footer: **"This product uses the TMDB API but is not
  endorsed or certified by TMDB."** + TMDB logo (required by TMDB terms).
- Posters/backdrops hotlinked from `image.tmdb.org` (no redistribution).
- Trakt API terms: attribute Trakt; respect rate limits; personal use.
- Code: MIT (your choice). Data belongs to TMDB/Trakt respectively.

---

## 19. Future upgrades (post-v1, optional)
1. **LLM chat row / companion web chat:** retrieve top-20 → Gemini Flash free
   tier or local Ollama picks 5 + explains. LLM never invents titles (retrieval
   provides freshness). Not in the Stremio protocol — separate mini web UI.
2. **Semantic text search row:** free embedding API (Jina/HF/Gemini free tier)
   for query-time embedding; or keep TMDB keyword search (v1 default).
3. **Better embeddings:** bge-small-en-v1.5 → re-embed catalog once (~1h job),
   dual-collection zero-downtime swap.
4. **Collaborative boost:** if ever multi-user, nightly co-watch counts from
   interaction logs → extra ranker term. Architecture already supports it.
5. **Watch providers row:** TMDB `/watch/providers` payload → "New on Netflix
   for your taste" (JustWatch data, daily freshness).
6. **Snapshots/backup:** monthly Qdrant snapshot download + user_state JSON
   export button on /configure.
7. Multi-taste clusters (k-means, enable at 100+ positive signals).
8. League/tier v2: major-studio flag + budget-when-known + votes percentile.

---

## 20. Constants quick reference
```
EMBED_MODEL      = all-MiniLM-L6-v2 (384d)      POINT_ID = uuid5(NAMESPACE_URL, "{type}:{tmdb_id}")
CATALOG_MOVIES   = votes≥100 | last 24mo | Trakt   CATALOG_SERIES = votes≥50 | last 24mo | Trakt
BAYES            = m:1000(movie)/300(series), C=catalog mean (~6.4) · GEMS m=300, floor avg≥7.0 OR bayes_gems≥6.5
DECAY = stable 365d (ratings 730d) / recent 60d; long = 0.6·stable + 0.4·recent
                                                 SESSION = last 3 actions / 72h, blend 0.7/0.3
CLUSTERS = v2 (100+ positives)                 ANN_FUNNEL = 200 → rank → 15 shown
WEIGHTS = blockbuster .45/.25/.20/.10 (sim/q/reach/era) · gems .60/.35/0/.05
BONUSES          = director +0.15, cast +0.05×min(2,overlap)
PENALTIES        = dislike-sim>0.8 → −0.30, language mismatch −0.20
MMR              = λ0.7, dup-threshold 0.90    EXPLORATION = 2/15 slots
GUARD            = vote_count≥50 (waived if year≥current−1)
CACHES           = catalogs 900s · manifest 300s · trending 3600s · Trakt/taste 900s
NIGHTLY_CRON     = 15 9 * * * UTC              TMDB_PACING = 15 req/s + 429 backoff
```