# DEPLOY.md — GitHub → Koyeb, step by step

Order matters: **Koyeb deploys a working-but-empty addon first**, then you fill the
catalog and wire Trakt. That way a broken deploy is a 2-minute fix, not a debugging
session over three services at once.

```
1  GitHub repo        →  push code
2  Koyeb service      →  deployed; /health green (fixture catalog)
3  Install in Stremio →  rows render (fixture data)
4  Qdrant + TMDB      →  bootstrap catalog → real rows
5  Trakt              →  personalisation
```

---

## 1. Push to GitHub

Everything is already committed locally. Create an **empty** repo (no README, no
.gitignore) at https://github.com/new, then:

```powershell
git remote add origin https://github.com/<your-username>/moviematch.git
git branch -M main
git commit -m "M0/M1: addon serving layer, ingest job, deploy config"   # if not committed yet
git push -u origin main
```

Public repo is fine and keeps GitHub Actions minutes unlimited (§2 of the plan).

## 2. Create the Koyeb service

1. Sign up at https://app.koyeb.com (no card for the free tier).
2. **Create Service → GitHub →** pick `moviematch`.
3. Settings:
   - **Builder:** Buildpack is enough (Python), but the repo ships a `Dockerfile` —
     choosing **Dockerfile** gives you a reproducible image. Either works.
   - **Port:** `8000` (the Dockerfile already honours `$PORT` if Koyeb injects one).
   - **Instance:** the free `nano` / free-tier instance.
   - **Health check path:** `/health`
   - **Regions:** pick the one closest to you.
4. **Environment variables** — generate the secret first:
   ```powershell
   openssl rand -hex 32
   ```
   | Key | Value | Required |
   |---|---|---|
   | `INSTALL_SECRET` | the hex string above | **yes** — without it `/configure` uses a public dev secret |
   | `MODE_DEFAULT` | `blockbuster` | optional |
   | `USER_LANGS` | `en` | optional |
   | `QDRANT_URL` | *(step 4)* | later |
   | `QDRANT_API_KEY` | *(step 4)* | later |
   | `TRAKT_CLIENT_ID` | *(step 5)* | later |
   | `TRAKT_ACCESS_TOKEN` | *(step 5)* | later |
5. Deploy. First build takes ~2 min.

**Verify:** `https://<app>.koyeb.app/health` should return `"status": "ok"` and
`"qdrant": {"configured": false, "mode": "fixture (QDRANT_URL unset)"}` — that is
correct at this stage, not an error.

> Scale-to-zero: the first request after idle takes a few seconds. If Stremio
> times out, add a free cron (e.g. cron-job.org) hitting `/health` every 10 min.

## 3. Install in Stremio

Open `https://<app>.koyeb.app/configure`. It prints an install URL of the form
`stremio://<app>.koyeb.app/<config>/manifest.json` — click it. The addon appears in
Discover with fixture rows.

If rows are empty, that is expected until step 4: the bundled fixture catalog is
only 12 titles.

## 4. Real catalog (M1)

```powershell
$env:TMDB_TOKEN    = "<v4 bearer token>"
$env:QDRANT_URL    = "https://xxxx.cloud.qdrant.io"
$env:QDRANT_API_KEY = "<key>"
$py -m pip install -r jobs/requirements.txt

# prove it cheaply first
& $py jobs\sync_tmdb.py --mode bootstrap --types movie --limit 200 --stats-out b200.json
```

Check `b200.json` (`fetched` high, `dropped` low), then run the full bootstrap —
it is resumable, so re-running after a failure continues where it stopped.

For daily updates, add three **GitHub repo secrets** (`TMDB_TOKEN`, `QDRANT_URL`,
`QDRANT_API_KEY`) — `.github/workflows/bootstrap.yml` and `nightly.yml` already use
those names. Then set `QDRANT_URL` / `QDRANT_API_KEY` on Koyeb too and redeploy:
`/health` will report the real catalog size.

## 5. Trakt (personalisation)

1. https://trakt.tv/oauth/applications → **New Application**.
   - Redirect URI: `urn:ietf:wg:oauth:2.0:oob` (device flow lands in M2)
   - Permissions: none needed for read-only sync
2. Set `TRAKT_CLIENT_ID` on Koyeb.
3. Get an access token. Quickest route is the device flow by hand:
   ```powershell
   Invoke-RestMethod -Method Post https://api.trakt.tv/oauth/device/code `
     -Headers @{'trakt-api-version'='2';'trakt-api-key'='<CLIENT_ID>'} `
     -Body @{client_id='<CLIENT_ID>'}
   ```
   Visit the returned `verification_url`, enter the `user_code`, then poll
   `POST /oauth/device/token` with `{code, client_id, client_secret}` until it returns
   `access_token`. Set that as `TRAKT_ACCESS_TOKEN`.
4. Once set, the addon reads your real history/ratings/watchlist and falls back to
   the fixture if Trakt errors — it never 500s Stremio.

---

## Pre-deploy checklist

Run locally before pushing (no cloud needed):

```powershell
& $py -m pytest tests -q      # 73 passing
& $py deploy_check.py         # /health, redirect, X-Forwarded, secret gate, Trakt-off
```

## Known gaps at this commit

These are real and will bite on first contact with live data:

- **`/health` lies about catalog size** when Qdrant is configured — it reports the
  fixture store (12), not Qdrant. The Qdrant block sits alongside it; trust
  `qdrant.catalog_size` once `configured: true`.
- **Rows still come from the fixture store**, not Qdrant. Wiring `build_row` to a
  Qdrant-backed store is the remaining M1→M3 gap. Rows will look correct but only
  contain the 12 fixture titles until that lands.
- **`/configure` is a placeholder** — no Trakt connect button, no favourites picker
  (M4).
- **Trakt token refresh (§7.1) is not implemented** — the access token is static, so
  when it expires (90 days) you must update the env var by hand.
