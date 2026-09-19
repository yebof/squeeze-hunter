# Disaster Recovery Drill

Run this drill at least once before the first live dollar, and quarterly
thereafter.

## Goal

Prove that the system can be reconstructed from cold storage with **zero
hands on the original machine**, and that the restored runtime reconciles
its book against the broker instead of trusting a stale snapshot.

## What is backed up

`scripts/backup.sh` (P10) writes timestamped archives to `backups/` and
prunes anything older than `BACKUP_KEEP_DAYS` (default 14):

| Archive | Contents | Why it matters |
| --- | --- | --- |
| `state-<ts>.tar.gz` | `data/state/` — the runtime snapshot: positions, pending exits and buys, killswitch lockout, equity history | Without it a restart still reconciles positions from the broker, but loses the killswitch cooldown and the per-position stops context (entry score, bars held) |
| `parquet-<ts>.tar.gz` | `data/parquet/` — bars, short interest, earnings, the live decision log, freshness stamps | Re-ingesting from scratch takes hours and the FINRA CDN may be unreachable |
| `postgres-<ts>.sql.gz` | only when `SH_DB_URL` is set and a server answers | The runtime does not read Postgres yet; kept for the schema |

Nothing is automated by the repository itself. Install the cron entry on the
always-on machine:

```cron
0 2 * * * cd /path/to/squeeze-hunter && ./scripts/backup.sh >> backups/backup.log 2>&1
5 2 * * * rclone sync /path/to/squeeze-hunter/backups/ b2:squeeze-hunter-backups/
```

(`rclone` to any off-site bucket; the script itself never uploads.)

## Drill

1. Pick a clean directory: `mkdir -p /tmp/dr-drill && cd /tmp/dr-drill`
2. Pull the most recent backups from cold storage:
   ```bash
   rclone copy b2:squeeze-hunter-backups/ ./backups/ --include "$(date +%Y-%m)*"
   ```
3. Clone the repo and install:
   ```bash
   git clone https://github.com/yebof/squeeze-hunter.git .src
   cd .src && uv sync --all-extras
   ```
4. Restore the data volumes into the clone:
   ```bash
   tar xzf ../backups/$(ls ../backups/parquet-*.tar.gz | tail -1) -C .
   tar xzf ../backups/$(ls ../backups/state-*.tar.gz | tail -1) -C .
   ```
5. Run a scan against the restored cache and compare with a known-good scan:
   ```bash
   uv run squeeze-hunter scan --date 2025-04-21
   ```
6. Start the paper runtime against the paper account and watch the startup
   reconciliation in the logs (`state_restored`, then `reconcile_drift` if
   the broker disagrees with the snapshot — every drift is also alerted):
   ```bash
   cp ../.env .env   # credentials from the password manager, never from git
   uv run squeeze-hunter paper
   ```
7. (Optional) Postgres, if a dump exists:
   ```bash
   docker run -d --name dr-pg -p 5433:5432 \
     -e POSTGRES_USER=squeeze -e POSTGRES_PASSWORD=squeeze -e POSTGRES_DB=squeeze \
     postgres:14
   sleep 5
   gunzip -c ../backups/$(ls ../backups/postgres-*.sql.gz | tail -1) | \
     docker exec -i dr-pg psql -U squeeze -d squeeze
   ```

## Pass criterion

- Scan output matches production within rounding on the top 10 candidates.
- The runtime starts, restores the snapshot, reconciles against the broker
  with no unexpected drift, and serves `GET /health` with `ok: true`.
- No errors in stderr / log.
- Under 30 minutes wall clock from "machine lost" to "runtime ticking".

## Containerised deployment

`docker/compose.yml` runs the app as the `squeeze-hunter` service
(`restart: unless-stopped`, `.env` mounted, `./data` as the data volume,
`/health` as the healthcheck) next to Prometheus, Grafana, Postgres and the
IB Gateway. The monitor endpoint is reachable only on the compose network
(`squeeze-hunter:8080`); it is not published to the host.

```bash
docker compose -f docker/compose.yml up -d --build
docker compose -f docker/compose.yml logs -f squeeze-hunter
```

## Cleanup

```bash
docker rm -f dr-pg
rm -rf /tmp/dr-drill
```
