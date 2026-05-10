# Disaster Recovery Drill

Run this drill at least once before declaring Phase 3 complete, and quarterly thereafter.

## Goal

Prove that the system can be reconstructed from cold storage with **zero hands on the original machine**.

## Backup procedure (already automated via cron)

```bash
# Run nightly at 02:00 ET
pg_dump -U squeeze squeeze | gzip > ~/backups/$(date +%Y-%m-%d).sql.gz
tar czf ~/backups/parquet-$(date +%Y-%m-%d).tar.gz data/parquet
rclone sync ~/backups/ b2:squeeze-hunter-backups/
```

## Drill

1. Pick a clean directory: `mkdir -p /tmp/dr-drill && cd /tmp/dr-drill`
2. Pull the most recent backups from cold storage:
   ```bash
   rclone copy b2:squeeze-hunter-backups/ ./backups/ --include "$(date +%Y-%m)*"
   ```
3. Spin up a fresh postgres on a non-default port to avoid collisions:
   ```bash
   docker run -d --name dr-pg -p 5433:5432 \
     -e POSTGRES_USER=squeeze -e POSTGRES_PASSWORD=squeeze -e POSTGRES_DB=squeeze \
     postgres:14
   sleep 5
   gunzip -c backups/$(ls backups/*.sql.gz | tail -1) | \
     docker exec -i dr-pg psql -U squeeze -d squeeze
   ```
4. Restore parquet:
   ```bash
   tar xzf backups/$(ls backups/parquet-*.tar.gz | tail -1) -C .
   ```
5. Clone the repo, install:
   ```bash
   git clone https://github.com/yebof/squeeze-hunter.git .src
   cd .src && uv sync --all-extras
   SH_DB_URL=postgresql+psycopg://squeeze:squeeze@localhost:5433/squeeze \
     uv run alembic upgrade head
   ```
6. Run a scan:
   ```bash
   SH_DB_URL=... uv run squeeze-hunter scan --date 2025-04-21 \
     --data /tmp/dr-drill/data/parquet
   ```
   Compare output to a known-good scan from production.

## Pass criterion

- Scan output matches production within rounding on top 10 candidates.
- No errors in stderr/log.
- Drill takes < 30 minutes wall clock from "machine lost" to "scan working".

## Cleanup

```bash
docker rm -f dr-pg
rm -rf /tmp/dr-drill
```
