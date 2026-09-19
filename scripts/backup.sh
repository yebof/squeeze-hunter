#!/usr/bin/env bash
# Nightly backup (P10): runtime snapshot + parquet cache (+ Postgres if reachable).
#
#   scripts/backup.sh [BACKUP_DIR]      default: ./backups
#
# Cron example (02:00 ET, from the repo root):
#   0 2 * * * cd /path/to/squeeze-hunter && ./scripts/backup.sh >> backups/backup.log 2>&1
#
# Off-site copy is deliberately left to the operator (e.g. `rclone sync
# backups/ b2:squeeze-hunter-backups/` as the next cron line); this script
# never uploads anything.
set -euo pipefail

cd "$(dirname "$0")/.."
BACKUP_DIR="${1:-backups}"
STAMP="$(date -u +%Y-%m-%dT%H%M%SZ)"
KEEP_DAYS="${BACKUP_KEEP_DAYS:-14}"
mkdir -p "$BACKUP_DIR"

# 1) Runtime state (positions, pending orders, killswitch lockout, telemetry).
if [ -d data/state ]; then
  tar -czf "$BACKUP_DIR/state-$STAMP.tar.gz" data/state
  echo "state:    $BACKUP_DIR/state-$STAMP.tar.gz"
fi

# 2) Parquet cache (bars, short interest, earnings, decisions, freshness).
if [ -d data/parquet ]; then
  tar -czf "$BACKUP_DIR/parquet-$STAMP.tar.gz" data/parquet
  echo "parquet:  $BACKUP_DIR/parquet-$STAMP.tar.gz"
fi

# 3) Postgres, only if a server answers (the runtime does not use it yet).
if command -v pg_dump >/dev/null 2>&1 && [ -n "${SH_DB_URL:-}" ]; then
  if pg_dump "$SH_DB_URL" 2>/dev/null | gzip > "$BACKUP_DIR/postgres-$STAMP.sql.gz"; then
    echo "postgres: $BACKUP_DIR/postgres-$STAMP.sql.gz"
  else
    rm -f "$BACKUP_DIR/postgres-$STAMP.sql.gz"
    echo "postgres: skipped (pg_dump failed or server unreachable)"
  fi
fi

# 4) Prune.
find "$BACKUP_DIR" -maxdepth 1 -type f \( -name '*.tar.gz' -o -name '*.sql.gz' \) \
  -mtime "+$KEEP_DAYS" -print -delete | sed 's/^/pruned:   /' || true
echo "done $STAMP"
