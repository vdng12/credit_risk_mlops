#!/usr/bin/env bash
# Standalone source-system (OLTP) Postgres.
#
# The lending source is the *external* system this platform ingests from, so it
# lives outside the compose stack (which runs the platform Postgres for
# mlflow/feast/metastore). This script latches the source onto `risk-net` so the
# platform can reach it as `postgres-source:5432`.
#
# Boots the OLTP schema at first boot from infra/postgres/init/.
#
# Usage:
#   bash scripts/run-source-postgres.sh
#   SOURCE_PG_PORT=5432 bash scripts/run-source-postgres.sh
#   DATA_GEN_IMAGE=data-generator:latest \
#     DATA_GEN_DATA_DIR=~/.cache/kagglehub/datasets/.../versions/1 \
#     DATA_GEN_OUTPUT_DIR=./outputs/generated \
#     bash scripts/run-source-postgres.sh
set -euo pipefail

# Pick up .env (POSTGRES_USER/PASSWORD/DB, S3 keys, ...) if present, like compose does.
if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  . ./.env
  set +a
fi

CONTAINER="postgres-source"
VOLUME="postgres-source-data"
NETWORK="risk-net"
IMAGE="postgres:16-alpine"
INIT_DIR="${PWD}/infra/postgres/init"

HOST_PORT="${SOURCE_PG_PORT:-5432}"
PG_USER="${POSTGRES_USER:-lending}"
PG_PASSWORD="${POSTGRES_PASSWORD:-lending123}"
PG_DB="${POSTGRES_DB:-lending_db}"

# 1. Make sure the shared network exists (normally created by `docker compose up`).
docker network inspect "$NETWORK" >/dev/null 2>&1 || docker network create "$NETWORK"

# 2. Create the data volume if missing.
docker volume inspect "$VOLUME" >/dev/null 2>&1 || docker volume create "$VOLUME"

# 3. Start (or resume) the container.
if docker inspect "$CONTAINER" >/dev/null 2>&1; then
  docker start "$CONTAINER" >/dev/null
else
  docker run -d \
    --name "$CONTAINER" \
    --network "$NETWORK" \
    -p "${HOST_PORT}:5432" \
    -e POSTGRES_USER="$PG_USER" \
    -e POSTGRES_PASSWORD="$PG_PASSWORD" \
    -e POSTGRES_DB="$PG_DB" \
    -v "$VOLUME":/var/lib/postgresql/data \
    -v "$INIT_DIR":/docker-entrypoint-initdb.d:ro \
    --restart unless-stopped \
    "$IMAGE"
fi

# 4. Wait until the server accepts connections.
echo "waiting for $CONTAINER on :$HOST_PORT ..."
for _ in $(seq 1 30); do
  if docker exec "$CONTAINER" pg_isready -U "$PG_USER" >/dev/null 2>&1; then
    break
  fi
  sleep 1
done

# 5. Optional: run the one-shot data generator (source-side population).
if [[ -n "${DATA_GEN_IMAGE:-}" ]]; then
  : "${DATA_GEN_DATA_DIR:?set DATA_GEN_DATA_DIR to the dir with the Home Credit CSVs}"
  : "${DATA_GEN_OUTPUT_DIR:?set DATA_GEN_OUTPUT_DIR to a writable output dir}"
  echo "running data generator: $DATA_GEN_IMAGE"
  docker run --rm \
    --network "$NETWORK" \
    -v "$DATA_GEN_DATA_DIR:/data:ro" \
    -v "$DATA_GEN_OUTPUT_DIR:/out" \
    -e DATA_DIR=/data \
    -e OUTPUT_DIR=/out \
    -e N_SAMPLES="${DATA_GEN_N_SAMPLES:-100000}" \
    -e SEED="${DATA_GEN_SEED:-42}" \
    "$DATA_GEN_IMAGE"
else
  echo "data generator skipped (build it, then set DATA_GEN_IMAGE to run it)"
fi

echo
echo "postgres-source ready"
echo "  host:   postgres://$PG_USER:$PG_PASSWORD@localhost:$HOST_PORT/$PG_DB"
echo "  in-net: $CONTAINER:5432  (reachable from anything on $NETWORK)"