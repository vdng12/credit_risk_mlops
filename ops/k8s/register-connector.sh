#!/usr/bin/env bash
# register-connector.sh — Register the Debezium Postgres CDC connector
# Runs from inside the debezium pod via kubectl exec.
set -euo pipefail

DEBEZIUM_POD="${1:-$(kubectl get pods --no-headers -o name | grep debezium | sed 's/pod\///')}"

echo "Registering Debezium connector..."
kubectl exec "$DEBEZIUM_POD" -- curl -s -X POST http://localhost:8083/connectors/ \
  -H "Content-Type: application/json" \
  -d '{
  "name": "lending-cdc-connector",
  "config": {
    "connector.class": "io.debezium.connector.postgresql.PostgresConnector",
    "database.hostname": "postgres-source",
    "database.port": "5432",
    "database.user": "lending",
    "database.password": "lending123",
    "database.dbname": "lending_db",
    "topic.prefix": "lending",
    "plugin.name": "pgoutput",
    "publication.name": "dbz_publication",
    "slot.name": "dbz_slot",
    "table.include.list": "public.clients,public.loan_applications,public.application_decisions,public.payments,public.card_authorizations",
    "tombstones.enabled": "false"
  }
}' | python3 -m json.tool

echo
echo "Connector list:"
kubectl exec "$DEBEZIUM_POD" -- curl -s http://localhost:8083/connectors/ | python3 -m json.tool