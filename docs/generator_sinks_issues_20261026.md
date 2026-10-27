# Generator Sinks Issues — 2026-10-26

Audit of the simulator's sink/output components against the intended
source-system-first architecture.

---

## Architecture reference

```
simulator ──► postgres-source (OLTP source tables only)
                  │
                  ├──► CDC (Debezium) ──► Redpanda ──► Flink (real-time)
                  │
                  └──► Batch dump ──► SeaweedFS S3 (parquet) ──► dbt (daily)
```

Source tables (go to Postgres, then CDC):
- clients
- application_decisions
- payments
- card_authorizations (stream)
- loan_applications (stream)

Warehouse tables (S3 only):
- application_master (analytical join)
- pos_balances
- card_balances
- bureau

---

## Issues

### 1. `cold_batches()` mixes source and warehouse entities

**File:** `simulator/src/core.py` — `MonthBatch.cold_batches()`

All 7 entities share a single dict and get identical treatment. The
`to_sinks` loop iterates all of them without distinguishing which are source
(need Postgres) and which are warehouse (S3 only).

### 2. `to_sinks` never calls `to_postgres`

**File:** `simulator/src/core.py` — `MonthBatch.to_sinks()`

```
# cold path: batch snapshots -> object store
for name, df in self.cold_batches().items():
    sink.to_parquet(...)    # local file
    sink.to_s3(...)         # S3
    # ← NO to_postgres() call anywhere
```

`SinkManager.to_postgres()` exists in `sinks.py` but is never invoked from
the main pipeline. Source tables never reach Postgres, so CDC has nothing to
capture.

### 3. Backfill path bypasses Redpanda for stream events

**File:** `simulator/src/core.py` — `MonthBatch.to_sinks()`

In backfill mode, stream events (`loan_applications`, `card_authorizations`)
are written to parquet + S3, never to Kafka/Redpanda. Historical stream data
doesn't flow through the message bus.

### 4. S3 failures are silently swallowed

**File:** `simulator/src/core.py` — `MonthBatch.to_sinks()`

```python
try:
    sink.to_s3(df, name, month_str, run_id)
except Exception:
    pass
```

If S3 is unreachable, the error is silently ignored. This means a run with
the kind cluster down silently skips all S3 writes without warning.

---

## Entity destination table

| Entity | Should go to | Actually goes to | Gap |
|---|---|---|---|
| clients | Postgres → CDC | parquet + S3 | Never reaches Postgres |
| application_decisions | Postgres → CDC | parquet + S3 | Never reaches Postgres |
| payments | Postgres → CDC | parquet + S3 | Never reaches Postgres |
| loan_applications (stream) | Redpanda (incr) / S3 (backfill) | parquet + S3 (backfill) | Backfill skips Kafka |
| card_authorizations (stream) | Redpanda (incr) / S3 (backfill) | parquet + S3 (backfill) | Backfill skips Kafka |
| application_master | S3 only | parquet + S3 | Correct |
| pos_balances | S3 only | parquet + S3 | Correct |
| card_balances | S3 only | parquet + S3 | Correct |
| bureau | S3 only | parquet + S3 | Correct |