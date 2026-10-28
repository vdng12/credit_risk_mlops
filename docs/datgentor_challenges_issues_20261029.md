# Data Generator Challenges - Implementation Issues Log (2026-10-29)

Status of the data-problems sprint on branch `agent/datgen`: what was
achieved, the infra wire-test run, and the findings that the test
surface (issues not visible in unit tests).

---

## 1. Sprint summary (what we built)

18 commits on `agent/datgen` (nothing merged, nothing pushed), 59/59 tests
green. The work follows `docs/datgentor_improv_plan_20261028.md`.

| Phase | Work | Commit |
|---|---|---|
| 0 | **Indexed deterministic RNG** - per-(entity, month) streams so months are pure functions of (seed, entity, month); replay/resume/retry safe | a18dd9a |
| 1 | **YAML scenarios + subcommands** (`generate/validate/resume/calibrate`), config\_loader precedence (defaults < calibration < YAML < env < CLI), calibration wired via the previously orphaned `apply_to_config`, `run_id` in config | 746518b |
| - | **Problem -> entity scope contract** (annotated + enforced: duplicates = payments/applications only, per-entity rates) | 13b7374 |
| - | **Dead code removed** (`calibrate_children.py` superseded by `calibrate_structures.py`) | 1d924a8 |
| 3 | **Sinks package + lane paths** - `sinks/` (parquet/S3/postgres) and `paths/` (OfflinePath batch, OnlinePath stream); routing left MonthBatch | b2786bb |
| 4 | **Offline problems** - schema evolution (bureau v1/v2/v3), content payload hash, region skew knob | 82f780c |
| - | **play_forward fix** - forward-only runs no longer skip the configured start month | 7f16108 |
| - | **SimClock revived** - yields completed months, month-end landing fix (was off-by-one), restart-proof date predicates | 1d3d0a0 |
| 5 | **Streaming problems** - burst / duplicate replay / late arrival + day-granular emitter | 0b23df6 |
| - | **Canonical S3 cold-facts set** (object-store rule; docs first, then code) | b1208d3, b1c38bb |
| 5b | **LiveStage** - SimClock dispatcher: HOT per day / NIGHTLY payments+decisions / MONTHLY close, all to the correct lanes | ab91fef |
| - | **Loan due-day spread** - each loan pays on a stable 1..28 due day so monthly payment rows spread across the month (nightly files carry content) | e3c4ad7 |
| - | **stream -> live rename** (the command runs the whole live stage, not just streaming) | 3201e33 |

### Design decisions locked during the sprint

- **Object-store rule**: S3 = every business artifact `{clients,
  application_master, application_decisions, payments, pos_balances,
  card_balances, bureau}`; pure-stream-only = `{loan_applications,
  card_authorizations}` (Kafka + write-through archive, never S3).
  Payments/decisions/clients are dual-lane (file on S3 AND CDC stream).
- **Lane model**: clean generator -> batch lane applies OFFLINE problems ->
  stream lane emits CLEAN + STREAMING problems. (Not yet fully implemented -
  see F2.)
- **Live stage = three rhythms**: hot stream (per day, SimClock), nightly
  batch (payments + decisions files), monthly close (balances, bureau,
  clients, gold). Routing lives in `paths/`, dispatch in `livestage.py`.
- **Problem-interface split** (post-Phase-5 refactor, documented) - frame vs
  generation problem shapes; the lane separation above is its motivation.

---

## 2. Infra wire-test run (2026-10-29)

Environment: k3d `risk-cluster` (7/7 pods up), Sandbox `../credit-risk-datgen`.
Debezium connector was NOT registered (cluster recreated at some point since
the last `register-connector.sh`) - re-registered. Sandbox venv extended with
the `sinks` extra (sqlalchemy, psycopg2, boto3, pandas, pyarrow).

### Executed

1. `generate` (clean path, `--no-problems`, 200 customers, 2 months) against
   real postgres-source + SeaweedFS S3 + Debezium CDC.
2. Verified parquet archive, S3 objects, postgres row counts, and Redpanda
   topics.

### Result: write accounting per month = 22 paths

| Lane | Writes | Entities |
|---|---|---|
| parquet write-through | 9 | all |
| S3 (canonical set) | 7 | clients, master, decisions, payments, pos/card\_balances, bureau |
| postgres (OLTP -> CDC) | 6 | clients, applications, auths, decisions, payments, master (gold) |

- The canonical S3 routing verified live: `loan_applications` and
  `card_authorizations` correctly ABSENT from S3; master + payments now
  present.
- CDC: 5 topics auto-created (`lending.public.*`), Debezium envelopes with
  `op:"c"` and PK-keyed messages (sampled `clients` - full payload incl.
  Debezium Date serialization).
- Loan due-date spread visible in real postgres rows (payment_date Jan
  2..27).
- Postgres counts sane for 200 customers (clients 200, loan_applications 6,
  card_authorizations 6305, decisions 6, payments 44, master 45).

### Result: multi-month / problems-on FAIL

Month 1 clean-path writes succeed; month 2 clients collide on
`clients_pkey`; with problems enabled, payments collide on
`payments_pkey` even on a truncated DB.

---

## 3. Findings and issues

### F1 - Monthly snapshot PK collision (blocker for multi-month, clean path)

The OLTP append-only sink + stable `client_id` means month 2's clients
snapshot collides with month 1's appended rows (same id, new
`snapshot_date`); same-seed reruns collide the same way.

Fix direction: raw-zone upsert semantics - either key raw rows by
`(id, snapshot)` and `ON CONFLICT DO UPDATE`, or (with Phase-0
determinism) `ON CONFLICT DO NOTHING` for identical snapshots. Decision
needed: raw table keying policy.

### F2 - OFFLINE exact-duplicate leaks into the ONLINE lane (root cause)

Duplicates are an OFFLINE problem (rubric + contract). But the generic
injector runs BEFORE lane routing inside `_generate_month`, so the
duplicated payments/applications frames also feed postgres -> PK
collision. The lane model says: offline problems mutate ONLY the batch
artifacts; the stream lane stays clean and receives only streaming
problems. (This is the motivation to pull the Problem-interface split
forward.)

### F3 - Duplicate replay would collide with OLTP PKs (conceptual)

All raw OLTP tables have single-column PKs (checked: clients,
payments, auths, apps, decisions). Debezium KEYS messages by those PKs -
so duplicates cannot live inside the source tables at all. At-least-once
replay must be a TRANSPORT-level duplicate: re-emit the already-published
CDC message to the topic (never a second DB insert). Flink then sees the
identical message pair - the honest at-least-once signature.

### F4 - Burst is the wrong model as implemented

`events_for_tick` repeats the same day bundle N times (identical
auth_ids) - that is duplicates, not a volume spike, and it collides with
`card_authorizations_pkey`. A real burst = Nx events with DISTINCT ids,
scaled at generation (Phase-0-safe: ids = f(seed, entity, month, day,
slot)).

### F3/F4 resolution shape

- late arrival: generator-side timestamp shift (already correct).
- replay: topic-level re-emit after the postgres insert.
- burst: generation-scale the day's volume with distinct ids.

### F5 - SQLAlchemy parameter logging flood (cosmetic)

`[parameters: ...]` spam during postgres inserts; silence with a
logging config for the engine.

### F6 - Debezium connector is not persistent across cluster recreates

`kubectl exec ... curl /connectors` returned `[]` after the recreate; the
wire test required re-running `ops/debezium/register-connector.sh`. Known
ops behavior (AGENTS.md), confirmed again.

### Legacy - batch late-arrival forward date-shift

The legacy generic late-arrival injector moves event timestamps FORWARD
(up to late\_arrival\_max\_days), bleeding rows into the next month and
breaking the emitter's within-month day slicing. Scheduled for removal by
the Problem-interface refactor; emitter tests currently isolate it off.

---

## 4. Open decisions (for the next sprint)

1. Raw OLTP keying policy (F1): upsert vs snapshot-keyed raw rows.
2. Lane-scoped problem application (F2): transport injection out of
   `_generate_month` into the batch path - pull the Problem-interface
   split forward.
3. Replay -> topic-level re-emit (F3), burst -> distinct-id scaling (F4).
4. Sink logging config (F5).
5. Guard the connector re-registration in the ops flow (F6).
