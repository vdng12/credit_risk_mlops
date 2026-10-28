# Improve the Data Generator - 2026-10-28

Plan to upgrade the credit-risk simulator into a rubric-ready data generator:
deterministic, scenario-configurable, with controlled data-quality problems,
a drift scenario, and ground-truth reporting.

**Related docs**
- `docs/characteristics.md` - dataset description, offline/streaming division, problem mapping
- `docs/data_problem_statement_20261028.md` - problem-injection principles (extracted from the RecSys reference)
- `docs/coursework_gap_analysis_20261028.md` - rubric gap analysis (edai-1: ~17/100 pts in hand)
- `docs/statistics.md` - t-copula statistical method

---

## 1. Goals

Cover the rubric's "Implement Data Generator" (edai-1) and the final-coursework
"Improve the Data Generator" items:

1. **Deterministic, resumable, idempotent generation** - the seed is the state.
2. **One scenario = one YAML** - externalized config, subcommand entrypoint,
   calibration wired in.
3. **Two lanes** - offline (batch, parquet/S3) and streaming (CDC -> Redpanda),
   with the correct problems on each.
4. **The four offline problems** - skew, high cardinality (documented), schema
   evolution, exact duplicates (+ payload hash).
5. **The three streaming problems** - late arrival, duplicate replay, burst.
6. **Drift scenario** - deterministic controller, PD-logit shift, per-row
   metadata, PSI-ready.
7. **Ground truth everywhere** - ChallengeStats, injected-vs-observed report,
   characteristics report, manifest.

Out of scope this round: Spark/Flink jobs, DP1-3 pipelines, DataHub, storage
optimization, feature store (deferred - see Section 9).

---

## 2. Current state (facts)

- `Simulator` (core.py): t-copula -> per-client factor vectors; monthly ledger
  state machine (loans, DPD, bureau Markov, cards); per-month `MonthBatch`.
- Problems today: `ProblemInjector` (missing values, duplicates, outliers, late
  arrivals) - a generic noise layer, not the rubric's problem taxonomy.
- Quality: `QualityValidator` (FK + per-entity checks), `report.py` summaries.
- Sinks: one `SinkManager`; routing sets baked into `MonthBatch.to_sinks()`.
- Config: dataclasses, `SIM_*` env overrides; **no YAML**; `run_id` hardcoded.
- Calibration: `calibrate.py` -> `calibrated_copula.json`; `calibrate_structures.py`
  -> `calibrated_params.json` consumed by `Params`. **But `apply_to_config` has
  zero callers** - the copula calibration is never merged into a run.
- CLI: flat argparse (`--n-customers/--months/--backfill-until/...`);
  **backfill writes nothing** (in-memory only); no resume; no bookmark.

### Known gaps (from gap analysis)

| Gap | Status |
|---|---|
| Skew / cardinality / volume report | [ ] (t-copula skews exist, undocumented) |
| Schema evolution over time | [ ] |
| Offline duplicates (payments/apps) | [ ] |
| Streaming problems (late/replay/burst) | [ ] (late exists as batch date-shift only) |
| Deterministic replay / resume | [ ] (sequential RNG; in-memory ledger) |
| YAML config + subcommands | [ ] |
| Calibration wired into runs | [ ] (`apply_to_config` uncalled) |
| DQ report (injected vs observed) + manifest | [ ] |
| Drift scenario | [ ] |

---

## 3. Architecture (the connected picture)

```
                     ONE CONFIG (YAML) + ONE SEED
                     +------------------------------+
                     | seed | run_id | calibration |
                     | offline{} streaming{} drift{}|
                     +--------------+---------------+
                                    |  subcommands: generate / validate / resume / calibrate
                     +--------------v---------------+
                     |  PHASE 0 - INDEXED DETERMINISM|
                     |  every draw = f(seed, entity,  |
                     |  month, day, slot) -> months   |
                     |  are pure functions            |
                     +--------------+-----------------+
                                    |
                     +--------------v-----------------+
                     |  SIMULATOR = CLEAN DATA FACTORY|
                     |  t-copula factors | monthly    |
                     |  ledger (loans, DPD, bureau)   |
                     |  no problems                   |
                     +-------+----------+-------------+
                             |          |
             +---------------+v-+   +---v------------------+
             | OFFLINE PATH      |   | ONLINE PATH           |
             | [start, live_start)|   | [live_start, now]     |
             | all entities      |   | stream entities,      |
             |                   |   | pull-based per-day    |
             | problems: skew,   |   | emitter (grain =      |
             | schema evolution, |   | choice)               |
             | duplicate         |   | problems: late,       |
             |                   |   | replay, burst         |
             +---------+---------+   +---------+-------------+
                       |                        |
             +---------v---------+   +---------v-------------+
             | sinks: parquet+S3 |   | sink: postgres OLTP   |
             | (batch facts)     |   | -> Debezium CDC       |
             |                   |   | -> Redpanda topics    |
             +---------+---------+   +---------+-------------+
                       |                        |
                       +-----------+------------+
                                   | union: disjoint time ranges, same stream
                     +-------------v-------------+
                     | FEATURE STORE / DP1-3     |  == DEFERRED ==
                     | parquet + CDC             |
                     +-------------+-------------+
                                   |
                     +-------------v-------------+
                     | Spark/Flink, Dagster,     |  == FUTURE ==
                     | DataHub, storage          |
                     +---------------------------+
```

### Lane terminology

- **Offline** = consumed by batch jobs (Spark/DP), Parquet on S3; history +
  monthly/nightly artifacts.
- **Streaming** = consumed by stream processors (Flink), CDC topics; recurring
  events.
- **Cadence is orthogonal**: payments/decisions are nightly files *and* streamed;
  balances/bureau are monthly snapshots consumed by batch only. See
  `docs/characteristics.md`.

---

## 4. Build phases

### Phase 0 - Indexed determinism (foundation)

Replace the sequential RNG with per-entity deterministic streams:
`draw = np.random.default_rng(hash(seed, entity, month, day, slot))`.
~15 call sites in `state.py`/`core.py` (hazard draws, Markov transitions, card
utilization, application draws).

Consequences:
- months are pure functions -> backfill == live, replay == byte-identical
- idempotent months -> retries/resume safe
- bookmark = one row `(run_id, seed, last_month, config_hash)` in postgres-source
  (optional - recomputation always possible)
- regression test: re-running a month from the seed produces identical output

### Phase 1 - Config + entrypoint

- Subcommands: `generate`, `validate`, `resume`, `calibrate`.
- YAML scenario (`configs/simulator/*.yaml`) mirroring the dataclasses:
  `seed`, `run_id`, `time`, `entities`, `copula`, `problems{offline, streaming}`, `drift`, `sinks`.
- Precedence: **CLI > env (SIM_*) > YAML > calibration > defaults**.
- Config hash in the manifest: `sha256(json.dumps(asdict(config), sort_keys=True))`.
- Fix CLI gaps: backfill months written (batch lane), `run_id` first-class.

### Phase 2 - Calibration wiring

- `calibrate` subcommand runs both stages (copula + structures) ->
  `configs/simulator/calibration/copula.json` + `params.json`.
- Config loader calls the currently-orphaned `apply_to_config()` (defaults ->
  calibration -> YAML -> env -> CLI).
- `Params.load()` path made overridable via config; manifest records
  `params.source` ("calibrated" vs "fallback").

### Phase 3 - Path isolation + sinks package

- `paths/offline.py`, `paths/online.py`; `sinks/parquet.py`, `sinks/s3.py`,
  `sinks/postgres.py`. Routing leaves `core.py`.
- Offline path: backfill history + cold artifacts -> parquet/S3.
- Online path: **pull-based per-day hot emitter** (grain = runtime choice:
  monthly / daily / per-tick) -> streaming problems -> postgres -> CDC.
- Live stage has **three rhythms**: hot stream (per-day/tick), **nightly batch**
  (payments file, decisions), **monthly close** (bureau drop lag 21d, balances,
  clients master) - the existing cadence knobs (`payment_file_hour_utc`,
  `bureau_drop_lag_days`) already encode this.
- Seam at `live_start_date`: **bootstrap dimensions** (clients, cards, loans)
  into postgres once, then live emission. No data replay.

### Phase 4 - Offline problems (on the batch/history lane)

| Problem | Injection | Details |
|---|---|---|
| Skew | generation-time | configurable `top_*_ratio` (top region, top segment), Pareto/heavy-tail exposure, preference-weighted selection - layered on t-copula marginals |
| High cardinality | documented characteristic | no key changes; cardinality ratios in the characteristics report (`bureau_id`/`auth_id`/`payment_id` ~ unique per row) |
| Schema evolution | bureau only | v1 omits new field (null), non-breaking at `change_date`, breaking later; per-row `schema_version`; validator tolerates legacy-null |
| Exact duplicate | payments 1.5-2%, applications 0.5% | per-month Bernoulli; byte-identical clones (all invariants preserved) |
| Payload hash | DQ-report time | content hash over row fields; used for observed duplicate count + dedup detection |

### Phase 5 - Streaming problems (on the online lane)

| Problem | Injection | Details |
|---|---|---|
| Late arrival | emit-time | event timestamp shifted into the past (5-45 min), ingestion = now; `late` flag + delay seconds recorded |
| Duplicate replay | emit-time | whole bundle re-delivered: original event timestamps, new ingestion/created timestamps; occupies an emit slot |
| Burst traffic | tick rule | periodic Nx volume multiplier (e.g. month-end closings); pure tick rule, no RNG |

### Phase 6 - Ground truth + reports

- `ChallengeStats` per month + per run (duplicates injected, version cohorts,
  per-entity rates).
- `data_quality_report.json` - **injected vs observed** (hash-dup count, null
  legacy fields, skew ratios, cardinality ratios, volume) - in
  `outputs/generated/<run_id>/`.
- `manifest.json` - seed, config hash, calibration source, row counts, paths.
- Characteristics report (JSON + markdown): which columns skewed + ratios,
  cardinality ratios, schema before/after, duplicate rate, volume per entity.
- Validation stays green: duplicates preserve all fields; schema evolution
  only nulls the new bureau field; IDs never null; drift metadata recomputed
  from timestamps must match.

### Phase 7 - Drift scenario

- Deterministic controller (reference pattern): `drift_factor(date)` gradual
  (ramp) / abrupt (step), phases `baseline` / `pre_drift` / `post_drift`.
- Applied to **one calibrated knob**: $P_{eff}(f_6, t) = \mathrm{logit}^{-1}\left(\mathrm{logit}(P_{base}(f_6)) \cdot factor(t)\right)$
  - preserves ranking, drift emerges cohort-by-cohort in DPD/labels/features.
- Every loan/application stamped: `drift_enabled`, `drift_scenario`,
  `drift_phase`, `drift_factor`. Validator re-derives and checks.
- Later (final coursework): PSI of PD-score distribution vs baseline window ->
  drift detection API + monitoring.

---

## 5. Timeline / run lifecycle

| Step | When | What happens |
|---|---|---|
| 1. calibrate | one-time (HC data available) | copula + params -> `calibration/` artifacts |
| 2. backfill | one batch run | months `[start, live_start)` -> offline problems -> validate -> **parquet/S3** |
| 3. bootstrap | once at boundary | dimensions (clients, cards, loans) -> postgres |
| 4. live | recurring (daily/tick) | hot stream per grain + nightly batch + monthly close; resume via bookmark |
| 5. consume | continuous | Spark/Flink read the union; DP1-3; feature store (deferred) |

---

## 6. Config YAML sketch

```yaml
seed: 42
run_id: challenges_v1

time:
  start_date: 2020-01-01
  bureau_drop_lag_days: 21
  payment_file_hour_utc: 2

entities:
  n_customers: 50000
  target_default_rate: 0.08
  currency: VND

copula:
  nu: 30.0            # omitted -> calibration artifact -> defaults
  corr_matrix: [...]  # omitted -> calibration artifact -> defaults

offline:
  problems:
    skew:
      top_region_ratio: 0.35
    schema_evolution:
      change_date: 2022-06-01
      breaking_change_date: 2023-01-01
    exact_duplicate:
      payments_rate: 0.02
      applications_rate: 0.005

streaming:
  emitter:
    grain: daily              # or per-tick
    live_start_date: 2023-06-01
  problems:
    late_arrival: { rate: 0.02, delay_minutes_min: 5, delay_minutes_max: 45 }
    duplicate_replay: { rate: 0.015, history_size: 1000 }
    burst_traffic: { every_n_ticks: 30, multiplier: 3 }

drift:
  enabled: true
  scenario: economic_downturn
  drift_start_date: 2022-06-01
  drift_mode: gradual
  default_propensity_multiplier: 2.0
  ramp_up_days: 90
  baseline_start_date: 2020-01-01
  baseline_end_date: 2022-05-31

sinks:
  enable_s3: true
  enable_postgres: true
  # DSNs stay in env (SIM_POSTGRES_DSN, SIM_S3_*)
```

---

## 7. Deferred details (future work)

### Feature store union + DP pipelines

The DP1-3 pipelines (Dagster, not Airflow - our tool of choice) consume the
two lanes as a **union**: parquet + CDC, disjoint time ranges, identical stream
identity (Phase 0 guarantees). The feature store computes as-of windows over
the union. Three required handlers (the rubric's graded jobs):

1. content-hash dedup for batch duplicates (Spark "handle duplicates")
2. schema-version normalization for evolution (Spark "handle schema evolution")
3. ingestion-time as-of / watermark for late arrivals (Flink "handle late arrival")

Boundary correctness: 90-day windows spanning `live_start_date` are covered by
the union; a materialize job (offline->online incremental) syncs the seam.
The generator must expose: `live_start_date`, shared seed/config, full-history
batch lane, dimension bootstrap.

### Other deferred items

- Spark/Flink jobs with baseline->optimized documentation (Spark/Flink UI)
- Dagster orchestration: backfill job (one-shot), nightly job (hot-day emit +
  cold-day close), monthly job (bureau/balances/ledger close)
- DataHub lineage + data validation per DP
- Storage optimization (Iceberg compaction / z-order via polaris, indexing)
- Novel ideas (2 x 10 pts edai-1; 2 x 2 pts elsewhere) - candidates: Dagster
  instead of Airflow, deterministic-indexed RNG as a documented technique

---

## 8. Acceptance criteria (this round)

- [ ] Re-running a month from the seed produces byte-identical output (Phase 0 test)
- [ ] `uv run simulator calibrate --data-dir ...` writes calibration artifacts;
      `generate --config <yaml>` merges calibration + YAML in precedence order
- [ ] Backfill months persist to the batch lane; `run_id` is scenario-keyed
- [ ] Offline problems inject skew, schema evolution (bureau), exact duplicates;
      validator stays green
- [ ] Streaming problems inject late/replay/burst on the online lane
- [ ] `data_quality_report.json` shows injected vs observed; characteristics
      report lists skew/cardinality/schema/duplicate/volume metrics
- [ ] Drift scenario: controller deterministic, per-row metadata verified,
      PD-shift emerges in labels
- [ ] Existing statistical regression tests still pass