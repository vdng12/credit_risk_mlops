# Data Problem Statement - 2026-10-28

## 0. Purpose and provenance

This document extracts the **core data generator requirements** and
**data-quality problem injection principles** from the reference
implementation:

> **Reference**: `RecSys-MLops/apps/data-platform/data-generator` (deterministic
> historical + streaming synthetic data generator with deliberate data-quality
> challenge injection and drift simulation). Domain: recommender system
> (users, sessions, impressions, orders, behavior events).

The reference is **domain-specific in content but domain-neutral in principle**.
This statement intentionally captures only the requirements and principles -
**not the code design** (no class/module structure, no implementation shapes).
Section 3 cross-references each principle against this repository's
credit-risk simulator (`simulator/`).

Guiding idea of the reference: *generate clean, relationally consistent data
first, then inject controlled, measurable problems* - so every artifact of
dirtiness has known ground truth and can be scored.

---

## 1. Core data generator requirements (domain-neutral)

### 1.1 Determinism and reproducibility
- One root seed drives the entire run; every stochastic choice (including
  problem injection and streaming problems) derives from that seed.
- Output ordering is deterministic (sorted by ingestion time + stable ID tiebreak).
- Each run records its seed, full resolved config, and a config hash in a
  manifest, so a run is fully reproducible and auditable.

### 1.2 Clean-first, relationally consistent generation
- Generate **clean** records first: all foreign keys resolve, referential
  integrity holds across the entity graph, and business invariants are exact
  (e.g., aggregated totals equal the sum of their line items).
- Contamination is a **separate, later stage** - never mixed into generation.

### 1.3 Canonical dimension metadata
- Denormalized fields stamped on fact rows (category, brand, price bucket)
  must **match the canonical dimension row** - downstream joins and aggregates
  must not disagree with the reference dimensions.
- All generated business IDs are positive; a reserved value (0 / empty) is used
  for "unknown/pad" and never collides with real IDs.

### 1.4 Volume control with tolerance and calibration
- The generator targets a configured event volume and stops cleanly at a
  natural boundary (end of a session/unit), so exact counts drift within a
  configured tolerance.
- **Targets are defined on the boundary consumers see (post-injection)**:
  the clean generation volume is calibrated *down* by the expected injected
  duplicate volume so the observed (dirty) total lands inside tolerance.

### 1.5 Schema versioning
- Data evolves over time: a **non-breaking** schema change at one date, an
  optional **breaking** change later, each row carrying an explicit
  `schema_version`.
- Legacy rows **omit** the new fields (null), not fill them with junk - so
  downstream can detect and handle both old and new shapes.
- Artifacts are version-tagged (feature version constants), and schema
  validation has an explicit legacy-mode escape hatch when the versioned
  feature is disabled.

### 1.6 Drift scenario as a first-class concept
- Drift is a **scenario knob, separate from data-quality problems**.
- A deterministic controller maps any business timestamp to:
  - a **phase** (`disabled` / `baseline` / `pre_drift` / `post_drift`), and
  - a **factor** (effective multiplier), in **gradual** (linear ramp over N
    days, capped) or **abrupt** (step) mode.
- The factor applies to **one transition probability** (e.g., cart->purchase),
  never by rewriting feature values - so feature/outcome drift *emerges*
  naturally from upstream behavior.
- Every affected row is stamped with drift metadata (enabled, scenario,
  phase, factor) so persistence can be verified against the controller
  (recomputed from the timestamp).

### 1.7 Drift observability
- Per-user daily **rolling features** are materialized point-in-time (90-day
  windows), and monitored daily per feature:
  - mean / stddev,
  - **PSI vs the pooled baseline distribution** (buckets learned from the
    baseline quantiles, with explicit protection against all-zero and
    duplicate breakpoints),
  - a severity classification from PSI thresholds (stable -> detected ->
    strong -> alert),
  - the configured factor for that date.
- **Alerts** are emitted only for rows at/after the drift start, crossing the
  configured PSI alert threshold.
- A **summary** compares baseline vs post-ramp conversion rates, factor
  min/max, max PSI, and alert count.

### 1.8 Ground truth: manifest and data-quality report
- **Manifest**: seed, config hash, full resolved config, row counts, schema
  versions, artifact paths.
- **Data-quality report**: **injected** ground truth (what problems were
  applied, with counts) **side-by-side with observed** signals (what a
  consumer would see in the dirty data: duplicates, null legacy fields,
  skew ratios). The injected/observed split is what makes the dataset a
  scorable benchmark.

### 1.9 Two lanes, one seed
- **Historical/batch lane**: writes partitioned explicit-schema Parquet,
  cold-first.
- **Streaming lane**: writes row-level to a database (which may be
  CDC-published downstream), with the **same root seed and scenario config**.
- Both lanes share problem taxonomy where it makes sense; streaming-specific
  problems exist only in the live lane.

---

## 2. Problem injection principles

### 2.1 Clean-then-inject separation
The clean dataset is the ground truth. Problems are applied afterwards, in a
pipeline whose **order is explicit and deterministic** (later problems see the
output of earlier ones). Never mutate generation logic to produce dirt.

### 2.2 Problems map to real-world failure modes
Each problem is a faithful model of a real pipeline failure, not random noise:

| Problem | Real-world failure it models |
|---|---|
| **Exact duplicate** | at-least-once delivery / retries: identical row (same ID and content hash) repeated |
| **High-cardinality keys** | opaque, collision-resistant natural keys (deterministic but UUID-like), so naive sequential-key assumptions fail |
| **Schema evolution** | producers upgrading over time: old rows missing new fields, then a breaking version change |
| **Skew** | uneven real-world distributions: top-value bias (city/category), heavy-tailed popularity (Pareto), and exposure weighting that amplifies popular/preferred items |
| **Payload hash** | content-derived identity: the hash must reflect the row *including* its schema version, so "same content" is well-defined |
| **Streaming: burst traffic** | load spikes: periodic `Nx` volume multiplier |
| **Streaming: duplicate replay** | re-delivery of a whole bundle: original event timestamps kept, ingestion/created timestamps rewritten to now |
| **Streaming: late arrival** | out-of-order data / watermark pressure: event timestamp shifted into the past, ingestion time unchanged |

### 2.3 Injection preserves the invariants being validated
Injected problems must **not break the invariants that are supposed to hold**.
Duplicates are byte-identical (all FKs, totals, drift metadata intact); schema
evolution nulls new fields rather than corrupting existing ones. This keeps
validation meaningful: the validator's failure signals can be attributed to
genuine breakage, never to the injection itself.

### 2.4 Problems are detectable - but not trivially
Each problem is constructed so the *right* downstream technique detects it:
content-hash dedup finds exact duplicates; point-in-time re-derivation catches
rolling-feature mismatch; version columns catch schema cohorts; monotonicity
and bounds catch drift factors. The intent is to exercise the consumer's
detection machinery, not to hide the problems.

### 2.5 Calibration on post-injection volumes
Configured targets describe what the consumer sees (post-injection). The
generation stage compensates so the sum of clean + injected lands within
tolerance. Reporting the post-injection count to the validator keeps the
contract consistent.

### 2.6 Deterministic output ordering
The dirty output is sorted deterministically (ingestion time, stable ID
tiebreak), so duplicates are adjacent to their originals and downstream
comparisons are reproducible regardless of insertion order.

### 2.7 Ground-truth instrumentation is separate from observed signals
The injector records ground truth (clean count, duplicates injected, version
cohort counts; for streaming: late flags and delay seconds) as **metadata
alongside** the dirty data. The report then juxtaposes injected vs observed -
this is the scoring contract.

### 2.8 Canonicalization before aggregation
When computing "true" metrics over dirty data, deduplicate by content identity
(event ID + payload hash) before aggregating. Aggregate over canonical rows;
report raw row counts separately.

### 2.9 Explicit legacy handling
When versioned features are disabled, schema and artifact validation fall back
to the legacy shape explicitly - no silent assumption that the current shape
is the only shape.

---

## 3. Cross-reference: mapping to this repository's credit-risk simulator

The reference principles above are domain-neutral; this repo's simulator
(`simulator/`) is a different domain (lending default risk: t-copula with 7
latent factors -> clients, applications, bureau, loans, DPD paths -> labels).
Mapping principle -> current state -> gap:

| Principle (section) | Credit-risk simulator today | Gap |
|---|---|---|
| 1.1 Determinism | Root seed; t-copula latent vectors stable per client | No run manifest (seed/config hash) emitted per run |
| 1.2 Clean, FK-consistent | 8 entity types with child-table links; label windowing | No *explicit* validation stage for FK/invariant checks at generation time (tests are statistical only) |
| 1.3 Canonical metadata | Application features derived from latent factors | Denormalized fact metadata (e.g., bureau STATUS vs client features) not cross-checked against canonical sources |
| 1.4 Volume + calibration | Default-rate knob solved by `brentq` so approval-weighted mean hits target | Target is a rate, not a post-injection count; no tolerance band / boundary-stop semantics |
| 1.5 Schema versioning | Gold `application_master` layer exists, no CDC | No versioned schema evolution over calendar time; bureau STATUS is static-shaped |
| 1.6 Drift scenario | None | No economy/macro regime knob (phase + factor over time) affecting PD curves |
| 1.7 Drift observability | None | No per-day feature health / PSI vs baseline monitoring artifacts |
| 1.8 Manifest + DQ report | Parquet write-through + S3/Postgres sinks only | No `manifest.json` / `data_quality_report.json` with injected-vs-observed |
| 1.9 Two lanes | Yes: local parquet + Postgres OLTP -> Debezium CDC -> Redpanda; S3 snapshots | Streaming lane has no problem injection (no burst/replay/late) |
| 2.1-2.9 Injection | None | No problem injector at all; no ground-truth stats per run |

### Natural ports to the credit-risk domain (principles, not design)
- **Exact duplicate** -> duplicate loan-application / payment rows (same natural
  key + content hash) - models retried ingest at the CDC boundary.
- **High-cardinality keys** -> opaque UUID-style bureau record IDs instead of
  sequential ints.
- **Schema evolution** -> bureau/application producers upgrading over time:
  v1 omits newly-added fields; a breaking version later; per-row
  `schema_version` with a gold-layer legacy escape hatch.
- **Skew** -> regional/segment top-value bias (e.g., top-city share), heavy-tail
  exposure, and preference-weighted candidate selection - all currently
  uniform-ish under the copula.
- **Payload hash** -> content-derived identity for dedup (event_id +
  payload_hash) so downstream "same event" is well-defined across schema
  versions.
- **Streaming burst/replay/late** -> bursty CDC batches, re-delivered
  transaction bundles with old business time + new ingestion time, and late
  payments/default-markers arriving outside their window - directly testable
  against the existing Redpanda CDC topics.
- **Drift scenario** -> an economic-regime factor applied to the PD curve
  (single probability knob), with per-month PD-score PSI vs a baseline window,
  and per-row drift metadata verified against a deterministic controller.
- **Validation stage** -> a generation-time invariant validator (FKs, amount
  math, timestamps, drift metadata recomputation) that runs on the *dirtied*
  output, plus injected-vs-observed DQ reporting.

---

## 4. Summary

The reference's core proposition: **a deterministic clean generator + an
explicit, ordered, ground-truth-instrumented problem injector + a validator
that understands the difference between injected artifacts and real breakage**.
The credit-risk simulator already has the hard part - a calibrated, realistic
clean generator - but none of the challenge-injection, drift-scenario,
monitoring, or ground-truth reporting layers. Porting the *principles* (not
the RecSys domain shapes) would turn the simulator into a scorable, end-to-end
data-quality benchmark for the lending pipeline, aligned with the existing
Kafka/CDC streaming lane.