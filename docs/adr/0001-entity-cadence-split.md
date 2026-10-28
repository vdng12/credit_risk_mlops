# Split entity delivery into daily / nightly / monthly cadences via explicit lanes

## Status

Accepted (2026-10-29) - decided on branch `agent/datgen`, pending final
review on `main`.

## Context

The initial model wrote **every entity once per month** through a single
routing method (`MonthBatch.to_sinks`), with the routing sets baked into
`core.py` and a monolithic `SinkManager`. That produced three pains:

1. A bank's real rhythm (transactions all day, a nightly payment file, a
   monthly bureau drop) was invisible - everything landed in monthly
   clumps, so the CDC stream was not "real-time" to consumers.
2. Routing was not explicit or testable, and the hot/cold naming collided
   (`MonthBatch.hot_events()` = stream-shaped subset vs the CDC lane set).
3. Data problems were applied before any lane split, so offline
   corruption leaked into the stream lane (duplicates hitting the OLTP
   PKs).

Alternatives considered:

- **A. Single monthly flush (the old model).** Simple, but no per-day /
  nightly distinction and no testable routing contract. Rejected.
- **B. A dedicated per-tick event producer (reference-style), emitting hot
  events tick-by-tick.** Faithful to the reference, but nothing consumes
  the stream yet (no Flink), so a second producer would be dead weight.
  Rejected for now; the chosen design keeps the same emitter shape so B
  remains a drop-in later.
- **C. Replay the history tail into the live lane to close the seam.** Data
  duplication and dedup burden for no benefit; the feature-store union
  (deferred) handles the seam instead. Rejected.
- **D. SimClock-driven day-granular dispatch with three rhythms (chosen).**
  Reuses the generator's clean monthly frames, re-paces delivery per day
  via the clock's boundary flags, and keeps deterministic identity with
  the batch lane (Phase 0). Also makes the cadence unit-testable (one
  nightly file per day, one monthly close per month).

## Decision

Separate **when an entity is delivered** from **which entity it is**:

- Delivery is split into three rhythms driven by SimClock boundary flags:
  a **daily hot** stream, a **nightly batch** (payment + decision files),
  and a **monthly close** (balances, bureau, clients, gold master).
- Dispatch (`livestage.py`) hands pre-grouped payloads to explicit lanes
  (`paths/`): OfflinePath (parquet archive + S3) and OnlinePath (postgres
  -> Debezium CDC). Sinks stay dumb writers. Routing sets live in `paths/`
  as unit-tested constants - not in the batch class.
- The canonical routing (which cadence maps to which destination):

  | Rhythm | Entities | Destinations |
  |---|---|---|
  | daily hot | loan_applications, card_authorizations | online (postgres -> CDC) + archive; never S3 |
  | nightly | payments, application_decisions | S3 file + archive + online (CDC) |
  | monthly | clients, application_master, pos/card_balances, bureau | archive + S3 (+ gold -> online) |

- Problems are lane-scoped in principle: offline problems mutate only the
  batch artifacts; the stream lane emits clean rows and receives the
  streaming problems (pending follow-up, see Consequences).

## Consequences

### Good

- Cadence now maps 1:1 to reality and was verified against postgres, S3,
  and CDC in the wire test (22 writes/month, canonical routing intact).
- Per-day / nightly / monthly behavior is unit-testable via the clock's
  boundary predicates; nightly files demonstrably carry content (loan
  due-day spread).
- Operational cadence is a runtime knob (fast-forward / throttled /
  scheduled), so the batch and live paths share one codebase.

### Bad / follow-ons

- **F1 (new ADR needed)**: raw OLTP keying policy - monthly snapshot PK
  collisions and reruns need an upsert / snapshot-keyed raw row decision.
- **F2 (triggers the Problem-interface ADR)**: lane-scoped problem
  application is still pending; offline duplicates leak into the stream
  lane until the split is implemented.
- **F3/F4**: streaming replay should be a transport-level topic re-emit and
  burst should scale generation volume (distinct ids) - both currently
  collide with OLTP PKs or repeat identical rows.
- **F5/F6**: silence the SQLAlchemy executemany logger; re-register the
  Debezium connector after each cluster recreate.

All follow-ups are tracked in
`docs/datgentor_challenges_issues_20261029.md`.

## References

- Old model: `MonthBatch.to_sinks` + `SinkManager` (pre-Phase-3 git
  history, before b2786bb).
- New model: `simulator/paths/`, `simulator/livestage.py`,
  `docs/characteristics.md`, `docs/simclock_discovery_20261029.md`.
