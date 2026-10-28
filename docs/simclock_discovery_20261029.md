# SimClock Discovery

Discovery note for `simulator/src/simulator/clock.py` - what the pretend
clock was designed to do, what it actually does, what is broken, and how it
plugs into the planned online path.

---

## 1. TL;DR

`SimClock` is a **calendar-aware pretend clock**: it advances over simulated
dates and fires booleans when month / bureau / payment **boundaries** occur.
It is the batch-domain analogue of the reference data generator's tick loop
(in the RecSys producer: `tick += 1` per emission round with
`BurstTrafficProblem.events_for_tick(tick, n)`).

**Current state: dormant.** The `Simulator` constructs it
(`core.py: self.clock = SimClock(...)`) but NOTHING ever calls its methods.
Month generation is hand-rolled in `backfill`/`play_forward` via
`_next_month_date`, which bypasses the clock entirely. The clock is the
"lost in translation" remainder of an intended day-granularity driver.

---

## 2. State (what the clock remembers)

```python
class SimClock:
    def __init__(self, start_date, bureau_lag_days=21):
        self.start_date = start_date
        self.current_date = start_date        # the pretend "now"
        self.bureau_lag_days = bureau_lag_days
        self._last_month_processed = None     # last time a monthly boundary fired
        self._last_bureau_processed = None   # last time a bureau boundary fired
        self._last_payment_processed = None  # last time a payment boundary fired
```

The clock is **stateful, not derived**: boundaries are detected by comparing
`current_date` against the "last processed" markers, and `tick()` updates
those markers when boundaries fire.

---

## 3. The three boundary rules

| Property          | Rule                                                   | Models |
|-------------------|--------------------------------------------------------|--------|
| `is_month_boundary`  | current month != last processed month              | a new calendar month started |
| `is_payment_boundary`| current_date > last payment processed date         | a new day arrived (nightly file) |
| `is_bureau_boundary` | (current_date - 21d)'s month != last bureau month  | the vendor drop, as-of 21 days ago |

### The bureau lag (the interesting one)

Bureau data comes from an external vendor as a **monthly extract**. The
extract for month M is *published about 21 days into M* (snapshot as-of the
1st, delivered on the ~22nd). So "is there a new bureau drop?" is answered
on the **lagged date**, not the current date:

```
month 1 (Jan)                month 2 (Feb)
|----|--B1--|                |----|--B2--|
1    21    22                1    21    22
      ^ drop for Jan          ^ drop for Feb
```

`is_bureau_boundary` goes True on the first date where
`current_date - 21 days` enters a new month - i.e. roughly the 22nd.
The boundary *count* per tick call is still one boolean (see section 5).

---

## 4. `tick(days)` - the heartbeat

```python
def tick(self, days=1) -> dict[str, bool]:
    self.current_date += timedelta(days=days)   # advance pretend-now
    monthly = self.is_month_boundary
    bureau  = self.is_bureau_boundary
    payment = self.is_payment_boundary
    if monthly: self._last_month_processed  = self.current_date.replace(day=1)
    if bureau:  self._last_bureau_processed  = self.current_date
    if payment: self._last_payment_processed = self.current_date
    return {"monthly": monthly, "bureau": bureau, "payment": payment}
```

- Advances the calendar, recomputes all three flags, updates the markers
  for whichever boundaries fired, returns the flags as a dict.
- The caller reacts: e.g. `if flags["bureau"]: emit_bureau_drop(...)`.

---

## 5. Boundary flags are booleans, not counts

`tick(40)` that jumps over two months sets `monthly=True` **once** even
though two month boundaries were crossed. The design is "one boundary event
per tick call" - the caller decides the granularity it needs. This matters
for replay/resume: if the clock ever drives generation, a resumed run must
re-tick day by day to re-fire the same boundary sequence, or the flags must
be derived from the register of (month, bureau-cycle, day).

---

## 6. `tick_to_month_end()` - fast-forward to month end

Computes the remaining days in the current month and ticks exactly that far:

```
current = Jan 15
  next_month = (Jan 28) + 4d = Feb 1     # any month's 28th + 4d >= next month
  last_day   = Feb 1 - 1d  = Jan 31
  days       = (Jan 31 - Jan 15) + 1 = 17
  tick(17)  -> lands on Jan 31, monthly=True
```

Use case: a batch loop that wants "advance one full simulated month and
land exactly at month end."

---

## 7. `month_iterator(n)` - broken

```python
def month_iterator(self, n_months: int) -> Iterator[date]:
    for _ in range(n_months):
        self.tick_to_month_end()
```

It ticks to month-end n times but **never yields** - despite the
`Iterator[date]` annotation it is an empty generator. Almost certainly
meant to be `yield self.current_month` after each tick. This is the
clearest sign the clock was abandoned mid-design: someone started the
honest-clock path and hand-rolled the month loop instead.

**Fix when revived:** `yield self.current_month` (and decide whether the
yielded date is the month just completed or the next month to start).

---

## 8. How the clock was meant to drive generation

```
INTENDED (day-granularity driver)          ACTUAL (current code)

sim.clock = SimClock(start, lag)           backfill(until) / play_forward(n):
loop:                                      month = next via _next_month_date()
    flags = clock.tick(days=1)             batch = _generate_month(month)
    if flags["monthly"]:                       # everything in one call:
        month_close(ledger, month)             #   applications + auths (hot)
        emit_month_entities()                 #   payments, balances, bureau
    if flags["bureau"]:                        # (bureau lag ignored)
        emit_bureau_drop(lagged_month)
    if flags["payment"]:
        emit_nightly_payment_file()
```

The clock gives **day granularity with correct relative offsets** (bureau on
the ~22nd, payments nightly, month close at month-end). The current code
collapses all of it into one monthly call, so the clock's value is zero
until the online path / pull-based emitter is built.

---

## 9. Lineage from the reference code

The RecSys streaming producer uses a plain integer tick:

```python
tick += 1
events_this_tick = problems.burst.events_for_tick(tick, events_per_tick)
now = datetime.now(utc)
replay = problems.replay(now)          # duplicate replay
timing = problems.event_time(now)      # late arrival
rows = factory.create(counter, now, timing.event_timestamp)
write_bundle(cursor, rows)             # postgres -> CDC
```

|             | reference producer         | SimClock                          |
|-------------|----------------------------|-----------------------------------|
| tick unit   | plain int counter          | calendar days                     |
| driver      | event volume + burst       | month/bureau/payment boundaries   |
| time source | real wall clock (`now`)    | pretend simulated dates           |
| grain       | per-event emission         | day ticks aggregated into months  |

The conceptual model is the same - a pretend clock pacing a stream of
generated data - at different granularity. When the online path is built,
`burst_traffic.every_n_ticks` maps to "every Nth tick/day/hour", and the
clock decides what to flush.

---

## 10. How it fits the plan (online path / three rhythms)

The plan's online lane has three rhythms, and each maps to a clock boundary:

| Rhythm                 | Clock boundary          | Scheduled job (later)     |
|------------------------|-------------------------|---------------------------|
| hot stream (per day)   | `tick(days=1)`          | per-day emit              |
| nightly batch          | `is_payment_boundary`   | nightly payment file      |
| monthly close          | `is_month_boundary`     | bureau drop (start+lag)   |
|                        | `is_bureau_boundary`    | balances + ledger advance  |

The `_STREAM_OLTP` / `_S3_COLD_FACTS` routing in `simulator/paths/` is
orthogonal to this: the paths decide *where* each emitted frame lands; the
clock decides *when* each frame is produced.

### Determinism and the clock

Under Phase 0 (indexed RNG), month content is a pure function of
(seed, entity, month). The clock's markers are **timeline position, not
data**. For batch replay the markers are cosmetic (re-derivable from the
month). For a resumed live run they matter: a resumed emitter must recreate
the same boundary sequence - either by re-ticking from the last processed
date, or by deriving "is it a bureau day / month end" from the date alone
(pure functions of the date, which is cleaner and restart-proof).

---

## 11. Status and decision points

Known issues:

1. `month_iterator` yields nothing (should `yield self.current_month`).
2. `bureau_drop_lag_days` is configured and passed into the clock but not
   enforced anywhere - bureau emit uses the month directly.
3. Boundary flags are booleans (one event per tick call), not counters;
   multi-period jumps collapse to one event.
4. No caller exists - `backfill`/`play_forward` bypass the clock.

Plan (when the online path is built):

- [ ] Fix `month_iterator` to yield the completed month
- [ ] Wire `tick(days=1)` into the pull-based emitter; flush hot/nightly/
      monthly on the three flags
- [ ] Enforce the 21-day bureau lag in the bureau drop emission
- [ ] Prefer date-derived boundary predicates (e.g. `is_bureau_day(date)`)
      for restart-proof resume; keep the clock for pacing only
