"""Deterministic per-(entity, month) random streams.

Phase 0 (indexed determinism): instead of one shared sequential RNG tape,
every draw site derives its generator from ``(root seed, entity, month)``.
Consequences:

- a month's random content is a pure function of (seed, entity, month) -
  it does not depend on how many draws happened in earlier months
- re-running a month reproduces byte-identical output (idempotent months)
- months can be generated in any order / in parallel / on retry
- the live emitter can pull per-day without disturbing other streams
- a resume only needs a bookmark (seed + last month), never the tape
"""

from __future__ import annotations

import hashlib
from datetime import date

import numpy as np


def month_rng(root_seed: int, entity: str, month: date) -> np.random.Generator:
    """RFC-style derived generator for one (entity, month) stream.

    Fixed hash of ``root_seed:entity:ISO-month`` -> a stable 64-bit seed,
    so the same inputs always yield the same generator (and orders of
    generation inside one entity/month are the only thing left to guard).
    """
    key = f"{root_seed}:{entity}:{month.isoformat()}"
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "big"))


def month_seed(root_seed: int, entity: str, month: date) -> int:
    """Return just the derived seed (for tests / independent streams)."""
    return int.from_bytes(
        hashlib.sha256(f"{root_seed}:{entity}:{month.isoformat()}".encode("utf-8")
                       ).digest()[:8], "big"
    )
