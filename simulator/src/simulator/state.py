"""Account ledger and state machines: temporal + FK structure.

This is what makes child entities structurally dependent:

  * loans/cards/bureau accounts persist across months (FK integrity)
  * DPD and bureau STATUS evolve via fitted Markov/conditional models
  * each account stores its client's 7-factor vector at origination, so a
    high-default-propensity client stays risky month over month
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.random import Generator
from scipy.optimize import brentq
from scipy.special import expit, logit

from simulator.constants import LATENT_DEFAULT_PROPENSITY
from simulator.params import Params
from simulator.randstreams import month_rng


# ---------------------------------------------------------------------------
# Conditional GLM draws (outcome ~ copula factor scores)
# ---------------------------------------------------------------------------

def _binned_rate(
    model: dict,
    F: np.ndarray,
    factor_idx: dict[str, int],
) -> np.ndarray:
    """Return P(y=1) per row for a binned/WOE model (no sampling)."""
    f = F[:, factor_idx[model["factor"]]]
    bins = np.clip(np.searchsorted(model["edges"], f, side="right") - 1,
                   0, len(model["rates"]) - 1)
    rates = np.asarray(model["rates"], dtype=np.float64)
    return rates[bins]


class RunHazard:
    """Exact map between a per-month delinquency hazard h and the
    probability of default within a performance window.

    default := at least ``trigger`` consecutive delinquent months within
    the first ``n`` instalments (n = min(term, window)). With i.i.d.
    Bernoulli(h) months, g_n(h) = P(run >= trigger in n trials) is
    computed exactly by a small DP over the current run length, on a dense
    h grid. g_n is strictly increasing, so h = g_n^{-1}(PD) by interpolation.

    This replaces the old binary search over a shared intercept alpha
    (with beta = 0.24 taken from a per-instalment partial-payment proxy),
    which (a) used the wrong target (lifetime default vs. the windowed
    label), (b) used beta on the wrong scale (z-scored PD, not N(0,1)), and
    (c) capped the latent AUC at ~0.63 vs. ~0.75 for the real PD score.
    """

    def __init__(self, window: int, trigger: int = 3, grid_size: int = 4000):
        self.window, self.trigger = int(window), int(trigger)
        self.h = expit(np.linspace(-14.0, 9.0, grid_size))
        self.table = {n: self._g(n) for n in range(1, self.window + 1)}

    def _g(self, n: int) -> np.ndarray:
        r = self.trigger
        h = self.h
        state = np.zeros((len(h), r))   # P(run length = k, not yet absorbed)
        state[:, 0] = 1.0
        absorbed = np.zeros(len(h))
        for _ in range(n):
            new = np.zeros_like(state)
            new[:, 0] = state.sum(1) * (1 - h)
            new[:, 1:] = state[:, :-1] * h[:, None]
            absorbed = absorbed + state[:, -1] * h
            state = new
        return absorbed

    def hazard(self, pd: np.ndarray, n: np.ndarray) -> np.ndarray:
        pd = np.asarray(pd, dtype=np.float64)
        n = np.clip(np.asarray(n, dtype=int), 1, self.window)
        out = np.empty_like(pd)
        for nv in np.unique(n):
            m = n == nv
            g = self.table[int(nv)]
            if nv < self.trigger:            # default impossible in window
                out[m] = 0.0
                continue
            out[m] = np.interp(np.clip(pd[m], g[0], g[-1]), g, self.h)
        return out


def solve_default_shift(pd_base: np.ndarray, weights: np.ndarray,
                        target: float) -> float:
    """delta such that the weighted mean of expit(logit(pd) + delta) = target.

    A constant logit shift keeps the ranking (≈ AUC) of the real PD curve
    and moves only the level: this is the ``target_default_rate`` knob.
    Deterministic (no Monte-Carlo noise), monotone, solved with brentq.
    """
    lp = logit(np.clip(pd_base, 1e-9, 1 - 1e-9))
    w = weights / weights.sum()
    f = lambda d: float((w * expit(lp + d)).sum()) - target
    return float(brentq(f, -20.0, 20.0))


def draw_from_model(
    model: dict,
    F: np.ndarray,
    rng: Generator,
    factor_idx: dict[str, int],
) -> np.ndarray:
    """Sample outcomes conditionally on factor scores F (n x 7).

    model: dict with kind in {linear, logistic, poisson}, intercept, coef.
    """
    kind = model["kind"]

    if kind == "binned":
        rates = _binned_rate(model, F, factor_idx)
        return rng.random(len(F)) < rates

    lp = np.full(F.shape[0], np.float64(model["intercept"]))
    for fname, c in model.get("coef", {}).items():
        lp = lp + c * F[:, factor_idx[fname]]

    if kind == "linear":
        sigma = model.get("resid_sigma", 0.5)
        return lp + rng.standard_normal(len(lp)) * sigma
    if kind == "logistic":
        p = 1.0 / (1.0 + np.exp(-lp))
        return rng.random(len(p)) < p
    if kind == "poisson":
        return rng.poisson(np.exp(lp))
    raise ValueError(f"unknown kind {kind}")


# ---------------------------------------------------------------------------
# Markov chain for discrete states (bureau STATUS)
# ---------------------------------------------------------------------------

class MarkovChain:
    """Discrete-state chain, optionally with one transition matrix per
    factor bucket (``T`` of shape (K, S, S)) so account state evolution
    depends on the client's latent risk."""

    def __init__(self, states: list[str], T, rng: Generator) -> None:
        self.states = states
        T = np.asarray(T, dtype=np.float64)
        if T.ndim == 2:
            T = T[None]
        self.T = T / T.sum(axis=2, keepdims=True)
        self.rng = rng

    def sample_states(self, start_idx: np.ndarray, n_steps: int = 1,
                      bucket: np.ndarray | None = None) -> np.ndarray:
        cur = np.asarray(start_idx, dtype=int).copy()
        b = np.zeros_like(cur) if bucket is None else np.asarray(bucket, dtype=int)
        S = self.T.shape[1]
        for _ in range(n_steps):
            cum = np.cumsum(self.T[b, cur], axis=1)
            r = self.rng.random(len(cur))[:, None]
            # clamp: float round-off can leave cum[-1] < r
            cur = np.minimum((r > cum).sum(axis=1), S - 1)
        return cur

    def label(self, idx: np.ndarray) -> list[str]:
        return [self.states[i] for i in idx]


# ---------------------------------------------------------------------------
# Ledger records
# ---------------------------------------------------------------------------

@dataclass
class Loan:
    loan_id: str
    client_id: str
    factors: np.ndarray          # 7-factor vector at origination
    term_months: int
    principal: float
    annuity: float
    application_id: str = ""     # FK back to the application master (label re-join)
    hazard: float = 0.0          # P(delinquent month), from the PD curve
    months_paid: int = 0
    dpd_months: int = 0          # consecutive missed/partial months
    paid_off: bool = False       # True once months_paid reaches term_months


@dataclass
class Card:
    card_id: str
    client_id: str
    factors: np.ndarray
    limit: float
    utilization: float
    dpd_months: int = 0


@dataclass
class BureauAccount:
    bureau_id: str
    client_id: str
    factors: np.ndarray
    status_idx: int              # index into MarkovChain.states
    credit_sum: float
    credit_type: str = "Consumer credit"
    age_months: int = 0
    months_dpd_status: int = 0   # numeric part of STATUS when overdue (0-5)
    bucket: int = 0              # f6 bucket selecting the transition matrix


# ---------------------------------------------------------------------------
# Ledger
# ---------------------------------------------------------------------------

class AccountLedger:
    """Persistent loans/cards/bureau accounts advanced monthly."""

    def __init__(
        self,
        params: Params,
        rng: Generator,
        root_seed: int | None = None,
        target_default_rate: float = 0.08,
        performance_window: int = 12,
        trigger_months: int = 3,
    ) -> None:
        self.p = params
        self.rng = rng
        self.root_seed = root_seed
        self.fidx = params.factor_idx
        self.loans: list[Loan] = []
        self.cards: list[Card] = []
        self.bureau: list[BureauAccount] = []

        states, T_by, init_by, bfactor = params.conditional_markov("bureau_balance")
        self.bureau_mc = MarkovChain(states, T_by, rng)
        self._bureau_init = init_by
        self._bureau_factor = self.fidx[bfactor]
        self._n_buckets = T_by.shape[0]

        self.target_default_rate = float(target_default_rate)
        self.window = int(performance_window)
        self.trigger = int(trigger_months)
        self.run_hazard = RunHazard(self.window, self.trigger)
        self.default_shift = 0.0   # set by calibrate_default_level()

    # ------------------------------------------------------------------
    # Default-level calibration (the target_default_rate knob)
    # ------------------------------------------------------------------

    def calibrate_default_level(self, client_factors: np.ndarray,
                                weights: np.ndarray | None = None) -> float:
        """Solve the logit shift so the expected windowed default rate over
        the population that actually gets loans equals the target.

        ``weights``: relative probability that each client becomes a
        borrower (e.g. the approval probability). Selection matters because
        approval depends on factors correlated with f6.
        """
        z = client_factors[:, self.fidx["default_propensity"]]
        w = np.ones(len(z)) if weights is None else np.asarray(weights, float)
        self.default_shift = solve_default_shift(
            self.p.pd_curve(z), w, self.target_default_rate)
        return self.default_shift

    def loan_pd(self, F: np.ndarray) -> np.ndarray:
        z = F[:, self.fidx["default_propensity"]]
        return expit(logit(np.clip(self.p.pd_curve(z), 1e-9, 1 - 1e-9))
                     + self.default_shift)

    def loan_hazard(self, F: np.ndarray, term: np.ndarray) -> np.ndarray:
        n = np.minimum(np.asarray(term, int), self.window)
        return self.run_hazard.hazard(self.loan_pd(F), n)

    def _bucket(self, F: np.ndarray) -> np.ndarray:
        from scipy.stats import norm
        edges = norm.ppf(np.linspace(0, 1, self._n_buckets + 1)[1:-1])
        return np.searchsorted(edges, F[:, self._bureau_factor], side="right")

    # ------------------------------------------------------------------
    # Origination
    # ------------------------------------------------------------------

    def originate_loans(self, approvals: dict) -> None:
        """Create loans from approved applications.

        approvals: dict with client_id, factors (n,7), and term/principal/
        annuity arrays computed by the caller.
        """
        n = len(approvals["client_id"])
        if n == 0:
            return
        app_ids = approvals.get("application_id")
        hz = self.loan_hazard(np.asarray(approvals["factors"]),
                              approvals["term_months"])
        for i in range(n):
            self.loans.append(Loan(
                loan_id=f"LN-{approvals['client_id'][i]}-{len(self.loans):06d}",
                client_id=approvals["client_id"][i],
                factors=approvals["factors"][i],
                term_months=int(approvals["term_months"][i]),
                principal=float(approvals["principal"][i]),
                annuity=float(approvals["annuity"][i]),
                application_id=(app_ids[i] if app_ids is not None else ""),
                hazard=float(hz[i]),
            ))

    def burn_in(self, loans: list[Loan], months_paid: np.ndarray) -> None:
        """Advance pre-existing loans through their unobserved history with
        the same hazard, so the DPD state carried into month 1 is consistent
        with each client's risk (previously: uniform random 0-2)."""
        for loan, m in zip(loans, months_paid):
            m = int(min(m, loan.term_months - 1))
            dpd = 0
            draws = self.rng.random(m) < loan.hazard
            for d in draws:
                dpd = dpd + 1 if d else 0
            loan.months_paid, loan.dpd_months = m, dpd

    def originate_cards(self, cardholders: dict) -> None:
        n = len(cardholders["client_id"])
        for i in range(n):
            self.cards.append(Card(
                card_id=f"CC-{cardholders['client_id'][i]}-{len(self.cards):06d}",
                client_id=cardholders["client_id"][i],
                factors=cardholders["factors"][i],
                limit=float(cardholders["limit"][i]),
                utilization=float(cardholders["utilization"][i]),
            ))

    def originate_bureau(self, clients: dict) -> None:
        """Give each client a bureau history: n_accounts ~ poisson(stability).

        Initial status distribution: mostly Closed (C) or current (0),
        matching the real bureau (62% closed, 37% active).
        """
        n_accounts = draw_from_model(
            self.p.model("bureau", "n_accounts"),
            clients["factors"], self.rng, self.fidx)
        n_accounts = np.maximum(n_accounts, 0).astype(int)

        states = self.bureau_mc.states
        n_states = len(states)
        buckets = self._bucket(np.asarray(clients["factors"]))
        cum_init = np.cumsum(self._bureau_init, axis=1)

        amounts = np.exp(draw_from_model(
            self.p.model("bureau", "log_amt_credit_sum"),
            clients["factors"], self.rng, self.fidx))

        # credit type from fitted frequencies
        types = list(self.p._data["bureau"]["credit_type_freq"].keys())
        type_probs = np.array(list(self.p._data["bureau"]["credit_type_freq"].values()))
        type_probs = type_probs / type_probs.sum()

        k = 0
        for i, client_id in enumerate(clients["client_id"]):
            for _ in range(n_accounts[i]):
                status_idx = int(min(np.searchsorted(
                    cum_init[buckets[i]], self.rng.random(), side="right"),
                    n_states - 1))
                self.bureau.append(BureauAccount(
                    bureau_id=f"BU-{client_id}-{len(self.bureau):08d}",
                    client_id=client_id,
                    factors=clients["factors"][i],
                    status_idx=status_idx,
                    credit_sum=float(amounts[i]),
                    credit_type=str(self.rng.choice(types, p=type_probs)),
                    age_months=int(self.rng.integers(0, 240)),
                    bucket=int(buckets[i]),
                ))
                k += 1

    # ------------------------------------------------------------------
    # Monthly advance
    # ------------------------------------------------------------------

    def advance_month(self, payment_date) -> list[dict]:
        """Advance all accounts one month.

        Returns the payment rows produced this month (one per active loan
        installment due), so the final installment is emitted exactly once
        and paid-off loans stop producing payments.

        Phase 0: the month's draws come from a stream derived from
        (root_seed, "ledger", month) instead of the sequential bootstrap
        tape, so each month replays byte-identically on its own.
        """
        if self.root_seed is not None:
            month = payment_date.replace(day=1)
            self.rng = month_rng(self.root_seed, "ledger", month)
        p = self.p
        payment_rows: list[dict] = []

        # --- loans: payment behavior -> DPD ---
        for loan in self.loans:
            if loan.paid_off:
                continue
            # per-loan hazard set at origination from the PD curve, so
            # P(default within window | f6) reproduces the calibrated curve
            if self.rng.random() < loan.hazard:
                loan.dpd_months += 1
            else:
                loan.dpd_months = 0
            loan.months_paid += 1

            # payment amount for this installment
            ratio = 1.0 if loan.dpd_months == 0 else float(self.rng.uniform(0.3, 0.8))
            amount = loan.annuity * ratio
            payment_rows.append({
                "payment_id": f"PMT-{loan.loan_id}-{loan.months_paid:03d}",
                "loan_id": loan.loan_id,
                "client_id": loan.client_id,
                "application_id": loan.application_id,
                "payment_date": payment_date,
                "amount": float(amount),
                "principal_portion": float(amount * 0.65),
                "interest_portion": float(amount * 0.35),
                "days_late": loan.dpd_months * 30,
                "payment_method": "auto_debit",
                "num_payment": loan.months_paid,
            })
            if loan.months_paid >= loan.term_months:
                loan.paid_off = True

        # --- cards: DPD ---
        card_dpd = self.p.model("credit_card_balance", "p_dpd_gt0")
        for card in self.cards:
            F = card.factors[None, :]
            if bool(draw_from_model(card_dpd, F, self.rng, self.fidx)[0]):
                card.dpd_months += 1
            else:
                card.dpd_months = 0
            # utilization drifts slightly (AR-ish)
            card.utilization = float(np.clip(
                card.utilization + self.rng.standard_normal() * 0.05, 0.0, 1.2))

        # --- bureau: STATUS markov ---
        if self.bureau:
            idx = np.array([b.status_idx for b in self.bureau])
            bkt = np.array([b.bucket for b in self.bureau])
            new_idx = self.bureau_mc.sample_states(idx, n_steps=1, bucket=bkt)
            for b, ni in zip(self.bureau, new_idx):
                b.status_idx = int(ni)
                b.age_months += 1
                # track numeric overdue months from STATUS label
                lbl = self.bureau_mc.states[int(ni)]
                b.months_dpd_status = int(lbl) if lbl.lstrip("-").isdigit() else 0

        return payment_rows

    # ------------------------------------------------------------------
    # Snapshot emission
    # ------------------------------------------------------------------

    def emit_pos_balances(self, snapshot_date, active_loan_ids=None) -> "pl.DataFrame":
        import polars as pl
        rows = []
        for loan in self.loans:
            if active_loan_ids is not None:
                if loan.loan_id not in active_loan_ids:
                    continue
            elif loan.months_paid <= 0 or loan.paid_off:
                continue
            sk_dpd = loan.dpd_months * 30
            rows.append({
                "pos_balance_id": f"PB-{loan.loan_id}",
                "client_id": loan.client_id,
                "loan_id": loan.loan_id,
                "snapshot_date": snapshot_date,
                "months_balance": -(loan.term_months - loan.months_paid),
                "cnt_instalment": loan.term_months,
                "cnt_instalment_future": max(0, loan.term_months - loan.months_paid),
                "sk_dpd": sk_dpd,
                "sk_dpd_def": int(sk_dpd > 90),
            })
        return pl.DataFrame(rows)

    def emit_card_balances(self, snapshot_date) -> "pl.DataFrame":
        import polars as pl
        rows = []
        for card in self.cards:
            balance = card.limit * np.clip(card.utilization, 0, 1.2)
            rows.append({
                "card_balance_id": f"CB-{card.card_id}",
                "card_id": card.card_id,
                "client_id": card.client_id,
                "snapshot_date": snapshot_date,
                "months_balance": 0,
                "amt_balance": float(balance),
                "amt_credit_limit_actual": card.limit,
                "amt_drawings_current": float(balance * 0.7),
                "amt_payment_current": float(balance * 0.15),
                "cnt_drawings_current": int(self.rng.integers(1, 25)),
                "sk_dpd": card.dpd_months * 30,
            })
        return pl.DataFrame(rows)

    def emit_bureau(self, snapshot_date) -> "pl.DataFrame":
        import polars as pl
        from datetime import timedelta
        rows = []
        for b in self.bureau:
            status = self.bureau_mc.states[b.status_idx]
            rows.append({
                "bureau_id": b.bureau_id,
                "client_id": b.client_id,
                "snapshot_date": snapshot_date,
                "credit_account_id": b.bureau_id,
                "credit_type": b.credit_type,
                "credit_amount": b.credit_sum,
                "credit_currency": "VND",
                "credit_start_date": snapshot_date - timedelta(days=30 * b.age_months),
                "credit_duration_days": 30 * max(1, b.age_months),
                "days_credit": -30 * b.age_months,
                "credit_updated_date": snapshot_date,
                "amt_credit_max_overdue": float(b.months_dpd_status > 0) * b.credit_sum * 0.3,
                "amt_credit_sum": b.credit_sum,
                "amt_credit_sum_overdue": float(b.months_dpd_status > 0) * b.credit_sum * 0.2,
                "cnt_credit_prolong": int(self.rng.integers(0, 3)),
                "credit_active": status != "C",
                "status": status,
            })
        return pl.DataFrame(rows)

