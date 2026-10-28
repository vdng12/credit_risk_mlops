# Simulator - Statistical Methods

This document describes the statistical machinery behind the synthetic
lending-data generator in `simulator/`: *what* each method does, *why*, and
*how well* it works, including known gaps. Measured numbers come from
`tests/test_statistics.py` and synthetic-only benchmark runs; anything that
needs the real Home Credit files is marked **(re-calibrate)**.

> Modules: `copula.py`, `calibrate.py`, `calibrate_structures.py`,
> `params.py`, `state.py`, `label.py`, `entities.py`, `core.py`.

---

## 1. Overview: one generative spine, many observed tables

```
t-copula (7 latent factors, N(0,1) margins)
        |  correlated, optionally tail-dependent
        v
per-client factor vector f in R^7      (drawn once, stable for life)
        |-- client / application features      f_0..f_5  -> marginals
        |-- child-table links                  f          -> binned / GLM curves
        |-- bureau STATUS                      Markov chain, matrix chosen by f_6 bucket
        `-- loan default                       f_6 -> PD curve -> per-loan monthly hazard
                                                         -> DPD path -> windowed label (gold)
```

Each client draws a latent factor vector $\mathbf{f} \in \mathbb{R}^7$ once
and keeps it for life. Every observed variable is a function of a factor (or
a child-table link keyed to one), so all tables downstream of a client are
correlated by construction.

| idx | Factor | "+" means | Built from |
|---|---|---|---|
| 0 | `creditworthiness` | safer | PC1 of `EXT_SOURCE_1/2/3` |
| 1 | `income_capacity` | richer | PC1 of log income, days employed, income type, occupation |
| 2 | `loan_demand` | bigger loan | PC1 of log credit / annuity / goods price |
| 3 | `asset_quality` | more assets | PC1 of car/realty flags, car age, housing |
| 4 | `stability` | older | PC1 of age, children, family, gender, education |
| 5 | `risk_appetite` | revolving | PC1 of contract type, suite |
| 6 | `default_propensity` | riskier | out-of-fold GBM PD score (all features $\rightarrow$ `TARGET`) |

Every PC is **sign-anchored**: after PCA the score is flipped if its
correlation with a reference column (EXT_SOURCE_2, log income, log credit,
own-realty, age, revolving flag) has the wrong sign. PCA signs are
arbitrary, and the generators hard-code the "+" directions above.

---

## 2. Why a t-copula

A copula separates marginals from dependence. The t-copula adds **tail
dependence** (joint extremes more frequent than under a Gaussian copula),
controlled by $\nu$: small ($3$-$5$) = strong tail dependence (stress),
large ($\gtrsim 30$) $\approx$ Gaussian.

### Sampling (`copula.py`)

$$ \mathbf{z} \sim \mathcal{N}(\mathbf{0}, I), \qquad
   \mathbf{z}_c = \mathbf{z}L^\top \quad (L = \operatorname{chol}\Sigma), \qquad
   w \sim \frac{\chi_\nu^2}{\nu}, \qquad
   \mathbf{t} = \frac{\mathbf{z}_c}{\sqrt{w}} $$

$$ u = F_\nu(\mathbf{t}; \nu), \qquad \mathbf{f} = \Phi^{-1}(u) $$

The tail content lives in the division by $\sqrt{w}$: a small $\nu$ makes
$w$ heavy-tailed, so $\mathbf{t}$ (and hence $\mathbf{f}$) produces joint
extremes. The margins are exactly $\Phi^{-1}(u) \sim \mathcal{N}(0,1)$ for
any $\nu$.

### Marginal samplers

| Outcome | Model |
|---|---|
| continuous | $\log X = \mu + \sigma\left(w\,\mathbf{f} + \sqrt{1-w^2}\,\varepsilon\right)$: exact Lognormal $(\mu, \sigma)$ marginal, $\operatorname{corr}(\log X, \mathbf{f}) = w$ |
| categorical | $\operatorname{logit}_j = \log p_j + \mathrm{shift}_j \cdot \mathbf{f}$, Gumbel-max sampling |
| binary | $p = \sigma\!\left(\operatorname{logit} p_0 + w \cdot \mathbf{f}\right)$ |
| count | Poisson, $\lambda = \lambda_0 \exp(w \cdot \mathbf{f})$ |

This "one linear inner product per column" design is what makes the whole
table correlated: any pair of columns that share a factor are dependent
through it, and the weight $w$ calibrates how strongly each column tracks
its factor.

---

## 3. Calibration (`calibrate.py`)

1. Factor scores (Section 1). Heavy-tailed amounts are $\log(1+x)$'d before
   standardising. Y/N flags become 0/1.
2. **PD score out-of-fold**: 5-fold `cross_val_predict` with
   `HistGradientBoostingClassifier`. An in-sample score overstates
   discrimination, and that error would carry straight into the synthetic
   label.
3. Pseudo-observations from mid-ranks, $u = \mathrm{rank}/(n+1)$.
4. **$\Sigma$ by Kendall's-$\tau$ inversion**, $\rho = \sin(\pi\tau/2)$:
   valid for every elliptical copula and robust to heavy tails. Then
   repaired to the nearest correlation matrix.
5. **$\nu$ by profile likelihood**, $\Sigma$ fixed, bounds $(2.05, 200)$.
   Compared with the Gaussian copula by a boundary-corrected LR test
   $\tfrac{1}{2}\chi_0^2 + \tfrac{1}{2}\chi_1^2$. The result is flagged
   `gaussian_preferred` when $\nu$ hits the bound or $p > 0.05$.

Recovery on synthetic ground truth (10k obs, the shipped $\Sigma$):
$\nu = 4 \rightarrow 4.07$, $\nu = 10 \rightarrow 9.84$, Gaussian -> flagged;
max $|\Delta\Sigma| \le 0.024$.

**Current shipped values are provisional (re-calibrate).** The previous
fit reported $\nu = 29.94$. That was the optimizer sitting on the old search
bound of 30, not an estimate: it means "Gaussian preferred". Two factors had
inverted signs (creditworthiness had $\operatorname{corr} = +0.558$ with PD;
stability $+0.118$). Their rows/columns of $\Sigma$ have been flipped, which
keeps $\Sigma$ a valid correlation matrix. Signs of factors 1, 2, 3 and 5
are unverified until the anchored calibration is re-run.

---

## 4. Child-table links (`calibrate_structures.py`)

**Scale fix.** All links are fit against **normal scores**
$\Phi^{-1}\!\left(\mathrm{rank}/(n+1)\right)$ of the factors, the same scale
the simulator samples. Previously they were fit against z-scored raw scores.
For the PD factor that scale is strongly right-skewed: its decile edges ran
from $-0.76$ to $+1.01$, so $\mathcal{N}(0,1)$ draws landed in the wrong bins
(22% instead of 10% in the bottom bin).

| Link | Model | Notes |
|---|---|---|
| bureau `p_overdue` | decile curve on $f_6$ | ~12x spread (fallback) |
| prior apps `p_approved` | decile curve on $f_0$ | weak (1.1x) |
| prior apps amount / term | linear / Poisson on $f_1$, $f_2$ | **approved, non-zero only**: refused rows (amount 0, term null) had made log-amount bimodal ($\sigma = 4.8$, so most draws hit the 5k/500k clip) |
| instalments `p_partial`, `p_late` | decile curve on $f_6$ | split payments aggregated per instalment first; diagnostic only |
| POS / card `p_dpd>0` | decile curve on $f_6$ | |
| bureau `STATUS` | **conditional Markov chain** | one $8 \times 8$ matrix per $f_6$ quintile, Dirichlet-shrunk ($\kappa = 50$) to the pooled matrix; per-bucket initial status |

The bureau chain used to ignore the factors entirely, so bureau history
carried no information about the client's risk.

---

## 5. Default model (`state.py`): the label mechanism

**Target quantity.** $\mathrm{default} = 1$ if a loan reaches **3 consecutive
delinquent months within its first $W = 12$ instalments**, a fixed
performance window like Home Credit's `TARGET`.

**Default curve.** From real data, $P(\mathrm{default} \mid f_6 = z) = Q_{PD}\!\left(\Phi(z)\right)$:
the quantile function of the out-of-fold PD score. This reproduces the real
data's discrimination by construction. Fallback (no JSON):
logit-normal,

$$ \operatorname{logit} PD = -2.80 + 0.969\,z, $$

matched to HC's 8.07% base rate and AUC 0.745.

**Exact hazard inversion (`RunHazard`).** With i.i.d. monthly delinquency
at hazard $h$,

$$ g_n(h) = P(\text{run} \ge 3 \text{ in } n \text{ trials}) $$

is computed exactly by a DP over the current run length. $g_n$ is strictly
increasing, so each loan gets $h = g_n^{-1}(PD)$ with $n = \min(\mathrm{term}, W)$.
A Monte-Carlo check (200k paths) matches to $\pm 0.003$.

**Default-rate knob.** `target_default_rate` solves a constant logit shift
$\delta$ by `brentq` so that the approval-weighted mean of
$\sigma\!\left(\operatorname{logit} PD + \delta\right)$ over the client pool
equals the target. This is deterministic, preserves the ranking, and accounts
for approval selection.

| target | measured (final labels) |
|---|---|
| 0.03 | 0.0303 |
| 0.08 | 0.0783 (pooled over 6 seeds, 143k loans: 0.0792 vs expected 0.0802, $z = -1.4$) |
| 0.15 | 0.1473 |

Latent AUC($f_6$ $\rightarrow$ default) = 0.745 (target 0.745).

**What this replaced.** The previous model was $\sigma(\alpha + \beta f_6)$
with $\beta = 0.24$, taken from per-instalment partial payments, and $\alpha$
found by binary search on a fresh Monte-Carlo draw each step. It had four
problems: $\beta$ was fitted on the skewed scale (Section 4); the proxy (any
underpayment, including split payments) is not the label; the calibration
target (lifetime default) differed from what the label measured; and the
approval-selected loan population was ignored. The result: latent AUC 0.63,
and 6.1% realised for an 8% target. Taking $\beta$ as an "authentic"
property of the data was the methodological error: the property to preserve
is the real $P(\mathrm{TARGET} \mid PD)$.

**Historical book.** Pre-existing loans are burned in with their own
hazard. Their DPD state is consistent with risk (previously random 0-2).

---

## 6. Label derivation (`label.py`)

1. Payments carry `application_id` -> re-join to the application master.
2. $\mathrm{default}$ = any instalment with $\mathrm{num\_payment} \le W$ and
   $\mathrm{days\_late} \ge 90$.
3. `label_status`:
   - `final`: the full window $\min(\mathrm{term}, W)$ has been observed.
   - `pending`: the window is still open.
   - `left_truncated`: the loan started before the data, so its window is
     not fully observed.
   Defaults seen early are **not** final until the window closes. Finalising
   them early over-samples defaults in young vintages (measured: 9.6% vs 8%).
4. Training uses **only `final` rows**. `scripts/eval_synthetic.py` used to
   fill missing labels with 0, so declined applications and immature loans
   became "non-defaults". That gave a 4.7% rate and mixed the approval
   policy into the target.

---

## 7. Structural fixes in the generators

- Client attributes are drawn once. Non-factor noise (age, children,
  employment) used to be re-drawn every month, so ages jumped by years
  between snapshots. Now only the snapshot date and age move.
- Card-auth amounts and declines use the cardholder's factors. They used
  fresh factors per event, which made them independent of the client.
- Unused legacy generators that drew factor-independent POS, card, bureau
  and payment rows were removed.
- Markov sampler index is clamped against float round-off.

---

## 8. Validation

- `tests/test_statistics.py`: 11 tests covering copula recovery, Gaussian
  flagging, $\mathcal{N}(0,1)$ margins, exact lognormal marginal, hazard vs
  Monte-Carlo, knob accuracy, latent AUC, EXT_SOURCE direction, and client
  stability.
- `problems.py` / `quality.py`: unchanged (corruption injection,
  per-entity + FK checks).

---

## 9. Known limitations

1. **Feature information bottleneck: the main AUC limiter.** Application
   features are generated from $f_0 \dots f_5$ only. Under the shipped
   $\Sigma$ these explain $R^2 = 0.37$ of $f_6$. So even a perfect model on
   application features has AUC $\le$ **0.654** (measured ceiling,
   $E[f_6 \mid f_0, \dots, f_5]$) against a latent 0.745. LightGBM reaches
   0.617. In real data the PD score is a function of the features
   ($R^2 \rightarrow 1$), so real AUC $\approx 0.75$. Fix: calibrate a
   per-feature loading matrix $\Lambda$ on all 7 factors (feature
   normal-scores $\sim \Lambda\mathbf{f} + \varepsilon$, mapped through
   empirical quantiles) instead of the hand-coded feature mappings in
   `entities.generate_clients`.
2. **Hand-coded feature mappings.** EXT_SOURCE, credit score, region rating
   and interest-rate formulas are priors, not fitted. `EXT_SOURCE_2` is
   driven by stability/asset (weakly protective) rather than
   creditworthiness.
3. **Discrete margins.** risk_appetite and asset_quality PCs are mostly
   dummies. Mid-ranks plus $\tau$-b still attenuate $\rho$ by about 15% on
   a 4-level margin.
4. **Weak trait coherence.** PC1 explained variance is 0.09 for income,
   0.18 stability, 0.22 asset (pre-fix values; log-transforming income
   should help) **(re-calibrate)**.
5. **i.i.d. monthly hazard.** Real delinquency is sticky (cure 46%, 1->1
   36%). A two-state hazard (enter / persist), solved by the same DP, would
   match the roll-rate structure.
6. Card DPD is still per-month independent given $f_6$. Card utilization is
   a random walk.