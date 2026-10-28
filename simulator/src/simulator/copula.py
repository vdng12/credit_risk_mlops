"""t-Copula engine: latent factor sampling and marginal transforms.

The t-copula captures tail dependence appropriate for credit risk:
extreme co-movements (e.g. systemic default waves) are more likely
than under a Gaussian copula.
"""

from __future__ import annotations

import numpy as np
from numpy.random import Generator
from scipy.stats import norm, t as tdist


class CopulaEngine:
    """Stateful t-copula sampler for correlated latent factor scores.

    Parameters
    ----------
    nu : float
        Degrees of freedom. Lower = heavier tails.
    corr_matrix : np.ndarray or None
        Square correlation matrix for latent factors.
    seed : int
        Seed for internal Generator.
    """

    def __init__(
        self,
        nu: float = 4.0,
        corr_matrix: np.ndarray | None = None,
        seed: int = 42,
    ) -> None:
        self.nu = float(nu)
        self._rng: Generator = np.random.default_rng(seed)

        if corr_matrix is None:
            corr_matrix = np.array([
                [ 1.00,  0.45,  0.00,  0.30,  0.35, -0.20, -0.60],
                [ 0.45,  1.00,  0.40,  0.50,  0.25,  0.00, -0.25],
                [ 0.00,  0.40,  1.00,  0.15, -0.10,  0.25,  0.05],
                [ 0.30,  0.50,  0.15,  1.00,  0.20,  0.05, -0.10],
                [ 0.35,  0.25, -0.10,  0.20,  1.00, -0.30, -0.30],
                [-0.20,  0.00,  0.25,  0.05, -0.30,  1.00,  0.15],
                [-0.60, -0.25,  0.05, -0.10, -0.30,  0.15,  1.00],
            ], dtype=np.float64)

        self._corr = np.asarray(corr_matrix, dtype=np.float64)
        self._n_factors = self._corr.shape[0]
        self._L = self._cholesky_psd(self._corr)

    @staticmethod
    def _cholesky_psd(mat: np.ndarray) -> np.ndarray:
        """Cholesky factor, with nearest-PSD fallback."""
        try:
            return np.linalg.cholesky(mat)
        except np.linalg.LinAlgError:
            eigs = np.linalg.eigvalsh(mat)
            eigs = np.maximum(eigs, 1e-6)
            v = np.linalg.eigh(mat)[1]
            fixed = v @ np.diag(eigs) @ v.T
            d = np.sqrt(np.diag(fixed))
            fixed = fixed / np.outer(d, d)
            return np.linalg.cholesky(fixed)

    def sample_uniform(self, n: int) -> np.ndarray:
        """Draw n samples from the t-copula, returning uniform [0,1] values.

        Returns shape (n, n_factors).
        """
        z = self._rng.standard_normal((n, self._n_factors))
        z_corr = z @ self._L.T
        w = self._rng.chisquare(self.nu, size=n) / self.nu
        t_samples = z_corr / np.sqrt(w[:, None])
        u = tdist.cdf(t_samples, df=self.nu)
        return np.clip(u, 1e-15, 1.0 - 1e-15)

    def sample_latent_factors(self, n: int) -> np.ndarray:
        """Draw n samples of latent factor z-scores.

        Returns shape (n, n_factors) with values roughly in [-4, 4].
        """
        u = self.sample_uniform(n)
        return norm.ppf(u)

    def lognormal(
        self,
        factor: np.ndarray,
        log_mu: float,
        log_sigma: float,
        factor_weight: float = 0.5,
    ) -> np.ndarray:
        """Log-normal values driven by a latent factor.

        log X = log_mu + log_sigma * (w * f + sqrt(1 - w^2) * eps), so with
        f ~ N(0,1) the marginal is exactly Lognormal(log_mu, log_sigma) and
        corr(log X, f) = w. (Previously the residual sd was a fixed
        0.3 * log_sigma, so the marginal variance depended on w.)
        """
        w = float(np.clip(factor_weight, -1.0, 1.0))
        mean = log_mu + log_sigma * w * np.asarray(factor, dtype=np.float64)
        return self._rng.lognormal(mean=mean,
                                   sigma=log_sigma * np.sqrt(1.0 - w * w))

    def categorical(
        self,
        factor: np.ndarray,
        categories: list[str],
        base_probs: np.ndarray,
        sensitivity: float = 0.15,
    ) -> np.ndarray:
        """Categorical values driven by a latent factor.

        Higher factor scores increase probability of higher-index categories.
        """
        n = len(factor)
        k = len(categories)
        factor = np.asarray(factor, dtype=np.float64)
        logits_base = np.log(np.maximum(np.asarray(base_probs, dtype=np.float64), 1e-10))
        shift = np.linspace(-sensitivity, sensitivity, k)
        # (n, k) logits; Gumbel-max trick = exact vectorized softmax sampling
        logits = logits_base[None, :] + shift[None, :] * factor[:, None]
        gumbel = self._rng.gumbel(size=(n, k))
        idx = (logits + gumbel).argmax(axis=1)
        cats = np.asarray(categories)
        return cats[idx.astype(int)]

    def bernoulli(
        self,
        factor: np.ndarray,
        base_prob: float,
        factor_weight: float = 0.3,
    ) -> np.ndarray:
        """Bernoulli (boolean) values driven by a latent factor."""
        n = len(factor)
        base_logit = np.log(base_prob / (1 - base_prob))
        logits = base_logit + factor * factor_weight
        probs = 1 / (1 + np.exp(-logits))
        return self._rng.random(n) < probs

    def poisson(
        self,
        factor: np.ndarray,
        base_lambda: float,
        factor_weight: float = 0.2,
    ) -> np.ndarray:
        """Poisson counts with a log link: lambda = base * exp(w * f)."""
        lam = base_lambda * np.exp(factor_weight * np.asarray(factor, dtype=np.float64))
        return self._rng.poisson(lam)