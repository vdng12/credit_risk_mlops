"""Configuration and parametric knobs for the t-copula simulator."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date


@dataclass
class CopulaConfig:
    """t-Copula parameters controlling latent-factor dependence structure.

    Attributes:
        nu: Degrees of freedom. Lower = heavier tails.
        corr_matrix: 7x7 correlation matrix for latent factors:
            [0] creditworthiness, [1] income_capacity, [2] loan_demand,
            [3] asset_quality, [4] stability, [5] risk_appetite,
            [6] default_propensity
    """

    # PROVISIONAL until calibrate.py is re-run on Home Credit:
    #  * nu: the old fit hit the search upper bound (30) => the data prefers
    #    a Gaussian copula; 30 is kept as a mildly tail-dependent default.
    #  * row/col 0 sign-flipped: the old PCA score for "creditworthiness"
    #    had corr +0.558 with the PD score, i.e. it pointed the wrong way
    #    (every generator treats +f0 as *safer*). Flipping one row/col of a
    #    correlation matrix keeps it a valid correlation matrix.
    #  * row/col 4 (stability) sign-flipped for the same reason: it had
    #    corr +0.118 with PD, consistent with a "younger / larger family"
    #    PC, while generators use +stability = older (older clients default
    #    less in Home Credit). Signs of factors 1,2,3,5 are unverified until
    #    the sign-anchored calibrate.py is re-run.
    nu: float = 30.0

    corr_matrix: list[list[float]] = field(default_factory=lambda: [
        [ 1.000, -0.008,  0.162, -0.014,  0.011,  0.018, -0.558],
        [-0.008,  1.000, -0.137,  0.020,  0.270,  0.013, -0.114],
        [ 0.162, -0.137,  1.000, -0.031, -0.114, -0.144, -0.236],
        [-0.014,  0.020, -0.031,  1.000,  0.082, -0.011,  0.056],
        [ 0.011,  0.270, -0.114,  0.082,  1.000, -0.047, -0.118],
        [ 0.018,  0.013, -0.144, -0.011, -0.047,  1.000, -0.064],
        [-0.558, -0.114, -0.236,  0.056, -0.118, -0.064,  1.000],
    ])

    def __post_init__(self) -> None:
        n = len(self.corr_matrix)
        for i in range(n):
            self.corr_matrix[i][i] = 1.0
            for j in range(i + 1, n):
                avg = (self.corr_matrix[i][j] + self.corr_matrix[j][i]) / 2
                self.corr_matrix[i][j] = avg
                self.corr_matrix[j][i] = avg


@dataclass
class EntityConfig:
    """Per-entity cardinality and behaviour knobs."""

    # Client pool
    n_customers: int = 50000
    card_penetration: float = 0.55
    cards_per_cardholder_lambda: float = 1.3
    loan_penetration: float = 0.18

    # Application stream
    monthly_app_rate: float = 0.03

    # Card auth stream
    auths_per_card_per_day: float = 1.8
    auth_decline_rate: float = 0.12
    merchant_categories: dict[str, float] = field(default_factory=lambda: {
        "groceries": 0.22,
        "dining": 0.14,
        "retail": 0.18,
        "transport": 0.10,
        "utilities": 0.08,
        "entertainment": 0.09,
        "travel": 0.06,
        "healthcare": 0.05,
        "education": 0.03,
        "other": 0.05,
    })

    # Bureau
    bureau_records_per_customer_lambda: float = 4.5

    # Default definition: ``default_trigger_months`` consecutive delinquent
    # months (3 = 90 DPD) within the first ``performance_window_months``
    # instalments. label.py derives exactly this quantity.
    default_trigger_months: int = 3
    performance_window_months: int = 12
    # Expected windowed default rate over *originated* loans (accounts for
    # approval selection). Solved exactly at init as a logit shift of the
    # calibrated PD curve, so discrimination is unchanged.
    # Home Credit's application_train.csv sits around 8%.
    target_default_rate: float = 0.08

    # Entity generation switches (for lean training-data runs that only
    # need the application-level entities: clients + applications + loans +
    # payments). Card auths / cards / bureau are skipped when False, which
    # keeps large runs fast and memory-bounded.
    generate_card_auths: bool = True
    generate_cards: bool = True
    generate_bureau: bool = True


@dataclass
class TimeConfig:
    """Simulation clock and schedule knobs."""

    start_date: date = field(default_factory=lambda: date(2020, 1, 1))
    bureau_drop_lag_days: int = 21
    payment_file_hour_utc: int = 2


@dataclass
class ProblemConfig:
    """Data problem injection knobs."""

    enabled: bool = True
    client_missing_rate: float = 0.005
    application_missing_rate: float = 0.01
    card_auth_missing_rate: float = 0.02
    pos_balance_missing_rate: float = 0.008
    card_balance_missing_rate: float = 0.008
    bureau_missing_rate: float = 0.015
    payment_missing_rate: float = 0.003
    duplicate_prob: float = 0.0005
    schema_drift_rate: float = 0.1
    late_arrival_max_days: int = 5
    late_arrival_prob: float = 0.02
    outlier_rate: float = 0.001
    column_swap_prob: float = 0.0002


@dataclass
class SinkConfig:
    """Output sink configuration."""

    postgres_dsn: str = "postgresql://lending:lending123@localhost:5432/lending_db"
    s3_endpoint: str = "http://localhost:8333"
    s3_access_key: str = "seaweedfs"
    s3_secret_key: str = "seaweedfs123"
    s3_bucket: str = "risk-data"
    s3_region: str = "us-east-1"
    kafka_brokers: str = "localhost:9092"
    kafka_applications_topic: str = "loan_applications"
    kafka_card_auths_topic: str = "card_authorizations"
    file_output_dir: str = "data/generated"


@dataclass
class SimConfig:
    """Top-level configuration aggregating all knobs."""

    copula: CopulaConfig = field(default_factory=CopulaConfig)
    entities: EntityConfig = field(default_factory=EntityConfig)
    time: TimeConfig = field(default_factory=TimeConfig)
    problems: ProblemConfig = field(default_factory=ProblemConfig)
    sinks: SinkConfig = field(default_factory=SinkConfig)
    seed: int = 42

    # Flat-kwarg routing for common overrides
    _FLAT_TO_NESTED = {
        "n_customers": ("entities", "n_customers"),
        "start_date": ("time", "start_date"),
        "nu": ("copula", "nu"),
        "card_penetration": ("entities", "card_penetration"),
        "loan_penetration": ("entities", "loan_penetration"),
        "monthly_app_rate": ("entities", "monthly_app_rate"),
        "auths_per_card_per_day": ("entities", "auths_per_card_per_day"),
        "bureau_drop_lag_days": ("time", "bureau_drop_lag_days"),
        "target_default_rate": ("entities", "target_default_rate"),
        "performance_window_months": ("entities", "performance_window_months"),
    }

    def __init__(self, **kwargs) -> None:
        self.copula = CopulaConfig()
        self.entities = EntityConfig()
        self.time = TimeConfig()
        self.problems = ProblemConfig()
        self.sinks = SinkConfig()
        self.seed = 42

        for flat_key, (nested_obj, nested_key) in self._FLAT_TO_NESTED.items():
            if flat_key in kwargs:
                setattr(getattr(self, nested_obj), nested_key, kwargs.pop(flat_key))
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.__post_init__()

    def __post_init__(self) -> None:
        import numpy as np
        n = len(self.copula.corr_matrix)
        mat = np.array(self.copula.corr_matrix, dtype=np.float64)
        eigs = np.linalg.eigvalsh(mat)
        if eigs.min() < -1e-8:
            raise ValueError(
                f"Correlation matrix not PSD (min eig={eigs.min():.6f}). "
                "Heavy-tailed t-copulas need a valid correlation matrix."
            )