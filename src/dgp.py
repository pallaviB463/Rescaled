
"""Data-generating process with exact treatment-effect ground truth."""

from functools import lru_cache

import numpy as np
from scipy.optimize import brentq
from scipy.special import expit


GH_X, GH_W = np.polynomial.hermite_e.hermegauss(40)
GH_W = GH_W / np.sqrt(2 * np.pi)

SCENARIOS = [
    "null",
    "homogeneous",
    "extensive",
    "intensive",
    "aligned",
    "opposed",
]


def effects(X, scenario):
    """Return treatment shifts in purchase log-odds and log-spend."""
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario!r}")

    z3, z4 = X[:, 3], X[:, 4]
    zero = np.zeros(len(X))

    if scenario == "null":
        return zero, zero
    if scenario == "homogeneous":
        return zero + 0.4, zero + 0.15
    if scenario == "extensive":
        return 0.3 + 0.8 * z3, zero
    if scenario == "intensive":
        return zero, 0.2 + 0.4 * z4
    if scenario == "aligned":
        return 0.3 + 0.8 * z3, 0.2 + 0.4 * z3
    if scenario == "opposed":
        return 0.3 + 0.8 * z3, 0.2 - 0.4 * z3

    raise ValueError(f"Unknown scenario: {scenario!r}")


def _base(X):
    """Return baseline purchase log-odds shape and log-spend mean."""
    return (
        0.8 * X[:, 0] - 0.5 * X[:, 1],
        2.5 + 0.5 * X[:, 2],
    )


@lru_cache(maxsize=None)
def calibrate_b0(zero_rate, d=10, seed=123):
    """Calibrate the intercept to the requested control purchase rate."""
    Xp = np.random.default_rng(seed).normal(size=(200_000, d))
    shape, _ = _base(Xp)

    return brentq(
        lambda b: expit(b + shape).mean() - (1 - zero_rate),
        -30,
        10,
    )


def _validate_inputs(X, W, scenario, zero_rate, sigma):
    """Validate covariates, treatment assignment, and DGP parameters."""
    X = np.asarray(X, dtype=float)
    W = np.asarray(W)

    if X.ndim != 2 or X.shape[1] < 5:
        raise ValueError("X must be a 2D array with at least 5 columns.")
    if X.shape[0] == 0:
        raise ValueError("X must contain at least one row.")
    if not np.isfinite(X).all():
        raise ValueError("X must contain only finite values.")

    if W.ndim != 1 or len(W) != len(X):
        raise ValueError("W must be a 1D array with the same number of rows as X.")
    if not np.isin(W, [0, 1]).all():
        raise ValueError("W must contain only 0 and 1.")
    W = W.astype(int)

    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario!r}")
    if not 0 < zero_rate < 1:
        raise ValueError("zero_rate must be between 0 and 1.")
    if not np.isfinite(sigma) or sigma < 0:
        raise ValueError("sigma must be finite and non-negative.")

    return X, W


def simulate_from_X(
    X,
    W,
    scenario="opposed",
    zero_rate=0.9,
    sigma=0.8,
    seed=0,
):
    """Generate outcomes for supplied covariates and treatment assignments.

    Parameters
    ----------
    X : array-like, shape (n, d)
        Supplied covariates. The first five columns follow the DGP definition.
    W : array-like, shape (n,)
        Supplied binary treatment assignments (0 = control, 1 = treatment).
    scenario : str
        One of the six supported treatment-effect scenarios.
    zero_rate : float
        Target probability of zero outcome in the control population.
    sigma : float
        Standard deviation of log-spend noise.
    seed : int
        Random seed for outcome generation.

    Returns
    -------
    Y : ndarray, shape (n,)
        Simulated nonnegative outcomes.
    truth : dict
        Exact level-scale CATE and baseline/treatment outcome parameters.
    """
    X, W = _validate_inputs(X, W, scenario, zero_rate, sigma)
    rng = np.random.default_rng(seed)

    shape, mu0 = _base(X)
    b0 = calibrate_b0(zero_rate, d=X.shape[1])
    de, di = effects(X, scenario)

    p0 = expit(b0 + shape)
    p1 = expit(b0 + shape + de)
    mu1 = mu0 + di

    p = np.where(W == 1, p1, p0)
    mu = np.where(W == 1, mu1, mu0)

    buy = rng.binomial(1, p)
    Y = buy * np.exp(rng.normal(mu, sigma))

    expected_spend = np.exp(sigma**2 / 2)
    tau_level = (
        p1 * np.exp(mu1) * expected_spend
        - p0 * np.exp(mu0) * expected_spend
    )

    truth = {
        "tau_level": tau_level,
        "p0": p0,
        "p1": p1,
        "mu0": mu0,
        "mu1": mu1,
        "sigma": sigma,
    }

    return Y, truth


def simulate(
    n,
    scenario="opposed",
    zero_rate=0.9,
    d=10,
    sigma=0.8,
    e=0.5,
    seed=0,
):
    """Simulate X, W, Y, and exact treatment-effect truth."""
    if n < 1:
        raise ValueError("n must be positive.")
    if d < 5:
        raise ValueError("d must be at least 5.")
    if not np.isfinite(e) or not 0 < e < 1:
        raise ValueError("e must be finite and strictly between 0 and 1.")

    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    W = rng.binomial(1, e, n)

    Y, truth = simulate_from_X(
        X=X,
        W=W,
        scenario=scenario,
        zero_rate=zero_rate,
        sigma=sigma,
        seed=seed + 1,
    )

    return X, W, Y, truth


def true_transformed_cate(truth, m, a=1.0):
    """Compute exact transformed-outcome CATE using Gauss-Hermite quadrature.

    The transform m must satisfy m(0) = 0.
    """
    if a <= 0 or not np.isfinite(a):
        raise ValueError("a must be finite and positive.")

    def expected_transform(p, mu):
        spend = np.exp(mu[:, None] + truth["sigma"] * GH_X[None, :])
        transformed = m(a * spend)
        return p * (transformed * GH_W[None, :]).sum(axis=1)

    return (
        expected_transform(truth["p1"], truth["mu1"])
        - expected_transform(truth["p0"], truth["mu0"])
    )
