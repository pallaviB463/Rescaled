
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


def simulate(
    n,
    scenario="opposed",
    zero_rate=0.9,
    d=10,
    sigma=0.8,
    e=0.5,
    seed=0,
):
    """Simulate outcomes and return X, W, Y, and exact treatment-effect truth."""
    if n < 1:
        raise ValueError("n must be positive.")
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario!r}")
    if not 0 < zero_rate < 1:
        raise ValueError("zero_rate must be between 0 and 1.")
    if d < 5:
        raise ValueError("d must be at least 5.")
    if sigma < 0:
        raise ValueError("sigma must be non-negative.")
    if not 0 < e < 1:
        raise ValueError("e must be between 0 and 1.")

    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    W = rng.binomial(1, e, n)

    shape, mu0 = _base(X)
    b0 = calibrate_b0(zero_rate, d)
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

    return X, W, Y, truth


def true_transformed_cate(truth, m, a=1.0):
    """Compute the exact transformed-outcome CATE using Gauss-Hermite quadrature.

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
