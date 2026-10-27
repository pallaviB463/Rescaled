"""dgp.py  (OWNER: Member P).  Stand-in copy so Q can work before Hand-off 1.
Zero-inflated lognormal spend with known true effects. Fix applied: calibrate_b0 is cached."""
import numpy as np
from functools import lru_cache
from scipy.special import expit
from scipy.optimize import brentq

GH_X, GH_W = np.polynomial.hermite_e.hermegauss(40)
GH_W = GH_W / np.sqrt(2 * np.pi)
SCENARIOS = ["null", "homogeneous", "extensive", "intensive", "aligned", "opposed"]


def effects(X, scenario):
    """Treatment shifts: (delta on logit of P(buy), delta on log-spend mean)."""
    z3, z4 = X[:, 3], X[:, 4]
    zero = np.zeros(len(X))
    if scenario == "null": return zero, zero
    if scenario == "homogeneous": return zero + 0.4, zero + 0.15
    if scenario == "extensive": return 0.3 + 0.8 * z3, zero
    if scenario == "intensive": return zero, 0.2 + 0.4 * z4
    if scenario == "aligned": return 0.3 + 0.8 * z3, 0.2 + 0.4 * z3
    if scenario == "opposed": return 0.3 + 0.8 * z3, 0.2 - 0.4 * z3
    raise ValueError(scenario)


def _base(X):
    return 0.8 * X[:, 0] - 0.5 * X[:, 1], 2.5 + 0.5 * X[:, 2]  # logit shape, log-spend mean (control)


@lru_cache(maxsize=None)
def calibrate_b0(zero_rate, d=10, seed=123):
    Xp = np.random.default_rng(seed).normal(size=(200_000, d))
    shape, _ = _base(Xp)
    return brentq(lambda b: expit(b + shape).mean() - (1 - zero_rate), -30, 10)


def simulate(n, scenario="opposed", zero_rate=0.9, d=10, sigma=0.8, e=0.5, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    W = rng.binomial(1, e, n)
    shape, mu0 = _base(X)
    b0 = calibrate_b0(zero_rate, d)
    de, di = effects(X, scenario)
    p0, p1 = expit(b0 + shape), expit(b0 + shape + de)
    mu1 = mu0 + di
    p, mu = np.where(W == 1, p1, p0), np.where(W == 1, mu1, mu0)
    buy = rng.binomial(1, p)
    Y = buy * np.exp(rng.normal(mu, sigma))
    truth = dict(p0=p0, p1=p1, mu0=mu0, mu1=mu1, sigma=sigma,
                 tau_level=p1 * np.exp(mu1 + sigma**2 / 2) - p0 * np.exp(mu0 + sigma**2 / 2))
    return X, W, Y, truth


def true_transformed_cate(truth, m, a=1.0):
    """E[m(aY)|x,W=1] - E[m(aY)|x,W=0]; requires m(0)=0 (identity/log1p/arcsinh/sqrt)."""
    def e_m(p, mu):
        s = np.exp(mu[:, None] + truth["sigma"] * GH_X[None, :])
        return p * (m(a * s) * GH_W[None, :]).sum(axis=1)
    return e_m(truth["p1"], truth["mu1"]) - e_m(truth["p0"], truth["mu0"])
