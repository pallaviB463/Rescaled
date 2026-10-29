
"""Naive log1p back-transformation baseline for causal effects."""

import numpy as np
import pandas as pd

from src.backtransform import backtransform_log1p
from src.dgp import simulate
from src.metrics import rank_corr, topk_overlap, policy_value, regret
from src.models import L2


def run_backtransform(
    scenario="opposed",
    n=5000,
    zero_rate=0.9,
    sigma=0.8,
    e=0.5,
    seed=0,
):
    """Fit arm-specific log1p outcome models and evaluate on the level scale.

    Predictions are transformed back using expm1. This is a naive baseline:
    exp(E[log1p(Y) | X]) - 1 is not generally E[Y | X].
    """
    if n < 2:
        raise ValueError("n must be at least 2.")
    if not 0 < e < 1:
        raise ValueError("e must be strictly between 0 and 1.")
    if not 0 < zero_rate < 1:
        raise ValueError("zero_rate must be strictly between 0 and 1.")
    if not np.isfinite(sigma) or sigma < 0:
        raise ValueError("sigma must be finite and nonnegative.")

    X, W, y, _ = simulate(
        n=n,
        scenario=scenario,
        zero_rate=zero_rate,
        sigma=sigma,
        e=e,
        seed=seed,
    )
    X_test, _, _, truth_test = simulate(
        n=max(2000, n // 2),
        scenario=scenario,
        zero_rate=zero_rate,
        sigma=sigma,
        e=e,
        seed=seed + 1_000_003,
    )

    y = np.asarray(y, dtype=float).ravel()
    W = np.asarray(W).ravel()
    tau_true = np.asarray(truth_test["tau_level"], dtype=float).ravel()

    if not np.isfinite(y).all() or (y < 0).any():
        raise ValueError("Observed outcomes must be finite and nonnegative.")
    if not np.isfinite(tau_true).all():
        raise ValueError("Ground-truth effects must be finite.")

    control = W == 0
    treated = W == 1

    if not control.any() or not treated.any():
        raise ValueError("Training data must contain both treatment arms.")

    model_0 = L2()
    model_1 = L2()

    model_0.fit(X[control], np.log1p(y[control]))
    model_1.fit(X[treated], np.log1p(y[treated]))

    pred_0_log = model_0.predict(X_test)
    pred_1_log = model_1.predict(X_test)

    mu_0_hat = backtransform_log1p(pred_0_log)
    mu_1_hat = backtransform_log1p(pred_1_log)
    tau_hat = mu_1_hat - mu_0_hat

    if not np.isfinite(tau_hat).all():
        raise ValueError("Back-transformed treatment effects are not finite.")

    if tau_true.size < 2 or np.ptp(tau_true) == 0:
        correlation = np.nan
        overlap = np.nan
    else:
        correlation = rank_corr(tau_hat, tau_true)
        overlap = topk_overlap(tau_hat, tau_true, k=0.2)

    result = {
        "scenario": scenario,
        "learner": "NaiveBackTransform",
        "transform": "log1p",
        "scale": 1.0,
        "rank_corr": correlation,
        "topk_overlap": overlap,
        "policy_value": policy_value(tau_hat, tau_true, k=0.2),
        "regret": regret(tau_hat, tau_true, k=0.2),
        "status": "ok",
        "error": "",
        "seed": seed,
        "n": n,
        "zero_rate": zero_rate,
        "sigma": sigma,
        "e": e,
    }

    return pd.DataFrame([result])
