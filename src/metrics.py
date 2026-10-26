
"""Evaluation metrics for treatment-effect estimation and policy learning."""

import numpy as np
from scipy.stats import spearmanr


def rank_corr(a, b):
    """Return Spearman rank correlation, guarding against constant inputs."""
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()

    if a.shape != b.shape:
        raise ValueError("a and b must have the same shape.")
    if a.size < 2:
        return 0.0
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError("Inputs must contain only finite values.")

    # Spearman correlation is undefined if either vector is constant.
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return 0.0

    return float(spearmanr(a, b).statistic)


def topk_mask(score, k):
    """Select the top k fraction of scores."""
    score = np.asarray(score, dtype=float).ravel()

    if score.size == 0:
        raise ValueError("score must not be empty.")
    if not 0 < k <= 1:
        raise ValueError("k must be in (0, 1].")
    if not np.all(np.isfinite(score)):
        raise ValueError("Scores must be finite.")

    n = max(1, int(round(k * len(score))))
    mask = np.zeros(len(score), dtype=bool)
    mask[np.argsort(-score)[:n]] = True
    return mask


def topk_overlap(a, b, k=0.2):
    """Compute intersection-over-union of two top-k selections."""
    A = topk_mask(a, k)
    B = topk_mask(b, k)
    return float((A & B).sum() / (A | B).sum())


def policy_value(tau_hat, tau_true, k=0.2):
    """Mean true treatment effect among units selected by predicted effects."""
    tau_hat = np.asarray(tau_hat, dtype=float).ravel()
    tau_true = np.asarray(tau_true, dtype=float).ravel()

    if tau_hat.shape != tau_true.shape:
        raise ValueError("tau_hat and tau_true must have the same shape.")

    return float(tau_true[topk_mask(tau_hat, k)].mean())


def regret(tau_hat, tau_true, k=0.2):
    """Difference between oracle and estimated-policy value."""
    return policy_value(tau_true, tau_true, k) - policy_value(
        tau_hat, tau_true, k
    )


def qini_auc(tau_hat, W, y, e=0.5):
    """Compute the starter's normalized IPW cumulative-gain summary."""
    tau_hat = np.asarray(tau_hat, dtype=float).ravel()
    W = np.asarray(W).ravel()
    y = np.asarray(y, dtype=float).ravel()

    if not (len(tau_hat) == len(W) == len(y)) or len(y) == 0:
        raise ValueError("Inputs must have the same nonzero length.")
    if not 0 < e < 1:
        raise ValueError("e must be in (0, 1].")
    if not np.all(np.isin(W, [0, 1])):
        raise ValueError("W must contain only 0 and 1.")

    order = np.argsort(-tau_hat)
    w_sorted, y_sorted = W[order], y[order]
    contribution = y_sorted * (
        w_sorted / e - (1 - w_sorted) / (1 - e)
    )

    return float(np.cumsum(contribution).sum() / len(y) ** 2)


def ipw_policy_value(tau_hat, W, y, k=0.2, e=0.5):
    """Estimate real-data policy value using inverse-propensity weighting."""
    tau_hat = np.asarray(tau_hat, dtype=float).ravel()
    W = np.asarray(W).ravel()
    y = np.asarray(y, dtype=float).ravel()

    if not (len(tau_hat) == len(W) == len(y)) or len(y) == 0:
        raise ValueError("Inputs must have the same nonzero length.")
    if not 0 < e < 1:
        raise ValueError("e must be in (0, 1).")
    if not np.all(np.isin(W, [0, 1])):
        raise ValueError("W must contain only 0 and 1.")

    treatment_policy = topk_mask(tau_hat, k).astype(int)
    matches_policy = W == treatment_policy
    propensity = np.where(W == 1, e, 1 - e)

    return float(np.mean(y * matches_policy / propensity))
