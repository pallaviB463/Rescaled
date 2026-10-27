"""metrics.py  (OWNER: Member P). Stand-in copy. Fix applied: rank_corr guards constant vectors."""
import numpy as np
from scipy.stats import spearmanr


def rank_corr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.std() == 0 or b.std() == 0:
        return np.nan  # undefined (e.g. null scenario truth); aggregate with nanmean
    return spearmanr(a, b)[0]


def topk_mask(score, k):
    n = max(1, int(round(k * len(score))))
    m = np.zeros(len(score), bool)
    m[np.argsort(-score)[:n]] = True
    return m


def topk_overlap(a, b, k=0.2):
    A, B = topk_mask(a, k), topk_mask(b, k)
    return (A & B).sum() / (A | B).sum()


def policy_value(tau_hat, tau_true, k=0.2):  # mean TRUE level effect among those treated
    return tau_true[topk_mask(tau_hat, k)].mean()


def regret(tau_hat, tau_true, k=0.2):
    return policy_value(tau_true, tau_true, k) - policy_value(tau_hat, tau_true, k)


def qini_auc(tau_hat, W, y, e=0.5):  # IPW-based cumulative gain, x-axis normalised to [0,1]
    o = np.argsort(-tau_hat)
    w, yy = W[o], y[o]
    contrib = yy * (w / e - (1 - w) / (1 - e))
    return np.cumsum(contrib).sum() / len(y) ** 2


def ipw_policy_value(tau_hat, W, y, k=0.2, e=0.5):  # real-data value of "treat top k%, don't treat rest"
    t = topk_mask(tau_hat, k).astype(int)
    match = (W == t)
    return np.mean(y * match / np.where(W == 1, e, 1 - e))
