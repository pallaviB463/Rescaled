"""run_real.py  (OWNER: Member Q). Hillstrom MineThatData e-mail test (task Q6).
Binary comparison: Men's e-mail vs control (primary), Women's vs control (replicate).
No ground truth, so we measure: rank stability across unit scales, IPW policy value, Qini.
All scores are OUT-OF-FOLD (repeated K-fold). Transforms are applied inside the training fold only (no leakage).
  python src/run_real.py data/Hillstrom.csv mens results/hillstrom_mens.csv"""
import sys
import numpy as np, pandas as pd
from sklearn.model_selection import KFold

import models
from metalearners import LEARNERS
from metrics import rank_corr, topk_overlap, ipw_policy_value, qini_auc
from transforms import apply_transform
from run_sim import pipelines

ARMS = {"mens": "Mens E-Mail", "womens": "Womens E-Mail"}
NUM = ["recency", "history", "mens", "womens", "newbie"]   # mens/womens = past-purchase flags (covariates)
CAT = ["history_segment", "zip_code", "channel"]


def load_hillstrom(csv, arm="mens"):
    df = pd.read_csv(csv)
    keep = df[df["segment"].isin([ARMS[arm], "No E-Mail"])].reset_index(drop=True)
    W = (keep["segment"] == ARMS[arm]).astype(int).to_numpy()
    X = pd.get_dummies(keep[NUM + CAT], columns=CAT, dtype=float).to_numpy(float)
    y = keep["spend"].to_numpy(float)
    return X, W, y, float(W.mean())      # e computed from the data (about 0.5 within two arms)


def run_hillstrom(csv, arm="mens", repeats=3, scales=(0.01, 1.0, 100.0), count_scales=(0.01, 1.0, 100.0),
                  learners=("T", "DR"), k_grid=(0.1, 0.2, 0.3, 0.5), folds=5, seed=0, count_models=None):
    assert 1.0 in scales and 1.0 in count_scales
    X, W, y, e = load_hillstrom(csv, arm)
    pls = [p for p in pipelines(list(scales), list(count_scales))
           if count_models is None or p[0] == "TransformedL2" or p[0] in count_models]
    rows = []
    for rep in range(repeats):
        oof = {}
        for tr, te in KFold(folds, shuffle=True, random_state=seed + rep).split(X):
            for fam, t, a, M in pls:
                ytr = y[tr] * a if t == "level" else apply_transform(t, y[tr], a)   # train rows only
                for L in learners:
                    oof.setdefault((fam, t, a, L), np.zeros(len(y)))[te] = \
                        LEARNERS[L](M, X[tr], W[tr], ytr, X[te], e=e)
        for (fam, t, a, L), tau in oof.items():
            ref = oof[(fam, t, 1.0, L)]
            r = dict(arm=arm, repeat=rep, family=fam, transform=t, scale=a, learner=L, backend=models.BACKEND,
                     e=e, rho_vs_a1=rank_corr(tau, ref), top20_overlap_vs_a1=topk_overlap(tau, ref, 0.2),
                     qini=qini_auc(tau, W, y, e))
            for k in k_grid:
                r[f"ipw_value_k{int(k*100)}"] = ipw_policy_value(tau, W, y, k, e)
            rows.append(r)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    csv, arm, out = sys.argv[1], sys.argv[2], sys.argv[3]
    df = run_hillstrom(csv, arm)
    df.to_csv(out, index=False)
    print(df.groupby(["family", "transform", "scale", "learner"])[["rho_vs_a1", "ipw_value_k20", "qini"]].mean().round(4))
