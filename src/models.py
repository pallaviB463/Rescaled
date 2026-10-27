"""models.py  (OWNER: Member P). Stand-in copy.
Backend: LightGBM (the planned stack). If LightGBM is not installed we fall back to scikit-learn so the
code still runs; the fallback is for development only (Tweedie is replaced by Poisson) and BACKEND is
written into every results row so fallback results can never be mistaken for real ones."""
import warnings
import numpy as np

try:
    from lightgbm import LGBMRegressor, LGBMClassifier
    BACKEND = "lightgbm"
    KW = dict(n_estimators=200, learning_rate=0.05, num_leaves=15, min_child_samples=50, verbose=-1,
              random_state=0, deterministic=True, force_row_wise=True, n_jobs=1)

    def _reg(objective=None, **extra):
        kw = dict(KW, **extra)
        if objective: kw["objective"] = objective
        return LGBMRegressor(**kw)

    def _clf(): return LGBMClassifier(**KW)
except ImportError:  # development fallback
    from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
    BACKEND = "sklearn-fallback"
    warnings.warn("LightGBM not found: using scikit-learn fallback. Do NOT report these results.")
    _SK = dict(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=50,
               early_stopping=False, random_state=0)

    def _reg(objective=None, **extra):
        loss = {None: "squared_error", "poisson": "poisson", "gamma": "gamma", "tweedie": "poisson"}[objective]
        return HistGradientBoostingRegressor(loss=loss, **_SK)

    def _clf(): return HistGradientBoostingClassifier(**_SK)


class L2:  # plain regression (transformed outcomes, pseudo-outcomes)
    def fit(self, X, y): self.m = _reg().fit(X, y); return self
    def predict(self, X): return self.m.predict(X)


class Poisson:  # log-link mean model on the level scale (fine for non-integer y >= 0)
    def fit(self, X, y): self.m = _reg("poisson").fit(X, y); return self
    def predict(self, X): return self.m.predict(X)


class Tweedie:
    def __init__(self, p=1.3): self.p = p
    def fit(self, X, y): self.m = _reg("tweedie", tweedie_variance_power=self.p).fit(X, y); return self
    def predict(self, X): return self.m.predict(X)


class Hurdle:  # P(Y>0) * E[Y | Y>0]
    def fit(self, X, y):
        pos = y > 0
        self.c = _clf().fit(X, pos)
        self.r = _reg("gamma").fit(X[pos], y[pos])
        return self
    def predict(self, X): return self.c.predict_proba(X)[:, 1] * self.r.predict(X)
