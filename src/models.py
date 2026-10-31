
"""Outcome models for the Neurolytics project."""

import numpy as np
import lightgbm as lgb
import statsmodels.api as sm

from sklearn.base import BaseEstimator, RegressorMixin

BACKEND = "lightgbm"


LGBM_PARAMS = {
    "n_estimators": 200,
    "learning_rate": 0.05,
    "num_leaves": 15,
    "min_child_samples": 50,
    "verbosity": -1,
    "random_state": 42,
    "deterministic": True,
    "force_row_wise": True,
    "n_jobs": 1,
}


def _validate_xy(X, y, allow_negative=False):
    """Validate features and outcomes before model fitting."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()

    if X.ndim != 2:
        raise ValueError("X must be a two-dimensional array.")

    if len(X) != len(y):
        raise ValueError("X and y must have the same number of rows.")

    if len(y) == 0:
        raise ValueError("Training data cannot be empty.")

    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(y)):
        raise ValueError("X and y must contain only finite values.")

    if not allow_negative and np.any(y < 0):
        raise ValueError("Outcomes must be nonnegative.")

    return X, y


class L2(BaseEstimator, RegressorMixin):
    """LightGBM squared-error regression.

    Negative targets are allowed because this model is also used to
    estimate treatment-effect pseudo-outcomes.
    """

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def fit(self, X, y):
        X, y = _validate_xy(X, y, allow_negative=True)

        self.model_ = lgb.LGBMRegressor(
            objective="regression",
            **{**LGBM_PARAMS, **self.kwargs},
        )
        self.model_.fit(X, y)
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)

        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError("Prediction features have an incompatible shape.")

        if getattr(self, "all_zero_", False):
            return np.zeros(len(X))
        return self.model_.predict(X)


class Poisson(BaseEstimator, RegressorMixin):
    """Poisson regression for nonnegative integer-valued counts."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def fit(self, X, y):
        X, y = _validate_xy(X, y)
        self.n_features_in_ = X.shape[1]
        if np.all(y == 0):
            self.all_zero_ = True
            return self
        self.all_zero_ = False
        self.model_ = lgb.LGBMRegressor(
            objective="poisson",
            **{**LGBM_PARAMS, **self.kwargs},
        )
        self.model_.fit(X, y)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)

        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError("Prediction features have an incompatible shape.")

        if getattr(self, "all_zero_", False):
            return np.zeros(len(X))
        return self.model_.predict(X)


class Tweedie(BaseEstimator, RegressorMixin):
    """Tweedie regression for nonnegative outcomes."""

    def __init__(self, power=1.5, **kwargs):
        if not 1.0 < power < 2.0:
            raise ValueError("Tweedie power must be between 1 and 2.")

        self.power = power
        self.kwargs = kwargs

    def fit(self, X, y):
        X, y = _validate_xy(X, y)
        self.n_features_in_ = X.shape[1]
        if np.all(y == 0):
            self.all_zero_ = True
            return self
        self.all_zero_ = False
        self.model_ = lgb.LGBMRegressor(
            objective="tweedie",
            tweedie_variance_power=self.power,
            **{**LGBM_PARAMS, **self.kwargs},
        )
        self.model_.fit(X, y)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)

        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError("Prediction features have an incompatible shape.")

        if getattr(self, "all_zero_", False):
            return np.zeros(len(X))

        return self.model_.predict(X)


class Hurdle(BaseEstimator, RegressorMixin):
    """Two-part model: probability of a positive outcome times its mean."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def fit(self, X, y):
        X, y = _validate_xy(X, y)
        positive = y > 0
        self.n_features_in_ = X.shape[1]

        if not np.any(positive):
            raise ValueError(
                "Hurdle model needs at least one positive training outcome."
            )
        elif np.all(positive):
            self.all_positive_ = True
            self.occurrence_model_ = None
        else:
            self.all_positive_ = None
            self.occurrence_model_ = lgb.LGBMClassifier(
                **{**LGBM_PARAMS, **self.kwargs}
            )
            self.occurrence_model_.fit(X, positive.astype(int))

        self.positive_model_ = lgb.LGBMRegressor(
            objective="gamma",
            **{**LGBM_PARAMS, **self.kwargs},
        )
        self.positive_model_.fit(X[positive], y[positive])
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)

        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError("Prediction features have an incompatible shape.")

        if self.all_positive_ is False:
            return np.zeros(len(X))
        elif self.all_positive_ is True:
            positive_mean = self.positive_model_.predict(X)
            return np.maximum(positive_mean, 0.0)
        else:
            probability_positive = self.occurrence_model_.predict_proba(X)[:, 1]
            positive_mean = self.positive_model_.predict(X)
            return probability_positive * np.maximum(positive_mean, 0.0)


class NegBin(BaseEstimator, RegressorMixin):
    """Negative Binomial GLM for integer-valued count outcomes."""

    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y):
        X, y = _validate_xy(X, y)

        if not np.allclose(y, np.round(y)):
            raise ValueError(
                "Negative Binomial regression requires integer-valued outcomes."
            )

        if self.alpha <= 0:
            raise ValueError("alpha must be positive.")

        X_with_intercept = sm.add_constant(X, has_constant="add")

        self.model_ = sm.GLM(
            y,
            X_with_intercept,
            family=sm.families.NegativeBinomial(alpha=self.alpha),
        ).fit()

        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)

        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError("Prediction features have an incompatible shape.")

        X_with_intercept = sm.add_constant(X, has_constant="add")

        return np.asarray(self.model_.predict(X_with_intercept))
