
"""Meta-learners for conditional average treatment effect estimation."""

import numpy as np
from sklearn.base import clone

from src.models import L2
from src.transforms import apply_transform


def _validate_propensity(e):
    """Validate a constant treatment propensity score."""
    if not np.isfinite(e) or not 0 < e < 1:
        raise ValueError("e must be strictly between 0 and 1.")


def _validate_data(X, W, y, e=0.5):
    """Validate common meta-learner inputs."""
    _validate_propensity(e)

    X = np.asarray(X, dtype=float)
    W = np.asarray(W).ravel()
    y = np.asarray(y, dtype=float).ravel()

    if X.ndim != 2:
        raise ValueError("X must be a two-dimensional array.")

    if len(X) != len(W) or len(W) != len(y):
        raise ValueError("X, W, and y must have the same number of rows.")

    if len(y) == 0:
        raise ValueError("Training data cannot be empty.")

    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(y)):
        raise ValueError("X and y must contain only finite values.")

    if not np.all(np.isin(W, [0, 1])):
        raise ValueError("W must contain only 0 and 1.")

    if not np.any(W == 0) or not np.any(W == 1):
        raise ValueError("Both treatment groups must be present.")

    return X, W.astype(int), y


def _fit_model(model, X, y):
    """Fit a fresh model instance to avoid reusing fitted state."""
    fitted = clone(model)
    fitted.fit(X, y)
    return fitted


class SLearner:
    """One outcome model with treatment included as a feature."""

    def __init__(self, model=None, e=0.5):
        _validate_propensity(e)
        self.model = model if model is not None else L2()
        self.e = e

    def fit(self, X, W, y):
        X, W, y = _validate_data(X, W, y, self.e)

        X_augmented = np.column_stack([X, W])
        self.model_ = _fit_model(self.model, X_augmented, y)
        self.n_features_in_ = X.shape[1]

        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)

        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError("X has an incompatible shape.")

        X0 = np.column_stack([X, np.zeros(len(X))])
        X1 = np.column_stack([X, np.ones(len(X))])

        return self.model_.predict(X1) - self.model_.predict(X0)


class TLearner:
    """Separate outcome models for control and treatment groups."""

    def __init__(self, model=None, e=0.5):
        _validate_propensity(e)
        self.model = model if model is not None else L2()
        self.e = e

    def fit(self, X, W, y):
        X, W, y = _validate_data(X, W, y, self.e)

        self.model0_ = _fit_model(self.model, X[W == 0], y[W == 0])
        self.model1_ = _fit_model(self.model, X[W == 1], y[W == 1])
        self.n_features_in_ = X.shape[1]

        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)

        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError("X has an incompatible shape.")

        return self.model1_.predict(X) - self.model0_.predict(X)


class XLearner:
    """X-learner with a constant propensity score."""

    def __init__(self, model=None, e=0.5):
        _validate_propensity(e)
        self.model = model if model is not None else L2()
        self.e = e

    def fit(self, X, W, y):
        X, W, y = _validate_data(X, W, y, self.e)

        X0, y0 = X[W == 0], y[W == 0]
        X1, y1 = X[W == 1], y[W == 1]

        self.mu0_ = _fit_model(self.model, X0, y0)
        self.mu1_ = _fit_model(self.model, X1, y1)

        # Imputed individual treatment effects.
        d1 = y1 - self.mu0_.predict(X1)
        d0 = self.mu1_.predict(X0) - y0

        self.tau1_ = _fit_model(self.model, X1, d1)
        self.tau0_ = _fit_model(self.model, X0, d0)
        self.n_features_in_ = X.shape[1]

        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)

        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError("X has an incompatible shape.")

        tau0 = self.tau0_.predict(X)
        tau1 = self.tau1_.predict(X)

        return self.e * tau0 + (1.0 - self.e) * tau1


class DRLearner:
    """Doubly robust learner using a constant propensity score."""

    def __init__(self, model=None, e=0.5):
        _validate_propensity(e)
        self.model = model if model is not None else L2()
        self.e = e

    def fit(self, X, W, y):
        X, W, y = _validate_data(X, W, y, self.e)

        X0, y0 = X[W == 0], y[W == 0]
        X1, y1 = X[W == 1], y[W == 1]

        self.mu0_ = _fit_model(self.model, X0, y0)
        self.mu1_ = _fit_model(self.model, X1, y1)

        mu0 = self.mu0_.predict(X)
        mu1 = self.mu1_.predict(X)

        pseudo_outcome = (
            mu1 - mu0
            + W * (y - mu1) / self.e
            - (1 - W) * (y - mu0) / (1.0 - self.e)
        )

        self.tau_model_ = _fit_model(self.model, X, pseudo_outcome)
        self.n_features_in_ = X.shape[1]

        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)

        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError("X has an incompatible shape.")

        return self.tau_model_.predict(X)


class TransformedOutcomeLearner:
    """Estimate treatment effects from transformed outcomes."""

    def __init__(self, transform="identity", scale=1.0, model=None, e=0.5):
        _validate_propensity(e)

        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("scale must be finite and positive.")

        self.transform = transform
        self.scale = scale
        self.model = model if model is not None else L2()
        self.e = e

    def fit(self, X, W, y):
        X, W, y = _validate_data(X, W, y, self.e)

        transformed_y = apply_transform(self.transform, y, a=self.scale)

        pseudo_outcome = transformed_y * (W - self.e) / (
            self.e * (1.0 - self.e)
        )

        self.tau_model_ = _fit_model(self.model, X, pseudo_outcome)
        self.n_features_in_ = X.shape[1]

        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)

        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError("X has an incompatible shape.")

        return self.tau_model_.predict(X)
