
"""Outcome transformations for the Neurolytics project."""

import numpy as np
from scipy.stats import rankdata

TRANSFORMS = {
    "identity": lambda y: y,
    "log1p": np.log1p,
    "arcsinh": np.arcsinh,
    "sqrt": np.sqrt,
}

SCALES = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0]


def apply_transform(name, y, a=1.0):
    """Apply a supported transformation after multiplying outcomes by scale a.

    Parameters
    ----------
    name : str
        One of identity, log1p, arcsinh, sqrt, or rank.
    y : array-like
        Non-negative outcome values.
    a : float
        Positive scale factor.

    Returns
    -------
    numpy.ndarray
        Transformed outcomes.
    """
    y = np.asarray(y, dtype=float)

    if y.ndim != 1:
        raise ValueError("y must be a one-dimensional array.")
    if y.size == 0:
        raise ValueError("y must not be empty.")
    if not np.all(np.isfinite(y)):
        raise ValueError("y must contain only finite values.")
    if np.any(y < 0):
        raise ValueError("y must contain non-negative values.")
    if not np.isfinite(a) or a <= 0:
        raise ValueError("a must be a finite positive number.")

    if name == "rank":
        # Rank is invariant to positive rescaling.
        return rankdata(y, method="average") / len(y)

    if name not in TRANSFORMS:
        raise ValueError(
            f"Unknown transformation {name!r}. "
            f"Choose from {list(TRANSFORMS)} or 'rank'."
        )

    return np.asarray(TRANSFORMS[name](a * y), dtype=float)
