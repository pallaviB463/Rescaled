
"""Back-transformation baseline for log1p outcome models."""

import numpy as np


def backtransform_log1p(predictions, clip=True):
    """Convert predictions of log1p(Y) back to the original outcome scale.

    Parameters
    ----------
    predictions : array-like
        Predictions on the log1p outcome scale.
    clip : bool
        If True, clip negative back-transformed predictions to zero.
        This is useful because fitted predictions may be below zero.

    Returns
    -------
    numpy.ndarray
        Predictions on the original outcome scale.
    """
    predictions = np.asarray(predictions, dtype=float)

    if predictions.ndim != 1:
        raise ValueError("predictions must be a one-dimensional array.")
    if predictions.size == 0:
        raise ValueError("predictions must not be empty.")
    if not np.isfinite(predictions).all():
        raise ValueError("predictions must contain only finite values.")

    # expm1(z) is numerically preferable to exp(z) - 1.
    with np.errstate(over="raise", invalid="raise"):
        try:
            result = np.expm1(predictions)
        except FloatingPointError as exc:
            raise ValueError(
                "Back-transformation overflowed; predictions are too large."
            ) from exc

    if not np.isfinite(result).all():
        raise ValueError("Back-transformed predictions are not finite.")

    if clip:
        result = np.maximum(result, 0.0)

    return result
