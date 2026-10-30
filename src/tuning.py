
"""Reproducible hyperparameter tuning for outcome models."""

from itertools import product

import numpy as np
from sklearn.metrics import (
    mean_squared_error,
    mean_tweedie_deviance,
)

from src.models import L2, Tweedie, Hurdle


MODEL_CLASSES = {
    "L2": L2,
    "Tweedie": Tweedie,
    "Hurdle": Hurdle,
}

BASE_SEARCH_SPACE = {
    "n_estimators": [100, 200, 400],
    "learning_rate": [0.03, 0.05, 0.10],
    "num_leaves": [7, 15, 31],
    "max_depth": [3, 5, 7, -1],
    "min_child_samples": [20, 50, 100],
}

TWEEDIE_POWERS = [1.1, 1.3, 1.5, 1.7]


def candidate_configurations(model_name="L2", n_candidates=12, seed=42):
    """Generate a reproducible sample of unique candidate configurations."""
    if model_name not in MODEL_CLASSES:
        raise ValueError(f"Unknown model {model_name!r}.")

    if (
        not isinstance(n_candidates, (int, np.integer))
        or isinstance(n_candidates, (bool, np.bool_))
        or n_candidates < 1
    ):
        raise ValueError("n_candidates must be a positive integer.")

    keys = list(BASE_SEARCH_SPACE)
    values = [BASE_SEARCH_SPACE[key] for key in keys]

    configurations = []

    for combination in product(*values):
        config = dict(zip(keys, combination))

        if model_name == "Tweedie":
            for power in TWEEDIE_POWERS:
                configurations.append({**config, "power": power})
        else:
            configurations.append(config)

    if n_candidates > len(configurations):
        raise ValueError(
            f"Requested {n_candidates} candidates, but only "
            f"{len(configurations)} unique configurations exist."
        )

    rng = np.random.default_rng(seed)
    chosen = rng.choice(
        len(configurations),
        size=n_candidates,
        replace=False,
    )

    return [configurations[int(index)] for index in chosen]


def _new_model(model_name, config):
    """Create a fresh model from one candidate configuration."""
    config = dict(config)

    if model_name == "Tweedie":
        power = config.pop("power")
        return Tweedie(power=power, **config)

    return MODEL_CLASSES[model_name](**config)


def _validate_data(X, W, y, label):
    """Validate one treatment-outcome dataset."""
    X = np.asarray(X, dtype=float)
    W = np.asarray(W).ravel()
    y = np.asarray(y, dtype=float).ravel()

    if X.ndim != 2 or len(X) == 0:
        raise ValueError(f"{label}: X must be a nonempty 2D array.")

    if len(X) != len(W) or len(X) != len(y):
        raise ValueError(f"{label}: X, W, and y must have equal lengths.")

    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError(f"{label}: X and y must be finite.")

    if not np.isin(W, [0, 1]).all():
        raise ValueError(f"{label}: W must contain only 0 and 1.")

    return X, W.astype(int), y


def _selection_metric(model_name, config):
    """Return the validation metric used to select a candidate."""
    if model_name == "Tweedie":
        return "tweedie_deviance"

    return "rmse"


def _calculate_selection_score(model_name, config, y_true, predictions):
    """Calculate the candidate's validation selection score."""
    if model_name == "Tweedie":
        power = config["power"]

        if np.any(y_true < 0) or np.any(predictions < 0):
            raise ValueError(
                "Tweedie deviance requires nonnegative outcomes "
                "and predictions."
            )

        return float(
            mean_tweedie_deviance(
                y_true,
                predictions,
                power=power,
            )
        )

    return float(np.sqrt(mean_squared_error(y_true, predictions)))


def tune_outcome_model(
    X_train,
    W_train,
    y_train,
    X_val,
    W_val,
    y_val,
    model_name="L2",
    n_candidates=12,
    seed=42,
    candidate_configs=None,
):
    """Select outcome-model hyperparameters using validation data.

    Fits separate outcome models in the control and treatment arms.

    L2 and Hurdle candidates are selected using validation RMSE.
    Tweedie candidates are selected using validation Tweedie deviance
    calculated with each candidate's own variance power.

    Returns selected parameters, validation score, candidate history,
    and the two fitted arm-specific models.
    """
    if model_name not in MODEL_CLASSES:
        raise ValueError(f"Unknown model {model_name!r}.")

    X_train, W_train, y_train = _validate_data(
        X_train, W_train, y_train, "Training data"
    )

    X_val, W_val, y_val = _validate_data(
        X_val, W_val, y_val, "Validation data"
    )

    if X_train.shape[1] != X_val.shape[1]:
        raise ValueError("Training and validation features must match.")

    if set(np.unique(W_train)) != {0, 1}:
        raise ValueError("Training data must contain both treatment arms.")

    if set(np.unique(W_val)) != {0, 1}:
        raise ValueError("Validation data must contain both treatment arms.")

    if candidate_configs is None:
        candidates = candidate_configurations(
            model_name=model_name,
            n_candidates=n_candidates,
            seed=seed,
        )
    else:
        candidates = [dict(config) for config in candidate_configs]

        if not candidates:
            raise ValueError("candidate_configs cannot be empty.")

    metric_name = _selection_metric(model_name, {})

    history = []

    for candidate_id, config in enumerate(candidates):
        record = {
            "candidate_id": candidate_id,
            "model": model_name,
            "parameters": repr(config),
            "selection_metric": metric_name,
            "validation_score": np.nan,
            "validation_rmse": np.nan,
            "validation_deviance": np.nan,
            "status": "failed",
            "error": "",
        }

        try:
            arm_models = {}

            for arm in (0, 1):
                arm_mask = W_train == arm

                arm_model = _new_model(model_name, config)
                arm_model.fit(
                    X_train[arm_mask],
                    y_train[arm_mask],
                )

                arm_models[arm] = arm_model

            predictions = np.empty(len(y_val), dtype=float)

            for arm in (0, 1):
                mask = W_val == arm

                predictions[mask] = arm_models[arm].predict(
                    X_val[mask]
                )

            if not np.isfinite(predictions).all():
                raise ValueError(
                    "Validation predictions are not finite."
                )

            rmse = float(
                np.sqrt(mean_squared_error(y_val, predictions))
            )

            score = _calculate_selection_score(
                model_name,
                config,
                y_val,
                predictions,
            )

            record["validation_rmse"] = rmse
            record["validation_score"] = score

            if model_name == "Tweedie":
                record["validation_deviance"] = score

            record["status"] = "ok"
            record["_models"] = arm_models

        except Exception as exc:
            record["error"] = (
                f"{type(exc).__name__}: {exc}"
            )

        history.append(record)

    successful = [
        record for record in history
        if record["status"] == "ok"
    ]

    if not successful:
        errors = "; ".join(
            record["error"]
            for record in history
            if record["error"]
        )

        raise RuntimeError(
            f"Every tuning candidate failed. {errors}"
        )

    best = min(
        successful,
        key=lambda record: record["validation_score"],
    )

    public_history = [
        {
            key: value
            for key, value in record.items()
            if key != "_models"
        }
        for record in history
    ]

    return {
        "model": model_name,
        "best_params": dict(candidates[best["candidate_id"]]),
        "best_validation_score": best["validation_score"],
        "selection_metric": metric_name,
        "best_validation_rmse": best["validation_rmse"],
        "best_validation_deviance": best["validation_deviance"],
        "models": best["_models"],
        "history": public_history,
        "requested_candidates": len(candidates),
        "successful_candidates": len(successful),
        "failed_candidates": len(history) - len(successful),
        "seed": seed,
    }


def run_tuning_experiment(
    X,
    W,
    y,
    model_name="L2",
    n_candidates=12,
    seed=42,
):
    """Split data, tune on validation data, refit, and test once."""
    from sklearn.model_selection import train_test_split

    X, W, y = _validate_data(
        X, W, y, "Full dataset"
    )

    if len(X) < 20:
        raise ValueError("At least 20 observations are required.")

    if set(np.unique(W)) != {0, 1}:
        raise ValueError("Full dataset must contain both treatment arms.")

    indices = np.arange(len(y))

    # Reserve 20% for the final test set.
    train_val_idx, test_idx = train_test_split(
        indices,
        test_size=0.20,
        random_state=seed,
        stratify=W,
    )

    # Split the remaining 80% into 60% training and 20% validation.
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=0.25,
        random_state=seed + 1,
        stratify=W[train_val_idx],
    )

    tuned = tune_outcome_model(
        X[train_idx],
        W[train_idx],
        y[train_idx],
        X[val_idx],
        W[val_idx],
        y[val_idx],
        model_name=model_name,
        n_candidates=n_candidates,
        seed=seed,
    )

    # Refit both arm-specific models on training + validation data.
    refit_idx = np.concatenate([train_idx, val_idx])

    final_models = {}

    for arm in (0, 1):
        arm_mask = W[refit_idx] == arm

        model = _new_model(
            model_name,
            tuned["best_params"],
        )

        model.fit(
            X[refit_idx][arm_mask],
            y[refit_idx][arm_mask],
        )

        final_models[arm] = model

    # Evaluate factual predictions on the untouched test set.
    test_predictions = np.empty(len(test_idx), dtype=float)

    for arm in (0, 1):
        arm_mask = W[test_idx] == arm

        test_predictions[arm_mask] = final_models[arm].predict(
            X[test_idx][arm_mask]
        )

    if not np.isfinite(test_predictions).all():
        raise ValueError(
            "Final test predictions contain non-finite values."
        )

    test_rmse = float(
        np.sqrt(
            mean_squared_error(
                y[test_idx],
                test_predictions,
            )
        )
    )

    return {
        "model": model_name,
        "best_params": tuned["best_params"],
        "best_validation_score": tuned["best_validation_score"],
        "selection_metric": tuned["selection_metric"],
        "best_validation_rmse": tuned["best_validation_rmse"],
        "best_validation_deviance": tuned["best_validation_deviance"],
        "test_rmse": test_rmse,
        "models": final_models,
        "history": tuned["history"],
        "requested_candidates": tuned["requested_candidates"],
        "successful_candidates": tuned["successful_candidates"],
        "failed_candidates": tuned["failed_candidates"],
        "seed": seed,
        "n_train": len(train_idx),
        "n_validation": len(val_idx),
        "n_test": len(test_idx),
        "test_indices": test_idx,
        "test_predictions": test_predictions,
    }
