
from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.model_selection import (
    KFold,
    cross_val_predict,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from lightgbm import LGBMClassifier, LGBMRegressor

warnings.filterwarnings("ignore", category=UserWarning)

# ============================================================
# CONFIGURATION
# ============================================================

SEEDS = [
    42, 123, 2026, 7, 99,
    314, 2718, 867, 1501, 2025,
    88, 456, 789, 1001, 2024,
    17, 256, 512, 1024, 4096,
]

TEST_SIZE = 0.20
TOP_FRACTION = 0.20
N_BOOTSTRAP = 2000

# Men's E-Mail = treatment; No E-Mail = control.
# The comparison is approximately 50/50 randomized.
TREATMENT_PROPENSITY = 0.5

DATA_PATH = Path("data/hillstrom/hillstrom_mens_control.csv")
OUT_DIR = Path("results/hillstrom")
OUT_DIR.mkdir(parents=True, exist_ok=True)

NUMERIC = ["recency", "history", "newbie"]
CATEGORICAL = ["history_segment", "zip_code", "channel"]
FEATURES = NUMERIC + CATEGORICAL
TARGETS = ["spend", "visit", "conversion"]


# ============================================================
# PREPROCESSING AND MODELS
# ============================================================

def make_preprocessor():
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer([
        ("num", numeric_pipe, NUMERIC),
        ("cat", categorical_pipe, CATEGORICAL),
    ])


def make_regressor(kind, seed):
    common = {
        "n_estimators": 250,
        "learning_rate": 0.04,
        "num_leaves": 15,
        "min_child_samples": 50,
        "verbosity": -1,
        "random_state": seed,
    }

    if kind == "L2":
        return LGBMRegressor(
            objective="regression",
            **common,
        )

    if kind == "Tweedie":
        return LGBMRegressor(
            objective="tweedie",
            tweedie_variance_power=1.5,
            **common,
        )

    raise ValueError(f"Unknown regression model: {kind}")


def make_regression_pipeline(model):
    return Pipeline([
        ("prep", make_preprocessor()),
        ("model", model),
    ])


# ============================================================
# FIT MODELS FOR ONE TREATMENT ARM
# ============================================================

def fit_regression_arm(X, y, kind, transform, seed):
    """
    Fit an L2 or Tweedie outcome model.

    For log1p outcomes, estimate a Duan smearing correction
    from out-of-fold residuals using training data only.
    """
    y = np.asarray(y, dtype=float)

    if np.any(y < 0):
        raise ValueError("Outcome values must be nonnegative.")

    y_fit = np.log1p(y) if transform == "log1p" else y

    pipe = make_regression_pipeline(make_regressor(kind, seed))

    if transform == "log1p":
        cv = KFold(
            n_splits=3,
            shuffle=True,
            random_state=seed,
        )

        oof_pred = cross_val_predict(
            clone(pipe),
            X.reset_index(drop=True),
            y_fit,
            cv=cv,
            n_jobs=1,
        )

        residuals = y_fit - oof_pred
        smear = float(
            np.mean(np.exp(np.clip(residuals, -30, 30)))
        )
    else:
        smear = 1.0

    pipe.fit(X, y_fit)

    return {
        "kind": "regression",
        "pipeline": pipe,
        "transform": transform,
        "smear": smear,
    }


def fit_hurdle_arm(X, y, transform, seed):
    """
    Fit a two-part model:
      1. Probability of positive spending.
      2. Spending amount conditional on positive spending.

    Expected spend is predicted probability multiplied by the
    predicted conditional positive-spend amount.
    """
    y = np.asarray(y, dtype=float)

    if np.any(y < 0):
        raise ValueError("Hurdle outcomes must be nonnegative.")

    positive = y > 0

    if positive.sum() < 10:
        raise ValueError(
            "Too few positive-spend observations for Hurdle model."
        )

    if positive.sum() == len(y):
        raise ValueError(
            "Hurdle occurrence model requires zero-spend observations."
        )

    occurrence_pipe = Pipeline([
        ("prep", make_preprocessor()),
        ("model", LGBMClassifier(
            n_estimators=250,
            learning_rate=0.04,
            num_leaves=15,
            min_child_samples=50,
            verbosity=-1,
            random_state=seed,
        )),
    ])

    occurrence_pipe.fit(X, positive.astype(int))

    X_positive = X.loc[positive].reset_index(drop=True)
    y_positive = y[positive]

    amount_target = (
        np.log1p(y_positive)
        if transform == "log1p"
        else y_positive
    )

    amount_pipe = Pipeline([
        ("prep", make_preprocessor()),
        ("model", LGBMRegressor(
            objective="regression_l1",
            n_estimators=250,
            learning_rate=0.04,
            num_leaves=10,
            min_child_samples=20,
            verbosity=-1,
            random_state=seed,
        )),
    ])

    if transform == "log1p":
        cv = KFold(
            n_splits=3,
            shuffle=True,
            random_state=seed,
        )

        oof_pred = cross_val_predict(
            clone(amount_pipe),
            X_positive,
            amount_target,
            cv=cv,
            n_jobs=1,
        )

        residuals = amount_target - oof_pred
        smear = float(
            np.mean(np.exp(np.clip(residuals, -30, 30)))
        )
    else:
        smear = 1.0

    amount_pipe.fit(X_positive, amount_target)

    return {
        "kind": "hurdle",
        "occurrence": occurrence_pipe,
        "amount": amount_pipe,
        "transform": transform,
        "smear": smear,
    }


def fit_arm(X, y, model_kind, transform, seed):
    if model_kind == "Hurdle":
        return fit_hurdle_arm(X, y, transform, seed)

    return fit_regression_arm(
        X,
        y,
        model_kind,
        transform,
        seed,
    )


# ============================================================
# PREDICTION ON THE ORIGINAL OUTCOME SCALE
# ============================================================

def predict_arm(fitted, X):
    transform = fitted["transform"]

    if fitted["kind"] == "hurdle":
        p_positive = fitted["occurrence"].predict_proba(X)[:, 1]
        amount_prediction = fitted["amount"].predict(X)

        if transform == "log1p":
            # Approximate conditional arithmetic mean using
            # the training-only smearing correction.
            positive_mean = (
                np.exp(np.clip(amount_prediction, -30, 30))
                * fitted["smear"]
                - 1.0
            )
        else:
            positive_mean = amount_prediction

        return np.maximum(
            p_positive * np.maximum(positive_mean, 0.0),
            0.0,
        )

    prediction = fitted["pipeline"].predict(X)

    if transform == "log1p":
        prediction = (
            np.exp(np.clip(prediction, -30, 30))
            * fitted["smear"]
            - 1.0
        )

    return np.maximum(
        np.asarray(prediction, dtype=float),
        0.0,
    )


# ============================================================
# POLICY EVALUATION
# ============================================================

def evaluate_policy_with_random_baseline(
    y,
    W,
    score,
    seed,
    fraction=TOP_FRACTION,
    n_bootstrap=N_BOOTSTRAP,
):
    """
    Compare uplift-ranked targeting against expected random targeting.

    Policy value is an inverse-probability-weighted estimate of
    incremental outcome versus treating nobody.

    Bootstrap intervals are conditional on the fitted model and
    test split. They do not account for model-training uncertainty.
    """
    y = np.asarray(y, dtype=float)
    W = np.asarray(W, dtype=int)
    score = np.asarray(score, dtype=float)

    if not (len(y) == len(W) == len(score)):
        raise ValueError("y, W, and score must have equal lengths.")

    if len(y) == 0:
        raise ValueError("Cannot evaluate an empty test set.")

    if not np.isin(W, [0, 1]).all():
        raise ValueError("W must contain only 0 and 1.")

    if not np.isfinite(y).all() or not np.isfinite(score).all():
        raise ValueError("Outcome and score arrays must be finite.")

    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1].")

    if not np.any(W == 1) or not np.any(W == 0):
        raise ValueError(
            "Test set must contain both treatment and control customers."
        )

    n = len(y)
    n_selected = max(1, int(np.ceil(n * fraction)))
    actual_fraction = n_selected / n

    # Approximate known randomization propensity for this comparison.
    p1 = TREATMENT_PROPENSITY
    p0 = 1.0 - TREATMENT_PROPENSITY

    # IPW contribution for each customer.
    contribution = W * y / p1 - (1 - W) * y / p0

    # Select the customers with the highest predicted uplift.
    selected = np.zeros(n, dtype=bool)
    selected[np.argsort(score, kind="mergesort")[-n_selected:]] = True

    model_contrib = selected.astype(float) * contribution

    model_value = float(model_contrib.mean())
    treat_all_value = float(contribution.mean())

    # Expected value of a random policy selecting the same fraction.
    random_expected_value = actual_fraction * treat_all_value
    model_minus_random = model_value - random_expected_value

    # Paired customer bootstrap: model policy minus expected random policy.
    paired_difference = (
        selected.astype(float) - actual_fraction
    ) * contribution

    rng = np.random.default_rng(seed + 200_000)

    model_boot = np.empty(n_bootstrap)
    random_boot = np.empty(n_bootstrap)
    difference_boot = np.empty(n_bootstrap)

    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)

        model_boot[b] = model_contrib[idx].mean()
        random_boot[b] = (
            actual_fraction * contribution[idx].mean()
        )
        difference_boot[b] = paired_difference[idx].mean()

    model_ci = np.quantile(model_boot, [0.025, 0.975])
    random_ci = np.quantile(random_boot, [0.025, 0.975])
    difference_ci = np.quantile(
        difference_boot, [0.025, 0.975]
    )

    # Descriptive diagnostics for the selected group's observed outcomes.
    # These are not causal effects because observed outcomes mix arms.
    selected_y = y[selected]
    selected_positive_rate = float((selected_y > 0).mean())
    selected_max_outcome = float(selected_y.max())

    if selected_y.sum() > 0:
        top_count = max(1, int(np.ceil(len(selected_y) * 0.01)))
        top_values = np.sort(selected_y)[-top_count:]
        top_one_percent_share = float(
            top_values.sum() / selected_y.sum()
        )
    else:
        top_one_percent_share = 0.0

    return {
        "policy_value_incremental": model_value,
        "policy_value_ci_low": float(model_ci[0]),
        "policy_value_ci_high": float(model_ci[1]),

        "random_policy_value": float(random_expected_value),
        "random_policy_ci_low": float(random_ci[0]),
        "random_policy_ci_high": float(random_ci[1]),

        "model_minus_random": float(model_minus_random),
        "model_minus_random_ci_low": float(difference_ci[0]),
        "model_minus_random_ci_high": float(difference_ci[1]),

        "treat_all_incremental": treat_all_value,
        "selected_fraction": float(selected.mean()),
        "random_selected_fraction": float(actual_fraction),

        "selected_positive_outcome_rate": selected_positive_rate,
        "selected_max_observed_outcome": selected_max_outcome,
        "selected_top_1pct_outcome_share": top_one_percent_share,
    }


# ============================================================
# EXPERIMENT
# ============================================================

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Cannot find {DATA_PATH}. "
            "Run this script from the project root."
        )

    df = pd.read_csv(DATA_PATH)
    df = df.dropna(
        subset=FEATURES + TARGETS + ["treatment"]
    ).copy()

    X_all = df[FEATURES].copy()
    W_all = df["treatment"].astype(int).to_numpy()

    results_rows = []
    customer_rows = []

    for seed in SEEDS:
        idx_train, idx_test = train_test_split(
            np.arange(len(df)),
            test_size=TEST_SIZE,
            random_state=seed,
            stratify=W_all,
        )

        X_train = X_all.iloc[idx_train].reset_index(drop=True)
        X_test = X_all.iloc[idx_test].reset_index(drop=True)

        W_train = W_all[idx_train]
        W_test = W_all[idx_test]

        print(
            f"\nSeed {seed}: "
            f"train={len(idx_train)}, "
            f"test={len(idx_test)}, "
            f"test treatment rate={W_test.mean():.4f}"
        )

        for target in TARGETS:
            y_all = df[target].astype(float).to_numpy()
            y_train = y_all[idx_train]
            y_test = y_all[idx_test]

            if target == "spend":
                configurations = [
                    ("L2", "identity"),
                    ("L2", "log1p"),
                    ("Tweedie", "identity"),
                    ("Tweedie", "log1p"),
                    ("Hurdle", "identity"),
                    ("Hurdle", "log1p"),
                ]
            else:
                configurations = [("L2", "identity")]

            for model_kind, transform in configurations:
                print(f"  {target}: {model_kind}, {transform}")

                # Fit separate outcome models for treatment and control.
                fitted = {}

                for arm in [0, 1]:
                    mask = W_train == arm

                    fitted[arm] = fit_arm(
                        X_train.loc[mask].reset_index(drop=True),
                        y_train[mask],
                        model_kind,
                        transform,
                        seed + arm,
                    )

                mu0 = predict_arm(fitted[0], X_test)
                mu1 = predict_arm(fitted[1], X_test)
                uplift = mu1 - mu0

                evaluation = evaluate_policy_with_random_baseline(
                    y=y_test,
                    W=W_test,
                    score=uplift,
                    seed=seed,
                    fraction=TOP_FRACTION,
                    n_bootstrap=N_BOOTSTRAP,
                )

                results_rows.append({
                    "seed": seed,
                    "target": target,
                    "model": model_kind,
                    "transform": transform,
                    "n_train": len(idx_train),
                    "n_test": len(idx_test),
                    "mean_pred_control": float(mu0.mean()),
                    "mean_pred_treated": float(mu1.mean()),
                    "mean_pred_uplift": float(uplift.mean()),
                    "uplift_std": float(uplift.std()),
                    **evaluation,
                })

                # Save test-set predictions for later analysis.
                for j in range(len(idx_test)):
                    customer_rows.append({
                        "seed": seed,
                        "test_row_index": int(idx_test[j]),
                        "target": target,
                        "model": model_kind,
                        "transform": transform,
                        "treatment": int(W_test[j]),
                        "observed_outcome": float(y_test[j]),
                        "pred_control": float(mu0[j]),
                        "pred_treated": float(mu1[j]),
                        "pred_uplift": float(uplift[j]),
                    })

    # --------------------------------------------------------
    # SAVE DETAILED RESULTS
    # --------------------------------------------------------

    result = pd.DataFrame(results_rows)

    corrected_path = (
        OUT_DIR / "hillstrom_model_results_corrected.csv"
    )
    result.to_csv(corrected_path, index=False)

    customer_path = OUT_DIR / "hillstrom_test_predictions.csv"
    pd.DataFrame(customer_rows).to_csv(
        customer_path, index=False
    )

    # --------------------------------------------------------
    # SUMMARY ACROSS SPLITS
    # --------------------------------------------------------

    summary = (
        result.groupby(
            ["target", "model", "transform"],
            dropna=False,
        )
        .agg(
            seeds=("seed", "nunique"),

            policy_value_mean=(
                "policy_value_incremental", "mean"
            ),
            policy_value_sd=(
                "policy_value_incremental", "std"
            ),

            random_value_mean=(
                "random_policy_value", "mean"
            ),
            random_value_sd=(
                "random_policy_value", "std"
            ),

            model_minus_random_mean=(
                "model_minus_random", "mean"
            ),
            model_minus_random_sd=(
                "model_minus_random", "std"
            ),

            positive_advantage_splits=(
                "model_minus_random",
                lambda x: int((x > 0).sum()),
            ),

            # Descriptive averages of split-specific interval endpoints.
            # These are NOT pooled confidence intervals.
            mean_split_bootstrap_ci_low=(
                "model_minus_random_ci_low", "mean"
            ),
            mean_split_bootstrap_ci_high=(
                "model_minus_random_ci_high", "mean"
            ),

            treat_all_mean=(
                "treat_all_incremental", "mean"
            ),
            treat_all_sd=(
                "treat_all_incremental", "std"
            ),

            selected_positive_rate_mean=(
                "selected_positive_outcome_rate", "mean"
            ),
            selected_max_outcome_mean=(
                "selected_max_observed_outcome", "mean"
            ),
            selected_top_1pct_share_mean=(
                "selected_top_1pct_outcome_share", "mean"
            ),
        )
        .reset_index()
    )

    summary["positive_advantage_split_fraction"] = (
        summary["positive_advantage_splits"] / summary["seeds"]
    )

    summary_path = (
        OUT_DIR / "hillstrom_model_summary_corrected.csv"
    )
    summary.to_csv(summary_path, index=False)

    # --------------------------------------------------------
    # REPRODUCIBILITY METADATA
    # --------------------------------------------------------

    metadata = {
        "seeds": SEEDS,
        "number_of_splits": len(SEEDS),
        "test_size": TEST_SIZE,
        "top_fraction": TOP_FRACTION,
        "n_bootstrap": N_BOOTSTRAP,
        "treatment_propensity_used": TREATMENT_PROPENSITY,
        "features": FEATURES,
        "targets": TARGETS,
        "treatment_definition": (
            "Mens E-Mail=1; No E-Mail=0"
        ),
        "log1p_retransformation": (
            "Duan smearing estimated from out-of-fold "
            "training residuals"
        ),
        "policy_metric": (
            "Held-out inverse-probability-weighted incremental "
            "value versus treating nobody"
        ),
        "random_baseline": (
            "Expected value of a random policy selecting the same "
            "fraction of customers"
        ),
        "uncertainty_note": (
            "Bootstrap intervals are conditional on each test split "
            "and fitted model. Summary interval columns are averages "
            "of split-specific endpoints, not pooled confidence "
            "intervals. Splits may overlap and are not independent "
            "replications."
        ),
    }

    metadata_path = OUT_DIR / "run_metadata_corrected.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # --------------------------------------------------------
    # PRINT SUMMARY
    # --------------------------------------------------------

    print("\n=== POLICY VS EXPECTED RANDOM TARGETING ===")

    display_cols = [
        "target",
        "model",
        "transform",
        "policy_value_mean",
        "random_value_mean",
        "model_minus_random_mean",
        "model_minus_random_sd",
        "positive_advantage_splits",
        "positive_advantage_split_fraction",
        "mean_split_bootstrap_ci_low",
        "mean_split_bootstrap_ci_high",
        "selected_positive_rate_mean",
        "selected_top_1pct_share_mean",
    ]

    print(
        summary[display_cols]
        .round(5)
        .to_string(index=False)
    )

    print("\nSaved detailed results:", corrected_path)
    print("Saved customer predictions:", customer_path)
    print("Saved summary:", summary_path)
    print("Saved metadata:", metadata_path)


if __name__ == "__main__":
    main()
