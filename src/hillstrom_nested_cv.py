"""
Nested cross-validation for Hillstrom spend uplift.

Purpose:
- Estimate out-of-fold policy advantage for the preselected L2 + log1p model.
- Estimate performance of a model-selection procedure that chooses among six
  spend configurations using inner folds and evaluates that choice on outer folds.

Important:
- This is not a truly untouched independent holdout: the Hillstrom dataset has
  already been used in earlier model comparisons. Nested CV is a more defensible
  re-evaluation of the selection procedure, but it cannot erase prior data reuse.
- Uses the existing fit_arm/predict_arm implementation from hillstrom_validation.py.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from hillstrom_validation import (
    DATA_PATH,
    FEATURES,
    fit_arm,
    predict_arm,
    TREATMENT_PROPENSITY,
    TOP_FRACTION,
)

OUT_DIR = Path("results/hillstrom")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUTER_FOLDS = 5
INNER_FOLDS = 3
RANDOM_SEED = 620261026
TARGET = "spend"
REFERENCE = ("L2", "log1p")
CONFIGURATIONS = [
    ("L2", "identity"),
    ("L2", "log1p"),
    ("Tweedie", "identity"),
    ("Tweedie", "log1p"),
    ("Hurdle", "identity"),
    ("Hurdle", "log1p"),
]


def policy_advantage(y, treatment, uplift, fraction=TOP_FRACTION,
                     propensity=TREATMENT_PROPENSITY):
    y = np.asarray(y, dtype=float)
    w = np.asarray(treatment, dtype=int)
    score = np.asarray(uplift, dtype=float)

    n = len(y)
    if n == 0:
        raise ValueError("Empty evaluation fold.")
    if not (len(w) == len(score) == n):
        raise ValueError("y, treatment and uplift lengths differ.")
    if not np.isin(w, [0, 1]).all():
        raise ValueError("Treatment must be binary.")
    if not (np.any(w == 0) and np.any(w == 1)):
        raise ValueError("Evaluation fold must contain both treatment arms.")

    n_selected = max(1, int(np.ceil(n * fraction)))
    selected = np.zeros(n, dtype=float)
    selected[np.argsort(score, kind="mergesort")[-n_selected:]] = 1.0
    actual_fraction = selected.mean()

    contribution = (
        w * y / propensity
        - (1 - w) * y / (1 - propensity)
    )
    advantage = float(np.mean((selected - actual_fraction) * contribution))
    policy_value = float(np.mean(selected * contribution))
    random_value = float(actual_fraction * np.mean(contribution))
    return {
        "advantage": advantage,
        "policy_value": policy_value,
        "random_value": random_value,
        "selected_fraction": actual_fraction,
        "n": n,
        "n_selected": n_selected,
    }


def fit_predict_config(X_train, y_train, W_train, X_eval, model, transform, seed):
    fitted = {}
    for arm in (0, 1):
        mask = W_train == arm
        fitted[arm] = fit_arm(
            X_train.iloc[np.flatnonzero(mask)].reset_index(drop=True),
            y_train[mask],
            model,
            transform,
            seed + arm,
        )
    mu0 = predict_arm(fitted[0], X_eval)
    mu1 = predict_arm(fitted[1], X_eval)
    return mu1 - mu0


def main():
    if not Path(DATA_PATH).exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    needed = list(FEATURES) + [TARGET, "treatment"]
    df = df.dropna(subset=needed).copy().reset_index(drop=True)

    X = df[list(FEATURES)].copy()
    y = df[TARGET].astype(float).to_numpy()
    W = df["treatment"].astype(int).to_numpy()
    row_ids = np.arange(len(df))

    if not np.isin(W, [0, 1]).all():
        raise ValueError("Treatment must be binary.")
    if np.any(y < 0):
        raise ValueError("Spend outcome must be nonnegative.")

    outer = StratifiedKFold(
        n_splits=OUTER_FOLDS, shuffle=True, random_state=RANDOM_SEED
    )

    outer_rows = []
    inner_rows = []

    print("=" * 72)
    print("HILLSTROM NESTED CROSS-VALIDATION — SPEND")
    print("=" * 72)
    print(f"Rows: {len(df):,}; outer folds: {OUTER_FOLDS}; inner folds: {INNER_FOLDS}")
    print(f"Treatment propensity used: {TREATMENT_PROPENSITY}")
    print("Candidate configurations:", CONFIGURATIONS)
    print("\nNote: this is a re-evaluation on a dataset already used in earlier analysis,")
    print("not a truly untouched independent holdout.")

    for outer_fold, (outer_train, outer_test) in enumerate(
        outer.split(row_ids, W), start=1
    ):
        print(f"\n--- Outer fold {outer_fold}/{OUTER_FOLDS} ---")
        X_otrain = X.iloc[outer_train].reset_index(drop=True)
        y_otrain = y[outer_train]
        W_otrain = W[outer_train]
        X_otest = X.iloc[outer_test].reset_index(drop=True)
        y_otest = y[outer_test]
        W_otest = W[outer_test]

        inner = StratifiedKFold(
            n_splits=INNER_FOLDS,
            shuffle=True,
            random_state=RANDOM_SEED + outer_fold,
        )

        config_inner_scores = []
        for model, transform in CONFIGURATIONS:
            scores = []
            for inner_fold, (inner_train, inner_val) in enumerate(
                inner.split(np.arange(len(outer_train)), W_otrain), start=1
            ):
                uplift = fit_predict_config(
                    X_otrain.iloc[inner_train].reset_index(drop=True),
                    y_otrain[inner_train],
                    W_otrain[inner_train],
                    X_otrain.iloc[inner_val].reset_index(drop=True),
                    model,
                    transform,
                    RANDOM_SEED + outer_fold * 1000 + inner_fold * 10,
                )
                metric = policy_advantage(
                    y_otrain[inner_val], W_otrain[inner_val], uplift
                )
                scores.append(metric["advantage"])
            mean_inner = float(np.mean(scores))
            config_inner_scores.append((model, transform, mean_inner))
            inner_rows.append({
                "outer_fold": outer_fold,
                "model": model,
                "transform": transform,
                "inner_mean_advantage": mean_inner,
                "inner_fold_scores": ";".join(f"{x:.8f}" for x in scores),
            })
            print(f"  Inner selection {model:8s} {transform:8s}: {mean_inner:+.5f}")

        selected_model, selected_transform, selected_inner_score = max(
            config_inner_scores, key=lambda x: x[2]
        )
        print(f"  Selected by inner folds: {selected_model} + {selected_transform}")

        # Evaluate both the inner-selected configuration and the previously
        # preselected reference on the same outer fold.
        for evaluation_role, model, transform in [
            ("inner_selected", selected_model, selected_transform),
            ("fixed_L2_log1p", REFERENCE[0], REFERENCE[1]),
        ]:
            uplift = fit_predict_config(
                X_otrain, y_otrain, W_otrain, X_otest,
                model, transform,
                RANDOM_SEED + outer_fold * 10000 + (0 if evaluation_role == "inner_selected" else 5000),
            )
            metric = policy_advantage(y_otest, W_otest, uplift)
            outer_rows.append({
                "outer_fold": outer_fold,
                "evaluation_role": evaluation_role,
                "model": model,
                "transform": transform,
                "inner_selected_model": selected_model,
                "inner_selected_transform": selected_transform,
                "inner_selected_mean_advantage": selected_inner_score,
                **metric,
            })
            print(
                f"  Outer evaluation {evaluation_role:14s} {model}+{transform}: "
                f"advantage={metric['advantage']:+.5f}"
            )

    outer_df = pd.DataFrame(outer_rows)
    inner_df = pd.DataFrame(inner_rows)
    outer_path = OUT_DIR / "hillstrom_nested_cv_outer_results.csv"
    inner_path = OUT_DIR / "hillstrom_nested_cv_inner_selection.csv"
    summary_path = OUT_DIR / "hillstrom_nested_cv_summary.csv"
    outer_df.to_csv(outer_path, index=False)
    inner_df.to_csv(inner_path, index=False)

    summary = (
        outer_df.groupby(["evaluation_role", "model", "transform"], as_index=False)
        .agg(
            n_outer_folds=("outer_fold", "nunique"),
            mean_outer_advantage=("advantage", "mean"),
            sd_outer_advantage=("advantage", "std"),
            positive_outer_folds=("advantage", lambda s: int((s > 0).sum())),
            mean_policy_value=("policy_value", "mean"),
            mean_random_value=("random_value", "mean"),
        )
    )
    summary.to_csv(summary_path, index=False)

    selected_counts = (
        outer_df.loc[outer_df["evaluation_role"] == "inner_selected"]
        .groupby(["model", "transform"]).size().rename("outer_folds_selected")
        .reset_index()
    )

    print("\n" + "=" * 72)
    print("NESTED-CV SUMMARY")
    print("=" * 72)
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.5f}"))
    print("\nConfiguration selection counts across outer folds:")
    print(selected_counts.to_string(index=False))
    print("\nSaved:")
    print(" ", outer_path)
    print(" ", inner_path)
    print(" ", summary_path)
    print("\nInterpretation:")
    print("- fixed_L2_log1p estimates performance of the previously selected configuration")
    print("  under this nested-CV split scheme; it is not a pristine external holdout.")
    print("- inner_selected estimates the performance of the model-selection procedure.")
    print("- Five outer folds provide a small number of correlated fold estimates; do not")
    print("  treat their standard deviation as a definitive confidence interval.")
    print("- Stronger confirmation requires an independent external dataset or a prospectively")
    print("  untouched holdout collected before model selection.")


if __name__ == "__main__":
    main()
