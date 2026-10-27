from pathlib import Path
import numpy as np
import pandas as pd

PREDICTIONS_FILE = Path("results/hillstrom/hillstrom_test_predictions.csv")
DATA_FILE = Path("data/hillstrom/hillstrom_mens_control.csv")
OUTPUT_DIR = Path("results/hillstrom")
TOP_FRACTION = 0.20
P_DEFAULT = 0.50


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not PREDICTIONS_FILE.exists():
        raise FileNotFoundError(f"Missing file: {PREDICTIONS_FILE.resolve()}")

    pred = pd.read_csv(PREDICTIONS_FILE)
    required = {
        "target", "model", "transform", "seed", "test_row_index",
        "treatment", "observed_outcome", "pred_uplift",
    }
    missing = required - set(pred.columns)
    if missing:
        raise ValueError(f"Prediction file missing columns: {sorted(missing)}")

    spend = pred.loc[pred["target"] == "spend"].copy()
    for col in ["treatment", "observed_outcome", "pred_uplift"]:
        spend[col] = pd.to_numeric(spend[col], errors="raise")

    if spend[["treatment", "observed_outcome", "pred_uplift"]].isna().any().any():
        raise ValueError("Missing values in required prediction columns.")
    if not spend["treatment"].isin([0, 1]).all():
        raise ValueError("Treatment must contain only 0 and 1.")

    customer_check = spend.groupby("test_row_index").agg(
        treatment_n=("treatment", "nunique"),
        outcome_n=("observed_outcome", "nunique"),
    )
    if (customer_check["treatment_n"] > 1).any() or (customer_check["outcome_n"] > 1).any():
        raise ValueError("Customer treatment or observed spend varies across prediction rows.")

    pred_customers = (
        spend[["test_row_index", "treatment", "observed_outcome"]]
        .drop_duplicates(subset=["test_row_index"])
        .copy()
    )

    # Use the full filtered dataset for the empirical arm fraction if available.
    # The prediction file only contains customers appearing in at least one test split.
    source = "prediction-file customer union"
    audit_customers = pred_customers.copy()
    if DATA_FILE.exists():
        full = pd.read_csv(DATA_FILE)
        if {"treatment", "spend"}.issubset(full.columns):
            full = full[["treatment", "spend"]].copy()
            full["treatment"] = pd.to_numeric(full["treatment"], errors="raise")
            full["spend"] = pd.to_numeric(full["spend"], errors="raise")
            full = full.dropna(subset=["treatment", "spend"])
            if full["treatment"].isin([0, 1]).all():
                audit_customers = full.rename(columns={"spend": "observed_outcome"})
                source = str(DATA_FILE)

    n = len(audit_customers)
    treated_n = int((audit_customers["treatment"] == 1).sum())
    control_n = int((audit_customers["treatment"] == 0).sum())
    p_empirical = treated_n / n

    treated_y = audit_customers.loc[
        audit_customers["treatment"] == 1, "observed_outcome"
    ].to_numpy(dtype=float)
    control_y = audit_customers.loc[
        audit_customers["treatment"] == 0, "observed_outcome"
    ].to_numpy(dtype=float)

    print("=== RANDOMIZATION / PROPENSITY CHECK ===")
    print(f"Source: {source}")
    print(f"Rows/customers: {n:,}")
    print(f"Men's Email treated: {treated_n:,}")
    print(f"No Email control: {control_n:,}")
    print(f"Observed treated fraction: {p_empirical:.8f}")
    print(f"Difference from 0.5: {p_empirical - 0.5:+.8f}")
    print(
        "The empirical fraction is a diagnostic. For the two-arm subset, "
        "the randomized design supports a treatment probability close to 0.5."
    )

    print("\n=== OBSERVED SPEND CHECK ===")
    print(f"Treatment mean spend: ${treated_y.mean():.6f}")
    print(f"Control mean spend:   ${control_y.mean():.6f}")
    print(f"Difference in means:  ${treated_y.mean() - control_y.mean():.6f}")
    print(f"Treatment positive-spend rate: {(treated_y > 0).mean():.6%}")
    print(f"Control positive-spend rate:   {(control_y > 0).mean():.6%}")
    print(f"Overall zero-spend fraction: {(audit_customers['observed_outcome'] == 0).mean():.6%}")

    # Compare the two algebraically equivalent definitions of policy advantage.
    rows = []
    for (model, transform, seed), group in spend.groupby(
        ["model", "transform", "seed"], sort=True
    ):
        group = group.reset_index(drop=True)
        n_test = len(group)
        k = max(1, int(np.ceil(TOP_FRACTION * n_test)))
        uplift = group["pred_uplift"].to_numpy(dtype=float)
        y = group["observed_outcome"].to_numpy(dtype=float)
        treatment = group["treatment"].to_numpy(dtype=float)

        selected = np.zeros(n_test, dtype=float)
        selected[np.argsort(uplift)[-k:]] = 1.0
        selected_fraction = selected.mean()

        for p in [P_DEFAULT, p_empirical]:
            ipw = treatment * y / p - (1.0 - treatment) * y / (1.0 - p)

            centered = np.mean((selected - selected_fraction) * ipw)
            policy_value = np.mean(selected * ipw)
            random_value = selected_fraction * np.mean(ipw)
            difference = policy_value - random_value

            rows.append({
                "model": model,
                "transform": transform,
                "seed": int(seed),
                "propensity_used": p,
                "n_test": n_test,
                "selected_fraction": selected_fraction,
                "policy_value_ipw": policy_value,
                "random_value_ipw": random_value,
                "advantage_centered": centered,
                "advantage_policy_minus_random": difference,
                "formula_abs_difference": abs(centered - difference),
            })

    by_split = pd.DataFrame(rows)
    max_formula_error = by_split["formula_abs_difference"].max()

    print("\n=== POLICY-VALUE FORMULA CHECK ===")
    print(f"Max difference between equivalent formulas: {max_formula_error:.12g}")
    print("Expected: approximately zero (floating-point precision).")

    summary = (
        by_split.groupby(["model", "transform", "propensity_used"])
        .agg(
            n_splits=("seed", "nunique"),
            mean_policy_value=("policy_value_ipw", "mean"),
            mean_random_value=("random_value_ipw", "mean"),
            mean_advantage=("advantage_centered", "mean"),
            sd_advantage_across_splits=("advantage_centered", "std"),
            positive_splits=("advantage_centered", lambda x: int((x > 0).sum())),
            max_formula_abs_difference=("formula_abs_difference", "max"),
        )
        .reset_index()
    )

    print("\n=== POLICY RESULTS: P=0.5 VS EMPIRICAL ARM FRACTION ===")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    # Summarize test split overlap. Each customer's repeated appearances
    # must be treated as dependent when quantifying uncertainty.
    overlap_rows = []
    for (model, transform), group in spend.groupby(["model", "transform"], sort=True):
        appearances = group.groupby("test_row_index")["seed"].nunique()
        overlap_rows.append({
            "model": model,
            "transform": transform,
            "test_records": len(group),
            "unique_customers_seen": group["test_row_index"].nunique(),
            "mean_test_splits_per_seen_customer": float(appearances.mean()),
            "max_test_splits_per_customer": int(appearances.max()),
            "customers_seen_in_10plus_splits": int((appearances >= 10).sum()),
        })

    overlap = pd.DataFrame(overlap_rows)
    print("\n=== REPEATED TEST-SPLIT OVERLAP ===")
    print(overlap.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    by_split_path = OUTPUT_DIR / "hillstrom_evaluation_audit_by_split.csv"
    summary_path = OUTPUT_DIR / "hillstrom_evaluation_audit_summary.csv"
    overlap_path = OUTPUT_DIR / "hillstrom_test_split_overlap.csv"

    by_split.to_csv(by_split_path, index=False)
    summary.to_csv(summary_path, index=False)
    overlap.to_csv(overlap_path, index=False)

    print("\nSaved files:")
    print(f"  {by_split_path}")
    print(f"  {summary_path}")
    print(f"  {overlap_path}")
    print(
        "\nLimits: this audit checks propensity diagnostics, formula arithmetic, "
        "and overlap. It does not prove bootstrap confidence-interval coverage "
        "or account for model-training uncertainty."
    )


if __name__ == "__main__":
    main()
