from pathlib import Path
import numpy as np
import pandas as pd

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
INPUT_FILE = Path("results/hillstrom/hillstrom_test_predictions.csv")
OUTPUT_DIR = Path("results/hillstrom")
CAPS = [100.0, 200.0, 300.0]
TOP_FRACTION = 0.20
TREATMENT_PROPENSITY = 0.50


def load_data():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Could not find input file: {INPUT_FILE.resolve()}"
        )

    df = pd.read_csv(INPUT_FILE)
    required = {
        "target", "model", "transform", "seed", "test_row_index",
        "treatment", "observed_outcome", "pred_uplift",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df.loc[df["target"] == "spend"].copy()
    for column in ["observed_outcome", "pred_uplift", "treatment"]:
        df[column] = pd.to_numeric(df[column], errors="raise")

    if df.empty:
        raise ValueError("No spend prediction rows were found.")
    if df[["observed_outcome", "pred_uplift", "treatment"]].isna().any().any():
        raise ValueError("Missing values in outcome, uplift, or treatment.")
    if not df["treatment"].isin([0, 1]).all():
        raise ValueError("Treatment must contain only 0 and 1.")
    if (df["observed_outcome"] < 0).any():
        raise ValueError("Spend outcomes must be non-negative.")

    consistency = df.groupby("test_row_index").agg(
        treatment_count=("treatment", "nunique"),
        outcome_count=("observed_outcome", "nunique"),
    )
    if (
        (consistency["treatment_count"] > 1).any()
        or (consistency["outcome_count"] > 1).any()
    ):
        raise ValueError(
            "Treatment or observed spend is inconsistent across repeated "
            "records for a customer."
        )

    print(f"Loaded spend prediction rows: {len(df):,}")
    print(f"Unique customers: {df['test_row_index'].nunique():,}")
    print("Seeds:", sorted(df["seed"].unique().tolist()))
    return df


def calculate_advantage(group, outcome_values):
    """Estimate top-20%-targeting advantage over random targeting."""
    n = len(group)
    if n == 0:
        return np.nan

    n_selected = max(1, int(np.ceil(TOP_FRACTION * n)))
    uplift = group["pred_uplift"].to_numpy(dtype=float)
    treatment = group["treatment"].to_numpy(dtype=float)
    y = np.asarray(outcome_values, dtype=float)

    selected = np.zeros(n, dtype=float)
    selected[np.argsort(uplift)[-n_selected:]] = 1.0

    p = TREATMENT_PROPENSITY
    contribution = treatment * y / p - (1.0 - treatment) * y / (1.0 - p)
    selected_fraction = selected.mean()
    advantage = np.mean((selected - selected_fraction) * contribution)
    return float(advantage)


def calculate_split_results(df):
    rows = []
    group_columns = ["model", "transform", "seed"]

    for (model, transform, seed), group in df.groupby(group_columns, sort=True):
        group = group.reset_index(drop=True)
        original = group["observed_outcome"].to_numpy(dtype=float)
        original_advantage = calculate_advantage(group, original)

        base = {
            "model": model,
            "transform": transform,
            "seed": int(seed),
            "n_test": len(group),
            "original_advantage": original_advantage,
            "original_mean_spend": float(original.mean()),
            "positive_spend_customers": int(np.sum(original > 0)),
        }

        rows.append({
            **base,
            "cap": "original",
            "cap_value": np.nan,
            "advantage": original_advantage,
            "mean_evaluated_spend": float(original.mean()),
            "fraction_outcomes_capped": 0.0,
        })

        for cap in CAPS:
            capped = np.minimum(original, cap)
            advantage = calculate_advantage(group, capped)
            rows.append({
                **base,
                "cap": f"cap_{int(cap)}",
                "cap_value": cap,
                "advantage": advantage,
                "mean_evaluated_spend": float(capped.mean()),
                "fraction_outcomes_capped": float(np.mean(original > cap)),
            })

    return pd.DataFrame(rows)


def summarize_results(split_results):
    """
    Summarize estimates across existing splits. This fast version
    deliberately omits bootstrap confidence intervals. Split SD is
    descriptive variability, not a confidence interval.
    """
    summary_rows = []
    for (model, transform, cap_name), group in split_results.groupby(
        ["model", "transform", "cap"], sort=True
    ):
        advantages = group["advantage"].dropna()
        original_advantages = group["original_advantage"].dropna()
        cap_value = (
            np.nan if cap_name == "original"
            else float(cap_name.replace("cap_", ""))
        )

        summary_rows.append({
            "target": "spend",
            "model": model,
            "transform": transform,
            "cap": cap_name,
            "cap_value": cap_value,
            "n_splits": len(advantages),
            "advantage_mean": float(advantages.mean()),
            "advantage_sd": (
                float(advantages.std(ddof=1)) if len(advantages) > 1 else np.nan
            ),
            "positive_splits": int((advantages > 0).sum()),
            "positive_split_fraction": float((advantages > 0).mean()),
            "mean_evaluated_spend": float(
                group["mean_evaluated_spend"].mean()
            ),
            "mean_fraction_outcomes_capped": float(
                group["fraction_outcomes_capped"].mean()
            ),
            "original_advantage_mean": float(original_advantages.mean()),
        })

    return pd.DataFrame(summary_rows)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data()

    print("\n=== SPEND DISTRIBUTION ===")
    unique_customers = df[
        ["test_row_index", "observed_outcome"]
    ].drop_duplicates(subset=["test_row_index"])
    positive_count = int((unique_customers["observed_outcome"] > 0).sum())
    max_spend = float(unique_customers["observed_outcome"].max())
    print(f"Unique customers with positive spend: {positive_count:,}")
    print(f"Maximum observed spend: {max_spend:.2f}")
    print(f"Caps being evaluated: {CAPS}")

    print("\n=== CALCULATING CAP SENSITIVITY ===")
    split_results = calculate_split_results(df)
    split_path = OUTPUT_DIR / "hillstrom_cap_sensitivity_by_split.csv"
    split_results.to_csv(split_path, index=False)

    summary = summarize_results(split_results)
    summary_path = OUTPUT_DIR / "hillstrom_cap_sensitivity_summary.csv"
    summary.to_csv(summary_path, index=False)

    display_columns = [
        "model", "transform", "cap", "n_splits",
        "advantage_mean", "advantage_sd", "positive_splits",
        "mean_fraction_outcomes_capped", "original_advantage_mean",
    ]

    print("\n=== CAP SENSITIVITY SUMMARY ===")
    print(
        summary[display_columns]
        .sort_values(["model", "transform", "cap"])
        .to_string(index=False, float_format=lambda x: f"{x:.5f}")
    )
    print(f"\nSaved split-level results: {split_path}")
    print(f"Saved summary: {summary_path}")
    print(
        "\nInterpretation: compare each capped advantage with its original-spend "
        "advantage. These are point estimates across existing splits; this fast "
        "version does not calculate bootstrap confidence intervals. Capping "
        "changes the evaluated outcome, so use original spend as the primary "
        "business metric."
    )


if __name__ == "__main__":
    main()
