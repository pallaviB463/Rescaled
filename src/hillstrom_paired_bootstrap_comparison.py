from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
INPUT_FILE = Path("results/hillstrom/hillstrom_test_predictions.csv")
OUTPUT_DIR = Path("results/hillstrom")

TOP_FRACTION = 0.20
TREATMENT_PROPENSITY = 0.50
BOOTSTRAP_REPS = 2000
RANDOM_SEED = 20261026

REFERENCE = ("L2", "log1p")
N_COMPARISONS = 5
BONFERRONI_ALPHA = 0.05 / N_COMPARISONS  # 99% CI per comparison


# ---------------------------------------------------------
# LOAD AND VALIDATE DATA
# ---------------------------------------------------------
def load_data():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Could not find prediction file: {INPUT_FILE.resolve()}"
        )

    df = pd.read_csv(INPUT_FILE)

    required = {
        "target", "model", "transform", "seed",
        "test_row_index", "treatment", "observed_outcome",
        "pred_uplift",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df.loc[df["target"] == "spend"].copy()

    for col in ["treatment", "observed_outcome", "pred_uplift"]:
        df[col] = pd.to_numeric(df[col], errors="raise")

    if df.empty:
        raise ValueError("No spend rows were found.")
    if df[["treatment", "observed_outcome", "pred_uplift"]].isna().any().any():
        raise ValueError("Missing values found in required numeric columns.")
    if not df["treatment"].isin([0, 1]).all():
        raise ValueError("Treatment must contain only 0 and 1.")
    if (df["observed_outcome"] < 0).any():
        raise ValueError("Spend outcomes must be non-negative.")

    consistency = df.groupby("test_row_index").agg(
        treatment_n=("treatment", "nunique"),
        outcome_n=("observed_outcome", "nunique"),
    )
    if (consistency["treatment_n"] > 1).any() or (consistency["outcome_n"] > 1).any():
        raise ValueError(
            "A customer's treatment or observed spend differs across rows."
        )

    print(f"Loaded spend prediction rows: {len(df):,}")
    print(f"Unique customers: {df['test_row_index'].nunique():,}")
    print(f"Bootstrap repetitions: {BOOTSTRAP_REPS}")
    print("Models:", sorted(df["model"].unique().tolist()))
    print("Transforms:", sorted(df["transform"].unique().tolist()))
    print("Seeds:", sorted(df["seed"].unique().tolist()))
    return df


# ---------------------------------------------------------
# PRECOMPUTE SPLIT ARRAYS
# ---------------------------------------------------------
def prepare_splits(df):
    """
    Fix each split's ranking and top-20% membership from its fitted
    predictions. The bootstrap resamples customer IDs, keeping a
    customer's multiplicity consistent across models and seeds.
    """
    prepared = {}

    for (model, transform, seed), group in df.groupby(
        ["model", "transform", "seed"], sort=True
    ):
        group = group.reset_index(drop=True)
        n = len(group)
        n_selected = max(1, int(np.ceil(TOP_FRACTION * n)))

        uplift = group["pred_uplift"].to_numpy(dtype=float)
        y = group["observed_outcome"].to_numpy(dtype=float)
        treatment = group["treatment"].to_numpy(dtype=float)
        customer_ids = group["test_row_index"].to_numpy()

        selected = np.zeros(n, dtype=float)
        selected[np.argsort(uplift)[-n_selected:]] = 1.0

        p = TREATMENT_PROPENSITY
        contribution = treatment * y / p - (1.0 - treatment) * y / (1.0 - p)

        prepared[(model, transform, int(seed))] = {
            "customer_ids": customer_ids,
            "selected": selected,
            "contribution": contribution,
            "n": n,
            "observed_advantage": float(
                np.mean((selected - selected.mean()) * contribution)
            ),
        }

    return prepared


def weighted_policy_advantage(selected, contribution, weights):
    """
    Weighted policy advantage. Crucially, re-centers selection using
    its bootstrap-weighted fraction instead of the original 20% fraction.
    """
    total_weight = weights.sum()
    if total_weight <= 0:
        return np.nan

    selected_fraction = np.sum(weights * selected) / total_weight
    return float(
        np.sum(weights * (selected - selected_fraction) * contribution)
        / total_weight
    )


# ---------------------------------------------------------
# CUSTOMER-CLUSTER BOOTSTRAP
# ---------------------------------------------------------
def run_paired_cluster_bootstrap(df, prepared):
    """
    Resample unique customer IDs and apply the same multiplicities to all
    model/transform/seed records. The model rankings remain fixed.

    CIs are conditional on existing fitted predictions. They do not
    include model-training uncertainty or split-generation uncertainty.
    """
    customer_ids = df["test_row_index"].drop_duplicates().to_numpy()
    n_customers = len(customer_ids)
    customer_position = {
        customer_id: i for i, customer_id in enumerate(customer_ids)
    }

    splits = {}
    for key, item in prepared.items():
        positions = np.fromiter(
            (customer_position[cid] for cid in item["customer_ids"]),
            dtype=np.int32,
            count=len(item["customer_ids"]),
        )
        splits[key] = {
            **item,
            "customer_positions": positions,
        }

    split_groups = {}
    point_split_values = {}

    for (model, transform, seed), item in splits.items():
        model_key = (model, transform)
        split_groups.setdefault(model_key, []).append(item)
        point_split_values.setdefault(model_key, []).append(
            item["observed_advantage"]
        )

    point_means = {
        key: float(np.mean(values))
        for key, values in point_split_values.items()
    }

    if REFERENCE not in point_means:
        raise ValueError(
            f"Reference {REFERENCE} not found. Available: {sorted(point_means)}"
        )

    model_keys = sorted(point_means)
    bootstrap_values = {
        key: np.empty(BOOTSTRAP_REPS, dtype=float)
        for key in model_keys
    }
    difference_values = {
        key: np.empty(BOOTSTRAP_REPS, dtype=float)
        for key in model_keys if key != REFERENCE
    }

    rng = np.random.default_rng(RANDOM_SEED)

    print("\n=== PAIRED CUSTOMER-CLUSTER BOOTSTRAP ===")
    print(f"Resampling {n_customers:,} customer IDs...")
    print(
        "Same bootstrap multiplicities are used across models and splits; "
        "selection is re-centered within each bootstrap split."
    )

    for b in range(BOOTSTRAP_REPS):
        if (b + 1) % 100 == 0:
            print(f"Bootstrap repetition {b + 1}/{BOOTSTRAP_REPS}")

        sampled_positions = rng.integers(
            0, n_customers, size=n_customers
        )
        customer_weights = np.bincount(
            sampled_positions, minlength=n_customers
        ).astype(float)

        boot_means = {}

        for model_key, items in split_groups.items():
            split_advantages = []

            for item in items:
                weights = customer_weights[item["customer_positions"]]
                estimate = weighted_policy_advantage(
                    item["selected"],
                    item["contribution"],
                    weights,
                )
                if np.isfinite(estimate):
                    split_advantages.append(estimate)

            boot_means[model_key] = float(np.mean(split_advantages))
            bootstrap_values[model_key][b] = boot_means[model_key]

        ref_value = boot_means[REFERENCE]
        for model_key in difference_values:
            difference_values[model_key][b] = (
                ref_value - boot_means[model_key]
            )

    # Individual model advantage intervals: descriptive 95% CIs.
    model_rows = []
    for key in model_keys:
        values = bootstrap_values[key]
        low95, high95 = np.percentile(values, [2.5, 97.5])
        model, transform = key
        split_values = point_split_values[key]

        model_rows.append({
            "model": model,
            "transform": transform,
            "n_splits": len(split_values),
            "observed_mean_advantage": point_means[key],
            "split_sd": float(np.std(split_values, ddof=1)),
            "positive_splits": int(np.sum(np.asarray(split_values) > 0)),
            "bootstrap_ci95_low": float(low95),
            "bootstrap_ci95_high": float(high95),
            "bootstrap_reps": BOOTSTRAP_REPS,
        })

    model_summary = pd.DataFrame(model_rows)

    # Paired comparisons: report both 95% and Bonferroni 99% CIs.
    comparison_rows = []
    for key in model_keys:
        if key == REFERENCE:
            continue

        values = difference_values[key]
        low95, high95 = np.percentile(values, [2.5, 97.5])
        low_adj, high_adj = np.percentile(
            values,
            [100 * BONFERRONI_ALPHA / 2,
             100 * (1 - BONFERRONI_ALPHA / 2)],
        )

        model, transform = key
        observed_difference = point_means[REFERENCE] - point_means[key]

        comparison_rows.append({
            "reference_model": REFERENCE[0],
            "reference_transform": REFERENCE[1],
            "comparison_model": model,
            "comparison_transform": transform,
            "reference_mean_advantage": point_means[REFERENCE],
            "comparison_mean_advantage": point_means[key],
            "observed_difference_reference_minus_comparison": observed_difference,
            "paired_ci95_low": float(low95),
            "paired_ci95_high": float(high95),
            "bonferroni_ci99_low": float(low_adj),
            "bonferroni_ci99_high": float(high_adj),
            "bonferroni_ci_excludes_zero": bool(
                low_adj > 0 or high_adj < 0
            ),
            "bootstrap_reps": BOOTSTRAP_REPS,
        })

    return model_summary, pd.DataFrame(comparison_rows)


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data()

    print("\n=== PREPARING SPLITS ===")
    prepared = prepare_splits(df)
    print(f"Prepared model/split groups: {len(prepared):,}")

    model_summary, comparisons = run_paired_cluster_bootstrap(
        df, prepared
    )

    model_path = OUTPUT_DIR / "hillstrom_paired_bootstrap_model_summary_corrected.csv"
    comparison_path = OUTPUT_DIR / "hillstrom_paired_bootstrap_comparisons_corrected.csv"

    model_summary.to_csv(model_path, index=False)
    comparisons.to_csv(comparison_path, index=False)

    print("\n=== MODEL ADVANTAGE SUMMARY ===")
    print(
        model_summary.sort_values(
            "observed_mean_advantage", ascending=False
        ).to_string(index=False, float_format=lambda x: f"{x:.5f}")
    )

    print("\n=== PAIRED COMPARISONS: L2 + log1p MINUS EACH ALTERNATIVE ===")
    print(
        comparisons.to_string(
            index=False, float_format=lambda x: f"{x:.5f}"
        )
    )

    print(f"\nSaved model summary: {model_path}")
    print(f"Saved paired comparisons: {comparison_path}")
    print(
        "\nInterpretation: paired 95% intervals are descriptive. The 99% "
        "Bonferroni intervals adjust for the five comparisons. An adjusted "
        "interval entirely above zero supports a higher estimated advantage "
        "for L2 + log1p than that alternative, conditional on the fitted "
        "predictions. These intervals do not include model-training or "
        "split-generation uncertainty."
    )


if __name__ == "__main__":
    main()
