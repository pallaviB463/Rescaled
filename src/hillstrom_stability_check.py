
from pathlib import Path

import numpy as np
import pandas as pd

# ============================================================
# CONFIGURATION
# ============================================================

RESULTS_PATH = Path(
    "results/hillstrom/hillstrom_test_predictions.csv"
)
OUT_DIR = Path("results/hillstrom")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TREATMENT_PROPENSITY = 0.5
TOP_FRACTION = 0.20
N_BOOTSTRAP = 1000
BOOTSTRAP_SEED = 20261026

# ============================================================
# LOAD AND VALIDATE PREDICTIONS
# ============================================================

def load_predictions():
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(
            f"Cannot find {RESULTS_PATH}. "
            "Run the Hillstrom experiment first."
        )

    df = pd.read_csv(RESULTS_PATH)

    required = {
        "seed",
        "test_row_index",
        "target",
        "model",
        "transform",
        "treatment",
        "observed_outcome",
        "pred_uplift",
    }

    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Prediction file is missing columns: {sorted(missing)}"
        )

    df = df.copy()
    df["test_row_index"] = df["test_row_index"].astype(int)
    df["seed"] = df["seed"].astype(int)

    numeric_cols = [
        "treatment",
        "observed_outcome",
        "pred_uplift",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="raise")

    if df[numeric_cols].isna().any().any():
        raise ValueError("Missing values found in required numeric columns.")

    if not df["treatment"].isin([0, 1]).all():
        raise ValueError("Treatment must contain only 0 and 1.")

    if not np.isfinite(df["observed_outcome"]).all():
        raise ValueError("Observed outcomes contain non-finite values.")

    if not np.isfinite(df["pred_uplift"]).all():
        raise ValueError("Predicted uplift contains non-finite values.")

    if (df["observed_outcome"] < 0).any():
        raise ValueError("Expected nonnegative outcomes.")

    return df


# ============================================================
# PREPARE POLICY CONTRIBUTIONS
# ============================================================

def prepare_policy_rows(df):
    """
    Reconstruct each split's top-20% policy and its difference
    from expected random targeting.

    contribution_i = Y_i / 0.5 for treatment,
                     -Y_i / 0.5 for control.

    difference_i = (selected_i - selected_fraction) * contribution_i.

    The split-level mean of difference_i estimates the policy's
    incremental value relative to expected random targeting.
    """
    frames = []

    group_cols = [
        "target",
        "model",
        "transform",
        "seed",
    ]

    for keys, group in df.groupby(group_cols, sort=False):
        group = group.copy()
        n = len(group)

        n_selected = max(1, int(np.ceil(n * TOP_FRACTION)))
        selected_fraction = n_selected / n

        order = np.argsort(
            group["pred_uplift"].to_numpy(),
            kind="mergesort",
        )

        selected = np.zeros(n, dtype=bool)
        selected[order[-n_selected:]] = True

        y = group["observed_outcome"].to_numpy(dtype=float)
        w = group["treatment"].to_numpy(dtype=int)

        contribution = (
            w * y / TREATMENT_PROPENSITY
            - (1 - w) * y / (1 - TREATMENT_PROPENSITY)
        )

        group["selected"] = selected.astype(int)
        group["ipw_contribution"] = contribution
        group["policy_contribution"] = selected * contribution
        group["difference_contribution"] = (
            selected.astype(float) - selected_fraction
        ) * contribution

        group["selected_fraction_actual"] = selected_fraction

        frames.append(group)

    if not frames:
        raise ValueError("No prediction groups were found.")

    return pd.concat(frames, ignore_index=True)


# ============================================================
# CLUSTER BOOTSTRAP ACROSS UNIQUE CUSTOMERS
# ============================================================

def cluster_bootstrap_summary(policy_rows):
    """
    Bootstrap original customer IDs, not individual seed-row records.

    A customer can appear in multiple test splits. Resampling their
    ID as a cluster preserves that dependence across splits.

    The resulting interval describes the average split-level
    advantage for these fitted models and this dataset. It does not
    include model retraining uncertainty or create a new experiment.
    """
    rng = np.random.default_rng(BOOTSTRAP_SEED)

    customer_ids = np.sort(
        policy_rows["test_row_index"].unique()
    )
    n_customers = len(customer_ids)

    if n_customers < 2:
        raise ValueError("Too few unique customers for bootstrap.")

    # Integer customer IDs are mapped to positions for fast lookup.
    id_to_pos = {
        customer_id: pos
        for pos, customer_id in enumerate(customer_ids)
    }

    rows = policy_rows.copy()
    rows["_customer_pos"] = (
        rows["test_row_index"].map(id_to_pos).astype(int)
    )

    group_cols = ["target", "model", "transform", "seed"]

    groups = []
    for keys, group in rows.groupby(group_cols, sort=False):
        groups.append({
            "keys": keys,
            "customer_pos": group["_customer_pos"].to_numpy(),
            "difference": group[
                "difference_contribution"
            ].to_numpy(dtype=float),
            "policy": group[
                "policy_contribution"
            ].to_numpy(dtype=float),
            "contribution": group[
                "ipw_contribution"
            ].to_numpy(dtype=float),
        })

    # Original split-level estimates.
    original_rows = []

    for group in groups:
        original_rows.append({
            "target": group["keys"][0],
            "model": group["keys"][1],
            "transform": group["keys"][2],
            "seed": group["keys"][3],
            "model_minus_random": float(
                group["difference"].mean()
            ),
            "policy_value": float(group["policy"].mean()),
            "treat_all_value": float(
                group["contribution"].mean()
            ),
        })

    split_results = pd.DataFrame(original_rows)

    # Bootstrap average advantage across splits for each configuration.
    bootstrap_values = {}

    for config, config_rows in split_results.groupby(
        ["target", "model", "transform"],
        sort=False,
    ):
        bootstrap_values[config] = np.empty(N_BOOTSTRAP)

    for b in range(N_BOOTSTRAP):
        # Resample unique customer IDs with replacement.
        sampled_positions = rng.integers(
            0,
            n_customers,
            size=n_customers,
        )
        multiplicities = np.bincount(
            sampled_positions,
            minlength=n_customers,
        )

        config_seed_values = {}

        for group in groups:
            config = group["keys"][:3]
            positions = group["customer_pos"]
            weights = multiplicities[positions]

            weight_sum = weights.sum()
            if weight_sum == 0:
                continue

            weighted_difference = np.average(
                group["difference"],
                weights=weights,
            )

            config_seed_values.setdefault(config, []).append(
                weighted_difference
            )

        for config, values in config_seed_values.items():
            if values:
                bootstrap_values[config][b] = np.mean(values)
            else:
                bootstrap_values[config][b] = np.nan

    summary_rows = []

    for config, group in split_results.groupby(
        ["target", "model", "transform"],
        sort=False,
    ):
        boot = bootstrap_values[config]
        boot = boot[np.isfinite(boot)]

        if len(boot) == 0:
            ci_low = np.nan
            ci_high = np.nan
        else:
            ci_low, ci_high = np.quantile(
                boot, [0.025, 0.975]
            )

        advantages = group["model_minus_random"].to_numpy()

        summary_rows.append({
            "target": config[0],
            "model": config[1],
            "transform": config[2],
            "n_splits": int(group["seed"].nunique()),
            "mean_policy_value": float(
                group["policy_value"].mean()
            ),
            "mean_treat_all_value": float(
                group["treat_all_value"].mean()
            ),
            "mean_advantage_vs_random": float(advantages.mean()),
            "sd_advantage_across_splits": float(
                advantages.std(ddof=1)
            ) if len(advantages) > 1 else np.nan,
            "positive_advantage_splits": int(
                (advantages > 0).sum()
            ),
            "positive_advantage_fraction": float(
                (advantages > 0).mean()
            ),
            "customer_cluster_bootstrap_ci_low": float(ci_low),
            "customer_cluster_bootstrap_ci_high": float(ci_high),
            "bootstrap_replicates_valid": int(len(boot)),
        })

    summary = pd.DataFrame(summary_rows)

    summary_path = (
        OUT_DIR / "hillstrom_customer_cluster_bootstrap.csv"
    )
    summary.to_csv(summary_path, index=False)

    split_path = OUT_DIR / "hillstrom_split_level_diagnostics.csv"
    split_results.to_csv(split_path, index=False)

    print("\n=== CUSTOMER-CLUSTER BOOTSTRAP ===")
    print(
        summary.round(5).to_string(index=False)
    )
    print("\nSaved:", summary_path)
    print("Saved:", split_path)

    return summary, split_results


# ============================================================
# CUSTOMER SELECTION RECURRENCE
# ============================================================

def selection_recurrence(policy_rows):
    """
    For the L2 + log1p spend policy, count how often a customer
    appears in a test split and how often they are selected.

    This checks whether the policy repeatedly selects the same
    customers, but recurrence alone does not establish influence
    on the policy-value estimate.
    """
    subset = policy_rows[
        (policy_rows["target"] == "spend")
        & (policy_rows["model"] == "L2")
        & (policy_rows["transform"] == "log1p")
    ].copy()

    if subset.empty:
        print(
            "\nNo L2 + log1p spend rows found; "
            "skipping recurrence analysis."
        )
        return

    recurrence = (
        subset.groupby("test_row_index")
        .agg(
            test_split_count=("seed", "nunique"),
            selected_split_count=("selected", "sum"),
            mean_observed_spend=("observed_outcome", "mean"),
            max_observed_spend=("observed_outcome", "max"),
            treatment_rate=("treatment", "mean"),
        )
        .reset_index()
    )

    recurrence["selection_rate_when_tested"] = (
        recurrence["selected_split_count"]
        / recurrence["test_split_count"]
    )

    recurrence = recurrence.sort_values(
        [
            "selected_split_count",
            "max_observed_spend",
        ],
        ascending=[False, False],
    )

    path = OUT_DIR / "hillstrom_customer_selection_recurrence.csv"
    recurrence.to_csv(path, index=False)

    print("\n=== L2 + LOG1P SPEND: REPEATEDLY SELECTED CUSTOMERS ===")
    print("Unique customers evaluated:", len(recurrence))
    print(
        "Customers selected in at least 5 test splits:",
        int((recurrence["selected_split_count"] >= 5).sum()),
    )
    print(
        "Customers selected in at least 10 test splits:",
        int((recurrence["selected_split_count"] >= 10).sum()),
    )
    print("\nTop 20 recurring selected customers:")
    print(
        recurrence.head(20).round(4).to_string(index=False)
    )
    print("\nSaved:", path)


# ============================================================
# TOP-SPEND INFLUENCE SENSITIVITY CHECK
# ============================================================


def top_spend_sensitivity(policy_rows):
    """
    Diagnostic only: compare the original split-level advantage
    with an estimate excluding exactly the highest 1% of observed
    spend rows in each split.

    The trimmed analysis changes the evaluation population and
    is not the primary causal policy-value estimate.
    """
    subset = policy_rows[
        policy_rows["target"] == "spend"
    ].copy()

    output = []
    group_cols = ["target", "model", "transform", "seed"]

    for keys, group in subset.groupby(group_cols, sort=False):
        group = group.copy()

        y = group["observed_outcome"].to_numpy(dtype=float)
        diff = group["difference_contribution"].to_numpy(dtype=float)

        n = len(group)
        n_exclude = max(1, int(np.ceil(0.01 * n)))

        # Rank by observed outcome. Break ties deterministically so
        # exactly n_exclude rows are excluded.
        order = np.argsort(y, kind="mergesort")
        keep = np.ones(n, dtype=bool)
        keep[order[-n_exclude:]] = False

        output.append({
            "target": keys[0],
            "model": keys[1],
            "transform": keys[2],
            "seed": keys[3],
            "original_advantage": float(diff.mean()),
            "trimmed_advantage": float(diff[keep].mean()),
            "rows_kept": int(keep.sum()),
            "rows_excluded": int((~keep).sum()),
            "max_excluded_spend": float(y[~keep].min()),
        })

    detail = pd.DataFrame(output)

    summary = (
        detail.groupby(
            ["target", "model", "transform"],
            sort=False,
        )
        .agg(
            n_splits=("seed", "nunique"),
            original_advantage_mean=("original_advantage", "mean"),
            trimmed_advantage_mean=("trimmed_advantage", "mean"),
            trimmed_advantage_sd=("trimmed_advantage", "std"),
            mean_rows_excluded=("rows_excluded", "mean"),
            mean_max_excluded_spend=("max_excluded_spend", "mean"),
        )
        .reset_index()
    )

    detail_path = (
        OUT_DIR / "hillstrom_top_spend_sensitivity_by_split.csv"
    )
    summary_path = (
        OUT_DIR / "hillstrom_top_spend_sensitivity_summary.csv"
    )

    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)

    print("\n=== TOP-SPEND SENSITIVITY CHECK ===")
    print(summary.round(5).to_string(index=False))
    print("\nSaved:", detail_path)
    print("Saved:", summary_path)



# ============================================================
# MAIN
# ============================================================

def main():
    df = load_predictions()

    print("Prediction rows:", len(df))
    print("Unique customer IDs:", df["test_row_index"].nunique())
    print("Seeds:", sorted(df["seed"].unique().tolist()))

    policy_rows = prepare_policy_rows(df)

    cluster_bootstrap_summary(policy_rows)
    selection_recurrence(policy_rows)
    top_spend_sensitivity(policy_rows)

    print("\nDiagnostics complete.")


if __name__ == "__main__":
    main()
