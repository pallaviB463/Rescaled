
from pathlib import Path

import numpy as np
import pandas as pd

# ============================================================
# CONFIGURATION
# ============================================================

PREDICTIONS_PATH = Path(
    "results/hillstrom/hillstrom_test_predictions.csv"
)
OUT_DIR = Path("results/hillstrom")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TREATMENT_PROPENSITY = 0.5
TOP_FRACTION = 0.20
WINSOR_QUANTILE = 0.99
N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 20261026

# ============================================================
# LOAD AND VALIDATE
# ============================================================

def load_predictions():
    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(
            f"Cannot find {PREDICTIONS_PATH}. "
            "Run the Hillstrom validation first."
        )

    df = pd.read_csv(PREDICTIONS_PATH)

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
            f"Missing required columns: {sorted(missing)}"
        )

    df = df.copy()
    df["seed"] = df["seed"].astype(int)
    df["test_row_index"] = df["test_row_index"].astype(int)

    for col in [
        "treatment",
        "observed_outcome",
        "pred_uplift",
    ]:
        df[col] = pd.to_numeric(df[col], errors="raise")

    if df[
        ["treatment", "observed_outcome", "pred_uplift"]
    ].isna().any().any():
        raise ValueError("Missing numeric values in predictions.")

    if not np.isfinite(
        df[["observed_outcome", "pred_uplift"]].to_numpy()
    ).all():
        raise ValueError("Non-finite outcome or uplift predictions.")

    if not df["treatment"].isin([0, 1]).all():
        raise ValueError("Treatment must contain only 0 and 1.")

    if (df["observed_outcome"] < 0).any():
        raise ValueError("Expected nonnegative outcomes.")

    return df


# ============================================================
# DETERMINE THE CAP FROM UNIQUE CUSTOMERS
# ============================================================


def get_spend_cap(df):
    spend = df[df["target"] == "spend"].copy()

    # Use one observation per original customer.
    unique = spend[
        ["test_row_index", "treatment", "observed_outcome"]
    ].drop_duplicates()

    consistency = unique.groupby("test_row_index").agg(
        treatment_values=("treatment", "nunique"),
        outcome_values=("observed_outcome", "nunique"),
    )

    if (
        (consistency["treatment_values"] > 1).any()
        or (consistency["outcome_values"] > 1).any()
    ):
        raise ValueError(
            "A customer has inconsistent treatment or observed outcomes."
        )

    unique = unique.drop_duplicates(subset=["test_row_index"])

    positive_spend = unique.loc[
        unique["observed_outcome"] > 0,
        "observed_outcome",
    ]

    if positive_spend.empty:
        raise ValueError("No positive spend observations found.")

    cap = float(positive_spend.quantile(WINSOR_QUANTILE))

    print("\n=== POSITIVE-SPEND WINSORIZATION THRESHOLD ===")
    print("Unique customers:", len(unique))
    print("Customers with positive spend:", len(positive_spend))
    print(f"Quantile among positive spenders: {WINSOR_QUANTILE:.2%}")
    print(f"Spend cap: {cap:.4f}")
    print(
        "Fraction of all customers above cap:",
        round(
            float((unique["observed_outcome"] > cap).mean()),
            5,
        ),
    )

    return cap



# ============================================================
# POLICY EVALUATION FOR ORIGINAL AND CAPPED SPEND
# ============================================================

def evaluate_split(group, outcome, seed):
    """
    Evaluate a top-fraction policy using the supplied outcome.

    The expected random baseline selects the same fraction of
    customers. Both values use the same IPW contribution.
    """
    y = np.asarray(outcome, dtype=float)
    w = group["treatment"].to_numpy(dtype=int)
    score = group["pred_uplift"].to_numpy(dtype=float)

    n = len(group)
    n_selected = max(1, int(np.ceil(n * TOP_FRACTION)))
    fraction = n_selected / n

    if not np.any(w == 0) or not np.any(w == 1):
        raise ValueError(
            f"Seed {seed} test set lacks one treatment arm."
        )

    p1 = TREATMENT_PROPENSITY
    p0 = 1.0 - p1

    contribution = w * y / p1 - (1 - w) * y / p0

    selected = np.zeros(n, dtype=bool)
    order = np.argsort(score, kind="mergesort")
    selected[order[-n_selected:]] = True

    model_contribution = selected.astype(float) * contribution
    difference_contribution = (
        selected.astype(float) - fraction
    ) * contribution

    return {
        "policy_value": float(model_contribution.mean()),
        "random_value": float(fraction * contribution.mean()),
        "treat_all_value": float(contribution.mean()),
        "advantage": float(difference_contribution.mean()),
        "difference_contribution": difference_contribution,
        "customer_ids": group["test_row_index"].to_numpy(dtype=int),
    }


# ============================================================
# BUILD SPLIT-LEVEL RESULTS
# ============================================================

def build_split_results(df, spend_cap):
    spend = df[df["target"] == "spend"].copy()
    rows = []
    bootstrap_groups = []

    group_cols = ["target", "model", "transform", "seed"]

    for keys, group in spend.groupby(group_cols, sort=False):
        group = group.copy().reset_index(drop=True)

        y_original = group["observed_outcome"].to_numpy(dtype=float)
        y_capped = np.minimum(y_original, spend_cap)

        original = evaluate_split(
            group, y_original, int(keys[3])
        )
        capped = evaluate_split(
            group, y_capped, int(keys[3])
        )

        rows.append({
            "target": keys[0],
            "model": keys[1],
            "transform": keys[2],
            "seed": int(keys[3]),
            "n_test": len(group),

            "original_policy_value": original["policy_value"],
            "original_random_value": original["random_value"],
            "original_treat_all_value": original["treat_all_value"],
            "original_advantage": original["advantage"],

            "winsorized_policy_value": capped["policy_value"],
            "winsorized_random_value": capped["random_value"],
            "winsorized_treat_all_value": capped["treat_all_value"],
            "winsorized_advantage": capped["advantage"],

            "spend_cap": spend_cap,
            "fraction_capped_in_split": float(
                (y_original > spend_cap).mean()
            ),
        })

        bootstrap_groups.append({
            "config": keys[:3],
            "seed": int(keys[3]),
            "customer_ids": group["test_row_index"].to_numpy(dtype=int),
            "original_difference": original["difference_contribution"],
            "winsorized_difference": capped["difference_contribution"],
        })

    return pd.DataFrame(rows), bootstrap_groups


# ============================================================
# CUSTOMER-CLUSTER BOOTSTRAP
# ============================================================

def customer_cluster_bootstrap(split_results, bootstrap_groups):
    """
    Resample original customer IDs as clusters.

    This preserves repeated appearances of the same customer across
    splits. The interval remains conditional on the fitted models
    and the existing dataset; it does not represent new-population
    or model-training uncertainty.
    """
    all_ids = np.unique(
        np.concatenate([
            item["customer_ids"] for item in bootstrap_groups
        ])
    )

    id_to_position = {
        customer_id: pos
        for pos, customer_id in enumerate(all_ids)
    }

    for item in bootstrap_groups:
        item["positions"] = np.array([
            id_to_position[cid]
            for cid in item["customer_ids"]
        ], dtype=int)

    configs = sorted(
        set(item["config"] for item in bootstrap_groups)
    )

    original_boot = {
        config: np.empty(N_BOOTSTRAP)
        for config in configs
    }
    capped_boot = {
        config: np.empty(N_BOOTSTRAP)
        for config in configs
    }

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    n_customers = len(all_ids)

    for b in range(N_BOOTSTRAP):
        sampled = rng.integers(
            0, n_customers, size=n_customers
        )
        multiplicities = np.bincount(
            sampled, minlength=n_customers
        )

        by_config = {
            config: {"original": [], "capped": []}
            for config in configs
        }

        for item in bootstrap_groups:
            weights = multiplicities[item["positions"]]

            if weights.sum() == 0:
                continue

            original_value = np.average(
                item["original_difference"],
                weights=weights,
            )
            capped_value = np.average(
                item["winsorized_difference"],
                weights=weights,
            )

            by_config[item["config"]]["original"].append(
                original_value
            )
            by_config[item["config"]]["capped"].append(
                capped_value
            )

        for config in configs:
            original_values = by_config[config]["original"]
            capped_values = by_config[config]["capped"]

            original_boot[config][b] = (
                np.mean(original_values)
                if original_values else np.nan
            )
            capped_boot[config][b] = (
                np.mean(capped_values)
                if capped_values else np.nan
            )

    summary_rows = []

    for config in configs:
        group = split_results[
            (split_results["target"] == config[0])
            & (split_results["model"] == config[1])
            & (split_results["transform"] == config[2])
        ]

        original_values = original_boot[config]
        original_values = original_values[
            np.isfinite(original_values)
        ]

        capped_values = capped_boot[config]
        capped_values = capped_values[
            np.isfinite(capped_values)
        ]

        original_ci = (
            np.quantile(original_values, [0.025, 0.975])
            if len(original_values)
            else [np.nan, np.nan]
        )
        capped_ci = (
            np.quantile(capped_values, [0.025, 0.975])
            if len(capped_values)
            else [np.nan, np.nan]
        )

        summary_rows.append({
            "target": config[0],
            "model": config[1],
            "transform": config[2],
            "n_splits": int(group["seed"].nunique()),

            "original_advantage_mean": float(
                group["original_advantage"].mean()
            ),
            "original_advantage_sd": float(
                group["original_advantage"].std()
            ),
            "original_positive_splits": int(
                (group["original_advantage"] > 0).sum()
            ),
            "original_cluster_ci_low": float(original_ci[0]),
            "original_cluster_ci_high": float(original_ci[1]),

            "winsorized_advantage_mean": float(
                group["winsorized_advantage"].mean()
            ),
            "winsorized_advantage_sd": float(
                group["winsorized_advantage"].std()
            ),
            "winsorized_positive_splits": int(
                (group["winsorized_advantage"] > 0).sum()
            ),
            "winsorized_cluster_ci_low": float(capped_ci[0]),
            "winsorized_cluster_ci_high": float(capped_ci[1]),
        })

    return pd.DataFrame(summary_rows)


# ============================================================
# MAIN
# ============================================================

def main():
    df = load_predictions()

    print("Loaded prediction rows:", len(df))
    print("Unique customer IDs:", df["test_row_index"].nunique())
    print("Seeds:", sorted(df["seed"].unique().tolist()))

    spend_cap = get_spend_cap(df)

    split_results, bootstrap_groups = build_split_results(
        df, spend_cap
    )

    split_path = (
        OUT_DIR / "hillstrom_robustness_split_results.csv"
    )
    split_results.to_csv(split_path, index=False)

    summary = customer_cluster_bootstrap(
        split_results, bootstrap_groups
    )

    summary_path = (
        OUT_DIR / "hillstrom_robustness_summary.csv"
    )
    summary.to_csv(summary_path, index=False)

    print("\n=== ORIGINAL VS WINSORIZED SPEND ===")
    print(
        summary.round(5).to_string(index=False)
    )

    print("\nSaved split results:", split_path)
    print("Saved robustness summary:", summary_path)

    print(
        "\nInterpretation note: winsorization changes the outcome "
        "being evaluated. Use the original-spend result as the "
        "primary business metric and the capped result only as "
        "a sensitivity analysis."
    )


if __name__ == "__main__":
    main()
