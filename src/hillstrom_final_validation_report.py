from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(".")
RESULTS = ROOT / "results" / "hillstrom"
PRED_PATH = RESULTS / "hillstrom_test_predictions.csv"
BOOT_MODEL_PATH = RESULTS / "hillstrom_paired_bootstrap_model_summary_corrected.csv"
BOOT_COMPARE_PATH = RESULTS / "hillstrom_paired_bootstrap_comparisons_corrected.csv"
OUT_MD = RESULTS / "hillstrom_final_validation_report.md"
OUT_CSV = RESULTS / "hillstrom_final_validation_policy_summary.csv"

TOP_FRACTION = 0.20
PROPENSITY = 0.50
REFERENCE = ("L2", "log1p")


def require_file(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found: {path}\n"
            "Run the model validation and corrected paired-bootstrap scripts first."
        )


def fmt(x, digits=5):
    return "NA" if pd.isna(x) else f"{x:.{digits}f}"


def main():
    for path in [PRED_PATH, BOOT_MODEL_PATH, BOOT_COMPARE_PATH]:
        require_file(path)

    df = pd.read_csv(PRED_PATH)
    df = df.loc[df["target"].eq("spend")].copy()
    required = {
        "seed", "test_row_index", "model", "transform",
        "treatment", "observed_outcome", "pred_uplift"
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Prediction CSV missing columns: {sorted(missing)}")

    for col in ["seed", "test_row_index", "treatment", "observed_outcome", "pred_uplift"]:
        df[col] = pd.to_numeric(df[col], errors="raise")

    if df.empty:
        raise ValueError("No spend rows found.")
    if not df["treatment"].isin([0, 1]).all():
        raise ValueError("Treatment must contain only 0 and 1.")
    if df[["treatment", "observed_outcome", "pred_uplift"]].isna().any().any():
        raise ValueError("Missing values in required columns.")

    # Ensure treatment and observed outcome are fixed for each customer.
    customer_check = df.groupby("test_row_index").agg(
        treatment_n=("treatment", "nunique"),
        outcome_n=("observed_outcome", "nunique"),
    )
    inconsistent_treatment = int((customer_check["treatment_n"] > 1).sum())
    inconsistent_outcome = int((customer_check["outcome_n"] > 1).sum())
    if inconsistent_treatment or inconsistent_outcome:
        raise ValueError(
            "Customer-level treatment/outcome inconsistency detected: "
            f"treatment={inconsistent_treatment}, outcome={inconsistent_outcome}"
        )

    rows = []
    for (model, transform, seed), g in df.groupby(
        ["model", "transform", "seed"], sort=True
    ):
        g = g.reset_index(drop=True)
        n = len(g)
        n_selected = max(1, int(np.ceil(n * TOP_FRACTION)))
        selected = np.zeros(n, dtype=float)
        selected[np.argsort(g["pred_uplift"].to_numpy(), kind="mergesort")[-n_selected:]] = 1.0

        y = g["observed_outcome"].to_numpy(dtype=float)
        w = g["treatment"].to_numpy(dtype=float)
        contribution = w * y / PROPENSITY - (1 - w) * y / (1 - PROPENSITY)
        frac = selected.mean()
        advantage = float(np.mean((selected - frac) * contribution))
        policy_value = float(np.mean(selected * contribution))
        treat_all_value = float(np.mean(contribution))
        random_value = float(frac * treat_all_value)

        rows.append({
            "model": model,
            "transform": transform,
            "seed": int(seed),
            "n_test_rows": n,
            "n_selected": int(n_selected),
            "selected_fraction": frac,
            "policy_value_incremental": policy_value,
            "random_policy_value": random_value,
            "policy_advantage": advantage,
            "treated_mean_spend_test": float(y[w == 1].mean()),
            "control_mean_spend_test": float(y[w == 0].mean()),
        })

    split = pd.DataFrame(rows)
    split.to_csv(OUT_CSV, index=False)

    summary = (
        split.groupby(["model", "transform"], as_index=False)
        .agg(
            n_splits=("seed", "nunique"),
            mean_advantage=("policy_advantage", "mean"),
            split_sd=("policy_advantage", "std"),
            positive_splits=("policy_advantage", lambda s: int((s > 0).sum())),
            mean_policy_value=("policy_value_incremental", "mean"),
            mean_random_value=("random_policy_value", "mean"),
            mean_test_rows=("n_test_rows", "mean"),
        )
        .sort_values("mean_advantage", ascending=False)
    )

    boot_models = pd.read_csv(BOOT_MODEL_PATH)
    boot_comps = pd.read_csv(BOOT_COMPARE_PATH)

    expected_model_cols = {
        "model", "transform", "observed_mean_advantage",
        "bootstrap_ci95_low", "bootstrap_ci95_high", "positive_splits"
    }
    if expected_model_cols - set(boot_models.columns):
        raise ValueError(
            f"Bootstrap model summary missing columns: {sorted(expected_model_cols - set(boot_models.columns))}"
        )

    expected_comp_cols = {
        "comparison_model", "comparison_transform",
        "observed_difference_reference_minus_comparison",
        "paired_ci95_low", "paired_ci95_high",
        "bonferroni_ci99_low", "bonferroni_ci99_high",
        "bonferroni_ci_excludes_zero"
    }
    if expected_comp_cols - set(boot_comps.columns):
        raise ValueError(
            f"Bootstrap comparison file missing columns: {sorted(expected_comp_cols - set(boot_comps.columns))}"
        )

    merged = summary.merge(
        boot_models[
            ["model", "transform", "observed_mean_advantage",
             "bootstrap_ci95_low", "bootstrap_ci95_high", "positive_splits"]
        ].rename(columns={"positive_splits": "bootstrap_positive_splits"}),
        on=["model", "transform"], how="left"
    )

    ref = merged[
        merged["model"].eq(REFERENCE[0]) & merged["transform"].eq(REFERENCE[1])
    ]
    if ref.empty:
        raise ValueError(f"Reference model {REFERENCE} missing from results.")
    ref = ref.iloc[0]

    n_customers = int(df["test_row_index"].nunique())
    n_seeds = int(df["seed"].nunique())
    n_rows = len(df)
    per_seed_counts = df.groupby("seed")["test_row_index"].nunique()
    all_seeds_same_size = bool(per_seed_counts.nunique() == 1)

    lines = []
    lines += [
        "# Hillstrom final validation report",
        "",
        "## Scope",
        "",
        "This report summarizes the saved spend predictions and the corrected paired customer-cluster bootstrap outputs. It does not refit models.",
        "",
        "## Data and split checks",
        "",
        f"- Spend prediction records: **{n_rows:,}**.",
        f"- Unique customers represented across test splits: **{n_customers:,}**.",
        f"- Seeds/splits: **{n_seeds}**.",
        f"- Unique test rows per seed: **{int(per_seed_counts.iloc[0]):,}** (same size across seeds: **{all_seeds_same_size}**).",
        f"- Treatment consistency across repeated customer records: **{'PASS' if inconsistent_treatment == 0 else 'FAIL'}**.",
        f"- Observed-outcome consistency across repeated customer records: **{'PASS' if inconsistent_outcome == 0 else 'FAIL'}**.",
        f"- Propensity used in this summary: **{PROPENSITY:.2f}**.",
        "",
        "## Policy advantage by model",
        "",
        "| Model | Transform | Mean advantage ($/customer) | Split SD | Positive splits | Conditional bootstrap 95% CI |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for _, r in merged.iterrows():
        lines.append(
            f"| {r['model']} | {r['transform']} | {fmt(r['mean_advantage'])} | "
            f"{fmt(r['split_sd'])} | {int(r['positive_splits_x']) if 'positive_splits_x' in r and pd.notna(r['positive_splits_x']) else int(r['positive_splits']) if pd.notna(r.get('positive_splits')) else 'NA'}/"
            f"{int(r['n_splits'])} | [{fmt(r['bootstrap_ci95_low'])}, {fmt(r['bootstrap_ci95_high'])}] |"
        )

    lines += [
        "",
        "## Reference model",
        "",
        f"The highest-ranked reference candidate is **{REFERENCE[0]} + {REFERENCE[1]}**.",
        f"- Mean advantage across saved splits: **${fmt(float(ref['mean_advantage']))} per customer**.",
        f"- Positive splits: **{int(ref['positive_splits'])}/{int(ref['n_splits'])}**.",
        f"- Conditional customer-cluster bootstrap 95% interval: **[${fmt(float(ref['bootstrap_ci95_low']))}, ${fmt(float(ref['bootstrap_ci95_high']))}]**.",
        "",
        "If this interval includes zero, the saved-prediction analysis does not establish positive policy advantage over random targeting.",
        "",
        "## Paired model comparisons",
        "",
        "The following table compares L2 + log1p against each alternative. The 99% intervals use Bonferroni adjustment for five comparisons.",
        "",
        "| Alternative | Difference in mean advantage | Paired 95% CI | Bonferroni 99% CI | Adjusted interval excludes zero? |",
        "|---|---:|---:|---:|---|",
    ]
    for _, r in boot_comps.iterrows():
        lines.append(
            f"| {r['comparison_model']} + {r['comparison_transform']} | "
            f"${fmt(float(r['observed_difference_reference_minus_comparison']))} | "
            f"[${fmt(float(r['paired_ci95_low']))}, ${fmt(float(r['paired_ci95_high']))}] | "
            f"[${fmt(float(r['bonferroni_ci99_low']))}, ${fmt(float(r['bonferroni_ci99_high']))}] | "
            f"{'Yes' if bool(r['bonferroni_ci_excludes_zero']) else 'No'} |"
        )

    lines += [
        "",
        "## Interpretation and limitations",
        "",
        "1. These are estimates from the saved repeated train/test splits; the test sets overlap across seeds.",
        "2. The customer-cluster bootstrap reuses customer multiplicities across models and splits, and re-centers the selected fraction within each bootstrap split.",
        "3. Rankings and selected-customer membership are held fixed in the bootstrap. The intervals are conditional on the existing fitted predictions.",
        "4. The intervals do not include model-training uncertainty or uncertainty from regenerating the splits. Their coverage for a broader retraining/generalization claim is not established by this analysis alone.",
        "5. A model may rank first by mean advantage without having a statistically established positive advantage over random targeting.",
        "6. Spend caps and propensity checks are sensitivity analyses; they do not substitute for independent final evaluation.",
        "",
        "## Recommended conclusion",
        "",
        f"Treat **{REFERENCE[0]} + {REFERENCE[1]}** as the leading candidate based on the current point estimates, not as a proven successful policy unless its policy-advantage interval excludes zero under a pre-specified, defensible inference procedure. For a stronger generalization claim, run a pre-specified untouched holdout or nested repeated evaluation, with model selection kept separate from final evaluation.",
        "",
        "## Output",
        "",
        f"- Per-split policy summary: `{OUT_CSV.as_posix()}`",
        f"- This report: `{OUT_MD.as_posix()}`",
        "",
    ]

    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print("Saved final validation report:", OUT_MD)
    print("Saved per-split policy summary:", OUT_CSV)
    print("\n=== POLICY SUMMARY ===")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    print("\n=== REFERENCE MODEL ===")
    print(f"Mean advantage: ${float(ref['mean_advantage']):.5f} per customer")
    print(f"Conditional bootstrap 95% CI: [{float(ref['bootstrap_ci95_low']):.5f}, {float(ref['bootstrap_ci95_high']):.5f}]")
    print(f"Positive splits: {int(ref['positive_splits'])}/{int(ref['n_splits'])}")
    print("\nImportant: this report summarizes saved predictions; it does not refit models or establish training/split-generation uncertainty.")


if __name__ == "__main__":
    main()
