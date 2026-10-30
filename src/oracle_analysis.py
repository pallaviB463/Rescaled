"""Oracle ranking and policy-value analysis across scenarios and scales."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.dgp import SCENARIOS, simulate, true_transformed_cate
from src.metrics import policy_value, rank_corr, topk_overlap
from src.transforms import SCALES, TRANSFORMS


OUTPUT_DIR = Path("results")
ORACLE_SCENARIO = "opposed"
ZERO_RATES = (0.70, 0.90, 0.98)
TOP_K = 0.20


def _validate_analysis_inputs(
    scenario, n, zero_rate, sigma, scales
):
    """Validate inputs shared by oracle analysis functions."""
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario!r}")
    if not isinstance(n, (int, np.integer)) or n < 2:
        raise ValueError("n must be an integer of at least 2.")
    if not np.isfinite(zero_rate) or not 0 < zero_rate < 1:
        raise ValueError("zero_rate must be strictly between 0 and 1.")
    if not np.isfinite(sigma) or sigma < 0:
        raise ValueError("sigma must be finite and nonnegative.")

    scales = list(SCALES if scales is None else scales)
    if not scales:
        raise ValueError("At least one scale is required.")
    if any(not np.isfinite(a) or a <= 0 for a in scales):
        raise ValueError("Scales must be finite and positive.")

    return scales


def calculate_oracle_table(
    scenario=ORACLE_SCENARIO,
    n=50_000,
    zero_rate=0.9,
    sigma=0.8,
    seed=2026,
    scales=None,
):
    """Compare transformed-CATE rankings with level-CATE rankings.

    The oracle policy value is the mean true level-scale treatment
    effect among the top TOP_K fraction selected by the true effect.
    """
    scales = _validate_analysis_inputs(
        scenario, n, zero_rate, sigma, scales
    )

    # Use one common covariate sample for every transformation and scale.
    _, _, _, truth = simulate(
        n=n,
        scenario=scenario,
        zero_rate=zero_rate,
        sigma=sigma,
        seed=seed,
    )

    tau_level = np.asarray(truth["tau_level"], dtype=float)

    if not np.all(np.isfinite(tau_level)):
        raise ValueError("The true level treatment effects must be finite.")

    # Oracle benchmark: select units using their true level-scale effects.
    oracle_value = policy_value(
        tau_level, tau_level, k=TOP_K
    )

    ranking_defined = np.ptp(tau_level) > 0
    rows = []

    for transform_name, transform in TRANSFORMS.items():
        for scale in scales:
            tau_transformed = true_transformed_cate(
                truth, m=transform, a=scale
            )

            if ranking_defined:
                rank_corr_value = rank_corr(
                    tau_transformed, tau_level
                )
                top20_iou_value = topk_overlap(
                    tau_transformed, tau_level, k=TOP_K
                )
            else:
                rank_corr_value = np.nan
                top20_iou_value = np.nan

            rows.append(
                {
                    "scenario": scenario,
                    "zero_rate": float(zero_rate),
                    "transform": transform_name,
                    "scale": float(scale),
                    "log10_scale": float(np.log10(scale)),
                    "rank_corr_vs_level": rank_corr_value,
                    "top20_iou_vs_level": top20_iou_value,
                    "oracle_policy_value_top20": oracle_value,
                }
            )

    return pd.DataFrame(rows)


def calculate_full_oracle_analysis(
    scenarios=None,
    zero_rates=ZERO_RATES,
    n=50_000,
    sigma=0.8,
    seed=2026,
    scales=None,
):
    """Run oracle ranking analysis for every scenario and zero rate.

    Returns
    -------
    ranking_table : pandas.DataFrame
        One row per scenario, zero rate, transformation, and scale.
    policy_table : pandas.DataFrame
        One row per scenario and zero rate, containing the oracle
        top-20% policy value.
    """
    scenarios = list(SCENARIOS if scenarios is None else scenarios)
    zero_rates = list(zero_rates)

    if not scenarios:
        raise ValueError("At least one scenario is required.")
    if not zero_rates:
        raise ValueError("At least one zero rate is required.")

    invalid_scenarios = [s for s in scenarios if s not in SCENARIOS]
    if invalid_scenarios:
        raise ValueError(f"Unknown scenarios: {invalid_scenarios}")

    if any(
        not np.isfinite(rate) or not 0 < rate < 1
        for rate in zero_rates
    ):
        raise ValueError("Zero rates must be strictly between 0 and 1.")

    ranking_tables = []

    for scenario_index, scenario in enumerate(scenarios):
        for rate_index, zero_rate in enumerate(zero_rates):
            # Derive a reproducible but distinct seed for each cell.
            cell_seed = seed + scenario_index * 100 + rate_index

            table = calculate_oracle_table(
                scenario=scenario,
                n=n,
                zero_rate=zero_rate,
                sigma=sigma,
                seed=cell_seed,
                scales=scales,
            )
            ranking_tables.append(table)

    ranking_table = pd.concat(
        ranking_tables, ignore_index=True
    )

    policy_table = (
        ranking_table[
            [
                "scenario",
                "zero_rate",
                "oracle_policy_value_top20",
            ]
        ]
        .drop_duplicates()
        .sort_values(["scenario", "zero_rate"])
        .reset_index(drop=True)
    )

    return ranking_table, policy_table


def save_oracle_figure(table, output_path):
    """Save ranking sensitivity plots for one scenario and zero rate."""
    required = {
        "transform",
        "log10_scale",
        "rank_corr_vs_level",
        "top20_iou_vs_level",
    }
    missing = required - set(table.columns)

    if missing:
        raise ValueError(
            f"Oracle table is missing columns: {sorted(missing)}"
        )
    if table.empty:
        raise ValueError("Cannot plot an empty oracle table.")

    fig, axes = plt.subplots(
        1, 2, figsize=(12, 4.8), sharex=True
    )

    for name, group in table.groupby("transform", sort=False):
        group = group.sort_values("log10_scale")

        axes[0].plot(
            group["log10_scale"],
            group["rank_corr_vs_level"],
            marker="o",
            label=name,
        )
        axes[1].plot(
            group["log10_scale"],
            group["top20_iou_vs_level"],
            marker="o",
            label=name,
        )

    axes[0].set_title("Oracle rank correlation")
    axes[0].set_ylabel("Spearman correlation")
    axes[1].set_title("Oracle top-20% overlap")
    axes[1].set_ylabel("Intersection over union")

    for ax in axes:
        ax.set_xlabel(r"$\log_{10}(a)$")
        ax.grid(True, alpha=0.3)
        ax.set_xticks(sorted(table["log10_scale"].unique()))

    axes[0].set_ylim(-1.05, 1.05)
    axes[1].set_ylim(0, 1.05)
    axes[1].legend(title="Transformation", loc="best")

    scenario = (
        str(table["scenario"].iloc[0])
        if "scenario" in table.columns
        else "unspecified"
    )
    zero_rate = (
        float(table["zero_rate"].iloc[0])
        if "zero_rate" in table.columns
        else None
    )

    subtitle = f"Scenario: {scenario}"
    if zero_rate is not None:
        subtitle += f" | Target zero rate: {zero_rate:.0%}"

    fig.suptitle(
        "Figure 1. Oracle ranking sensitivity to outcome scale\n"
        + subtitle
    )
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_full_oracle_outputs(
    ranking_table,
    policy_table,
    output_dir=OUTPUT_DIR,
):
    """Save combined tables and one ranking figure per scenario/rate."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ranking_path = output_dir / "oracle_ranking_full_grid.csv"
    policy_path = output_dir / "oracle_policy_values.csv"

    ranking_table.to_csv(ranking_path, index=False)
    policy_table.to_csv(policy_path, index=False)

    for (scenario, zero_rate), group in ranking_table.groupby(
        ["scenario", "zero_rate"], sort=False
    ):
        rate_label = f"{zero_rate:.2f}".replace(".", "p")
        figure_path = (
            output_dir
            / f"figure1_oracle_{scenario}_zero_{rate_label}.png"
        )
        save_oracle_figure(group, figure_path)

    print(f"Full ranking table saved to: {ranking_path}")
    print(f"Oracle policy values saved to: {policy_path}")
    print(
        "Ranking figures saved under: "
        f"{output_dir.resolve()}"
    )
    print("\nOracle policy-value table:")
    print(policy_table.to_string(index=False))


def main():
    ranking_table, policy_table = calculate_full_oracle_analysis(
        scenarios=SCENARIOS,
        zero_rates=ZERO_RATES,
        n=50_000,
        sigma=0.8,
        seed=2026,
    )

    save_full_oracle_outputs(
        ranking_table,
        policy_table,
        output_dir=OUTPUT_DIR,
    )


if __name__ == "__main__":
    main()