
"""Oracle ranking analysis across outcome transformations and scales."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.dgp import SCENARIOS, simulate, true_transformed_cate
from src.metrics import rank_corr, topk_overlap
from src.transforms import SCALES, TRANSFORMS


OUTPUT_DIR = Path("results")
ORACLE_SCENARIO = "opposed"


def calculate_oracle_table(
    scenario=ORACLE_SCENARIO,
    n=50_000,
    zero_rate=0.9,
    sigma=0.8,
    seed=2026,
    scales=None,
):
    """Compare true transformed CATE rankings with level-scale CATE rankings."""
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario!r}")
    if n < 2:
        raise ValueError("n must be at least 2.")

    scales = list(SCALES if scales is None else scales)
    if not scales:
        raise ValueError("At least one scale is required.")
    if any(not np.isfinite(a) or a <= 0 for a in scales):
        raise ValueError("Scales must be finite and positive.")

    # Use one common covariate sample for every transformation and scale.
    _, _, _, truth = simulate(
        n=n,
        scenario=scenario,
        zero_rate=zero_rate,
        sigma=sigma,
        seed=seed,
    )

    tau_level = np.asarray(truth["tau_level"], dtype=float)
    rows = []

    # If the level-scale CATE is constant, there is no meaningful
    # level-scale ranking against which to compare transformed rankings.
    ranking_defined = np.ptp(tau_level) > 0

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
                    tau_transformed, tau_level, k=0.2
                )
            else:
                rank_corr_value = np.nan
                top20_iou_value = np.nan

            rows.append(
                {
                    "scenario": scenario,
                    "transform": transform_name,
                    "scale": float(scale),
                    "log10_scale": float(np.log10(scale)),
                    "rank_corr_vs_level": rank_corr_value,
                    "top20_iou_vs_level": top20_iou_value,
                }
            )

    # Positive scaling cannot change ranks under the rank transform.
    # This is a useful invariant check, not an additional oracle curve.
    return pd.DataFrame(rows)


def save_oracle_figure(table, output_path):
    """Save Figure 1: oracle ranking metrics against log10 outcome scale."""
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

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True)

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

    fig.suptitle("Figure 1. Oracle ranking sensitivity to outcome scale")
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    table = calculate_oracle_table()
    csv_path = OUTPUT_DIR / "oracle_ranking_table.csv"
    figure_path = OUTPUT_DIR / "figure1_oracle_ranking.png"

    table.to_csv(csv_path, index=False)
    save_oracle_figure(table, figure_path)

    print(f"Oracle table saved to: {csv_path}")
    print(f"Figure saved to: {figure_path}")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
