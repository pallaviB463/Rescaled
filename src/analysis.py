"""analysis.py (OWNER: Member Q).
Aggregates simulation and real-data results, calculates bootstrap confidence intervals,
computes paired differences, and generates Figures 2, 3, and 4 for the manuscript.

Usage:
    python src/analysis.py results/pilot --out-dir figures
    python src/analysis.py results/main --out-dir figures
"""
import argparse
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_sim import aggregate


def bootstrap_ci(arr, n_boot=1000, ci=95, seed=42):
    """Compute bootstrap mean and confidence interval for a 1D array."""
    clean = np.asarray(arr, dtype=float)
    clean = clean[~np.isnan(clean)]
    if len(clean) == 0:
        return np.nan, np.nan, np.nan
    if len(clean) == 1:
        return float(clean[0]), float(clean[0]), float(clean[0])

    rng = np.random.default_rng(seed)
    boot_means = np.empty(n_boot)
    n = len(clean)
    for i in range(n_boot):
        sample = rng.choice(clean, size=n, replace=True)
        boot_means[i] = sample.mean()

    alpha = (100 - ci) / 2.0
    low = np.percentile(boot_means, alpha)
    high = np.percentile(boot_means, 100 - alpha)
    return float(np.mean(clean)), float(low), float(high)


def compute_paired_differences(df, baseline_family="Hurdle", metric="regret"):
    """Compute paired differences within the same replication seed (common random numbers)."""
    piv = df.pivot_table(index=["scenario", "zero_rate", "seed", "scale", "learner"],
                         columns=["family", "transform"], values=metric).reset_index()
    diffs = []
    # Identify baseline column
    base_col = None
    for col in piv.columns:
        if isinstance(col, tuple) and col[0] == baseline_family:
            base_col = col
            break

    if base_col is None:
        return pd.DataFrame()

    for col in piv.columns:
        if isinstance(col, tuple) and col != base_col:
            fam, tr = col
            diff = piv[col] - piv[base_col]
            for (sc, zr, sca, L), grp in piv.groupby(["scenario", "zero_rate", "scale", "learner"]):
                d_vals = (grp[col] - grp[base_col]).dropna()
                if len(d_vals) > 0:
                    mean_d, low_d, high_d = bootstrap_ci(d_vals)
                    diffs.append({
                        "scenario": sc, "zero_rate": zr, "scale": sca, "learner": L,
                        "family": fam, "transform": tr, "paired_diff_mean": mean_d,
                        "paired_diff_ci_low": low_d, "paired_diff_ci_high": high_d
                    })
    return pd.DataFrame(diffs)


def plot_fig2_rank_stability(df, out_path, scenario="opposed"):
    """Fig 2: Rank stability rho_vs_a1 vs log10(a), one line per family and learner."""
    sub = df[df["scenario"] == scenario].copy()
    if len(sub) == 0:
        return

    plt.figure(figsize=(9, 5.5))
    styles = {
        ("TransformedL2", "log1p"): ("blue", "-", "o"),
        ("TransformedL2", "arcsinh"): ("purple", "--", "s"),
        ("TransformedL2", "sqrt"): ("darkorange", "-.", "^"),
        ("TransformedL2", "rank"): ("gray", ":", "d"),
        ("Poisson", "level"): ("green", "-", "v"),
        ("Tweedie", "level"): ("teal", "--", "x"),
        ("Hurdle", "level"): ("red", "-", "*"),
    }

    for (fam, tr), grp in sub.groupby(["family", "transform"]):
        color, ls, marker = styles.get((fam, tr), ("black", "-", "."))
        pts = []
        for a, agrp in grp.groupby("scale"):
            m, low, high = bootstrap_ci(agrp["rho_vs_a1"])
            pts.append((a, m, low, high))
        pts.sort(key=lambda x: x[0])
        a_vals = [p[0] for p in pts]
        means = [p[1] for p in pts]
        lows = [p[2] for p in pts]
        highs = [p[3] for p in pts]

        lbl = f"{fam} ({tr})" if fam == "TransformedL2" else fam
        plt.plot(a_vals, means, label=lbl, color=color, linestyle=ls, marker=marker, linewidth=1.8, markersize=6)
        plt.fill_between(a_vals, lows, highs, color=color, alpha=0.15)

    plt.xscale("log")
    plt.xlabel("Unit Scale $a$ (log scale, $a=1$ is reference)", fontsize=11)
    plt.ylabel(r"Rank Stability $\rho_{\mathrm{vs\_a1}}$ with $a=1.0$", fontsize=11)
    plt.title(f"Figure 2: Rank Stability Across Unit Scales ({scenario.capitalize()} Scenario)", fontsize=12, fontweight="bold")
    plt.ylim(0.45, 1.05)
    plt.axhline(1.0, color="gray", linestyle=":", alpha=0.7)
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True, fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved Figure 2 to {out_path}")


def plot_fig4_regret_heatmap(df, out_path):
    """Fig 4: Regret heatmap by family and learner per scenario at scale a=1."""
    sub = df[df["scale"] == 1.0].copy()
    if len(sub) == 0:
        return

    piv = sub.pivot_table(index=["family", "transform", "learner"], columns="scenario", values="regret", aggfunc="mean")
    if piv.empty:
        return

    plt.figure(figsize=(10, 6))
    im = plt.imshow(piv.values, cmap="YlOrRd", aspect="auto")
    plt.colorbar(im, label="Mean Regret (lower is better)")
    plt.xticks(range(len(piv.columns)), piv.columns, rotation=30, ha="right", fontsize=10)
    row_labels = [f"{fam}-{tr} ({L})" for fam, tr, L in piv.index]
    plt.yticks(range(len(piv.index)), row_labels, fontsize=9)
    plt.title("Figure 4: Policy Regret Heatmap by Model Pipeline (Scale $a=1.0$)", fontsize=12, fontweight="bold")

    # Annotate values
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            val = piv.values[i, j]
            if not np.isnan(val):
                txt_color = "white" if val > (piv.values.max() * 0.65) else "black"
                plt.text(j, i, f"{val:.2f}", ha="center", va="center", color=txt_color, fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved Figure 4 to {out_path}")


def run_analysis(results_dir, out_dir="figures"):
    os.makedirs(out_dir, exist_ok=True)
    df = aggregate(results_dir)
    print(f"Loaded {len(df)} rows from {results_dir}.")

    # Fig 2
    plot_fig2_rank_stability(df, os.path.join(out_dir, "fig2_rank_stability.png"), scenario="opposed")

    # Fig 4
    plot_fig4_regret_heatmap(df, os.path.join(out_dir, "fig4_regret_heatmap.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", help="Directory with simulation parquet files")
    parser.add_argument("--out-dir", default="figures", help="Directory to save figures")
    args = parser.parse_args()
    run_analysis(args.results_dir, args.out_dir)
