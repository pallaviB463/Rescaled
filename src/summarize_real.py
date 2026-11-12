"""summarize_real.py (OWNER: Member Q).
Aggregates and formats real-data experiment results (Task Q6 / Q10, Table 3).

Usage:
    python src/summarize_real.py results/hillstrom_mens.csv
    python src/summarize_real.py results/hillstrom_womens.csv
    python src/summarize_real.py results/hillstrom_mens.csv results/hillstrom_womens.csv
"""
import sys
import os
import pandas as pd
import numpy as np


def print_table3(csv_paths):
    dfs = []
    for p in csv_paths:
        if os.path.exists(p):
            dfs.append(pd.read_csv(p))
        else:
            print(f"Warning: File not found: {p}")
    if not dfs:
        return

    df = pd.concat(dfs, ignore_index=True)

    print("=" * 95)
    print(" TABLE 3: REAL DATA BENCHMARK (HILLSTROM METH-DATA FIELD EXPERIMENT)")
    print("=" * 95)
    print("Metrics: rho_vs_a1 (rank stability vs a=1), top20_overlap (budget k=0.20), IPW policy value, Qini")
    print("-" * 95)

    group_cols = ["arm", "family", "transform", "scale", "learner"]
    agg = df.groupby(group_cols).agg(
        rho_vs_a1=("rho_vs_a1", "mean"),
        top20_overlap=("top20_overlap_vs_a1", "mean"),
        ipw_value_k20=("ipw_value_k20", "mean"),
        qini=("qini", "mean")
    ).reset_index()

    # Sort for clear presentation
    agg = agg.sort_values(by=["arm", "learner", "family", "transform", "scale"])
    agg["scale"] = agg["scale"].apply(lambda s: f"{s:g}")
    agg["rho_vs_a1"] = agg["rho_vs_a1"].apply(lambda x: f"{x:.4f}")
    agg["top20_overlap"] = agg["top20_overlap"].apply(lambda x: f"{x:.4f}")
    agg["ipw_value_k20"] = agg["ipw_value_k20"].apply(lambda x: f"{x:.4f}")
    agg["qini"] = agg["qini"].apply(lambda x: f"{x:.4f}")

    print(agg.to_string(index=False))
    print("=" * 95)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        paths = ["results/hillstrom_mens.csv", "results/hillstrom_womens.csv"]
    else:
        paths = sys.argv[1:]
    print_table3(paths)
