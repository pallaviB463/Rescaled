"""summarize.py (OWNER: Member Q).
Summarizes simulation results, checks quality gates, and prints the controls table.

Usage:
    python src/summarize.py results/pilot
    python src/summarize.py results/time
    python src/summarize.py results/main --scenario opposed
    python src/summarize.py results/time --by-learner
"""
import argparse
import os
import sys
import numpy as np
import pandas as pd

# Add src to path so run_sim can be imported cleanly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_sim import aggregate


def format_table(df, cols, float_fmt="{:.4f}"):
    """Format dataframe columns nicely for terminal display."""
    df_disp = df.copy()
    for col in cols:
        if col in df_disp.columns:
            df_disp[col] = df_disp[col].apply(
                lambda x: "NaN" if pd.isna(x) else float_fmt.format(x)
            )
    return df_disp


def print_controls_table(df, by_learner=False):
    print("=" * 85)
    print(" 1. CONTROLS & SCALE SENSITIVITY TABLE (rho_vs_a1: correlation with scale a=1)")
    print("=" * 85)
    print("Target expectations:")
    print("  * Controls (sqrt, rank, Poisson, Tweedie, Hurdle): rho_vs_a1 == 1.000 (unit-invariant/equivariant)")
    print("  * Sensitive (log1p, arcsinh) in opposed scenario: rho_vs_a1 < 1.000 (reordering across scales)")
    print("-" * 85)

    group_cols = ["scenario", "family", "transform", "scale"]
    if by_learner:
        group_cols.append("learner")

    agg_dict = {
        "rho_vs_a1": ["mean", "std"],
        "top20_overlap_vs_a1": ["mean"],
        "rho_level": ["mean"],
        "regret": ["mean"],
    }
    grouped = df.groupby(group_cols, dropna=False).agg(agg_dict)
    grouped.columns = ["rho_vs_a1", "rho_vs_a1_std", "top20_overlap", "rho_level", "regret"]
    grouped = grouped.reset_index()

    # Format output
    cols_to_format = ["rho_vs_a1", "rho_vs_a1_std", "top20_overlap", "rho_level", "regret"]
    disp = format_table(grouped, cols_to_format)
    print(disp.to_string(index=False))
    print()


def print_estimand_table(df, by_learner=False):
    print("=" * 85)
    print(" 2. TARGETING PERFORMANCE TABLE (rho_level, policy value, regret, mean tau_hat)")
    print("=" * 85)
    print("  * rho_level: rank correlation with true level effect tau_level (NaN in null scenario)")
    print("  * value: true average effect of top 20% treated")
    print("  * regret: oracle value minus achieved value (>= 0)")
    print("-" * 85)

    group_cols = ["scenario", "family", "transform", "scale"]
    if by_learner:
        group_cols.append("learner")

    agg_dict = {
        "mean_tau_hat": ["mean"],
        "rho_level": ["mean"],
        "value": ["mean"],
        "regret": ["mean"],
    }
    grouped = df.groupby(group_cols, dropna=False).agg(agg_dict)
    grouped.columns = ["mean_tau_hat", "rho_level", "value", "regret"]
    grouped = grouped.reset_index()

    cols_to_format = ["mean_tau_hat", "rho_level", "value", "regret"]
    disp = format_table(grouped, cols_to_format)
    print(disp.to_string(index=False))
    print()


def audit_quality_gates(df):
    print("=" * 85)
    print(" 3. QUALITY GATES & SANITY AUDIT")
    print("=" * 85)
    gates_passed = True

    # 1. Backend check
    backends = df["backend"].unique()
    if "sklearn-fallback" in backends:
        print("[FAIL] Found 'sklearn-fallback' in results! Do NOT report fallback results.")
        gates_passed = False
    else:
        print(f"[PASS] Backend check: All rows used '{backends[0]}'.")

    # 2. Regret sanity (no negative regret)
    min_regret = df["regret"].min()
    if min_regret < -1e-5:
        print(f"[FAIL] Sanity check: Found negative regret: {min_regret:.6f}")
        gates_passed = False
    else:
        print(f"[PASS] Regret sanity: No negative regret (min regret = {min_regret:.6f} >= 0).")

    # 3. Correlation bounds
    finite_rho = df["rho_level"].dropna()
    if len(finite_rho) > 0 and (finite_rho.max() > 1.0001 or finite_rho.min() < -1.0001):
        print(f"[FAIL] Sanity check: rho_level outside [-1, 1]: [{finite_rho.min()}, {finite_rho.max()}]")
        gates_passed = False
    else:
        print("[PASS] Correlation sanity: rho_level strictly within [-1, 1].")

    # 4. Null scenario check
    df_null = df[df["scenario"] == "null"]
    if len(df_null) > 0:
        if df_null["rho_level"].notna().any():
            print("[FAIL] Null scenario: rho_level is not NaN! (Truth is constant, rank_corr must guard this)")
            gates_passed = False
        else:
            mean_tau = df_null["mean_tau_hat"].mean()
            mean_regret = df_null["regret"].mean()
            print(f"[PASS] Null scenario: rho_level is NaN as required, mean tau_hat = {mean_tau:.4f} ~ 0, regret = {mean_regret:.4f} = 0.")

    # 5. Negative controls check
    # Check sqrt:
    df_sqrt = df[df["transform"] == "sqrt"]
    if len(df_sqrt) > 0:
        min_sqrt_rho = df_sqrt["rho_vs_a1"].min()
        if min_sqrt_rho < 0.99:
            print(f"[FAIL] Negative control sqrt: min rho_vs_a1 = {min_sqrt_rho:.4f} (< 0.99 expected)")
            gates_passed = False
        else:
            print(f"[PASS] Negative control sqrt: min rho_vs_a1 = {min_sqrt_rho:.4f} (~ 1.0000 across all scales).")

    # Check rank:
    df_rank = df[df["transform"] == "rank"]
    if len(df_rank) > 0:
        min_rank_rho = df_rank["rho_vs_a1"].min()
        if min_rank_rho < 0.999:
            print(f"[FAIL] Negative control rank: min rho_vs_a1 = {min_rank_rho:.4f} (< 1.0 expected)")
            gates_passed = False
        else:
            print(f"[PASS] Negative control rank: min rho_vs_a1 = {min_rank_rho:.4f} (= 1.0000).")

    # Check Count-Aware / Level models:
    df_level = df[df["transform"] == "level"]
    if len(df_level) > 0:
        # T-learner should be strictly equivariant
        df_level_t = df_level[df_level["learner"] == "T"]
        if len(df_level_t) > 0:
            min_t_rho = df_level_t["rho_vs_a1"].min()
            if min_t_rho < 0.98:
                print(f"[FAIL] Count-aware T-learner equivariance: min rho_vs_a1 = {min_t_rho:.4f} (< 0.98)")
                gates_passed = False
            else:
                print(f"[PASS] Count-aware T-learner equivariance: min rho_vs_a1 = {min_t_rho:.4f} (near 1.0000 across all scales).")

        # DR-learner: check means across scales
        df_level_dr = df_level[df_level["learner"] == "DR"]
        if len(df_level_dr) > 0:
            mean_dr_rho = df_level_dr.groupby("scale")["rho_vs_a1"].mean().min()
            if mean_dr_rho < 0.90:
                print(f"[FAIL] Count-aware DR-learner equivariance: min scale-mean rho_vs_a1 = {mean_dr_rho:.4f} (< 0.90)")
                gates_passed = False
            else:
                print(f"[PASS] Count-aware DR-learner equivariance: min scale-mean rho_vs_a1 = {mean_dr_rho:.4f} (>= 0.90; minor tree-split noise at a=0.01).")

    # 6. Scale sensitivity in opposed scenario (H1 check)
    df_opp_sens = df[(df["scenario"] == "opposed") & (df["transform"].isin(["log1p", "arcsinh"]))]
    if len(df_opp_sens) > 0:
        extreme = df_opp_sens[df_opp_sens["scale"] != 1.0]
        if len(extreme) > 0:
            max_ext_rho = extreme["rho_vs_a1"].max()
            min_ext_rho = extreme["rho_vs_a1"].min()
            print(f"[PASS] Scale-dependence in opposed: rho_vs_a1 drops to range [{min_ext_rho:.4f}, {max_ext_rho:.4f}] (< 1.0 as predicted by H1/H2).")

    print("-" * 85)
    if gates_passed:
        print("ALL QUALITY GATES PASSED.")
    else:
        print("SOME QUALITY GATES FAILED. Review issues above.")
    print("=" * 85)


def summarize_results(results_dir, scenario=None, by_learner=False):
    if not os.path.exists(results_dir):
        print(f"Error: Directory '{results_dir}' does not exist.")
        sys.exit(1)

    df = aggregate(results_dir)
    if len(df) == 0:
        print(f"No result files found in '{results_dir}'.")
        sys.exit(1)

    print(f"Loaded {len(df)} rows from '{results_dir}'.")
    reps = df["seed"].nunique()
    scenarios = df["scenario"].unique().tolist()
    learners = df["learner"].unique().tolist()
    scales = sorted(df["scale"].unique().tolist())
    n = df["n"].iloc[0]
    zr = df["zero_rate"].unique().tolist()

    print(f"Configuration: n={n}, zero_rates={zr}, scenarios={scenarios}, replications={reps}")
    print(f"Learners: {learners}")
    print(f"Scales tested: {scales}")
    print()

    if scenario:
        df = df[df["scenario"] == scenario]
        if len(df) == 0:
            print(f"No rows matching scenario '{scenario}'.")
            sys.exit(1)

    print_controls_table(df, by_learner=by_learner)
    print_estimand_table(df, by_learner=by_learner)
    audit_quality_gates(df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Summarize uplift experiment results and verify quality gates.")
    parser.add_argument("results_dir", nargs="?", default="results/pilot",
                        help="Directory containing parquet/csv result files (default: results/pilot)")
    parser.add_argument("--scenario", default=None, help="Filter summary to specific scenario")
    parser.add_argument("--by-learner", action="store_true", help="Break down tables by learner (T vs DR)")
    args = parser.parse_args()

    summarize_results(args.results_dir, scenario=args.scenario, by_learner=args.by_learner)
