"""rebuild_figures.py (OWNER: Member Q).
Clean reproducibility script to regenerate all manuscript figures and tables from results.

Usage:
    python src/rebuild_figures.py
    python src/rebuild_figures.py --results-dir results/main --out-dir figures
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analysis import run_analysis
from summarize_real import print_table3
from summarize import summarize_results


def rebuild_all(results_dir="results/main", real_dir="results", out_dir="figures"):
    os.makedirs(out_dir, exist_ok=True)
    print("=" * 80)
    print(" REBUILDING ALL MANUSCRIPT FIGURES AND TABLES")
    print("=" * 80)

    # 1. Simulation analysis & figures (Fig 2, Fig 4)
    if os.path.exists(results_dir) and any(os.scandir(results_dir)):
        print(f"\n[1/3] Generating simulation figures from '{results_dir}'...")
        run_analysis(results_dir, out_dir=out_dir)
        print(f"Summary of quality gates for '{results_dir}':")
        summarize_results(results_dir)
    elif os.path.exists("results/pilot"):
        print(f"\n[1/3] '{results_dir}' not populated yet; generating figures from 'results/pilot'...")
        run_analysis("results/pilot", out_dir=out_dir)
    else:
        print(f"\n[1/3] Warning: No simulation results found in '{results_dir}'.")

    # 2. Real-data benchmark tables (Table 3)
    mens_csv = os.path.join(real_dir, "hillstrom_mens.csv")
    womens_csv = os.path.join(real_dir, "hillstrom_womens.csv")
    existing_real = [p for p in [mens_csv, womens_csv] if os.path.exists(p)]
    if existing_real:
        print(f"\n[2/3] Formatting Table 3 from {existing_real}...")
        print_table3(existing_real)
    else:
        print(f"\n[2/3] Warning: Real-data results not found in '{real_dir}'.")

    print("\n[3/3] Rebuild process complete. Artifacts saved in:", out_dir)
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rebuild manuscript figures and tables.")
    parser.add_argument("--results-dir", default="results/main", help="Path to simulation results")
    parser.add_argument("--real-dir", default="results", help="Directory containing real-data results")
    parser.add_argument("--out-dir", default="figures", help="Output directory for generated figures")
    args = parser.parse_args()

    rebuild_all(results_dir=args.results_dir, real_dir=args.real_dir, out_dir=args.out_dir)
