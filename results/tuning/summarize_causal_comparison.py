import pandas as pd
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent

files = {
    "L2": RESULTS_DIR / "causal_tuning_compare_L2.csv",
    "Tweedie": RESULTS_DIR / "causal_tuning_compare_Tweedie.csv",
    "Hurdle": RESULTS_DIR / "causal_tuning_compare_Hurdle.csv",
}

frames = []

for model, filename in files.items():
    df = pd.read_csv(filename)
    df["outcome_model"] = model
    frames.append(df)

data = pd.concat(frames, ignore_index=True)

metrics = ["rank_corr", "topk_overlap", "policy_value", "regret"]
valid = data[data["status"] == "ok"]

summary = (
    valid.groupby(["scenario", "outcome_model"])[metrics]
    .agg(["mean", "std"])
    .round(3)
)

print("\nCAUSAL COMPARISON: MEAN AND STANDARD DEVIATION ACROSS SEEDS")
print(summary.to_string())

output = RESULTS_DIR / "causal_tuning_compare_all_summary.csv"
summary.to_csv(output)

print(f"\nSummary saved to: {output}")