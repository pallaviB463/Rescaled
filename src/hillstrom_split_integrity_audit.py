from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(".")
PRED_PATH = ROOT / "results" / "hillstrom" / "hillstrom_test_predictions.csv"
DATA_PATH = ROOT / "data" / "hillstrom" / "hillstrom_mens_control.csv"
OUT_DIR = ROOT / "results" / "hillstrom"
OUT_DIR.mkdir(parents=True, exist_ok=True)

if not PRED_PATH.exists():
    raise FileNotFoundError(f"Missing prediction file: {PRED_PATH}")

df = pd.read_csv(PRED_PATH)
required = {
    "seed", "test_row_index", "target", "model", "transform", "treatment",
    "observed_outcome", "pred_control", "pred_treated", "pred_uplift"
}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Prediction file is missing columns: {sorted(missing)}")

print("=" * 72)
print("HILLSTROM SPLIT-INTEGRITY AUDIT")
print("=" * 72)
print(f"Prediction rows: {len(df):,}")
print(f"Seeds: {df['seed'].nunique()} | targets: {df['target'].nunique()} | "
      f"models: {df['model'].nunique()} | transforms: {df['transform'].nunique()}")
print("Targets:", sorted(df["target"].dropna().astype(str).unique().tolist()))
print("\nRows per target/model/transform (should be consistent across seeds):")
counts = (df.groupby(["target", "model", "transform", "seed"], dropna=False)
            .agg(rows=("test_row_index", "size"),
                 unique_rows=("test_row_index", "nunique"))
            .reset_index())
print(counts.groupby(["target", "model", "transform"])[["rows", "unique_rows"]]
      .agg(["min", "max"]).to_string())

# 1. Is test_row_index stable across targets and modelling configurations?
print("\n" + "=" * 72)
print("1. ROW-INDEX AND CONFIGURATION CONSISTENCY")
print("=" * 72)
print("Unique test_row_index overall:", df["test_row_index"].nunique())
print("Unique (seed, test_row_index) pairs:",
      df[["seed", "test_row_index"]].drop_duplicates().shape[0])
print("Unique targets per (seed, test_row_index), distribution:")
target_per_row = df.groupby(["seed", "test_row_index"])["target"].nunique()
print(target_per_row.value_counts().sort_index().to_string())
print("Unique model/transform configurations per (seed, test_row_index, target), distribution:")
cfg_per_row = df.groupby(["seed", "test_row_index", "target"])[["model", "transform"]].apply(
    lambda x: len(x.drop_duplicates())
)
print(cfg_per_row.value_counts().sort_index().to_string())

# 2. Prediction arithmetic and finite values
print("\n" + "=" * 72)
print("2. NUMERIC / PREDICTION CONSISTENCY")
print("=" * 72)
finite_cols = ["observed_outcome", "pred_control", "pred_treated", "pred_uplift"]
for col in finite_cols:
    vals = pd.to_numeric(df[col], errors="coerce")
    print(f"{col}: nonnumeric/NaN={vals.isna().sum():,}; infinite={np.isinf(vals).sum():,}")
uplift_err = (df["pred_uplift"] - (df["pred_treated"] - df["pred_control"])).abs()
print("Max |pred_uplift - (pred_treated - pred_control)|:", float(uplift_err.max()))
print("Negative pred_control:", int((df["pred_control"] < 0).sum()))
print("Negative pred_treated:", int((df["pred_treated"] < 0).sum()))
print("Negative observed outcomes:", int((df["observed_outcome"] < 0).sum()))

# 3. Customer-level treatment/outcome consistency across appearances
print("\n" + "=" * 72)
print("3. CUSTOMER CONSISTENCY ACROSS SPLITS")
print("=" * 72)
consistency = (df.groupby(["target", "test_row_index"])
                 .agg(n_treatment=("treatment", "nunique"),
                      n_outcome=("observed_outcome", "nunique"),
                      n_seeds=("seed", "nunique"))
                 .reset_index())
print("Target/row groups with inconsistent treatment:", int((consistency["n_treatment"] > 1).sum()))
print("Target/row groups with inconsistent observed outcome:", int((consistency["n_outcome"] > 1).sum()))
print("Appearances per row within target: quantiles")
appearances = df.groupby(["target", "test_row_index"]).size()
print(appearances.quantile([0, .25, .5, .75, .9, .99, 1]).to_string())

# 4. Are same rows repeated across seeds, and is overlap expected from nominal test size?
print("\n" + "=" * 72)
print("4. TEST-SET OVERLAP BY SEED")
print("=" * 72)
seed_sizes = df.groupby(["target", "model", "transform", "seed"]).agg(
    records=("test_row_index", "size"),
    unique_rows=("test_row_index", "nunique")
).reset_index()
print(seed_sizes.groupby(["target", "model", "transform"])[["records", "unique_rows"]]
      .agg(["min", "max"]).to_string())

# Test-row set overlap should be same across configs if split assignment is shared.
sets = {}
for (target, seed), sub in df[["target", "seed", "test_row_index"]].drop_duplicates().groupby(["target", "seed"]):
    sets[(target, seed)] = set(sub["test_row_index"].tolist())
overlap_rows = []
for target in sorted(df["target"].dropna().unique(), key=str):
    seeds = sorted(df.loc[df["target"] == target, "seed"].dropna().unique().tolist())
    for i, s1 in enumerate(seeds):
        for s2 in seeds[i+1:]:
            a, b = sets[(target, s1)], sets[(target, s2)]
            inter = len(a & b)
            union = len(a | b)
            overlap_rows.append({
                "target": target, "seed_a": s1, "seed_b": s2,
                "n_a": len(a), "n_b": len(b), "intersection": inter,
                "jaccard": inter / union if union else np.nan
            })
overlap = pd.DataFrame(overlap_rows)
overlap_path = OUT_DIR / "hillstrom_split_pairwise_overlap.csv"
overlap.to_csv(overlap_path, index=False)
if not overlap.empty:
    print("Pairwise intersection size by target (across seed pairs):")
    print(overlap.groupby("target")["intersection"].agg(["min", "median", "mean", "max"]).to_string())
    print("Pairwise Jaccard by target:")
    print(overlap.groupby("target")["jaccard"].agg(["min", "median", "mean", "max"]).to_string())
    print("Saved pairwise overlap:", overlap_path)

# 5. Cross-check row index and observed labels against original filtered dataset, if possible.
print("\n" + "=" * 72)
print("5. ORIGINAL DATASET CROSS-CHECK")
print("=" * 72)
if DATA_PATH.exists():
    original = pd.read_csv(DATA_PATH)
    print("Original filtered file shape:", original.shape)
    print("Original columns:", original.columns.tolist())
    if "treatment" in original.columns and "spend" in original.columns:
        if pd.api.types.is_numeric_dtype(original.index):
            # test_row_index is presumed positional only for diagnostic; explicitly report this assumption.
            idx = pd.to_numeric(df["test_row_index"], errors="coerce")
            valid = idx.notna() & (idx >= 0) & (idx < len(original)) & (idx % 1 == 0)
            print("test_row_index values in positional range:", int(valid.sum()), "/", len(df))
            probe = df.loc[valid, ["target", "test_row_index", "treatment", "observed_outcome"]].copy()
            probe["pos"] = probe["test_row_index"].astype(int)
            probe = probe.drop_duplicates(["target", "pos"])
            # Dataset target mapping may vary; only compare when target column has a matching source column.
            for target in sorted(probe["target"].dropna().astype(str).unique()):
                if target not in original.columns:
                    print(f"Target '{target}': no matching original column; outcome check skipped.")
                    continue
                part = probe[probe["target"].astype(str) == target]
                ix = part["pos"].to_numpy()
                source_outcome = pd.to_numeric(original.iloc[ix][target], errors="coerce").to_numpy()
                pred_outcome = pd.to_numeric(part["observed_outcome"], errors="coerce").to_numpy()
                match = np.isclose(source_outcome, pred_outcome, equal_nan=True)
                print(f"Target '{target}': outcome matches at same positional index: "
                      f"{int(match.sum()):,}/{len(match):,}")
            if "spend" in original.columns:
                # Specifically verify the spend rows if target is spend.
                part = probe[probe["target"].astype(str) == "spend"]
                if not part.empty:
                    ix = part["pos"].to_numpy()
                    source_t = pd.to_numeric(original.iloc[ix]["treatment"], errors="coerce").to_numpy()
                    source_y = pd.to_numeric(original.iloc[ix]["spend"], errors="coerce").to_numpy()
                    row_t = pd.to_numeric(part["treatment"], errors="coerce").to_numpy()
                    row_y = pd.to_numeric(part["observed_outcome"], errors="coerce").to_numpy()
                    print("Spend target: treatment matches at same positional index:",
                          int(np.isclose(source_t, row_t, equal_nan=True).sum()), "/", len(part))
                    print("Spend target: outcome matches at same positional index:",
                          int(np.isclose(source_y, row_y, equal_nan=True).sum()), "/", len(part))
        else:
            print("Original data lacks expected treatment/spend columns; row-index cross-check skipped.")
    else:
        print("Original data lacks expected treatment/spend columns; row-index cross-check skipped.")
else:
    print("Filtered original dataset not found; row-index cross-check skipped.")

print("\n" + "=" * 72)
print("INTERPRETATION NOTES")
print("=" * 72)
print("- This audit can validate saved-row consistency and overlap, but cannot prove a model excluded each test customer during training.")
print("- Confirming out-of-sample prediction requires inspecting the training/splitting code that generated the CSV.")
print("- Pairwise overlap is expected for independently generated random test splits; the question is whether the split design and inference account for it.")
print("- This script does not compute confidence intervals.")
