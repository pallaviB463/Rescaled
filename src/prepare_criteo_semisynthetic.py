
"""Generate a reproducible semi-synthetic Criteo experiment."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from dgp import SCENARIOS, simulate_from_X


ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = ROOT / "data" / "processed" / "criteo_covariates_200k.csv"
OUTPUT_FILE = ROOT / "data" / "processed" / "criteo_semisynthetic_200k.csv"
MANIFEST_FILE = ROOT / "data" / "processed" / "criteo_semisynthetic_manifest.json"

SCENARIO = "opposed"
ZERO_RATE = 0.90
SIGMA = 0.8
TREATMENT_PROBABILITY = 0.5
SEED = 2026


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Sample not found: {INPUT_FILE}\n"
            "Run src/prepare_criteo_sample.py first."
        )

    raw = pd.read_csv(INPUT_FILE)
    feature_names = [f"f{i}" for i in range(12)]

    if not all(name in raw.columns for name in feature_names):
        raise ValueError("Expected Criteo features f0 through f11.")

    if len(raw) != 200_000:
        raise ValueError(f"Expected 200,000 rows; found {len(raw):,}.")

    raw_features = raw[feature_names].to_numpy(dtype=float)

    if not np.isfinite(raw_features).all():
        raise ValueError("Criteo features contain missing or infinite values.")

    # Standardize real covariates to match the scale expected by the DGP.
    feature_mean = raw_features.mean(axis=0)
    feature_std = raw_features.std(axis=0)
    feature_std[feature_std == 0] = 1.0

    X = (raw_features - feature_mean) / feature_std

    # Generate randomized synthetic treatment assignment.
    rng = np.random.default_rng(SEED)
    W = rng.binomial(1, TREATMENT_PROBABILITY, size=len(X))

    # Generate synthetic outcomes and exact level-scale CATE.
    Y, truth = simulate_from_X(
        X=X,
        W=W,
        scenario=SCENARIO,
        zero_rate=ZERO_RATE,
        sigma=SIGMA,
        seed=SEED + 1,
    )

    # Standardized f0...f11 are the model inputs.
    result = pd.DataFrame(X, columns=feature_names)

    # Preserve raw features for traceability.
    for j, name in enumerate(feature_names):
        result[f"raw_{name}"] = raw_features[:, j]

    result["W"] = W
    result["Y"] = Y
    result["tau_level"] = truth["tau_level"]
    result["p0"] = truth["p0"]
    result["p1"] = truth["p1"]

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_FILE, index=False)

    manifest = {
        "dataset": "Criteo uplift covariates with synthetic treatment/outcomes",
        "input_file": str(INPUT_FILE.relative_to(ROOT)),
        "output_file": str(OUTPUT_FILE.relative_to(ROOT)),
        "n_rows": int(len(result)),
        "n_features": len(feature_names),
        "feature_columns": feature_names,
        "raw_feature_columns": [f"raw_{name}" for name in feature_names],
        "treatment_column": "W",
        "outcome_column": "Y",
        "ground_truth_column": "tau_level",
        "scenario": SCENARIO,
        "zero_rate_target": ZERO_RATE,
        "sigma": SIGMA,
        "treatment_probability": TREATMENT_PROBABILITY,
        "seed": SEED,
        "standardization": "sample mean and standard deviation",
        "original_treatment_and_outcomes_used": False,
        "note": (
            "Original Criteo treatment/conversion/visit/exposure columns "
            "were not used. Synthetic treatment and outcomes were generated "
            "from standardized real covariates."
        ),
    }

    MANIFEST_FILE.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("\nSemi-synthetic dataset generated.")
    print(f"Rows: {len(result):,}")
    print(f"Scenario: {SCENARIO}")
    print(f"Treatment rate: {result['W'].mean():.4f}")
    print(f"Observed zero-outcome rate: {(result['Y'] == 0).mean():.4f}")
    print(f"Mean outcome: {result['Y'].mean():.6f}")
    print(f"Mean true level CATE: {result['tau_level'].mean():.6f}")
    print(f"Null scenario: {SCENARIO == 'null'}")
    print(f"Output: {OUTPUT_FILE}")
    print(f"Manifest: {MANIFEST_FILE}")
    print("\nMissing values:")
    print(result.isna().sum().to_string())


if __name__ == "__main__":
    main()
