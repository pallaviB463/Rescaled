
"""Validate the Criteo semi-synthetic dataset for P12."""

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "processed" / "criteo_semisynthetic_200k.csv"
MANIFEST_FILE = ROOT / "data" / "processed" / "criteo_semisynthetic_manifest.json"


def main():
    assert DATA_FILE.exists(), f"Missing dataset: {DATA_FILE}"
    assert MANIFEST_FILE.exists(), f"Missing manifest: {MANIFEST_FILE}"

    df = pd.read_csv(DATA_FILE)
    manifest = json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))

    features = [f"f{i}" for i in range(12)]
    required = features + [
        "W", "Y", "tau_level", "p0", "p1"
    ]

    assert len(df) == 200_000, f"Unexpected row count: {len(df)}"
    assert all(c in df.columns for c in required), "Missing required columns"
    assert df[required].notna().all().all(), "Missing required values"
    assert np.isfinite(df[required].to_numpy(dtype=float)).all(), "Non-finite values"
    assert set(df["W"].unique()).issubset({0, 1}), "Treatment must be binary"
    assert (df["Y"] >= 0).all(), "Outcomes must be nonnegative"
    assert df["p0"].between(0, 1).all(), "Invalid p0 probabilities"
    assert df["p1"].between(0, 1).all(), "Invalid p1 probabilities"
    assert manifest["scenario"] == "opposed", "Unexpected scenario"
    assert manifest["seed"] == 2026, "Unexpected seed"

    # Under randomized treatment, observed purchase rates should track
    # the mean conditional purchase probabilities.
    for treatment, probability_column in [(0, "p0"), (1, "p1")]:
        group = df[df["W"] == treatment]
        observed_rate = (group["Y"] > 0).mean()
        expected_rate = group[probability_column].mean()

        print(
            f"W={treatment}: observed purchase rate={observed_rate:.5f}, "
            f"mean predicted probability={expected_rate:.5f}"
        )

        # Allow a conservative tolerance for finite-sample variation.
        assert abs(observed_rate - expected_rate) < 0.015, (
            f"Purchase rate mismatch for W={treatment}"
        )

    print("\nP12 structural validation passed.")
    print(f"Rows: {len(df):,}")
    print(f"Feature count: {len(features)}")
    print(f"Treatment rate: {df['W'].mean():.4f}")
    print(f"Zero-outcome rate: {(df['Y'] == 0).mean():.4f}")
    print(f"Duplicate complete records: {df.duplicated().sum():,}")
    print(f"Output: {DATA_FILE}")


if __name__ == "__main__":
    main()
