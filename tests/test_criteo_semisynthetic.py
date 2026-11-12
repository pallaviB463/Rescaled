
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "processed" / "criteo_semisynthetic_200k.csv"
MANIFEST_FILE = ROOT / "data" / "processed" / "criteo_semisynthetic_manifest.json"

FEATURES = [f"f{i}" for i in range(12)]
RAW_FEATURES = [f"raw_f{i}" for i in range(12)]


def load_dataset():
    assert DATA_FILE.exists(), "Run the Criteo semi-synthetic preparation script first."
    return pd.read_csv(DATA_FILE)


def test_dataset_has_expected_shape_and_columns():
    df = load_dataset()

    assert len(df) == 200_000
    assert all(c in df.columns for c in FEATURES + RAW_FEATURES)
    assert all(c in df.columns for c in ["W", "Y", "tau_level", "p0", "p1"])


def test_dataset_has_no_missing_or_nonfinite_values():
    df = load_dataset()
    numeric = df.select_dtypes(include=[np.number]).to_numpy()

    assert np.isfinite(numeric).all()
    assert not df.isna().any().any()


def test_treatment_and_outcomes_are_valid():
    df = load_dataset()

    assert set(df["W"].unique()).issubset({0, 1})
    assert 0.45 < df["W"].mean() < 0.55
    assert (df["Y"] >= 0).all()


def test_ground_truth_probabilities_are_valid():
    df = load_dataset()

    assert df["p0"].between(0, 1).all()
    assert df["p1"].between(0, 1).all()
    assert np.isfinite(df["tau_level"]).all()


def test_observed_purchase_rates_match_predicted_probabilities():
    df = load_dataset()

    for treatment, column in [(0, "p0"), (1, "p1")]:
        group = df[df["W"] == treatment]
        observed_rate = (group["Y"] > 0).mean()
        expected_rate = group[column].mean()

        assert abs(observed_rate - expected_rate) < 0.015


def test_standardized_features_have_approximately_zero_mean_and_unit_sd():
    df = load_dataset()
    X = df[FEATURES]

    assert np.all(np.abs(X.mean().to_numpy()) < 1e-8)
    assert np.all(np.abs(X.std(ddof=0).to_numpy() - 1) < 1e-8)


def test_manifest_records_reproducible_generation_settings():
    assert MANIFEST_FILE.exists()
    manifest = json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))

    assert manifest["n_rows"] == 200_000
    assert manifest["scenario"] == "opposed"
    assert manifest["seed"] == 2026
    assert manifest["original_treatment_and_outcomes_used"] is False
