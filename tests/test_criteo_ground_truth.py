
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "processed" / "criteo_semisynthetic_200k.csv"


def test_level_cate_matches_dgp_equation():
    df = pd.read_csv(DATA_FILE)

    sigma = 0.8
    mu0 = 2.5 + 0.5 * df["f2"].to_numpy()

    # Reconstruct the opposed-scenario treatment shifts.
    z3 = df["f3"].to_numpy()
    z4 = df["f4"].to_numpy()

    de = 0.3 + 0.8 * z3
    di = 0.2 - 0.4 * z3

    # Reconstruct the calibrated baseline intercept from stored p0.
    shape = 0.8 * df["f0"].to_numpy() - 0.5 * df["f1"].to_numpy()
    p0 = df["p0"].to_numpy()
    b0 = np.log(p0 / (1 - p0)) - shape

    p1 = 1 / (1 + np.exp(-(b0 + shape + de)))
    mu1 = mu0 + di

    expected = (
        p1 * np.exp(mu1 + sigma**2 / 2)
        - p0 * np.exp(mu0 + sigma**2 / 2)
    )

    actual = df["tau_level"].to_numpy()

    np.testing.assert_allclose(actual, expected, rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(df["p1"].to_numpy(), p1, rtol=1e-8, atol=1e-8)
