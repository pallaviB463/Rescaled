import os, tempfile
import numpy as np, pandas as pd
import run_real


def fake_hillstrom(path, n=3000, seed=0):
    rng = np.random.default_rng(seed)
    seg = rng.choice(["Mens E-Mail", "Womens E-Mail", "No E-Mail"], n)
    df = pd.DataFrame(dict(recency=rng.integers(1, 13, n), history_segment=rng.choice(["1) $0 - $100", "2) $100 - $200"], n),
                           history=rng.gamma(2, 100, n), mens=rng.integers(0, 2, n), womens=rng.integers(0, 2, n),
                           zip_code=rng.choice(["Urban", "Rural", "Surburban"], n), newbie=rng.integers(0, 2, n),
                           channel=rng.choice(["Web", "Phone", "Multichannel"], n), segment=seg))
    buy = rng.random(n) < (0.04 + 0.03 * (seg == "Mens E-Mail"))
    df["visit"] = 0; df["conversion"] = buy.astype(int)
    df["spend"] = buy * rng.gamma(4, 25, n)
    df.to_csv(path, index=False)


def test_hillstrom_pipeline_runs():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "h.csv"); fake_hillstrom(p)
        X, W, y, e = run_real.load_hillstrom(p, "mens")
        assert 0.4 < e < 0.6 and X.shape[0] == len(y) == len(W)      # two arms only, e computed from data
        df = run_real.run_hillstrom(p, "mens", repeats=1, scales=(1.0,), count_scales=(1.0,),
                                    learners=("T",), count_models=("Hurdle",), folds=3)
        assert {"rho_vs_a1", "qini", "ipw_value_k20", "backend"} <= set(df.columns)
        assert (df.rho_vs_a1 > 0.999).all()                              # only a = 1 here: stability vs itself
