import os, tempfile, yaml
import numpy as np
import run_sim


def test_one_rep_columns_and_controls():
    rows = run_sim.one_rep("opposed", 0.9, 2500, seed=0, scales=[0.01, 1.0, 100.0],
                           count_scales=[0.01, 1.0, 100.0], learners=("T",))
    import pandas as pd
    df = pd.DataFrame(rows)
    need = {"scenario", "zero_rate", "n", "seed", "family", "transform", "scale", "learner", "backend",
            "rho_level", "value", "regret", "rho_vs_a1", "top20_overlap_vs_a1"}
    assert need <= set(df.columns)
    assert (df[df.scale == 1.0].rho_vs_a1 > 0.999).all()                      # a=1 vs itself
    assert (df[df["transform"] == "rank"].rho_vs_a1 > 0.999).all()                # unit-invariant control
    assert (df[df["transform"] == "sqrt"].rho_vs_a1 > 0.99).all(), df[df["transform"] == "sqrt"][["scale", "rho_vs_a1"]]
    assert (df[df["transform"] == "level"].rho_vs_a1 > 0.95).all(), df[df["transform"] == "level"][["family", "scale", "rho_vs_a1"]]


def test_null_scenario_does_not_crash():
    rows = run_sim.one_rep("null", 0.9, 1500, seed=0, scales=[1.0], count_scales=[1.0], learners=("T",))
    assert len(rows) > 0 and all(np.isnan(r["rho_level"]) for r in rows)


def test_run_grid_is_resumable():
    cfg = dict(n=800, k=0.2, scales=[1.0], count_scales=[1.0], learners=["T"],
               cells=[dict(scenarios=["null"], zero_rates=[0.9], reps=2, tier="must")])
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "g.yaml"); yaml.safe_dump(cfg, open(p, "w"))
        out = os.path.join(d, "out")
        run_sim.run_grid(p, out, jobs=1)
        assert len(run_sim.grid_tasks(cfg, out)) == 0          # everything finished -> nothing left to do
        assert len(run_sim.aggregate(out)) > 0
