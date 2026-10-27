"""run_sim.py  (OWNER: Member Q). Grid-driven, parallel, resumable experiment runner.
  python src/run_sim.py config/grid_pilot.yaml results/pilot --jobs 4
  python src/run_sim.py config/grid.yaml results/main --jobs 8 --tier must
One file per replication, so a crash loses at most the replications in flight; re-running skips finished ones.
Common random numbers: the SAME simulated dataset feeds every transform, scale and learner in a replication."""
import argparse, os, glob, itertools, time
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))  # silences a harmless joblib warning on Windows
import numpy as np, pandas as pd, yaml
from joblib import Parallel, delayed

from dgp import simulate
from transforms import apply_transform, SCALES
import models
from models import L2, Poisson, Tweedie, Hurdle
from metalearners import LEARNERS
from metrics import rank_corr, topk_overlap, policy_value, regret

COUNT_AWARE = {"Poisson": Poisson, "Tweedie": Tweedie, "Hurdle": Hurdle}
try:
    import pyarrow  # noqa
    EXT = "parquet"
except ImportError:
    EXT = "csv"  # fallback; install pyarrow for the real runs


def pipelines(scales, count_scales):
    """Yield (family, transform, scale, ModelClass). One pipeline = how the analyst prepares y and models it."""
    for t in ["log1p", "arcsinh", "sqrt"]:
        for a in scales: yield "TransformedL2", t, a, L2
    yield "TransformedL2", "identity", 1.0, L2   # untransformed least squares
    yield "TransformedL2", "rank", 1.0, L2       # unit-invariant control
    for name, M in COUNT_AWARE.items():
        for a in count_scales: yield name, "level", a, M   # level-scale; a only changes units


def one_rep(scenario, zero_rate, n, seed, scales=SCALES, count_scales=(0.01, 1.0, 100.0),
            learners=("T", "DR"), k=0.2, e=0.5):
    assert 1.0 in scales and 1.0 in count_scales, "a = 1 is the reference for rho_vs_a1"
    X, W, Y, T = simulate(n, scenario, zero_rate, e=e, seed=seed)
    Xt, _, _, Tt = simulate(n, scenario, zero_rate, e=e, seed=seed + 10_000)  # fresh test set with known truth
    rows, store = [], {}
    for fam, t, a, M in pipelines(scales, count_scales):
        y = Y * a if t == "level" else apply_transform(t, Y, a)
        for L in learners:
            tau = LEARNERS[L](M, X, W, y, Xt, e=e)
            store[(fam, t, a, L)] = tau
            rows.append(dict(scenario=scenario, zero_rate=zero_rate, n=n, seed=seed, family=fam, transform=t,
                             scale=a, learner=L, backend=models.BACKEND, mean_tau_hat=float(np.mean(tau)),
                             rho_level=rank_corr(tau, Tt["tau_level"]),
                             value=policy_value(tau, Tt["tau_level"], k), regret=regret(tau, Tt["tau_level"], k)))
    for r in rows:  # ranking stability: compare each scale with the a = 1 version of the same pipeline
        ref = store[(r["family"], r["transform"], 1.0, r["learner"])]
        tau = store[(r["family"], r["transform"], r["scale"], r["learner"])]
        r["rho_vs_a1"] = rank_corr(tau, ref)
        r["top20_overlap_vs_a1"] = topk_overlap(tau, ref, k)
    return rows


def _path(out_dir, scenario, zr, n, seed, ext=None):
    return os.path.join(out_dir, f"{scenario}_zr{int(round(zr*100))}_n{n}_seed{seed}.{ext or EXT}")


def _done(out_dir, scenario, zr, n, seed):
    return any(os.path.exists(_path(out_dir, scenario, zr, n, seed, x)) for x in ("parquet", "csv"))


def _task(cfg, out_dir, scenario, zr, seed):
    n = cfg["n"]
    rows = one_rep(scenario, zr, n, seed, scales=cfg["scales"], count_scales=cfg["count_scales"],
                   learners=tuple(cfg["learners"]), k=cfg.get("k", 0.2))
    df, p = pd.DataFrame(rows), _path(out_dir, scenario, zr, n, seed)
    tmp = p + ".tmp"
    (df.to_parquet(tmp, index=False) if EXT == "parquet" else df.to_csv(tmp, index=False))
    os.replace(tmp, p)  # atomic: a half-written file is never mistaken for a finished one
    return p


def grid_tasks(cfg, out_dir, tier="all"):
    tasks = []
    for cell in cfg["cells"]:
        if tier != "all" and cell.get("tier", "must") != tier: continue
        scs = ["null" if x is None else x for x in cell["scenarios"]]   # YAML reads bare null as None
        for sc, zr in itertools.product(scs, cell["zero_rates"]):
            for seed in range(cell["reps"]):
                if not _done(out_dir, sc, zr, cfg["n"], seed): tasks.append((sc, zr, seed))
    return tasks


def run_grid(cfg_path, out_dir, jobs=1, tier="all"):
    cfg = yaml.safe_load(open(cfg_path))
    os.makedirs(out_dir, exist_ok=True)
    tasks = grid_tasks(cfg, out_dir, tier)
    print(f"{len(tasks)} replications to run (finished ones are skipped); backend={models.BACKEND}; files={EXT}")
    t0 = time.time()
    Parallel(n_jobs=jobs, verbose=5)(delayed(_task)(cfg, out_dir, *t) for t in tasks)
    print(f"done in {(time.time()-t0)/60:.1f} min")


def aggregate(out_dir):
    fs = glob.glob(os.path.join(out_dir, "*.parquet")) + glob.glob(os.path.join(out_dir, "*.csv"))
    rd = lambda f: pd.read_parquet(f) if f.endswith(".parquet") else pd.read_csv(f)
    return pd.concat([rd(f) for f in fs], ignore_index=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("config"); ap.add_argument("out_dir")
    ap.add_argument("--jobs", type=int, default=1); ap.add_argument("--tier", default="all")
    a = ap.parse_args()
    run_grid(a.config, a.out_dir, a.jobs, a.tier)
