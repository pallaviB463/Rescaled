
"""Run Neurolytics simulation experiments."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.dgp import SCENARIOS, simulate, true_transformed_cate
from src.metrics import rank_corr, topk_overlap, policy_value, regret
from src.models import L2
from src.transforms import SCALES, TRANSFORMS
from src.metalearners import (
    SLearner,
    TLearner,
    XLearner,
    DRLearner,
    TransformedOutcomeLearner,
)


LEARNERS = {
    "S": SLearner,
    "T": TLearner,
    "X": XLearner,
    "DR": DRLearner,
}

OUTPUT_DIR = Path("results")


def pipelines(scales=None, transforms=None):
    """Build the transformation and scale configurations."""
    scales = list(SCALES if scales is None else scales)
    transforms = list(TRANSFORMS if transforms is None else transforms)

    if 1.0 not in scales:
        raise ValueError("Scales must include the reference scale 1.0.")

    configurations = []

    for transform in transforms:
        for scale in scales:
            configurations.append({
                "transform": transform,
                "scale": float(scale),
            })

    # Rank transformation is scale invariant. Keep one reference scale.
    configurations.append({
        "transform": "rank",
        "scale": 1.0,
    })

    return configurations


def _make_model():
    """Create a fresh base learner model."""
    return L2(
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=15,
        min_child_samples=50,
    )


def _fit_predict_learner(name, X, W, y, X_test, e):
    """Fit a learner and return its estimated treatment effects."""
    learner_class = LEARNERS[name]
    learner = learner_class(model=_make_model(), e=e)
    learner.fit(X, W, y)
    return np.asarray(learner.predict(X_test), dtype=float)


def one_rep(
    scenario="opposed",
    n=5000,
    zero_rate=0.9,
    sigma=0.8,
    e=0.5,
    seed=0,
    scales=None,
    learners=("T",),
    include_transformed=True,
):
    """Run one simulation replicate and return a results DataFrame."""
    if 1.0 not in (SCALES if scales is None else scales):
        raise ValueError("Scales must include the reference scale 1.0.")

    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario}")

    if not 0 < e < 1:
        raise ValueError("e must be strictly between 0 and 1.")

    rng = np.random.default_rng(seed)

    X, W, y, truth = simulate(
        n=n,
        scenario=scenario,
        zero_rate=zero_rate,
        sigma=sigma,
        e=e,
        seed=seed,
    )

    # Use an independent test set to assess generalization.
    X_test, _, _, truth_test = simulate(
        n=max(2000, n // 2),
        scenario=scenario,
        zero_rate=zero_rate,
        sigma=sigma,
        e=e,
        seed=seed + 1_000_003,
    )

    tau_true = truth_test["tau_level"]
    rows = []

    for learner_name in learners:
        if learner_name not in LEARNERS:
            raise ValueError(f"Unknown learner: {learner_name}")

        try:
            tau_hat = _fit_predict_learner(
                learner_name, X, W, y, X_test, e
            )

            rows.append({
                "scenario": scenario,
                "learner": learner_name,
                "transform": "level",
                "scale": 1.0,
                "rank_corr": rank_corr(tau_hat, tau_true),
                "topk_overlap": topk_overlap(tau_hat, tau_true, k=0.2),
                "policy_value": policy_value(tau_hat, tau_true, k=0.2),
                "regret": regret(tau_hat, tau_true, k=0.2),
                "status": "ok",
                "error": "",
            })
        except (ValueError, RuntimeError) as exc:
            rows.append({
                "scenario": scenario,
                "learner": learner_name,
                "transform": "level",
                "scale": 1.0,
                "rank_corr": np.nan,
                "topk_overlap": np.nan,
                "policy_value": np.nan,
                "regret": np.nan,
                "status": "failed",
                "error": str(exc),
            })

    if include_transformed:
        for configuration in pipelines(scales=scales):
            transform = configuration["transform"]
            scale = configuration["scale"]

            try:
                tau_transformed_true = true_transformed_cate(
                    truth_test,
                    m=lambda z: _transform_values(transform, z),
                    a=scale,
                )

                learner = TransformedOutcomeLearner(
                    transform=transform,
                    scale=scale,
                    model=_make_model(),
                    e=e,
                )
                learner.fit(X, W, y)
                tau_hat = np.asarray(learner.predict(X_test), dtype=float)

                rows.append({
                    "scenario": scenario,
                    "learner": "TransformedOutcome",
                    "transform": transform,
                    "scale": scale,
                    "rank_corr": rank_corr(
                        tau_hat, tau_transformed_true
                    ),
                    "topk_overlap": topk_overlap(
                        tau_hat, tau_transformed_true, k=0.2
                    ),
                    "policy_value": policy_value(
                        tau_hat, tau_true, k=0.2
                    ),
                    "regret": regret(tau_hat, tau_true, k=0.2),
                    "status": "ok",
                    "error": "",
                })
            except (ValueError, RuntimeError) as exc:
                rows.append({
                    "scenario": scenario,
                    "learner": "TransformedOutcome",
                    "transform": transform,
                    "scale": scale,
                    "rank_corr": np.nan,
                    "topk_overlap": np.nan,
                    "policy_value": np.nan,
                    "regret": np.nan,
                    "status": "failed",
                    "error": str(exc),
                })

    return pd.DataFrame(rows)


def _transform_values(name, values):
    """Apply a transformation to values for quadrature ground truth."""
    values = np.asarray(values, dtype=float)

    if name == "rank":
        # Population rank transformation is not pointwise identifiable
        # from a finite test sample. Use the rank-CATE fallback only
        # for a finite, sample-relative approximation.
        from scipy.stats import rankdata
        return rankdata(values, method="average") / len(values)

    return TRANSFORMS[name](values)


def run_experiment(
    scenarios=None,
    n=5000,
    zero_rate=0.9,
    sigma=0.8,
    e=0.5,
    seeds=(0,),
    scales=None,
    learners=("T",),
    include_transformed=True,
    output_path=None,
):
    """Run multiple scenarios and seeds, then save CSV results."""
    scenarios = list(SCENARIOS if scenarios is None else scenarios)
    seeds = list(seeds)

    if not seeds:
        raise ValueError("At least one seed is required.")

    if 1.0 not in (SCALES if scales is None else scales):
        raise ValueError("Scales must include the reference scale 1.0.")

    frames = []

    for scenario in scenarios:
        for seed in seeds:
            frame = one_rep(
                scenario=scenario,
                n=n,
                zero_rate=zero_rate,
                sigma=sigma,
                e=e,
                seed=seed,
                scales=scales,
                learners=learners,
                include_transformed=include_transformed,
            )
            frame["seed"] = seed
            frame["n"] = n
            frames.append(frame)

    results = pd.concat(frames, ignore_index=True)

    destination = Path(output_path) if output_path else (
        OUTPUT_DIR / "simulation_results.csv"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(destination, index=False)

    summary = (
        results[results["status"] == "ok"]
        .groupby(["scenario", "learner", "transform", "scale"], dropna=False)
        [["rank_corr", "topk_overlap", "policy_value", "regret"]]
        .mean()
        .reset_index()
    )

    summary_path = destination.with_name(
        f"{destination.stem}_summary.csv"
    )
    summary.to_csv(summary_path, index=False)

    return results, summary


def main():
    parser = argparse.ArgumentParser(
        description="Run Neurolytics treatment-effect simulations."
    )
    parser.add_argument("--n", type=int, default=5000)
    parser.add_argument("--zero-rate", type=float, default=0.9)
    parser.add_argument("--sigma", type=float, default=0.8)
    parser.add_argument("--e", type=float, default=0.5)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=SCENARIOS,
        default=["opposed"],
    )
    parser.add_argument(
        "--learners",
        nargs="+",
        choices=list(LEARNERS),
        default=["T"],
    )
    parser.add_argument(
        "--no-transformed",
        action="store_true",
        help="Skip transformed-outcome learner experiments.",
    )
    parser.add_argument(
        "--output",
        default="results/simulation_results.csv",
    )

    args = parser.parse_args()

    results, summary = run_experiment(
        scenarios=args.scenarios,
        n=args.n,
        zero_rate=args.zero_rate,
        sigma=args.sigma,
        e=args.e,
        seeds=args.seeds,
        learners=args.learners,
        include_transformed=not args.no_transformed,
        output_path=args.output,
    )

    print("\nSimulation results:")
    print(results.head(10).to_string(index=False))
    print(f"\nFull results saved to: {args.output}")

    summary_path = Path(args.output).with_name(
        f"{Path(args.output).stem}_summary.csv"
    )
    print(f"Summary saved to: {summary_path}")
    print(f"Successful rows: {(results['status'] == 'ok').sum()}")
    print(f"Failed rows: {(results['status'] == 'failed').sum()}")


if __name__ == "__main__":
    main()
