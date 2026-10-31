
"""Compare transformed-outcome and naive back-transform baselines."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.dgp import simulate
from src.metrics import rank_corr, topk_overlap, policy_value, regret
from src.metalearners import (
    TransformedOutcomeLearner,
    BackTransformLearner,
)
from src.models import L2


DEFAULT_SEEDS = (42, 123, 2026, 2027, 2028)


def _rmse(y_true, y_pred):
    """Calculate root mean squared prediction error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def run_backtransform_ablation(
    scenario="opposed",
    n=5000,
    zero_rate=0.9,
    sigma=0.8,
    e=0.5,
    seeds=DEFAULT_SEEDS,
    test_size=0.2,
    output="results/p13_backtransform_ablation.csv",
):
    """Compare both learners on identical train/test splits."""
    if n < 100:
        raise ValueError("n must be at least 100.")

    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")

    if not seeds:
        raise ValueError("At least one seed is required.")

    rows = []

    for seed in seeds:
        print(f"Running P13 comparison: scenario={scenario}, seed={seed}")

        X, W, y, truth = simulate(
            n=n,
            scenario=scenario,
            zero_rate=zero_rate,
            sigma=sigma,
            e=e,
            seed=seed,
        )

        indices = np.arange(n)

        train_idx, test_idx = train_test_split(
            indices,
            test_size=test_size,
            random_state=seed,
            stratify=W,
        )

        X_train, X_test = X[train_idx], X[test_idx]
        W_train = W[train_idx]
        y_train = y[train_idx]

        tau_true = truth["tau_level"][test_idx]

        learners = {
            "TransformedOutcome": TransformedOutcomeLearner(
                transform="log1p",
                scale=1.0,
                model=L2(),
                e=e,
            ),
            "BackTransform": BackTransformLearner(
                model=L2(),
                e=e,
            ),
        }

        oracle_value = policy_value(
            tau_true,
            tau_true,
            k=0.2,
        )

        for method, learner in learners.items():
            learner.fit(X_train, W_train, y_train)
            tau_pred = learner.predict(X_test)

            rows.append({
                "scenario": scenario,
                "seed": seed,
                "method": method,
                "n": n,
                "zero_rate_parameter": zero_rate,
                "sigma": sigma,
                "propensity": e,
                "train_size": len(train_idx),
                "test_size": len(test_idx),
                "cate_rmse": _rmse(tau_true, tau_pred),
                "rank_corr": rank_corr(tau_pred, tau_true),
                "top20_overlap": topk_overlap(
                    tau_pred,
                    tau_true,
                    k=0.2,
                ),
                "policy_value": policy_value(
                    tau_pred,
                    tau_true,
                    k=0.2,
                ),
                "oracle_policy_value": oracle_value,
                "regret": regret(
                    tau_pred,
                    tau_true,
                    k=0.2,
                ),
                "negative_prediction_fraction": float(
                    np.mean(tau_pred < 0)
                ),
                "prediction_mean": float(np.mean(tau_pred)),
                "prediction_std": float(np.std(tau_pred)),
                "status": "ok",
            })

    results = pd.DataFrame(rows)

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)

    summary = (
        results.groupby("method")[
            [
                "cate_rmse",
                "rank_corr",
                "top20_overlap",
                "policy_value",
                "oracle_policy_value",
                "regret",
                "negative_prediction_fraction",
            ]
        ]
        .agg(["mean", "std"])
    )

    print("\nP13 results by seed:")
    print(results.to_string(index=False))

    print("\nP13 aggregate summary:")
    print(summary.to_string())

    print(f"\nResults saved to: {output_path}")

    return results, summary


def main():
    parser = argparse.ArgumentParser(
        description="Run the P13 back-transform baseline ablation."
    )

    parser.add_argument("--scenario", default="opposed")
    parser.add_argument("--n", type=int, default=5000)
    parser.add_argument("--zero-rate", type=float, default=0.9)
    parser.add_argument("--sigma", type=float, default=0.8)
    parser.add_argument("--e", type=float, default=0.5)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_SEEDS),
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument(
        "--output",
        default="results/p13_backtransform_ablation.csv",
    )

    args = parser.parse_args()

    run_backtransform_ablation(
        scenario=args.scenario,
        n=args.n,
        zero_rate=args.zero_rate,
        sigma=args.sigma,
        e=args.e,
        seeds=args.seeds,
        test_size=args.test_size,
        output=args.output,
    )


if __name__ == "__main__":
    main()
