
"""Run reproducible hyperparameter tuning experiments."""

import argparse
from pathlib import Path

import pandas as pd

from src.dgp import SCENARIOS, simulate
from src.tuning import run_tuning_experiment


def run_experiment(
    scenarios=None,
    n=5000,
    zero_rate=0.9,
    sigma=0.8,
    e=0.5,
    seeds=(0,),
    models=("L2", "Tweedie", "Hurdle"),
    n_candidates=12,
    output="results/tuning_results.csv",
):
    """Tune outcome models and save candidate-level and summary results."""
    scenarios = list(SCENARIOS if scenarios is None else scenarios)
    seeds = list(seeds)
    models = list(models)

    if not scenarios:
        raise ValueError("At least one scenario is required.")
    if not seeds:
        raise ValueError("At least one seed is required.")
    if not models:
        raise ValueError("At least one model is required.")

    unknown = set(scenarios) - set(SCENARIOS)
    if unknown:
        raise ValueError(f"Unknown scenarios: {sorted(unknown)}")

    if any(model not in ("L2", "Tweedie", "Hurdle") for model in models):
        raise ValueError("Models must be L2, Tweedie, or Hurdle.")

    summary_rows = []
    history_rows = []

    for scenario in scenarios:
        for seed in seeds:
            X, W, y, _ = simulate(
                n=n,
                scenario=scenario,
                zero_rate=zero_rate,
                sigma=sigma,
                e=e,
                seed=seed,
            )

            for model_name in models:
                print(
                    f"Tuning model={model_name}, scenario={scenario}, "
                    f"seed={seed}, candidates={n_candidates}"
                )

                try:
                    result = run_tuning_experiment(
                        X=X,
                        W=W,
                        y=y,
                        model_name=model_name,
                        n_candidates=n_candidates,
                        seed=seed,
                    )

                    summary_rows.append({
                        "scenario": scenario,
                        "seed": seed,
                        "model": model_name,
                        "n": n,
                        "zero_rate": zero_rate,
                        "sigma": sigma,
                        "propensity": e,
                        "requested_candidates": n_candidates,
                        "successful_candidates":
                            result["successful_candidates"],
                        "failed_candidates":
                            result["failed_candidates"],
                        "best_params": repr(result["best_params"]),
                        "validation_rmse":
                            result["best_validation_rmse"],
                        "test_rmse": result["test_rmse"],
                        "n_train": result["n_train"],
                        "n_validation": result["n_validation"],
                        "n_test": result["n_test"],
                        "status": "ok",
                        "error": "",
                    })

                    for row in result["history"]:
                        history_rows.append({
                            "scenario": scenario,
                            "seed": seed,
                            "n": n,
                            "model": model_name,
                            **row,
                        })

                except Exception as exc:
                    summary_rows.append({
                        "scenario": scenario,
                        "seed": seed,
                        "model": model_name,
                        "n": n,
                        "zero_rate": zero_rate,
                        "sigma": sigma,
                        "propensity": e,
                        "requested_candidates": n_candidates,
                        "successful_candidates": 0,
                        "failed_candidates": n_candidates,
                        "best_params": "",
                        "validation_rmse": float("nan"),
                        "test_rmse": float("nan"),
                        "n_train": float("nan"),
                        "n_validation": float("nan"),
                        "n_test": float("nan"),
                        "status": "failed",
                        "error": f"{type(exc).__name__}: {exc}",
                    })
                    print(f"  FAILED: {type(exc).__name__}: {exc}")

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_path, index=False)

    history_path = output_path.with_name(
        f"{output_path.stem}_history.csv"
    )
    pd.DataFrame(history_rows).to_csv(history_path, index=False)

    print("\nTuning summary:")
    print(summary.to_string(index=False))
    print(f"\nSummary saved to: {output_path}")
    print(f"Candidate history saved to: {history_path}")
    print(f"Successful experiments: {(summary['status'] == 'ok').sum()}")
    print(f"Failed experiments: {(summary['status'] == 'failed').sum()}")

    return summary, pd.DataFrame(history_rows)


def main():
    parser = argparse.ArgumentParser(
        description="Tune outcome models on simulated data."
    )
    parser.add_argument("--n", type=int, default=5000)
    parser.add_argument("--zero-rate", type=float, default=0.9)
    parser.add_argument("--sigma", type=float, default=0.8)
    parser.add_argument("--e", type=float, default=0.5)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument(
        "--scenarios",
        nargs="+",
        choices=list(SCENARIOS),
        default=["opposed"],
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["L2", "Tweedie", "Hurdle"],
        default=["L2", "Tweedie", "Hurdle"],
    )
    parser.add_argument("--candidates", type=int, default=12)
    parser.add_argument(
        "--output",
        default="results/tuning_results.csv",
    )
    args = parser.parse_args()

    if args.n < 20:
        parser.error("--n must be at least 20.")
    if args.candidates < 1:
        parser.error("--candidates must be positive.")

    run_experiment(
        scenarios=args.scenarios,
        n=args.n,
        zero_rate=args.zero_rate,
        sigma=args.sigma,
        e=args.e,
        seeds=args.seeds,
        models=args.models,
        n_candidates=args.candidates,
        output=args.output,
    )


if __name__ == "__main__":
    main()
