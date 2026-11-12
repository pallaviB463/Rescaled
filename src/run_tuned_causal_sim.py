
"""Evaluate tuned outcome models on independent causal test populations."""

import argparse
import ast
from pathlib import Path

import numpy as np
import pandas as pd

from src.dgp import SCENARIOS, simulate
from src.metrics import rank_corr, topk_overlap, policy_value, regret
from src.models import L2, Tweedie, Hurdle

MODEL_CLASSES = {"L2": L2, "Tweedie": Tweedie, "Hurdle": Hurdle}


def make_model(model_name, params):
    """Create a fresh outcome model from selected hyperparameters."""
    params = dict(params)

    if model_name == "Tweedie":
        power = params.pop("power")
        return Tweedie(power=power, **params)

    return MODEL_CLASSES[model_name](**params)


def load_best_configs(history_path):
    """Select minimum-validation-RMSE candidate per scenario, seed, model."""
    history = pd.read_csv(history_path, keep_default_na=False)

    required = {
        "scenario", "seed", "model", "parameters",
        "validation_rmse", "status",
    }
    missing = required - set(history.columns)
    if missing:
        raise ValueError(f"History file is missing columns: {sorted(missing)}")

    history["validation_rmse"] = pd.to_numeric(
        history["validation_rmse"], errors="coerce"
    )
    successful = history[
        (history["status"] == "ok")
        & np.isfinite(history["validation_rmse"])
    ].copy()

    if successful.empty:
        raise ValueError("No successful tuning candidates were found.")

    best_indices = successful.groupby(
        ["scenario", "seed", "model"]
    )["validation_rmse"].idxmin()

    best = successful.loc[best_indices].copy()
    best["best_params"] = best["parameters"].map(ast.literal_eval)
    return best


def evaluate_one(scenario, seed, model_name, params, n=5000,
                 zero_rate=0.9, sigma=0.8, e=0.5):
    """Fit tuned T-learner arm models and evaluate on an independent population."""
    X, W, y, _ = simulate(
        n=n, scenario=scenario, zero_rate=zero_rate,
        sigma=sigma, e=e, seed=seed,
    )

    # Match run_sim.py's independent evaluation-population construction.
    X_test, _, _, truth_test = simulate(
        n=max(2000, n // 2),
        scenario=scenario,
        zero_rate=zero_rate,
        sigma=sigma,
        e=e,
        seed=seed + 1_000_003,
    )
    tau_true = np.asarray(truth_test["tau_level"], dtype=float).ravel()

    predictions = np.empty(len(X_test), dtype=float)

    for arm in (0, 1):
        mask = W == arm
        model = make_model(model_name, params)
        model.fit(X[mask], y[mask])
        predictions_arm = model.predict(X_test)
        if arm == 0:
            mu0 = np.asarray(predictions_arm, dtype=float).ravel()
        else:
            mu1 = np.asarray(predictions_arm, dtype=float).ravel()

    tau_hat = mu1 - mu0

    if not np.isfinite(tau_hat).all():
        raise ValueError("Predicted treatment effects contain non-finite values.")

    if tau_true.size < 2 or np.ptp(tau_true) == 0:
        correlation = np.nan
        overlap = np.nan
    else:
        correlation = rank_corr(tau_hat, tau_true)
        overlap = topk_overlap(tau_hat, tau_true, k=0.2)

    return {
        "scenario": scenario,
        "seed": seed,
        "model": model_name,
        "configuration": "tuned",
        "best_params": repr(params),
        "rank_corr": correlation,
        "topk_overlap": overlap,
        "policy_value": policy_value(tau_hat, tau_true, k=0.2),
        "regret": regret(tau_hat, tau_true, k=0.2),
        "status": "ok",
        "error": "",
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate tuned outcome models on causal metrics."
    )
    parser.add_argument(
        "--history",
        default="results/tuning_grid_seeds_0_4_history.csv",
    )
    parser.add_argument("--n", type=int, default=5000)
    parser.add_argument("--zero-rate", type=float, default=0.9)
    parser.add_argument("--sigma", type=float, default=0.8)
    parser.add_argument("--e", type=float, default=0.5)
    parser.add_argument("--output", default="results/causal_tuned.csv")
    args = parser.parse_args()

    best = load_best_configs(args.history)
    rows = []

    for _, candidate in best.iterrows():
        scenario = candidate["scenario"]
        seed = int(candidate["seed"])
        model_name = candidate["model"]
        params = candidate["best_params"]

        try:
            result = evaluate_one(
                scenario, seed, model_name, params,
                n=args.n,
                zero_rate=args.zero_rate,
                sigma=args.sigma,
                e=args.e,
            )
        except Exception as exc:
            result = {
                "scenario": scenario,
                "seed": seed,
                "model": model_name,
                "configuration": "tuned",
                "best_params": repr(params),
                "rank_corr": np.nan,
                "topk_overlap": np.nan,
                "policy_value": np.nan,
                "regret": np.nan,
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            }
        rows.append(result)

    results = pd.DataFrame(rows)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output, index=False)

    successful = results[results["status"] == "ok"]
    summary = (
        successful.groupby(["scenario", "model"])[
            ["rank_corr", "topk_overlap", "policy_value", "regret"]
        ]
        .agg(["mean", "std"])
        .round(4)
    )
    summary_path = output.with_name(f"{output.stem}_summary.csv")
    summary.to_csv(summary_path)

    print("\nTuned causal evaluation")
    print("Successful rows:", len(successful))
    print("Failed rows:", len(results) - len(successful))
    print(summary.to_string())
    print(f"\nDetailed results: {output}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
