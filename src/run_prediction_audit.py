
"""Measure estimated CATE magnitudes under the P14 null scenario."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.dgp import simulate
from src.metalearners import (
    SLearner,
    TLearner,
    XLearner,
    DRLearner,
    TransformedOutcomeLearner,
    BackTransformLearner,
)
from src.transforms import SCALES


STANDARD_LEARNERS = {
    "S": SLearner,
    "T": TLearner,
    "X": XLearner,
    "DR": DRLearner,
}

TRANSFORMS = ["identity", "log1p", "arcsinh", "sqrt", "rank"]


def prediction_metrics(predictions):
    """Calculate magnitude and dispersion of predicted CATEs."""
    predictions = np.asarray(predictions, dtype=float)

    if predictions.ndim != 1 or not np.isfinite(predictions).all():
        raise ValueError("Predictions must be a finite one-dimensional array.")

    return {
        "mean_cate": float(np.mean(predictions)),
        "mean_abs_cate": float(np.mean(np.abs(predictions))),
        "cate_rmse_vs_null": float(np.sqrt(np.mean(predictions**2))),
        "cate_std": float(np.std(predictions)),
        "max_abs_cate": float(np.max(np.abs(predictions))),
        "positive_fraction": float(np.mean(predictions > 0)),
        "negative_fraction": float(np.mean(predictions < 0)),
    }


def fit_and_record(
    learner_name,
    learner,
    X_train,
    W_train,
    y_train,
    X_test,
    seed,
    rows,
    transform="level",
    scale=1.0,
):
    """Fit one learner and append its prediction-magnitude results."""
    learner.fit(X_train, W_train, y_train)
    predictions = learner.predict(X_test)

    row = {
        "scenario": "null",
        "learner": learner_name,
        "transform": transform,
        "scale": scale,
        "seed": seed,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "status": "ok",
        "error": "",
    }
    row.update(prediction_metrics(predictions))
    rows.append(row)


def run_audit(
    n=5000,
    zero_rate=0.9,
    sigma=0.8,
    e=0.5,
    seeds=(42, 123, 2026, 2027, 2028),
):
    """Run all learner configurations under the null scenario."""
    rows = []

    for seed in seeds:
        X_train, W_train, y_train, _ = simulate(
            n=n,
            scenario="null",
            zero_rate=zero_rate,
            sigma=sigma,
            e=e,
            seed=seed,
        )

        X_test, _, _, _ = simulate(
            n=n,
            scenario="null",
            zero_rate=zero_rate,
            sigma=sigma,
            e=e,
            seed=seed + 1_000_003,
        )

        for name, learner_class in STANDARD_LEARNERS.items():
            try:
                learner = learner_class(e=e)
                fit_and_record(
                    name,
                    learner,
                    X_train,
                    W_train,
                    y_train,
                    X_test,
                    seed,
                    rows,
                )
            except Exception as exc:
                rows.append({
                    "scenario": "null",
                    "learner": name,
                    "transform": "level",
                    "scale": 1.0,
                    "seed": seed,
                    "n_train": len(X_train),
                    "n_test": len(X_test),
                    "status": "failed",
                    "error": repr(exc),
                })

        for transform in TRANSFORMS:
            scales = [1.0] if transform == "rank" else SCALES

            for scale in scales:
                try:
                    learner = TransformedOutcomeLearner(
                        transform=transform,
                        scale=scale,
                        e=e,
                    )
                    fit_and_record(
                        "TransformedOutcome",
                        learner,
                        X_train,
                        W_train,
                        y_train,
                        X_test,
                        seed,
                        rows,
                        transform=transform,
                        scale=scale,
                    )
                except Exception as exc:
                    rows.append({
                        "scenario": "null",
                        "learner": "TransformedOutcome",
                        "transform": transform,
                        "scale": scale,
                        "seed": seed,
                        "n_train": len(X_train),
                        "n_test": len(X_test),
                        "status": "failed",
                        "error": repr(exc),
                    })

        try:
            learner = BackTransformLearner(e=e)
            fit_and_record(
                "BackTransform",
                learner,
                X_train,
                W_train,
                y_train,
                X_test,
                seed,
                rows,
            )
        except Exception as exc:
            rows.append({
                "scenario": "null",
                "learner": "BackTransform",
                "transform": "log1p_inverse",
                "scale": 1.0,
                "seed": seed,
                "n_train": len(X_train),
                "n_test": len(X_test),
                "status": "failed",
                "error": repr(exc),
            })

    return pd.DataFrame(rows)


def summarize_results(results):
    """Aggregate successful runs by learner configuration."""
    successful = results[results["status"] == "ok"].copy()

    metrics = [
        "mean_cate",
        "mean_abs_cate",
        "cate_rmse_vs_null",
        "cate_std",
        "max_abs_cate",
        "positive_fraction",
        "negative_fraction",
    ]

    summary = (
        successful.groupby(
            ["learner", "transform", "scale"],
            dropna=False,
        )[metrics]
        .agg(["mean", "std"])
        .reset_index()
    )

    summary.columns = [
        "_".join(str(part) for part in col if str(part))
        if isinstance(col, tuple)
        else col
        for col in summary.columns
    ]

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="P14 null prediction-magnitude audit."
    )
    parser.add_argument("--n", type=int, default=5000)
    parser.add_argument("--zero-rate", type=float, default=0.9)
    parser.add_argument("--sigma", type=float, default=0.8)
    parser.add_argument("--e", type=float, default=0.5)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 123, 2026, 2027, 2028],
    )
    parser.add_argument(
        "--output",
        default="results/p14_prediction_magnitudes.csv",
    )

    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = run_audit(
        n=args.n,
        zero_rate=args.zero_rate,
        sigma=args.sigma,
        e=args.e,
        seeds=args.seeds,
    )

    summary = summarize_results(results)

    results.to_csv(output_path, index=False)

    summary_path = output_path.with_name(
        output_path.stem + "_summary.csv"
    )
    summary.to_csv(summary_path, index=False)

    print("\nPrediction magnitude audit:")
    print(
        results["status"]
        .value_counts(dropna=False)
        .to_string()
    )

    print(f"\nFull results saved to: {output_path}")
    print(f"Summary saved to: {summary_path}")

    failed = results[results["status"] != "ok"]
    if not failed.empty:
        print("\nFailed configurations:")
        print(failed[["learner", "transform", "scale", "seed", "error"]])

    print(f"\nSuccessful rows: {(results['status'] == 'ok').sum()}")
    print(f"Failed rows: {len(failed)}")


if __name__ == "__main__":
    main()
