"""Run Neurolytics simulation experiments."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from src.dgp import SCENARIOS, simulate, true_transformed_cate
from src.metrics import rank_corr, topk_overlap, policy_value, regret
from src.models import L2, Tweedie, Hurdle
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
RANK_TRANSFORM = "rank"
REFERENCE_SCALE = 1.0
OUTCOME_MODELS = {"L2": L2, "Tweedie": Tweedie, "Hurdle": Hurdle}



def pipelines(scales=None, transforms=None):
    """Build transformation/scale configurations, including a=1 reference."""
    scales = list(SCALES if scales is None else scales)
    transforms = list(
        (list(TRANSFORMS) + ["rank"])
        if transforms is None
        else transforms
    )

    if not scales:
        raise ValueError("At least one scale is required.")
    if any(not np.isfinite(a) or a <= 0 for a in scales):
        raise ValueError("All scales must be finite and positive.")
    if len(set(scales)) != len(scales):
        raise ValueError("Scales must not contain duplicates.")
    if REFERENCE_SCALE not in scales:
        raise ValueError("Scales must include the reference scale 1.0.")

    allowed = set(TRANSFORMS) | {RANK_TRANSFORM}
    unknown = set(transforms) - allowed
    if unknown:
        raise ValueError(f"Unknown transformations: {sorted(unknown)}")

    if len(set(transforms)) != len(transforms):
        raise ValueError("Transformations must not contain duplicates.")

    configs = []
    for name in transforms:
        if name == RANK_TRANSFORM:
            # Positive scaling cannot change ranks, so evaluate rank once.
            configs.append({
                "transform": name,
                "scale": REFERENCE_SCALE,
            })
        else:
            for scale in scales:
                configs.append({
                    "transform": name,
                    "scale": float(scale),
                })

    return configs



def _make_model(model_name="L2"):
    """Create a fresh base model for level-scale outcome modeling.

    Tweedie and Hurdle are intended for the T-learner, which fits models
    directly to nonnegative observed outcomes. DR/X pseudo-outcomes may be
    negative, so those learners must keep L2.
    """
    if model_name not in OUTCOME_MODELS:
        raise ValueError(
            f"Unknown outcome model {model_name!r}. "
            f"Choose from {list(OUTCOME_MODELS)}."
        )
    model_class = OUTCOME_MODELS[model_name]
    params = dict(
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=15,
        min_child_samples=50,
    )
    if model_name == "Tweedie":
        return model_class(power=1.5, **params)
    return model_class(**params)


def _fit_predict_learner(name, X, W, y, X_test, e, outcome_model="L2"):
    """Fit a meta-learner and return its estimated treatment effects."""
    model_name = outcome_model if name == "T" else "L2"
    learner = LEARNERS[name](model=_make_model(model_name), e=e)
    learner.fit(X, W, y)
    return np.asarray(learner.predict(X_test), dtype=float).ravel()


def _transform_values(name, values, a=1.0):
    """Apply a supported outcome transformation."""
    values = np.asarray(values, dtype=float)

    if not np.isfinite(values).all():
        raise ValueError("Transformation inputs must be finite.")
    if (values < 0).any():
        raise ValueError("Transformation inputs must be nonnegative.")
    if not np.isfinite(a) or a <= 0:
        raise ValueError("Scale must be finite and positive.")

    scaled = a * values
    if name == RANK_TRANSFORM:
        if scaled.size == 0:
            return scaled
        return rankdata(scaled, method="average") / len(scaled)
    if name not in TRANSFORMS:
        raise ValueError(f"Unknown transformation: {name!r}")
    return np.asarray(TRANSFORMS[name](scaled), dtype=float)


def _result_row(
    scenario,
    learner,
    transform,
    scale,
    tau_hat,
    tau_true,
    ranking_truth=None,
    rho_vs_a1=np.nan,
):
    """Create a consistent row; ranking metrics are undefined for constant truth."""
    tau_hat = np.asarray(tau_hat, dtype=float).ravel()
    tau_true = np.asarray(tau_true, dtype=float).ravel()

    if tau_hat.shape != tau_true.shape:
        raise ValueError("Prediction and level-truth arrays must have matching shapes.")

    if ranking_truth is None:
        correlation = np.nan
        overlap = np.nan
    else:
        ranking_truth = np.asarray(ranking_truth, dtype=float).ravel()
        if tau_hat.shape != ranking_truth.shape:
            raise ValueError("Prediction and ranking-truth arrays must match.")
        if not np.isfinite(ranking_truth).all():
            raise ValueError("Ranking truth must contain only finite values.")
        # A constant true effect has no unique ranking (e.g. the null DGP).
        if ranking_truth.size < 2 or np.ptp(ranking_truth) == 0:
            correlation = np.nan
            overlap = np.nan
        else:
            correlation = rank_corr(tau_hat, ranking_truth)
            overlap = topk_overlap(tau_hat, ranking_truth, k=0.2)

    return {
        "scenario": scenario,
        "learner": learner,
        "transform": transform,
        "scale": float(scale),
        "rank_corr": correlation,
        "topk_overlap": overlap,
        # Always evaluate policies against original, level-scale CATE truth.
        "policy_value": policy_value(tau_hat, tau_true, k=0.2),
        "regret": regret(tau_hat, tau_true, k=0.2),
        "rho_vs_a1": rho_vs_a1,
        "status": "ok",
        "error": "",
    }


def _failed_row(scenario, learner, transform, scale, exc):
    """Return a standardized row for a failed model/configuration."""
    return {
        "scenario": scenario,
        "learner": learner,
        "transform": transform,
        "scale": float(scale),
        "rank_corr": np.nan,
        "topk_overlap": np.nan,
        "policy_value": np.nan,
        "regret": np.nan,
        "rho_vs_a1": np.nan,
        "status": "failed",
        "error": f"{type(exc).__name__}: {exc}",
    }


def one_rep(
    scenario="opposed",
    n=5000,
    zero_rate=0.9,
    sigma=0.8,
    e=0.5,
    seed=0,
    scales=None,
    transforms=None,
    learners=("T",),
    include_transformed=True,
    outcome_model="L2",
):
    """Run one simulation replicate and return a results DataFrame."""
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario!r}")
    if n < 2:
        raise ValueError("n must be at least 2.")
    if not 0 < zero_rate < 1:
        raise ValueError("zero_rate must be strictly between 0 and 1.")
    if not np.isfinite(sigma) or sigma < 0:
        raise ValueError("sigma must be finite and nonnegative.")
    if not 0 < e < 1:
        raise ValueError("e must be strictly between 0 and 1.")
    if not learners:
        raise ValueError("At least one learner must be specified.")

    for learner_name in learners:
        if learner_name not in LEARNERS:
            raise ValueError(f"Unknown learner: {learner_name}")

    if outcome_model not in OUTCOME_MODELS:
        raise ValueError(
            f"Unknown outcome model {outcome_model!r}. "
            f"Choose from {list(OUTCOME_MODELS)}."
        )
    if outcome_model != "L2" and any(name != "T" for name in learners):
        raise ValueError(
            "Tweedie and Hurdle outcome models are supported only with "
            "the T-learner. Use --learners T or --outcome-model L2."
        )

    configurations = pipelines(scales=scales, transforms=transforms)

    X, W, y, _ = simulate(
        n=n, scenario=scenario, zero_rate=zero_rate,
        sigma=sigma, e=e, seed=seed,
    )
    X_test, _, y_test, truth_test = simulate(
        n=max(2000, n // 2),
        scenario=scenario,
        zero_rate=zero_rate,
        sigma=sigma,
        e=e,
        seed=seed + 1_000_003,
    )

    tau_true = np.asarray(truth_test["tau_level"], dtype=float).ravel()
    rows = []

    # Standard level-scale meta-learners.
    for learner_name in learners:
        try:
            tau_hat = _fit_predict_learner(
                learner_name, X, W, y, X_test, e, outcome_model=outcome_model
            )
            rows.append(
                _result_row(
                    scenario=scenario,
                    learner=(f"{learner_name}:{outcome_model}" if outcome_model != "L2" else learner_name),
                    transform="level",
                    scale=REFERENCE_SCALE,
                    tau_hat=tau_hat,
                    tau_true=tau_true,
                    ranking_truth=tau_true,
                )
            )
        except Exception as exc:
            rows.append(
                _failed_row(
                    scenario, (f"{learner_name}:{outcome_model}" if outcome_model != "L2" else learner_name), "level", REFERENCE_SCALE, exc
                )
            )

    if include_transformed:
        predictions = {}

        for config in configurations:
            transform = config["transform"]
            scale = config["scale"]

            try:
                # TransformedOutcomeLearner applies its transform and scale.
                learner = TransformedOutcomeLearner(
                    transform=transform,
                    scale=scale,
                    model=_make_model(),
                    e=e,
                )
                learner.fit(X, W, y)
                tau_hat = np.asarray(learner.predict(X_test), dtype=float).ravel()

                # Evaluate every learned ranking against the appropriate truth.
                # Rank-transformed predictions are evaluated against level-scale CATE,
                # since no exact transformed-scale population truth is implemented.
                if transform == RANK_TRANSFORM:
                    ranking_truth = tau_true
                else:
                    ranking_truth = true_transformed_cate(
                        truth_test,
                        m=TRANSFORMS[transform],
                        a=scale,
                    )

                predictions[(transform, scale)] = tau_hat
                rows.append(
                    _result_row(
                        scenario=scenario,
                        learner="TransformedOutcome",
                        transform=transform,
                        scale=scale,
                        tau_hat=tau_hat,
                        tau_true=tau_true,
                        ranking_truth=ranking_truth,
                    )
                )
            except Exception as exc:
                rows.append(
                    _failed_row(
                        scenario, "TransformedOutcome", transform, scale, exc
                    )
                )

        # Compare estimated rankings to the same transformation at scale a=1.
        for row in rows:
            if row["learner"] != "TransformedOutcome":
                continue

            transform = row["transform"]
            scale = row["scale"]

            if transform == RANK_TRANSFORM:
                # Rank is configured once; no distinct-scale comparison exists.
                row["rho_vs_a1"] = np.nan
                continue

            current = predictions.get((transform, scale))
            reference = predictions.get((transform, REFERENCE_SCALE))
            if current is not None and reference is not None:
                row["rho_vs_a1"] = rank_corr(current, reference)

    return pd.DataFrame(rows)


def run_experiment(
    scenarios=None,
    n=5000,
    zero_rate=0.9,
    sigma=0.8,
    e=0.5,
    seeds=(0,),
    scales=None,
    transforms=None,
    learners=("T",),
    include_transformed=True,
    output_path=None,
    outcome_model="L2",
):
    """Run multiple scenarios/seeds and save detailed and summary CSV files."""
    scenarios = list(SCENARIOS if scenarios is None else scenarios)
    seeds = list(seeds)

    if not scenarios:
        raise ValueError("At least one scenario is required.")
    if not seeds:
        raise ValueError("At least one seed is required.")
    unknown = set(scenarios) - set(SCENARIOS)
    if unknown:
        raise ValueError(f"Unknown scenarios: {sorted(unknown)}")

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
                transforms=transforms,
                learners=learners,
                include_transformed=include_transformed,
                outcome_model=outcome_model,
            )
            frame["seed"] = seed
            frame["n"] = n
            frame["zero_rate"] = zero_rate
            frame["sigma"] = sigma
            frames.append(frame)

    results = pd.concat(frames, ignore_index=True)
    destination = (
        Path(output_path)
        if output_path
        else OUTPUT_DIR / "simulation_results.csv"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(destination, index=False)

    metric_columns = [
        "rank_corr", "topk_overlap", "policy_value", "regret", "rho_vs_a1"
    ]
    summary = (
        results[results["status"] == "ok"]
        .groupby(
            ["scenario", "learner", "transform", "scale"],
            dropna=False,
        )[metric_columns]
        .mean()
        .reset_index()
    )
    summary_path = destination.with_name(f"{destination.stem}_summary.csv")
    summary.to_csv(summary_path, index=False)
    return results, summary


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description="Run Neurolytics treatment-effect simulations."
    )
    parser.add_argument("--n", type=int, default=5000)
    parser.add_argument("--zero-rate", type=float, default=0.9)
    parser.add_argument("--sigma", type=float, default=0.8)
    parser.add_argument("--e", type=float, default=0.5)
    parser.add_argument(
        "--outcome-model",
        choices=sorted(OUTCOME_MODELS),
        default="L2",
        help=(
            "Outcome model for the level-scale T-learner. Tweedie and Hurdle "
            "are supported only with --learners T. Transformed-outcome "
            "experiments continue to use L2."
        ),
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument(
        "--scenarios", nargs="+", choices=list(SCENARIOS), default=["opposed"]
    )
    parser.add_argument(
        "--learners", nargs="+", choices=list(LEARNERS), default=["T"]
    )
    parser.add_argument(
        "--scales", type=float, nargs="+", default=None,
        help="Optional scale values; must include 1.0.",
    )
    parser.add_argument(
        "--transforms", nargs="+", choices=sorted(set(TRANSFORMS) | {RANK_TRANSFORM}),
        default=None,
    )
    parser.add_argument(
        "--no-transformed",
        action="store_true",
        help="Skip transformed-outcome learner experiments.",
    )
    parser.add_argument("--output", default="results/simulation_results.csv")
    args = parser.parse_args()

    results, _ = run_experiment(
        scenarios=args.scenarios,
        n=args.n,
        zero_rate=args.zero_rate,
        sigma=args.sigma,
        e=args.e,
        seeds=args.seeds,
        scales=args.scales,
        transforms=args.transforms,
        learners=args.learners,
        include_transformed=not args.no_transformed,
        output_path=args.output,
        outcome_model=args.outcome_model,
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
