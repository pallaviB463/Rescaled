"""Hillstrom real-data experiment.

Quick diagnostic:
    python -u src/run_real.py data/Hillstrom.csv mens results/hillstrom_mens.csv

Full experiment:
    python -u src/run_real.py data/Hillstrom.csv mens results/hillstrom_mens.csv --full

The quick mode is intentionally small so that integration can be verified promptly.
Use --full for the larger repeated-cross-validation experiment.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

# Support both:
#   python src/run_real.py ...
# and:
#   python -m src.run_real ...
try:
    from . import models
    from .metalearners import LEARNERS
    from .metrics import rank_corr, topk_overlap, ipw_policy_value, qini_auc
    from .transforms import TRANSFORMS, apply_transform
except ImportError:
    import models
    from metalearners import LEARNERS
    from metrics import rank_corr, topk_overlap, ipw_policy_value, qini_auc
    from transforms import TRANSFORMS, apply_transform


ARMS = {
    "mens": "Mens E-Mail",
    "womens": "Womens E-Mail",
}

NUM = ["recency", "history", "mens", "womens", "newbie"]
CAT = ["history_segment", "zip_code", "channel"]

# Only these model classes are available in the project's models.py API.
COUNT_MODEL_NAMES = ("Tweedie", "Hurdle")


def load_hillstrom(csv_path: str | Path, arm: str = "mens"):
    """Load one treatment arm against the no-email control group."""
    if arm not in ARMS:
        raise ValueError(
            f"Unknown arm {arm!r}. Choose one of: {', '.join(ARMS)}"
        )

    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Hillstrom CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    required = set(["segment", "spend", *NUM, *CAT])
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Hillstrom CSV is missing required columns: {missing}")

    keep = df.loc[df["segment"].isin([ARMS[arm], "No E-Mail"])].copy()
    if keep.empty:
        raise ValueError(
            f"No rows found for {ARMS[arm]!r} and 'No E-Mail'. "
            "Check the segment values in the CSV."
        )

    W = (keep["segment"] == ARMS[arm]).astype(int).to_numpy()
    if W.min() == W.max():
        raise ValueError(
            "The selected data contains only one group; both treatment and "
            "No E-Mail control rows are required."
        )

    # Keep the same one-hot encoding across the selected subset.
    X_df = pd.get_dummies(
        keep[NUM + CAT],
        columns=CAT,
        dtype=float,
    )
    X = X_df.to_numpy(dtype=float)
    y = pd.to_numeric(keep["spend"], errors="raise").to_numpy(dtype=float)

    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError("Hillstrom features/outcome contain NaN or infinite values.")
    if np.any(y < 0):
        raise ValueError("Hillstrom spend must be non-negative for this experiment.")

    return X, W, y, float(W.mean())


def _build_pipelines(scales, count_scales, transforms, count_models):
    """Build the local pipeline list; do not use run_sim.pipelines here.

    run_sim.pipelines has a different API and returns dictionaries for the
    simulation experiment, not (family, transform, scale, model) tuples.
    """
    pipelines = []

    for transform_name in transforms:
        if transform_name not in TRANSFORMS and transform_name != "rank":
            raise ValueError(f"Unknown outcome transform: {transform_name}")
        for scale in scales:
            pipelines.append(
                ("TransformedL2", transform_name, float(scale), models.L2)
            )

    # Also evaluate the untransformed outcome on its original scale.
    for scale in scales:
        pipelines.append(("LevelL2", "level", float(scale), models.L2))

    available_count_models = {
        name: getattr(models, name)
        for name in COUNT_MODEL_NAMES
        if hasattr(models, name)
    }
    for name in count_models:
        if name not in available_count_models:
            raise ValueError(
                f"Count model {name!r} is unavailable. "
                f"Available: {sorted(available_count_models)}"
            )
        for scale in count_scales:
            pipelines.append(
                (name, "level", float(scale), available_count_models[name])
            )

    return pipelines


def run_hillstrom(
    csv_path,
    arm="mens",
    repeats=1,
    scales=(0.01, 1.0, 100.0),
    count_scales=(1.0,),
    learners=("T",),
    k_grid=(0.1, 0.2, 0.3, 0.5),
    folds=3,
    seed=0,
    transforms=("identity",),
    count_models=(),
):
    """Run out-of-fold Hillstrom uplift estimates and return a results DataFrame."""
    if repeats < 1:
        raise ValueError("repeats must be at least 1")
    if folds < 2:
        raise ValueError("folds must be at least 2")
    if 1.0 not in scales:
        raise ValueError("scales must include 1.0 as the reference scale")
    if count_models and 1.0 not in count_scales:
        raise ValueError("count_scales must include 1.0 as the reference scale")

    unknown_learners = [name for name in learners if name not in LEARNERS]
    if unknown_learners:
        raise ValueError(
            f"Unknown learners {unknown_learners}. "
            f"Available: {sorted(LEARNERS)}"
        )

    X, W, y, e = load_hillstrom(csv_path, arm)
    if len(y) < folds:
        raise ValueError(f"Only {len(y)} rows available for {folds}-fold CV.")

    pipelines = _build_pipelines(
        scales=scales,
        count_scales=count_scales,
        transforms=transforms,
        count_models=count_models,
    )

    print(
        f"Loaded {len(y):,} rows | features={X.shape[1]} | "
        f"treatment_rate={e:.4f}",
        flush=True,
    )
    print(
        f"Configuration: repeats={repeats}, folds={folds}, "
        f"pipelines={len(pipelines)}, learners={list(learners)}",
        flush=True,
    )

    rows = []
    splitter_count = repeats * folds
    completed_folds = 0

    for rep in range(repeats):
        oof = {}

        splitter = KFold(
            n_splits=folds,
            shuffle=True,
            random_state=seed + rep,
        )

        for fold_num, (train_idx, test_idx) in enumerate(splitter.split(X), start=1):
            completed_folds += 1
            print(
                f"[repeat {rep + 1}/{repeats}] "
                f"[fold {fold_num}/{folds}] "
                f"overall fold {completed_folds}/{splitter_count}",
                flush=True,
            )

            X_train, X_test = X[train_idx], X[test_idx]
            W_train, y_train_raw = W[train_idx], y[train_idx]

            for pipeline_num, (family, transform_name, scale, model_class) in enumerate(
                pipelines, start=1
            ):
                if transform_name == "level":
                    y_fit = y_train_raw * scale
                else:
                    y_fit = apply_transform(transform_name, y_train_raw, scale)

                for learner_name in learners:
                    key = (family, transform_name, scale, learner_name)
                    if key not in oof:
                        oof[key] = np.full(len(y), np.nan, dtype=float)

                    # LEARNERS contains functional APIs. They expect a model
                    # constructor (e.g. models.L2), not an instantiated model.
                    prediction = LEARNERS[learner_name](
                        model_class,
                        X_train,
                        W_train,
                        y_fit,
                        X_test,
                        e=e,
                    )
                    prediction = np.asarray(prediction, dtype=float).reshape(-1)
                    if len(prediction) != len(test_idx):
                        raise RuntimeError(
                            f"{learner_name} returned {len(prediction)} predictions "
                            f"for a test fold of {len(test_idx)} rows."
                        )
                    oof[key][test_idx] = prediction

                # Progress within the current fold; flush so PowerShell displays it.
                if pipeline_num == len(pipelines):
                    print(
                        f"    completed {pipeline_num}/{len(pipelines)} pipelines",
                        flush=True,
                    )

        for (family, transform_name, scale, learner_name), tau in oof.items():
            if not np.isfinite(tau).all():
                raise RuntimeError(
                    "OOF predictions are incomplete or non-finite for "
                    f"{family}/{transform_name}/scale={scale}/{learner_name}."
                )

            reference_key = (family, transform_name, 1.0, learner_name)
            if reference_key not in oof:
                raise RuntimeError(
                    f"Missing scale=1.0 reference for {family}/{transform_name}/"
                    f"{learner_name}."
                )
            reference = oof[reference_key]

            result = {
                "arm": arm,
                "repeat": rep,
                "family": family,
                "transform": transform_name,
                "scale": scale,
                "learner": learner_name,
                "backend": getattr(models, "BACKEND", "unknown"),
                "e": e,
                "rho_vs_a1": rank_corr(tau, reference),
                "top20_overlap_vs_a1": topk_overlap(tau, reference, 0.2),
                "qini": qini_auc(tau, W, y, e),
            }

            for k in k_grid:
                result[f"ipw_value_k{int(k * 100)}"] = ipw_policy_value(
                    tau, W, y, k, e
                )

            rows.append(result)

        print(f"Finished repeat {rep + 1}/{repeats}.", flush=True)

    return pd.DataFrame(rows)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the Hillstrom real-data uplift experiment."
    )
    parser.add_argument("csv", help="Path to data/Hillstrom.csv")
    parser.add_argument("arm", choices=sorted(ARMS), help="mens or womens")
    parser.add_argument("out", help="Output CSV path")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run the larger experiment: 3 repeats, 5 folds, all transforms, "
             "both T/DR learners, and available count models.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.full:
        settings = {
            "repeats": 3,
            "folds": 5,
            "learners": ("T", "DR"),
            "scales": (0.01, 1.0, 100.0),
            "count_scales": (0.01, 1.0, 100.0),
            "transforms": tuple(TRANSFORMS.keys()) + ("rank",),
            "count_models": tuple(
                name for name in COUNT_MODEL_NAMES if hasattr(models, name)
            ),
        }
    else:
        # Small integration run: checks loading, transforms, learner, metrics,
        # cross-validation and CSV output without launching the full workload.
        settings = {
            "repeats": 1,
            "folds": 3,
            "learners": ("T",),
            "scales": (0.01, 1.0, 100.0),
            "count_scales": (1.0,),
            "transforms": ("identity",),
            "count_models": (),
        }

    print(f"Starting Hillstrom {args.arm} experiment.", flush=True)
    result_df = run_hillstrom(args.csv, arm=args.arm, **settings)

    result_df.to_csv(output_path, index=False)
    print(f"\nSaved {len(result_df)} result rows to: {output_path}", flush=True)

    summary_columns = [
        "rho_vs_a1",
        "top20_overlap_vs_a1",
        "ipw_value_k20",
        "qini",
    ]
    print(
        result_df.groupby(
            ["family", "transform", "scale", "learner"]
        )[summary_columns].mean().round(4).to_string(),
        flush=True,
    )


if __name__ == "__main__":
    main()
