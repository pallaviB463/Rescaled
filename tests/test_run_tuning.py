
import numpy as np
import pandas as pd
import pytest

from src.run_tuning import run_experiment


def test_tuning_runner_rejects_unknown_scenario():
    with pytest.raises(ValueError, match="Unknown scenarios"):
        run_experiment(scenarios=["not_a_scenario"])


def test_tuning_runner_rejects_unknown_model():
    with pytest.raises(ValueError, match="Models must be"):
        run_experiment(models=["unknown"])


def test_tuning_runner_rejects_invalid_candidate_budget():
    with pytest.raises(ValueError, match="n_candidates must be positive"):
        run_experiment(n_candidates=0)


def test_tuning_runner_writes_summary_and_history(tmp_path, monkeypatch):
    import src.run_tuning as runner

    def fake_simulate(**kwargs):
        rng = np.random.default_rng(kwargs["seed"])
        n = kwargs["n"]
        X = rng.normal(size=(n, 3))
        W = np.tile([0, 1], n // 2)

        if len(W) < n:
            W = np.append(W, 0)

        y = np.maximum(
            1 + X[:, 0] + W + rng.normal(0, 0.1, n),
            0,
        )

        return X, W, y, {}

    def fake_tune(**kwargs):
        return {
            "best_params": {"n_estimators": 100},
            "best_validation_score": 1.2,
            "selection_metric": "rmse",
            "best_validation_rmse": 1.2,
            "best_validation_deviance": np.nan,
            "test_rmse": 1.3,
            "requested_candidates": kwargs["n_candidates"],
            "successful_candidates": 2,
            "failed_candidates": 0,
            "n_train": 60,
            "n_validation": 20,
            "n_test": 20,
            "history": [
                {
                    "candidate_id": 0,
                    "parameters": "{'n_estimators': 100}",
                    "selection_metric": "rmse",
                    "validation_score": 1.2,
                    "validation_rmse": 1.2,
                    "validation_deviance": np.nan,
                    "status": "ok",
                    "error": "",
                }
            ],
        }

    monkeypatch.setattr(runner, "simulate", fake_simulate)
    monkeypatch.setattr(runner, "run_tuning_experiment", fake_tune)

    summary_path = tmp_path / "tuning.csv"

    summary, history = run_experiment(
        scenarios=["null"],
        n=100,
        models=["L2"],
        seeds=[0],
        n_candidates=2,
        output=str(summary_path),
    )

    assert summary_path.exists()
    assert (tmp_path / "tuning_history.csv").exists()

    assert summary.loc[0, "status"] == "ok"
    assert summary.loc[0, "requested_candidates"] == 2
    assert summary.loc[0, "successful_candidates"] == 2
    assert summary.loc[0, "failed_candidates"] == 0
    assert summary.loc[0, "selection_metric"] == "rmse"
    assert summary.loc[0, "validation_score"] == 1.2
    assert summary.loc[0, "validation_rmse"] == 1.2
    assert summary.loc[0, "test_rmse"] == 1.3

    assert len(history) == 1
    assert history.loc[0, "selection_metric"] == "rmse"

    saved_summary = pd.read_csv(summary_path)
    assert saved_summary.loc[0, "model"] == "L2"
    assert saved_summary.loc[0, "requested_candidates"] == 2
    assert saved_summary.loc[0, "selection_metric"] == "rmse"


def test_tuning_runner_passes_equal_budget_to_all_models(
    tmp_path,
    monkeypatch,
):
    import src.run_tuning as runner

    received_budgets = []

    def fake_simulate(**kwargs):
        rng = np.random.default_rng(kwargs["seed"])
        n = kwargs["n"]
        X = rng.normal(size=(n, 3))
        W = np.tile([0, 1], n // 2)
        y = np.maximum(
            1 + X[:, 0] + W + rng.normal(0, 0.1, n),
            0,
        )
        return X, W, y, {}

    def fake_tune(**kwargs):
        received_budgets.append(
            (kwargs["model_name"], kwargs["n_candidates"])
        )

        return {
            "best_params": {"n_estimators": 100},
            "best_validation_score": 1.2,
            "selection_metric": (
                "tweedie_deviance"
                if kwargs["model_name"] == "Tweedie"
                else "rmse"
            ),
            "best_validation_rmse": 1.2,
            "best_validation_deviance": (
                0.5 if kwargs["model_name"] == "Tweedie" else np.nan
            ),
            "test_rmse": 1.3,
            "requested_candidates": kwargs["n_candidates"],
            "successful_candidates": kwargs["n_candidates"],
            "failed_candidates": 0,
            "n_train": 60,
            "n_validation": 20,
            "n_test": 20,
            "history": [],
        }

    monkeypatch.setattr(runner, "simulate", fake_simulate)
    monkeypatch.setattr(runner, "run_tuning_experiment", fake_tune)

    summary, _ = run_experiment(
        scenarios=["null"],
        n=100,
        models=["L2", "Tweedie", "Hurdle"],
        seeds=[0],
        n_candidates=4,
        output=str(tmp_path / "fairness.csv"),
    )

    assert received_budgets == [
        ("L2", 4),
        ("Tweedie", 4),
        ("Hurdle", 4),
    ]

    assert summary["requested_candidates"].tolist() == [4, 4, 4]
    assert summary["status"].tolist() == ["ok", "ok", "ok"]
