
import pandas as pd
import pytest

from src.run_tuning import run_experiment


def test_tuning_runner_rejects_unknown_scenario():
    with pytest.raises(ValueError, match="Unknown scenarios"):
        run_experiment(scenarios=["not_a_scenario"])


def test_tuning_runner_rejects_unknown_model():
    with pytest.raises(ValueError, match="Models must be"):
        run_experiment(models=["unknown"])


def test_tuning_runner_writes_summary_and_history(tmp_path, monkeypatch):
    import src.run_tuning as runner

    def fake_simulate(**kwargs):
        import numpy as np

        rng = np.random.default_rng(kwargs["seed"])
        n = kwargs["n"]
        X = rng.normal(size=(n, 3))
        W = np.tile([0, 1], n // 2)
        if len(W) < n:
            W = np.append(W, 0)
        y = np.maximum(1 + X[:, 0] + W + rng.normal(0, 0.1, n), 0)
        return X, W, y, {}

    def fake_tune(**kwargs):
        return {
            "best_params": {"n_estimators": 100},
            "best_validation_rmse": 1.2,
            "test_rmse": 1.3,
            "successful_candidates": 2,
            "failed_candidates": 0,
            "n_train": 60,
            "n_validation": 20,
            "n_test": 20,
            "history": [
                {
                    "candidate_id": 0,
                    "parameters": "{'n_estimators': 100}",
                    "status": "ok",
                    "validation_rmse": 1.2,
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
    assert len(history) == 1

    saved_summary = pd.read_csv(summary_path)
    assert saved_summary.loc[0, "model"] == "L2"
