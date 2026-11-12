
import numpy as np
import pytest

from src.tuning import (
    BASE_SEARCH_SPACE,
    TWEEDIE_POWERS,
    candidate_configurations,
    tune_outcome_model,
)


def test_candidate_configurations_are_reproducible():
    first = candidate_configurations("L2", n_candidates=4, seed=17)
    second = candidate_configurations("L2", n_candidates=4, seed=17)

    assert first == second
    assert len(first) == 4
    assert len({repr(config) for config in first}) == 4


def test_search_space_includes_required_depth_and_learning_rates():
    assert "max_depth" in BASE_SEARCH_SPACE
    assert BASE_SEARCH_SPACE["max_depth"] == [3, 5, 7, -1]

    assert BASE_SEARCH_SPACE["learning_rate"] == [0.03, 0.05, 0.10]


def test_tweedie_candidates_include_required_powers():
    candidates = candidate_configurations(
        "Tweedie", n_candidates=12, seed=17
    )

    assert len(candidates) == 12
    assert TWEEDIE_POWERS == [1.1, 1.3, 1.5, 1.7]
    assert all(config["power"] in TWEEDIE_POWERS for config in candidates)


def test_all_model_families_receive_same_candidate_budget():
    budget = 12

    for model_name in ("L2", "Tweedie", "Hurdle"):
        candidates = candidate_configurations(
            model_name,
            n_candidates=budget,
            seed=42,
        )
        assert len(candidates) == budget


def test_candidate_generation_rejects_unknown_model():
    with pytest.raises(ValueError, match="Unknown model"):
        candidate_configurations("Unknown")


def test_candidate_generation_rejects_invalid_budget():
    for invalid_budget in (0, -1, 1.5, True):
        with pytest.raises(ValueError, match="positive integer"):
            candidate_configurations(
                "L2",
                n_candidates=invalid_budget,
            )


def test_tuning_rejects_missing_treatment_arm():
    X = np.arange(40, dtype=float).reshape(20, 2)
    W = np.zeros(20, dtype=int)
    y = np.ones(20)

    with pytest.raises(ValueError, match="both treatment arms"):
        tune_outcome_model(
            X, W, y,
            X, W, y,
            model_name="L2",
            candidate_configs=[{
                "n_estimators": 5,
                "learning_rate": 0.1,
                "num_leaves": 7,
                "max_depth": 3,
                "min_child_samples": 2,
            }],
        )


def test_tuning_selects_a_successful_candidate():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(100, 5))
    W = np.tile([0, 1], 50)
    y = np.maximum(
        0.1 + X[:, 0] + 0.4 * W + rng.normal(0, 0.1, 100),
        0,
    )

    X_train, X_val = X[:70], X[70:]
    W_train, W_val = W[:70], W[70:]
    y_train, y_val = y[:70], y[70:]

    configs = [{
        "n_estimators": 5,
        "learning_rate": 0.1,
        "num_leaves": 7,
        "max_depth": 3,
        "min_child_samples": 2,
    }]

    result = tune_outcome_model(
        X_train, W_train, y_train,
        X_val, W_val, y_val,
        model_name="L2",
        candidate_configs=configs,
    )

    assert result["successful_candidates"] == 1
    assert result["failed_candidates"] == 0
    assert np.isfinite(result["best_validation_rmse"])
    assert result["selection_metric"] == "rmse"
    assert set(result["models"]) == {0, 1}
    assert len(result["history"]) == 1


def test_tweedie_uses_deviance_for_candidate_selection():
    rng = np.random.default_rng(19)
    X = rng.normal(size=(400, 5))
    W = np.tile([0, 1], 200)

    y = np.zeros(400)
    positive = rng.random(400) < 0.5
    y[positive] = np.exp(
        1.0 + 0.2 * X[positive, 0]
        + rng.normal(0, 0.3, positive.sum())
    )

    configs = [
        {
            "n_estimators": 10,
            "learning_rate": 0.05,
            "num_leaves": 7,
            "max_depth": 3,
            "min_child_samples": 5,
            "power": 1.1,
        },
        {
            "n_estimators": 10,
            "learning_rate": 0.05,
            "num_leaves": 7,
            "max_depth": 3,
            "min_child_samples": 5,
            "power": 1.7,
        },
    ]

    result = tune_outcome_model(
        X[:280], W[:280], y[:280],
        X[280:], W[280:], y[280:],
        model_name="Tweedie",
        candidate_configs=configs,
    )

    successful = [
        row for row in result["history"]
        if row["status"] == "ok"
    ]

    assert result["selection_metric"] == "tweedie_deviance"
    assert result["successful_candidates"] == 2

    winner = min(
        successful,
        key=lambda row: row["validation_deviance"],
    )

    assert result["best_params"] == configs[winner["candidate_id"]]
    assert np.isfinite(result["best_validation_deviance"])
    assert all(
        np.isfinite(row["validation_deviance"])
        for row in successful
    )


def test_full_tuning_workflow_refits_and_evaluates_on_test():
    from src.tuning import run_tuning_experiment

    rng = np.random.default_rng(23)
    X = rng.normal(size=(120, 5))
    W = np.tile([0, 1], 60)
    y = np.maximum(
        1.0 + X[:, 0] + 0.5 * W + rng.normal(0, 0.2, 120),
        0,
    )

    result = run_tuning_experiment(
        X,
        W,
        y,
        model_name="L2",
        n_candidates=1,
        seed=23,
    )

    assert result["n_train"] == 72
    assert result["n_validation"] == 24
    assert result["n_test"] == 24
    assert np.isfinite(result["best_validation_rmse"])
    assert np.isfinite(result["test_rmse"])
    assert len(result["test_predictions"]) == 24
    assert set(result["models"]) == {0, 1}
    assert result["requested_candidates"] == 1


def test_tuning_returns_parameters_of_winning_candidate():
    rng = np.random.default_rng(31)
    X = rng.normal(size=(100, 5))
    W = np.tile([0, 1], 50)
    y = np.maximum(
        1.0 + X[:, 0] + 0.3 * W + rng.normal(0, 0.1, 100),
        0,
    )

    configs = [
        {
            "n_estimators": 5,
            "learning_rate": 0.1,
            "num_leaves": 7,
            "max_depth": 3,
            "min_child_samples": 2,
        },
        {
            "n_estimators": 10,
            "learning_rate": 0.05,
            "num_leaves": 7,
            "max_depth": 3,
            "min_child_samples": 2,
        },
    ]

    result = tune_outcome_model(
        X[:70], W[:70], y[:70],
        X[70:], W[70:], y[70:],
        model_name="L2",
        candidate_configs=configs,
    )

    successful = [
        row for row in result["history"]
        if row["status"] == "ok"
    ]

    winner = min(
        successful,
        key=lambda row: row["validation_score"],
    )

    assert result["best_params"] == configs[winner["candidate_id"]]
