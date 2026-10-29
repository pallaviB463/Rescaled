
import numpy as np
import pytest

from src.models import L2, Poisson, Tweedie, Hurdle, NegBin


@pytest.fixture
def training_data():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(250, 5))
    # Zero-inflated, nonnegative integer outcomes
    y = rng.poisson(np.exp(0.3 + 0.2 * X[:, 0]))
    y[rng.random(250) < 0.65] = 0
    return X, y


@pytest.mark.parametrize(
    "model_class",
    [L2, Poisson, Tweedie, Hurdle, NegBin],
)
def test_model_fit_and_predict(model_class, training_data):
    X, y = training_data

    model = model_class()
    model.fit(X, y)
    predictions = model.predict(X[:20])

    assert predictions.shape == (20,)
    assert np.all(np.isfinite(predictions))
    assert np.all(predictions >= 0)


def test_poisson_accepts_non_integer_outcomes():
    X = np.array([[0.0], [1.0], [2.0]])
    y = np.array([0.0, 1.5, 2.0])

    model = Poisson(n_estimators=20)
    model.fit(X, y)

    predictions = model.predict(X)

    assert predictions.shape == (3,)
    assert np.all(np.isfinite(predictions))
    assert np.all(predictions >= 0)


def test_negative_binomial_rejects_non_integer_outcomes():
    X = np.array([[0.0], [1.0], [2.0]])
    y = np.array([0.0, 1.5, 2.0])

    with pytest.raises(ValueError, match="integer-valued"):
        NegBin().fit(X, y)


def test_hurdle_rejects_all_zero_outcomes():
    X = np.ones((20, 3))
    y = np.zeros(20)

    with pytest.raises(ValueError, match="positive"):
        Hurdle().fit(X, y)


def test_tweedie_rejects_invalid_power():
    with pytest.raises(ValueError):
        Tweedie(power=2.0)
def test_poisson_accepts_nonnegative_continuous_outcomes():
    import numpy as np

    from src.models import Poisson

    rng = np.random.default_rng(42)
    X = rng.normal(size=(120, 5))
    y = np.exp(0.2 * X[:, 0]) * 0.7

    model = Poisson(n_estimators=30, verbosity=-1)
    model.fit(X, y)

    predictions = model.predict(X)

    assert predictions.shape == (len(y),)
    assert np.all(np.isfinite(predictions))
    assert np.all(predictions >= 0)

@pytest.mark.parametrize(
    "model_class,extra_params",
    [
        (Poisson, {}),
        (Tweedie, {"power": 1.5}),
        (Hurdle, {}),
    ],
)
def test_count_aware_models_are_approximately_scale_equivariant(
    model_class, extra_params
):
    """Scaling outcomes should approximately scale predicted conditional means."""
    rng = np.random.default_rng(2026)
    X = rng.normal(size=(600, 5))

    # A zero-inflated, nonnegative continuous outcome.
    positive = rng.random(len(X)) < 0.4
    y = np.zeros(len(X))
    y[positive] = np.exp(
        1.0 + 0.35 * X[positive, 0] - 0.2 * X[positive, 1]
        + rng.normal(0, 0.2, positive.sum())
    )

    scale = 100.0
    params = {
        "n_estimators": 80,
        "learning_rate": 0.05,
        "num_leaves": 10,
        "min_child_samples": 15,
        "verbosity": -1,
        "random_state": 42,
    }

    original = model_class(**extra_params, **params)
    rescaled = model_class(**extra_params, **params)

    original.fit(X, y)
    rescaled.fit(X, scale * y)

    pred_original = original.predict(X)
    pred_rescaled = rescaled.predict(X)

    assert np.all(np.isfinite(pred_original))
    assert np.all(np.isfinite(pred_rescaled))
    assert np.all(pred_original >= 0)
    assert np.all(pred_rescaled >= 0)

    # Allow modest numerical and tree-fitting differences.
    np.testing.assert_allclose(
        pred_rescaled,
        scale * pred_original,
        rtol=0.20,
        atol=0.05,
    )


def test_hurdle_occurrence_predictions_are_invariant_to_outcome_scale():
    """Rescaling positive outcomes must not change the zero/positive labels."""
    rng = np.random.default_rng(2027)
    X = rng.normal(size=(600, 5))

    positive = rng.random(len(X)) < 0.4
    y = np.zeros(len(X))
    y[positive] = np.exp(
        1.0 + 0.35 * X[positive, 0] - 0.2 * X[positive, 1]
    )

    params = {
        "n_estimators": 80,
        "learning_rate": 0.05,
        "num_leaves": 10,
        "min_child_samples": 15,
        "verbosity": -1,
        "random_state": 42,
    }

    original = Hurdle(**params).fit(X, y)
    rescaled = Hurdle(**params).fit(X, 100.0 * y)

    if original.occurrence_model_ is not None:
        np.testing.assert_allclose(
            rescaled.occurrence_model_.predict_proba(X),
            original.occurrence_model_.predict_proba(X),
            rtol=0,
            atol=1e-12,
        )

    np.testing.assert_allclose(
        rescaled.positive_model_.predict(X),
        100.0 * original.positive_model_.predict(X),
        rtol=0.20,
        atol=0.05,
    )
