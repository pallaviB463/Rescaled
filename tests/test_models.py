
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


def test_poisson_rejects_non_integer_outcomes():
    X = np.array([[0.0], [1.0], [2.0]])
    y = np.array([0.0, 1.5, 2.0])

    with pytest.raises(ValueError, match="integer-valued"):
        Poisson().fit(X, y)


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
