
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
    rng = np.random.default_rng(42)
    X = rng.normal(size=(120, 5))
    y = np.exp(0.2 * X[:, 0]) * 0.7

    model = Poisson(n_estimators=30, verbosity=-1)
    model.fit(X, y)

    predictions = model.predict(X)

    assert predictions.shape == (len(y),)
    assert np.all(np.isfinite(predictions))
    assert np.all(predictions >= 0)


# ============================================================
# P9: SCALE EQUIVARIANCE TESTS
# ============================================================

P9_SCALES = [
    0.001,
    0.01,
    0.1,
    1.0,
    10.0,
    100.0,
    1000.0,
]


@pytest.mark.parametrize(
    "model_class,extra_params",
    [
        (Poisson, {}),
        (Tweedie, {"power": 1.5}),
        (Hurdle, {}),
    ],
)
@pytest.mark.parametrize("scale", P9_SCALES)
def test_count_aware_models_are_scale_equivariant(
    model_class,
    extra_params,
    scale,
):
    """
    Verify prediction scaling and ranking invariance across all P9 scales.

    Required properties:
    1. Predictions approximately scale by the same positive factor.
    2. Prediction rankings remain unchanged.
    3. Hurdle occurrence probabilities remain unchanged.
    """

    rng = np.random.default_rng(2026)

    X = rng.normal(size=(600, 5))

    # Generate zero-inflated, nonnegative continuous outcomes.
    positive = rng.random(len(X)) < 0.4

    y = np.zeros(len(X))

    y[positive] = np.exp(
        1.0
        + 0.35 * X[positive, 0]
        - 0.2 * X[positive, 1]
        + rng.normal(0, 0.2, positive.sum())
    )

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

    # Fit both models on exactly the same features.
    original.fit(X, y)
    rescaled.fit(X, scale * y)

    pred_original = original.predict(X)
    pred_rescaled = rescaled.predict(X)

    # Basic prediction validity.
    assert np.all(np.isfinite(pred_original))
    assert np.all(np.isfinite(pred_rescaled))

    assert np.all(pred_original >= 0)
    assert np.all(pred_rescaled >= 0)

    assert pred_original.shape == pred_rescaled.shape

    # Property 1:
    # f(aY) approximately equals a * f(Y).
    np.testing.assert_allclose(
        pred_rescaled,
        scale * pred_original,
        rtol=0.20,
        atol=0.05 * scale,
    )

    # Property 2:
    # Rankings must remain unchanged.
    ranks_original = np.argsort(
        np.argsort(pred_original, kind="stable"),
        kind="stable",
    )

    ranks_rescaled = np.argsort(
        np.argsort(pred_rescaled, kind="stable"),
        kind="stable",
    )

    np.testing.assert_array_equal(
        ranks_rescaled,
        ranks_original,
    )

    # Property 3:
    # Hurdle occurrence predictions depend on zero/positive labels.
    # These labels do not change under positive rescaling.
    if (
        model_class is Hurdle
        and original.occurrence_model_ is not None
    ):
        np.testing.assert_allclose(
            rescaled.occurrence_model_.predict_proba(X),
            original.occurrence_model_.predict_proba(X),
            rtol=0,
            atol=1e-12,
        )


# ============================================================
# P7: HURDLE MEAN CALIBRATION TEST
# ============================================================

def test_hurdle_mean_is_within_five_percent_of_observed_mean():
    rng = np.random.default_rng(2028)

    X = rng.normal(size=(5000, 5))

    positive = rng.random(len(X)) < 0.4

    y = np.zeros(len(X))

    y[positive] = np.exp(
        1.0
        + 0.35 * X[positive, 0]
        - 0.2 * X[positive, 1]
        + rng.normal(0, 0.2, positive.sum())
    )

    model = Hurdle(
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=15,
        min_child_samples=50,
        verbosity=-1,
        random_state=42,
    )

    model.fit(X, y)

    predictions = model.predict(X)

    observed_mean = np.mean(y)
    predicted_mean = np.mean(predictions)

    relative_error = (
        abs(predicted_mean - observed_mean) / observed_mean
    )

    assert relative_error <= 0.05, (
        f"Hurdle mean calibration error is {relative_error:.2%}; "
        "expected <= 5%."
    )
