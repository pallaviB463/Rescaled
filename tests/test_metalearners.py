
import numpy as np
import pytest

from src.models import L2
from src.metalearners import (
    SLearner,
    TLearner,
    XLearner,
    DRLearner,
    TransformedOutcomeLearner,
)


@pytest.fixture
def training_data():
    rng = np.random.default_rng(123)
    X = rng.normal(size=(400, 5))
    W = rng.binomial(1, 0.5, size=400)

    # A treatment effect that varies with the first feature.
    baseline = 2.0 + X[:, 0] - 0.5 * X[:, 1]
    treatment_effect = 1.0 + 0.8 * X[:, 0]
    y = baseline + W * treatment_effect + rng.normal(0, 0.5, 400)

    # The transformed-outcome learner requires nonnegative outcomes.
    y = y - y.min() + 0.1
    return X, W, y


@pytest.mark.parametrize(
    "learner_class",
    [SLearner, TLearner, XLearner, DRLearner],
)
def test_meta_learner_fit_and_predict(learner_class, training_data):
    X, W, y = training_data

    learner = learner_class(model=L2(n_estimators=50, min_child_samples=10))
    learner.fit(X, W, y)
    predictions = learner.predict(X[:30])

    assert predictions.shape == (30,)
    assert np.all(np.isfinite(predictions))


def test_transformed_outcome_learner(training_data):
    X, W, y = training_data

    learner = TransformedOutcomeLearner(
        transform="log1p",
        scale=1.0,
        model=L2(n_estimators=50, min_child_samples=10),
        e=0.5,
    )
    learner.fit(X, W, y)
    predictions = learner.predict(X[:30])

    assert predictions.shape == (30,)
    assert np.all(np.isfinite(predictions))


def test_learner_rejects_missing_treatment_group(training_data):
    X, _, y = training_data
    W = np.zeros(len(y), dtype=int)

    with pytest.raises(ValueError, match="Both treatment groups"):
        TLearner().fit(X, W, y)


def test_learner_rejects_invalid_propensity():
    with pytest.raises(ValueError):
        SLearner(e=1.0)


def test_transformed_learner_rejects_negative_outcomes():
    X = np.ones((20, 3))
    W = np.tile([0, 1], 10)
    y = np.arange(20, dtype=float) - 5

    learner = TransformedOutcomeLearner(transform="log1p")
    with pytest.raises(ValueError):
        learner.fit(X, W, y)
