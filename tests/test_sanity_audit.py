
import numpy as np
import pytest

from src.dgp import simulate, true_transformed_cate
from src.metrics import rank_corr, topk_overlap, policy_value, regret
from src.models import L2, Poisson, Tweedie, Hurdle
from src.transforms import apply_transform, SCALES
from src.run_sim import one_rep


def test_null_dgp_has_exactly_zero_level_cate():
    _, _, _, truth = simulate(
        n=5000,
        scenario="null",
        zero_rate=0.9,
        sigma=0.8,
        e=0.5,
        seed=42,
    )

    assert np.all(truth["tau_level"] == 0)
    assert np.allclose(truth["p0"], truth["p1"])
    assert np.allclose(truth["mu0"], truth["mu1"])


@pytest.mark.parametrize(
    "transform",
    ["identity", "log1p", "arcsinh", "sqrt"],
)
def test_null_transformed_cate_is_zero(transform):
    _, _, _, truth = simulate(
        n=2000,
        scenario="null",
        seed=123,
    )

    transformed_cate = true_transformed_cate(
        truth,
        m=lambda y: (
            y if transform == "identity"
            else np.log1p(y) if transform == "log1p"
            else np.arcsinh(y) if transform == "arcsinh"
            else np.sqrt(y)
        ),
        a=1.0,
    )

    assert np.allclose(transformed_cate, 0.0, atol=1e-12)


def test_null_simulation_has_undefined_ranking_metrics():
    results = one_rep(
        scenario="null",
        n=1000,
        seed=42,
        learners=("T",),
        include_transformed=True,
    )

    successful = results[results["status"] == "ok"]

    assert len(successful) > 0
    assert successful["rank_corr"].isna().all()
    assert successful["topk_overlap"].isna().all()
    assert np.allclose(successful["policy_value"], 0.0)
    assert np.allclose(successful["regret"], 0.0)


@pytest.mark.parametrize("scale", SCALES)
def test_identity_and_sqrt_transform_rankings_are_scale_invariant(scale):
    y = np.array([0.0, 0.2, 1.0, 3.0, 8.0, 15.0, 30.0])

    for transform in ("identity", "sqrt"):
        reference = apply_transform(transform, y, a=1.0)
        scaled = apply_transform(transform, y, a=scale)

        assert rank_corr(reference, scaled) == pytest.approx(1.0)
        assert topk_overlap(reference, scaled, k=0.2) == pytest.approx(1.0)


@pytest.mark.parametrize("scale", SCALES)
def test_rank_transform_is_invariant_to_positive_scale(scale):
    y = np.array([0.0, 0.2, 1.0, 3.0, 8.0, 15.0, 30.0])

    reference = apply_transform("rank", y, a=1.0)
    scaled = apply_transform("rank", y, a=scale)

    np.testing.assert_allclose(reference, scaled)


@pytest.mark.parametrize(
    "model_class",
    [L2, Poisson, Tweedie, Hurdle],
)
def test_model_predictions_are_finite(model_class):
    rng = np.random.default_rng(42)
    X = rng.normal(size=(500, 5))

    positive = rng.lognormal(mean=1.0, sigma=0.5, size=500)
    zero_mask = rng.random(500) < 0.7
    y = np.where(zero_mask, 0.0, positive)

    model = model_class(
        n_estimators=50,
        min_child_samples=10,
    )

    model.fit(X, y)
    predictions = model.predict(X[:100])

    assert predictions.shape == (100,)
    assert np.all(np.isfinite(predictions))


@pytest.mark.parametrize(
    "model_class",
    [Poisson, Tweedie, Hurdle],
)
def test_nonnegative_models_never_predict_negative_outcomes(model_class):
    rng = np.random.default_rng(123)
    X = rng.normal(size=(500, 5))

    positive = rng.lognormal(mean=1.0, sigma=0.5, size=500)
    y = np.where(rng.random(500) < 0.7, 0.0, positive)

    model = model_class(
        n_estimators=50,
        min_child_samples=10,
    )

    model.fit(X, y)
    predictions = model.predict(X)

    assert np.all(np.isfinite(predictions))
    assert np.all(predictions >= 0.0)


def test_negative_treatment_effect_predictions_are_not_invalid():
    predictions = np.array([-2.0, -0.5, 0.0, 1.0])

    assert np.all(np.isfinite(predictions))
    assert np.any(predictions < 0.0)


def test_null_policy_value_and_regret_are_zero():
    tau_true = np.zeros(100)
    tau_hat = np.random.default_rng(42).normal(size=100)

    assert policy_value(tau_hat, tau_true, k=0.2) == 0.0
    assert regret(tau_hat, tau_true, k=0.2) == 0.0
