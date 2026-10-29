
import numpy as np

from src.dgp import SCENARIOS, simulate, true_transformed_cate


def test_all_scenarios_return_expected_shapes():
    for scenario in SCENARIOS:
        X, W, Y, truth = simulate(
            n=200,
            scenario=scenario,
            zero_rate=0.9,
            seed=42,
        )

        assert X.shape == (200, 10)
        assert W.shape == (200,)
        assert Y.shape == (200,)
        assert truth["tau_level"].shape == (200,)


def test_null_has_exactly_zero_level_effect():
    _, _, _, truth = simulate(n=200, scenario="null", seed=1)

    np.testing.assert_array_equal(truth["tau_level"], np.zeros(200))


def test_outcomes_are_nonnegative_and_finite():
    _, _, Y, _ = simulate(n=500, scenario="opposed", seed=2)

    assert np.all(np.isfinite(Y))
    assert np.all(Y >= 0)


def test_transformed_cate_is_finite():
    _, _, _, truth = simulate(n=500, scenario="opposed", seed=3)

    tau = true_transformed_cate(truth, np.log1p, a=1.0)

    assert tau.shape == (500,)
    assert np.all(np.isfinite(tau))

def test_control_zero_rate_is_calibrated_within_one_percentage_point():
    import numpy as np
    from src.dgp import simulate

    _, W, Y, _ = simulate(
        n=100_000,
        scenario="opposed",
        zero_rate=0.9,
        seed=42,
    )

    observed_control_zero_rate = np.mean(Y[W == 0] == 0)

    assert abs(observed_control_zero_rate - 0.9) <= 0.01

def test_level_cate_matches_observed_treated_minus_control_mean():
    import numpy as np
    from src.dgp import simulate

    _, W, Y, truth = simulate(
        n=400_000,
        scenario="opposed",
        zero_rate=0.9,
        seed=123,
    )

    observed_difference = Y[W == 1].mean() - Y[W == 0].mean()
    expected_difference = truth["tau_level"].mean()

    assert abs(observed_difference - expected_difference) < 0.15

def test_transformed_cate_matches_observed_log1p_difference():
    import numpy as np
    from src.dgp import simulate, true_transformed_cate

    _, W, Y, truth = simulate(
        n=400_000,
        scenario="opposed",
        zero_rate=0.9,
        seed=456,
    )

    true_cate = true_transformed_cate(
        truth,
        m=np.log1p,
        a=1.0,
    )

    observed_difference = (
        np.log1p(Y[W == 1]).mean()
        - np.log1p(Y[W == 0]).mean()
    )

    expected_difference = true_cate.mean()

    assert abs(observed_difference - expected_difference) < 0.02

def test_simulate_from_X_preserves_supplied_covariates_and_assignments():
    from src.dgp import simulate_from_X

    rng = np.random.default_rng(101)
    X = rng.normal(size=(300, 10))
    W = np.tile([0, 1], 150)

    Y, truth = simulate_from_X(
        X, W, scenario="opposed", zero_rate=0.9, seed=42
    )

    assert Y.shape == (300,)
    assert truth["tau_level"].shape == (300,)
    assert np.all(np.isfinite(Y))
    assert np.all(Y >= 0)
    assert set(np.unique(W)) == {0, 1}


def test_simulate_from_X_is_reproducible():
    from src.dgp import simulate_from_X

    rng = np.random.default_rng(102)
    X = rng.normal(size=(100, 10))
    W = rng.binomial(1, 0.5, size=100)

    Y1, truth1 = simulate_from_X(X, W, seed=17)
    Y2, truth2 = simulate_from_X(X, W, seed=17)

    np.testing.assert_array_equal(Y1, Y2)
    np.testing.assert_array_equal(truth1["tau_level"], truth2["tau_level"])


def test_simulate_from_X_null_has_zero_treatment_effect():
    from src.dgp import simulate_from_X

    rng = np.random.default_rng(103)
    X = rng.normal(size=(100, 10))
    W = rng.binomial(1, 0.5, size=100)

    _, truth = simulate_from_X(X, W, scenario="null")

    np.testing.assert_array_equal(truth["tau_level"], np.zeros(100))


def test_simulate_from_X_rejects_invalid_inputs():
    import pytest
    from src.dgp import simulate_from_X

    X = np.zeros((10, 10))
    W = np.zeros(10)

    with pytest.raises(ValueError):
        simulate_from_X(X, W, scenario="unknown")

    with pytest.raises(ValueError):
        simulate_from_X(X, np.full(10, 2))

    with pytest.raises(ValueError):
        simulate_from_X(X, np.zeros(9))

    with pytest.raises(ValueError):
        simulate_from_X(X, W, sigma=-1)

def test_identity_transformed_cate_equals_scaled_level_cate():
    from src.dgp import true_transformed_cate

    _, _, _, truth = simulate(
        n=200,
        scenario="opposed",
        zero_rate=0.9,
        seed=17,
    )

    for scale in (0.001, 0.1, 1.0, 100.0):
        transformed_cate = true_transformed_cate(
            truth,
            m=lambda y: y,
            a=scale,
        )

        np.testing.assert_allclose(
            transformed_cate,
            scale * truth["tau_level"],
            rtol=1e-10,
            atol=1e-10,
        )


def test_null_scenario_has_zero_transformed_cate():
    from src.dgp import true_transformed_cate

    _, _, _, truth = simulate(
        n=200,
        scenario="null",
        zero_rate=0.9,
        seed=18,
    )

    for transform in (np.log1p, np.arcsinh, np.sqrt):
        transformed_cate = true_transformed_cate(
            truth,
            m=transform,
            a=10.0,
        )

        np.testing.assert_allclose(
            transformed_cate,
            np.zeros(200),
            atol=1e-10,
        )


def test_nonlinear_transformed_cate_is_finite_across_scales():
    from src.dgp import true_transformed_cate

    _, _, _, truth = simulate(
        n=200,
        scenario="opposed",
        zero_rate=0.9,
        seed=19,
    )

    for transform in (np.log1p, np.arcsinh):
        for scale in (0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0):
            transformed_cate = true_transformed_cate(
                truth,
                m=transform,
                a=scale,
            )

            assert transformed_cate.shape == (200,)
            assert np.all(np.isfinite(transformed_cate))
