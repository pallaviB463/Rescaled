
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
