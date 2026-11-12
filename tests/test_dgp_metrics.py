import numpy as np
from dgp import simulate, true_transformed_cate, SCENARIOS
from metrics import rank_corr, regret, policy_value, topk_overlap, qini_auc


def test_null_effect_is_exactly_zero():
    _, _, _, T = simulate(2000, "null", 0.9, seed=1)
    assert np.all(T["tau_level"] == 0)


def test_zero_rate_calibrated():
    for zr in (0.7, 0.9, 0.98):
        # calibration targets the CONTROL arm's zero rate; in "null" treated = control
        _, _, Y, _ = simulate(200_000, "null", zr, seed=2)
        assert abs((Y == 0).mean() - zr) < 0.01, zr


def _se(a, b): return np.sqrt(a.var() / len(a) + b.var() / len(b))


def test_transformed_ate_matches_observed_difference():
    X, W, Y, T = simulate(400_000, "opposed", 0.9, seed=3)
    y = np.log1p(Y)
    diff = y[W == 1].mean() - y[W == 0].mean()
    assert abs(diff - true_transformed_cate(T, np.log1p, 1.0).mean()) < 4 * _se(y[W == 1], y[W == 0])


def test_level_ate_matches_observed_difference():
    X, W, Y, T = simulate(400_000, "homogeneous", 0.9, seed=4)
    diff = Y[W == 1].mean() - Y[W == 0].mean()
    assert abs(diff - T["tau_level"].mean()) < 4 * _se(Y[W == 1], Y[W == 0])


def test_metrics_basics():
    rng = np.random.default_rng(0)
    tau = rng.normal(size=1000)
    assert rank_corr(tau, tau) == 1
    assert regret(tau, tau) == 0
    assert regret(rng.normal(size=1000), tau) >= -1e-12
    assert topk_overlap(tau, tau) == 1
    assert np.isnan(rank_corr(np.zeros(10), tau[:10]))      # constant vector guard (null scenario)
