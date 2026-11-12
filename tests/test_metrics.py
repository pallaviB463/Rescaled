
import numpy as np
import pytest

from src.metrics import (
    rank_corr,
    topk_mask,
    topk_overlap,
    policy_value,
    regret,
    qini_auc,
    ipw_policy_value,
)


def test_rank_corr_for_perfect_rankings():
    assert rank_corr([1, 2, 3], [10, 20, 30]) == pytest.approx(1.0)


def test_rank_corr_handles_constant_vectors():
    assert np.isnan(rank_corr([0, 0, 0], [1, 2, 3]))
    assert np.isnan(rank_corr([1, 2, 3], [0, 0, 0]))


def test_topk_mask_selects_correct_number():
    mask = topk_mask([1, 4, 2, 3, 5], 0.4)

    assert mask.sum() == 2
    assert np.array_equal(mask, [False, True, False, False, True])


def test_topk_overlap_identical_rankings():
    scores = [1, 5, 3, 2, 4]

    assert topk_overlap(scores, scores, 0.4) == pytest.approx(1.0)


def test_policy_value_and_regret():
    true_effect = np.array([5.0, 4.0, 1.0, 0.0])
    predicted = np.array([4.0, 5.0, 0.0, 1.0])

    assert policy_value(predicted, true_effect, 0.5) == pytest.approx(4.5)
    assert regret(predicted, true_effect, 0.5) == pytest.approx(0.0)


def test_qini_auc_is_finite():
    result = qini_auc(
        tau_hat=[0.9, 0.8, 0.2, 0.1],
        W=[1, 0, 1, 0],
        y=[3.0, 1.0, 4.0, 2.0],
        e=0.5,
    )

    assert np.isfinite(result)


def test_ipw_policy_value_is_finite():
    result = ipw_policy_value(
        tau_hat=[0.9, 0.8, 0.2, 0.1],
        W=[1, 0, 1, 0],
        y=[3.0, 1.0, 4.0, 2.0],
        k=0.5,
        e=0.5,
    )

    assert np.isfinite(result)


def test_rank_corr_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        rank_corr([1, 2], [1, 2, 3])


def test_rank_corr_returns_nan_for_constant_truth():
    result = rank_corr([0.1, 0.2, 0.3], [0.0, 0.0, 0.0])

    assert np.isnan(result)


def test_rank_corr_returns_nan_for_constant_predictions():
    result = rank_corr([1.0, 1.0, 1.0], [0.1, 0.2, 0.3])

    assert np.isnan(result)


def test_topk_overlap_matches_intersection_over_union():
    # Each ranking selects two of five observations.
    # The selected sets overlap in one observation.
    # IoU = intersection / union = 1 / 3.
    a = [5, 4, 3, 2, 1]
    b = [1, 5, 4, 3, 2]

    assert topk_overlap(a, b, k=0.4) == pytest.approx(1 / 3)


def test_policy_regret_is_positive_for_suboptimal_selection():
    true_effect = np.array([10.0, 8.0, 2.0, 0.0])
    predicted = np.array([0.0, 1.0, 9.0, 10.0])

    # Predicted policy selects true effects 2 and 0.
    # Oracle policy selects true effects 10 and 8.
    assert policy_value(predicted, true_effect, k=0.5) == pytest.approx(1.0)
    assert regret(predicted, true_effect, k=0.5) == pytest.approx(8.0)


@pytest.mark.parametrize(
    "metric",
    [qini_auc, ipw_policy_value],
)
@pytest.mark.parametrize(
    "kwargs",
    [
        {"e": 0.0},
        {"e": 1.0},
    ],
)
def test_ipw_metrics_reject_invalid_propensity(metric, kwargs):
    with pytest.raises(ValueError):
        metric(
            tau_hat=[0.9, 0.1],
            W=[1, 0],
            y=[2.0, 1.0],
            **kwargs,
        )


@pytest.mark.parametrize(
    "metric",
    [qini_auc, ipw_policy_value],
)
def test_ipw_metrics_reject_mismatched_lengths(metric):
    with pytest.raises(ValueError):
        metric(
            tau_hat=[0.9, 0.1],
            W=[1, 0],
            y=[2.0],
        )


@pytest.mark.parametrize(
    "metric",
    [qini_auc, ipw_policy_value],
)
def test_ipw_metrics_reject_nonbinary_treatment(metric):
    with pytest.raises(ValueError):
        metric(
            tau_hat=[0.9, 0.1],
            W=[1, 2],
            y=[2.0, 1.0],
        )


@pytest.mark.parametrize(
    "metric",
    [qini_auc, ipw_policy_value],
)
def test_ipw_metrics_reject_nonfinite_inputs(metric):
    with pytest.raises(ValueError):
        metric(
            tau_hat=[0.9, np.nan],
            W=[1, 0],
            y=[2.0, 1.0],
        )

    with pytest.raises(ValueError):
        metric(
            tau_hat=[0.9, 0.1],
            W=[1, 0],
            y=[2.0, np.inf],
        )


def test_ipw_policy_value_matches_hand_calculation():
    # The top two observations receive treatment.
    # Observations 1 and 3 match their assigned policy.
    # Contributions: 2/0.5 = 4 and 1/0.5 = 2.
    # Final estimate: (4 + 2) / 4 = 1.5.
    result = ipw_policy_value(
        tau_hat=[0.9, 0.8, 0.1, 0.0],
        W=[1, 0, 0, 1],
        y=[2.0, 4.0, 1.0, 3.0],
        k=0.5,
        e=0.5,
    )

    assert result == pytest.approx(1.5)


def test_random_score_regret_matches_oracle_minus_average_effect():
    true_effect = np.array([10.0, 8.0, 2.0, 0.0])
    random_scores = np.array([0.1, 0.9, 0.2, 0.8])

    # Oracle targets the top 50%.
    oracle_value = policy_value(
        true_effect, true_effect, k=0.5
    )

    # Selecting everyone gives the average treatment effect.
    random_policy_value = policy_value(
        random_scores, true_effect, k=1.0
    )

    expected_regret = oracle_value - np.mean(true_effect)

    assert oracle_value == pytest.approx(9.0)
    assert random_policy_value == pytest.approx(5.0)
    assert expected_regret == pytest.approx(4.0)
    assert (
        oracle_value - random_policy_value
        == pytest.approx(expected_regret)
    )


def test_regret_is_nonnegative_for_suboptimal_predictions():
    true_effect = np.array([10.0, 8.0, 2.0, 0.0])
    predicted = np.array([0.0, 1.0, 9.0, 10.0])

    assert regret(predicted, true_effect, k=0.5) >= 0.0


def test_rank_corr_identical_values():
    values = np.array([3.0, 1.0, 5.0, 2.0, 4.0])

    assert rank_corr(values, values) == pytest.approx(1.0)


def test_rank_corr_rejects_nonfinite_values():
    with pytest.raises(ValueError):
        rank_corr([1.0, np.nan], [1.0, 2.0])


def test_policy_value_rejects_nonfinite_truth():
    with pytest.raises(ValueError):
        policy_value([1.0, 2.0], [1.0, np.nan])


def test_topk_mask_rejects_invalid_k():
    with pytest.raises(ValueError):
        topk_mask([1.0, 2.0], 0.0)

    with pytest.raises(ValueError):
        topk_mask([1.0, 2.0], 1.5)


def test_qini_auc_oracle_exceeds_random_ranking():
    tau_true = np.array([10.0, 8.0, 2.0, 0.0, -2.0, -4.0])
    W = np.array([1, 0, 1, 0, 1, 0])
    y = np.array([10.0, 8.0, 2.0, 0.0, -2.0, -4.0])

    oracle = qini_auc(tau_true, W, y, e=0.5)

    random = qini_auc(
        np.array([0.2, 0.8, 0.1, 0.9, 0.3, 0.7]),
        W,
        y,
        e=0.5,
    )

    assert oracle > random
