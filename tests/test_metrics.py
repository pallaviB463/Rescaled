
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
    assert rank_corr([0, 0, 0], [1, 2, 3]) == 0.0
    assert rank_corr([1, 2, 3], [5, 5, 5]) == 0.0
    assert rank_corr([0, 0, 0], [0, 0, 0]) == 0.0


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
