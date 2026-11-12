
import numpy as np
import pytest

from src.run_backtransform import run_backtransform


def test_backtransform_experiment_returns_valid_result():
    results = run_backtransform(
        scenario="opposed",
        n=300,
        seed=17,
    )

    required = {
        "scenario",
        "learner",
        "transform",
        "scale",
        "rank_corr",
        "topk_overlap",
        "policy_value",
        "regret",
        "status",
        "error",
    }

    assert required.issubset(results.columns)
    assert len(results) == 1
    assert results.loc[0, "learner"] == "NaiveBackTransform"
    assert results.loc[0, "transform"] == "log1p"
    assert results.loc[0, "status"] == "ok"
    assert np.isfinite(results.loc[0, "policy_value"])
    assert np.isfinite(results.loc[0, "regret"])


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"n": 1}, "n must be at least 2"),
        ({"e": 0.0}, "e must be strictly between 0 and 1"),
        ({"zero_rate": 1.0}, "zero_rate must be strictly between 0 and 1"),
        ({"sigma": -0.1}, "sigma must be finite and nonnegative"),
    ],
)
def test_backtransform_experiment_rejects_invalid_parameters(kwargs, message):
    with pytest.raises(ValueError, match=message):
        run_backtransform(**kwargs)
