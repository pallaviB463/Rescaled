"""Tests for count-aware outcome-model selection in the simulation runner."""

import pytest

from src.models import Hurdle, L2, Tweedie
from src.run_sim import _make_model, one_rep


@pytest.mark.parametrize(
    ("name", "expected_type"),
    [
        ("L2", L2),
        ("Tweedie", Tweedie),
        ("Hurdle", Hurdle),
    ],
)
def test_make_model_supports_registered_outcome_models(name, expected_type):
    model = _make_model(name)
    assert isinstance(model, expected_type)


def test_make_model_rejects_unknown_model():
    with pytest.raises(ValueError, match="Unknown outcome model"):
        _make_model("Poisson")


def test_count_aware_model_rejects_non_t_learner():
    with pytest.raises(ValueError, match="supported only with the T-learner"):
        one_rep(
            scenario="opposed",
            n=10,
            learners=("DR",),
            include_transformed=False,
            outcome_model="Tweedie",
        )


def test_l2_remains_default_model():
    assert isinstance(_make_model(), L2)
