
import numpy as np
import pytest

from src.run_sim import pipelines, one_rep


def test_pipelines_include_reference_scale():
    configurations = pipelines()
    assert any(config["scale"] == 1.0 for config in configurations)


def test_pipelines_reject_missing_reference_scale():
    with pytest.raises(ValueError, match="reference scale"):
        pipelines(scales=[0.01, 0.1, 10.0])


def test_rank_transformation_has_one_scale_configuration():
    configurations = pipelines()
    rank_configs = [
        config for config in configurations
        if config["transform"] == "rank"
    ]
    assert len(rank_configs) == 1
    assert rank_configs[0]["scale"] == 1.0


def test_one_rep_returns_expected_columns():
    results = one_rep(
        scenario="null",
        n=300,
        seed=11,
        learners=("T",),
        include_transformed=False,
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


def test_one_rep_rejects_missing_reference_scale():
    with pytest.raises(ValueError, match="reference scale"):
        one_rep(
            scenario="null",
            n=300,
            scales=[0.01, 0.1, 10.0],
            include_transformed=False,
        )


def test_one_rep_rejects_invalid_scenario():
    with pytest.raises(ValueError, match="Unknown scenario"):
        one_rep(scenario="not_a_scenario", n=300)
