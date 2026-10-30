import numpy as np
import pandas as pd
import pytest

from src.oracle_analysis import (
    ZERO_RATES,
    calculate_full_oracle_analysis,
    calculate_oracle_table,
    save_full_oracle_outputs,
    save_oracle_figure,
)
from src.dgp import SCENARIOS
from src.transforms import SCALES, TRANSFORMS


def test_oracle_table_contains_all_transform_scale_combinations():
    table = calculate_oracle_table(n=500, seed=11)

    assert len(table) == len(TRANSFORMS) * len(SCALES)
    assert set(table["transform"]) == set(TRANSFORMS)
    assert set(table["scale"]) == set(SCALES)
    assert np.all(np.isfinite(table["log10_scale"]))
    assert np.all(table["top20_iou_vs_level"].between(0, 1))
    assert table["oracle_policy_value_top20"].notna().all()


def test_oracle_identity_ranking_is_scale_invariant():
    table = calculate_oracle_table(
        scenario="opposed",
        n=500,
        seed=12,
        scales=[0.001, 1.0, 1000.0],
    )
    identity = table[table["transform"] == "identity"]

    np.testing.assert_allclose(
        identity["rank_corr_vs_level"],
        1.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        identity["top20_iou_vs_level"],
        1.0,
        atol=1e-12,
    )


def test_oracle_null_scenario_has_undefined_rank_correlation():
    table = calculate_oracle_table(
        scenario="null",
        n=200,
        seed=13,
        scales=[1.0],
    )

    assert table["rank_corr_vs_level"].isna().all()
    assert table["top20_iou_vs_level"].isna().all()


def test_oracle_rejects_invalid_scenario_and_scale():
    with pytest.raises(ValueError, match="Unknown scenario"):
        calculate_oracle_table(scenario="invalid", n=100)

    with pytest.raises(ValueError, match="finite and positive"):
        calculate_oracle_table(n=100, scales=[0.0])


def test_oracle_figure_is_written(tmp_path):
    table = pd.DataFrame({
        "scenario": ["opposed", "opposed"],
        "zero_rate": [0.9, 0.9],
        "transform": ["identity", "identity"],
        "log10_scale": [-1.0, 0.0],
        "rank_corr_vs_level": [1.0, 1.0],
        "top20_iou_vs_level": [1.0, 1.0],
    })

    output_path = tmp_path / "figure1.png"
    save_oracle_figure(table, output_path)

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_full_oracle_analysis_covers_all_scenarios_and_zero_rates():
    ranking, policy = calculate_full_oracle_analysis(
        n=300,
        seed=21,
        scales=[0.1, 1.0],
    )

    expected_ranking_rows = (
        len(SCENARIOS)
        * len(ZERO_RATES)
        * len(TRANSFORMS)
        * 2
    )

    assert len(ranking) == expected_ranking_rows
    assert set(ranking["scenario"]) == set(SCENARIOS)
    assert set(ranking["zero_rate"]) == set(ZERO_RATES)
    assert set(ranking["transform"]) == set(TRANSFORMS)

    assert len(policy) == len(SCENARIOS) * len(ZERO_RATES)
    assert policy["oracle_policy_value_top20"].notna().all()


def test_oracle_policy_value_is_constant_across_transforms_and_scales():
    table = calculate_oracle_table(
        scenario="opposed",
        n=500,
        seed=31,
        zero_rate=0.9,
    )

    assert table["oracle_policy_value_top20"].nunique() == 1


def test_full_oracle_analysis_rejects_invalid_zero_rate():
    with pytest.raises(ValueError, match="Zero rates"):
        calculate_full_oracle_analysis(
            scenarios=["opposed"],
            zero_rates=[1.0],
            n=100,
        )


def test_full_oracle_outputs_are_saved(tmp_path):
    ranking, policy = calculate_full_oracle_analysis(
        scenarios=["opposed"],
        zero_rates=[0.7],
        n=300,
        seed=41,
        scales=[0.1, 1.0],
    )

    save_full_oracle_outputs(
        ranking,
        policy,
        output_dir=tmp_path,
    )

    assert (tmp_path / "oracle_ranking_full_grid.csv").exists()
    assert (tmp_path / "oracle_policy_values.csv").exists()
    assert (
        tmp_path / "figure1_oracle_opposed_zero_0p70.png"
    ).exists()