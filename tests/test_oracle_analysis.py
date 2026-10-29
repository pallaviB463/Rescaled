
import numpy as np
import pandas as pd
import pytest

from src.oracle_analysis import calculate_oracle_table, save_oracle_figure
from src.transforms import SCALES, TRANSFORMS


def test_oracle_table_contains_all_transform_scale_combinations():
    table = calculate_oracle_table(n=500, seed=11)

    assert len(table) == len(TRANSFORMS) * len(SCALES)
    assert set(table["transform"]) == set(TRANSFORMS)
    assert set(table["scale"]) == set(SCALES)
    assert np.all(np.isfinite(table["log10_scale"]))
    assert np.all(table["top20_iou_vs_level"].between(0, 1))


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


def test_oracle_rejects_invalid_scenario_and_scale():
    with pytest.raises(ValueError, match="Unknown scenario"):
        calculate_oracle_table(scenario="invalid", n=100)

    with pytest.raises(ValueError, match="finite and positive"):
        calculate_oracle_table(n=100, scales=[0.0])


def test_oracle_figure_is_written(tmp_path):
    table = pd.DataFrame({
        "transform": ["identity", "identity"],
        "log10_scale": [-1.0, 0.0],
        "rank_corr_vs_level": [1.0, 1.0],
        "top20_iou_vs_level": [1.0, 1.0],
    })

    output_path = tmp_path / "figure1.png"
    save_oracle_figure(table, output_path)

    assert output_path.exists()
    assert output_path.stat().st_size > 0
