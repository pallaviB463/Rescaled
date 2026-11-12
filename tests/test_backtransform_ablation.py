
import numpy as np
import pytest

from src.run_backtransform_ablation import run_backtransform_ablation


def test_backtransform_ablation_returns_both_methods(tmp_path):
    output = tmp_path / "p13_results.csv"

    results, summary = run_backtransform_ablation(
        scenario="opposed",
        n=1000,
        zero_rate=0.9,
        sigma=0.8,
        e=0.5,
        seeds=[42],
        output=output,
    )

    assert set(results["method"]) == {
        "TransformedOutcome",
        "BackTransform",
    }

    assert len(results) == 2
    assert output.exists()
    assert set(results["status"]) == {"ok"}

    assert np.all(np.isfinite(results["cate_rmse"]))
    assert np.all(np.isfinite(results["policy_value"]))
    assert np.all(results["regret"] >= -1e-10)

    assert set(summary.index) == {
        "TransformedOutcome",
        "BackTransform",
    }


def test_backtransform_ablation_uses_same_test_size(tmp_path):
    results, _ = run_backtransform_ablation(
        n=1000,
        seeds=[42],
        output=tmp_path / "results.csv",
    )

    assert results["test_size"].nunique() == 1
    assert results["test_size"].iloc[0] == 200


def test_backtransform_ablation_rejects_invalid_test_size(tmp_path):
    with pytest.raises(ValueError, match="test_size"):
        run_backtransform_ablation(
            n=1000,
            seeds=[42],
            test_size=1.0,
            output=tmp_path / "results.csv",
        )
