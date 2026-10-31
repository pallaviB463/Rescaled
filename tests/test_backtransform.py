
import numpy as np
import pytest

from src.backtransform import backtransform_log1p


def test_backtransform_recovers_known_values():
    original = np.array([0.0, 0.5, 1.0, 5.0, 20.0])
    log_predictions = np.log1p(original)

    result = backtransform_log1p(log_predictions, clip=False)

    np.testing.assert_allclose(result, original, rtol=1e-12, atol=1e-12)


def test_backtransform_clips_negative_predictions_by_default():
    result = backtransform_log1p(np.array([-2.0, 0.0, 1.0]))

    assert result[0] == 0.0
    assert result[1] == 0.0
    assert result[2] == pytest.approx(np.expm1(1.0))


def test_backtransform_can_disable_clipping():
    result = backtransform_log1p(np.array([-1.0]), clip=False)

    assert result[0] == pytest.approx(np.expm1(-1.0))


@pytest.mark.parametrize(
    "values",
    [
        np.array([]),
        np.array([[1.0, 2.0]]),
        np.array([np.nan]),
        np.array([np.inf]),
    ],
)
def test_backtransform_rejects_invalid_inputs(values):
    with pytest.raises(ValueError):
        backtransform_log1p(values)


def test_backtransform_rejects_overflow():
    with pytest.raises(ValueError, match="overflow"):
        backtransform_log1p(np.array([1000.0]))
