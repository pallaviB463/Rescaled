
import numpy as np
import pytest

from src.transforms import SCALES, TRANSFORMS, apply_transform


def test_all_required_transformations_are_available():
    assert set(TRANSFORMS) == {"identity", "log1p", "arcsinh", "sqrt"}
    assert len(SCALES) == 7
    assert 1.0 in SCALES


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("identity", np.array([0.0, 1.0, 4.0])),
        ("log1p", np.log1p([0.0, 1.0, 4.0])),
        ("arcsinh", np.arcsinh([0.0, 1.0, 4.0])),
        ("sqrt", np.sqrt([0.0, 1.0, 4.0])),
    ],
)
def test_transform_values(name, expected):
    result = apply_transform(name, [0.0, 1.0, 4.0])
    np.testing.assert_allclose(result, expected)


def test_scale_is_applied_before_transformation():
    y = np.array([1.0, 2.0, 3.0])
    np.testing.assert_allclose(
        apply_transform("log1p", y, a=10.0),
        np.log1p(10.0 * y),
    )


def test_rank_is_invariant_to_positive_scale():
    y = np.array([0.0, 4.0, 2.0, 2.0, 10.0])
    reference = apply_transform("rank", y, a=1.0)

    for scale in SCALES:
        np.testing.assert_allclose(
            apply_transform("rank", y, a=scale),
            reference,
        )


def test_rank_handles_ties():
    result = apply_transform("rank", [1.0, 1.0, 3.0])
    np.testing.assert_allclose(result, [0.5, 0.5, 1.0])


@pytest.mark.parametrize("name", ["log1p", "arcsinh", "sqrt"])
def test_transform_output_is_finite(name):
    result = apply_transform(name, [0.0, 1.0, 10.0], a=100.0)
    assert np.all(np.isfinite(result))


def test_invalid_scale_is_rejected():
    with pytest.raises(ValueError):
        apply_transform("log1p", [1.0, 2.0], a=0)


def test_negative_outcomes_are_rejected():
    with pytest.raises(ValueError):
        apply_transform("sqrt", [-1.0, 2.0])


def test_unknown_transform_is_rejected():
    with pytest.raises(ValueError):
        apply_transform("unknown", [1.0, 2.0])
