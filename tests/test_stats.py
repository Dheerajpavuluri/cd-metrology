import numpy as np
import pytest

from cdmetro import stats


def test_robust_z_flags_single_outlier():
    rng = np.random.default_rng(0)
    v = list(rng.normal(20, 0.3, 50)) + [14.0]
    z, out = stats.flag_outliers(v)
    assert out[-1]
    assert out.sum() == 1


def test_robust_z_constant_data():
    z, out = stats.flag_outliers([5.0] * 10)
    assert not out.any()
    assert (z == 0).all()


def test_robust_z_empty():
    z, out = stats.flag_outliers([])
    assert z.size == 0 and out.size == 0


def test_control_limits_use_moving_range():
    v = [10, 12, 10, 12, 10]
    lim = stats.control_limits(v)
    assert lim["centre"] == pytest.approx(10.8)
    assert lim["sigma"] == pytest.approx(2 / 1.128)
    assert lim["ucl"] == pytest.approx(10.8 + 3 * 2 / 1.128)


def test_control_limits_baseline():
    lim = stats.control_limits([1, 1, 1, 50, 60], baseline=3)
    assert lim["centre"] == 1 and lim["sigma"] == 0


def test_accuracy():
    acc = stats.accuracy([1.1, 2.1, 2.9], [1, 2, 3])
    assert acc["bias"] == pytest.approx(0.1 / 3)
    assert acc["max_abs"] == pytest.approx(0.1)
