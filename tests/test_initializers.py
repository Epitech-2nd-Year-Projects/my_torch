from __future__ import annotations

import numpy as np
import pytest

from my_torch import get_initializer, initialize_bias, initialize_weights
from my_torch.initializers import _fan_in_out


def test_xavier_uniform_variance_matches_expectation() -> None:
    rng = np.random.default_rng(0)
    shape = (128, 64)
    weights = initialize_weights(shape, mode="xavier", rng=rng)
    fan_in, fan_out = 64, 128
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    expected_variance = (limit**2) / 3
    assert np.isclose(np.var(weights), expected_variance, rtol=0.1)


def test_he_normal_variance_matches_expectation() -> None:
    rng = np.random.default_rng(1)
    shape = (64, 128)
    weights = initialize_weights(shape, mode="he", rng=rng)
    fan_in = 128
    expected_variance = 2.0 / fan_in
    assert np.isclose(np.var(weights), expected_variance, rtol=0.1)


def test_xavier_normal_variance_matches_expectation() -> None:
    rng = np.random.default_rng(7)
    shape = (128, 64)
    weights = initialize_weights(shape, mode="xavier_normal", rng=rng)
    fan_in, fan_out = 64, 128
    expected_variance = 2.0 / (fan_in + fan_out)
    assert np.isclose(np.var(weights), expected_variance, rtol=0.1)


def test_normal_fallback_uses_small_stddev() -> None:
    rng = np.random.default_rng(2)
    weights = initialize_weights((10_000,), mode="normal", rng=rng)
    assert np.isclose(np.std(weights), 0.01, rtol=0.1)


def test_he_uniform_variance_matches_expectation() -> None:
    rng = np.random.default_rng(5)
    shape = (256, 128)
    weights = initialize_weights(shape, mode="he_uniform", rng=rng)
    fan_in = 128
    limit = np.sqrt(6.0 / fan_in)
    expected_variance = (limit**2) / 3
    assert np.isclose(np.var(weights), expected_variance, rtol=0.1)


def test_simple_uniform_respects_bounds() -> None:
    rng = np.random.default_rng(6)
    weights = initialize_weights((5_000,), mode="uniform", rng=rng)
    assert np.all(weights <= 0.05)
    assert np.all(weights >= -0.05)
    assert np.var(weights) > 0


def test_initialize_bias_modes() -> None:
    zeros = initialize_bias((3,))
    normal = initialize_bias((3,), mode="normal", rng=np.random.default_rng(3))
    uniform = initialize_bias((3,), mode="uniform", rng=np.random.default_rng(4))
    assert np.all(zeros == 0.0)
    assert zeros.shape == normal.shape == uniform.shape
    assert not np.all(normal == normal[0])
    assert np.all(np.abs(uniform) <= 0.05)
    assert not np.all(uniform == uniform[0])


def test_initialize_bias_invalid_mode_raises_value_error() -> None:
    with pytest.raises(ValueError):
        initialize_bias((2,), mode="invalid")


def test_unknown_initializer_key_raises_value_error() -> None:
    with pytest.raises(ValueError):
        get_initializer("does-not-exist")((2, 2))


def test_fan_in_out_empty_shape_raises_value_error() -> None:
    with pytest.raises(ValueError):
        _fan_in_out(())


def test_fan_in_out_1d_shape() -> None:
    fan_in, fan_out = _fan_in_out((16,))
    assert fan_in == 16
    assert fan_out == 16


def test_fan_in_out_2d_shape() -> None:
    fan_in, fan_out = _fan_in_out((32, 64))
    assert fan_in == 64
    assert fan_out == 32
