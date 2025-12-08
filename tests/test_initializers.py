from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from my_torch import get_initializer, initialize_bias, initialize_weights


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


def test_normal_fallback_uses_small_stddev() -> None:
    rng = np.random.default_rng(2)
    weights = initialize_weights((10_000,), mode="normal", rng=rng)
    assert np.isclose(np.std(weights), 0.01, rtol=0.1)


def test_initialize_bias_modes() -> None:
    zeros = initialize_bias((3,))
    normal = initialize_bias((3,), mode="normal", rng=np.random.default_rng(3))
    uniform = initialize_bias((3,), mode="uniform", rng=np.random.default_rng(4))
    assert np.all(zeros == 0.0)
    assert zeros.shape == normal.shape == uniform.shape
    assert not np.all(normal == normal[0])
    assert np.all(np.abs(uniform) <= 0.05)
    assert not np.all(uniform == uniform[0])


def test_unknown_initializer_key_raises_value_error() -> None:
    with pytest.raises(ValueError):
        get_initializer("does-not-exist")((2, 2))
