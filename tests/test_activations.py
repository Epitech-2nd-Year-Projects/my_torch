from __future__ import annotations

import numpy as np

from my_torch import (
    relu,
    relu_derivative,
    sigmoid,
    sigmoid_derivative,
    tanh,
    tanh_derivative,
    softmax,
    softmax_derivative,
)


def test_relu_monotonicity_and_derivative_shape() -> None:
    x = np.linspace(-2, 2, 9)
    y = relu(x)
    assert y.shape == x.shape
    assert np.all(np.diff(y) >= 0)

    dy = relu_derivative(x)
    assert dy.shape == x.shape
    assert np.array_equal(dy[x < 0], np.zeros_like(dy[x < 0]))
    assert np.all(dy[x > 0] == 1)


def test_sigmoid_range_monotonicity_and_derivative() -> None:
    x = np.linspace(-8, 8, 17)
    y = sigmoid(x)
    assert y.shape == x.shape
    assert np.all((0 <= y) & (y <= 1))
    assert np.all(np.diff(y) >= 0)

    dy = sigmoid_derivative(x)
    assert dy.shape == x.shape
    assert np.all(dy > 0)
    assert dy[np.argmax(dy)] == np.max(dy)


def test_tanh_range_monotonicity_and_derivative() -> None:
    x = np.linspace(-4, 4, 17)
    y = tanh(x)
    assert y.shape == x.shape
    assert np.all((-1 <= y) & (y <= 1))
    assert np.all(np.diff(y) >= 0)

    dy = tanh_derivative(x)
    assert dy.shape == x.shape
    assert np.all(dy > 0)
    assert np.isclose(np.max(dy), 1.0, atol=1e-6)


def test_softmax_probabilities_and_derivative_shape() -> None:
    x = np.array([[1.0, 0.5, -1.0], [0.2, 0.2, 0.2]])
    y = softmax(x, axis=1)
    assert y.shape == x.shape
    assert np.all(y >= 0)
    assert np.allclose(np.sum(y, axis=1), 1.0)

    jacobian = softmax_derivative(x, axis=1)
    assert jacobian.shape == x.shape + (x.shape[1],)

    row_sums = jacobian.sum(axis=-1)
    assert np.allclose(row_sums, 0.0)

    diagonal = np.diagonal(jacobian, axis1=-2, axis2=-1)
    assert np.all(diagonal > 0)
    assert np.all(jacobian[..., ~np.eye(x.shape[1], dtype=bool)] <= 0)


def test_softmax_monotonic_in_target_coordinate() -> None:
    base = np.array([0.1, -0.2, 0.0])
    boosted = base.copy()
    boosted[0] += 1.5

    p_base = softmax(base)
    p_boosted = softmax(boosted)

    assert p_boosted.shape == p_base.shape
    assert p_boosted[0] > p_base[0]
