from __future__ import annotations

import numpy as np
import pytest

from my_torch import SGD


def test_update_applies_learning_rate_and_weight_decay() -> None:
    optimizer = SGD(lr=0.1, weight_decay=0.01)
    param = np.array([[1.0, -2.0]])
    grad = np.array([[0.5, 0.5]])

    updated = optimizer.update(param, grad)

    expected = param - 0.1 * (grad + 0.01 * param)
    assert np.allclose(updated, expected)


def test_update_requires_matching_shapes() -> None:
    optimizer = SGD(lr=0.1)
    with pytest.raises(ValueError, match="same shape"):
        optimizer.update(np.ones((2, 2)), np.ones((2,)))


def test_step_updates_parameters_in_place() -> None:
    optimizer = SGD(lr=0.5)
    params = [np.array([1.0, -1.0]), np.array([[0.5]])]
    grads = [np.array([0.2, 0.2]), np.array([[1.0]])]

    optimizer.step(params, grads)

    assert np.allclose(params[0], np.array([0.9, -1.1]))
    assert np.allclose(params[1], np.array([0.0]))


def test_zero_grad_clears_all_gradients() -> None:
    grads = [np.full((2, 2), 3.0), np.array([[-1.0, 2.0]])]
    SGD.zero_grad(grads)
    assert all(np.all(g == 0) for g in grads)


def test_step_requires_matching_lengths() -> None:
    optimizer = SGD(lr=0.1)
    with pytest.raises(ValueError, match="same length"):
        optimizer.step([np.ones((2, 2))], [])


@pytest.mark.parametrize("lr, weight_decay", [(-0.1, 0.0), (0.0, 0.0), (0.1, -0.5)])
def test_invalid_hyperparameters_raise(lr: float, weight_decay: float) -> None:
    with pytest.raises(ValueError):
        SGD(lr=lr, weight_decay=weight_decay)
