from __future__ import annotations

from typing import Callable

import numpy as np
import pytest

from my_torch import cross_entropy_grad, cross_entropy_loss, mse_grad, mse_loss


def numerical_gradient(
    func: Callable[[np.ndarray], float], values: np.ndarray, epsilon: float = 1e-5
) -> np.ndarray:
    grad = np.zeros_like(values, dtype=float)
    for index in np.ndindex(values.shape):
        original = values[index]
        values[index] = original + epsilon
        plus = func(values)
        values[index] = original - epsilon
        minus = func(values)
        values[index] = original
        grad[index] = (plus - minus) / (2 * epsilon)
    return grad


def test_cross_entropy_gradient_matches_finite_difference() -> None:
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(3, 4))
    target = np.array([0, 2, 3])

    def loss_fn(current_logits: np.ndarray) -> float:
        return cross_entropy_loss(current_logits, target)

    numerical = numerical_gradient(loss_fn, logits.copy())
    analytical = cross_entropy_grad(logits, target)
    assert analytical.shape == logits.shape
    assert np.allclose(analytical, numerical, atol=1e-6, rtol=1e-6)


def test_cross_entropy_loss_matches_manual_value() -> None:
    logits = np.array([[2.0, 0.0, -2.0], [0.0, 0.0, 2.0]])
    target = np.array([0, 2])
    probs = np.array(
        [
            np.exp([2.0, 0.0, -2.0]) / np.sum(np.exp([2.0, 0.0, -2.0])),
            np.exp([0.0, 0.0, 2.0]) / np.sum(np.exp([0.0, 0.0, 2.0])),
        ]
    )
    manual_loss = -np.log(probs[0, 0]) - np.log(probs[1, 2])
    manual_loss /= 2
    computed_loss = cross_entropy_loss(logits, target)
    assert np.allclose(computed_loss, manual_loss)


def test_mse_gradient_matches_finite_difference() -> None:
    rng = np.random.default_rng(1)
    prediction = rng.normal(size=(2, 3))
    target = rng.normal(size=(2, 3))

    def loss_fn(current_prediction: np.ndarray) -> float:
        return mse_loss(current_prediction, target)

    numerical = numerical_gradient(loss_fn, prediction.copy())
    analytical = mse_grad(prediction, target)
    assert analytical.shape == prediction.shape
    assert np.allclose(analytical, numerical, atol=1e-9, rtol=1e-7)


def test_mse_loss_matches_manual_value() -> None:
    prediction = np.array([[1.0, 2.0], [3.0, 4.0]])
    target = np.array([[0.0, 1.0], [4.0, 2.0]])
    manual_loss = np.mean((prediction - target) ** 2)
    computed_loss = mse_loss(prediction, target)
    assert np.allclose(computed_loss, manual_loss)


def test_cross_entropy_invalid_logits_shape_raises() -> None:
    logits = np.array([1.0, 2.0, 3.0])  # 1D instead of 2D
    target = np.array([0])
    with pytest.raises(ValueError):
        cross_entropy_loss(logits, target)
    with pytest.raises(ValueError):
        cross_entropy_grad(logits, target)


def test_cross_entropy_mismatched_batch_raises() -> None:
    logits = np.zeros((2, 3))
    target = np.array([0, 1, 2])
    with pytest.raises(ValueError):
        cross_entropy_loss(logits, target)


def test_cross_entropy_non_integer_target_raises() -> None:
    logits = np.zeros((1, 2))
    target = np.array([0.5])
    with pytest.raises(TypeError):
        cross_entropy_loss(logits, target)


def test_cross_entropy_out_of_bounds_target_raises() -> None:
    logits = np.zeros((2, 2))
    target = np.array([0, 2])
    with pytest.raises(ValueError):
        cross_entropy_grad(logits, target)


def test_mse_shape_mismatch_raises() -> None:
    prediction = np.zeros((2, 2))
    target = np.zeros((3, 2))
    with pytest.raises(ValueError):
        mse_loss(prediction, target)
    with pytest.raises(ValueError):
        mse_grad(prediction, target)
