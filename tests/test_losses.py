from __future__ import annotations

import numpy as np

from my_torch import cross_entropy_grad, cross_entropy_loss, mse_grad, mse_loss


def numerical_gradient(func, values: np.ndarray, epsilon: float = 1e-5) -> np.ndarray:
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
