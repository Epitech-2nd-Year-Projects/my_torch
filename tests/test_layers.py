from __future__ import annotations

import numpy as np
import pytest

from my_torch import DenseLayer, relu, relu_derivative


def test_forward_computes_linear_activation() -> None:
    layer = DenseLayer(in_features=2, out_features=3)
    layer.weights = np.array([[1.0, 2.0], [3.0, -1.0], [0.5, 0.5]])
    layer.bias = np.array([0.5, -0.5, 1.0])

    inputs = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    output = layer.forward(inputs)

    expected = inputs @ layer.weights.T + layer.bias
    assert output.shape == expected.shape
    assert np.allclose(output, expected)


def test_backward_returns_gradients_matching_manual_derivation() -> None:
    layer = DenseLayer(in_features=2, out_features=2, activation=relu, activation_derivative=relu_derivative)
    layer.weights = np.array([[1.0, -2.0], [0.5, 1.5]])
    layer.bias = np.array([0.0, 0.5])

    inputs = np.array([[2.0, -1.0], [0.0, 1.0]])
    grad_output = np.array([[1.0, 2.0], [-1.0, 0.5]])

    output = layer.forward(inputs)
    grad_input = layer.backward(grad_output)

    pre_activation = inputs @ layer.weights.T + layer.bias
    local_grad = relu_derivative(pre_activation)
    grad_z = grad_output * local_grad

    expected_grad_weights = grad_z.T @ inputs
    expected_grad_bias = grad_z.sum(axis=0)
    expected_grad_input = grad_z @ layer.weights

    assert output.shape == grad_output.shape
    assert np.allclose(layer.grad_weights, expected_grad_weights)
    assert np.allclose(layer.grad_bias, expected_grad_bias)
    assert np.allclose(grad_input, expected_grad_input)


def test_apply_updates_and_zero_grad() -> None:
    layer = DenseLayer(in_features=3, out_features=1)
    layer.grad_weights = np.full_like(layer.grad_weights, 0.1)
    layer.grad_bias = np.array([0.2])

    class SGD:
        def __init__(self, lr: float) -> None:
            self.lr = lr

        def update(self, param: np.ndarray, grad: np.ndarray) -> np.ndarray:
            return param - self.lr * grad

    optimizer = SGD(lr=0.5)
    original_weights = layer.weights.copy()
    original_bias = layer.bias.copy()

    layer.apply_updates(optimizer)
    assert np.allclose(layer.weights, original_weights - 0.5 * layer.grad_weights)
    assert np.allclose(layer.bias, original_bias - 0.5 * layer.grad_bias)

    layer.zero_grad()
    assert np.all(layer.grad_weights == 0)
    assert np.all(layer.grad_bias == 0)


def test_backward_without_forward_raises() -> None:
    layer = DenseLayer(in_features=2, out_features=2)
    with pytest.raises(RuntimeError):
        layer.backward(np.ones((1, 2)))
