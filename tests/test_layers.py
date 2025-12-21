from __future__ import annotations

import numpy as np
import pytest

from my_torch import SGD, DenseLayer, relu, relu_derivative


def test_dense_layer_preserves_shapes() -> None:
    rng = np.random.default_rng(123)
    batch_size, in_features, out_features = 5, 4, 3
    layer = DenseLayer(in_features=in_features, out_features=out_features, rng=rng)

    inputs = rng.standard_normal(size=(batch_size, in_features))
    outputs = layer.forward(inputs, training=False)
    assert outputs.shape == (batch_size, out_features)

    grad_output = rng.standard_normal(size=outputs.shape)
    grad_input = layer.backward(grad_output)
    assert grad_input.shape == (batch_size, in_features)
    assert layer.grad_weights.shape == (out_features, in_features)
    assert layer.grad_bias.shape == (out_features,)


def test_forward_computes_linear_activation() -> None:
    layer = DenseLayer(in_features=2, out_features=3)
    layer.weights = np.array([[1.0, 2.0], [3.0, -1.0], [0.5, 0.5]])
    layer.bias = np.array([0.5, -0.5, 1.0])

    inputs = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    output = layer.forward(inputs, training=False)

    expected = inputs @ layer.weights.T + layer.bias
    assert output.shape == expected.shape
    assert np.allclose(output, expected)


def test_backward_returns_gradients_matching_manual_derivation() -> None:
    layer = DenseLayer(
        in_features=2,
        out_features=2,
        activation=relu,
        activation_derivative=relu_derivative,
    )
    layer.weights = np.array([[1.0, -2.0], [0.5, 1.5]])
    layer.bias = np.array([0.0, 0.5])

    inputs = np.array([[2.0, -1.0], [0.0, 1.0]])
    grad_output = np.array([[1.0, 2.0], [-1.0, 0.5]])

    output = layer.forward(inputs, training=False)
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


def test_forward_invalid_dimensions_raise() -> None:
    layer = DenseLayer(in_features=2, out_features=2)
    with pytest.raises(ValueError, match="2D array"):
        layer.forward(np.array([1.0, 2.0, 3.0]), training=False)
    with pytest.raises(ValueError, match="expected input feature dimension"):
        layer.forward(np.ones((1, 3)), training=False)


def test_backward_mismatched_grad_output_raises() -> None:
    layer = DenseLayer(in_features=2, out_features=2)
    layer.forward(np.ones((1, 2)), training=False)
    with pytest.raises(ValueError, match="grad_output shape"):
        layer.backward(np.ones((2, 2)))


def test_invalid_feature_dimensions_raise_in_post_init() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        DenseLayer(in_features=0, out_features=1)
    with pytest.raises(ValueError, match="must be positive"):
        DenseLayer(in_features=1, out_features=-2)


def test_parameters_and_gradients_return_expected_references() -> None:
    layer = DenseLayer(in_features=2, out_features=2)
    params = layer.parameters()
    grads = layer.gradients()
    assert params[0] is layer.weights
    assert params[1] is layer.bias
    assert grads[0] is layer.grad_weights
    assert grads[1] is layer.grad_bias
