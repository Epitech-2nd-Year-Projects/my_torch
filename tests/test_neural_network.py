from __future__ import annotations

import numpy as np
import pytest

from my_torch import DenseLayer, NeuralNetwork, relu, relu_derivative


def test_forward_and_backward_chain() -> None:
    layer1 = DenseLayer(in_features=2, out_features=3)
    layer2 = DenseLayer(in_features=3, out_features=1)

    layer1.weights = np.array([[0.2, -0.4], [1.0, 0.5], [-0.3, 0.8]])
    layer1.bias = np.array([0.1, -0.2, 0.05])
    layer2.weights = np.array([[0.7, -1.2, 0.3]])
    layer2.bias = np.array([0.4])

    network = NeuralNetwork([layer1, layer2])

    inputs = np.array([[1.0, 2.0], [0.5, -1.0]])
    output = network.forward(inputs)

    expected_hidden = inputs @ layer1.weights.T + layer1.bias
    expected_output = expected_hidden @ layer2.weights.T + layer2.bias
    assert output.shape == expected_output.shape
    assert np.allclose(output, expected_output)

    grad_loss = np.array([[1.0], [-0.5]])
    grad_input = network.backward(grad_loss)

    expected_grad_layer2_weights = grad_loss.T @ expected_hidden
    expected_grad_layer2_bias = grad_loss.sum(axis=0)
    grad_to_hidden = grad_loss @ layer2.weights
    expected_grad_layer1_weights = grad_to_hidden.T @ inputs
    expected_grad_layer1_bias = grad_to_hidden.sum(axis=0)
    expected_grad_input = grad_to_hidden @ layer1.weights

    assert np.allclose(layer2.grad_weights, expected_grad_layer2_weights)
    assert np.allclose(layer2.grad_bias, expected_grad_layer2_bias)
    assert np.allclose(layer1.grad_weights, expected_grad_layer1_weights)
    assert np.allclose(layer1.grad_bias, expected_grad_layer1_bias)
    assert np.allclose(grad_input, expected_grad_input)


def test_parameters_and_gradients_iterate_in_order() -> None:
    first = DenseLayer(in_features=2, out_features=2)
    second = DenseLayer(in_features=2, out_features=1)
    network = NeuralNetwork([first, second])

    params = network.parameters()
    grads = network.gradients()

    assert params[0] is first.weights
    assert params[1] is first.bias
    assert params[2] is second.weights
    assert params[3] is second.bias

    assert grads[0] is first.grad_weights
    assert grads[1] is first.grad_bias
    assert grads[2] is second.grad_weights
    assert grads[3] is second.grad_bias


def test_configuration_constructor_builds_layers() -> None:
    configs = [
        {
            "type": "dense",
            "in_features": 4,
            "out_features": 2,
            "activation": relu,
            "activation_derivative": relu_derivative,
        },
        {"in_features": 2, "out_features": 1},
    ]
    network = NeuralNetwork(layer_configs=configs)

    assert len(network.layers) == 2
    assert isinstance(network.layers[0], DenseLayer)
    assert isinstance(network.layers[1], DenseLayer)

    sample = np.ones((1, 4))
    output = network.forward(sample)
    assert output.shape == (1, 1)


def test_init_with_both_layers_and_configs_raises() -> None:
    layer = DenseLayer(in_features=2, out_features=1)
    configs = [{"in_features": 2, "out_features": 1}]
    with pytest.raises(ValueError, match="either layers or layer_configs"):
        NeuralNetwork([layer], layer_configs=configs)


def test_config_with_unsupported_layer_type_raises() -> None:
    with pytest.raises(ValueError, match="unsupported layer type"):
        NeuralNetwork(layer_configs=[{"type": "conv", "in_features": 2, "out_features": 1}])


def test_config_missing_required_keys_raises() -> None:
    with pytest.raises(ValueError, match="requires 'in_features' and 'out_features' keys"):
        NeuralNetwork(layer_configs=[{"type": "dense"}])


def test_config_with_unexpected_keys_raises() -> None:
    with pytest.raises(ValueError, match="unexpected keys"):
        NeuralNetwork(layer_configs=[{"in_features": 2, "out_features": 1, "invalid_key": 123}])


def test_config_with_non_callable_activation_raises() -> None:
    with pytest.raises(TypeError, match="must be callable"):
        NeuralNetwork(layer_configs=[{"in_features": 2, "out_features": 1, "activation": "not_callable"}])


def test_zero_grad_clears_all_layer_gradients() -> None:
    first = DenseLayer(in_features=2, out_features=2)
    second = DenseLayer(in_features=2, out_features=1)
    network = NeuralNetwork([first, second])

    first.grad_weights = np.full_like(first.grad_weights, 0.5)
    first.grad_bias = np.full_like(first.grad_bias, 0.3)
    second.grad_weights = np.full_like(second.grad_weights, -0.2)
    second.grad_bias = np.full_like(second.grad_bias, -0.1)

    network.zero_grad()

    assert np.all(first.grad_weights == 0)
    assert np.all(first.grad_bias == 0)
    assert np.all(second.grad_weights == 0)
    assert np.all(second.grad_bias == 0)
