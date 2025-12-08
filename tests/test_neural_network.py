from __future__ import annotations

import numpy as np

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
