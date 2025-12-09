import numpy as np
import pytest

from my_torch.layers import DenseLayer
from my_torch.losses import cross_entropy_grad, cross_entropy_loss
from my_torch.neural_network import NeuralNetwork
from my_torch.optimizers import SGD
from my_torch.training import train


def test_l2_loss_reporting() -> None:
    network = NeuralNetwork([DenseLayer(2, 2, bias_initializer="zeros")])
    assert isinstance(network.layers[0], DenseLayer)
    network.layers[0].weights.fill(1.0)
    network.layers[0].bias.fill(0.0)

    inputs = np.array([[1.0, 1.0]])
    labels = np.array([0], dtype=int)

    logits = network.forward(inputs)
    data_loss = cross_entropy_loss(logits, labels)

    weight_decay = 0.1
    expected_reg_loss = 0.5 * weight_decay * 4.0

    expected_total_loss = data_loss + expected_reg_loss

    optimizer = SGD(lr=0.01, weight_decay=weight_decay)
    history = train(
        network,
        optimizer,
        inputs,
        labels,
        val_inputs=inputs,
        val_labels=labels,
        loss_fn=cross_entropy_loss,
        loss_grad_fn=cross_entropy_grad,
        epochs=1,
        batch_size=1,
        weight_decay=weight_decay,
        shuffle=False,
    )

    reported_loss = history.train[0].loss
    assert reported_loss == pytest.approx(expected_total_loss)

    val_loss = history.validation[0].loss
    assert val_loss < 0.7
    assert val_loss < 0.95


def test_l2_weight_decay_update() -> None:
    network = NeuralNetwork([DenseLayer(2, 2, bias_initializer="zeros")])
    assert isinstance(network.layers[0], DenseLayer)
    network.layers[0].weights.fill(1.0)
    network.layers[0].bias.fill(0.0)

    inputs = np.array([[0.0, 0.0]])
    labels = np.array([0], dtype=int)

    weight_decay = 0.1
    lr = 0.1
    optimizer = SGD(lr=lr, weight_decay=weight_decay)
    train(
        network,
        optimizer,
        inputs,
        labels,
        val_inputs=inputs,
        val_labels=labels,
        loss_fn=cross_entropy_loss,
        loss_grad_fn=cross_entropy_grad,
        epochs=1,
        batch_size=1,
        weight_decay=weight_decay,
        shuffle=False,
    )

    expected_weight = 1.0 - lr * weight_decay * 1.0
    assert np.allclose(network.layers[0].weights, expected_weight)
