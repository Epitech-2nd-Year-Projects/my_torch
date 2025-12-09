from __future__ import annotations

from typing import cast

import numpy as np
import pytest

from my_torch.layers import DenseLayer
from my_torch.losses import cross_entropy_grad, cross_entropy_loss
from my_torch.neural_network import NeuralNetwork
from my_torch.training import (
    TrainingHistory,
    _classification_accuracy,
    _validate_inputs,
    train,
)


def test_validate_inputs_accepts_matching_shapes() -> None:
    inputs = np.array([[1.0, 2.0], [3.0, 4.0]])
    labels = np.array([0, 1], dtype=int)
    validated_inputs, validated_labels = _validate_inputs(inputs, labels)
    assert validated_inputs.shape == inputs.shape
    assert validated_labels.shape == labels.shape
    assert validated_labels.dtype == labels.dtype


def test_validate_inputs_rejects_mismatched_batch_dimension() -> None:
    inputs = np.array([[1.0, 2.0], [3.0, 4.0]])
    labels = np.array([0], dtype=int)
    with pytest.raises(ValueError):
        _validate_inputs(inputs, labels)


def test_validate_inputs_rejects_non_1d_labels() -> None:
    inputs = np.array([[1.0, 2.0], [3.0, 4.0]])
    labels = np.array([[0, 1]], dtype=int)
    with pytest.raises(ValueError):
        _validate_inputs(inputs, labels)


def test_validate_inputs_rejects_non_integer_labels() -> None:
    inputs = np.array([[1.0, 2.0], [3.0, 4.0]])
    labels = np.array([0.1, 0.2])
    with pytest.raises(TypeError):
        _validate_inputs(inputs, labels)


def test_classification_accuracy_basic() -> None:
    logits = np.array([[2.0, 0.5], [0.1, 1.5]])
    labels = np.array([0, 1], dtype=int)
    assert _classification_accuracy(logits, labels) == pytest.approx(1.0)


def test_classification_accuracy_handles_errors() -> None:
    logits = np.array([1.0, 2.0, 3.0])
    labels = np.array([0, 1, 2], dtype=int)
    with pytest.raises(ValueError):
        _classification_accuracy(logits, labels)


def test_train_returns_metrics_for_train_and_validation() -> None:
    inputs = np.array([[1.0, 0.0], [0.0, 1.0]])
    labels = np.array([0, 1], dtype=int)
    network = NeuralNetwork([DenseLayer(2, 2)])
    layer = cast(DenseLayer, network.layers[0])
    layer.weights[...] = np.array([[1.0, 0.0], [0.0, 1.0]])
    layer.bias.fill(0.0)

    class NoOpOptimizer:
        def step(self, parameters, gradients) -> None:
            return None

    history = train(
        network,
        NoOpOptimizer(),
        inputs,
        labels,
        val_inputs=inputs,
        val_labels=labels,
        loss_fn=cross_entropy_loss,
        loss_grad_fn=cross_entropy_grad,
        epochs=1,
        batch_size=1,
        shuffle=False,
    )

    assert isinstance(history, TrainingHistory)
    assert len(history.train) == 1
    assert len(history.validation) == 1

    expected_loss = cross_entropy_loss(inputs, labels)
    assert history.train[0].loss == pytest.approx(expected_loss)
    assert history.validation[0].loss == pytest.approx(expected_loss)
    assert history.train[0].accuracy == pytest.approx(1.0)
    assert history.validation[0].accuracy == pytest.approx(1.0)
