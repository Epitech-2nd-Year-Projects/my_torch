from __future__ import annotations

from typing import Any, Sequence, cast

import numpy as np
import pytest

from my_torch.layers import DenseLayer
from my_torch.losses import cross_entropy_grad, cross_entropy_loss
from my_torch.neural_network import NeuralNetwork
from my_torch.training import (
    TrainingHistory,
    _classification_accuracy,
    _validate_inputs,
    compute_class_weights,
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


def test_compute_class_weights_inverse_frequency() -> None:
    labels = np.array([0, 0, 1, 2, 2, 2], dtype=int)
    weights = compute_class_weights(labels, num_classes=3)

    assert weights.shape == (3,)
    assert weights.dtype == np.float32
    assert weights[1] > weights[0] > weights[2]
    assert np.isclose(weights.mean(), 1.0)


def test_train_returns_metrics_for_train_and_validation() -> None:
    inputs = np.array([[1.0, 0.0], [0.0, 1.0]])
    labels = np.array([0, 1], dtype=int)
    network = NeuralNetwork([DenseLayer(2, 2)])
    layer = cast(DenseLayer, network.layers[0])
    layer.weights[...] = np.array([[1.0, 0.0], [0.0, 1.0]])
    layer.bias.fill(0.0)

    class NoOpOptimizer:
        def step(self, parameters: object, gradients: object) -> None:
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


def test_train_stops_early_when_patience_is_reached() -> None:
    inputs = np.array([[1.0, 0.0], [0.0, 1.0]])
    labels = np.array([0, 1], dtype=int)
    network = NeuralNetwork([DenseLayer(2, 2)])

    class NoOpOptimizer:
        def step(self, parameters: object, gradients: object) -> None:
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
        epochs=5,
        batch_size=2,
        shuffle=False,
        early_stopping_patience=1,
    )

    assert len(history.train) == 2
    assert len(history.validation) == 2
    assert history.best_epoch == 0
    assert history.best_parameters is not None


def test_train_tracks_best_model_parameters() -> None:
    inputs = np.array([[1.0, 0.0], [0.0, 1.0]])
    labels = np.array([0, 1], dtype=int)
    network = NeuralNetwork([DenseLayer(2, 2)])
    layer = cast(DenseLayer, network.layers[0])
    layer.weights.fill(0.0)
    layer.bias.fill(0.0)

    desired_weights = np.array([[3.0, -1.0], [-1.0, 3.0]])
    desired_bias = np.array([0.0, 0.0])

    class ToggleOptimizer:
        def __init__(self) -> None:
            self.calls = 0

        def step(
            self,
            parameters: Sequence[np.ndarray[Any, Any]],
            gradients: Sequence[np.ndarray[Any, Any]],
        ) -> None:
            self.calls += 1
            if self.calls == 1:
                weights, bias = parameters
                weights[...] = desired_weights
                bias[...] = desired_bias
            else:
                for param in parameters:
                    param.fill(0.0)

    history = train(
        network,
        ToggleOptimizer(),
        inputs,
        labels,
        val_inputs=inputs,
        val_labels=labels,
        loss_fn=cross_entropy_loss,
        loss_grad_fn=cross_entropy_grad,
        epochs=3,
        batch_size=2,
        shuffle=False,
    )

    assert history.best_epoch == 0
    assert history.best_parameters is not None
    best_weights, best_bias = history.best_parameters
    assert np.allclose(best_weights, desired_weights)
    assert np.allclose(best_bias, desired_bias)
    assert not np.allclose(layer.weights, desired_weights)
