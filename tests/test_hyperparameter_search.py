from __future__ import annotations

import numpy as np
import pytest

from my_torch.hyperparameter_search import (
    HyperparameterSearchSummary,
    format_search_summary,
    search_hyperparameters,
)
from my_torch.losses import cross_entropy_grad, cross_entropy_loss


class DummyNetwork:
    """Minimal network whose accuracy is determined by a single parameter."""

    def __init__(self, num_features: int, num_classes: int) -> None:
        self.num_features = num_features
        self.num_classes = num_classes
        self.param = np.zeros(1, dtype=float)
        self.grad = np.zeros_like(self.param)
        self._last_input_shape: tuple[int, ...] | None = None

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        inputs_array = np.asarray(inputs, dtype=float)
        if inputs_array.ndim != 2 or inputs_array.shape[1] != self.num_features:
            raise ValueError("unexpected input shape")
        self._last_input_shape = inputs_array.shape
        logits = np.zeros((inputs_array.shape[0], self.num_classes))
        class_idx = int(np.clip(round(self.param[0]), 0, self.num_classes - 1))
        logits[:, class_idx] = 5.0
        return logits

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        if self._last_input_shape is None:
            raise RuntimeError("forward must be called before backward")
        batch_size = grad_output.shape[0]
        return np.zeros((batch_size, self.num_features))

    def parameters(self) -> tuple[np.ndarray, ...]:
        return (self.param,)

    def gradients(self) -> tuple[np.ndarray, ...]:
        return (self.grad,)

    def zero_grad(self) -> None:
        self.grad.fill(0.0)


class ScriptedOptimizer:
    """Optimizer that overwrites parameters with a predetermined value."""

    def __init__(self, target_value: float) -> None:
        self.target_value = target_value

    def step(self, parameters, gradients) -> None:  # pragma: no cover - exercised
        if len(parameters) != len(gradients):
            raise ValueError("parameters and gradients must have the same length")
        for param in parameters:
            np.asarray(param)[...] = self.target_value


def make_builder(num_features: int = 2, num_classes: int = 3):
    def builder() -> DummyNetwork:
        return DummyNetwork(num_features=num_features, num_classes=num_classes)

    return builder


def _common_data() -> tuple[np.ndarray, np.ndarray]:
    features = np.zeros((4, 2), dtype=float)
    labels = np.array([1, 1, 1, 0], dtype=int)
    return features, labels


def test_grid_search_selects_highest_accuracy_configuration() -> None:
    train_inputs, train_labels = _common_data()
    val_inputs, val_labels = train_inputs.copy(), train_labels.copy()

    def optimizer_factory(config):
        target_class = 1 if config.learning_rate >= 0.05 else 0
        return ScriptedOptimizer(target_value=float(target_class))

    summary = search_hyperparameters(
        network_builder=make_builder(),
        train_inputs=train_inputs,
        train_labels=train_labels,
        val_inputs=val_inputs,
        val_labels=val_labels,
        loss_fn=cross_entropy_loss,
        loss_grad_fn=cross_entropy_grad,
        learning_rates=[0.01, 0.1],
        batch_sizes=[2],
        weight_decays=[0.0],
        epochs=1,
        optimizer_factory=optimizer_factory,
    )

    assert len(summary.trials) == 2
    assert summary.best_trial.config.learning_rate == pytest.approx(0.1)
    assert summary.best_trial.validation_accuracy == pytest.approx(0.75)
    report = format_search_summary(summary)
    assert "learning_rate=0.1" in report
    assert "val_acc=0.7500" in report


def test_random_search_is_reproducible_with_seed() -> None:
    train_inputs, train_labels = _common_data()
    val_inputs, val_labels = train_inputs.copy(), train_labels.copy()

    def optimizer_factory(config):
        target_class = 1 if config.learning_rate >= 0.03 else 0
        return ScriptedOptimizer(target_value=float(target_class))

    summary_one = search_hyperparameters(
        network_builder=make_builder(),
        train_inputs=train_inputs,
        train_labels=train_labels,
        val_inputs=val_inputs,
        val_labels=val_labels,
        loss_fn=cross_entropy_loss,
        loss_grad_fn=cross_entropy_grad,
        learning_rates=[0.01, 0.02, 0.03, 0.04],
        batch_sizes=[2],
        weight_decays=[0.0],
        epochs=1,
        mode="random",
        num_samples=2,
        optimizer_factory=optimizer_factory,
        seed=42,
    )
    summary_two = search_hyperparameters(
        network_builder=make_builder(),
        train_inputs=train_inputs,
        train_labels=train_labels,
        val_inputs=val_inputs,
        val_labels=val_labels,
        loss_fn=cross_entropy_loss,
        loss_grad_fn=cross_entropy_grad,
        learning_rates=[0.01, 0.02, 0.03, 0.04],
        batch_sizes=[2],
        weight_decays=[0.0],
        epochs=1,
        mode="random",
        num_samples=2,
        optimizer_factory=optimizer_factory,
        seed=42,
    )

    def _config_signature(
        summary: HyperparameterSearchSummary,
    ) -> list[tuple[float, int]]:
        return [
            (trial.config.learning_rate, trial.config.batch_size)
            for trial in summary.trials
        ]

    assert len(summary_one.trials) == 2
    assert len(summary_two.trials) == 2
    assert _config_signature(summary_one) == _config_signature(summary_two)
    assert summary_one.best_trial.config.learning_rate == pytest.approx(
        summary_two.best_trial.config.learning_rate
    )
