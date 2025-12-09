from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterator, Protocol, Sequence

import numpy as np
from numpy.typing import NDArray

from .neural_network import NeuralNetwork

ArrayFloat = NDArray[np.floating]
ArrayInt = NDArray[np.integer]
LossFn = Callable[[ArrayFloat, ArrayInt], float]
LossGradFn = Callable[[ArrayFloat, ArrayInt], ArrayFloat]
AccuracyFn = Callable[[ArrayFloat, ArrayInt], float]

__all__ = ["EpochMetrics", "TrainingHistory", "train_validation_split", "train"]


class Optimizer(Protocol):
    def step(
        self, parameters: Sequence[ArrayFloat], gradients: Sequence[ArrayFloat]
    ) -> None: ...


@dataclass(slots=True)
class EpochMetrics:
    """Loss and accuracy metrics for a single epoch."""

    loss: float
    accuracy: float


@dataclass(slots=True)
class TrainingHistory:
    """Per-epoch metrics for a full training run."""

    train: list[EpochMetrics]
    validation: list[EpochMetrics]


def _resolve_rng(rng: np.random.Generator | None) -> np.random.Generator:
    return rng if rng is not None else np.random.default_rng()


def _validate_inputs(
    inputs: ArrayFloat, labels: ArrayInt
) -> tuple[ArrayFloat, ArrayInt]:
    inputs_array = np.asarray(inputs, dtype=float)
    labels_array = np.asarray(labels)
    if inputs_array.shape[0] != labels_array.shape[0]:
        raise ValueError("inputs and labels must share the first dimension")
    if labels_array.ndim != 1:
        raise ValueError("labels must be a 1D array of class indices")
    if not np.issubdtype(labels_array.dtype, np.integer):
        raise TypeError("labels must contain integer class indices")
    return inputs_array, labels_array


def train_validation_split(
    inputs: ArrayFloat,
    labels: ArrayInt,
    *,
    val_ratio: float = 0.2,
    shuffle: bool = True,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[ArrayFloat, ArrayFloat, ArrayInt, ArrayInt]:
    """
    Split inputs and labels into training and validation subsets

    Args:
        inputs: Feature array shaped (num_samples, num_features)
        labels: Integer class labels shaped (num_samples,)
        val_ratio: Fraction of data to place in the validation set, must be between 0
            and 1 (exclusive)
        shuffle: Whether to shuffle before splitting
        seed: Optional seed used when constructing the RNG
        rng: Optional NumPy Generator; cannot be combined with seed
    Returns:
        Tuple of (train_inputs, val_inputs, train_labels, val_labels)
    Raises:
        ValueError: when ratios are invalid or splits would be empty
        TypeError: when labels are not integer class indices
    """
    if rng is not None and seed is not None:
        raise ValueError("provide either rng or seed, not both")
    inputs_array, labels_array = _validate_inputs(inputs, labels)
    num_samples = inputs_array.shape[0]
    if num_samples == 0:
        raise ValueError("cannot split an empty dataset")
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be between 0 and 1 (exclusive)")

    val_size = int(np.floor(num_samples * val_ratio))
    if val_size == 0 or val_size == num_samples:
        raise ValueError("val_ratio produces an empty train or validation split")

    rng_instance = rng if rng is not None else np.random.default_rng(seed)
    indices = (
        rng_instance.permutation(num_samples) if shuffle else np.arange(num_samples)
    )
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]

    return (
        inputs_array[train_indices],
        inputs_array[val_indices],
        labels_array[train_indices],
        labels_array[val_indices],
    )


def _iter_batches(
    inputs: ArrayFloat, labels: ArrayInt, batch_size: int
) -> Iterator[tuple[ArrayFloat, ArrayInt]]:
    for start in range(0, inputs.shape[0], batch_size):
        end = start + batch_size
        yield inputs[start:end], labels[start:end]


def _classification_accuracy(logits: ArrayFloat, labels: ArrayInt) -> float:
    logits_array = np.asarray(logits, dtype=float)
    if logits_array.ndim != 2:
        raise ValueError(
            "logits must be 2D (batch_size, num_classes) to compute accuracy"
        )
    predictions = np.argmax(logits_array, axis=1)
    return float(np.mean(predictions == labels))


def _compute_l2_loss(network: NeuralNetwork, weight_decay: float) -> float:
    if weight_decay <= 0.0:
        return 0.0
    l2_sum = sum(float(np.sum(np.square(p))) for p in network.parameters())
    return 0.5 * weight_decay * l2_sum


def _evaluate(
    network: NeuralNetwork,
    inputs: ArrayFloat,
    labels: ArrayInt,
    batch_size: int,
    loss_fn: LossFn,
    accuracy_fn: AccuracyFn,
    weight_decay: float = 0.0,
) -> EpochMetrics:
    total_loss = 0.0
    weighted_accuracy = 0.0
    total_samples = inputs.shape[0]

    reg_loss = _compute_l2_loss(network, weight_decay)

    for batch_inputs, batch_labels in _iter_batches(inputs, labels, batch_size):
        logits = network.forward(batch_inputs)
        batch_size_actual = batch_labels.shape[0]
        total_loss += (loss_fn(logits, batch_labels) + reg_loss) * batch_size_actual
        weighted_accuracy += accuracy_fn(logits, batch_labels) * batch_size_actual

    if total_samples == 0:
        return EpochMetrics(loss=0.0, accuracy=0.0)
    return EpochMetrics(
        loss=total_loss / total_samples, accuracy=weighted_accuracy / total_samples
    )


def train(
    network: NeuralNetwork,
    optimizer: Optimizer,
    train_inputs: ArrayFloat,
    train_labels: ArrayInt,
    *,
    val_inputs: ArrayFloat,
    val_labels: ArrayInt,
    loss_fn: LossFn,
    loss_grad_fn: LossGradFn,
    epochs: int,
    batch_size: int,
    shuffle: bool = True,
    rng: np.random.Generator | None = None,
    accuracy_fn: AccuracyFn | None = None,
    weight_decay: float = 0.0,
) -> TrainingHistory:
    """
    Train a network with mini-batch updates and return per-epoch metrics

    Args:
        network: Model to optimize
        optimizer: Optimizer called after each backward pass
        train_inputs: Training features shaped (num_samples, num_features)
        train_labels: Integer class labels for the training set
        val_inputs: Validation features used for evaluation
        val_labels: Integer class labels for validation
        loss_fn: Callable returning a scalar loss from logits and labels
        loss_grad_fn: Callable returning dLoss/dLogits for a batch
        epochs: Number of passes over the training data
        batch_size: Number of samples per batch
        shuffle: Whether to shuffle training data each epoch
        rng: Optional random generator for deterministic shuffling
        accuracy_fn: Optional accuracy metric; defaults to argmax-based accuracy
        weight_decay: L2 regularization strength. If > 0,
                      adds L2 loss to reported metrics.
                      Optimizers supporting weight decay should be configured with it
                      directly during their instantiation.
    Returns:
        TrainingHistory containing train and validation metrics per epoch
    Raises:
        ValueError: when shapes are inconsistent or hyperparameters are invalid
        TypeError: when labels are not integer class indices
    """
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if weight_decay < 0:
        raise ValueError("weight_decay must be non-negative")

    train_inputs_array, train_labels_array = _validate_inputs(
        train_inputs, train_labels
    )
    val_inputs_array, val_labels_array = _validate_inputs(val_inputs, val_labels)
    metric = accuracy_fn or _classification_accuracy
    rng_instance = _resolve_rng(rng)
    num_train = train_inputs_array.shape[0]

    train_history: list[EpochMetrics] = []
    val_history: list[EpochMetrics] = []

    for _ in range(epochs):
        if shuffle:
            indices = rng_instance.permutation(num_train)
            epoch_inputs = train_inputs_array[indices]
            epoch_labels = train_labels_array[indices]
        else:
            epoch_inputs = train_inputs_array
            epoch_labels = train_labels_array

        epoch_loss = 0.0
        epoch_accuracy = 0.0
        seen = 0

        for batch_inputs, batch_labels in _iter_batches(
            epoch_inputs, epoch_labels, batch_size
        ):
            logits = network.forward(batch_inputs)
            batch_loss = loss_fn(logits, batch_labels)

            reg_loss = _compute_l2_loss(network, weight_decay)

            grad_logits = loss_grad_fn(logits, batch_labels)

            network.zero_grad()
            network.backward(grad_logits)
            optimizer.step(network.parameters(), network.gradients())

            batch_size_actual = batch_labels.shape[0]

            epoch_loss += (batch_loss + reg_loss) * batch_size_actual
            epoch_accuracy += metric(logits, batch_labels) * batch_size_actual
            seen += batch_size_actual

        if seen == 0:
            train_history.append(EpochMetrics(loss=0.0, accuracy=0.0))
        else:
            train_history.append(
                EpochMetrics(loss=epoch_loss / seen, accuracy=epoch_accuracy / seen)
            )
        val_history.append(
            _evaluate(
                network, val_inputs_array, val_labels_array, batch_size, loss_fn, metric
            )
        )

    return TrainingHistory(train=train_history, validation=val_history)
