from __future__ import annotations

import inspect
import math
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Protocol, Sequence

import numpy as np
from numpy.typing import NDArray

from .neural_network import NeuralNetwork

ArrayFloat = NDArray[np.floating]
ArrayInt = NDArray[np.integer]
IndexArray = NDArray[np.signedinteger[Any]]
LossFn = Callable[[ArrayFloat, ArrayInt], float]
LossGradFn = Callable[[ArrayFloat, ArrayInt], ArrayFloat]
AccuracyFn = Callable[[ArrayFloat, ArrayInt], float]

__all__ = [
    "EpochMetrics",
    "TrainingHistory",
    "compute_class_weights",
    "train_validation_split",
    "train",
    "Optimizer",
]


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
    """Per-epoch metrics plus best model tracking for a training run."""

    train: list[EpochMetrics]
    validation: list[EpochMetrics]
    best_epoch: int | None = None
    best_parameters: tuple[ArrayFloat, ...] | None = None


def _resolve_rng(rng: np.random.Generator | None) -> np.random.Generator:
    return rng if rng is not None else np.random.default_rng()


def _forward(
    network: NeuralNetwork, inputs: ArrayFloat, *, training: bool
) -> ArrayFloat:
    forward = network.forward
    try:
        signature = inspect.signature(forward)
    except (TypeError, ValueError):
        try:
            return forward(inputs, training=training)
        except TypeError:
            return forward(inputs)

    params = signature.parameters
    if "training" in params:
        return forward(inputs, training=training)
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values()):
        return forward(inputs, training=training)
    return forward(inputs)


def _validate_inputs(
    inputs: ArrayFloat, labels: ArrayInt
) -> tuple[ArrayFloat, ArrayInt]:
    inputs_array = np.asarray(inputs)
    labels_array = np.asarray(labels)
    if inputs_array.shape[0] != labels_array.shape[0]:
        raise ValueError("inputs and labels must share the first dimension")
    if labels_array.ndim != 1:
        raise ValueError("labels must be a 1D array of class indices")
    if not np.issubdtype(labels_array.dtype, np.integer):
        raise TypeError("labels must contain integer class indices")
    return inputs_array, labels_array


def _stratified_split_indices(
    labels: ArrayInt,
    *,
    val_ratio: float,
    shuffle: bool,
    rng: np.random.Generator,
    num_classes: int | None,
) -> tuple[IndexArray, IndexArray]:
    labels_array = np.asarray(labels)
    num_samples = labels_array.shape[0]
    if num_classes is not None:
        if num_classes <= 0:
            raise ValueError("num_classes must be positive")
        if np.any((labels_array < 0) | (labels_array >= num_classes)):
            raise ValueError("labels must be in [0, num_classes)")
        class_values = np.arange(num_classes)
    else:
        class_values = np.unique(labels_array)

    class_groups: list[IndexArray] = []
    class_counts: list[int] = []
    for class_value in class_values:
        indices: IndexArray = np.flatnonzero(labels_array == class_value).astype(
            np.int64, copy=False
        )
        if indices.size == 0:
            continue
        if shuffle:
            rng.shuffle(indices)
        class_groups.append(indices)
        class_counts.append(int(indices.size))

    val_counts = []
    for count in class_counts:
        val_count = int(np.floor(count * val_ratio))
        if count > 1:
            val_count = min(max(val_count, 1), count - 1)
        val_counts.append(val_count)

    total_val = sum(val_counts)
    if total_val == 0:
        if num_samples < 2:
            raise ValueError("val_ratio produces an empty train or validation split")
        fallback_index = int(np.argmax(class_counts))
        val_counts[fallback_index] = 1
        total_val = 1
    if total_val >= num_samples:
        if num_samples < 2:
            raise ValueError("val_ratio produces an empty train or validation split")
        fallback_index = int(np.argmax(val_counts))
        if val_counts[fallback_index] > 0:
            val_counts[fallback_index] -= 1
            total_val -= 1
    if total_val == 0 or total_val == num_samples:
        raise ValueError("val_ratio produces an empty train or validation split")

    val_indices: list[IndexArray] = []
    train_indices: list[IndexArray] = []
    for indices, val_count in zip(class_groups, val_counts):
        if val_count > 0:
            val_indices.append(indices[:val_count])
        if val_count < indices.size:
            train_indices.append(indices[val_count:])

    val_indices_array: IndexArray = (
        np.concatenate(val_indices)
        if val_indices
        else np.array([], dtype=np.int64)
    )
    train_indices_array: IndexArray = (
        np.concatenate(train_indices)
        if train_indices
        else np.array([], dtype=np.int64)
    )
    return train_indices_array, val_indices_array


def train_validation_split(
    inputs: ArrayFloat,
    labels: ArrayInt,
    *,
    val_ratio: float = 0.2,
    shuffle: bool = True,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
    stratify: bool = False,
    num_classes: int | None = None,
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
        stratify: Whether to split data while preserving class proportions
        num_classes: Optional number of classes used when stratifying
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

    rng_instance = rng if rng is not None else np.random.default_rng(seed)
    if stratify:
        train_indices, val_indices = _stratified_split_indices(
            labels_array,
            val_ratio=val_ratio,
            shuffle=shuffle,
            rng=rng_instance,
            num_classes=num_classes,
        )
    else:
        val_size = int(np.floor(num_samples * val_ratio))
        if val_size == 0 or val_size == num_samples:
            raise ValueError("val_ratio produces an empty train or validation split")
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


def compute_class_weights(
    labels: ArrayInt,
    num_classes: int,
    *,
    strategy: str = "inverse_freq",
) -> NDArray[np.floating]:
    """
    Compute per-class weights for imbalanced classification

    Args:
        labels: Integer class labels shaped (num_samples,)
        num_classes: Total number of classes
        strategy: Weighting strategy identifier
    Returns:
        Array of class weights shaped (num_classes,) with mean weight 1.0
    Raises:
        ValueError: when labels are empty or arguments are invalid
        TypeError: when labels are not integer class indices
    """
    labels_array = np.asarray(labels)
    if labels_array.ndim != 1:
        raise ValueError("labels must be a 1D array of class indices")
    if not np.issubdtype(labels_array.dtype, np.integer):
        raise TypeError("labels must contain integer class indices")
    if num_classes <= 0:
        raise ValueError("num_classes must be positive")
    if labels_array.size == 0:
        raise ValueError("labels must be non-empty")
    if np.any((labels_array < 0) | (labels_array >= num_classes)):
        raise ValueError("labels must be in [0, num_classes)")
    if strategy != "inverse_freq":
        raise ValueError("strategy must be 'inverse_freq'")

    counts = np.bincount(
        labels_array.astype(np.int64), minlength=num_classes
    ).astype(np.float32)
    weights = np.zeros(num_classes, dtype=np.float32)
    nonzero = counts > 0
    if not np.any(nonzero):
        raise ValueError("labels must contain at least one class")
    weights[nonzero] = 1.0 / counts[nonzero]
    mean_weight = float(np.mean(weights))
    if mean_weight <= 0.0:
        raise ValueError("class weights cannot be normalized")
    return weights / mean_weight


def _iter_batches(
    inputs: ArrayFloat, labels: ArrayInt, batch_size: int
) -> Iterator[tuple[ArrayFloat, ArrayInt]]:
    for start in range(0, inputs.shape[0], batch_size):
        end = start + batch_size
        yield inputs[start:end], labels[start:end]


def _classification_accuracy(logits: ArrayFloat, labels: ArrayInt) -> float:
    logits_array = np.asarray(logits)
    if logits_array.ndim != 2:
        raise ValueError(
            "logits must be 2D (batch_size, num_classes) to compute accuracy"
        )
    predictions = np.argmax(logits_array, axis=1)
    return float(np.mean(predictions == labels))


def _compute_l2_loss(network: NeuralNetwork, weight_decay: float) -> float:
    """
    Compute L2 regularization loss for weight tensors.

    Args:
        network: NeuralNetwork whose parameters will be regularized.
        weight_decay: Regularization strength (lambda).
    Returns:
        Total L2 loss (scalar). Returns 0.0 if weight_decay <= 0.
    """
    if weight_decay <= 0.0:
        return 0.0
    l2_sum = sum(
        float(np.sum(np.square(p)))
        for p in network.parameters()
        if p.ndim >= 2
    )
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
        logits = _forward(network, batch_inputs, training=False)
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
    early_stopping_patience: int | None = None,
    max_grad_norm: float | None = None,
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
        early_stopping_patience: Optional number of consecutive epochs without
            validation-loss improvement to tolerate before stopping early.
        max_grad_norm: Optional global norm threshold for gradient clipping.
    Returns:
        TrainingHistory containing train/validation metrics and best model data
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
    if early_stopping_patience is not None and early_stopping_patience <= 0:
        raise ValueError("early_stopping_patience must be positive when provided")
    if max_grad_norm is not None and max_grad_norm <= 0:
        raise ValueError("max_grad_norm must be positive when provided")

    train_inputs_array, train_labels_array = _validate_inputs(
        train_inputs, train_labels
    )
    val_inputs_array, val_labels_array = _validate_inputs(val_inputs, val_labels)
    metric = accuracy_fn or _classification_accuracy
    rng_instance = _resolve_rng(rng)
    num_train = train_inputs_array.shape[0]

    train_history: list[EpochMetrics] = []
    val_history: list[EpochMetrics] = []

    best_val_loss = float("inf")
    best_epoch: int | None = None
    best_parameters: tuple[ArrayFloat, ...] | None = None
    epochs_since_improvement = 0

    for epoch_index in range(epochs):
        indices = (
            rng_instance.permutation(num_train)
            if shuffle
            else np.arange(num_train, dtype=int)
        )

        epoch_loss = 0.0
        epoch_accuracy = 0.0
        seen = 0

        reg_loss = _compute_l2_loss(network, weight_decay)

        for start in range(0, num_train, batch_size):
            batch_indices = indices[start : start + batch_size]
            batch_inputs = train_inputs_array[batch_indices]
            batch_labels = train_labels_array[batch_indices]
            logits = _forward(network, batch_inputs, training=True)
            batch_loss = loss_fn(logits, batch_labels)

            grad_logits = loss_grad_fn(logits, batch_labels)

            network.zero_grad()
            network.backward(grad_logits)
            gradients = network.gradients()
            if max_grad_norm is not None:
                total_norm_sq = sum(
                    float(np.sum(np.square(grad))) for grad in gradients
                )
                if total_norm_sq > 0.0:
                    total_norm = math.sqrt(total_norm_sq)
                    if total_norm > max_grad_norm:
                        scale = max_grad_norm / total_norm
                        for grad in gradients:
                            grad *= scale
            optimizer.step(network.parameters(), gradients)

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
        current_val_loss = val_history[-1].loss
        improved = current_val_loss < best_val_loss
        if improved or best_parameters is None:
            best_val_loss = current_val_loss
            best_epoch = epoch_index
            best_parameters = tuple(
                np.array(param, copy=True) for param in network.parameters()
            )
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1
        if (
            early_stopping_patience is not None
            and epochs_since_improvement >= early_stopping_patience
        ):
            break

    return TrainingHistory(
        train=train_history,
        validation=val_history,
        best_epoch=best_epoch,
        best_parameters=best_parameters,
    )
