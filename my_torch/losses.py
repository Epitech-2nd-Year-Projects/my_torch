from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

ArrayFloat = NDArray[np.floating]
ArrayInt = NDArray[np.integer]

__all__ = [
    "cross_entropy_loss",
    "cross_entropy_grad",
    "mse_loss",
    "mse_grad",
]


def _validate_class_targets(logits: ArrayFloat, target: ArrayInt) -> tuple[ArrayFloat, ArrayInt]:
    logits_array = np.asarray(logits, dtype=float)
    target_array = np.asarray(target)
    if logits_array.ndim != 2:
        raise ValueError("logits must be 2D (batch_size, num_classes)")
    if target_array.shape != (logits_array.shape[0],):
        raise ValueError("target must match the batch dimension of logits")
    if not np.issubdtype(target_array.dtype, np.integer):
        raise TypeError("target must contain integer class indices")
    num_classes = logits_array.shape[1]
    if np.any((target_array < 0) | (target_array >= num_classes)):
        raise ValueError("target indices must be in [0, num_classes)")
    return logits_array, target_array


def cross_entropy_loss(logits: ArrayFloat, target: ArrayInt) -> float:
    """
    Mean cross-entropy from unnormalized logits and integer class targets

    Args:
        logits: 2D array shaped (batch_size, num_classes)
        target: 1D integer array of length batch_size
    Returns:
        Mean scalar loss
    Raises:
        ValueError: when shapes do not align
        TypeError: when targets are not integer indices
    """
    logits_array, target_array = _validate_class_targets(logits, target)
    shifted = logits_array - np.max(logits_array, axis=1, keepdims=True)
    log_sum_exp = np.log(np.sum(np.exp(shifted), axis=1))
    correct_logit = shifted[np.arange(logits_array.shape[0]), target_array]
    losses = -correct_logit + log_sum_exp
    return float(np.mean(losses))


def cross_entropy_grad(logits: ArrayFloat, target: ArrayInt) -> ArrayFloat:
    """
    Gradient of mean cross-entropy loss with respect to logits

    Args:
        logits: 2D array shaped (batch_size, num_classes)
        target: 1D integer array of length batch_size
    Returns:
        Array matching logits shape containing dL/dlogits, averaged over the batch
    Raises:
        ValueError: when shapes do not align
        TypeError: when targets are not integer indices
    """
    logits_array, target_array = _validate_class_targets(logits, target)
    shifted = logits_array - np.max(logits_array, axis=1, keepdims=True)
    exp_shifted = np.exp(shifted)
    probabilities = exp_shifted / np.sum(exp_shifted, axis=1, keepdims=True)
    grad = probabilities.copy()
    grad[np.arange(logits_array.shape[0]), target_array] -= 1
    grad /= logits_array.shape[0]
    return grad


def mse_loss(prediction: ArrayFloat, target: ArrayFloat) -> float:
    """
    Mean squared error between predictions and targets

    Args:
        prediction: array of predictions
        target: array with the same shape as prediction
    Returns:
        Mean scalar loss
    Raises:
        ValueError: when shapes differ
    """
    prediction_array = np.asarray(prediction, dtype=float)
    target_array = np.asarray(target, dtype=float)
    if prediction_array.shape != target_array.shape:
        raise ValueError("prediction and target must share the same shape")
    squared = np.square(prediction_array - target_array)
    return float(np.mean(squared))


def mse_grad(prediction: ArrayFloat, target: ArrayFloat) -> ArrayFloat:
    """
    Gradient of mean squared error with respect to predictions

    Args:
        prediction: array of predictions
        target: array with the same shape as prediction
    Returns:
        Array matching prediction shape containing dL/dprediction
    Raises:
        ValueError: when shapes differ
    """
    prediction_array = np.asarray(prediction, dtype=float)
    target_array = np.asarray(target, dtype=float)
    if prediction_array.shape != target_array.shape:
        raise ValueError("prediction and target must share the same shape")
    return 2.0 * (prediction_array - target_array) / prediction_array.size
