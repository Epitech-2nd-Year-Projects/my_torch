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
    return logits_array, target_array


def cross_entropy_loss(logits: ArrayFloat, target: ArrayInt) -> float:
    """
    Compute the mean cross-entropy loss from unnormalized logits and class indices.
    """
    logits_array, target_array = _validate_class_targets(logits, target)
    shifted = logits_array - np.max(logits_array, axis=1, keepdims=True)
    log_sum_exp = np.log(np.sum(np.exp(shifted), axis=1))
    correct_logit = shifted[np.arange(logits_array.shape[0]), target_array]
    losses = -correct_logit + log_sum_exp
    return float(np.mean(losses))


def cross_entropy_grad(logits: ArrayFloat, target: ArrayInt) -> ArrayFloat:
    """
    Compute the gradient of mean cross-entropy loss with respect to logits.
    """
    logits_array, target_array = _validate_class_targets(logits, target)
    shifted = logits_array - np.max(logits_array, axis=1, keepdims=True)
    exp_shifted = np.exp(shifted)
    probabilities = exp_shifted / np.sum(exp_shifted, axis=1, keepdims=True)
    grad = probabilities
    grad[np.arange(logits_array.shape[0]), target_array] -= 1
    grad /= logits_array.shape[0]
    return grad


def mse_loss(prediction: ArrayFloat, target: ArrayFloat) -> float:
    """
    Compute mean squared error between predictions and targets.
    """
    prediction_array = np.asarray(prediction, dtype=float)
    target_array = np.asarray(target, dtype=float)
    if prediction_array.shape != target_array.shape:
        raise ValueError("prediction and target must share the same shape")
    squared = np.square(prediction_array - target_array)
    return float(np.mean(squared))


def mse_grad(prediction: ArrayFloat, target: ArrayFloat) -> ArrayFloat:
    """
    Compute the gradient of mean squared error with respect to predictions.
    """
    prediction_array = np.asarray(prediction, dtype=float)
    target_array = np.asarray(target, dtype=float)
    if prediction_array.shape != target_array.shape:
        raise ValueError("prediction and target must share the same shape")
    return 2.0 * (prediction_array - target_array) / prediction_array.size
