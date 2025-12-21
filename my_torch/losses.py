from __future__ import annotations

from typing import cast

import numpy as np
from numpy.typing import NDArray

ArrayFloat = NDArray[np.floating]
ArrayInt = NDArray[np.integer]

__all__ = [
    "cross_entropy_loss",
    "cross_entropy_grad",
    "softmax_cross_entropy_with_logits",
    "mse_loss",
    "mse_grad",
]


def _validate_class_targets(
    logits: ArrayFloat, target: ArrayInt
) -> tuple[ArrayFloat, ArrayInt]:
    logits_array = np.asarray(logits)
    if not np.issubdtype(logits_array.dtype, np.floating):
        logits_array = logits_array.astype(np.float32)
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


def softmax_cross_entropy_with_logits(
    logits: ArrayFloat,
    target: ArrayInt,
    *,
    class_weights: np.ndarray | None = None,
    label_smoothing: float = 0.0,
) -> tuple[float, ArrayFloat]:
    """
    Mean cross-entropy and gradient from logits and integer class targets

    Args:
        logits: 2D array shaped batch_size by num_classes
        target: 1D integer array of length batch_size
        class_weights: optional 1D array shaped num_classes
        label_smoothing: smoothing factor in [0, 1]
    Returns:
        Tuple of mean loss and gradient w.r.t logits
        Weighted results are normalized by the sum of per-sample weights
    Raises:
        ValueError: when shapes do not align or parameters are invalid
        TypeError: when targets are not integer indices
    """
    logits_array, target_array = _validate_class_targets(logits, target)
    if label_smoothing < 0.0 or label_smoothing > 1.0:
        raise ValueError("label_smoothing must be in [0, 1]")
    batch_size, num_classes = logits_array.shape
    shifted = logits_array - np.max(logits_array, axis=1, keepdims=True)
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(
        exp_shifted, axis=1, keepdims=True, dtype=logits_array.dtype
    )
    probabilities = exp_shifted / sum_exp
    log_probs = shifted - np.log(sum_exp)

    if label_smoothing == 0.0:
        per_sample_loss = -log_probs[np.arange(batch_size), target_array]
        grad = probabilities
        grad[np.arange(batch_size), target_array] -= 1.0
    else:
        smoothing = float(label_smoothing)
        uniform = smoothing / num_classes
        per_sample_loss = -(1.0 - smoothing) * log_probs[
            np.arange(batch_size), target_array
        ]
        per_sample_loss -= uniform * np.sum(
            log_probs, axis=1, dtype=logits_array.dtype
        )
        grad = probabilities - uniform
        grad[np.arange(batch_size), target_array] -= 1.0 - smoothing

    if class_weights is not None:
        weights_array = np.asarray(class_weights, dtype=logits_array.dtype)
        if weights_array.ndim != 1 or weights_array.shape[0] != num_classes:
            raise ValueError("class_weights must have shape (num_classes,)")
        sample_weights = weights_array[target_array]
        weight_total = float(np.sum(sample_weights))
        if weight_total <= 0.0:
            raise ValueError(
                "class_weights must sum to a positive value over targets"
            )
        per_sample_loss = per_sample_loss * sample_weights
        loss = float(np.sum(per_sample_loss) / weight_total)
        grad = grad * (sample_weights / weight_total)[:, None]
        return loss, cast(ArrayFloat, grad)

    loss = float(np.mean(per_sample_loss))
    grad = grad / batch_size
    return loss, cast(ArrayFloat, grad)


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
    loss, _ = softmax_cross_entropy_with_logits(logits, target)
    return loss


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
    _, grad = softmax_cross_entropy_with_logits(logits, target)
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
    prediction_array = np.asarray(prediction)
    if not np.issubdtype(prediction_array.dtype, np.floating):
        prediction_array = prediction_array.astype(np.float32)
    target_array = np.asarray(target, dtype=prediction_array.dtype)
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
    prediction_array = np.asarray(prediction)
    if not np.issubdtype(prediction_array.dtype, np.floating):
        prediction_array = prediction_array.astype(np.float32)
    target_array = np.asarray(target, dtype=prediction_array.dtype)
    if prediction_array.shape != target_array.shape:
        raise ValueError("prediction and target must share the same shape")
    return cast(
        ArrayFloat, 2.0 * (prediction_array - target_array) / prediction_array.size
    )
