from __future__ import annotations

from typing import cast

import numpy as np
from numpy.typing import NDArray

ArrayFloat = NDArray[np.floating]


def relu(x: ArrayFloat) -> ArrayFloat:
    """Rectified linear activation"""
    return np.maximum(0, x)


def relu_derivative(x: ArrayFloat) -> ArrayFloat:
    """Derivative of ReLU with respect to x"""
    return (x > 0).astype(x.dtype, copy=False)


def sigmoid(x: ArrayFloat) -> ArrayFloat:
    """Sigmoid activation with stable computation"""
    return np.where(
        x >= 0,
        1 / (1 + np.exp(-x)),
        np.exp(x) / (1 + np.exp(x)),
    )


def sigmoid_derivative(
    x: ArrayFloat, sigmoid_output: ArrayFloat | None = None
) -> ArrayFloat:
    """Derivative of sigmoid with optional precomputed output"""
    s = sigmoid_output if sigmoid_output is not None else sigmoid(x)
    return s * (1 - s)


def tanh(x: ArrayFloat) -> ArrayFloat:
    """Hyperbolic tangent activation"""
    return np.tanh(x)


def tanh_derivative(x: ArrayFloat, tanh_output: ArrayFloat | None = None) -> ArrayFloat:
    """Derivative of tanh with optional precomputed output"""
    t = tanh_output if tanh_output is not None else np.tanh(x)
    return 1 - t**2


def softmax(x: ArrayFloat, axis: int = -1) -> ArrayFloat:
    """Softmax over the given axis with overflow protection"""
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exps = np.exp(shifted)
    return cast(ArrayFloat, exps / np.sum(exps, axis=axis, keepdims=True))


def softmax_derivative(
    x: ArrayFloat, axis: int = -1, softmax_output: ArrayFloat | None = None
) -> ArrayFloat:
    """Jacobian of softmax with optional precomputed output"""
    s = softmax_output if softmax_output is not None else softmax(x, axis=axis)
    s_moved = np.moveaxis(s, axis, -1)
    outer = np.einsum("...i,...j->...ij", s_moved, s_moved)
    jacobian = -outer
    diag_indices = np.arange(s_moved.shape[-1])
    jacobian[..., diag_indices, diag_indices] += s_moved
    return cast(ArrayFloat, jacobian)
