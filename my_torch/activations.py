from __future__ import annotations

import numpy as np


def relu(x: np.ndarray) -> np.ndarray:
    """Rectified linear activation"""
    return np.maximum(0, x)


def relu_derivative(x: np.ndarray) -> np.ndarray:
    """Derivative of ReLU with respect to x"""
    return (x > 0).astype(x.dtype, copy=False)


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Sigmoid activation with stable computation"""
    return np.where(
        x >= 0,
        1 / (1 + np.exp(-x)),
        np.exp(x) / (1 + np.exp(x)),
    )


def sigmoid_derivative(
    x: np.ndarray, sigmoid_output: np.ndarray | None = None
) -> np.ndarray:
    """Derivative of sigmoid with optional precomputed output"""
    s = sigmoid_output if sigmoid_output is not None else sigmoid(x)
    return s * (1 - s)


def tanh(x: np.ndarray) -> np.ndarray:
    """Hyperbolic tangent activation"""
    return np.tanh(x)


def tanh_derivative(x: np.ndarray, tanh_output: np.ndarray | None = None) -> np.ndarray:
    """Derivative of tanh with optional precomputed output"""
    t = tanh_output if tanh_output is not None else np.tanh(x)
    return 1 - t**2


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Softmax over the given axis with overflow protection"""
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exps = np.exp(shifted)
    return exps / np.sum(exps, axis=axis, keepdims=True)


def softmax_derivative(
    x: np.ndarray, axis: int = -1, softmax_output: np.ndarray | None = None
) -> np.ndarray:
    """Jacobian of softmax with optional precomputed output"""
    s = softmax_output if softmax_output is not None else softmax(x, axis=axis)
    s_moved = np.moveaxis(s, axis, -1)
    outer = np.einsum("...i,...j->...ij", s_moved, s_moved)
    jacobian = -outer
    diag_indices = np.arange(s_moved.shape[-1])
    jacobian[..., diag_indices, diag_indices] += s_moved
    return jacobian
