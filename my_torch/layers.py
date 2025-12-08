from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal, Protocol

import numpy as np
from numpy.typing import NDArray

from .initializers import InitializerKey, initialize_bias, initialize_weights

ArrayFloat = NDArray[np.floating]
ActivationFn = Callable[[ArrayFloat], ArrayFloat]
ActivationDerivativeFn = Callable[..., ArrayFloat]

__all__ = ["DenseLayer"]


class Optimizer(Protocol):
    def update(self, param: ArrayFloat, grad: ArrayFloat) -> ArrayFloat:
        """Return the updated parameter given its gradient."""


def _identity(x: ArrayFloat) -> ArrayFloat:
    return x


@dataclass
class DenseLayer:
    """
    Fully connected layer with optional activation and gradient bookkeeping.

    Gradients are accumulated on backward passes and can be reset with `zero_grad`.
    Updates are delegated to any optimizer that implements an `update` method.

    Args:
        in_features: Number of input features (must be positive).
        out_features: Number of output features (must be positive).
        activation: Activation function to apply; defaults to identity.
        activation_derivative: Derivative of the activation function. If None, uses ones.
        weight_initializer: Key specifying the weight initialization strategy.
        bias_initializer: Policy for bias initialization; one of "zeros", "normal", or "uniform".
        rng: Optional random number generator for reproducibility.

    Raises:
        ValueError: If `in_features` or `out_features` are not positive.
    """

    in_features: int
    out_features: int
    activation: ActivationFn = _identity
    activation_derivative: ActivationDerivativeFn | None = None
    weight_initializer: InitializerKey = "xavier"
    bias_initializer: Literal["zeros", "normal", "uniform"] = "zeros"
    rng: np.random.Generator | None = None

    weights: ArrayFloat = field(init=False)
    bias: ArrayFloat = field(init=False)
    grad_weights: ArrayFloat = field(init=False)
    grad_bias: ArrayFloat = field(init=False)
    _last_input: ArrayFloat | None = field(default=None, init=False, repr=False)
    _pre_activation: ArrayFloat | None = field(default=None, init=False, repr=False)
    _output: ArrayFloat | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.in_features <= 0 or self.out_features <= 0:
            raise ValueError("in_features and out_features must be positive")
        self.weights = initialize_weights(
            (self.out_features, self.in_features), mode=self.weight_initializer, rng=self.rng
        )
        self.bias = initialize_bias((self.out_features,), mode=self.bias_initializer, rng=self.rng)
        self.grad_weights = np.zeros_like(self.weights)
        self.grad_bias = np.zeros_like(self.bias)

    def forward(self, inputs: ArrayFloat) -> ArrayFloat:
        x = np.asarray(inputs, dtype=float)
        if x.ndim != 2:
            raise ValueError("inputs must be a 2D array of shape (batch_size, in_features)")
        if x.shape[1] != self.in_features:
            raise ValueError(f"expected input feature dimension {self.in_features}, got {x.shape[1]}")

        self._last_input = x
        self._pre_activation = x @ self.weights.T + self.bias
        self._output = self.activation(self._pre_activation)
        return self._output

    def _activation_grad(self) -> ArrayFloat:
        if self._pre_activation is None:
            raise RuntimeError("forward must be called before backward")
        if self.activation_derivative is None:
            return np.ones_like(self._pre_activation)
        try:
            return self.activation_derivative(self._pre_activation, self._output)
        except TypeError:
            return self.activation_derivative(self._pre_activation)

    def backward(self, grad_output: ArrayFloat) -> ArrayFloat:
        if self._last_input is None or self._output is None or self._pre_activation is None:
            raise RuntimeError("forward must be called before backward")

        grad_out = np.asarray(grad_output, dtype=float)
        if grad_out.shape != self._output.shape:
            raise ValueError(f"grad_output shape {grad_out.shape} does not match output shape {self._output.shape}")

        local_grad = self._activation_grad()
        grad_z = grad_out * local_grad

        self.grad_weights = grad_z.T @ self._last_input
        self.grad_bias = np.sum(grad_z, axis=0)
        grad_input = grad_z @ self.weights
        return grad_input

    def parameters(self) -> tuple[ArrayFloat, ArrayFloat]:
        return self.weights, self.bias

    def gradients(self) -> tuple[ArrayFloat, ArrayFloat]:
        return self.grad_weights, self.grad_bias

    def zero_grad(self) -> None:
        self.grad_weights.fill(0.0)
        self.grad_bias.fill(0.0)

    def apply_updates(self, optimizer: Optimizer) -> None:
        self.weights = optimizer.update(self.weights, self.grad_weights)
        self.bias = optimizer.update(self.bias, self.grad_bias)
