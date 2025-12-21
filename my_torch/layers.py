from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Integral
from typing import Callable, Literal, Protocol

import numpy as np
from numpy.typing import NDArray

from . import activations
from .initializers import InitializerKey, initialize_bias, initialize_weights

ArrayFloat = NDArray[np.floating]
ActivationFn = Callable[[ArrayFloat], ArrayFloat]
ActivationDerivativeFn = Callable[..., ArrayFloat]

__all__ = [
    "Conv2DLayer",
    "DenseLayer",
    "DropoutLayer",
    "FlattenLayer",
    "GlobalAvgPool2D",
]


class Optimizer(Protocol):
    def update(self, param: ArrayFloat, grad: ArrayFloat) -> ArrayFloat:
        """Return the updated parameter given its gradient."""


def _identity(x: ArrayFloat) -> ArrayFloat:
    return x


def _pair(value: int | tuple[int, int], *, name: str) -> tuple[int, int]:
    if isinstance(value, tuple):
        if len(value) != 2:
            raise ValueError(f"{name} must be an int or a tuple of two ints")
        if not all(isinstance(entry, Integral) for entry in value):
            raise ValueError(f"{name} must contain integers")
        return int(value[0]), int(value[1])
    if not isinstance(value, Integral):
        raise ValueError(f"{name} must be an int or a tuple of two ints")
    return int(value), int(value)


def _compute_output_dim(
    input_size: int, kernel_size: int, padding: int, stride: int
) -> int:
    numerator = input_size + 2 * padding - kernel_size
    if numerator < 0:
        raise ValueError("kernel size exceeds padded input size")
    if numerator % stride != 0:
        raise ValueError("stride does not evenly divide the output size")
    return numerator // stride + 1


def _im2col(
    inputs: ArrayFloat,
    kernel_size: tuple[int, int],
    stride: tuple[int, int],
    padding: tuple[int, int],
) -> ArrayFloat:
    batch_size, channels, height, width = inputs.shape
    kernel_h, kernel_w = kernel_size
    stride_h, stride_w = stride
    pad_h, pad_w = padding
    out_h = _compute_output_dim(height, kernel_h, pad_h, stride_h)
    out_w = _compute_output_dim(width, kernel_w, pad_w, stride_w)

    padded = np.pad(
        inputs,
        ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)),
        mode="constant",
    )
    cols = np.empty(
        (batch_size, channels, kernel_h, kernel_w, out_h, out_w),
        dtype=inputs.dtype,
    )
    for y in range(kernel_h):
        y_max = y + stride_h * out_h
        for x in range(kernel_w):
            x_max = x + stride_w * out_w
            cols[:, :, y, x, :, :] = padded[:, :, y:y_max:stride_h, x:x_max:stride_w]
    return cols.reshape(batch_size, channels * kernel_h * kernel_w, out_h * out_w)


def _col2im(
    cols: ArrayFloat,
    input_shape: tuple[int, int, int, int],
    kernel_size: tuple[int, int],
    stride: tuple[int, int],
    padding: tuple[int, int],
) -> ArrayFloat:
    batch_size, channels, height, width = input_shape
    kernel_h, kernel_w = kernel_size
    stride_h, stride_w = stride
    pad_h, pad_w = padding
    out_h = _compute_output_dim(height, kernel_h, pad_h, stride_h)
    out_w = _compute_output_dim(width, kernel_w, pad_w, stride_w)

    cols_reshaped = cols.reshape(
        batch_size, channels, kernel_h, kernel_w, out_h, out_w
    )
    padded = np.zeros(
        (batch_size, channels, height + 2 * pad_h, width + 2 * pad_w),
        dtype=cols.dtype,
    )
    for y in range(kernel_h):
        y_max = y + stride_h * out_h
        for x in range(kernel_w):
            x_max = x + stride_w * out_w
            padded[:, :, y:y_max:stride_h, x:x_max:stride_w] += cols_reshaped[
                :, :, y, x, :, :
            ]
    if pad_h == 0 and pad_w == 0:
        return padded
    return padded[:, :, pad_h : pad_h + height, pad_w : pad_w + width]


@dataclass
class DenseLayer:
    """
    Fully connected layer with optional activation and gradient bookkeeping.

    Gradients are accumulated on backward passes and can be reset with `zero_grad`
    Updates are delegated to any optimizer that implements an `update` method.

    Args:
        in_features: Number of input features (must be positive).
        out_features: Number of output features (must be positive).
        activation: Activation function to apply; defaults to identity.
        activation_derivative: Derivative of the activation function. If None, uses
            ones.
        weight_initializer: Key specifying the weight initialization strategy.
        bias_initializer: Policy for bias initialization; one of "zeros", "normal", or
            "uniform".
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
            (self.out_features, self.in_features),
            mode=self.weight_initializer,
            rng=self.rng,
        )
        self.bias = initialize_bias(
            (self.out_features,), mode=self.bias_initializer, rng=self.rng
        )
        self.grad_weights = np.zeros_like(self.weights)
        self.grad_bias = np.zeros_like(self.bias)

    def forward(self, inputs: ArrayFloat, *, training: bool) -> ArrayFloat:
        x = np.asarray(inputs, dtype=self.weights.dtype)
        if x.ndim != 2:
            raise ValueError(
                "inputs must be a 2D array of shape (batch_size, in_features)"
            )
        if x.shape[1] != self.in_features:
            raise ValueError(
                f"expected input feature dimension {self.in_features}, got {x.shape[1]}"
            )

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
        if (
            self._last_input is None
            or self._output is None
            or self._pre_activation is None
        ):
            raise RuntimeError("forward must be called before backward")

        grad_out = np.asarray(grad_output, dtype=self.weights.dtype)
        if grad_out.shape != self._output.shape:
            raise ValueError(
                f"grad_output shape {grad_out.shape} does not match output shape "
                f"{self._output.shape}"
            )

        if self.activation is activations.softmax:
            s = self._output
            dot = np.sum(grad_out * s, axis=1, keepdims=True)
            grad_z = s * (grad_out - dot)
        else:
            local_grad = self._activation_grad()
            grad_z = grad_out * local_grad

        self.grad_weights[...] = grad_z.T @ self._last_input
        self.grad_bias[...] = np.sum(grad_z, axis=0, dtype=self.grad_bias.dtype)
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


@dataclass
class Conv2DLayer:
    """
    Apply 2D convolution over NCHW inputs with optional activation

    Gradients are accumulated on backward passes and can be reset with `zero_grad`

    Args:
        in_channels: Number of input channels (must be positive)
        out_channels: Number of output channels (must be positive)
        kernel_size: Kernel height and width
        stride: Stride for height and width
        padding: Zero padding for height and width
        activation: Activation function to apply; defaults to identity
        activation_derivative: Derivative of activation function; if None, uses ones
        weight_initializer: Key specifying the weight initialization strategy
        bias_initializer: Policy for bias initialization; one of "zeros", "normal", or
            "uniform"
        rng: Optional random number generator for reproducibility
    """

    in_channels: int
    out_channels: int
    kernel_size: int | tuple[int, int]
    stride: int | tuple[int, int] = 1
    padding: int | tuple[int, int] = 0
    activation: ActivationFn = _identity
    activation_derivative: ActivationDerivativeFn | None = None
    weight_initializer: InitializerKey = "he_normal"
    bias_initializer: Literal["zeros", "normal", "uniform"] = "zeros"
    rng: np.random.Generator | None = None

    weights: ArrayFloat = field(init=False)
    bias: ArrayFloat = field(init=False)
    grad_weights: ArrayFloat = field(init=False)
    grad_bias: ArrayFloat = field(init=False)
    _input_shape: tuple[int, int, int, int] | None = field(
        default=None, init=False, repr=False
    )
    _im2col_cache: ArrayFloat | None = field(default=None, init=False, repr=False)
    _pre_activation: ArrayFloat | None = field(default=None, init=False, repr=False)
    _output: ArrayFloat | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.in_channels <= 0 or self.out_channels <= 0:
            raise ValueError("in_channels and out_channels must be positive")
        kernel_h, kernel_w = _pair(self.kernel_size, name="kernel_size")
        stride_h, stride_w = _pair(self.stride, name="stride")
        pad_h, pad_w = _pair(self.padding, name="padding")
        if kernel_h <= 0 or kernel_w <= 0:
            raise ValueError("kernel_size values must be positive")
        if stride_h <= 0 or stride_w <= 0:
            raise ValueError("stride values must be positive")
        if pad_h < 0 or pad_w < 0:
            raise ValueError("padding values must be non-negative")

        self.kernel_size = (kernel_h, kernel_w)
        self.stride = (stride_h, stride_w)
        self.padding = (pad_h, pad_w)

        self.weights = initialize_weights(
            (self.out_channels, self.in_channels, kernel_h, kernel_w),
            mode=self.weight_initializer,
            rng=self.rng,
            dtype=np.float32,
        )
        self.bias = initialize_bias(
            (self.out_channels,),
            mode=self.bias_initializer,
            rng=self.rng,
            dtype=np.float32,
        )
        self.grad_weights = np.zeros_like(self.weights)
        self.grad_bias = np.zeros_like(self.bias)

    def forward(self, inputs: ArrayFloat, *, training: bool) -> ArrayFloat:
        x = np.asarray(inputs, dtype=self.weights.dtype)
        if x.ndim != 4:
            raise ValueError(
                "inputs must be a 4D array of shape "
                "(batch_size, channels, height, width)"
            )
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"expected input channels {self.in_channels}, got {x.shape[1]}"
            )

        kernel_size = self.kernel_size
        stride = self.stride
        padding = self.padding

        cols = _im2col(x, kernel_size, stride, padding)
        weights_matrix = self.weights.reshape(self.out_channels, -1)
        cols_matrix = cols.transpose(0, 2, 1)
        output = cols_matrix @ weights_matrix.T
        out_h = _compute_output_dim(
            x.shape[2], kernel_size[0], padding[0], stride[0]
        )
        out_w = _compute_output_dim(
            x.shape[3], kernel_size[1], padding[1], stride[1]
        )
        output = output.transpose(0, 2, 1).reshape(
            x.shape[0], self.out_channels, out_h, out_w
        )
        output = output + self.bias[None, :, None, None]

        self._input_shape = x.shape
        self._im2col_cache = cols
        self._pre_activation = output
        self._output = self.activation(output)
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
        if (
            self._input_shape is None
            or self._im2col_cache is None
            or self._output is None
            or self._pre_activation is None
        ):
            raise RuntimeError("forward must be called before backward")

        grad_out = np.asarray(grad_output, dtype=self.weights.dtype)
        if grad_out.shape != self._output.shape:
            raise ValueError(
                f"grad_output shape {grad_out.shape} does not match output shape "
                f"{self._output.shape}"
            )

        if self.activation is activations.softmax:
            s = self._output
            dot = np.sum(grad_out * s, axis=1, keepdims=True)
            grad_z = s * (grad_out - dot)
        else:
            local_grad = self._activation_grad()
            grad_z = grad_out * local_grad

        batch_size, _, out_h, out_w = grad_z.shape
        grad_z_flat = grad_z.reshape(batch_size, self.out_channels, out_h * out_w)

        self.grad_weights[...] = np.tensordot(
            grad_z_flat, self._im2col_cache, axes=([0, 2], [0, 2])
        ).reshape(self.weights.shape)
        self.grad_bias[...] = np.sum(
            grad_z, axis=(0, 2, 3), dtype=self.grad_bias.dtype
        )

        weights_matrix = self.weights.reshape(self.out_channels, -1)
        grad_cols = np.tensordot(grad_z_flat, weights_matrix, axes=([1], [0]))
        grad_cols = grad_cols.transpose(0, 2, 1)
        grad_input = _col2im(
            grad_cols,
            self._input_shape,
            self.kernel_size,
            self.stride,
            self.padding,
        )
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


@dataclass
class FlattenLayer:
    """
    Flatten 4D inputs into 2D batches

    Returns:
        Output shaped as batch size by flattened feature dimension
    """

    _input_shape: tuple[int, int, int, int] | None = field(
        default=None, init=False, repr=False
    )

    def forward(self, inputs: ArrayFloat, *, training: bool) -> ArrayFloat:
        x = np.asarray(inputs)
        if x.ndim != 4:
            raise ValueError(
                "inputs must be a 4D array of shape "
                "(batch_size, channels, height, width)"
            )
        self._input_shape = x.shape
        batch_size, channels, height, width = x.shape
        return x.reshape(batch_size, channels * height * width)

    def backward(self, grad_output: ArrayFloat) -> ArrayFloat:
        if self._input_shape is None:
            raise RuntimeError("forward must be called before backward")
        grad_out = np.asarray(grad_output)
        expected = (self._input_shape[0], int(np.prod(self._input_shape[1:])))
        if grad_out.shape != expected:
            raise ValueError(
                f"grad_output shape {grad_out.shape} does not match expected shape "
                f"{expected}"
            )
        return grad_out.reshape(self._input_shape)

    def parameters(self) -> tuple[ArrayFloat, ...]:
        return ()

    def gradients(self) -> tuple[ArrayFloat, ...]:
        return ()

    def zero_grad(self) -> None:
        return None


@dataclass
class GlobalAvgPool2D:
    """
    Average spatial dimensions for each channel

    Returns:
        Output shaped as batch size by channels
    """

    _input_shape: tuple[int, int, int, int] | None = field(
        default=None, init=False, repr=False
    )

    def forward(self, inputs: ArrayFloat, *, training: bool) -> ArrayFloat:
        x = np.asarray(inputs)
        if x.ndim != 4:
            raise ValueError(
                "inputs must be a 4D array of shape "
                "(batch_size, channels, height, width)"
            )
        self._input_shape = x.shape
        return np.mean(x, axis=(2, 3), dtype=x.dtype)

    def backward(self, grad_output: ArrayFloat) -> ArrayFloat:
        if self._input_shape is None:
            raise RuntimeError("forward must be called before backward")
        grad_out = np.asarray(grad_output)
        expected = (self._input_shape[0], self._input_shape[1])
        if grad_out.shape != expected:
            raise ValueError(
                f"grad_output shape {grad_out.shape} does not match expected shape "
                f"{expected}"
            )
        batch_size, channels, height, width = self._input_shape
        scale = np.asarray(1.0 / float(height * width), dtype=grad_out.dtype)
        return grad_out[:, :, None, None] * scale * np.ones(
            (batch_size, channels, height, width), dtype=grad_out.dtype
        )

    def parameters(self) -> tuple[ArrayFloat, ...]:
        return ()

    def gradients(self) -> tuple[ArrayFloat, ...]:
        return ()

    def zero_grad(self) -> None:
        return None


@dataclass
class DropoutLayer:
    """
    Randomly zero inputs during training using inverted dropout

    Args:
        p: Drop probability for each element
        rng: Optional random generator or seed for reproducibility
    """

    p: float
    rng: np.random.Generator | int | None = None

    _mask: ArrayFloat | None = field(default=None, init=False, repr=False)
    _input_shape: tuple[int, ...] | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if not 0.0 <= self.p < 1.0:
            raise ValueError("p must be in the range [0.0, 1.0)")

        if isinstance(self.rng, np.random.Generator):
            self.rng = self.rng
        elif self.rng is None:
            self.rng = np.random.default_rng()
        else:
            self.rng = np.random.default_rng(self.rng)

    def forward(self, inputs: ArrayFloat, *, training: bool) -> ArrayFloat:
        x = np.asarray(inputs)
        self._input_shape = x.shape
        mask_dtype = (
            x.dtype if np.issubdtype(x.dtype, np.floating) else np.dtype(np.float32)
        )
        if not training or self.p == 0.0:
            self._mask = np.ones(x.shape, dtype=mask_dtype)
            return x

        if self.rng is None:
            raise RuntimeError("rng not initialized")

        keep_prob = np.asarray(1.0 - self.p, dtype=mask_dtype)
        mask = (self.rng.random(x.shape) >= self.p).astype(mask_dtype, copy=False)
        mask = mask / keep_prob
        self._mask = mask
        return x * mask

    def backward(self, grad_output: ArrayFloat) -> ArrayFloat:
        if self._input_shape is None:
            raise RuntimeError("forward must be called before backward")
        grad_out = np.asarray(grad_output)
        if grad_out.shape != self._input_shape:
            raise ValueError(
                f"grad_output shape {grad_out.shape} does not match expected shape "
                f"{self._input_shape}"
            )
        if self._mask is None:
            raise RuntimeError("dropout mask missing; forward must be called first")
        return grad_out * self._mask

    def parameters(self) -> tuple[ArrayFloat, ...]:
        return ()

    def gradients(self) -> tuple[ArrayFloat, ...]:
        return ()

    def zero_grad(self) -> None:
        return None
