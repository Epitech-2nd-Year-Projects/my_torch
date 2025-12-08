from __future__ import annotations

from typing import Callable, Literal

import numpy as np
from numpy.typing import NDArray

ArrayFloat = NDArray[np.floating]
InitializerKey = Literal[
    "xavier",
    "xavier_uniform",
    "xavier_normal",
    "he",
    "he_uniform",
    "he_normal",
    "normal",
    "uniform",
]
InitializerFn = Callable[[tuple[int, ...], np.random.Generator | None], ArrayFloat]

_DEFAULT_STDDEV = 0.01
_DEFAULT_UNIFORM_LIMIT = 0.05

__all__ = [
    "InitializerKey",
    "get_initializer",
    "initialize_weights",
    "initialize_bias",
]


def _fan_in_out(shape: tuple[int, ...]) -> tuple[int, int]:
    if not shape:
        raise ValueError("shape must contain at least one dimension")
    if len(shape) == 1:
        fan_in = fan_out = shape[0]
    elif len(shape) == 2:
        fan_out, fan_in = shape
    else:
        receptive_field = int(np.prod(shape[2:]))
        fan_in = shape[1] * receptive_field
        fan_out = shape[0] * receptive_field
    return fan_in, fan_out


def _resolve_rng(rng: np.random.Generator | None) -> np.random.Generator:
    return rng if rng is not None else np.random.default_rng()


def _xavier_uniform(shape: tuple[int, ...], rng: np.random.Generator | None = None) -> ArrayFloat:
    fan_in, fan_out = _fan_in_out(shape)
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    return _resolve_rng(rng).uniform(-limit, limit, size=shape)


def _xavier_normal(shape: tuple[int, ...], rng: np.random.Generator | None = None) -> ArrayFloat:
    fan_in, fan_out = _fan_in_out(shape)
    std = np.sqrt(2.0 / (fan_in + fan_out))
    return _resolve_rng(rng).normal(0.0, std, size=shape)


def _he_uniform(shape: tuple[int, ...], rng: np.random.Generator | None = None) -> ArrayFloat:
    fan_in, _ = _fan_in_out(shape)
    limit = np.sqrt(6.0 / fan_in)
    return _resolve_rng(rng).uniform(-limit, limit, size=shape)


def _he_normal(shape: tuple[int, ...], rng: np.random.Generator | None = None) -> ArrayFloat:
    fan_in, _ = _fan_in_out(shape)
    std = np.sqrt(2.0 / fan_in)
    return _resolve_rng(rng).normal(0.0, std, size=shape)


def _simple_normal(shape: tuple[int, ...], rng: np.random.Generator | None = None) -> ArrayFloat:
    return _resolve_rng(rng).normal(0.0, _DEFAULT_STDDEV, size=shape)


def _simple_uniform(shape: tuple[int, ...], rng: np.random.Generator | None = None) -> ArrayFloat:
    return _resolve_rng(rng).uniform(-_DEFAULT_UNIFORM_LIMIT, _DEFAULT_UNIFORM_LIMIT, size=shape)


_INITIALIZER_REGISTRY: dict[str, InitializerFn] = {
    "xavier": _xavier_uniform,
    "xavier_uniform": _xavier_uniform,
    "xavier_normal": _xavier_normal,
    "he": _he_normal,
    "he_normal": _he_normal,
    "he_uniform": _he_uniform,
    "normal": _simple_normal,
    "uniform": _simple_uniform,
}


def get_initializer(mode: InitializerKey) -> InitializerFn:
    """
    Resolve a weight initializer by name

    Args:
        mode: Initialization strategy key such as "xavier", "he", or "uniform"
    Returns:
        Callable that generates an array for a given shape and optional RNG
    Raises:
        ValueError: when the mode is not registered
    """
    key = mode.lower()
    try:
        return _INITIALIZER_REGISTRY[key]
    except KeyError as exc:
        available = ", ".join(sorted(_INITIALIZER_REGISTRY))
        raise ValueError(f"unknown initialization strategy '{mode}', choose from: {available}") from exc


def initialize_weights(
    shape: tuple[int, ...], *, mode: InitializerKey = "xavier", rng: np.random.Generator | None = None
) -> ArrayFloat:
    """
    Initialize weights for the given shape using a selected strategy

    Args:
        shape: Target tensor shape, e.g. (out_features, in_features)
        mode: Initialization key, defaults to "xavier"
        rng: Optional NumPy random number generator
    Returns:
        Array shaped like `shape` populated according to the initializer
    Raises:
        ValueError: when the mode is not a supported initializer
    """
    initializer = get_initializer(mode)
    return initializer(shape, rng)


def initialize_bias(
    shape: tuple[int, ...],
    *,
    mode: Literal["zeros", "normal", "uniform"] = "zeros",
    rng: np.random.Generator | None = None,
    value: float = 0.0,
) -> ArrayFloat:
    """
    Initialize biases for the given shape using a specified policy

    Args:
        shape: Desired output shape for the bias array
        mode: Policy name; one of "zeros", "normal", or "uniform"
        rng: Optional NumPy random number generator for sampled modes
        value: Fill value used when mode is "zeros"
    Returns:
        Array shaped like `shape` initialized with the selected policy
    Raises:
        ValueError: when mode is not one of the supported options
    """
    if mode == "zeros":
        return np.full(shape, value, dtype=float)
    if mode == "normal":
        return _simple_normal(shape, rng)
    if mode == "uniform":
        return _simple_uniform(shape, rng)
    raise ValueError("mode must be one of: zeros, normal, uniform")
