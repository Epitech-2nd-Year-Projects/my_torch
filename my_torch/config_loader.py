from __future__ import annotations

import json
from numbers import Integral
from pathlib import Path
from typing import Any

from .activations import (
    identity,
    identity_derivative,
    relu,
    relu_derivative,
    sigmoid,
    sigmoid_derivative,
    softmax,
    softmax_derivative,
    tanh,
    tanh_derivative,
)

_ACTIVATIONS = {
    "relu": relu,
    "sigmoid": sigmoid,
    "tanh": tanh,
    "softmax": softmax,
    "identity": identity,
}

_DERIVATIVES = {
    "relu": relu_derivative,
    "sigmoid": sigmoid_derivative,
    "tanh": tanh_derivative,
    "softmax": softmax_derivative,
    "identity": identity_derivative,
}


def load_config(path: str | Path) -> list[dict[str, Any]]:
    """
    Load a neural network configuration from a JSON file.

    Args:
        path: Path to the JSON configuration file.

    Returns:
        A list of layer configuration dictionaries ready for NeuralNetwork.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        try:
            config = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in configuration file: {e}") from e

    if not isinstance(config, dict):
        raise ValueError("Configuration root must be a dictionary")

    if "layers" not in config:
        raise ValueError("Configuration must contain a 'layers' list")

    layers_config = config["layers"]
    if not isinstance(layers_config, list):
        raise ValueError("'layers' must be a list")

    processed_layers = []
    for i, layer_conf in enumerate(layers_config):
        if not isinstance(layer_conf, dict):
            raise ValueError(f"Layer {i} configuration must be a dictionary")
        processed_layers.append(_process_layer_config(layer_conf, i))

    return processed_layers


def _process_layer_config(conf: dict[str, Any], index: int) -> dict[str, Any]:
    """Valdiate and resolve callables for a single layer config."""
    layer = conf.copy()

    layer_type = layer.get("type", "dense")
    if not isinstance(layer_type, str):
        raise ValueError(f"Layer {index}: Layer type must be a string")
    layer_type = layer_type.lower()
    layer["type"] = layer_type

    if layer_type == "dense":
        return _process_dense_layer(layer, index)
    if layer_type == "conv2d":
        return _process_conv2d_layer(layer, index)
    if layer_type == "flatten":
        return _process_flatten_layer(layer, index)
    if layer_type == "dropout":
        return _process_dropout_layer(layer, index)
    if layer_type == "global_avg_pool2d":
        return _process_global_avg_pool2d_layer(layer, index)

    raise ValueError(f"Layer {index}: Unsupported layer type '{layer_type}'")


def _process_dense_layer(layer: dict[str, Any], index: int) -> dict[str, Any]:
    _validate_layer_keys(
        layer,
        index,
        "dense",
        required={"in_features", "out_features"},
        optional={
            "activation",
            "activation_derivative",
            "weight_initializer",
            "bias_initializer",
            "rng",
        },
    )

    _resolve_activation(layer, index)
    return layer


def _process_conv2d_layer(layer: dict[str, Any], index: int) -> dict[str, Any]:
    _validate_layer_keys(
        layer,
        index,
        "conv2d",
        required={"in_channels", "out_channels", "kernel_size"},
        optional={
            "stride",
            "padding",
            "activation",
            "activation_derivative",
            "weight_initializer",
            "bias_initializer",
            "rng",
        },
    )
    layer["kernel_size"] = _normalize_pair_value(
        layer["kernel_size"], index, "conv2d", "kernel_size"
    )
    if "stride" in layer:
        layer["stride"] = _normalize_pair_value(
            layer["stride"], index, "conv2d", "stride"
        )
    if "padding" in layer:
        layer["padding"] = _normalize_pair_value(
            layer["padding"], index, "conv2d", "padding"
        )
    _resolve_activation(layer, index)
    return layer


def _process_flatten_layer(layer: dict[str, Any], index: int) -> dict[str, Any]:
    _validate_layer_keys(layer, index, "flatten", required=set(), optional=set())
    return layer


def _process_dropout_layer(layer: dict[str, Any], index: int) -> dict[str, Any]:
    _validate_layer_keys(
        layer, index, "dropout", required={"p"}, optional={"rng"}
    )
    return layer


def _process_global_avg_pool2d_layer(
    layer: dict[str, Any], index: int
) -> dict[str, Any]:
    _validate_layer_keys(
        layer, index, "global_avg_pool2d", required=set(), optional=set()
    )
    return layer


def _validate_layer_keys(
    layer: dict[str, Any],
    index: int,
    layer_type: str,
    *,
    required: set[str],
    optional: set[str],
) -> None:
    allowed = required | optional | {"type"}
    unexpected = set(layer) - allowed
    if unexpected:
        unknown = ", ".join(sorted(unexpected))
        raise ValueError(
            f"Layer {index} ({layer_type}): Unexpected keys: {unknown}"
        )
    missing = sorted(required - set(layer))
    if missing:
        missing_list = ", ".join(missing)
        raise ValueError(
            f"Layer {index} ({layer_type}): Missing required keys: {missing_list}"
        )


def _normalize_pair_value(
    value: Any, index: int, layer_type: str, key: str
) -> int | tuple[int, int]:
    if isinstance(value, tuple):
        entries = value
    elif isinstance(value, list):
        entries = value
    else:
        return value

    if len(entries) != 2 or not all(isinstance(entry, Integral) for entry in entries):
        raise ValueError(
            f"Layer {index} ({layer_type}): '{key}' must be an int or "
            "a list of two ints"
        )
    return int(entries[0]), int(entries[1])


def _resolve_activation(layer: dict[str, Any], index: int) -> None:
    act_name = layer.get("activation")
    resolved_name = None
    if act_name is not None:
        if isinstance(act_name, str):
            key = act_name.lower()
            if key not in _ACTIVATIONS:
                raise ValueError(
                    f"Layer {index}: Unknown activation '{act_name}'. "
                    f"Available: {', '.join(sorted(_ACTIVATIONS.keys()))}"
                )
            layer["activation"] = _ACTIVATIONS[key]
            resolved_name = key
        elif callable(act_name):
            resolved_name = None
        else:
            raise TypeError(
                f"Layer {index}: activation must be a string or callable"
            )

    if "activation_derivative" not in layer and resolved_name in _DERIVATIVES:
        layer["activation_derivative"] = _DERIVATIVES[resolved_name]

    der_name = layer.get("activation_derivative")
    if isinstance(der_name, str):
        key = der_name.lower()
        if key not in _DERIVATIVES:
            raise ValueError(
                f"Layer {index}: Unknown activation derivative '{der_name}'"
            )
        layer["activation_derivative"] = _DERIVATIVES[key]
    elif der_name is not None and not callable(der_name):
        raise TypeError(
            f"Layer {index}: activation_derivative must be a string or callable"
        )
