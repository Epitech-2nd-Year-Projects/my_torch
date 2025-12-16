from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .activations import (
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
    "identity": lambda x: x,
}

_DERIVATIVES = {
    "relu": relu_derivative,
    "sigmoid": sigmoid_derivative,
    "tanh": tanh_derivative,
    "softmax": softmax_derivative,
    "identity": lambda x, *args: 1.0,
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

    if "type" not in layer:
        layer["type"] = "dense"

    if layer["type"] == "dense":
        return _process_dense_layer(layer, index)

    raise ValueError(f"Layer {index}: Unsupported layer type '{layer['type']}'")


def _process_dense_layer(layer: dict[str, Any], index: int) -> dict[str, Any]:
    if "in_features" not in layer or "out_features" not in layer:
        raise ValueError(
            f"Layer {index} (dense): Must specify 'in_features' and 'out_features'"
        )

    act_name = layer.get("activation")
    if act_name:
        if act_name not in _ACTIVATIONS:
            raise ValueError(
                f"Layer {index}: Unknown activation '{act_name}'. "
                f"Available: {', '.join(sorted(_ACTIVATIONS.keys()))}"
            )
        layer["activation"] = _ACTIVATIONS[act_name]

        if "activation_derivative" not in layer and act_name in _DERIVATIVES:
            layer["activation_derivative"] = _DERIVATIVES[act_name]

    der_name = layer.get("activation_derivative")
    if isinstance(der_name, str):
        if der_name not in _DERIVATIVES:
            raise ValueError(
                f"Layer {index}: Unknown activation derivative '{der_name}'"
            )
        layer["activation_derivative"] = _DERIVATIVES[der_name]

    return layer
