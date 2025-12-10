from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, cast

import numpy as np
from numpy.typing import NDArray

from . import activations as activation_module
from .layers import DenseLayer
from .neural_network import NeuralNetwork, TrainableLayer

ArrayFloat = NDArray[np.floating]
ActivationRegistry = Mapping[str, Callable[..., ArrayFloat]]

FORMAT_VERSION = 1
_METADATA_KEY = "metadata_json"
_IDENTITY_NAME = "identity"

DEFAULT_ACTIVATIONS: dict[str, Callable[[ArrayFloat], ArrayFloat]] = {
    _IDENTITY_NAME: lambda x: x,
    "relu": activation_module.relu,
    "sigmoid": activation_module.sigmoid,
    "tanh": activation_module.tanh,
    "softmax": activation_module.softmax,
}

DEFAULT_ACTIVATION_DERIVATIVES: dict[str, Callable[..., ArrayFloat]] = {
    "relu_derivative": activation_module.relu_derivative,
    "sigmoid_derivative": activation_module.sigmoid_derivative,
    "tanh_derivative": activation_module.tanh_derivative,
    "softmax_derivative": activation_module.softmax_derivative,
    f"{_IDENTITY_NAME}_derivative": lambda x, *_args, **_kwargs: np.ones_like(x),
}


@dataclass(slots=True)
class SerializedModelMetadata:
    """Describes non-parameter data embedded alongside a saved network."""

    architecture: dict[str, Any]
    training: Mapping[str, Any] | None
    extras: Mapping[str, Any] | None
    format_version: int = FORMAT_VERSION


def save_nn(
    network: NeuralNetwork,
    file_path: str | Path,
    *,
    training_metadata: Mapping[str, Any] | None = None,
    extra_metadata: Mapping[str, Any] | None = None,
) -> None:
    """
    Persist a network into a `.nn` file.

    The file is a NumPy `.npz` archive containing:
        * metadata_json: UTF-8 JSON with format version, architecture, and metadata
        * layer_{i}_weights / layer_{i}_bias arrays for each dense layer

    Args:
        network: NeuralNetwork to serialize.
        file_path: Target path ending with `.nn`.
        training_metadata: Optional dictionary with information such as epochs or
            validation scores.
        extra_metadata: Optional free-form metadata copied into the file.
    """

    architecture = _serialize_architecture(network)
    payload: dict[str, Any] = {
        "format": "my_torch.nn",
        "format_version": FORMAT_VERSION,
        "architecture": architecture,
    }
    if training_metadata is not None:
        payload["training"] = dict(training_metadata)
    if extra_metadata is not None:
        payload["extras"] = dict(extra_metadata)

    arrays = _serialize_parameters(network)
    arrays[_METADATA_KEY] = np.array(
        json.dumps(payload, default=_json_default_serializer), dtype=np.dtype("U")
    )

    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file_obj:
        np.savez(file_obj, **cast(dict[str, Any], dict(arrays)))


def load_nn(
    file_path: str | Path,
    *,
    activation_registry: ActivationRegistry | None = None,
    activation_derivative_registry: ActivationRegistry | None = None,
) -> tuple[NeuralNetwork, SerializedModelMetadata]:
    """
    Load a network and metadata previously saved with `save_nn`.

    Args:
        file_path: Path to the `.nn` archive.
        activation_registry: Optional mapping from activation names to callables.
        activation_derivative_registry: Optional mapping for activation derivatives.
    Returns:
        Tuple of (`NeuralNetwork`, `SerializedModelMetadata`).
    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: When metadata is missing or inconsistent with stored tensors.
    """

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(path)

    activation_lookup: dict[str, Callable[..., ArrayFloat]] = dict(DEFAULT_ACTIVATIONS)
    if activation_registry is not None:
        activation_lookup.update(activation_registry)

    derivative_lookup: dict[str, Callable[..., ArrayFloat]] = dict(
        DEFAULT_ACTIVATION_DERIVATIVES
    )
    if activation_derivative_registry is not None:
        derivative_lookup.update(activation_derivative_registry)

    with path.open("rb") as file_obj, np.load(file_obj, allow_pickle=False) as archive:
        try:
            metadata_raw = archive[_METADATA_KEY]
        except KeyError as exc:
            raise ValueError("missing metadata_json entry in .nn archive") from exc
        metadata = json.loads(str(metadata_raw.item()))
        architecture = metadata.get("architecture")
        if not isinstance(architecture, Mapping):
            raise ValueError("architecture metadata is missing or invalid")

        layers = []
        layer_defs = architecture.get("layers", [])
        if not isinstance(layer_defs, list):
            raise ValueError("architecture.layers must be a list")
        for idx, layer_cfg in enumerate(layer_defs):
            if not isinstance(layer_cfg, Mapping):
                raise ValueError("layer configuration entries must be mappings")
            weights_key = f"layer_{idx}_weights"
            bias_key = f"layer_{idx}_bias"
            try:
                weights = archive[weights_key]
                bias = archive[bias_key]
            except KeyError as exc:
                raise ValueError(
                    f"missing parameter arrays for layer {idx}: {exc.args[0]}"
                ) from exc
            layers.append(
                _deserialize_layer(
                    layer_cfg,
                    weights,
                    bias,
                    activation_lookup,
                    derivative_lookup,
                )
            )

    network = NeuralNetwork(layers=layers)
    metadata_obj = SerializedModelMetadata(
        architecture=dict(architecture),
        training=metadata.get("training"),
        extras=metadata.get("extras"),
        format_version=int(metadata.get("format_version", FORMAT_VERSION)),
    )
    return network, metadata_obj


def save_network(
    path: str | Path,
    network: NeuralNetwork,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    """
    Save a neural network and metadata to a file.

    Wrapper around `save_nn` to match the project requirements.
    """
    save_nn(network, path, extra_metadata=metadata)


def load_network(path: str | Path) -> NeuralNetwork:
    """
    Load a neural network from a file.

    Wrapper around `load_nn` to match the project requirements.
    Returns only the reconstructed NeuralNetwork instance.
    """
    network, _ = load_nn(path)
    return network


def _serialize_parameters(network: NeuralNetwork) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for idx, layer in enumerate(network.layers):
        dense = _require_dense_layer(layer)
        arrays[f"layer_{idx}_weights"] = np.asarray(dense.weights, dtype=float)
        arrays[f"layer_{idx}_bias"] = np.asarray(dense.bias, dtype=float)
    return arrays


def _serialize_architecture(network: NeuralNetwork) -> dict[str, Any]:
    return {
        "layers": [
            _serialize_dense_layer(_require_dense_layer(layer), idx)
            for idx, layer in enumerate(network.layers)
        ]
    }


def _require_dense_layer(layer: TrainableLayer) -> DenseLayer:
    if isinstance(layer, DenseLayer):
        return layer
    raise TypeError("only DenseLayer instances can be serialized")


def _serialize_dense_layer(layer: DenseLayer, idx: int) -> dict[str, Any]:
    activation_name = _callable_to_name(layer.activation, allow_identity=True)
    derivative_name = _callable_to_name(layer.activation_derivative)
    if activation_name == "_identity":
        activation_name = _IDENTITY_NAME
    return {
        "type": "dense",
        "index": idx,
        "in_features": layer.in_features,
        "out_features": layer.out_features,
        "activation": activation_name,
        "activation_derivative": derivative_name,
        "weight_initializer": layer.weight_initializer,
        "bias_initializer": layer.bias_initializer,
    }


def _callable_to_name(
    func: Callable[..., Any] | None, *, allow_identity: bool = False
) -> str | None:
    if func is None:
        return None
    name = getattr(func, "__name__", None)
    if name:
        if allow_identity and name == "<lambda>":
            return _IDENTITY_NAME
        return name
    raise ValueError(
        "callables used in serializable layers must define a __name__ attribute"
    )


def _deserialize_layer(
    config: Mapping[str, Any],
    weights: ArrayFloat,
    bias: ArrayFloat,
    activation_registry: ActivationRegistry,
    derivative_registry: ActivationRegistry,
) -> DenseLayer:
    layer_type = config.get("type", "dense")
    if layer_type != "dense":
        raise ValueError(f"unsupported layer type '{layer_type}' in .nn file")
    in_features = int(config["in_features"])
    out_features = int(config["out_features"])

    activation_name = config.get("activation")
    derivative_name = config.get("activation_derivative")

    activation_fn = _resolve_activation(
        activation_registry, activation_name, required=False
    )
    derivative_fn = _resolve_activation(
        derivative_registry, derivative_name, required=False
    )

    identity_fn = activation_registry.get(_IDENTITY_NAME, lambda x: x)
    dense = DenseLayer(
        in_features=in_features,
        out_features=out_features,
        activation=activation_fn if activation_fn is not None else identity_fn,
        activation_derivative=derivative_fn,
        weight_initializer=config.get("weight_initializer", "xavier"),
        bias_initializer=config.get("bias_initializer", "zeros"),
    )
    dense.weights = np.asarray(weights, dtype=float)
    dense.bias = np.asarray(bias, dtype=float)
    dense.grad_weights = np.zeros_like(dense.weights)
    dense.grad_bias = np.zeros_like(dense.bias)
    return dense


def _resolve_activation(
    registry: ActivationRegistry, name: str | None, *, required: bool
) -> Callable[..., ArrayFloat] | None:
    if name is None:
        return None if not required else registry[_IDENTITY_NAME]
    if name == _IDENTITY_NAME:
        return registry[_IDENTITY_NAME]
    try:
        return registry[name]
    except KeyError as exc:
        raise ValueError(
            f"activation '{name}' is not registered for deserialization"
        ) from exc


def _json_default_serializer(obj: Any) -> Any:
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"object of type {type(obj).__name__} is not JSON serializable")
