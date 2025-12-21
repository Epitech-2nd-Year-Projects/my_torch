from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, cast

import numpy as np
from numpy.typing import NDArray

from . import activations as activation_module
from .layers import Conv2DLayer, DenseLayer, DropoutLayer, FlattenLayer, GlobalAvgPool2D
from .neural_network import NeuralNetwork, TrainableLayer

ArrayFloat = NDArray[np.floating]
ActivationRegistry = Mapping[str, Callable[..., ArrayFloat]]

FORMAT_VERSION = 2
_LEGACY_FORMAT_VERSION = 1
_METADATA_KEY = "metadata_json"
_IDENTITY_NAME = "identity"

DEFAULT_ACTIVATIONS: dict[str, Callable[[ArrayFloat], ArrayFloat]] = {
    _IDENTITY_NAME: activation_module.identity,
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
    f"{_IDENTITY_NAME}_derivative": activation_module.identity_derivative,
}


@dataclass(slots=True)
class SerializedModelMetadata:
    """Describes non-parameter data embedded alongside a saved network."""

    architecture: dict[str, Any]
    training: Mapping[str, Any] | None
    extras: Mapping[str, Any] | None
    format_version: int = FORMAT_VERSION
    dtype: str = "float32"


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
        * layer_{i}_weights / layer_{i}_bias arrays for each parameterized layer

    Args:
        network: NeuralNetwork to serialize.
        file_path: Target path ending with `.nn`.
        training_metadata: Optional dictionary with information such as epochs or
            validation scores.
        extra_metadata: Optional free-form metadata copied into the file.
    """

    architecture = _serialize_architecture(network)
    dtype = _resolve_network_dtype(network)
    payload: dict[str, Any] = {
        "format": "my_torch.nn",
        "format_version": FORMAT_VERSION,
        "dtype": dtype.name,
        "architecture": architecture,
    }
    if training_metadata is not None:
        payload["training"] = dict(training_metadata)
    if extra_metadata is not None:
        payload["extras"] = dict(extra_metadata)

    arrays = _serialize_parameters(network, dtype=dtype)
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
        format_version = int(metadata.get("format_version", _LEGACY_FORMAT_VERSION))
        architecture = metadata.get("architecture")
        if not isinstance(architecture, Mapping):
            raise ValueError("architecture metadata is missing or invalid")

        layer_defs = architecture.get("layers", [])
        if not isinstance(layer_defs, list):
            raise ValueError("architecture.layers must be a list")

        dtype = _resolve_metadata_dtype(metadata, archive)
        if format_version == _LEGACY_FORMAT_VERSION:
            layers = _deserialize_layers_v1(
                layer_defs,
                archive,
                activation_lookup,
                derivative_lookup,
                dtype,
            )
        elif format_version == FORMAT_VERSION:
            layers = _deserialize_layers_v2(
                layer_defs,
                archive,
                activation_lookup,
                derivative_lookup,
                dtype,
            )
        else:
            raise ValueError(f"unsupported format version {format_version}")

    network = NeuralNetwork(layers=layers)
    metadata_obj = SerializedModelMetadata(
        architecture=dict(architecture),
        training=metadata.get("training"),
        extras=metadata.get("extras"),
        format_version=format_version,
        dtype=dtype.name,
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


def _serialize_parameters(
    network: NeuralNetwork, *, dtype: np.dtype[Any]
) -> dict[str, np.ndarray[Any, Any]]:
    arrays: dict[str, np.ndarray[Any, Any]] = {}
    for idx, layer in enumerate(network.layers):
        if isinstance(layer, DenseLayer):
            arrays[f"layer_{idx}_weights"] = np.asarray(layer.weights, dtype=dtype)
            arrays[f"layer_{idx}_bias"] = np.asarray(layer.bias, dtype=dtype)
            continue
        if isinstance(layer, Conv2DLayer):
            arrays[f"layer_{idx}_weights"] = np.asarray(layer.weights, dtype=dtype)
            arrays[f"layer_{idx}_bias"] = np.asarray(layer.bias, dtype=dtype)
            continue
        if isinstance(layer, (FlattenLayer, DropoutLayer, GlobalAvgPool2D)):
            continue
        raise TypeError(
            "only DenseLayer, Conv2DLayer, FlattenLayer, DropoutLayer, and "
            "GlobalAvgPool2D instances can be serialized"
        )
    return arrays


def _serialize_architecture(network: NeuralNetwork) -> dict[str, Any]:
    return {
        "layers": [
            _serialize_layer(layer, idx) for idx, layer in enumerate(network.layers)
        ]
    }


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


def _serialize_layer(layer: TrainableLayer, idx: int) -> dict[str, Any]:
    if isinstance(layer, DenseLayer):
        return _serialize_dense_layer(layer, idx)
    if isinstance(layer, Conv2DLayer):
        return _serialize_conv2d_layer(layer, idx)
    if isinstance(layer, FlattenLayer):
        return {"type": "flatten", "index": idx}
    if isinstance(layer, DropoutLayer):
        return {"type": "dropout", "index": idx, "p": layer.p}
    if isinstance(layer, GlobalAvgPool2D):
        return {"type": "global_avg_pool2d", "index": idx}
    raise TypeError(
        "only DenseLayer, Conv2DLayer, FlattenLayer, DropoutLayer, and "
        "GlobalAvgPool2D instances can be serialized"
    )


def _serialize_conv2d_layer(layer: Conv2DLayer, idx: int) -> dict[str, Any]:
    activation_name = _callable_to_name(layer.activation, allow_identity=True)
    derivative_name = _callable_to_name(layer.activation_derivative)
    if activation_name == "_identity":
        activation_name = _IDENTITY_NAME
    return {
        "type": "conv2d",
        "index": idx,
        "in_channels": layer.in_channels,
        "out_channels": layer.out_channels,
        "kernel_size": layer.kernel_size,
        "stride": layer.stride,
        "padding": layer.padding,
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
        return str(name)
    raise ValueError(
        "callables used in serializable layers must define a __name__ attribute"
    )


def _deserialize_layer(
    index: int,
    config: Mapping[str, Any],
    archive: Mapping[str, np.ndarray[Any, Any]],
    activation_registry: ActivationRegistry,
    derivative_registry: ActivationRegistry,
    dtype: np.dtype[Any],
) -> TrainableLayer:
    layer_type = str(config.get("type", "dense")).lower()
    if layer_type == "dense":
        weights, bias = _require_layer_parameters(archive, index)
        return _deserialize_dense_layer(
            index,
            config,
            weights,
            bias,
            activation_registry,
            derivative_registry,
            dtype,
        )
    if layer_type == "conv2d":
        weights, bias = _require_layer_parameters(archive, index)
        return _deserialize_conv2d_layer(
            index,
            config,
            weights,
            bias,
            activation_registry,
            derivative_registry,
            dtype,
        )
    if layer_type == "flatten":
        return FlattenLayer()
    if layer_type == "dropout":
        if "p" not in config:
            raise ValueError("dropout layer config missing required 'p' value")
        return DropoutLayer(p=float(config["p"]))
    if layer_type == "global_avg_pool2d":
        return GlobalAvgPool2D()
    raise ValueError(f"unsupported layer type '{layer_type}' in .nn file")


def _require_layer_parameters(
    archive: Mapping[str, np.ndarray[Any, Any]], index: int
) -> tuple[ArrayFloat, ArrayFloat]:
    weights_key = f"layer_{index}_weights"
    bias_key = f"layer_{index}_bias"
    try:
        weights = archive[weights_key]
        bias = archive[bias_key]
    except KeyError as exc:
        raise ValueError(
            f"missing parameter arrays for layer {index}: {exc.args[0]}"
        ) from exc
    return weights, bias


def _deserialize_dense_layer(
    index: int,
    config: Mapping[str, Any],
    weights: ArrayFloat,
    bias: ArrayFloat,
    activation_registry: ActivationRegistry,
    derivative_registry: ActivationRegistry,
    dtype: np.dtype[Any],
) -> DenseLayer:
    in_features = int(config["in_features"])
    out_features = int(config["out_features"])

    activation_name = config.get("activation")
    derivative_name = config.get("activation_derivative")
    if derivative_name == "<lambda>":
        derivative_name = f"{_IDENTITY_NAME}_derivative"

    expected_weights_shape = (out_features, in_features)
    if weights.shape != expected_weights_shape:
        raise ValueError(
            f"layer {index} weights shape {weights.shape} does not match "
            f"expected {expected_weights_shape}"
        )
    expected_bias_shape = (out_features,)
    if bias.shape != expected_bias_shape:
        raise ValueError(
            f"layer {index} bias shape {bias.shape} does not match "
            f"expected {expected_bias_shape}"
        )

    activation_fn = _resolve_activation(
        activation_registry, activation_name, required=False
    )
    derivative_fn = _resolve_activation(
        derivative_registry, derivative_name, required=False
    )

    identity_fn = activation_registry.get(
        _IDENTITY_NAME, activation_module.identity
    )
    dense = DenseLayer(
        in_features=in_features,
        out_features=out_features,
        activation=activation_fn if activation_fn is not None else identity_fn,
        activation_derivative=derivative_fn,
        weight_initializer=config.get("weight_initializer", "xavier"),
        bias_initializer=config.get("bias_initializer", "zeros"),
    )
    dense.weights = np.asarray(weights, dtype=dtype)
    dense.bias = np.asarray(bias, dtype=dtype)
    dense.grad_weights = np.zeros_like(dense.weights)
    dense.grad_bias = np.zeros_like(dense.bias)
    return dense


def _deserialize_conv2d_layer(
    index: int,
    config: Mapping[str, Any],
    weights: ArrayFloat,
    bias: ArrayFloat,
    activation_registry: ActivationRegistry,
    derivative_registry: ActivationRegistry,
    dtype: np.dtype[Any],
) -> Conv2DLayer:
    in_channels = int(config["in_channels"])
    out_channels = int(config["out_channels"])
    kernel_size = _normalize_pair_config(config["kernel_size"], "kernel_size")
    stride = _normalize_pair_config(config.get("stride", 1), "stride")
    padding = _normalize_pair_config(config.get("padding", 0), "padding")

    activation_name = config.get("activation")
    derivative_name = config.get("activation_derivative")
    if derivative_name == "<lambda>":
        derivative_name = f"{_IDENTITY_NAME}_derivative"

    expected_weights_shape = (
        out_channels,
        in_channels,
        kernel_size[0],
        kernel_size[1],
    )
    if weights.shape != expected_weights_shape:
        raise ValueError(
            f"layer {index} weights shape {weights.shape} does not match "
            f"expected {expected_weights_shape}"
        )
    expected_bias_shape = (out_channels,)
    if bias.shape != expected_bias_shape:
        raise ValueError(
            f"layer {index} bias shape {bias.shape} does not match "
            f"expected {expected_bias_shape}"
        )

    activation_fn = _resolve_activation(
        activation_registry, activation_name, required=False
    )
    derivative_fn = _resolve_activation(
        derivative_registry, derivative_name, required=False
    )

    identity_fn = activation_registry.get(
        _IDENTITY_NAME, activation_module.identity
    )
    conv = Conv2DLayer(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        activation=activation_fn if activation_fn is not None else identity_fn,
        activation_derivative=derivative_fn,
        weight_initializer=config.get("weight_initializer", "he_normal"),
        bias_initializer=config.get("bias_initializer", "zeros"),
    )
    conv.weights = np.asarray(weights, dtype=dtype)
    conv.bias = np.asarray(bias, dtype=dtype)
    conv.grad_weights = np.zeros_like(conv.weights)
    conv.grad_bias = np.zeros_like(conv.bias)
    return conv


def _deserialize_layers_v1(
    layer_defs: list[Any],
    archive: Mapping[str, np.ndarray[Any, Any]],
    activation_registry: ActivationRegistry,
    derivative_registry: ActivationRegistry,
    dtype: np.dtype[Any],
) -> list[TrainableLayer]:
    layers: list[TrainableLayer] = []
    for idx, layer_cfg in enumerate(layer_defs):
        if not isinstance(layer_cfg, Mapping):
            raise ValueError("layer configuration entries must be mappings")
        layer_type = str(layer_cfg.get("type", "dense")).lower()
        if layer_type != "dense":
            raise ValueError(
                f"format version {_LEGACY_FORMAT_VERSION} supports dense layers only"
            )
        weights, bias = _require_layer_parameters(archive, idx)
        layers.append(
            _deserialize_dense_layer(
                idx,
                layer_cfg,
                weights,
                bias,
                activation_registry,
                derivative_registry,
                dtype,
            )
        )
    return layers


def _deserialize_layers_v2(
    layer_defs: list[Any],
    archive: Mapping[str, np.ndarray[Any, Any]],
    activation_registry: ActivationRegistry,
    derivative_registry: ActivationRegistry,
    dtype: np.dtype[Any],
) -> list[TrainableLayer]:
    layers: list[TrainableLayer] = []
    for idx, layer_cfg in enumerate(layer_defs):
        if not isinstance(layer_cfg, Mapping):
            raise ValueError("layer configuration entries must be mappings")
        layers.append(
            _deserialize_layer(
                idx,
                layer_cfg,
                archive,
                activation_registry,
                derivative_registry,
                dtype,
            )
        )
    return layers


def _normalize_pair_config(value: Any, name: str) -> tuple[int, int]:
    if isinstance(value, list):
        if len(value) != 2:
            raise ValueError(f"{name} must be a list of two integers")
        return int(value[0]), int(value[1])
    if isinstance(value, tuple):
        if len(value) != 2:
            raise ValueError(f"{name} must be a tuple of two integers")
        return int(value[0]), int(value[1])
    return int(value), int(value)


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


def _resolve_network_dtype(network: NeuralNetwork) -> np.dtype[Any]:
    dtypes: list[np.dtype[Any]] = []
    for layer in network.layers:
        if isinstance(layer, (DenseLayer, Conv2DLayer)):
            dtypes.append(np.asarray(layer.weights).dtype)
            dtypes.append(np.asarray(layer.bias).dtype)
    if not dtypes:
        return np.dtype(np.float32)
    dtype = np.result_type(*dtypes)
    if not np.issubdtype(dtype, np.floating):
        raise ValueError("only floating point dtypes can be serialized")
    return np.dtype(dtype)


def _resolve_metadata_dtype(
    metadata: Mapping[str, Any],
    archive: Mapping[str, np.ndarray[Any, Any]],
) -> np.dtype[Any]:
    dtype_raw = metadata.get("dtype")
    if isinstance(dtype_raw, str):
        try:
            return np.dtype(dtype_raw)
        except TypeError as exc:
            raise ValueError(f"unsupported dtype '{dtype_raw}' in metadata") from exc
    inferred = _infer_archive_dtype(archive)
    if inferred is not None:
        return inferred
    return np.dtype(np.float32)


def _infer_archive_dtype(
    archive: Mapping[str, np.ndarray[Any, Any]]
) -> np.dtype[Any] | None:
    for key in archive.keys():
        if key == _METADATA_KEY:
            continue
        array = archive[key]
        return np.asarray(array).dtype
    return None
