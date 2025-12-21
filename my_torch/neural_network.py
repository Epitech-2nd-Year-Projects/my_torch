from __future__ import annotations

from typing import Iterable, Mapping, Protocol, Sequence

from .layers import (
    ActivationDerivativeFn,
    ActivationFn,
    ArrayFloat,
    Conv2DLayer,
    DenseLayer,
    DropoutLayer,
    FlattenLayer,
    GlobalAvgPool2D,
)

LayerConfig = Mapping[str, object]


class TrainableLayer(Protocol):
    """
    Lightweight structural protocol for layers used within NeuralNetwork.

    A concrete implementation must support forward/backward passes alongside
    access to its parameters and accumulated gradients.
    """

    def forward(self, inputs: ArrayFloat, *, training: bool) -> ArrayFloat: ...

    def backward(self, grad_output: ArrayFloat) -> ArrayFloat: ...

    def parameters(self) -> tuple[ArrayFloat, ...]: ...

    def gradients(self) -> tuple[ArrayFloat, ...]: ...

    def zero_grad(self) -> None: ...


class NeuralNetwork:
    """
    Simple feedforward neural network composed of ordered layers.

    Layers can be supplied directly or constructed from a configuration list
    containing dictionaries with a ``type`` key (``dense``, ``conv2d``,
    ``flatten``, ``dropout``, ``global_avg_pool2d``) and the arguments needed to
    instantiate each layer.

    Raises:
        ValueError: If both `layers` and `layer_configs` are provided, if an
            unsupported layer type is specified, if required config keys are
            missing, or if unexpected config keys are present.
        TypeError: If activation or activation_derivative in config are not
            callable.
    """

    layers: list[TrainableLayer]

    def __init__(
        self,
        layers: Iterable[TrainableLayer] | None = None,
        *,
        layer_configs: Sequence[LayerConfig] | None = None,
    ) -> None:
        if layers is not None and layer_configs is not None:
            raise ValueError("provide either layers or layer_configs, not both")
        if layer_configs is not None:
            self.layers = [self._build_layer_from_config(cfg) for cfg in layer_configs]
        else:
            self.layers = list(layers) if layers is not None else []

    def forward(self, inputs: ArrayFloat, *, training: bool = False) -> ArrayFloat:
        """Run a forward pass through all layers."""
        output = inputs
        for layer in self.layers:
            output = layer.forward(output, training=training)
        return output

    def backward(self, grad_loss: ArrayFloat) -> ArrayFloat:
        """Propagate loss gradients back through the network."""
        grad = grad_loss
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def parameters(self) -> list[ArrayFloat]:
        """Flattened list of parameters in layer order."""
        return [param for layer in self.layers for param in layer.parameters()]

    def gradients(self) -> list[ArrayFloat]:
        """Flattened list of gradients in layer order."""
        return [grad for layer in self.layers for grad in layer.gradients()]

    def zero_grad(self) -> None:
        """Reset gradients on all layers."""
        for layer in self.layers:
            layer.zero_grad()

    def _build_layer_from_config(self, config: LayerConfig) -> TrainableLayer:
        layer_type = str(config.get("type", "dense")).lower()
        if layer_type == "dense":
            return self._build_dense_layer(config)
        if layer_type == "conv2d":
            return self._build_conv2d_layer(config)
        if layer_type == "flatten":
            return self._build_flatten_layer(config)
        if layer_type == "dropout":
            return self._build_dropout_layer(config)
        if layer_type == "global_avg_pool2d":
            return self._build_global_avg_pool2d_layer(config)
        raise ValueError(f"unsupported layer type '{layer_type}'")

    def _build_dense_layer(self, config: LayerConfig) -> DenseLayer:
        try:
            in_features = int(config["in_features"])  # type: ignore[call-overload]
            out_features = int(config["out_features"])  # type: ignore[call-overload]
        except KeyError as exc:
            raise ValueError(
                "dense layer config requires 'in_features' and 'out_features' keys"
            ) from exc

        allowed_keys = {
            "in_features",
            "out_features",
            "activation",
            "activation_derivative",
            "weight_initializer",
            "bias_initializer",
            "rng",
        }
        unexpected = set(config) - (allowed_keys | {"type"})
        if unexpected:
            unknown = ", ".join(sorted(unexpected))
            raise ValueError(f"unexpected keys for dense layer config: {unknown}")

        activation = config.get("activation", None)
        activation_derivative = config.get("activation_derivative", None)

        dense_kwargs: dict[str, object] = {
            "in_features": in_features,
            "out_features": out_features,
            "weight_initializer": config.get("weight_initializer", "xavier"),
            "bias_initializer": config.get("bias_initializer", "zeros"),
            "rng": config.get("rng", None),
        }
        if activation is not None:
            dense_kwargs["activation"] = self._expect_callable("activation", activation)
        if activation_derivative is not None:
            dense_kwargs["activation_derivative"] = self._expect_callable(
                "activation_derivative", activation_derivative
            )
        return DenseLayer(**dense_kwargs)  # type: ignore[arg-type]

    def _build_conv2d_layer(self, config: LayerConfig) -> Conv2DLayer:
        try:
            in_channels = int(config["in_channels"])  # type: ignore[call-overload]
            out_channels = int(config["out_channels"])  # type: ignore[call-overload]
            kernel_size = config["kernel_size"]
        except KeyError as exc:
            raise ValueError(
                "conv2d layer config requires 'in_channels', 'out_channels', "
                "and 'kernel_size' keys"
            ) from exc

        allowed_keys = {
            "in_channels",
            "out_channels",
            "kernel_size",
            "stride",
            "padding",
            "activation",
            "activation_derivative",
            "weight_initializer",
            "bias_initializer",
            "rng",
        }
        unexpected = set(config) - (allowed_keys | {"type"})
        if unexpected:
            unknown = ", ".join(sorted(unexpected))
            raise ValueError(f"unexpected keys for conv2d layer config: {unknown}")

        stride = config.get("stride", 1)
        padding = config.get("padding", 0)
        kernel_size = _normalize_pair(kernel_size)
        stride = _normalize_pair(stride)
        padding = _normalize_pair(padding)

        activation = config.get("activation", None)
        activation_derivative = config.get("activation_derivative", None)

        conv_kwargs: dict[str, object] = {
            "in_channels": in_channels,
            "out_channels": out_channels,
            "kernel_size": kernel_size,
            "stride": stride,
            "padding": padding,
            "weight_initializer": config.get("weight_initializer", "he_normal"),
            "bias_initializer": config.get("bias_initializer", "zeros"),
            "rng": config.get("rng", None),
        }
        if activation is not None:
            conv_kwargs["activation"] = self._expect_callable("activation", activation)
        if activation_derivative is not None:
            conv_kwargs["activation_derivative"] = self._expect_callable(
                "activation_derivative", activation_derivative
            )
        return Conv2DLayer(**conv_kwargs)  # type: ignore[arg-type]

    def _build_flatten_layer(self, config: LayerConfig) -> FlattenLayer:
        allowed_keys: set[str] = set()
        unexpected = set(config) - (allowed_keys | {"type"})
        if unexpected:
            unknown = ", ".join(sorted(unexpected))
            raise ValueError(f"unexpected keys for flatten layer config: {unknown}")
        return FlattenLayer()

    def _build_dropout_layer(self, config: LayerConfig) -> DropoutLayer:
        try:
            p = float(config["p"])  # type: ignore[call-overload]
        except KeyError as exc:
            raise ValueError("dropout layer config requires 'p' key") from exc

        allowed_keys = {"p", "rng"}
        unexpected = set(config) - (allowed_keys | {"type"})
        if unexpected:
            unknown = ", ".join(sorted(unexpected))
            raise ValueError(f"unexpected keys for dropout layer config: {unknown}")
        return DropoutLayer(p=p, rng=config.get("rng", None))

    def _build_global_avg_pool2d_layer(self, config: LayerConfig) -> GlobalAvgPool2D:
        allowed_keys: set[str] = set()
        unexpected = set(config) - (allowed_keys | {"type"})
        if unexpected:
            unknown = ", ".join(sorted(unexpected))
            raise ValueError(
                f"unexpected keys for global_avg_pool2d layer config: {unknown}"
            )
        return GlobalAvgPool2D()

    @staticmethod
    def _expect_callable(
        name: str, candidate: object
    ) -> ActivationFn | ActivationDerivativeFn:
        if not callable(candidate):
            raise TypeError(f"{name} must be callable when provided")
        return candidate


def _normalize_pair(value: object) -> int | tuple[int, int]:
    if isinstance(value, list):
        if len(value) != 2:
            raise ValueError("expected a list of two integers")
        return int(value[0]), int(value[1])
    if isinstance(value, tuple):
        if len(value) != 2:
            raise ValueError("expected a tuple of two integers")
        return int(value[0]), int(value[1])
    return int(value)
