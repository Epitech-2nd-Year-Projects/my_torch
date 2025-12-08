from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, MutableSequence, Protocol, Sequence

import numpy as np
from numpy.typing import NDArray

from .layers import (
    ActivationDerivativeFn,
    ActivationFn,
    DenseLayer,
)

ArrayFloat = NDArray[np.floating]

LayerConfig = Mapping[str, object]


class TrainableLayer(Protocol):
    """
    Lightweight structural protocol for layers used within NeuralNetwork.

    A concrete implementation must support forward/backward passes alongside
    access to its parameters and accumulated gradients.
    """

    def forward(self, inputs: ArrayFloat) -> ArrayFloat:
        ...

    def backward(self, grad_output: ArrayFloat) -> ArrayFloat:
        ...

    def parameters(self) -> Sequence[ArrayFloat]:
        ...

    def gradients(self) -> Sequence[ArrayFloat]:
        ...

    def zero_grad(self) -> None:
        ...


@dataclass
class NeuralNetwork:
    """
    Simple feedforward neural network composed of ordered layers.

    Layers can be supplied directly or constructed from a configuration list
    containing dictionaries with a ``type`` key (currently only ``dense``) and
    the arguments needed to instantiate each layer.
    """

    layers: MutableSequence[TrainableLayer]

    def __init__(self, layers: Iterable[TrainableLayer] | None = None, *, layer_configs: Sequence[LayerConfig] | None = None) -> None:
        if layers is not None and layer_configs is not None:
            raise ValueError("provide either layers or layer_configs, not both")
        if layer_configs is not None:
            self.layers = [self._build_layer_from_config(cfg) for cfg in layer_configs]
        else:
            self.layers = list(layers) if layers is not None else []

    def forward(self, inputs: ArrayFloat) -> ArrayFloat:
        """Run a forward pass through all layers."""
        output = inputs
        for layer in self.layers:
            output = layer.forward(output)
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
        if layer_type != "dense":
            raise ValueError(f"unsupported layer type '{layer_type}'")
        return self._build_dense_layer(config)

    def _build_dense_layer(self, config: LayerConfig) -> DenseLayer:
        try:
            in_features = int(config["in_features"])  # type: ignore[index]
            out_features = int(config["out_features"])  # type: ignore[index]
        except KeyError as exc:
            raise ValueError("dense layer config requires 'in_features' and 'out_features'") from exc

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

    @staticmethod
    def _expect_callable(name: str, candidate: object) -> ActivationFn | ActivationDerivativeFn:
        if not callable(candidate):
            raise TypeError(f"{name} must be callable when provided")
        return candidate  # type: ignore[return-value]
