from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from numpy.typing import NDArray

ArrayFloat = NDArray[np.floating]

__all__ = ["SGD"]


@dataclass(slots=True)
class SGD:
    """
    Stochastic gradient descent optimizer with optional L2 weight decay.

    The optimizer can be used in two modes:
    - Call :meth:`update` to produce an updated parameter array (functional style).
    - Call :meth:`step` with parameter and gradient sequences to update in place.
    """

    lr: float
    weight_decay: float = 0.0

    def __post_init__(self) -> None:
        if self.lr <= 0:
            raise ValueError("lr must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")

    def update(self, param: ArrayFloat, grad: ArrayFloat) -> ArrayFloat:
        """
        Return a new parameter array after applying a single SGD step.

        Args:
            param: Array representing the parameter tensor.
            grad: Gradient of the loss with respect to ``param``.

        Raises:
            ValueError: If ``param`` and ``grad`` shapes differ.
        """
        param_array = np.asarray(param, dtype=float)
        grad_array = np.asarray(grad, dtype=float)
        if param_array.shape != grad_array.shape:
            raise ValueError("param and grad must share the same shape")

        if self.weight_decay:
            grad_array = grad_array + self.weight_decay * param_array
        return param_array - self.lr * grad_array

    def step(
        self, parameters: Sequence[ArrayFloat], gradients: Sequence[ArrayFloat]
    ) -> None:
        """
        Update all parameters in place using the provided gradients.

        Args:
            parameters: Sequence of parameter arrays to be updated.
            gradients: Sequence of gradients matching ``parameters`` by index.

        Raises:
            ValueError: If the parameter and gradient sequences differ in length.
        """
        if len(parameters) != len(gradients):
            raise ValueError("parameters and gradients must have the same length")

        for param, grad in zip(parameters, gradients):
            param[...] = self.update(param, grad)

    @staticmethod
    def zero_grad(gradients: Iterable[ArrayFloat]) -> None:
        """
        Zero out all gradient arrays in place.

        Args:
            gradients: Iterable of gradient arrays to be zeroed.
        """
        for grad in gradients:
            np.asarray(grad).fill(0.0)
