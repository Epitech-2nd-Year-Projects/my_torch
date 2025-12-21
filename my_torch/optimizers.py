from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np
from numpy.typing import NDArray

ArrayFloat = NDArray[np.floating]

__all__ = ["SGD", "SGDMomentum", "AdamW"]


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

        if self.weight_decay and param_array.ndim >= 2:
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
            ValueError: If sequence lengths differ or parameter/gradient shapes mismatch.
        """
        if len(parameters) != len(gradients):
            raise ValueError("parameters and gradients must have the same length")

        for param, grad in zip(parameters, gradients):
            if param.shape != grad.shape:
                raise ValueError("param and grad must share the same shape")
            if self.weight_decay and param.ndim >= 2:
                param *= 1.0 - self.lr * self.weight_decay
            if self.lr != 1.0:
                np.multiply(grad, self.lr, out=grad)
                param -= grad
                np.multiply(grad, 1.0 / self.lr, out=grad)
            else:
                param -= grad

    @staticmethod
    def zero_grad(gradients: Iterable[ArrayFloat]) -> None:
        """
        Zero out all gradient arrays in place.

        Args:
            gradients: Iterable of gradient arrays to be zeroed.
        """
        for grad in gradients:
            np.asarray(grad).fill(0.0)


@dataclass(slots=True)
class SGDMomentum:
    """
    SGD with momentum and optional L2 weight decay.

    Supports in-place updates via :meth:`step` or :meth:`update`.
    """

    lr: float
    momentum: float = 0.9
    weight_decay: float = 0.0
    _velocity: dict[int, ArrayFloat] = field(
        default_factory=dict, init=False, repr=False
    )

    def __post_init__(self) -> None:
        if self.lr <= 0:
            raise ValueError("lr must be positive")
        if not 0.0 <= self.momentum < 1.0:
            raise ValueError("momentum must be in the range [0, 1)")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")

    def _get_velocity(self, param: ArrayFloat) -> ArrayFloat:
        key = id(param)
        velocity = self._velocity.get(key)
        if velocity is None or velocity.shape != param.shape:
            velocity = np.zeros_like(param)
            self._velocity[key] = velocity
        return velocity

    def update(self, param: ArrayFloat, grad: ArrayFloat) -> ArrayFloat:
        """
        Update a single parameter array in place and return it.

        Args:
            param: Array representing the parameter tensor.
            grad: Gradient of the loss with respect to ``param``.
        """
        param_array = np.asarray(param)
        grad_array = np.asarray(grad)
        if param_array.shape != grad_array.shape:
            raise ValueError("param and grad must share the same shape")
        velocity = self._get_velocity(param_array)
        velocity *= self.momentum
        if self.weight_decay and param_array.ndim >= 2:
            velocity += grad_array + self.weight_decay * param_array
        else:
            velocity += grad_array
        param_array -= self.lr * velocity
        return param_array

    def step(
        self, parameters: Sequence[ArrayFloat], gradients: Sequence[ArrayFloat]
    ) -> None:
        """
        Update all parameters in place using the provided gradients.
        """
        if len(parameters) != len(gradients):
            raise ValueError("parameters and gradients must have the same length")
        for param, grad in zip(parameters, gradients):
            if param.shape != grad.shape:
                raise ValueError("param and grad must share the same shape")
            self.update(param, grad)


@dataclass(slots=True)
class _AdamState:
    m: ArrayFloat
    v: ArrayFloat
    step: int = 0


@dataclass(slots=True)
class AdamW:
    """
    AdamW optimizer with decoupled weight decay.

    Supports in-place updates via :meth:`step` or :meth:`update`.
    """

    lr: float
    weight_decay: float = 0.0
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    _state: dict[int, _AdamState] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.lr <= 0:
            raise ValueError("lr must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if not 0.0 <= self.beta1 < 1.0:
            raise ValueError("beta1 must be in the range [0, 1)")
        if not 0.0 <= self.beta2 < 1.0:
            raise ValueError("beta2 must be in the range [0, 1)")
        if self.eps <= 0:
            raise ValueError("eps must be positive")

    def _get_state(self, param: ArrayFloat) -> _AdamState:
        key = id(param)
        state = self._state.get(key)
        if state is None or state.m.shape != param.shape:
            state = _AdamState(np.zeros_like(param), np.zeros_like(param), 0)
            self._state[key] = state
        return state

    def update(self, param: ArrayFloat, grad: ArrayFloat) -> ArrayFloat:
        """
        Update a single parameter array in place and return it.

        Args:
            param: Array representing the parameter tensor.
            grad: Gradient of the loss with respect to ``param``.
        """
        param_array = np.asarray(param)
        grad_array = np.asarray(grad)
        if param_array.shape != grad_array.shape:
            raise ValueError("param and grad must share the same shape")

        state = self._get_state(param_array)
        state.step += 1

        if self.weight_decay and param_array.ndim >= 2:
            param_array *= 1.0 - self.lr * self.weight_decay

        state.m *= self.beta1
        state.m += (1.0 - self.beta1) * grad_array
        state.v *= self.beta2
        state.v += (1.0 - self.beta2) * (grad_array * grad_array)

        bias_correction1 = 1.0 - self.beta1**state.step
        bias_correction2 = 1.0 - self.beta2**state.step
        m_hat = state.m / bias_correction1
        v_hat = state.v / bias_correction2

        param_array -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
        return param_array

    def step(
        self, parameters: Sequence[ArrayFloat], gradients: Sequence[ArrayFloat]
    ) -> None:
        """
        Update all parameters in place using the provided gradients.
        """
        if len(parameters) != len(gradients):
            raise ValueError("parameters and gradients must have the same length")
        for param, grad in zip(parameters, gradients):
            if param.shape != grad.shape:
                raise ValueError("param and grad must share the same shape")
            self.update(param, grad)
