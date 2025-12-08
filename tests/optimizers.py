from __future__ import annotations

import numpy as np


class SGD:
    def __init__(self, lr: float) -> None:
        self.lr = lr

    def update(self, param: np.ndarray, grad: np.ndarray) -> np.ndarray:
        return param - self.lr * grad
