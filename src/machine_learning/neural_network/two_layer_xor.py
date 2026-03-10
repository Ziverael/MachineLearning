"""Two-Layer XOR Network implementation.

This module implements a manually-configured two-layer neural network
that solves the XOR problem, demonstrating how multi-layer networks
can solve non-linearly separable problems.
"""

from typing import cast

import numpy as np

from machine_learning.model import Vector


class TwoLayerXOR:
    def __init__(self) -> None:
        self.W1 = np.array([[1, 1], [1, 1]])
        self.b1 = np.array([-1.5, -0.5])
        self.W2 = np.array([[-1], [1]])
        self.b2 = np.array([-0.5])

    def heaviside(self, z: float | Vector) -> int | Vector:
        return (
            np.where(z >= 0, 1, 0)
            if isinstance(z, np.ndarray)
            else int(z >= 0.0)
        )

    def forward(self, X: Vector) -> tuple[Vector, Vector]:
        z1 = np.dot(X, self.W1) + self.b1
        h = cast("Vector", self.heaviside(z1))
        z2 = np.dot(h, self.W2) + self.b2
        y = cast("Vector", self.heaviside(z2))
        return y.flatten(), h

    def predict(self, X: Vector) -> Vector:
        y, _ = self.forward(X)
        return y

    def verify_xor(self) -> bool:
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y_true = np.array([0, 1, 1, 0])
        y_pred = self.predict(X)
        return np.array_equal(y_pred, y_true)
