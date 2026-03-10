"""Perceptron implementation.

This module implements Rosenblatt's Perceptron (1957), the first
trainable neural network model with a learning algorithm.
"""

from typing import cast

import numpy as np

from machine_learning.model import Vector


class Perceptron:
    def __init__(
        self,
        n_inputs: int,
        learning_rate: float = 0.1,
        random_state: int | None = None,
    ) -> None:
        self.n_inputs = n_inputs
        self.learning_rate = learning_rate
        self.weights = np.ndarray(self.n_inputs, dtype=np.float64)
        self.bias: np.float64 = np.float64(0.0)
        self.errors_: list[int] = []
        self.rng = np.random.default_rng(random_state)

    def _initialize_weights(self) -> None:
        self.bias = np.float64(0.0)
        self.weights = self.rng.normal(loc=0.0, scale=0.01, size=self.n_inputs)

    def activation(self, z: float | Vector) -> int | Vector:
        if isinstance(z, np.ndarray):
            return np.where(z >= 0, 1, 0)
        return int(z >= 0)

    def forward(self, X: Vector) -> Vector:
        return cast(
            "Vector", self.activation(np.dot(X, self.weights) + self.bias)
        )

    def fit(self, X: Vector, y: Vector, n_epochs: int = 100) -> "Perceptron":
        self._initialize_weights()
        self.errors_ = []
        for _ in range(n_epochs):
            epoch_errors = 0
            for X_train, y_train in zip(X, y, strict=False):
                y_pred = self.activation(self.weights @ X_train + self.bias)
                error = y_train - y_pred
                if error != 0:
                    self.weights += self.learning_rate * error * X_train
                    self.bias += self.learning_rate * error
                    epoch_errors += 1
            self.errors_.append(epoch_errors)
        return self

    def predict(self, X: Vector) -> Vector:
        return self.forward(X)

    def score(self, X: Vector, y: Vector) -> float:
        preds = self.predict(X)
        return np.mean(preds == y)
