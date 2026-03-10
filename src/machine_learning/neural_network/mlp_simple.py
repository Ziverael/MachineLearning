"""Simple Multi-Layer Perceptron implementation.

This module implements a simple MLP with one hidden layer,
trained using backpropagation and gradient descent.
Architecture:
- Input layer: receives features
- Hidden layer: performs intermediate computations
- Output layer: produces final prediction
"""

import logging

import numpy as np
from numpy.typing import NDArray


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

EPSILON = 1e-15
LIMIT = 500


class SimpleMLP:
    def __init__(
        self,
        n_input: int,
        n_hidden: int,
        n_output: int,
        learning_rate: float = 0.1,
        random_state: int | None = None,
    ) -> None:
        self.learning_rate = learning_rate
        self.rng = np.random.default_rng(random_state)
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output

        self.W1 = np.zeros((n_input, n_hidden))
        self.W2 = np.zeros((n_hidden, n_output))
        self.b1 = np.zeros(self.n_hidden)
        self.b2 = np.zeros(self.n_output)

    def _initialize_parameters(self) -> None:
        self.W1 = self.rng.normal(
            0, np.sqrt(1.0 / self.n_input), size=(self.n_input, self.n_hidden)
        )
        self.W2 = self.rng.normal(
            0, np.sqrt(1.0 / self.n_hidden), size=(self.n_hidden, self.n_output)
        )
        self.b1 = np.zeros(self.n_hidden)
        self.b2 = np.zeros(self.n_output)

    def sigmoid(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        z = np.clip(z, -LIMIT, LIMIT)
        return 1.0 / (1.0 + np.exp(-z))

    def sigmoid_derivative(self, a: NDArray[np.float64]) -> NDArray[np.float64]:
        return a * (1.0 - a)

    def forward(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        self.z1 = X @ self.W1 + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = self.sigmoid(self.z2)
        return self.a2

    def backward(
        self, X: NDArray[np.float64], y: NDArray[np.float64]
    ) -> dict[str, NDArray[np.float64]]:
        m = X.shape[0]
        delta2 = self.a2 - y
        dW2 = (1.0 / m) * np.dot(self.a1.T, delta2)
        db2 = (1.0 / m) * np.sum(delta2, axis=0)
        delta1 = np.dot(delta2, self.W2.T) * self.sigmoid_derivative(self.a1)
        dW1 = (1.0 / m) * np.dot(X.T, delta1)
        db1 = (1.0 / m) * np.sum(delta1, axis=0)
        return {
            "dW1": dW1,
            "dW2": dW2,
            "db1": db1,
            "db2": db2,
        }

    def fit(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.float64],
        n_epochs: int = 1000,
        verbose: bool = True,  # noqa: FBT001,FBT002
    ) -> "SimpleMLP":
        self._initialize_parameters()

        self.loss_history_ = []
        for epoch in range(n_epochs):
            predictions = self.forward(X)
            loss = self.compute_loss(y, predictions)
            self.loss_history_.append(loss)
            gradients = self.backward(X, y)
            self.W1 -= self.learning_rate * gradients["dW1"]
            self.b1 -= self.learning_rate * gradients["db1"]
            self.W2 -= self.learning_rate * gradients["dW2"]
            self.b2 -= self.learning_rate * gradients["db2"]
            if verbose and (epoch % 100 == 0):
                logger.info("Loss: %.2f", loss)
        return self

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        return self.forward(X)

    def compute_loss(
        self,
        y_true: NDArray[np.float64],
        y_pred: NDArray[np.float64],
    ) -> float:
        y_pred = np.clip(y_pred, EPSILON, 1 - EPSILON)
        loss = -np.mean(
            y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)
        )
        return float(loss)
