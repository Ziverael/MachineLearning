from abc import ABC

import numpy as np
from tqdm import tqdm

from machine_learning.gradient import compute_gradient
from machine_learning.metrics import mse_loss
from machine_learning.model import Matrix, Vector
from machine_learning.regression._regressor import (
    GDRegressor,
)
from machine_learning.utils import add_interception_to_matrix


class BaseRegressor(GDRegressor, ABC):
    """In default configuration we use basic gradient implementation
    and MSE loss.
    """

    def _fit_prepare(
        self, features_matrix: Matrix, target_values: Vector
    ) -> None:
        self.reset()
        _, dim = features_matrix.shape
        features_matrix_with_intercept = add_interception_to_matrix(
            features_matrix
        )
        self._coef = np.zeros(dim + 1, dtype=np.float64)
        loss = mse_loss(
            features_matrix_with_intercept, target_values, self.coef
        )
        self._loss_history.append(loss)

    def _compute_gradient(
        self,
        features_matrix: Matrix,
        target_values: Vector,
        theta: Vector,
    ):
        return compute_gradient(features_matrix, target_values, theta)

    def _compute_loss(
        self,
        features_matrix: Matrix,
        target_values: Vector,
        theta: Vector,
    ):
        return mse_loss(features_matrix, target_values, theta)

    def fit(self, features_matrix: Matrix, target_values: Vector):
        self._fit_prepare(features_matrix, target_values)
        features_matrix_with_intercept = add_interception_to_matrix(
            features_matrix
        )
        for idx in tqdm(range(self.fit_iter)):
            self._fit_step(features_matrix_with_intercept, target_values, idx)
            if self._is_early_stopped():
                break
