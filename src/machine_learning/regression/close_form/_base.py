from abc import ABC

from machine_learning.model import Matrix, Vector
from machine_learning.regression._regressor import Regressor


class BaseRegressor(Regressor, ABC):
    def _fit_prepare(self, features_matrix: Matrix) -> None:
        intercept_size: int = 1
        if (features_matrix.shape[-1] + intercept_size) != self.coef.shape[0]:
            msg = (
                f"Invalid target shape: {features_matrix.shape[-1]},"
                f"expects {self.coef.shape[0] - intercept_size}."
            )
            raise ValueError(msg)

    def fit(self, features_matrix: Matrix, target_values: Vector):
        self._fit_prepare(features_matrix)
        self._fit_step(features_matrix, target_values)
