from abc import ABC, abstractmethod

import numpy as np
from pydantic import PositiveInt, validate_call

from machine_learning.model import Matrix, Vector


class AbstractLinearRegressor(ABC):
    _coef: np.ndarray

    @validate_call
    def __init__(self, shape: PositiveInt):
        self._coef = np.ndarray((shape, 1), dtype=np.float32)

    @abstractmethod
    def fit(self, features_matrix: Matrix, target_values: Vector): ...

    def predict(self, features_matrix: Matrix):
        features_matrix_with_intercept = (
            np.ones((features_matrix.shape[0], 1))
            if features_matrix.shape[1] == 0
            else np.column_stack(
                [np.ones(features_matrix.shape[0]), features_matrix]
            )
        )
        return features_matrix_with_intercept @ self._coef

    @property
    def coef(self) -> np.ndarray:
        return self._coef
