import numpy as np

from machine_learning.model import Matrix, Vector
from machine_learning.regression.close_form._base import BaseRegressor
from machine_learning.utils import add_interception_to_matrix


class LinearRegressor(BaseRegressor):
    """Linear regression class which basis on closed form solution.
    Linear Regression using Ordinary Least Squares (OLS).

    Implements Equation 3.4 from "The Elements of Statistical Learning":
    coef = (X^T X)^(-1) X^T y
    """

    def _update_coef(self, features_matrix, target_values):
        return _update_coef(features_matrix, target_values)

    def _fit_step(self, features_matrix: Matrix, target_values: Vector):
        features_matrix_with_intercept = add_interception_to_matrix(
            features_matrix
        )
        self._coef = self._update_coef(
            features_matrix_with_intercept, target_values
        )


def _update_coef(features_matrix: Matrix, target_values: Vector) -> Vector:
    coef: Vector
    try:
        coef = (
            np.linalg.inv(features_matrix.T @ features_matrix)
            @ features_matrix.T
            @ target_values
        )
    except np.linalg.LinAlgError:
        coef, *_ = np.linalg.lstsq(features_matrix, target_values, rcond=None)
    return coef
