import numpy as np

from machine_learning.model import Matrix, Vector
from machine_learning.regression._regressor import AbstractLinearRegressor
from machine_learning.utils import add_interception_to_matrix


class LinearRegressor(AbstractLinearRegressor):
    """Linear regression class which basis on closed form solution.
    Linear Regression using Ordinary Least Squares (OLS).

    Implements Equation 3.4 from "The Elements of Statistical Learning":
    coef = (X^T X)^(-1) X^T target_values
    """

    def fit(self, features_matrix: Matrix, target_values: Vector):
        intercept_size: int = 1
        if (features_matrix.shape[-1] + intercept_size) != self._coef.shape[0]:
            msg = (
                f"Invalid target shape: {features_matrix.shape[-1]},"
                "expects {self._coef.shape[0] - intercept_size}."
            )
            raise ValueError(msg)
        features_matrix_with_intercept = add_interception_to_matrix(
            features_matrix
        )
        self._coef = _eval_coef(features_matrix_with_intercept, target_values)


def _eval_coef(features_matrix: Matrix, target_values: Vector) -> Vector:
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
