import numpy as np
from pydantic import PositiveFloat

from machine_learning.model import Matrix, Vector
from machine_learning.regression._regressor import AbstractLinearRegressor
from machine_learning.utils import add_interception_to_matrix


class RidgeRegressor(AbstractLinearRegressor):
    """
    Ridge Regression with L2 regularization.

    Implements Equation 3.44 from "The Elements of Statistical Learning":
    coef = (X^T X + αI)^(-1) X^T y
    """

    def __init__(self, *args, regularization_coef: PositiveFloat, **kwargs):
        super().__init__(*args, **kwargs)
        self._regularization_coef = regularization_coef

    def fit(self, features_matrix: Matrix, target_values: Vector):
        intercept_size: int = 1
        if (features_matrix.shape[-1] + intercept_size) != self._coef.shape[0]:
            msg = (
                f"Invalid target shape: {features_matrix.shape[-1]},"
                f"expects {self._coef.shape[0] - intercept_size}."
            )
            raise ValueError(msg)
        features_matrix_with_intercept = add_interception_to_matrix(
            features_matrix
        )
        self._coef = _eval_coef(
            features_matrix_with_intercept,
            target_values,
            self._regularization_coef,
        )


def _eval_coef(
    features_matrix: Matrix,
    target_values: Vector,
    regularization_coef: PositiveFloat,
) -> Vector:
    identity_matrix = np.identity(features_matrix.shape[1])
    # Intercept term is not regularized
    identity_matrix[0, 0] = 0.0
    return (
        np.linalg.inv(
            features_matrix.T @ features_matrix
            + regularization_coef * identity_matrix
        )
        @ features_matrix.T
        @ target_values
    )
