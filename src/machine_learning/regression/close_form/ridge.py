import numpy as np
from pydantic import NonNegativeFloat, PositiveFloat

from machine_learning.model import Matrix, Vector
from machine_learning.regression.close_form._base import BaseRegressor
from machine_learning.utils import add_interception_to_matrix


class RidgeRegressor(BaseRegressor):
    """
    Ridge Regression with L2 regularization.

    Implements Equation 3.44 from "The Elements of Statistical Learning":
    coef = (X^T X + αI)^(-1) X^T y
    """

    def __init__(
        self, *args, regularization_coef: NonNegativeFloat = 0.0, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self._regularization_coef = regularization_coef

    def _update_coef(
        self,
        features_matrix: Matrix,
        target_values: Vector,
        regularizaiton_coef: NonNegativeFloat,
    ) -> None:
        self._coef = _update_coef(
            features_matrix,
            target_values,
            regularizaiton_coef,
        )

    def _fit_step(self, features_matrix: Matrix, target_values: Vector):
        features_matrix_with_intercept = add_interception_to_matrix(
            features_matrix
        )
        self._update_coef(
            features_matrix_with_intercept,
            target_values,
            self._regularization_coef,
        )


def _update_coef(
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
