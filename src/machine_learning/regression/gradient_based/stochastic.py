import logging

import numpy as np
from pydantic import PositiveFloat, PositiveInt

from machine_learning.gradient import compute_gradient_ridge
from machine_learning.model import Matrix, Vector
from machine_learning.regression.gradient_based._base import BaseRegressor


logger = logging.getLogger(__name__)


class LinearRegressorSGD(BaseRegressor):
    """Have in mind, that Stochastic Gradient Descent no longer base on
    iterations concept. Instead we use epoch which represents

    In stochastic gradient descent method we choose one example for each epoch
    randomly and use it for gradient calculation.
    """

    def __init__(
        self,
        learning_rate: PositiveFloat = 0.01,
        num_epochs: PositiveInt = 50,
        tolerance: PositiveFloat = 1e-10,
        logger: logging.Logger = logger,
        rng_seed: int | None = None,
        **kwargs,
    ):
        super().__init__(
            learning_rate=learning_rate,
            tolerance=tolerance,
            logger=logger,
            num_epochs=num_epochs,
            **kwargs,
        )
        self.seed = rng_seed
        self._rng = np.random.default_rng(rng_seed)

    def reset(self):
        super().reset()
        self._rng = np.random.default_rng(self.seed)

    def _fit_step(
        self, features_matrix: Matrix, target_values: Vector, idx: PositiveInt
    ):
        m, _dim = features_matrix.shape
        indices = self._rng.permutation(m)
        features_matrix_shuffled = features_matrix[indices]
        target_values_shuffled = target_values[indices]
        for i in range(m):
            xi = features_matrix_shuffled[i].reshape(1, -1)
            yi = np.array([target_values_shuffled[i]])
            self._current_gradient = self._compute_gradient(xi, yi, self.coef)
            self._update_coef()
        epoch_loss = self._compute_loss(
            features_matrix, target_values, self.coef
        )
        self._loss_history.append(epoch_loss)
        if self._should_log(idx):
            self._log_progress(idx, epoch_loss)


class LinearRegressorRidgeGD(LinearRegressorSGD):
    def __init__(
        self,
        learning_rate: PositiveFloat = 0.01,
        tolerance: PositiveFloat = 1e-10,
        num_iterations: PositiveInt = 1000,
        alpha: PositiveFloat = 0.1,
        *,
        logger: logging.Logger = logger,
    ) -> None:
        super().__init__(
            learning_rate=learning_rate,
            tolerance=tolerance,
            num_iterations=num_iterations,
            logger=logger,
        )
        self._alpha = alpha

    @property
    def alpha(self) -> PositiveFloat:
        return self._alpha

    @alpha.setter
    def alpha(self, value: PositiveFloat) -> None:
        self._alpha = value

    def _compute_gradient(
        self,
        features_matrix: Matrix,
        target_values: Vector,
        theta: Vector,
    ):
        return compute_gradient_ridge(
            features_matrix, target_values, theta, self._alpha
        )
