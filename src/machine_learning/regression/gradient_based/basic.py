import logging

from pydantic import PositiveFloat, PositiveInt

from machine_learning.gradient import compute_gradient_ridge
from machine_learning.model import Matrix, Vector
from machine_learning.regression.gradient_based._base import BaseRegressor


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LinearRegressorGD(BaseRegressor):
    def __init__(
        self,
        learning_rate: PositiveFloat = 0.01,
        tolerance: PositiveFloat = 1e-10,
        num_iterations: PositiveInt = 1000,
        *,
        logger: logging.Logger = logger,
    ) -> None:
        super().__init__(
            learning_rate=learning_rate,
            tolerance=tolerance,
            num_iterations=num_iterations,
            logger=logger,
            num_epochs=None,
        )

    def _fit_step(
        self,
        features_matrix: Matrix,
        target_values: Vector,
        idx: int,
    ) -> None:
        self._current_gradient = self._compute_gradient(
            features_matrix,
            target_values,
            self.coef,
        )
        self._update_coef()
        it_loss = self._compute_loss(features_matrix, target_values, self.coef)
        self.loss_history.append(it_loss)
        if self._should_log(idx):
            self._log_progress(idx, it_loss)


class LinearRegressorRidgeGD(LinearRegressorGD):
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
