import logging
from typing import Final

import numpy as np
from pydantic import PositiveFloat, PositiveInt

from machine_learning.gradient import compute_gradient_ridge
from machine_learning.model import Matrix, Vector
from machine_learning.regression.gradient_based._base import BaseRegressor


LOG_INTERVAL: Final[int] = 5

logger = logging.getLogger(__name__)


class LinearRegressorMGD(BaseRegressor):
    """Minibatch implementation"""

    def __init__(  # noqa:PLR0913
        self,
        learning_rate: PositiveFloat = 0.01,
        num_epochs: PositiveInt = 50,
        batch_size: PositiveInt = 32,
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
        self._batch_size = batch_size
        self._log_interval = LOG_INTERVAL

    @property
    def batch_size(self) -> PositiveInt:
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value: PositiveInt) -> None:
        self._batch_size = value

    def reset(self):
        super().reset()
        self._rng = np.random.default_rng(self.seed)

    def _fit_step(
        self, features_matrix: Matrix, target_values: Vector, idx: PositiveInt
    ) -> None:
        m, _dim = features_matrix.shape
        indices = self._rng.permutation(m)
        features_matrix_shuffled = features_matrix[indices]
        target_values_shuffled = target_values[indices]

        for start in range(0, m, self._batch_size):
            end = start + self._batch_size
            features_matrix_batch = features_matrix_shuffled[start:end]
            target_values_batch = target_values_shuffled[start:end]
            self._current_gradient = self._compute_gradient(
                features_matrix_batch,
                target_values_batch,
                self.coef,
            )
            self._update_coef()
        epoch_loss = self._compute_loss(
            features_matrix, target_values, self.coef
        )
        self._loss_history.append(epoch_loss)
        if self._should_log(idx):
            self._log_progress(idx, epoch_loss)


class LinearRegressorRidgeMGD(LinearRegressorMGD):
    def __init__(  # noqa:PLR0913
        self,
        learning_rate: PositiveFloat = 0.01,
        num_epochs: PositiveInt = 50,
        batch_size: PositiveInt = 32,
        alpha: PositiveFloat = 0.1,
        tolerance: PositiveFloat = 1e-10,
        *,
        logger: logging.Logger = logger,
        rng_seed: int | None = None,
        **kwargs,
    ):
        super().__init__(
            learning_rate=learning_rate,
            tolerance=tolerance,
            num_epochs=num_epochs,
            batch_size=batch_size,
            logger=logger,
            rng_seed=rng_seed,
            **kwargs,
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
