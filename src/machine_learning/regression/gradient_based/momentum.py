import logging

from machine_learning.model import Matrix, Vector
from machine_learning.regression.gradient_based._base import BaseRegressor


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LinearRegressionGDMomentum(BaseRegressor):
    def __init__(self, *args, beta: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self._momentum_coef = beta
        self._momentums: list[float] = [0.0]

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
        self._momentums.append(self._compute_momentum())
        self._update_coef()
        it_loss = self._compute_loss(features_matrix, target_values, self.coef)
        self.loss_history.append(it_loss)
        if self._should_log(idx):
            self._log_progress(idx, it_loss)

    def _update_coef(self) -> None:
        self._coef = self.coef - self.learning_rate * self._momentums[-1]

    def _compute_momentum(self):
        return (
            self._momentum_coef * self._momentums[-1] + self._current_gradient
        )
