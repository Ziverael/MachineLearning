import logging
from abc import ABC, abstractmethod
from typing import Final, Literal, cast

import numpy as np
from pydantic import PositiveFloat, PositiveInt, validate_call

from machine_learning.error import ModelValidationError
from machine_learning.model import Matrix, Vector


LOG_INTERVAL: Final[int] = 100

logger = logging.getLogger("RegressionLogger")


class Regressor(ABC):
    _coef: Vector | None

    @validate_call
    def __init__(self, shape: PositiveInt):
        self._coef = np.ndarray((shape, 1), dtype=np.float32)

    @abstractmethod
    def fit(self, features_matrix: Matrix, target_values: Vector): ...

    @abstractmethod
    def _update_coef(
        self,
        *args,
        **kwargs,
    ) -> None: ...

    @abstractmethod
    def _fit_prepare(self, *args, **kwargs) -> None:
        """Steps before real fitting. This assumes to be
        a component of fit method."""

    @abstractmethod
    def _fit_step(self, features_matrix: Matrix, *args, **kwargs) -> None:
        """Real fitting part. This assumes to be
        a component of fit method."""

    def predict(self, features_matrix: Matrix):
        features_matrix_with_intercept = (
            np.ones((features_matrix.shape[0], 1))
            if features_matrix.shape[1] == 0
            else np.column_stack(
                [np.ones(features_matrix.shape[0]), features_matrix]
            )
        )
        return features_matrix_with_intercept @ self.coef

    @property
    def coef(self) -> np.ndarray:
        if self._coef is None:
            msg = "Coefficient is None."
            raise RuntimeError(msg)
        return cast("np.ndarray", self._coef)


class GDRegressor(Regressor):
    def __init__(
        self,
        learning_rate: PositiveFloat = 0.01,
        tolerance: PositiveFloat = 1e-10,
        num_epochs: PositiveInt | None = 50,
        *,
        num_iterations: PositiveInt | None = None,
        logger: logging.Logger = logger,
    ) -> None:
        if num_iterations is not None:
            self._num_epochs: PositiveInt | None = None
            self._num_iterations = num_iterations
            self._is_epoch_based = False
        else:
            self._num_epochs = num_epochs
            self._is_epoch_based = True
        self._learning_rate = learning_rate
        self._tolerance = tolerance
        self._coef: Vector | None = None
        self._loss_history: list[np.float32] = []
        self._current_gradient: np.ndarray | None = None
        self._verbose = False
        self._logger = logger
        self._log_interval = LOG_INTERVAL

    def validate_model_type(
        self, type_: Literal["epoch_based", "iter_based"]
    ) -> None:
        if (self._is_epoch_based and type_ == "epoch_based") or (
            not self._is_epoch_based and type_ == "iter_based"
        ):
            return
        msg = f"Class is not {type_}."
        raise ModelValidationError(msg)

    @abstractmethod
    def _compute_gradient(
        self,
        features_matrix: Matrix,
        target_values: Vector,
        theta: Vector,
    ): ...

    @abstractmethod
    def _compute_loss(
        self, features_matrix: Matrix, target_values: Vector, theta: Vector
    ): ...

    @property
    def learning_rate(self) -> PositiveFloat:
        return self._learning_rate

    @learning_rate.setter
    def learning_rate(self, value: PositiveFloat) -> None:
        self._learning_rate = value

    @property
    def num_iterations(self) -> PositiveInt:
        self.validate_model_type("iter_based")
        return cast("PositiveInt", self._num_iterations)

    @num_iterations.setter
    def num_iterations(self, value: PositiveInt) -> None:
        self.validate_model_type("iter_based")
        self._num_iterations = value

    @property
    def num_epochs(self) -> PositiveInt:
        self.validate_model_type("epoch_based")
        return cast("PositiveInt", self._num_epochs)

    @num_epochs.setter
    def num_epochs(self, value: PositiveInt) -> None:
        self.validate_model_type("epoch_based")
        self._num_epochs = value

    @property
    def tolerance(self) -> PositiveFloat:
        return self._tolerance

    @tolerance.setter
    def tolerance(self, value: PositiveFloat) -> None:
        self._tolerance = value

    @property
    def loss_history(self) -> list[np.float32]:
        return self._loss_history

    @property
    def current_gradient(self) -> np.ndarray:
        if self._current_gradient is None:
            msg = "Current gradient is None."
            raise RuntimeError(msg)
        return cast("np.ndarray", self._current_gradient)

    @property
    def log_interval(self) -> PositiveInt:
        return self._log_interval

    @log_interval.setter
    def log_interval(self, value: PositiveInt) -> None:
        self._log_interval = value

    @property
    def fit_iter(self) -> PositiveInt:
        return (
            cast("PositiveInt", self._num_epochs)
            if self._is_epoch_based
            else cast("PositiveInt", self._num_iterations)
        )

    def _should_log(self, it: int) -> bool:
        return bool(it % self._log_interval)

    def verbose_on(self) -> None:
        self._verbose = True

    def verbose_off(self) -> None:
        self._verbose = False

    def _log_progress(self, it: int, loss: PositiveFloat) -> None:
        iter_text = "Epoch:" if self._is_epoch_based else "Iteration:"
        if self._verbose:
            self._logger.info(
                "%s %s Loss: %s Coef: %s",
                iter_text,
                it,
                loss,
                ", ".join(f"{v:.2f}" for v in cast("np.ndarray", self._coef)),
            )
        else:
            self._logger.debug(
                "%s %s Loss: %s Coef: %s",
                iter_text,
                it,
                loss,
                ", ".join(f"{v:.2f}" for v in cast("np.ndarray", self._coef)),
            )

    def reset(self) -> None:
        self._coef = None
        self._loss_history = []
        self._current_gradient = None

    def _update_coef(self) -> None:
        self._coef = self.coef - self.learning_rate * self.current_gradient

    def _is_early_stopped(self) -> bool:
        """There is alternative condition:
        np.linalg.norm(self._current_gradient) < self.tolerance
        however this not pass tests.
        """
        if len(self.loss_history) > 0 and (
            abs(self.loss_history[-1] - self.loss_history[-2]) < self.tolerance
        ):
            if self._verbose:
                logger.info(
                    "Converged at iteration: %s.", len(self.loss_history)
                )
            return True
        return False
