import numpy as np
import pytest


@pytest.fixture
def closed_form_solution():
    """Return a function that computes the closed-form solution."""

    def _closed_form_solution(X, y):
        m = X.shape[0]
        X_with_intercept = np.column_stack([np.ones(m), X])
        return (
            np.linalg.inv(X_with_intercept.T @ X_with_intercept)
            @ X_with_intercept.T
            @ y
        )

    return _closed_form_solution


@pytest.fixture
def compute_mse():
    """Return a function that computes the closed-form solution."""

    def _compute_mse(y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    return _compute_mse
