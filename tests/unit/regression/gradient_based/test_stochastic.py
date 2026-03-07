import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler

from machine_learning.regression.gradient_based.stochastic import (
    LinearRegressorSGD,
)


def test_sgd_one_dimensional(closed_form_solution):
    # given
    X = np.array([1, 2, 3, 4, 5, 6, 7, 8]).reshape((8, 1)).astype(float)
    y = np.array([2, 4, 6, 8, 10, 12, 14, 16]).astype(float)  # y = 2x

    X_test = np.array([9, 10]).reshape((2, 1)).astype(float)
    expected_closed = closed_form_solution(X, y)
    X_test_with_intercept = np.column_stack([np.ones(2), X_test])
    expected = X_test_with_intercept @ expected_closed
    model = LinearRegressorSGD(learning_rate=0.01, num_epochs=100)

    # when
    model.fit(X, y)
    actual = model.predict(X_test)

    # then
    assert list(actual) == pytest.approx(list(expected), rel=0.1)


def test_sgd_general_loss_decrease():
    # given
    rng = np.random.default_rng(42)
    X = rng.standard_normal(size=(100, 2))
    y = X @ np.array([1.5, -2.0]) + 3 + 0.1 * rng.standard_normal(100)
    model = LinearRegressorSGD(learning_rate=0.01, num_epochs=50)

    # when
    model.fit(X, y)

    # then
    assert model.loss_history[-1] < model.loss_history[0]


def test_sgd_prediction():
    # given
    X = np.array([[1], [2], [3], [4]], dtype=float)
    y = np.array([2, 4, 6, 8], dtype=float)
    X_test = np.array([[5], [6]], dtype=float)
    model = LinearRegressorSGD(learning_rate=0.01, num_epochs=100)
    expected = np.array([10, 12], dtype=float)

    # when
    model.fit(X, y)
    predictions = model.predict(X_test)

    # then
    np.testing.assert_allclose(predictions, expected, rtol=0.1, atol=0.5)


def test_sgd_loss_general_decrease():
    # given
    rng = np.random.default_rng(42)
    X = rng.standard_normal(size=(100, 2))
    y = X @ np.array([1.5, -2.0]) + 3 + 0.1 * rng.standard_normal(100)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = LinearRegressorSGD(learning_rate=0.01, num_epochs=50)

    # when
    model.fit(X_scaled, y)

    # then
    assert len(model.loss_history) > 0
    assert model.loss_history[-1] < model.loss_history[0]
