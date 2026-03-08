import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler

from machine_learning.regression.gradient_based.basic import LinearRegressorGD
from machine_learning.regression.gradient_based.momentum import (
    LinearRegressionGDMomentum,
)


def test_momentum_gd__no_momentum():
    # given
    rng = np.random.default_rng(42)
    X = rng.standard_normal(size=(100, 2))
    y = X @ np.array([1.5, -2.0]) + 3
    X_test = np.array([0, 0, 1, 1]).reshape((2, 2)).astype(float)
    momentum_model = LinearRegressionGDMomentum(
        learning_rate=0.1,
        num_epochs=50,
    )
    gd_model = LinearRegressorGD(
        learning_rate=0.1,
        num_iterations=50,
    )

    # when
    momentum_model.fit(X, y)
    momentum_pred = momentum_model.predict(X_test)
    gd_model.fit(X, y)
    gd_pred = gd_model.predict(X_test)

    # then
    assert list(gd_pred) == pytest.approx(list(momentum_pred), rel=0.1)


def test_momentum_gd__faster_convergence_than_gd():
    # given
    rng = np.random.default_rng(42)
    X = rng.standard_normal(size=(100, 2))
    y = X @ np.array([1.5, -2.0]) + 3
    momentum_model = LinearRegressionGDMomentum(
        learning_rate=0.1,
        num_epochs=1_000,
        beta=0.7,
    )
    gd_model = LinearRegressorGD(
        learning_rate=0.1,
        num_iterations=1_000,
    )

    # when
    momentum_model.fit(X, y)
    gd_model.fit(X, y)

    # then
    assert len(momentum_model.loss_history) < len(gd_model.loss_history)


@pytest.mark.parametrize("beta", [0.1, 0.3, 0.5, 0.7, 0.9])
def test_momentum_gd_converges(closed_form_solution, beta: float):
    # given
    rng = np.random.default_rng(42)
    m, d = 200, 3
    X = rng.standard_normal(size=(m, d))
    true_coef = np.array([2.0, 1.5, -1.0, 0.5])
    X_with_intercept = np.column_stack([np.ones(m), X])
    y = X_with_intercept @ true_coef + 0.1 * rng.standard_normal(m)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = LinearRegressionGDMomentum(
        learning_rate=0.1,
        num_epochs=50,
        beta=beta,
    )

    # when
    model.fit(X_scaled, y)

    # then
    coef_closed = closed_form_solution(X_scaled, y)
    np.testing.assert_allclose(model.coef, coef_closed, rtol=0.05, atol=0.05)


@pytest.mark.parametrize("beta", [0.1, 0.3, 0.5, 0.7, 0.9])
def test_momentum_prediction(beta: float):
    # given
    X = np.array([[1], [2], [3], [4], [5], [6], [7], [8]], dtype=float)
    y = np.array([2, 4, 6, 8, 10, 12, 14, 16], dtype=float)
    X_test = np.array([[9], [10]], dtype=float)
    model = LinearRegressionGDMomentum(
        learning_rate=0.01,
        num_epochs=100,
        beta=beta,
    )

    # when
    model.fit(X, y)
    predictions = model.predict(X_test)

    # then
    expected = np.array([18, 20], dtype=float)
    np.testing.assert_allclose(predictions, expected, rtol=0.1, atol=0.5)
