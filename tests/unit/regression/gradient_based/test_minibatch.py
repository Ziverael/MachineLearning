import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler

from machine_learning.regression.gradient_based.minibatch import (
    LinearRegressorMGD,
)


def test_minibatch_gd_different_batch_sizes():
    # given
    rng = np.random.default_rng(42)
    X = rng.standard_normal(size=(100, 2))
    y = X @ np.array([1.5, -2.0]) + 3
    X_test = np.array([0, 0, 1, 1]).reshape((2, 2)).astype(float)
    model1 = LinearRegressorMGD(
        learning_rate=0.1,
        num_epochs=50,
        batch_size=16,
    )
    model2 = LinearRegressorMGD(
        learning_rate=0.1,
        num_epochs=50,
        batch_size=32,
    )

    # when
    model1.fit(X, y)
    pred1 = model1.predict(X_test)
    model2.fit(X, y)
    pred2 = model2.predict(X_test)

    # then
    assert list(pred1) == pytest.approx(list(pred2), rel=0.1)


def test_minibatch_gd_converges(closed_form_solution):
    # given
    rng = np.random.default_rng(42)
    m, d = 200, 3
    X = rng.standard_normal(size=(m, d))
    true_coef = np.array([2.0, 1.5, -1.0, 0.5])
    X_with_intercept = np.column_stack([np.ones(m), X])
    y = X_with_intercept @ true_coef + 0.1 * rng.standard_normal(m)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = LinearRegressorMGD(
        learning_rate=0.1,
        num_epochs=50,
        batch_size=32,
    )

    # when
    model.fit(X_scaled, y)

    # then
    coef_closed = closed_form_solution(X_scaled, y)
    np.testing.assert_allclose(model.coef, coef_closed, rtol=0.05, atol=0.05)


def test_minibatch_gd_different_batch_sizes__list():
    # given
    rng = np.random.default_rng(42)
    X = rng.standard_normal(size=(100, 2))
    y = X @ np.array([1.5, -2.0]) + 3 + 0.1 * rng.standard_normal(100)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    batch_sizes = [16, 32, 64]
    models = []

    # when
    for batch_size in batch_sizes:
        model = LinearRegressorMGD(
            learning_rate=0.1,
            num_epochs=50,
            batch_size=batch_size,
        )

        model.fit(X_scaled, y)
        models.append(model)

    # then
    for i in range(len(models) - 1):
        np.testing.assert_allclose(
            models[i].coef, models[i + 1].coef, rtol=0.1, atol=0.1
        )


def test_minibatch_prediction():
    # given
    X = np.array([[1], [2], [3], [4], [5], [6], [7], [8]], dtype=float)
    y = np.array([2, 4, 6, 8, 10, 12, 14, 16], dtype=float)
    X_test = np.array([[9], [10]], dtype=float)
    model = LinearRegressorMGD(
        learning_rate=0.01,
        num_epochs=100,
        batch_size=4,
    )

    # when
    model.fit(X, y)
    predictions = model.predict(X_test)

    # then
    expected = np.array([18, 20], dtype=float)
    np.testing.assert_allclose(predictions, expected, rtol=0.1, atol=0.5)
