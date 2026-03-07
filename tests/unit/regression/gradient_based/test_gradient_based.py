import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler

from machine_learning.regression.gradient_based.basic import LinearRegressorGD
from machine_learning.regression.gradient_based.minibatch import (
    LinearRegressorMGD,
)
from machine_learning.regression.gradient_based.stochastic import (
    LinearRegressorSGD,
)


def test_all_methods_similar_results():
    # given
    rng = np.random.default_rng(42)
    X = rng.standard_normal(size=(100, 3))
    y = X @ np.array([1.0, 2.0, -1.5]) + 3 + 0.1 * rng.standard_normal(100)
    X_test = np.array([0, 0, 0, 1, 1, 1]).reshape((2, 3)).astype(float)
    gd = LinearRegressorGD(learning_rate=0.1, num_iterations=1000)
    sgd = LinearRegressorSGD(learning_rate=0.01, num_epochs=50)
    mgd = LinearRegressorMGD(learning_rate=0.1, num_epochs=50, batch_size=20)

    # when
    gd.fit(X, y)
    sgd.fit(X, y)
    mgd.fit(X, y)
    pred_gd = gd.predict(X_test)
    pred_sgd = sgd.predict(X_test)
    pred_mgd = mgd.predict(X_test)

    # then
    assert list(pred_gd) == pytest.approx(list(pred_sgd), rel=0.15)
    assert list(pred_gd) == pytest.approx(list(pred_mgd), rel=0.1)


def test_sgd_converges_to_similar_solution():
    # given
    rng = np.random.default_rng(42)
    m, d = 200, 3
    X = rng.standard_normal(size=(m, d))
    true_coef = np.array([2.0, 1.5, -1.0, 0.5])
    X_with_intercept = np.column_stack([np.ones(m), X])
    y = X_with_intercept @ true_coef + 0.1 * rng.standard_normal(m)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model_batch = LinearRegressorGD(
        learning_rate=0.1,
        num_iterations=1000,
    )
    model_batch.fit(X_scaled, y)
    model_sgd = LinearRegressorSGD(
        learning_rate=0.01,
        num_epochs=50,
    )

    # when
    model_sgd.fit(X_scaled, y)

    # then
    np.testing.assert_allclose(
        model_sgd.coef, model_batch.coef, rtol=0.1, atol=0.1
    )


def test_all_methods_converge_similarly():
    # given
    rng = np.random.default_rng(42)
    m, d = 200, 3
    X = rng.standard_normal(size=(m, d))
    true_coef = np.array([2.0, 1.5, -1.0, 0.5])
    X_with_intercept = np.column_stack([np.ones(m), X])
    y = X_with_intercept @ true_coef + 0.1 * rng.standard_normal(m)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model_batch = LinearRegressorGD(
        learning_rate=0.1,
        num_iterations=1000,
    )
    model_sgd = LinearRegressorSGD(
        learning_rate=0.01,
        num_epochs=50,
    )
    model_minibatch = LinearRegressorMGD(
        learning_rate=0.1,
        num_epochs=50,
        batch_size=32,
    )

    # when
    model_batch.fit(X_scaled, y)
    model_sgd.fit(X_scaled, y)
    model_minibatch.fit(X_scaled, y)

    # then
    np.testing.assert_allclose(
        model_batch.coef, model_sgd.coef, rtol=0.1, atol=0.1
    )
    np.testing.assert_allclose(
        model_batch.coef, model_minibatch.coef, rtol=0.05, atol=0.05
    )
