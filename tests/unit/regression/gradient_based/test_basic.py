import numpy as np
import pytest
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from machine_learning.regression.gradient_based.basic import (
    LinearRegressorGD,
    LinearRegressorRidgeGD,
)


def test_batch_gd_one_dimensional(closed_form_solution):
    # given
    X = np.array([1, 2, 3, 4, 5]).reshape((5, 1)).astype(float)
    y = np.array([3, 5, 7, 9, 11]).astype(float)  # y = 2x + 1
    X_test = np.array([6, 7, 0]).reshape((3, 1)).astype(float)
    expected_closed = closed_form_solution(X, y)
    X_test_with_intercept = np.column_stack([np.ones(3), X_test])
    expected = X_test_with_intercept @ expected_closed
    model = LinearRegressorGD(
        learning_rate=0.1,
        num_iterations=1000,
    )

    # when
    model.fit(X, y)
    actual = model.predict(X_test)

    # then
    assert list(actual) == pytest.approx(list(expected), rel=1e-2)


def test_batch_gd_perfect_fit():
    X = np.array([1, 2, 3, 4, 5]).reshape((5, 1)).astype(float)
    y = np.array([3, 5, 7, 9, 11]).astype(float)
    model = LinearRegressorGD(learning_rate=0.1, num_iterations=1000)

    # when
    model.fit(X, y)
    predictions = model.predict(X)

    # then
    assert list(predictions) == pytest.approx(list(y), rel=1e-2)


def test_batch_gd_recovers_parameters():
    # given
    X = np.array([1, 2, 3, 4, 5]).reshape((5, 1)).astype(float)
    true_slope = 3.0
    true_intercept = 2.0
    y = true_slope * X.flatten() + true_intercept
    model = LinearRegressorGD(learning_rate=0.1, num_iterations=2000)

    # when
    model.fit(X, y)

    # then
    expected__coef = [true_intercept, true_slope]
    assert list(model._coef) == pytest.approx(expected__coef, rel=1e-2)


def test_batch_gd_loss_decreases():
    # given
    X = np.array([1, 2, 3, 4, 5]).reshape((5, 1)).astype(float)
    y = np.array([3, 5, 7, 9, 11]).astype(float)
    model = LinearRegressorGD(learning_rate=0.1, num_iterations=100)

    # when
    model.fit(X, y)

    # then
    assert model.loss_history[-1] < model.loss_history[0]


def test_convergence_to_closed_form(closed_form_solution):
    # given
    X = np.array([1, 3, 2, 5, 4, 6]).reshape((6, 1)).astype(float)
    y = np.array([2, 5, 3, 8, 7, 10]).astype(float)
    _coef_closed = closed_form_solution(X, y)
    model = LinearRegressorGD(learning_rate=0.1, num_iterations=2000)

    # when
    model.fit(X, y)

    # then
    assert list(model._coef) == pytest.approx(list(_coef_closed), rel=0.05)


def test_ridge_gd_one_dimensional():
    # given
    X = np.array([1, 3, 2, 5]).reshape((4, 1)).astype(float)
    y = np.array([2, 5, 3, 8]).astype(float)
    alpha = 0.3
    X_test = np.array([1, 2, 10]).reshape((3, 1)).astype(float)
    expected = Ridge(alpha=alpha).fit(X, y).predict(X_test)
    model = LinearRegressorRidgeGD(
        alpha=alpha,
        learning_rate=0.1,
        num_iterations=2000,
    )

    # when
    model.fit(X, y)
    actual = model.predict(X_test)

    # then
    assert list(actual) == pytest.approx(list(expected), rel=1e-2)


def test_ridge_gd_multi_dimensional():
    # given
    X = (
        np.array([1, 2, 3, 5, 4, 5, 4, 3, 3, 3, 2, 5])
        .reshape((4, 3))
        .astype(float)
    )
    y = np.array([2, 5, 3, 8]).astype(float)
    X_test = (
        np.array([1, 0, 0, 0, 1, 0, 0, 0, 1, 2, 5, 7, -2, 0, 3])
        .reshape((5, 3))
        .astype(float)
    )
    alpha = 0.4
    expected = Ridge(alpha=alpha).fit(X, y).predict(X_test)
    model = LinearRegressorRidgeGD(
        alpha=alpha,
        learning_rate=0.01,
        num_iterations=2000,
    )

    # when
    model.fit(X, y)
    actual = model.predict(X_test)

    # then
    assert list(actual) == pytest.approx(list(expected), rel=0.7)


def test_ridge_gd_zero_alpha():
    # given
    X = np.array([1, 3, 2, 5]).reshape((4, 1)).astype(float)
    y = np.array([2, 5, 3, 8]).astype(float)
    alpha = 0
    X_test = np.array([1, 2, 10]).reshape((3, 1)).astype(float)
    expected = Ridge(alpha=alpha).fit(X, y).predict(X_test)
    model = LinearRegressorRidgeGD(
        alpha=alpha,
        learning_rate=0.1,
        num_iterations=2000,
    )

    # when
    model.fit(X, y)
    actual = model.predict(X_test)

    # then
    assert list(actual) == pytest.approx(list(expected), rel=1e-2)


def test_ridge_reduces_coefficient_magnitude():
    # given
    rng = np.random.default_rng(42)
    X = rng.standard_normal(size=(50, 3))
    y = X @ np.array([3.0, -2.0, 1.5]) + 0.1 * rng.standard_normal(50)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model_ols = LinearRegressorRidgeGD(
        alpha=0,
        learning_rate=0.1,
        num_iterations=2000,
    )
    model_ridge = LinearRegressorRidgeGD(
        alpha=1.0,
        learning_rate=0.1,
        num_iterations=2000,
    )

    # when
    model_ols.fit(X_scaled, y)
    model_ridge.fit(X_scaled, y)

    # then
    ols_coef_magnitude = np.linalg.norm(model_ols.coef[1:])
    ridge_coef_magnitude = np.linalg.norm(model_ridge.coef[1:])
    assert ridge_coef_magnitude < ols_coef_magnitude


def test_ridge_handles_multicollinearity():
    # given
    rng = np.random.default_rng(42)
    X1 = rng.standard_normal(size=(50, 1))
    X2 = X1 + 0.01 * rng.standard_normal(
        size=(50, 1)
    )  # Highly correlated with X1
    X = np.column_stack([X1, X2])
    y = X1.flatten() + 0.1 * rng.standard_normal(50)
    model = LinearRegressorRidgeGD(
        alpha=0.5,
        learning_rate=0.1,
        num_iterations=2000,
    )
    model.fit(X, y)
    predictions = model.predict(X)

    # Should produce reasonable predictions
    mse = np.mean((predictions - y) ** 2)
    assert mse < 1.0  # Should fit reasonably well


def test_higher_alpha_more_regularization():
    """Test that higher alpha values lead to more regularization."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal(size=(50, 3))
    y = X @ np.array([3.0, -2.0, 1.5]) + 0.1 * rng.standard_normal(50)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model_weak = LinearRegressorRidgeGD(
        alpha=0.1,
        learning_rate=0.1,
        num_iterations=2000,
    )
    model_strong = LinearRegressorRidgeGD(
        alpha=10.0,
        learning_rate=0.1,
        num_iterations=2000,
    )

    # when
    model_weak.fit(X_scaled, y)
    model_strong.fit(X_scaled, y)

    # then
    weak_magnitude = np.linalg.norm(model_weak.coef[1:])
    strong_magnitude = np.linalg.norm(model_strong.coef[1:])
    assert strong_magnitude < weak_magnitude


def test_ridge_intercept_not_regularized():
    # given
    rng = np.random.default_rng(42)
    X = rng.standard_normal(size=(50, 2))
    y = (
        X @ np.array([1.5, -2.0]) + 5.0 + 0.1 * rng.standard_normal(50)
    )  # Intercept = 5.0
    model = LinearRegressorRidgeGD(
        alpha=100.0,
        learning_rate=0.1,
        num_iterations=2000,
    )

    # when
    model.fit(X, y)

    # then
    # Intercept should still be close to 5.0 (not shrunk to zero)
    # while slopes should be heavily regularized
    assert abs(model.coef[0] - 5.0) < 1.0  # Intercept not too affected
    assert np.linalg.norm(model.coef[1:]) < 1.0  # Slopes heavily regularized


def test_ridge_loss_decreases():
    # given
    X = np.array([1, 2, 3, 4, 5]).reshape((5, 1)).astype(float)
    y = np.array([3, 5, 7, 9, 11]).astype(float)
    model = LinearRegressorRidgeGD(
        alpha=0.5,
        learning_rate=0.1,
        num_iterations=500,
    )

    # when
    model.fit(X, y)

    # then
    # Loss should decrease
    assert len(model.loss_history) > 0
    assert model.loss_history[-1] < model.loss_history[0]


def test_ridge_matches_sklearn_simple():
    # given
    X = np.array([[1], [2], [3], [4], [5]]).astype(float)
    y = np.array([2, 4, 5, 4, 5]).astype(float)
    alpha = 1.0
    sklearn_model = Ridge(alpha=alpha)
    sklearn_model.fit(X, y)
    sklearn_pred = sklearn_model.predict(X)
    model = LinearRegressorRidgeGD(
        alpha=alpha,
        learning_rate=0.1,
        num_iterations=3000,
    )

    # when
    model.fit(X, y)
    our_pred = model.predict(X)

    # then
    assert list(our_pred) == pytest.approx(list(sklearn_pred), rel=0.05)


def test_ridge_matches_sklearn_random():
    # given
    rng = np.random.default_rng(42)
    X = rng.standard_normal(size=(100, 5))
    y = rng.standard_normal(100)
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
    alpha = 1.0
    sklearn_model = Ridge(alpha=alpha)
    model = LinearRegressorRidgeGD(
        alpha=alpha,
        learning_rate=0.1,
        num_iterations=2000,
    )
    sklearn_model.fit(X_scaled, y_scaled)
    X_test = rng.standard_normal(size=(10, 5))
    X_test_scaled = scaler_X.transform(X_test)
    sklearn_pred = sklearn_model.predict(X_test_scaled)

    # when
    model.fit(X_scaled, y_scaled)
    our_pred = model.predict(X_test_scaled)

    # then
    # Parameters should be close
    # Note: sklearn doesn't expose coef in the same format, so compare
    #  predictions
    assert list(our_pred) == pytest.approx(list(sklearn_pred), rel=0.1)


def test_ridge_with_negative_values():
    # given
    X = np.array([[-2], [-1], [0], [1], [2]]).astype(float)
    y = np.array([-5, -3, -1, 1, 3]).astype(float)
    alpha = 0.5
    model = LinearRegressorRidgeGD(
        alpha=alpha,
        learning_rate=0.1,
        num_iterations=2000,
    )

    # when
    model.fit(X, y)
    predictions = model.predict(X)

    # then
    assert list(predictions) == pytest.approx(list(y), rel=0.1)


def test_ridge_perfect_fit_with_regularization():
    # given
    X = np.array([[1], [2], [3], [4], [5]]).astype(float)
    y = np.array([3, 5, 7, 9, 11]).astype(float)  # Perfect line: y = 2x + 1
    model = LinearRegressorRidgeGD(
        alpha=10.0,
        learning_rate=0.1,
        num_iterations=2000,
    )

    # when
    model.fit(X, y)
    predictions = model.predict(X)

    # then
    # Should NOT achieve perfect fit due to regularization
    mse = np.mean((predictions - y) ** 2)
    assert mse > 0.01  # Some error due to regularization


@pytest.mark.parametrize("alpha", [0.1, 0.5, 1.0, 5.0])
def test_ridge_various_alphas(alpha):
    # given
    X = np.array([1, 3, 2, 5, 4]).reshape((5, 1)).astype(float)
    y = np.array([2, 5, 3, 8, 7]).astype(float)
    sklearn_model = Ridge(alpha=alpha)
    sklearn_model.fit(X, y)
    sklearn_pred = sklearn_model.predict(X)
    model = LinearRegressorRidgeGD(
        alpha=alpha,
        learning_rate=0.1,
        num_iterations=3000,
    )

    # when
    model.fit(X, y)
    our_pred = model.predict(X)

    # then
    assert list(our_pred) == pytest.approx(list(sklearn_pred), rel=0.1)


def test_batch_gd_simple_1d(closed_form_solution):
    # given
    X = np.array([[1], [2], [3], [4], [5]], dtype=float)
    y = np.array([3, 5, 7, 9, 11], dtype=float)  # y = 2x + 1
    model = LinearRegressorGD(
        learning_rate=0.1,
        num_iterations=1000,
    )
    coef_closed = closed_form_solution(X, y)

    # when
    model.fit(X, y)
    predictions = model.predict(X)

    # then
    # Coefficients should be close
    np.testing.assert_allclose(model.coef, coef_closed, rtol=1e-3, atol=1e-3)
    # Predictions should be very accurate
    np.testing.assert_allclose(predictions, y, rtol=1e-2, atol=1e-2)


def test_batch_gd_multiple_features(closed_form_solution):
    # given
    rng = np.random.default_rng(42)
    m, d = 100, 3
    X = rng.standard_normal(size=(m, d))
    true_coef = np.array([2.0, 1.5, -1.0, 0.5])
    X_with_intercept = np.column_stack([np.ones(m), X])
    y = X_with_intercept @ true_coef + 0.1 * rng.standard_normal(m)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = LinearRegressorGD(
        learning_rate=0.1,
        num_iterations=2000,
    )
    coef_closed = closed_form_solution(X_scaled, y)

    # when
    model.fit(X_scaled, y)

    # then
    np.testing.assert_allclose(model.coef, coef_closed, rtol=1e-2, atol=1e-2)


def test_batch_gd_prediction_on_new_data():
    # given
    X_train = np.array([[1], [2], [3], [4]], dtype=float)
    y_train = np.array([2, 4, 6, 8], dtype=float)  # y = 2x
    X_test = np.array([[5], [6], [0]], dtype=float)
    model = LinearRegressorGD(
        learning_rate=0.1,
        num_iterations=1000,
    )
    expected = np.array([10, 12, 0], dtype=float)

    # when
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    # then
    np.testing.assert_allclose(predictions, expected, rtol=0.05, atol=0.1)


def test_batch_gd_loss_decreases__different_case():
    # given
    rng = np.random.default_rng(42)
    X = rng.standard_normal(size=(50, 2))
    y = X @ np.array([1.5, -2.0]) + 3 + 0.1 * rng.standard_normal(50)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = LinearRegressorGD(
        learning_rate=0.1,
        num_iterations=500,
    )

    # when
    model.fit(X_scaled, y)

    # then
    loss_history = model.loss_history
    assert len(loss_history) > 0, "Loss history should not be empty"
    # Allow for small increases due to numerical issues, but overall trend
    # should be down
    assert loss_history[-1] < loss_history[0], (
        "Final loss should be less than initial loss"
    )
    # Check that most consecutive pairs show decrease
    decreases = sum(
        loss_history[i + 1] <= loss_history[i]
        for i in range(len(loss_history) - 1)
    )
    assert decreases > 0.9 * len(loss_history), (
        "Loss should decrease in most iterations"
    )


def test_batch_gd_convergence_tolerance():
    # given
    X = np.array([[1], [2], [3], [4], [5]], dtype=float)
    y = np.array([3, 5, 7, 9, 11], dtype=float)
    model = LinearRegressorGD(
        learning_rate=0.1,
        num_iterations=10000,
        tolerance=1e-6,
    )

    # when
    model.fit(X, y)

    # then
    # Should converge before max iterations
    assert len(model.loss_history) < 10000, (
        "Should converge before max iterations"
    )


def test_learning_rate_too_large():
    # given
    X = np.array([[1], [2], [3], [4]], dtype=float)
    y = np.array([2, 4, 6, 8], dtype=float)
    model = LinearRegressorGD(
        learning_rate=10.0,
        num_iterations=100,
    )

    # when
    model.fit(X, y)

    # then
    # Loss should either increase or oscillate wildly
    loss_history = model.loss_history
    # Check if loss explodes or doesn't converge properly
    # (either NaN, very large values, or non-decreasing pattern)
    if not np.any(np.isnan(loss_history)) and not np.any(
        np.isinf(loss_history)
    ):
        # If no NaN/Inf, check that it doesn't converge well
        final_loss = loss_history[-1]
        initial_loss = loss_history[0]
        # With bad learning rate, final loss shouldn't be much better
        assert final_loss > 0.1 * initial_loss or final_loss > 10


def test_batch_gd_perfect_fit__mse_comparison(compute_mse):
    # given
    X = np.array([[1], [2], [3], [4], [5]], dtype=float)
    y = np.array([3, 5, 7, 9, 11], dtype=float)  # Perfect: y = 2x + 1
    model = LinearRegressorGD(
        learning_rate=0.1,
        num_iterations=2000,
    )

    # when
    model.fit(X, y)
    predictions = model.predict(X)

    # then
    mse = compute_mse(y, predictions)
    assert mse < 1e-4, f"MSE should be very small for perfect fit, got {mse}"


def test_batch_gd_zero_features():
    # given
    X = np.empty((5, 0), dtype=float)
    y = np.array([1, 2, 3, 4, 5], dtype=float)
    model = LinearRegressorGD(
        learning_rate=0.1,
        num_iterations=1000,
    )
    expected_mean = np.mean(y)

    # when
    model.fit(X, y)
    predictions = model.predict(X)

    # then
    np.testing.assert_allclose(predictions, expected_mean, rtol=0.01, atol=0.01)


def test_batch_gd_negative_values():
    # given
    X = np.array([[-2], [-1], [0], [1], [2]], dtype=float)
    y = np.array([-5, -3, -1, 1, 3], dtype=float)  # y = 2x - 1
    model = LinearRegressorGD(
        learning_rate=0.1,
        num_iterations=1000,
    )

    # when
    model.fit(X, y)
    predictions = model.predict(X)

    # then
    np.testing.assert_allclose(predictions, y, rtol=0.05, atol=0.1)


@pytest.mark.parametrize(
    ("n_samples", "n_features"), [(50, 1), (100, 2), (200, 5)]
)
def test_batch_gd_various_sizes(n_samples, n_features, closed_form_solution):
    # given
    rng = np.random.default_rng(42)
    X = rng.standard_normal(size=(n_samples, n_features))
    y = rng.standard_normal(n_samples)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = LinearRegressorGD(
        learning_rate=0.1,
        num_iterations=1000,
    )

    # when
    model.fit(X_scaled, y)

    # then
    coef_closed = closed_form_solution(X_scaled, y)
    np.testing.assert_allclose(model.coef, coef_closed, rtol=0.05, atol=0.05)


@pytest.mark.parametrize("learning_rate", [0.01, 0.05, 0.1, 0.5])
def test_batch_gd_various_learning_rates(learning_rate, closed_form_solution):
    # given
    rng = np.random.default_rng(42)
    X = rng.standard_normal(size=(50, 2))
    y = X @ np.array([1.5, -2.0]) + 3 + 0.1 * rng.standard_normal(50)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = LinearRegressorGD(
        learning_rate=learning_rate,
        num_iterations=2000,
    )

    # when
    model.fit(X_scaled, y)

    # then
    coef_closed = closed_form_solution(X_scaled, y)
    np.testing.assert_allclose(model.coef, coef_closed, rtol=0.1, atol=0.1)
