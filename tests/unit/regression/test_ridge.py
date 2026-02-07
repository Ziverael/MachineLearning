from typing import cast

import numpy as np
import pytest
from sklearn.linear_model import (  # type: ignore[import-untyped]
    Ridge as SklearnRidge,
)

from machine_learning.regression.ridge import RidgeRegressor as RidgeRegression


@pytest.mark.parametrize("alpha", np.linspace(0.0, 2.0, 10))
def test_simple_1d_case(alpha: np.float32):
    # given
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([3, 5, 7, 9, 11])  # y = 2x + 1

    # when
    rr = RidgeRegression(2, regularization_coef=cast("float", alpha))
    rr.fit(X, y)
    predictions = rr.predict(X)
    sklearn_rr = SklearnRidge(alpha=alpha)
    sklearn_rr.fit(X, y)
    expected = sklearn_rr.predict(X)

    # then
    np.testing.assert_allclose(predictions, expected, rtol=1e-10)


@pytest.mark.parametrize("alpha", np.linspace(0.01, 2.0, 10))
def test_multidimensional_case(alpha: np.float32):
    """This test uses data where features are linearly dependent
    (each row follows pattern [n, n+1, n+2] scaled by 3), creating
    a singular or near-singular matrix X^T X. Standard OLS using
    matrix inversion will fail, while sklearn uses SVD-based lstsq
    which handles this gracefully.
    """
    # given
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
    y = np.array([14, 32, 50, 68])  # y = x1 + 2*x2 + 3*x3 + 2

    # when
    rr = RidgeRegression(4, regularization_coef=cast("float", alpha))
    rr.fit(X, y)
    predictions = rr.predict(X)
    sklearn_rr = SklearnRidge(alpha=alpha)
    sklearn_rr.fit(X, y)
    expected = sklearn_rr.predict(X)

    # then
    np.testing.assert_allclose(predictions, expected, rtol=1e-10)


def test_multidimensional_case__failing():
    """This test uses data where features are linearly dependent
    (each row follows pattern [n, n+1, n+2] scaled by 3), creating
    a singular or near-singular matrix X^T X. Standard OLS using
    matrix inversion will fail, while sklearn uses SVD-based lstsq
    which handles this gracefully.
    """
    # given
    alpha = 0.0
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
    y = np.array([14, 32, 50, 68])  # y = x1 + 2*x2 + 3*x3 + 2

    # when / then
    rr = RidgeRegression(4, regularization_coef=alpha)
    with pytest.raises(np.linalg.LinAlgError, match="Singular matrix"):
        rr.fit(X, y)


@pytest.mark.parametrize("alpha", np.linspace(0.0, 2.0, 10))
def test_prediction_on_new_data(alpha: np.float32):
    # given
    X_train = np.array([[1], [2], [3], [4]])
    y_train = np.array([2, 4, 6, 8])  # y = 2x
    X_test = np.array([[5], [6], [0]])

    # when
    rr = RidgeRegression(2, regularization_coef=cast("float", alpha))
    rr.fit(X_train, y_train)
    predictions = rr.predict(X_test)
    sklearn_rr = SklearnRidge(alpha=alpha)
    sklearn_rr.fit(X_train, y_train)
    expected = sklearn_rr.predict(X_test)

    # then
    np.testing.assert_allclose(predictions, expected, rtol=1e-10)


@pytest.mark.parametrize("alpha", np.linspace(0.1, 2.0, 10))
def test_single_data_point(alpha: np.float32):
    # given
    X = np.array([[1, 2]])
    y = np.array([5])

    # when
    rr = RidgeRegression(3, regularization_coef=cast("float", alpha))
    rr.fit(X, y)
    predictions = rr.predict(X)

    # then
    # Should predict the training point exactly
    np.testing.assert_allclose(predictions, y, rtol=1e-10)


def test_single_data_point__failing():
    # given
    alpha = 0.0
    X = np.array([[1, 2]])
    y = np.array([5])

    # when / then
    rr = RidgeRegression(3, regularization_coef=cast("float", alpha))
    with pytest.raises(np.linalg.LinAlgError, match="Singular matrix"):
        rr.fit(X, y)


def test_perfect_fit():
    """When data is perfectly linear, the model should achieve
    zero training error.
    """
    # given
    alpha = 0.0
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([3, 5, 7, 9, 11])  # Perfect line: y = 2x + 1

    # when
    rr = RidgeRegression(2, regularization_coef=cast("float", alpha))
    rr.fit(X, y)
    predictions = rr.predict(X)

    # then
    np.testing.assert_allclose(predictions, y, rtol=1e-10)


@pytest.mark.parametrize("alpha", np.linspace(0.1, 2.0, 10))
def test_perfect_fit__failing(alpha: np.float32):
    # given
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([3, 5, 7, 9, 11])  # Perfect line: y = 2x + 1

    # when
    rr = RidgeRegression(2, regularization_coef=cast("float", alpha))
    rr.fit(X, y)
    predictions = rr.predict(X)

    # Should predict training data exactly
    with pytest.raises(AssertionError, match="Mismatched elements"):
        np.testing.assert_allclose(predictions, y, rtol=1e-10)


@pytest.mark.parametrize("alpha", np.linspace(0.1, 2.0, 10))
def test_zero_features(alpha: np.float32):
    # given
    X = np.empty((5, 0))  # 5 samples, 0 features
    y = np.array([1, 2, 3, 4, 5])

    # when
    rr = RidgeRegression(1, regularization_coef=cast("float", alpha))
    rr.fit(X, y)
    predictions = rr.predict(X)
    expected_mean = np.mean(y)
    expected = np.full_like(y, expected_mean)

    # then
    np.testing.assert_allclose(predictions, expected, rtol=1e-10)


@pytest.mark.parametrize("alpha", np.linspace(0.1, 2.0, 10))
def test_negative_values(alpha: np.float32):
    # given
    X = np.array([[-2], [-1], [0], [1], [2]])
    y = np.array([-5, -3, -1, 1, 3])  # y = 2x - 1

    # when
    rr = RidgeRegression(2, regularization_coef=cast("float", alpha))
    rr.fit(X, y)
    predictions = rr.predict(X)
    sklearn_rr = SklearnRidge(alpha=alpha)
    sklearn_rr.fit(X, y)
    expected = sklearn_rr.predict(X)

    # then
    np.testing.assert_allclose(predictions, expected, rtol=1e-10)


def test_singular_matrix():
    """When features are linearly dependent, the matrix X^T X becomes singular
    and cannot be inverted. Standard OLS should fail in this case, while
    Ridge regression (with alpha > 0) can handle it.
    """
    # given
    alpha = 0.0
    # Create data where second column is exactly 2 times the first column
    X = np.array([[1, 2], [2, 4], [3, 6], [4, 8], [5, 10]])
    y = np.array([3, 5, 7, 9, 11])

    # when
    rr = RidgeRegression(3, regularization_coef=cast("float", alpha))

    # This should fail with standard OLS due to singular matrix
    # Expected to raise LinAlgError or similar numerical error
    # then
    with pytest.raises(np.linalg.LinAlgError, match="Singular matrix"):
        rr.fit(X, y)


@pytest.mark.parametrize("alpha", np.linspace(0.1, 2.0, 10))
def test_intercept_only_model(alpha: np.float32):
    # given
    X = np.zeros((10, 3))
    y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)

    # when
    rr = RidgeRegression(4, regularization_coef=cast("float", alpha))
    rr.fit(X, y)
    predictions = rr.predict(X)
    expected_mean = np.mean(y)
    expected = np.full_like(y, expected_mean)

    # then
    np.testing.assert_allclose(predictions, expected, rtol=1e-10)


def test_intercept_only_model__failing():
    # given
    alpha = 0.0
    X = np.zeros((10, 3))
    y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)

    # when
    rr = RidgeRegression(4, regularization_coef=cast("float", alpha))

    # then
    with pytest.raises(np.linalg.LinAlgError, match="Singular matrix"):
        rr.fit(X, y)


def test_known_analytical_solution():
    # given
    alpha = 0.5
    # Generate data from y = 3x + 2
    X = np.array([[1], [2], [3], [4], [5]])
    true_slope = 3
    true_intercept = 2
    y = true_slope * X.flatten() + true_intercept
    expected_theta = np.array([true_intercept, true_slope])

    # when
    rr = RidgeRegression(2, regularization_coef=cast("float", alpha))
    rr.fit(X, y)

    # then
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(rr._coef, expected_theta, rtol=1e-10)


def test_residuals_orthogonal_to_features():
    # given
    alpha = 0.0
    rng = np.random.default_rng(42)
    X = rng.normal(size=(20, 3))
    y = X @ np.array([1, 2, 3]) + rng.normal(size=20) * 0.1

    # when
    rr = RidgeRegression(4, regularization_coef=cast("float", alpha))
    rr.fit(X, y)
    predictions = rr.predict(X)
    residuals = y - predictions
    X_with_intercept = np.column_stack([np.ones(X.shape[0]), X])

    # then
    # Residuals should be orthogonal to each column of X_with_intercept
    for i in range(X_with_intercept.shape[1]):
        dot_product = np.dot(residuals, X_with_intercept[:, i])
        np.testing.assert_allclose(dot_product, 0, atol=1e-10)


def test_residuals_orthogonal_to_features__failing():
    # given
    alpha = 0.0
    rng = np.random.default_rng(42)
    X = rng.normal(size=(20, 3))
    y = X @ np.array([1, 2, 3]) + rng.normal(size=20) * 0.1

    # when
    rr = RidgeRegression(4, regularization_coef=cast("float", alpha))
    rr.fit(X, y)
    predictions = rr.predict(X)
    residuals = y - predictions
    X_with_intercept = np.column_stack([np.ones(X.shape[0]), X])

    # then
    # Residuals should be orthogonal to each column of X_with_intercept
    for i in range(X_with_intercept.shape[1]):
        dot_product = np.dot(residuals, X_with_intercept[:, i])
        np.testing.assert_allclose(dot_product, 0, atol=1e-10)


@pytest.mark.parametrize("alpha", np.linspace(0.0, 2.0, 10))
def test_linearity_property(alpha: np.float32):
    # given
    X = np.array([[1], [2], [3], [4]])
    y1 = np.array([2, 4, 6, 8])
    y2 = np.array([1, 3, 5, 7])

    # when
    rr1 = RidgeRegression(2, regularization_coef=cast("float", alpha))
    rr1.fit(X, y1)
    pred1 = rr1.predict(X)
    rr2 = RidgeRegression(2, regularization_coef=cast("float", alpha))
    rr2.fit(X, y2)
    pred2 = rr2.predict(X)
    rr_sum = RidgeRegression(2, regularization_coef=cast("float", alpha))
    rr_sum.fit(X, y1 + y2)
    pred_sum = rr_sum.predict(X)

    # then
    np.testing.assert_allclose(pred_sum, pred1 + pred2, rtol=1e-10)


@pytest.mark.parametrize("alpha", np.linspace(0.0, 2.0, 10))
def test_scale_invariance(alpha: np.float32):
    # given
    X = np.array([[1], [2], [3], [4]])
    y = np.array([2, 4, 6, 8])
    scale_factor = 3.5

    # when
    rr1 = RidgeRegression(2, regularization_coef=cast("float", alpha))
    rr1.fit(X, y)
    pred1 = rr1.predict(X)
    rr2 = RidgeRegression(2, regularization_coef=cast("float", alpha))
    rr2.fit(X, y * scale_factor)
    pred2 = rr2.predict(X)

    # then
    np.testing.assert_allclose(pred2, pred1 * scale_factor, rtol=1e-10)


@pytest.mark.parametrize("alpha", np.linspace(0.0, 2.0, 10))
def test_translation_invariance(alpha: np.float32):
    # given
    X = np.array([[1], [2], [3], [4]])
    y = np.array([2, 4, 6, 8])
    translation = 10.5

    # when
    rr1 = RidgeRegression(2, regularization_coef=cast("float", alpha))
    rr1.fit(X, y)
    pred1 = rr1.predict(X)
    rr2 = RidgeRegression(2, regularization_coef=cast("float", alpha))
    rr2.fit(X, y + translation)
    pred2 = rr2.predict(X)

    # then
    np.testing.assert_allclose(pred2, pred1 + translation, rtol=1e-10)


@pytest.mark.parametrize(
    ("n_samples", "n_features"), [(10, 1), (10, 2), (10, 5), (50, 3), (100, 10)]
)
def test_random_data_consistency(n_samples: int, n_features: int):
    # given
    alpha = 0.5
    rng = np.random.default_rng(42)
    X = rng.normal(size=(n_samples, n_features))
    y = rng.normal(size=(n_samples))

    # when
    rr = RidgeRegression(n_features + 1, regularization_coef=alpha)
    rr.fit(X, y)
    predictions = rr.predict(X)
    sklearn_rr = SklearnRidge(alpha=alpha)
    sklearn_rr.fit(X, y)
    expected = sklearn_rr.predict(X)

    # then
    np.testing.assert_allclose(predictions, expected, rtol=1e-10)
