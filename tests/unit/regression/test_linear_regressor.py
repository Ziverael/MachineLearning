import numpy as np
import pytest
from sklearn.linear_model import (  # type: ignore[import-untyped]
    LinearRegression as SklearnLinearRegression,
)

from machine_learning.regression.linear import (
    LinearRegressor as LinearRegression,
)


def test_simple_1d_case():
    # given
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([3, 5, 7, 9, 11])  # y = 2x + 1

    # when
    lr = LinearRegression(2)
    lr.fit(X, y)
    predictions = lr.predict(X)

    sklearn_lr = SklearnLinearRegression()
    sklearn_lr.fit(X, y)
    expected = sklearn_lr.predict(X)

    # then
    np.testing.assert_allclose(predictions, expected, rtol=1e-10)


def test_multidimensional_case():
    # given
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
    y = np.array([14, 32, 50, 68])  # y = x1 + 2*x2 + 3*x3 + 2

    # when
    lr = LinearRegression(4)
    lr.fit(X, y)
    predictions = lr.predict(X)

    sklearn_lr = SklearnLinearRegression()
    sklearn_lr.fit(X, y)
    expected = sklearn_lr.predict(X)

    # then
    np.testing.assert_allclose(predictions, expected, rtol=1e-10)


def test_prediction_on_new_data():
    # given
    X_train = np.array([[1], [2], [3], [4]])
    y_train = np.array([2, 4, 6, 8])  # y = 2x
    X_test = np.array([[5], [6], [0]])

    # when
    lr = LinearRegression(2)
    lr.fit(X_train, y_train)
    predictions = lr.predict(X_test)

    sklearn_lr = SklearnLinearRegression()
    sklearn_lr.fit(X_train, y_train)
    expected = sklearn_lr.predict(X_test)

    # then
    np.testing.assert_allclose(predictions, expected, rtol=1e-10)


def test_single_data_point():
    # given
    X = np.array([[1, 2]])
    y = np.array([5])

    # when
    lr = LinearRegression(3)
    lr.fit(X, y)
    predictions = lr.predict(X)

    # then
    # Should predict the training point exactly
    np.testing.assert_allclose(predictions, y, rtol=1e-10)


def test_perfect_fit():
    # given
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([3, 5, 7, 9, 11])  # Perfect line: y = 2x + 1

    # when
    lr = LinearRegression(2)
    lr.fit(X, y)
    predictions = lr.predict(X)

    # then
    # Should predict training data exactly
    np.testing.assert_allclose(predictions, y, rtol=1e-10)


def test_zero_features():
    # given
    X = np.empty((5, 0))  # 5 samples, 0 features
    y = np.array([1, 2, 3, 4, 5])

    # when
    lr = LinearRegression(1)
    lr.fit(X, y)
    predictions = lr.predict(X)
    expected_mean = np.mean(y)
    expected = np.full_like(y, expected_mean)

    # then
    np.testing.assert_allclose(predictions, expected, rtol=1e-10)


def test_negative_values():
    # given
    X = np.array([[-2], [-1], [0], [1], [2]])
    y = np.array([-5, -3, -1, 1, 3])  # y = 2x - 1

    # when
    lr = LinearRegression(2)
    lr.fit(X, y)
    predictions = lr.predict(X)
    sklearn_lr = SklearnLinearRegression()
    sklearn_lr.fit(X, y)
    expected = sklearn_lr.predict(X)

    # then
    np.testing.assert_allclose(predictions, expected, rtol=1e-10)


def test_singular_matrix():
    # given
    X = np.array([[1, 2], [2, 4], [3, 6], [4, 8], [5, 10]])
    y = np.array([3, 5, 7, 9, 11], dtype=np.float32)

    # when
    lr = LinearRegression(3)
    lr.fit(X, y)
    predictions = lr.predict(X)

    # then
    np.testing.assert_allclose(predictions, y, rtol=1e-10)


def test_intercept_only_model():
    # given
    X = np.zeros((10, 3))  # All features are zero
    y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)

    # when
    lr = LinearRegression(4)
    lr.fit(X, y)
    predictions = lr.predict(X)
    expected_mean = np.mean(y)
    expected = np.full_like(y, expected_mean)

    # then
    np.testing.assert_allclose(predictions, expected, rtol=1e-10)


def test_known_analytical_solution():
    # given
    X = np.array([[1], [2], [3], [4], [5]])
    true_slope = 3
    true_intercept = 2
    y = true_slope * X.flatten() + true_intercept

    # when
    lr = LinearRegression(2)
    lr.fit(X, y)
    expected_theta = np.array([true_intercept, true_slope])

    # then
    np.testing.assert_allclose(lr._coef, expected_theta, rtol=1e-10)


def test_residuals_orthogonal_to_features():
    # given
    rng = np.random.default_rng(42)
    X = rng.normal(size=(20, 3))
    y = X @ np.array([1, 2, 3]) + rng.normal(size=20) * 0.1

    # when
    lr = LinearRegression(4)
    lr.fit(X, y)
    predictions = lr.predict(X)
    residuals = y - predictions
    X_with_intercept = np.column_stack([np.ones(X.shape[0]), X])

    # then
    for i in range(X_with_intercept.shape[1]):
        dot_product = np.dot(residuals, X_with_intercept[:, i])
        np.testing.assert_allclose(dot_product, 0, atol=1e-10)


def test_linearity_property():
    # given
    X = np.array([[1], [2], [3], [4]])
    y1 = np.array([2, 4, 6, 8])
    y2 = np.array([1, 3, 5, 7])

    # when
    lr1 = LinearRegression(2)
    lr1.fit(X, y1)
    pred1 = lr1.predict(X)
    lr2 = LinearRegression(2)
    lr2.fit(X, y2)
    pred2 = lr2.predict(X)
    lr_sum = LinearRegression(2)
    lr_sum.fit(X, y1 + y2)
    pred_sum = lr_sum.predict(X)

    # then
    np.testing.assert_allclose(pred_sum, pred1 + pred2, rtol=1e-10)


def test_scale_invariance():
    # given
    X = np.array([[1], [2], [3], [4]])
    y = np.array([2, 4, 6, 8])
    scale_factor = 3.5

    # when
    lr1 = LinearRegression(2)
    lr1.fit(X, y)
    pred1 = lr1.predict(X)
    lr2 = LinearRegression(2)
    lr2.fit(X, y * scale_factor)
    pred2 = lr2.predict(X)

    # then
    np.testing.assert_allclose(pred2, pred1 * scale_factor, rtol=1e-10)


def test_translation_invariance():
    # given
    X = np.array([[1], [2], [3], [4]])
    y = np.array([2, 4, 6, 8])
    translation = 10.5

    # when
    lr1 = LinearRegression(2)
    lr1.fit(X, y)
    pred1 = lr1.predict(X)
    lr2 = LinearRegression(2)
    lr2.fit(X, y + translation)
    pred2 = lr2.predict(X)

    # then
    np.testing.assert_allclose(pred2, pred1 + translation, rtol=1e-10)


@pytest.mark.parametrize(
    ("n_samples", "n_features"), [(10, 1), (10, 2), (10, 5), (50, 3), (100, 10)]
)
def test_random_data_consistency(n_samples: int, n_features: int):
    # given
    rng = np.random.default_rng(42)
    X = rng.normal(size=(n_samples, n_features))
    y = rng.normal(size=(n_samples))

    # when
    lr = LinearRegression(n_features + 1)
    lr.fit(X, y)
    predictions = lr.predict(X)
    sklearn_lr = SklearnLinearRegression()
    sklearn_lr.fit(X, y)
    expected = sklearn_lr.predict(X)

    # then
    np.testing.assert_allclose(predictions, expected, rtol=1e-10)
