"""Tests for Perceptron implementation."""

import numpy as np

from machine_learning.neural_network.perceptron import Perceptron


def test_initialization():
    # given
    n_inputs = 3
    learning_rate = 0.1
    random_state = 42

    # when
    perceptron = Perceptron(n_inputs, learning_rate, random_state)

    # then
    assert perceptron.n_inputs == 3
    assert perceptron.learning_rate == 0.1
    assert hasattr(perceptron, "weights")
    assert hasattr(perceptron, "bias")


def test_weights_shape():
    # given
    n_inputs = 5
    random_state = 42

    # when
    perceptron = Perceptron(n_inputs=n_inputs, random_state=random_state)

    # then
    assert perceptron.weights.shape == (5,)
    assert isinstance(perceptron.bias, (int, float, np.number))


def test_activation_function():
    # given
    n_inputs = 2

    # when
    perceptron = Perceptron(n_inputs=n_inputs)

    # then
    assert perceptron.activation(1.5) in [0, 1]
    assert perceptron.activation(0.0) in [0, 1]
    assert perceptron.activation(-1.5) in [0, 1]


def test_forward_single_sample():
    # given
    perceptron = Perceptron(n_inputs=2, random_state=42)
    perceptron.weights = np.array([0.5, 0.5])
    perceptron.bias = 0.0
    x = np.array([1.0, 1.0])

    # when
    output = perceptron.forward(x.reshape(1, -1))

    # then
    assert output.shape == (1,)
    assert output[0] in [0, 1]


def test_forward_batch():
    # given
    perceptron = Perceptron(n_inputs=2, random_state=42)
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

    # when
    output = perceptron.forward(X)

    # then
    assert output.shape == (4,)
    assert all(val in [0, 1] for val in output)


def test_fit_and_gate():
    # given
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 0, 0, 1])  # AND truth table
    perceptron = Perceptron(n_inputs=2, learning_rate=0.1, random_state=42)

    # when
    perceptron.fit(X, y, n_epochs=100)
    predictions = perceptron.predict(X)

    # then
    assert hasattr(perceptron, "errors_")
    assert len(perceptron.errors_) <= 100
    assert np.array_equal(predictions, y)


def test_fit_or_gate():
    # given
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 1])  # OR truth table
    perceptron = Perceptron(n_inputs=2, learning_rate=0.1, random_state=42)

    # when
    perceptron.fit(X, y, n_epochs=100)
    predictions = perceptron.predict(X)

    # then
    assert np.array_equal(predictions, y)


def test_predict():
    # given
    X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_train = np.array([0, 0, 0, 1])
    perceptron = Perceptron(n_inputs=2, learning_rate=0.1, random_state=42)
    X_test = np.array([[0, 0], [1, 1]])

    # when
    perceptron.fit(X_train, y_train, n_epochs=100)
    predictions = perceptron.predict(X_test)

    # then
    assert predictions.shape == (2,)
    assert predictions[0] == 0
    assert predictions[1] == 1


def test_score():
    # given
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 0, 0, 1])
    perceptron = Perceptron(n_inputs=2, learning_rate=0.1, random_state=42)

    # when
    perceptron.fit(X, y, n_epochs=100)
    accuracy = perceptron.score(X, y)

    # then
    assert 0.0 <= accuracy <= 1.0
    assert accuracy == 1.0


def test_convergence():
    # given
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 1])
    perceptron = Perceptron(n_inputs=2, learning_rate=0.1, random_state=42)

    # when
    perceptron.fit(X, y, n_epochs=100)

    # then
    assert perceptron.errors_[-1] == 0


def test_errors_tracking():
    # given
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 0, 0, 1])
    perceptron = Perceptron(n_inputs=2, learning_rate=0.1, random_state=42)

    # when
    perceptron.fit(X, y, n_epochs=50)

    # then
    assert hasattr(perceptron, "errors_")
    assert len(perceptron.errors_) > 0
    assert len(perceptron.errors_) <= 50
    if len(perceptron.errors_) > 1:
        assert perceptron.errors_[-1] <= perceptron.errors_[0]


def test_learning_rate_effect():
    # given
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 1])
    p1 = Perceptron(n_inputs=2, learning_rate=0.01, random_state=42)
    p2 = Perceptron(n_inputs=2, learning_rate=0.5, random_state=42)

    # when
    p1.fit(X, y, n_epochs=100)
    p2.fit(X, y, n_epochs=100)

    # then
    assert p1.score(X, y) >= 0.75
    assert p2.score(X, y) >= 0.75


def test_random_state_reproducibility():
    # given
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 1])
    p1 = Perceptron(n_inputs=2, learning_rate=0.1, random_state=42)
    p2 = Perceptron(n_inputs=2, learning_rate=0.1, random_state=42)

    # when
    p1.fit(X, y, n_epochs=50)
    p2.fit(X, y, n_epochs=50)

    # then
    assert np.allclose(p1.weights, p2.weights)
    assert np.isclose(p1.bias, p2.bias)


def test_higher_dimensional_input():
    # given
    X = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 1, 1]])
    y = np.array([0, 0, 0, 1])
    perceptron = Perceptron(n_inputs=3, learning_rate=0.1, random_state=42)

    # when
    perceptron.fit(X, y, n_epochs=100)
    predictions = perceptron.predict(X)
    accuracy = np.mean(predictions == y)

    # then
    assert accuracy >= 0.75
