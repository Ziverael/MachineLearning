import numpy as np

from machine_learning.neural_network.mlp_simple import SimpleMLP


def test_initialization():
    # when
    mlp = SimpleMLP(
        n_input=2, n_hidden=4, n_output=1, learning_rate=0.1, random_state=42
    )

    # then
    assert mlp.n_input == 2
    assert mlp.n_hidden == 4
    assert mlp.n_output == 1
    assert mlp.learning_rate == 0.1
    assert hasattr(mlp, "W1")
    assert hasattr(mlp, "b1")
    assert hasattr(mlp, "W2")
    assert hasattr(mlp, "b2")


def test_weights_shapes():
    # when
    mlp = SimpleMLP(n_input=3, n_hidden=5, n_output=2, random_state=42)

    # then
    assert mlp.W1.shape == (3, 5)
    assert mlp.b1.shape == (5,)
    assert mlp.W2.shape == (5, 2)
    assert mlp.b2.shape == (2,)


def test_sigmoid_function():
    # given
    z = np.array([0, 1, -1])

    # when
    mlp = SimpleMLP(n_input=2, n_hidden=2, n_output=1)
    result = mlp.sigmoid(z)

    # then
    assert np.isclose(mlp.sigmoid(0), 0.5)
    assert mlp.sigmoid(100) > 0.99
    assert mlp.sigmoid(-100) < 0.01  # Large negative
    assert result.shape == (3,)
    assert np.all((result > 0) & (result < 1))


def test_sigmoid_derivative():
    # given
    a = np.array([0.5, 0.8, 0.2])
    expected = a * (1 - a)

    # when
    mlp = SimpleMLP(n_input=2, n_hidden=2, n_output=1)
    result = mlp.sigmoid_derivative(a)

    # then
    assert np.allclose(result, expected)


def test_forward_pass_shapes():
    # given
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

    # when
    mlp = SimpleMLP(n_input=2, n_hidden=3, n_output=1, random_state=42)
    output = mlp.forward(X)

    # then
    assert output.shape == (4, 1)
    assert np.all((output > 0) & (output < 1))


def test_forward_pass_stores_activations():
    # given
    X = np.array([[1, 0]])

    # when
    mlp = SimpleMLP(n_input=2, n_hidden=3, n_output=1, random_state=42)
    mlp.forward(X)

    # then
    assert hasattr(mlp, "z1")
    assert hasattr(mlp, "a1")
    assert hasattr(mlp, "z2")
    assert hasattr(mlp, "a2")
    assert mlp.z1.shape == (1, 3)
    assert mlp.a1.shape == (1, 3)
    assert mlp.z2.shape == (1, 1)
    assert mlp.a2.shape == (1, 1)


def test_compute_loss():
    # given
    y_true = np.array([[0], [1], [1], [0]])
    y_pred = np.array([[0.1], [0.9], [0.8], [0.2]])
    y_perfect = np.array([[0.01], [0.99], [0.99], [0.01]])

    # when
    mlp = SimpleMLP(n_input=2, n_hidden=2, n_output=1, random_state=42)
    loss = mlp.compute_loss(y_true, y_pred)
    loss_perfect = mlp.compute_loss(y_true, y_perfect)

    # then
    assert loss >= 0
    assert loss_perfect < loss


def test_backward_pass_returns_gradients():
    # given
    X = np.array([[0, 1], [1, 0]])
    y = np.array([[1], [0]])

    # when
    mlp = SimpleMLP(n_input=2, n_hidden=3, n_output=1, random_state=42)
    mlp.forward(X)
    gradients = mlp.backward(X, y)

    # then
    assert "dW1" in gradients
    assert "db1" in gradients
    assert "dW2" in gradients
    assert "db2" in gradients
    assert gradients["dW1"].shape == mlp.W1.shape
    assert gradients["db1"].shape == mlp.b1.shape
    assert gradients["dW2"].shape == mlp.W2.shape
    assert gradients["db2"].shape == mlp.b2.shape


def test_fit_xor_convergence():
    # given
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    # when
    mlp = SimpleMLP(
        n_input=2, n_hidden=4, n_output=1, learning_rate=0.5, random_state=42
    )
    mlp.fit(X, y, n_epochs=2000, verbose=False)
    predictions = mlp.predict(X)
    binary_pred = (predictions > 0.5).astype(int)
    accuracy = np.mean(binary_pred == y)

    # then
    assert accuracy >= 0.75, f"XOR accuracy too low: {accuracy}"


def test_loss_decreases_during_training():
    # given
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    # when
    mlp = SimpleMLP(
        n_input=2, n_hidden=4, n_output=1, learning_rate=0.5, random_state=42
    )
    mlp.fit(X, y, n_epochs=500, verbose=False)

    # then
    assert hasattr(mlp, "loss_history_")
    assert len(mlp.loss_history_) > 0
    assert mlp.loss_history_[-1] < mlp.loss_history_[0]


def test_predict_method():
    # given
    X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_train = np.array([[0], [1], [1], [0]])
    X_test = np.array([[0, 0], [1, 1]])

    # when
    mlp = SimpleMLP(
        n_input=2, n_hidden=4, n_output=1, learning_rate=0.5, random_state=42
    )
    mlp.fit(X_train, y_train, n_epochs=1000, verbose=False)
    predictions = mlp.predict(X_test)
    binary_pred = (predictions > 0.5).astype(int)

    # then
    assert predictions.shape == (2, 1)
    assert binary_pred[0, 0] == 0
    assert binary_pred[1, 0] == 0


def test_learning_rate_effect():
    # given
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])
    mlp1 = SimpleMLP(
        n_input=2, n_hidden=4, n_output=1, learning_rate=0.1, random_state=42
    )
    mlp2 = SimpleMLP(
        n_input=2, n_hidden=4, n_output=1, learning_rate=1.0, random_state=42
    )

    # when
    mlp1.fit(X, y, n_epochs=500, verbose=False)
    mlp2.fit(X, y, n_epochs=500, verbose=False)

    # then
    assert mlp1.loss_history_[-1] < mlp1.loss_history_[0]
    assert mlp2.loss_history_[-1] < mlp2.loss_history_[0]


def test_random_state_reproducibility():
    # given
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])
    mlp1 = SimpleMLP(
        n_input=2, n_hidden=4, n_output=1, learning_rate=0.5, random_state=42
    )
    mlp2 = SimpleMLP(
        n_input=2, n_hidden=4, n_output=1, learning_rate=0.5, random_state=42
    )

    # when
    mlp1.fit(X, y, n_epochs=100, verbose=False)
    mlp2.fit(X, y, n_epochs=100, verbose=False)

    # then
    assert np.allclose(mlp1.W1, mlp2.W1)
    assert np.allclose(mlp1.W2, mlp2.W2)


def test_hidden_layer_size_effect():
    # given
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])
    mlp_small = SimpleMLP(
        n_input=2, n_hidden=2, n_output=1, learning_rate=0.5, random_state=42
    )
    mlp_large = SimpleMLP(
        n_input=2, n_hidden=8, n_output=1, learning_rate=0.5, random_state=43
    )

    # when
    mlp_small.fit(X, y, n_epochs=1000, verbose=False)
    mlp_large.fit(X, y, n_epochs=1000, verbose=False)
    pred_small = (mlp_small.predict(X) > 0.5).astype(int)
    pred_large = (mlp_large.predict(X) > 0.5).astype(int)
    acc_small = np.mean(pred_small == y)
    acc_large = np.mean(pred_large == y)

    # then
    assert max(acc_small, acc_large) >= 0.75


def test_multi_output():
    # given
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array(
        [
            [0, 0],  # AND=0, OR=0
            [0, 1],  # AND=0, OR=1
            [0, 1],  # AND=0, OR=1
            [1, 1],  # AND=1, OR=1
        ]
    )
    mlp = SimpleMLP(
        n_input=2, n_hidden=4, n_output=2, learning_rate=0.5, random_state=42
    )

    # when
    mlp.fit(X, y, n_epochs=1000, verbose=False)
    predictions = mlp.predict(X)
    binary_pred = (predictions > 0.5).astype(int)
    accuracy = np.mean(binary_pred == y)

    # then
    assert predictions.shape == (4, 2)
    assert accuracy >= 0.6


def test_loss_history_length():
    # given
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])
    n_epochs = 100
    mlp = SimpleMLP(n_input=2, n_hidden=4, n_output=1, random_state=42)

    # when
    mlp.fit(X, y, n_epochs=n_epochs, verbose=False)

    # then
    assert len(mlp.loss_history_) == n_epochs


def test_weights_update_during_training():
    # given
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])
    mlp = SimpleMLP(n_input=2, n_hidden=4, n_output=1, random_state=42)
    W1_initial = mlp.W1.copy()
    W2_initial = mlp.W2.copy()

    # when
    mlp.fit(X, y, n_epochs=10, verbose=False)

    # then
    assert not np.allclose(mlp.W1, W1_initial)
    assert not np.allclose(mlp.W2, W2_initial)


def test_single_sample_training():
    # given
    X = np.array([[0.5, 0.5]])
    y = np.array([[1]])
    mlp = SimpleMLP(n_input=2, n_hidden=3, n_output=1, random_state=42)

    # when
    mlp.fit(X, y, n_epochs=10, verbose=False)
    prediction = mlp.predict(X)

    # then
    assert prediction.shape == (1, 1)
