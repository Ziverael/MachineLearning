import numpy as np

from machine_learning.neural_network.two_layer_xor import TwoLayerXOR


def test_initialization():
    # when
    network = TwoLayerXOR()

    # ten
    assert hasattr(network, "W1")
    assert hasattr(network, "b1")
    assert hasattr(network, "W2")
    assert hasattr(network, "b2")
    assert network.W1 is not None
    assert network.b1 is not None
    assert network.W2 is not None
    assert network.b2 is not None


def test_weights_shape():
    # when
    network = TwoLayerXOR()

    # then
    assert network.W1.shape == (2, 2)
    assert network.b1.shape == (2,)
    assert network.W2.shape == (2, 1)
    assert network.b2.shape == (1,)


def test_heaviside_function():
    # given
    z = np.array([-2, -1, 0, 1, 2])
    expected = np.array([0, 0, 1, 1, 1])

    # when
    network = TwoLayerXOR()
    result = network.heaviside(z)

    # then
    assert network.heaviside(1.0) == 1
    assert network.heaviside(0.0) == 1
    assert network.heaviside(-1.0) == 0
    assert np.array_equal(result, expected)


def test_forward_pass_structure():
    # given
    X = np.array([[0, 0]])

    # when
    network = TwoLayerXOR()
    y, h = network.forward(X)

    # then
    assert y.shape == (1,)
    assert h.shape == (1, 2)
    assert y[0] in [0, 1]
    assert h[0, 0] in [0, 1]
    assert h[0, 1] in [0, 1]


def test_forward_batch():
    # given
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

    # when
    network = TwoLayerXOR()
    y, h = network.forward(X)

    # then
    assert y.shape == (4,)
    assert h.shape == (4, 2)
    assert all(val in [0, 1] for val in y)


def test_xor_truth_table():
    # given
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    expected_y = np.array([0, 1, 1, 0])

    # when
    network = TwoLayerXOR()
    predictions = network.predict(X)

    # then
    assert np.array_equal(predictions, expected_y), (
        f"XOR failed: got {predictions}, expected {expected_y}"
    )


def test_xor_individual_cases():
    # when
    network = TwoLayerXOR()

    # then
    assert network.predict(np.array([[0, 0]]))[0] == 0
    assert network.predict(np.array([[0, 1]]))[0] == 1
    assert network.predict(np.array([[1, 0]]))[0] == 1
    assert network.predict(np.array([[1, 1]]))[0] == 0


def test_verify_xor_method():
    # when
    network = TwoLayerXOR()

    # then
    assert network.verify_xor() is True


def test_hidden_layer_representations():
    # given
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

    # when
    network = TwoLayerXOR()
    _, h = network.forward(X)

    # then
    assert all(val in [0, 1] for val in h.flatten())
    assert not np.all(h == h[0])


def test_linear_separability_in_hidden_space():
    # given
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_true = np.array([0, 1, 1, 0])

    # when
    network = TwoLayerXOR()
    _, h = network.forward(X)
    h_class_0 = h[y_true == 0]  # (0,0) and (1,1)
    h_class_1 = h[y_true == 1]  # (0,1) and (1,0)

    # then
    for h0 in h_class_0:
        for h1 in h_class_1:
            assert not np.array_equal(h0, h1), (
                "Hidden representations should differ between classes"
            )


def test_predict_wrapper():
    # given
    X = np.array([[0, 1], [1, 1]])

    # when
    network = TwoLayerXOR()
    predictions = network.predict(X)

    # then
    assert predictions.shape == (2,)
    assert predictions[0] == 1
    assert predictions[1] == 0


def test_network_deterministic():
    # given
    X = np.array([[0, 1]])

    # when
    network = TwoLayerXOR()
    y1 = network.predict(X)
    y2 = network.predict(X)
    y3 = network.predict(X)

    # then
    assert np.array_equal(y1, y2)
    assert np.array_equal(y2, y3)


def test_weight_values_reasonable():
    # when
    network = TwoLayerXOR()

    # then
    assert np.all(np.abs(network.W1) < 10)
    assert np.all(np.abs(network.b1) < 10)
    assert np.all(np.abs(network.W2) < 10)
    assert np.all(np.abs(network.b2) < 10)


def test_multiple_instances_independence():
    # given
    X = np.array([[0, 1], [1, 0]])

    # when
    network1 = TwoLayerXOR()
    network2 = TwoLayerXOR()

    # then
    assert network1.verify_xor() is True
    assert network2.verify_xor() is True
    assert np.array_equal(network1.predict(X), network2.predict(X))
