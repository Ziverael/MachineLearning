from itertools import product

import numpy as np
import pytest

from machine_learning.neural_network.mcculloch_pitts import McCullochPittsNeuron


def test_heaviside_scalar():
    # given
    n_inputs = 2
    threshold = 1

    # when
    neuron = McCullochPittsNeuron(n_inputs, threshold)

    # then
    assert neuron.heaviside(1.0) == 1
    assert neuron.heaviside(0.0) == 1
    assert neuron.heaviside(-1.0) == 0
    assert neuron.heaviside(0.5) == 1
    assert neuron.heaviside(-0.5) == 0


def test_heaviside_array():
    # given
    n_inputs = 2
    threshold = 1
    neuron = McCullochPittsNeuron(n_inputs, threshold)
    expected = np.array([0, 0, 1, 1, 1])
    z = np.array([-2, -1, 0, 1, 2])

    # when
    result = neuron.heaviside(z)

    # then
    assert np.array_equal(result, expected)


@pytest.mark.parametrize(
    ("inputs", "expected"), [([0, 0], 0), ([1, 0], 0), ([1, 1], 1)]
)
def test_forward_single_sample(inputs: list[int], expected: int):
    # given
    n_inputs = 2
    threshold = 1.5
    neuron = McCullochPittsNeuron(n_inputs, threshold)

    # when
    results = neuron.forward(np.array(inputs))

    # then
    assert results == expected


def test_forward_batch():
    # given
    n_inputs = 2
    threshold = 2
    neuron = McCullochPittsNeuron(n_inputs, threshold)
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    expected = np.array([0, 0, 0, 1])

    # when
    result = neuron.forward(X)

    # then
    assert np.array_equal(result, expected)


def test_and_gate():
    # given
    n_inputs = 2
    threshold = 2

    # when
    neuron = McCullochPittsNeuron(n_inputs, threshold)

    # then
    assert neuron.forward(np.array([0, 0])) == 0
    assert neuron.forward(np.array([0, 1])) == 0
    assert neuron.forward(np.array([1, 0])) == 0
    assert neuron.forward(np.array([1, 1])) == 1


def test_or_gate():
    # given
    n_inputs = 2
    threshold = 1

    # when
    neuron = McCullochPittsNeuron(n_inputs, threshold)

    # then
    assert neuron.forward(np.array([0, 0])) == 0
    assert neuron.forward(np.array([0, 1])) == 1
    assert neuron.forward(np.array([1, 0])) == 1
    assert neuron.forward(np.array([1, 1])) == 1


def test_truth_table_structure():
    # given
    n_inputs = 2
    threshold = 1
    neuron = McCullochPittsNeuron(n_inputs, threshold)

    # when
    truth_table = neuron.truth_table()

    # then
    assert isinstance(truth_table, dict)
    assert "inputs" in truth_table
    assert "outputs" in truth_table
    inputs = truth_table["inputs"]
    outputs = truth_table["outputs"]
    assert len(inputs) == 4
    assert len(outputs) == 4


def test_truth_table_and():
    # given
    n_inputs = 2
    threshold = 2
    neuron = McCullochPittsNeuron(n_inputs, threshold)
    expected_outputs = [0, 0, 0, 1]  # AND truth table

    # when
    truth_table = neuron.truth_table()
    inputs = truth_table["inputs"]
    outputs = truth_table["outputs"]

    # then
    for i, (inp, out) in enumerate(zip(inputs, outputs, strict=True)):
        assert len(inp) == 2
        assert out == expected_outputs[i]


def test_truth_table_or():
    # given
    n_inputs = 2
    threshold = 1
    neuron = McCullochPittsNeuron(n_inputs, threshold)
    expected_outputs = [0, 1, 1, 1]  # OR truth table

    # when
    truth_table = neuron.truth_table()
    outputs = truth_table["outputs"]

    # then
    assert outputs == expected_outputs


def test_three_input_neuron():
    # given
    n_inputs = 3
    threshold = 2

    # when
    neuron = McCullochPittsNeuron(n_inputs, threshold)

    # then
    assert neuron.forward(np.array([0, 0, 0])) == 0  # sum=0
    assert neuron.forward(np.array([1, 0, 0])) == 0  # sum=1
    assert neuron.forward(np.array([1, 1, 0])) == 1  # sum=2
    assert neuron.forward(np.array([1, 1, 1])) == 1  # sum=3


def test_truth_table_three_inputs():
    # given
    n_inputs = 3
    threshold = 2
    neuron = McCullochPittsNeuron(n_inputs, threshold)
    expected_inputs = list(product([0, 1], repeat=3))

    # when
    truth_table = neuron.truth_table()
    inputs = truth_table["inputs"]
    outputs = truth_table["outputs"]

    # then
    assert len(inputs) == 8
    assert len(outputs) == 8
    for expected in expected_inputs:
        assert expected in inputs


def test_initialization():
    # given
    n_inputs = 5
    threshold = 3.5

    # when
    neuron = McCullochPittsNeuron(n_inputs, threshold)

    # then
    assert neuron.n_inputs == 5
    assert neuron.threshold == 3.5
