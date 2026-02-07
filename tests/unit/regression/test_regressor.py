from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError

from machine_learning.regression import _regressor


@pytest.fixture
def dummy_linear_regressor():
    class DummyLinReg(_regressor.AbstractLinearRegressor):
        def fit(self):
            self._coef = np.array([[10.0], [3.4]])

    return DummyLinReg


@pytest.mark.parametrize("shape", [1, 2, 3, 10, 100])
def test_linear_init(dummy_linear_regressor, shape: int):
    # given / when
    dummy_linear_regressor(shape)

    # then
    assert True


@pytest.mark.parametrize("shape", [-1, 0, 2.2, "text", None])
def test_linear_init__fails(dummy_linear_regressor, shape: Any):
    # given / when / then
    with pytest.raises(ValidationError, match="Input should be"):
        dummy_linear_regressor(shape)


def test_coef(dummy_linear_regressor):
    # given
    reg = dummy_linear_regressor(2)

    # when
    results = reg._coef

    # then
    assert results.shape == (2, 1)


def test_coef__after_fit(dummy_linear_regressor):
    # given
    reg = dummy_linear_regressor(2)

    # when
    reg.fit()
    results = reg._coef

    # then
    assert (results == np.array([[10.0], [3.4]])).all()


def test_predict(dummy_linear_regressor):
    # given
    reg = dummy_linear_regressor(2)
    x_mat = np.array([[2.0]])

    # when
    reg.fit()
    results = reg.predict(x_mat)

    # then
    assert (results == np.array([[16.8]])).all()
