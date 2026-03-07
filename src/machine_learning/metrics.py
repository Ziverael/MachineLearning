from functools import partial

import numpy as np

from machine_learning.model import LossFunction, Matrix, Vector


def mse(exact: Vector, predicted: Vector) -> np.floating:
    m = len(exact)
    return (1 / (2 * m)) * np.sum((predicted - exact) ** 2)


def loss(
    features_matrix: Matrix,
    target_values: Vector,
    theta: Vector,
    loss_func: LossFunction,
) -> np.floating:
    predictions = features_matrix @ theta
    return loss_func(target_values, predictions)


mse_loss = partial(loss, loss_func=mse)
