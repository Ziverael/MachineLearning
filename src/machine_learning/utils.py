import numpy as np

from machine_learning.model import Matrix


def add_interception_to_matrix(matrix: Matrix) -> Matrix:
    return (
        np.ones((matrix.shape[0], 1))
        if matrix.shape[1] == 0
        else np.column_stack([np.ones(matrix.shape[0]), matrix])
    )
