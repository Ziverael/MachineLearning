import numpy as np

from machine_learning.model import Matrix


def add_interception_to_matrix(matrix: Matrix) -> Matrix:
    return np.column_stack([np.ones((matrix.shape[0], 1)), matrix])
