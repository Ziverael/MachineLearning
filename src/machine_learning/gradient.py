import numpy as np
from pydantic import NonNegativeFloat

from machine_learning.model import Matrix, Vector


def compute_gradient(
    features_matrix: Matrix, target_values: Vector, theta: Vector
) -> Vector:
    predictions = features_matrix @ theta
    m = len(target_values)
    return (1 / m) * features_matrix.T @ (predictions - target_values)


def compute_gradient_ridge(
    features_matrix: Matrix,
    target_values: Vector,
    theta: Vector,
    alpha: NonNegativeFloat,
) -> Vector:
    gradient = compute_gradient(features_matrix, target_values, theta)
    m = len(target_values)
    ridge_term = np.concatenate([[0], (alpha / m) * theta[1:]])
    return gradient + ridge_term
