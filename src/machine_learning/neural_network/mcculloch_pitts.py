"""McCulloch-Pitts Neuron implementation.

This module implements the McCulloch-Pitts neuron model (1943),
the first mathematical model of an artificial neuron.
"""

from collections import defaultdict
from functools import cached_property
from itertools import product
from typing import Any

import numpy as np

from machine_learning.model import Vector


class McCullochPittsNeuron:
    def __init__(self, n_inputs: int, threshold: float) -> None:
        self.n_inputs = n_inputs
        self.threshold = threshold

    def heaviside(self, z: float | Vector) -> int | Vector:
        if isinstance(z, np.ndarray):
            return (z >= 0).astype(np.int8)
        return int(z >= 0)

    def forward(self, x: Vector) -> int | Vector:
        return self.heaviside(x.sum(axis=-1) - self.threshold)

    @cached_property
    def _truth_table(self) -> dict[str, list[Any]]:
        table: dict[str, list[Any]] = defaultdict(list)
        for comb in product([0, 1], repeat=self.n_inputs):
            table["inputs"].append(comb)
            table["outputs"].append(self.forward(np.array(comb)))
        return table

    def truth_table(self) -> dict[str, list[Any]]:
        return self._truth_table
