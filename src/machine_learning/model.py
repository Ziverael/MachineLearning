from __future__ import annotations

import numpy as np
from jaxtyping import Float
from numpy.typing import NDArray


# We needs mathematical interpretation of vector which can be row vector or
# columnar vector
type RowVector = Float[NDArray[np.floating], "1 n"]  # noqa: F722
type ColumnVector = Float[NDArray[np.floating], "n 1"]  # noqa: F722
type Vector = RowVector | ColumnVector
type Matrix = NDArray[np.floating]
