from numpy.typing import NDArray
from typing import Optional

import numpy as np


# Array search helpers
def first_less_than(arr: NDArray, k: float) -> int:
    mask = arr < k
    idx = np.argmax(mask)
    return int(idx) if mask[idx] else int(arr.size)


def last_less_than(arr: NDArray, k: float) -> int:
    return int(np.argmin(arr < k)) - 1


def first_more_or_equal_than(arr: NDArray, k: float) -> int:
    mask = arr >= k
    idx = np.argmax(mask)
    return int(idx) if mask[idx] else int(arr.size)


def first_index_above_min(arr: NDArray, error: float = 1e-5) -> int:
    threshold = np.min(arr) + error
    for i in range(arr.size - 1, -1, -1):
        if arr[i] > threshold:
            return i
    return 0


# Geometry helpers
def area_under_curve(
    points: NDArray, arg_sort: bool = False, complement: Optional[float] = 1
) -> float:
    x, y = points[0], points[1]
    if arg_sort:
        indices = np.argsort(x)
        x, y = x[indices], y[indices]

    area = 0.5 * np.sum(np.diff(x) * (y[1:] + y[:-1]))
    if complement:
        area += (complement - x[-1]) * y[-1]
    return area
