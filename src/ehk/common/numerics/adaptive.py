from numpy.typing import NDArray
from typing import Optional, Tuple, Callable, TypeVar, List, Literal, cast, Any
from scipy.interpolate import interp1d
from collections import OrderedDict, deque
import numpy as np

from .kde import gaussian_kernel, compute_kde_density, compute_weighted_stats


# Moving statistics
def moving_average(
    data: NDArray,
    window_size: int,
    pad_mode: str = "edge",
    convolve_mode: Literal["valid", "same", "full"] = "valid",
) -> NDArray:
    if window_size < 2:
        return data
    pad_width = window_size // 2
    pad_data = np.pad(data, pad_width, mode=cast(Any, pad_mode))
    window = np.ones(window_size) / window_size
    return np.convolve(pad_data, window, convolve_mode)


def adaptive_moving_stats(
    x: NDArray,
    y: NDArray,
    h0: float,
    alpha: float = 0.5,
    g: float = 1.0,
    min: Optional[float] = None,
    max: Optional[float] = None,
    density_x: Optional[NDArray] = None,
    result_x: Optional[NDArray] = None,
    density_estimation_point: Optional[int] = 20,
    result_point: Optional[int] = 100,
    epsilon: float = 1e-14,
    edge_padding: float = 0.0,
    edge_method: str = "reflect",
    min_bandwidth: Optional[float] = None,
    max_bandwidth: Optional[float] = None,
) -> Tuple[NDArray, NDArray, NDArray]:
    """
    Compute moving statistics with KDE-adaptive bandwidths.

    h(x) = h0 * (ρ(x) / g)^(-α)
    """
    min_val = min if min is not None else x.min()
    max_val = max if max is not None else x.max()

    # Reflect samples at the domain boundaries.
    if edge_padding > 0:
        range_size = max_val - min_val
        padding = range_size * edge_padding
        min_padded, max_padded = min_val - padding, max_val + padding

        if edge_method == "reflect":
            left_mask = x < (min_val + padding)
            right_mask = x > (max_val - padding)
            x_extended = np.concatenate(
                [2 * min_val - x[left_mask], x, 2 * max_val - x[right_mask]]
            )
            y_extended = np.concatenate([y[left_mask], y, y[right_mask]])
        else:
            x_extended, y_extended = x, y
    else:
        min_padded, max_padded = min_val, max_val
        x_extended, y_extended = x, y

    # Density evaluation points.
    if density_x is None:
        density_x_pts = (
            x_extended
            if density_estimation_point is None
            else np.linspace(
                min_padded, max_padded, density_estimation_point, dtype=float
            )
        )
    else:
        density_x_pts = density_x

    density = compute_kde_density(density_x_pts, x_extended, h0, epsilon)

    # Output evaluation points.
    if result_x is None:
        result_x_pts = (
            x
            if result_point is None
            else np.linspace(min_val, max_val, result_point, dtype=float)
        )
    else:
        result_x_pts = result_x

    # Interpolate the density and derive an adaptive bandwidth.
    fill_value: Any = (
        "extrapolate"
        if edge_method in ["extrapolate", "none"]
        else (density[0], density[-1])
    )
    density_interp = interp1d(
        density_x_pts, density, fill_value=fill_value, bounds_error=False
    )
    density_resampled = np.maximum(density_interp(result_x_pts), epsilon)

    h_t = h0 * (density_resampled / g) ** (-alpha)
    if min_bandwidth is not None:
        h_t = np.maximum(h_t, min_bandwidth)
    if max_bandwidth is not None:
        h_t = np.minimum(h_t, max_bandwidth)

    means, variances = compute_weighted_stats(
        result_x_pts, x_extended, y_extended, h_t, epsilon
    )

    return result_x_pts, means, variances


# Adaptive sampling

T = TypeVar("T")


def adaptive_discrete_sampling(
    f: Callable[[float], T],
    error_threshold: float,
    t_start: float,
    t_end: float,
    min_interval: float = 1,
    max_interval: float | None = None,
    int_midpoint: bool = False,
    err_func: Callable[[T, T, T, float], float] | None = None,
) -> Tuple[List[float], List[T]]:
    if t_start >= t_end:
        raise ValueError("t_start must be less than t_end")
    if min_interval <= 0:
        raise ValueError("min_interval must be positive")
    if int_midpoint and min_interval < 1:
        raise ValueError("min_interval must be at least 1 when int_midpoint is True")

    samples = OrderedDict({t_start: f(t_start), t_end: f(t_end)})
    queue: deque[Tuple[float, float]] = deque([(t_start, t_end)])

    while queue:
        t_left, t_right = queue.popleft()
        if t_right - t_left <= min_interval:
            continue

        force_sample = max_interval is not None and (t_right - t_left) > max_interval
        t_mid: float
        if force_sample:
            t_mid = t_left + max_interval  # type: ignore
        elif int_midpoint:
            t_mid = round((t_left + t_right) / 2)
        else:
            t_mid = (t_left + t_right) / 2

        if t_mid in samples:
            continue

        f_left, f_right, f_mid = samples[t_left], samples[t_right], f(t_mid)
        t_mid_rate = (t_mid - t_left) / (t_right - t_left)

        if err_func:
            # this calculates the error of linear interpolation, but can be customized by passing a different err_func
            # e.g. if T is float, err_func=lambda a, b, mid, rate: abs(mid - (b * rate + a * (1 - rate)))
            error = err_func(f_left, f_right, f_mid, t_mid_rate)
        else:
            f_interp = cast(Any, f_right) * t_mid_rate + cast(Any, f_left) * (
                1 - t_mid_rate
            )
            error = abs(cast(Any, f_mid) - f_interp)

        if error > error_threshold or force_sample:
            samples[t_mid] = f_mid
            queue.extend([(t_left, t_mid), (t_mid, t_right)])

    t_arr = sorted(samples.keys())
    return t_arr, [samples[t] for t in t_arr]


# Data merging
def merge_data_with_axes(
    *data: Tuple[NDArray, NDArray]
) -> Tuple[NDArray, List[NDArray]]:
    x_merged = np.unique(np.concatenate([x for x, _ in data]))
    y_list = [
        interp1d(
            x,
            y,
            axis=0,
            kind="linear",
            bounds_error=False,
            fill_value=cast(Any, "extrapolate"),
        )(x_merged)
        for x, y in data
    ]
    return x_merged, y_list


# Force-field and potential estimation
def estimate_force_field_kde(
    x_data: NDArray, dx_data: NDArray, x_grid: NDArray, h: float, k: float = 1.0
) -> NDArray:
    """Estimate the force field with a kernel-weighted mean."""
    diff = x_grid.reshape(-1, 1) - x_data.reshape(1, -1)
    weights = gaussian_kernel(diff, h)
    F_i = k * dx_data
    return np.sum(weights * F_i.reshape(1, -1), axis=1) / (
        np.sum(weights, axis=1) + 1e-14
    )


def estimate_potential_from_force(x_grid: NDArray, F_grid: NDArray) -> NDArray:
    """Estimate potential by trapezoidal integration of the force field."""
    dx = np.diff(x_grid)
    integral_segments = dx * (F_grid[1:] + F_grid[:-1]) / 2
    V_grid = np.zeros_like(F_grid)
    V_grid[1:] = -np.cumsum(integral_segments)
    return V_grid
