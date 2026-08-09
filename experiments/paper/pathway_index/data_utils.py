from typing import Tuple
from numpy.typing import NDArray
import numpy as np
from scipy.stats import gaussian_kde


def linear_func(x, a, b):
  """Simple linear function: f(x) = a*x + b"""
  return a * x + b


def scale_array(
    arr: NDArray,
    range_from: Tuple[float, float],
    range_to: Tuple[float, float],
):
  """Scale array from one range to another"""
  a, b = range_from
  c, d = range_to
  return c + ((d-c) / (b-a)) * (arr-a)


def transform_kde(
    data: NDArray,
    lower: float | None = None,  # pyright: ignore[reportRedeclaration]
    upper: float | None = None,  # pyright: ignore[reportRedeclaration]
):
  """Transform data using KDE with arctanh transformation"""
  if lower is None:
    lower: float = data.min()
  if upper is None:
    upper: float = data.max()

  scaled_data = scale_array(data, (lower, upper), (-1, 1))
  transformed_data = np.arctanh(scaled_data)
  kde = gaussian_kde(transformed_data)

  def density(x):
    scaled_x = scale_array(x, (lower, upper), (-1, 1))
    transformed_x = np.arctanh(scaled_x)
    trsf_density_inv = 1 - np.tanh(scaled_x) ** 2
    return kde(transformed_x) * np.abs(
        2 / (upper - lower) *
        (1 / trsf_density_inv)
    )

  return density


def partition_data(
    grad: NDArray, y: NDArray, consensual: NDArray,
    threshold=0.8,
):
  """Partition data based on gradient and consensus"""
  p2_mask = grad < threshold
  p1_mask = grad >= threshold

  polarized = np.logical_not(consensual)

  d_p2_p = y[np.logical_and(p2_mask, polarized)]
  d_p2_c = y[np.logical_and(p2_mask, consensual)]
  d_p1_p = y[np.logical_and(p1_mask, polarized)]
  d_p1_c = y[np.logical_and(p1_mask, consensual)]

  d = [d_p2_p, d_p2_c, d_p1_p, d_p1_c]
  means = np.array([np.mean(dd) for dd in d])
  std_devs = np.array([np.std(dd) for dd in d])
  return means, std_devs


def piecewise_linear_integral_trapz(x: np.ndarray, y: np.ndarray, a: float, b: float):
  # Normalize the interval orientation.
  if a > b:
    a, b = b, a

  # Include interval endpoints.
  x_new = np.sort(np.concatenate([x, [a, b]]))
  y_new = np.interp(x_new, x, y, left=0, right=1)

  # Keep points inside the interval.
  mask = (x_new >= a) & (x_new <= b)
  x_new = x_new[mask]
  y_new = y_new[mask]

  # Integrate the clipped curve.
  return np.trapz(y_new, x_new)
