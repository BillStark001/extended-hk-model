from typing import Mapping, Optional, Tuple, List, overload, Union, Iterable, Any, Literal, Sequence
from numpy.typing import NDArray

import networkx as nx
import numpy as np

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ehk.common.numerics import adaptive_moving_stats

PAPER_WIDTH = 12.0  # inches
DEFAULT_HW_RATIO = 4 / 5  # height / width


def setup_paper_params():
  """Setup matplotlib parameters optimized for paper publication"""
  plt.rcParams.update({
      'font.size': 12,
      'font.family': 'serif',
      'text.usetex': False,  # Set to True if LaTeX is available
      'mathtext.fontset': 'cm',  # Computer Modern for math text
      'mathtext.rm': 'serif',
      'mathtext.it': 'serif:italic',
      'mathtext.bf': 'serif:bold',
      'axes.linewidth': 1.2,
      'axes.labelsize': 12,
      'axes.titlesize': 12,
      'xtick.labelsize': 10,
      'ytick.labelsize': 10,
      'legend.fontsize': 10,
      'figure.titlesize': 14,
      'savefig.dpi': 300,
      'savefig.bbox': 'tight',
  })


def plot_network_snapshot(
    pos: Mapping | None,
    opinion: NDArray,
    G: nx.Graph,
    ax: Optional[Axes] = None,
    step: int | float | None = None,
    cmap: str = 'coolwarm',
    colorbar: bool | Iterable[Axes] = True,
    node_size: int = 40,
    alpha: float = 0.36,
    rasterized: bool = True,
):
  if pos is None:
    pos = nx.spring_layout(G, pos=pos)
  norm = mpl.colors.Normalize(vmin=-1, vmax=1)  # type: ignore
  sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
  sm.set_array([])

  if ax is not None:
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)

  nx.draw_networkx_nodes(
      G, ax=ax, pos=pos, node_color=opinion,  # type: ignore
      cmap=cmap, vmin=-1, vmax=1, node_size=node_size,
  )
  nx.draw_networkx_edges(
      G, ax=ax, pos=pos, node_size=node_size, alpha=alpha
  )
  if ax is not None and rasterized:
    ax.set_rasterized(True)

  if ax is not None and step is not None:
    ax.set_xlabel(f'step = {step}')

  if colorbar:
    ax_colorbar = ax if colorbar is True else colorbar  # type: ignore
    if ax_colorbar is not None:
      plt.colorbar(sm, ticks=np.linspace(-1, 1, 5), ax=ax_colorbar)

  # plt.tight_layout()


def plot_opinion_colorbar(
    fig: Figure,
    axes: Union[Axes, List[Axes]],
    cmap: str = 'coolwarm',
    label: str = 'Opinion'
):
  """Plot a unified colorbar for opinion values across multiple axes

  Args:
    fig: The figure object
    axes: Single axis or list of axes to attach the colorbar to
    cmap: Colormap name (default: 'coolwarm')
    label: Label for the colorbar (default: 'Opinion')
  """
  norm = mpl.colors.Normalize(vmin=-1, vmax=1)  # type: ignore
  sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
  sm.set_array([])

  axes_list = axes if isinstance(axes, list) else [axes]
  cbar = fig.colorbar(sm, ax=axes_list, ticks=np.linspace(-1, 1, 5),
                      orientation="vertical", aspect=30, pad=0.01)
  cbar.set_label(label)
  return cbar


@overload
def plt_figure(
    n_row: Literal[1],
    n_col: Literal[1],
    hw_ratio: float = DEFAULT_HW_RATIO,
    total_width: float = PAPER_WIDTH,
    **kwargs
) -> Tuple[Figure, Axes]: ...


@overload
def plt_figure(
    n_row: Literal[1],
    n_col: int,
    hw_ratio: float = DEFAULT_HW_RATIO,
    total_width: float = PAPER_WIDTH,
    **kwargs
) -> Tuple[Figure, Sequence[Axes]]: ...


@overload
def plt_figure(
    n_row: int,
    n_col: Literal[1],
    hw_ratio: float = DEFAULT_HW_RATIO,
    total_width: float = PAPER_WIDTH,
    **kwargs
) -> Tuple[Figure, Sequence[Axes]]: ...


@overload
def plt_figure(
    n_row: int,
    n_col: int,
    hw_ratio: float = DEFAULT_HW_RATIO,
    total_width: float = PAPER_WIDTH,
    **kwargs
) -> Tuple[Figure, Sequence[Sequence[Axes]]]: ...


def plt_figure(
    n_row: int = 1,
    n_col: int = 1,
    hw_ratio: float = DEFAULT_HW_RATIO,
    total_width: float = PAPER_WIDTH,
    **kwargs
) -> Tuple[Figure, Union[Axes, Sequence[Axes], Sequence[Sequence[Axes]]]]:
  width = total_width / n_col
  height = width * hw_ratio
  total_height = height * n_row
  kwargs.setdefault('constrained_layout', True)
  return plt.subplots(
      n_row, n_col, figsize=(total_width, total_height), **kwargs
  )  # type: ignore


def plt_save_and_close(fig: Figure | None, path: str, jpg = False):
  """Save figure to path and close it"""
  plt.show()
  _fig = fig if fig is not None else plt
  _fig.savefig(path + ".pdf", dpi=300, bbox_inches="tight")
  _fig.savefig(path + ".png", dpi=300, bbox_inches="tight")
  if jpg:
    _fig.savefig(path + ".jpg", dpi=300, bbox_inches="tight")
  if fig is not None:
    plt.close(fig)
  else:
    plt.close()


def get_colormap(
    axes: Union[Axes, Iterable[Axes]],
    cmap='YlGnBu',
    vmin: float = -1, vmax: float = 1, seg: int = 5,
    fig: Any = plt,
    **kwargs
):
  norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)  # type: ignore
  sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
  sm.set_array([])

  cmap_arr = dict(cmap=cmap, vmin=vmin, vmax=vmax)

  def set_cmap_func(): return fig.colorbar(
      sm, ticks=np.linspace(vmin, vmax, seg), ax=axes, **kwargs)

  return cmap_arr, set_cmap_func


def linear_func(x, a, b):
  """Simple linear function: f(x) = a*x + b"""
  return a * x + b


def plot_line(ax: Axes, x_data: NDArray, y_data: NDArray):
  """Plot a linear best fit line through data points"""
  from scipy.optimize import curve_fit

  (a, b), _ = curve_fit(linear_func, x_data, y_data)
  min_x = np.min(x_data)
  max_x = np.max(x_data)
  x_line = np.array([min_x, max_x])
  ax.plot(x_line, a * x_line + b)
  return a, b


def draw_adaptive_moving_stats(
    axes: Axes,
    x: NDArray,
    y: NDArray,
    h0=0.1,
    color="tab:gray",
    alpha=0.1,
    log=False,
):
  """Draw adaptive moving statistics (mean and std)"""
  _points, mean, var = adaptive_moving_stats(x, y, h0)
  points = 10**_points if log else _points
  std = var**0.5
  axes.plot(points, mean, lw=1, color=color)
  axes.fill_between(points, mean - std, mean + std, color=color, alpha=alpha)
  return points, mean, std


def numpy_to_latex_table(
    data: NDArray,
    filename: str,
    caption="Table",
    row_labels: Optional[List[str]] = None,
    col_labels: Optional[List[str]] = None,
    scientific=True,
    precision=4
):
  # convert to string
  formatted_data = []
  for row in data:
    formatted_row = []
    for val in row:
      if scientific:
        formatted_row.append(f"{val:.{precision}e}")
      else:
        formatted_row.append(f"{val:.{precision}f}")
    formatted_data.append(formatted_row)

  # create table
  table = "\\begin{table}[h]\n\\centering\n"
  if row_labels is not None and col_labels is not None:
    table += "\\begin{tabular}{|l|" + "c|" * (data.shape[1]-1) + "c|}\n"
    table += "\\hline\n"
    table += " & " + " & ".join(col_labels) + "\\\\\n\\hline\n"
    for i, row in enumerate(formatted_data):
      table += row_labels[i] + " & " + " & ".join(row) + "\\\\\n\\hline\n"
  else:
    table += "\\begin{tabular}{" + "c" * data.shape[1] + "}\n"
    for row in formatted_data:
      table += " & ".join(row) + "\\\\\n"
  table += "\\end{tabular}\n\\caption{" + caption + "}\n\\end{table}"

  # save table
  with open(filename, "w") as f:
    f.write(table)
