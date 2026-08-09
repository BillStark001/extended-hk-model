from typing import Any, Optional
from numpy.typing import NDArray

import dataclasses

import numpy as np
import networkx as nx
from scipy.stats import norm

from ehk.common.numerics.kde import (
    get_kde_pdf,
    LINSPACE_SMPL_COUNT,
    js_divergence_continuous_fast,
    kl_divergence_continuous_fast,
)


# Ideal-distribution helpers
def sample_linear_pdf(n: int, k: float, random_state=None):
    rng = np.random.default_rng(random_state)
    u = rng.uniform(0, 1, size=n)
    x = k - k * np.sqrt(1 - u)
    return x


def ideal_dist_init_array(x: np.ndarray, k: float = 2):
    ret_array = x / -k + 1
    ret_array[x > k] = 0
    ret_array[x < 0] = 0
    return ret_array


def ideal_dist_worst_obj_array(
    axis: np.ndarray,
    cl: float,
    effective_sample_size: int,
    min_bw: float,
):
    m = max(1, effective_sample_size)
    std = cl / 2  # std. of dataset
    factor = max(
        m ** (-1.0 / (1 + 4)),  # scotts
        min_bw / std,
    )
    std_norm = std * factor

    o_worst_vals = 0.5 * norm.pdf(axis, 0, std_norm) + 0.5 * norm.pdf(
        axis, cl, std_norm
    )

    return o_worst_vals


# Opinion-distance result types
@dataclasses.dataclass
class DistanceResultDebugData:
    o_sample: NDArray
    s_sample: NDArray
    axis: NDArray
    rand_o_pdf: NDArray
    rand_s_pdf: NDArray
    worst_o_pdf: NDArray
    worst_s_pdf: NDArray


@dataclasses.dataclass
class DistanceResult:
    rand_o: float
    rand_s: float
    worst_o: float
    worst_s: float
    debug_data: Optional[DistanceResultDebugData] = None


class DistanceCalculator:
    def __init__(
        self,
        min_bandwidth: float = 0.05,
        sample_count: int = 500,
        max_kde_samples: int = 10000,
        t_err: float = 1e-10,
        k: float = 2.0,
        use_js_divergence: bool = False,
        use_debug_data: bool = False,
    ):
        self.min_bandwidth = min_bandwidth
        self.sample_count = sample_count
        self.max_kde_samples = max_kde_samples

        self.t_err = t_err
        self.k = k

        self.use_js_divergence = use_js_divergence
        self.use_debug_data = use_debug_data

        self.div = (
            js_divergence_continuous_fast
            if use_js_divergence
            else kl_divergence_continuous_fast
        )

        err_range = self.min_bandwidth * 4
        self.err_range = err_range
        self.axis = np.linspace(0 - err_range, k + err_range, LINSPACE_SMPL_COUNT)

        o_rand_smpl = sample_linear_pdf(int(sample_count**1.75), k)
        o_rand_pdf = get_kde_pdf(o_rand_smpl, self.min_bandwidth, 0, k)
        o_rand_vals = o_rand_pdf(self.axis)

        self.o_rand_vals = o_rand_vals
        self.triu_indices = np.triu_indices(sample_count, k=1)

        # Cache data-independent constants used by every collect() call.
        self._params_dict: Any = dict(
            xmin=0 - err_range, xmax=k + err_range, t_err=t_err, n=self.axis
        )
        s_worst_vals = norm.pdf(self.axis, 0, min_bandwidth)
        self.s_worst_vals = s_worst_vals
        if use_js_divergence:
            self._s_scale_rand = self._s_scale_worst = self.div(
                s_worst_vals, o_rand_vals, **self._params_dict
            )
        else:
            self._s_scale_rand = self.div(
                s_worst_vals, o_rand_vals, **self._params_dict
            )
            self._s_scale_worst = self.div(
                o_rand_vals, s_worst_vals, **self._params_dict
            )

    def calculate(
        self,
        digraph: nx.DiGraph,
        opinion: np.ndarray,
        t_opinion: float = 0.4,
    ) -> DistanceResult:

        k = self.k

        # Compute pairwise differences without allocating a full n-by-n matrix.
        i_idx, j_idx = self.triu_indices

        o_sample = np.abs(opinion[i_idx] - opinion[j_idx])
        o_sample = np.clip(o_sample, 0, k)

        neighbors = np.array(digraph.edges)
        s_sample = np.abs(opinion[neighbors[:, 1]] - opinion[neighbors[:, 0]])
        s_sample = np.clip(s_sample, 0, k)

        # Subsample large inputs to keep KDE fitting tractable.
        n_max = self.max_kde_samples
        if n_max > 0 and len(o_sample) > n_max:
            idx = np.random.choice(len(o_sample), n_max, replace=False)
            o_kde = o_sample[idx]
        else:
            o_kde = o_sample

        if n_max > 0 and len(s_sample) > n_max:
            idx = np.random.choice(len(s_sample), n_max, replace=False)
            s_kde = s_sample[idx]
        else:
            s_kde = s_sample

        # Fit the density to the subsampled values.
        o_pdf = get_kde_pdf(o_kde, self.min_bandwidth, 0, k)
        s_pdf = get_kde_pdf(s_kde, self.min_bandwidth, 0, k)

        # random cases, use pre-rendered
        axis = self.axis
        s_rand_vals = o_rand_vals = self.o_rand_vals

        # worst cases
        # objective
        o_worst_b = o_sample[o_sample >= t_opinion]
        o_worst_v = float(t_opinion if o_worst_b.size == 0 else np.mean(o_worst_b))
        o_worst_vals = ideal_dist_worst_obj_array(
            axis, o_worst_v, len(o_kde), self.min_bandwidth
        )

        # Use the cached worst-case subjective-polarization values.
        s_worst_vals = self.s_worst_vals

        # The subjective scale is cached; the objective scale is data-dependent.
        params_dict = self._params_dict
        o_scale_worst = o_scale_rand = self.div(
            o_worst_vals, o_rand_vals, **params_dict
        )
        s_scale_rand = self._s_scale_rand
        s_scale_worst = self._s_scale_worst
        if not self.use_js_divergence:
            o_scale_worst = self.div(o_rand_vals, o_worst_vals, **params_dict)

        o_vals = o_pdf(axis)
        s_vals = s_pdf(axis)

        debug_data = None
        if self.use_debug_data:
            debug_data = DistanceResultDebugData(
                o_sample=o_sample,
                s_sample=s_sample,
                axis=axis,
                rand_o_pdf=o_rand_vals,
                rand_s_pdf=o_rand_vals,
                worst_o_pdf=o_worst_vals,
                worst_s_pdf=s_worst_vals,
            )

        rand_o = self.div(o_vals, o_rand_vals, **params_dict) / o_scale_rand
        rand_s = self.div(s_vals, s_rand_vals, **params_dict) / s_scale_rand
        worst_o = self.div(o_vals, o_worst_vals, **params_dict) / o_scale_worst
        worst_s = self.div(s_vals, s_worst_vals, **params_dict) / s_scale_worst

        return DistanceResult(
            rand_o=rand_o,
            rand_s=rand_s,
            worst_o=worst_o,
            worst_s=worst_s,
            debug_data=debug_data,
        )
