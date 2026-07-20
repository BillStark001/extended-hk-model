from .array_utils import (
    first_less_than,
    last_less_than,
    first_more_or_equal_than,
    first_index_above_min,
    area_under_curve,
)
from .kde import (
    gaussian_kernel,
    compute_kde_density,
    compute_weighted_stats,
    kl_divergence_continuous,
    js_divergence_continuous,
    LINSPACE_SMPL_COUNT,
    fast_trapz,
    kl_divergence_continuous_fast,
    js_divergence_continuous_fast,
    kde_min_bw_factory,
    kde_min_bw_calc,
    min_bandwidth_enforcer,
    get_kde_pdf,
)
from .adaptive import (
    moving_average,
    adaptive_moving_stats,
    adaptive_discrete_sampling,
    merge_data_with_axes,
    estimate_force_field_kde,
    estimate_potential_from_force,
)
from .distance import (
    sample_linear_pdf,
    ideal_dist_init_array,
    ideal_dist_worst_obj_array,
    DistanceResultDebugData,
    DistanceResult,
    DistanceCalculator,
)
from .homophily import normalize_homophily, uniform_concordance_probability
