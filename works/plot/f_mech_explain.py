from typing import Callable, Tuple, Union

import numpy as np
from scipy import integrate, interpolate
from scipy.optimize import brentq
from scipy.stats import norm

from stats import adaptive_discrete_sampling
from utils.plot import plt_figure, plt_save_and_close, setup_paper_params


# region Integration Utilities

# P_{\epsilon, f}(x) := \int_{x-\epsilon}^{x+\epsilon} f(x, x')\,dx'
# \mu_{\epsilon, f}(x) := \frac{\int_{x-\epsilon}^{x+\epsilon} x' f(x, x')\,dx'}{P_{\epsilon, f}(x)}

_DENOM_TOL = 1e-14  # Denominator guard threshold


def compute_P_mu_pdf(
    f: Callable[[float, float], float], epsilon: float, x_values: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Pointwise adaptive integration with high accuracy but lower speed."""
    x_values = np.asarray(x_values, dtype=float)
    P = np.empty(len(x_values))
    mu = np.empty(len(x_values))

    for i, x in enumerate(x_values):
        a, b = x - epsilon, x + epsilon
        P_val, _ = integrate.quad(lambda xp: f(x, xp), a, b)
        P[i] = P_val
        if P_val < _DENOM_TOL:
            # [FIX] Denominator guard: fallback to current position when mass vanishes.
            mu[i] = x
            continue
        num, _ = integrate.quad(lambda xp: xp * f(x, xp), a, b)
        mu[i] = num / P_val

    return P, mu


def compute_P_mu_combined_pdf(
    f1: Callable[[float, float], float],
    f2: Callable[[float, float], float],
    k1: float,
    k2: float,
    epsilon: float,
    x_values: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute P and mu for f = k1*f1 + k2*f2.
    Uses linearity to combine two separately computed terms.
    """
    P1, mu1 = compute_P_mu_pdf(f1, epsilon, x_values)
    P2, mu2 = compute_P_mu_pdf(f2, epsilon, x_values)

    w1, w2 = k1 * P1, k2 * P2
    P_combined = w1 + w2

    # [FIX] Denominator guard
    safe_P = np.where(P_combined > _DENOM_TOL, P_combined, 1.0)
    mu_combined = np.where(
        P_combined > _DENOM_TOL,
        (w1 * mu1 + w2 * mu2) / safe_P,
        x_values,
    )
    return P_combined, mu_combined


def compute_P_mu_cdf(
    F: Callable[[float, float], float], epsilon: float, x_values: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    CDF-based pointwise computation of P and mu.

    P is given directly by endpoint differences:
        P = F(x, x+ε) - F(x, x-ε)

    The numerator is simplified by integration by parts:
        ∫ x'f dx' = (x+ε)·F(x,x+ε) - (x-ε)·F(x,x-ε) - ∫ F(x,x') dx'
    """
    x_values = np.asarray(x_values, dtype=float)
    P = np.empty(len(x_values))
    mu = np.empty(len(x_values))

    for i, x in enumerate(x_values):
        a, b = x - epsilon, x + epsilon
        Fa, Fb = F(x, a), F(x, b)
        P_val = Fb - Fa
        P[i] = P_val

        if P_val < _DENOM_TOL:
            # [FIX] Denominator guard
            mu[i] = x
            continue

        integral_F, _ = integrate.quad(lambda xp: F(x, xp), a, b)
        mu[i] = (b * Fb - a * Fa - integral_F) / P_val

    return P, mu


def compute_P_mu_combined_cdf(
    F1: Callable[[float, float], float],
    F2: Callable[[float, float], float],
    k1: float,
    k2: float,
    epsilon: float,
    x_values: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute P and mu for F = k1*F1 + k2*F2 (CDF version).

        P_combined = k1·P1 + k2·P2
        μ_combined = (k1·P1·μ1 + k2·P2·μ2) / (k1·P1 + k2·P2)
    """
    P1, mu1 = compute_P_mu_cdf(F1, epsilon, x_values)
    P2, mu2 = compute_P_mu_cdf(F2, epsilon, x_values)

    w1, w2 = k1 * P1, k2 * P2
    P_combined = w1 + w2

    # [FIX] Denominator guard
    safe_P = np.where(P_combined > _DENOM_TOL, P_combined, 1.0)
    mu_combined = np.where(
        P_combined > _DENOM_TOL,
        (w1 * mu1 + w2 * mu2) / safe_P,
        x_values,
    )
    return P_combined, mu_combined


def find_bounds(
    cdf: Callable[[float], float],
    x: float,
    k: float,
    tol: float = 1e-12,
    domain: Tuple[float, float] = (-np.inf, np.inf),
) -> Tuple[float, float]:
    """
    Given a CDF, solve distances a and b such that:
        [F(x) - F(x-a)] + [F(x+b) - F(x)] = 2k

    Return (a, b), ensuring L <= x-a and x+b <= R.
    Note: return values are distances, not absolute coordinates.
    """
    L, R = domain

    if not (L <= x <= R):
        raise ValueError(f"x={x} is outside domain [{L}, {R}]")
    if k <= 0:
        return 0.0, 0.0

    cdf_x = cdf(x)
    cdf_L = 0.0 if np.isinf(L) else cdf(L)
    cdf_R = 1.0 if np.isinf(R) else cdf(R)

    left_avail = cdf_x - cdf_L
    right_avail = cdf_R - cdf_x

    if 2 * k > left_avail + right_avail + tol:
        raise ValueError(
            f"k is too large: required 2k={2*k:.6g}, "
            f"but only {left_avail + right_avail:.6g} mass is available in-domain"
        )

    k_left = min(k, left_avail)
    k_right = min(2 * k - k_left, right_avail)
    k_left = 2 * k - k_right

    # Left-side distance a
    if k_left >= left_avail - tol:
        a = (x - L) if not np.isinf(L) else np.inf
    else:
        target_left = cdf_x - k_left
        if not np.isinf(L):
            a_hi = x - L
        else:
            a_hi = 1.0
            while cdf(x - a_hi) > target_left:
                a_hi *= 2
        # [FIX] Validate bracketing for brentq: f(0) > 0 and f(a_hi) should be < 0.
        if cdf(x - a_hi) - target_left >= 0:
            # Floating-point edge case: fallback to upper bound.
            a = a_hi
        else:
            a = brentq(lambda v: cdf(x - v) - target_left, 0.0, a_hi, xtol=tol)

    # Right-side distance b
    if k_right >= right_avail - tol:
        b = (R - x) if not np.isinf(R) else np.inf
    else:
        target_right = cdf_x + k_right
        if not np.isinf(R):
            b_hi = R - x
        else:
            b_hi = 1.0
            while cdf(x + b_hi) < target_right:
                b_hi *= 2
        # [FIX] Same as above: f(0) < 0 and f(b_hi) should be > 0.
        if cdf(x + b_hi) - target_right <= 0:
            b = b_hi
        else:
            b = brentq(lambda v: cdf(x + v) - target_right, 0.0, b_hi, xtol=tol)

    return a, b  # type: ignore


def build_cdf(
    pdf: Union[Callable[[float], float], Tuple[np.ndarray, np.ndarray]],
    x_start: float,
    x_end: float,
    n_points: int = 2000,
    normalize: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Callable[[Union[float, np.ndarray]], np.ndarray]]:
    """Numerically integrate a PDF into a CDF and return an interpolation function."""
    x = np.linspace(x_start, x_end, n_points)

    if callable(pdf):
        y = np.asarray(pdf(x), dtype=float)  # type: ignore
    else:
        x_data = np.asarray(pdf[0], dtype=float)
        y_data = np.asarray(pdf[1], dtype=float)
        if x_data.shape != y_data.shape or x_data.ndim != 1:
            raise ValueError(
                "For tuple input, x_arr and y_arr must be 1D arrays of equal length."
            )
        if len(x_data) < 4:
            interp_fn = interpolate.interp1d(
                x_data, y_data, kind="linear", bounds_error=False, fill_value=0.0
            )
        else:
            interp_fn = interpolate.CubicSpline(x_data, y_data, extrapolate=False)
        y = np.nan_to_num(interp_fn(x), nan=0.0)

    if np.any(y < 0):
        raise ValueError("PDF contains negative values; please check the input.")

    cdf_vals = integrate.cumulative_trapezoid(y, x, initial=0.0)
    total = cdf_vals[-1]
    if total == 0:
        raise ValueError(
            "PDF integrates to 0 on the specified interval; check x_start/x_end or the PDF definition."
        )
    if normalize:
        cdf_vals = cdf_vals / total

    cdf_fn = interpolate.interp1d(
        x,
        cdf_vals,
        kind="linear",
        bounds_error=False,
        fill_value=(0.0, cdf_vals[-1]),  # type: ignore
    )
    return x, cdf_vals, cdf_fn


# endregion

# region Parameters


def rate2(a: float) -> float:
    return a**2 + (1 - a) ** 2


def sigma(steepness: float) -> float:
    return 1 / steepness


def err_func_float(a: float, b: float, mid: float, rate: float) -> float:
    return abs(mid - (b * rate + a * (1 - rate)))


def err_func_float_tuple(
    l: Tuple[float, float],
    r: Tuple[float, float],
    mid: Tuple[float, float],
    rate: float,
) -> float:
    return max(
        err_func_float(l[0], r[0], mid[0], rate),
        err_func_float(l[1], r[1], mid[1], rate),
    )


def get_cdf_op(
    cdf_rand: Callable[[float, float], float],
) -> Callable[[float, float], float]:
    """
    Build an opinion-peer conditional CDF constrained by op_rec_mass.

    Fix summary
    -----------
    1. find_bounds returns distances (a, b), not absolute coordinates.
        Convert to absolute bounds lb = x - a and ub = x + b before interpolation.

    2. Normalization direction was wrong in the original version.
        Divide by the actual in-window mass (nominally op_rec_mass).

    3. fill_value=-1 is meaningless for distances.
        Use endpoint extrapolation to avoid invalid out-of-range coordinates.

    4. Performance: for fixed x, integrate.quad repeatedly calls cdf_op(x, ·).
        Use a per-x cache to avoid redundant evaluations.
    """
    smpl_x_list, smpl_bounds_list = adaptive_discrete_sampling(
        lambda x: find_bounds(
            lambda xp: cdf_rand(x, xp),
            x,
            op_rec_mass / 2,
            tol=1e-13,
            domain=(-1.0, 1.0),
        ),
        1e-3,
        -1.0,
        1.0,
        1e-2,
        int_midpoint=False,
        err_func=err_func_float_tuple,
    )

    smpl_x_arr = np.asarray(smpl_x_list, dtype=float)
    ab = np.asarray(smpl_bounds_list, dtype=float)  # shape (n, 2)
    smpl_a, smpl_b = ab[:, 0], ab[:, 1]

    # [FIX 1] Convert distances (a, b) to absolute coordinates.
    smpl_lb = smpl_x_arr - smpl_a  # absolute left bound
    smpl_ub = smpl_x_arr + smpl_b  # absolute right bound

    _ikw = dict(kind="linear", bounds_error=False)

    # [FIX 3] Use endpoint extrapolation instead of invalid fill values.
    lb_func = interpolate.interp1d(
        smpl_x_arr,
        smpl_lb,
        fill_value=(smpl_lb[0], smpl_lb[-1]),  # type: ignore
        **_ikw,  # type: ignore
    )
    ub_func = interpolate.interp1d(
        smpl_x_arr,
        smpl_ub,
        fill_value=(smpl_ub[0], smpl_ub[-1]),  # type: ignore
        **_ikw,  # type: ignore
    )

    # [FIX 4] Precompute lb_cdf and total mass on sampled points.
    smpl_lb_cdf = np.array(
        [cdf_rand(xi, lbi) for xi, lbi in zip(smpl_x_arr, smpl_lb)], dtype=float
    )
    smpl_total = np.array(
        [
            cdf_rand(xi, ubi) - cdf_rand(xi, lbi)
            for xi, ubi, lbi in zip(smpl_x_arr, smpl_ub, smpl_lb)
        ],
        dtype=float,
    )

    lb_cdf_func = interpolate.interp1d(
        smpl_x_arr,
        smpl_lb_cdf,
        fill_value=(smpl_lb_cdf[0], smpl_lb_cdf[-1]),  # type: ignore
        **_ikw,  # type: ignore
    )
    total_func = interpolate.interp1d(
        smpl_x_arr,
        smpl_total,
        fill_value=(smpl_total[0], smpl_total[-1]),  # type: ignore
        **_ikw,  # type: ignore
    )

    # Single-key cache: quad repeatedly calls cdf_op(x, ·) for fixed x.
    _cache: dict = {}

    def _refresh(x: float) -> None:
        if _cache.get("x") == x:
            return
        _cache["x"] = x
        _cache["lb"] = float(lb_func(x))
        _cache["ub"] = float(ub_func(x))
        _cache["lb_cdf"] = float(lb_cdf_func(x))
        _cache["total"] = float(total_func(x))

    def cdf_op(x: float, xp: float) -> float:
        _refresh(x)
        lb, ub = _cache["lb"], _cache["ub"]
        if xp <= lb:
            return 0.0
        if xp >= ub:
            return 1.0
        total = _cache["total"]
        if total < _DENOM_TOL:
            # [FIX 2] Degenerate-window guard
            return 0.5
        # [FIX 2] Divide by total mass (≈ op_rec_mass), not multiply by it.
        return float(
            np.clip(
                (cdf_rand(x, xp) - _cache["lb_cdf"]) / total,
                0.0,
                1.0,
            )
        )

    return cdf_op


def naive_integrate(x: np.ndarray, sf: np.ndarray) -> np.ndarray:
    ret = np.array(
        [
            integrate.trapezoid(sf[: i + 1], x[: i + 1]) if i > 0 else 0.0
            for i in range(len(x))
        ]
    )
    ret_normalizer = np.sum(ret) / len(x)
    ret -= ret_normalizer
    return ret


# endregion

# region Global Parameters

N = 500
k_n = 15
k_r = 10
k = k_n + k_r
eps = 0.45

op_rec_mass = k_r / (N - k_n)

steepness_cons_values = [8, 64]
rate_polarized_same_values = [1 - 0.1, 1 - 0.0125]


# endregion

# region Initial State


def cdf_init_neighbor(x: float, xp: float) -> float:
    if xp < -1:
        return 0.0
    elif xp > 1:
        return 1.0
    return 0.5 * (xp + 1)


cdf_init_rand = cdf_init_neighbor
cdf_init_st = cdf_init_neighbor

init_op_interval = 2 * op_rec_mass
init_op_interval_rev = 1.0 / init_op_interval
init_op_half_interval = init_op_interval / 2


def pdf_init_op(x: float, xp: float) -> float:
    if x < -1 + init_op_half_interval:
        if xp < -1 or xp > -1 + init_op_interval:
            return 0.0
        return init_op_interval_rev
    elif x > 1 - init_op_half_interval:
        if xp < 1 - init_op_interval or xp > 1:
            return 0.0
        return init_op_interval_rev
    if xp < x - init_op_half_interval or xp > x + init_op_half_interval:
        return 0.0
    return init_op_interval_rev


def cdf_init_op(x: float, xp: float) -> float:
    if x < -1 + init_op_half_interval:
        if xp < -1:
            return 0.0
        elif xp > -1 + init_op_interval:
            return 1.0
        return init_op_interval_rev * (xp + 1)
    elif x > 1 - init_op_half_interval:
        if xp < 1 - init_op_interval:
            return 0.0
        elif xp > 1:
            return 1.0
        return init_op_interval_rev * (xp - (1 - init_op_interval))
    if xp < x - init_op_half_interval:
        return 0.0
    elif xp > x + init_op_half_interval:
        return 1.0
    return init_op_interval_rev * (xp - (x - init_op_half_interval))


# endregion

# region Consensus State


def build_cons_cdfs(
    steepness_cons_val: float,
) -> Tuple[
    Callable[[float, float], float],
    Callable[[float, float], float],
    Callable[[float, float], float],
    Callable[[float, float], float],
]:
    s_cons = sigma(steepness_cons_val)

    def cdf_cons_neighbor(_x: float, xp: float) -> float:
        return float(norm.cdf(xp, loc=0, scale=s_cons))

    cdf_cons_rand = cdf_cons_neighbor
    cdf_cons_st = cdf_cons_neighbor
    cdf_cons_op = get_cdf_op(cdf_cons_rand)
    return cdf_cons_neighbor, cdf_cons_rand, cdf_cons_st, cdf_cons_op


# endregion

# region Polarized State


def build_div_cdfs(
    steepness_div_val: float,
    rate_polarized_same: float,
) -> Tuple[
    Callable[[float, float], float],
    Callable[[float, float], float],
    Callable[[float, float], float],
    Callable[[float, float], float],
]:
    s_div = sigma(steepness_div_val)

    rate_polarized_diff = 1 - rate_polarized_same

    rate2_polarized_same = rate2(rate_polarized_same)
    rate2_polarized_diff = 1 - rate2_polarized_same

    def cdf_div_base(x: float, xp: float, r1: float, r2: float) -> float:
        return float(
            r1 * norm.cdf(xp, loc=-0.5, scale=s_div)
            + r2 * norm.cdf(xp, loc=0.5, scale=s_div)
        )

    def cdf_div_neighbor(x: float, xp: float) -> float:
        r1, r2 = (
            (rate_polarized_same, rate_polarized_diff)
            if x < 0
            else (rate_polarized_diff, rate_polarized_same)
        )
        return cdf_div_base(x, xp, r1, r2)

    def cdf_div_rand(x: float, xp: float) -> float:
        return cdf_div_base(x, xp, 0.5, 0.5)

    def cdf_div_st(x: float, xp: float) -> float:
        r1, r2 = (
            (rate2_polarized_same, rate2_polarized_diff)
            if x < 0
            else (rate2_polarized_diff, rate2_polarized_same)
        )
        return cdf_div_base(x, xp, r1, r2)

    cdf_div_op = get_cdf_op(cdf_div_rand)
    return cdf_div_neighbor, cdf_div_rand, cdf_div_st, cdf_div_op


# endregion

if __name__ == "__main__":

    setup_paper_params()

    # region Social Forces

    x_axis = np.linspace(-1, 1, 1001)

    _, sf_init_rand = compute_P_mu_cdf(cdf_init_neighbor, eps, x_axis)
    sf_init_st = sf_init_rand
    _, sf_init_op = compute_P_mu_combined_cdf(
        cdf_init_neighbor, cdf_init_op, k_n / k, k_r / k, eps, x_axis
    )

    # Sweep steepness for e2-e5: consensual/diverged at steepness 8 and 32.
    pot_cons = {"Rand.": {}, "St.": {}, "Op.": {}}
    pot_div = {"Rand.": {}, "St.": {}, "Op.": {}}

    for steep_cons, rate_polarized_same in zip(
        steepness_cons_values, rate_polarized_same_values
    ):
        steep_div = steep_cons * 2

        cdf_cons_neighbor, _, cdf_cons_st, cdf_cons_op = build_cons_cdfs(steep_cons)
        _, sf_cons_rand = compute_P_mu_cdf(cdf_cons_neighbor, eps, x_axis)
        _, sf_cons_st = compute_P_mu_cdf(cdf_cons_st, eps, x_axis)
        _, sf_cons_op = compute_P_mu_combined_cdf(
            cdf_cons_neighbor, cdf_cons_op, k_n / k, k_r / k, eps, x_axis
        )

        cdf_div_neighbor, cdf_div_rand, cdf_div_st, cdf_div_op = build_div_cdfs(
            steep_div, rate_polarized_same
        )
        _, sf_div_rand = compute_P_mu_combined_cdf(
            cdf_div_neighbor, cdf_div_rand, k_n / k, k_r / k, eps, x_axis
        )
        _, sf_div_st = compute_P_mu_combined_cdf(
            cdf_div_neighbor, cdf_div_st, k_n / k, k_r / k, eps, x_axis
        )
        _, sf_div_op = compute_P_mu_combined_cdf(
            cdf_div_neighbor, cdf_div_op, k_n / k, k_r / k, eps, x_axis
        )

        pot_cons["Rand."][steep_cons] = naive_integrate(x_axis, x_axis - sf_cons_rand)
        pot_cons["St."][steep_cons] = naive_integrate(x_axis, x_axis - sf_cons_st)
        pot_cons["Op."][steep_cons] = naive_integrate(x_axis, x_axis - sf_cons_op)

        pot_div["Rand."][steep_cons] = naive_integrate(x_axis, x_axis - sf_div_rand)
        pot_div["St."][steep_cons] = naive_integrate(x_axis, x_axis - sf_div_st)
        pot_div["Op."][steep_cons] = naive_integrate(x_axis, x_axis - sf_div_op)

    # endregion

    # region Potential Landscapes

    pot_init_rand = naive_integrate(x_axis, x_axis - sf_init_rand)
    pot_init_st = naive_integrate(x_axis, x_axis - sf_init_st)
    pot_init_op = naive_integrate(x_axis, x_axis - sf_init_op)

    fig, axes = plt_figure(n_row=1, n_col=5, total_width=16)
    ax_init, ax_cons_1, ax_cons_2, ax_div_1, ax_div_2 = axes

    ax_init.plot(x_axis, pot_init_rand, label="Rand.", color="tab:blue")
    ax_init.plot(x_axis, pot_init_st, label="St.", color="tab:orange", linestyle="--")
    ax_init.plot(x_axis, pot_init_op, label="Op.", color="tab:green")

    # Keep original mechanism color/style mapping.
    for ax, steep_cons in [
        (ax_cons_1, steepness_cons_values[0]),
        (ax_cons_2, steepness_cons_values[1]),
    ]:
        ax.plot(x_axis, pot_cons["Rand."][steep_cons], color="tab:blue")
        ax.plot(x_axis, pot_cons["St."][steep_cons], color="tab:orange", linestyle="--")
        ax.plot(x_axis, pot_cons["Op."][steep_cons], color="tab:green")

    for ax, steep_cons in [
        (ax_div_1, steepness_cons_values[0]),
        (ax_div_2, steepness_cons_values[1]),
    ]:
        ax.plot(x_axis, pot_div["Rand."][steep_cons], color="tab:blue")
        ax.plot(x_axis, pot_div["St."][steep_cons], color="tab:orange", linestyle="--")
        ax.plot(x_axis, pot_div["Op."][steep_cons], color="tab:green")

    ax_init.set_title("(e1) Initial", loc="left")
    ax_cons_1.set_title(r"(e2) Consensual, $\kappa=8$", loc="left")
    ax_cons_2.set_title(r"(e3) Consensual, $\kappa=64$", loc="left")
    ax_div_1.set_title(r"(e4) Diverged, $\kappa=8, k_d=0.1$", loc="left")
    ax_div_2.set_title(r"(e5) Diverged, $\kappa=64, k_d=0.0125$", loc="left")

    for ax in axes:
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.set_xlim(-1, 1)
        ax.set_xlabel(r"$x$")

    # Unify y-axis across all five panels and show y ticks/label only on the leftmost axis.
    all_pots = [
        pot_init_rand,
        pot_init_st,
        pot_init_op,
        pot_cons["Rand."][steepness_cons_values[0]],
        pot_cons["St."][steepness_cons_values[0]],
        pot_cons["Op."][steepness_cons_values[0]],
        pot_cons["Rand."][steepness_cons_values[1]],
        pot_cons["St."][steepness_cons_values[1]],
        pot_cons["Op."][steepness_cons_values[1]],
        pot_div["Rand."][steepness_cons_values[0]],
        pot_div["St."][steepness_cons_values[0]],
        pot_div["Op."][steepness_cons_values[0]],
        pot_div["Rand."][steepness_cons_values[1]],
        pot_div["St."][steepness_cons_values[1]],
        pot_div["Op."][steepness_cons_values[1]],
    ]
    y_min = min(float(np.min(v)) for v in all_pots)
    y_max = max(float(np.max(v)) for v in all_pots)
    y_pad = 0.04 * max(y_max - y_min, 1e-9)
    for ax in axes:
        ax.set_ylim(y_min - y_pad, y_max + y_pad)

    ax_init.set_ylabel(r"$V(x)$")
    ax_init.legend()
    for ax in [ax_cons_1, ax_cons_2, ax_div_1, ax_div_2]:
        ax.set_ylabel("")
        ax.tick_params(axis="y", labelleft=False)

    plt_save_and_close(fig, "fig/f_mech_explain_potential")

    # endregion
