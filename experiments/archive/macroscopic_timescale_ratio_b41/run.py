"""Archived B=41 exploratory operator-resolved time-scale scan.

Retained for provenance only.  Do not use its numerical defaults for the
paper-aligned Figure 2 comparison.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shlex
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import rankdata, spearmanr

from ehk.common.plotting import setup_paper_params
from ehk.metrics import (
    calculate_channel_contributions,
    calculate_index_series,
    summarize_channel_window,
)
from ehk.modeling.mesoscopic import KineticParameters, solve
from theory.mesoscopic.cli_utils import (
    first_crossing_or_nan,
    pathway_label,
    write_run_metadata,
)
from theory.mesoscopic.phase_scan import RATES
from theory.mesoscopic.recommender_scan import (
    CONFIGURATIONS as BASE_CONFIGURATIONS,
)
from theory.mesoscopic.recommender_scan import (
    DISPLAY_NAMES as BASE_DISPLAY_NAMES,
)

L1_CONFIGURATIONS = (
    (
        "structure_random_l1_zeta1",
        "structure_random_l1_mean_power",
        1.0,
    ),
    (
        "structure_random_l1_zeta4",
        "structure_random_l1_mean_power",
        4.0,
    ),
)
CONFIGURATIONS = (*BASE_CONFIGURATIONS, *L1_CONFIGURATIONS)
DISPLAY_NAMES = {
    **BASE_DISPLAY_NAMES,
    "structure_random_l1_zeta1": r"L1-StructureRandom ($\zeta=1$)",
    "structure_random_l1_zeta4": (r"L1-StructureRandom ($\zeta=4$, mean-power)"),
}


@dataclass(frozen=True)
class TimescaleResult:
    configuration: str
    recsys: str
    steepness: float
    alpha: float
    q: float
    path: str
    pathway: float
    t_polarization: float
    t_homophily: float
    precedence: float
    opinion_drive: float
    rewiring_drive: float
    gamma: float
    gamma_0p10: float
    gamma_0p40: float
    gamma_initial: float
    gamma_at_window: float
    window_end: float


@dataclass(frozen=True)
class ThresholdScore:
    threshold: float
    accuracy: float
    balanced_accuracy: float
    auc: float
    count: int


def _configurations_in_results(
    results: list[TimescaleResult],
) -> tuple[tuple[str, str, float], ...]:
    present = {result.configuration for result in results}
    return tuple(item for item in CONFIGURATIONS if item[0] in present)


def record_schedule(
    steps: int,
    *,
    early_until: int,
    early_every: int,
    late_every: int,
) -> tuple[int, ...]:
    if steps < 1 or early_until < 0 or early_every < 1 or late_every < 1:
        raise ValueError("invalid record schedule")
    split = min(steps, early_until)
    selected = set(range(0, split + 1, early_every))
    selected.update(range(split, steps + 1, late_every))
    selected.update((0, split, steps))
    return tuple(sorted(selected))


def _precedence(t_polarization: float, t_homophily: float) -> float:
    if not np.isfinite(t_polarization) or not np.isfinite(t_homophily):
        return float("nan")
    total = t_polarization + t_homophily
    return (t_homophily - t_polarization) / total if total > 0 else float("nan")


def _solve_case(
    configuration: str,
    recsys: str,
    steepness: float,
    alpha: float,
    q: float,
    base: KineticParameters,
    selected_steps: tuple[int, ...],
    probe_dt: float,
    progress_threshold: float,
) -> TimescaleResult:
    params = replace(
        base,
        recsys=recsys,
        recommendation_steepness=steepness,
        influence=alpha,
        rewiring=q,
    )
    trajectory = solve(params, record_steps=selected_steps)
    indices = calculate_index_series(trajectory)
    contributions = calculate_channel_contributions(
        trajectory,
        indices,
        probe_dt=probe_dt,
        progress_threshold=progress_threshold,
    )
    sensitivity = {
        threshold: summarize_channel_window(
            trajectory.time,
            indices.polarization,
            indices.homophily,
            contributions.opinion_polarization_rate,
            contributions.rewiring_homophily_rate,
            progress_threshold=threshold,
        )
        for threshold in (0.10, 0.40)
    }
    t_p = first_crossing_or_nan(indices.time, indices.polarization)
    t_h = first_crossing_or_nan(indices.time, indices.homophily)
    return TimescaleResult(
        configuration=configuration,
        recsys=recsys,
        steepness=steepness,
        alpha=alpha,
        q=q,
        path=pathway_label(t_p, t_h),
        pathway=indices.pathway,
        t_polarization=t_p,
        t_homophily=t_h,
        precedence=_precedence(t_p, t_h),
        opinion_drive=contributions.opinion_drive,
        rewiring_drive=contributions.rewiring_drive,
        gamma=contributions.gamma_integrated,
        gamma_0p10=sensitivity[0.10].gamma_integrated,
        gamma_0p40=sensitivity[0.40].gamma_integrated,
        gamma_initial=float(contributions.gamma[0]),
        gamma_at_window=contributions.gamma_at_window,
        window_end=contributions.window_end,
    )


def _log_predictor(values: np.ndarray) -> np.ndarray:
    return np.log10(np.clip(np.asarray(values, dtype=float), 1e-12, 1e12))


def _binary_rows(
    results: list[TimescaleResult], predictor: str
) -> tuple[np.ndarray, np.ndarray, list[TimescaleResult]]:
    def predictor_value(result: TimescaleResult) -> float:
        return (
            result.q / result.alpha
            if predictor == "rate_ratio"
            else float(getattr(result, predictor))
        )

    selected = [
        result
        for result in results
        if result.path in {"PbS", "SbP"} and not np.isnan(predictor_value(result))
    ]
    values = _log_predictor(
        np.asarray([predictor_value(result) for result in selected])
    )
    labels = np.asarray([result.path == "SbP" for result in selected], dtype=bool)
    return values, labels, selected


def _auc(values: np.ndarray, labels: np.ndarray) -> float:
    positives = int(labels.sum())
    negatives = int((~labels).sum())
    if positives == 0 or negatives == 0:
        return float("nan")
    ranks = rankdata(values)
    statistic = float(ranks[labels].sum()) - positives * (positives + 1) / 2
    return statistic / (positives * negatives)


def _score_predictions(
    labels: np.ndarray, predictions: np.ndarray
) -> tuple[float, float]:
    if labels.size == 0:
        return float("nan"), float("nan")
    accuracy = float(np.mean(labels == predictions))
    recalls = []
    for category in (False, True):
        mask = labels == category
        if np.any(mask):
            recalls.append(float(np.mean(predictions[mask] == category)))
    return accuracy, float(np.mean(recalls))


def optimal_threshold(values: np.ndarray, labels: np.ndarray) -> ThresholdScore:
    """Fit a threshold in log space; larger values predict SbP."""

    values = np.asarray(values, dtype=float)
    labels = np.asarray(labels, dtype=bool)
    if values.size == 0 or np.all(labels == labels[0]):
        return ThresholdScore(
            float("nan"),
            float("nan"),
            float("nan"),
            _auc(values, labels),
            int(values.size),
        )
    unique = np.unique(values)
    candidates = np.concatenate(
        (
            [unique[0] - 1.0],
            0.5 * (unique[:-1] + unique[1:]),
            [unique[-1] + 1.0],
        )
    )
    best: tuple[float, float, float] | None = None
    for threshold in candidates:
        accuracy, balanced = _score_predictions(labels, values >= threshold)
        score = (balanced, accuracy, -abs(float(threshold)))
        if best is None or score > best:
            best = score
            selected_threshold = float(threshold)
    assert best is not None
    return ThresholdScore(
        threshold=10.0**selected_threshold,
        accuracy=best[1],
        balanced_accuracy=best[0],
        auc=_auc(values, labels),
        count=int(values.size),
    )


def _leave_one_configuration_out(
    results: list[TimescaleResult],
    predictor: str,
    configurations: tuple[tuple[str, str, float], ...],
) -> tuple[float, float]:
    predictions: list[bool] = []
    labels: list[bool] = []
    for configuration, _, _ in configurations:
        train = [result for result in results if result.configuration != configuration]
        test = [result for result in results if result.configuration == configuration]
        train_values, train_labels, _ = _binary_rows(train, predictor)
        score = optimal_threshold(train_values, train_labels)
        test_values, test_labels, _ = _binary_rows(test, predictor)
        if not np.isfinite(score.threshold) or test_values.size == 0:
            continue
        predictions.extend(test_values >= math.log10(score.threshold))
        labels.extend(test_labels)
    return _score_predictions(
        np.asarray(labels, dtype=bool), np.asarray(predictions, dtype=bool)
    )


def analyze_predictors(
    results: list[TimescaleResult],
    configurations: tuple[tuple[str, str, float], ...] | None = None,
) -> tuple[dict[str, dict[str, float | int]], list[dict[str, object]]]:
    if configurations is None:
        configurations = _configurations_in_results(results)
    metrics: dict[str, dict[str, float | int]] = {}
    threshold_rows: list[dict[str, object]] = []
    for predictor in (
        "rate_ratio",
        "gamma_initial",
        "gamma_at_window",
        "gamma_0p10",
        "gamma_0p40",
        "gamma",
    ):
        values, labels, selected = _binary_rows(results, predictor)
        score = optimal_threshold(values, labels)
        loco_accuracy, loco_balanced = _leave_one_configuration_out(
            results, predictor, configurations
        )
        pathway_values = np.asarray([result.pathway for result in selected])
        correlation = (
            float(spearmanr(values, pathway_values).statistic)
            if values.size > 1 and np.ptp(values) > 0 and np.ptp(pathway_values) > 0
            else float("nan")
        )
        per_configuration_logs = []
        for configuration, _, _ in configurations:
            subset = [
                result for result in results if result.configuration == configuration
            ]
            subset_values, subset_labels, _ = _binary_rows(subset, predictor)
            subset_score = optimal_threshold(subset_values, subset_labels)
            threshold_rows.append(
                {
                    "predictor": predictor,
                    "configuration": configuration,
                    "threshold": subset_score.threshold,
                    "log10_threshold": (
                        math.log10(subset_score.threshold)
                        if np.isfinite(subset_score.threshold)
                        and subset_score.threshold > 0
                        else float("nan")
                    ),
                    "balanced_accuracy": subset_score.balanced_accuracy,
                    "count": subset_score.count,
                }
            )
            if np.isfinite(subset_score.threshold) and subset_score.threshold > 0:
                per_configuration_logs.append(math.log10(subset_score.threshold))
        metrics[predictor] = {
            "threshold": score.threshold,
            "accuracy": score.accuracy,
            "balanced_accuracy": score.balanced_accuracy,
            "auc": score.auc,
            "count": score.count,
            "loco_accuracy": loco_accuracy,
            "loco_balanced_accuracy": loco_balanced,
            "spearman_I_w": correlation,
            "configuration_threshold_log10_std": (
                float(np.std(per_configuration_logs))
                if per_configuration_logs
                else float("nan")
            ),
            "configuration_threshold_log10_range": (
                float(np.ptp(per_configuration_logs))
                if per_configuration_logs
                else float("nan")
            ),
        }
    return metrics, threshold_rows


def fit_transition_offsets(
    results: list[TimescaleResult],
    predictor: str,
    configurations: tuple[tuple[str, str, float], ...] | None = None,
) -> tuple[dict[str, float | int | bool], list[dict[str, object]]]:
    """Fit one transition-center scalar per configuration.

    The binary pathway labels follow a common-slope logistic model with one
    intercept per recommendation configuration.  Centering the inferred
    transition locations gives the horizontal shift required to align each
    configuration with the equal-configuration master curve.
    """

    if configurations is None:
        configurations = _configurations_in_results(results)
    values, labels, selected = _binary_rows(results, predictor)
    names = tuple(item[0] for item in configurations)
    lookup = {name: index for index, name in enumerate(names)}
    retained = [
        index for index, result in enumerate(selected) if result.configuration in lookup
    ]
    values = values[retained]
    labels = labels[retained].astype(float)
    selected = [selected[index] for index in retained]
    if not names or values.size == 0:
        raise ValueError("transition-offset fit has no resolved cells")

    design = np.zeros((values.size, len(names) + 1), dtype=float)
    group_index = np.asarray(
        [lookup[result.configuration] for result in selected], dtype=int
    )
    design[np.arange(values.size), group_index] = 1.0
    design[:, -1] = values

    slope_initial = 3.0
    initial = np.zeros(len(names) + 1, dtype=float)
    for index in range(len(names)):
        mask = group_index == index
        probability = float(np.clip(np.mean(labels[mask]), 1e-3, 1 - 1e-3))
        initial[index] = math.log(
            probability / (1.0 - probability)
        ) - slope_initial * float(np.mean(values[mask]))
    initial[-1] = slope_initial

    def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        linear = design @ parameters
        value = float(np.sum(np.logaddexp(0.0, linear) - labels * linear))
        gradient = design.T @ (expit(linear) - labels)
        return value, gradient

    fit = minimize(
        objective,
        initial,
        method="BFGS",
        jac=True,
        options={"gtol": 1e-8, "maxiter": 2000},
    )
    parameters = np.asarray(fit.x, dtype=float)
    gradient_norm = float(np.max(np.abs(objective(parameters)[1])))
    slope = float(parameters[-1])
    if not np.isfinite(slope) or slope <= 0 or gradient_norm > 1e-4:
        raise RuntimeError(
            "common-slope transition fit failed: "
            f"success={fit.success}, slope={slope}, gradient={gradient_norm}"
        )

    linear = design @ parameters
    weights = expit(linear) * (1.0 - expit(linear))
    hessian = design.T @ (weights[:, None] * design)
    covariance = np.linalg.pinv(hessian, hermitian=True)
    intercepts = parameters[:-1]
    centers = -intercepts / slope
    global_center = float(np.mean(centers))
    offsets = centers - global_center
    rows: list[dict[str, object]] = []
    for index, (configuration, _, _) in enumerate(configurations):
        derivative = np.zeros(len(names) + 1, dtype=float)
        derivative[:-1] = 1.0 / (len(names) * slope)
        derivative[index] -= 1.0 / slope
        derivative[-1] = -offsets[index] / slope
        standard_error = float(
            np.sqrt(max(float(derivative @ covariance @ derivative), 0.0))
        )
        rows.append(
            {
                "predictor": predictor,
                "configuration": configuration,
                "transition_log10": float(centers[index]),
                "transition": float(10.0 ** centers[index]),
                "offset_log10": float(offsets[index]),
                "offset_standard_error": standard_error,
                "offset_lower_95": float(offsets[index] - 1.96 * standard_error),
                "offset_upper_95": float(offsets[index] + 1.96 * standard_error),
                "count": int(np.sum(group_index == index)),
            }
        )
    metrics: dict[str, float | int | bool] = {
        "common_slope": slope,
        "equal_configuration_center_log10": global_center,
        "equal_configuration_center": float(10.0**global_center),
        "offset_log10_range": float(np.ptp(offsets)),
        "offset_log10_std": float(np.std(offsets)),
        "negative_log_likelihood": float(fit.fun),
        "count": int(values.size),
        "converged": bool(fit.success or gradient_norm <= 1e-4),
        "gradient_infinity_norm": gradient_norm,
    }
    return metrics, rows


def _write_transition_offsets(
    results: list[TimescaleResult],
    configurations: tuple[tuple[str, str, float], ...],
    output_dir: Path,
) -> tuple[dict[str, dict[str, object]], list[dict[str, object]]]:
    metrics: dict[str, dict[str, object]] = {}
    rows: list[dict[str, object]] = []
    for predictor in ("rate_ratio", "gamma_initial"):
        try:
            predictor_metrics, predictor_rows = fit_transition_offsets(
                results, predictor, configurations
            )
            predictor_metrics["available"] = True
        except (ValueError, RuntimeError) as error:
            predictor_metrics = {
                "available": False,
                "reason": str(error),
            }
            predictor_rows = []
        metrics[predictor] = predictor_metrics
        rows.extend(predictor_rows)
    (output_dir / "transition_offset_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (output_dir / "transition_offsets.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        fieldnames = [
            "predictor",
            "configuration",
            "transition_log10",
            "transition",
            "offset_log10",
            "offset_standard_error",
            "offset_lower_95",
            "offset_upper_95",
            "count",
        ]
        writer = csv.DictWriter(
            stream,
            fieldnames=(list(rows[0]) if rows else fieldnames),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    return metrics, rows


def _result_row(result: TimescaleResult) -> dict[str, object]:
    return {
        **asdict(result),
        "I_w": result.pathway,
        "t_Ip_0.5": result.t_polarization,
        "t_Ih_0.5": result.t_homophily,
        "rate_ratio": result.q / result.alpha,
        "log10_rate_ratio": math.log10(result.q / result.alpha),
        "log10_gamma": math.log10(np.clip(result.gamma, 1e-12, 1e12)),
    }


def _write_outputs(
    results: list[TimescaleResult],
    rates: np.ndarray,
    configurations: tuple[tuple[str, str, float], ...],
    output_dir: Path,
) -> tuple[dict[str, dict[str, float | int]], list[dict[str, object]]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [_result_row(result) for result in results]
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    metrics, threshold_rows = analyze_predictors(results, configurations)
    (output_dir / "collapse_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (output_dir / "configuration_thresholds.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(threshold_rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(threshold_rows)
    controlled_results = [
        result for result in results if result.alpha < 1.0 and result.q < 1.0
    ]
    controlled_metrics, controlled_threshold_rows = analyze_predictors(
        controlled_results, configurations
    )
    (output_dir / "collapse_metrics_rates_below_one.json").write_text(
        json.dumps(controlled_metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "configuration_thresholds_rates_below_one.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(controlled_threshold_rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(controlled_threshold_rows)

    shape = (len(configurations), rates.size, rates.size)
    fields = {
        name: np.full(shape, np.nan)
        for name in (
            "I_w",
            "t_Ip_0.5",
            "t_Ih_0.5",
            "precedence",
            "opinion_drive",
            "rewiring_drive",
            "gamma",
            "gamma_0p10",
            "gamma_0p40",
            "gamma_initial",
            "gamma_at_window",
            "window_end",
        )
    }
    configuration_index = {
        configuration[0]: index for index, configuration in enumerate(configurations)
    }
    rate_index = {float(rate): index for index, rate in enumerate(rates)}
    for result in results:
        index = (
            configuration_index[result.configuration],
            rate_index[result.q],
            rate_index[result.alpha],
        )
        fields["I_w"][index] = result.pathway
        fields["t_Ip_0.5"][index] = result.t_polarization
        fields["t_Ih_0.5"][index] = result.t_homophily
        fields["precedence"][index] = result.precedence
        fields["opinion_drive"][index] = result.opinion_drive
        fields["rewiring_drive"][index] = result.rewiring_drive
        fields["gamma"][index] = result.gamma
        fields["gamma_0p10"][index] = result.gamma_0p10
        fields["gamma_0p40"][index] = result.gamma_0p40
        fields["gamma_initial"][index] = result.gamma_initial
        fields["gamma_at_window"][index] = result.gamma_at_window
        fields["window_end"][index] = result.window_end
    np.savez_compressed(
        output_dir / "timescale_ratio.npz",
        rates=rates,
        configuration=np.asarray([item[0] for item in configurations]),
        **fields,
    )
    return metrics, threshold_rows


def _plot_collapse(
    results: list[TimescaleResult],
    metrics: dict[str, dict[str, float | int]],
    configurations: tuple[tuple[str, str, float], ...],
    output_dir: Path,
) -> None:
    setup_paper_params()
    macro_predictors = (
        "gamma_initial",
        "gamma_at_window",
        "gamma_0p10",
        "gamma",
        "gamma_0p40",
    )
    macro_predictor = max(
        macro_predictors,
        key=lambda name: (
            float(metrics[name]["loco_balanced_accuracy"]),
            float(metrics[name]["balanced_accuracy"]),
        ),
    )
    macro_label = {
        "gamma_initial": r"$\log_{10}\Gamma(t=0)$",
        "gamma_at_window": r"$\log_{10}\Gamma(I_p+I_h=0.25)$",
        "gamma_0p10": r"$\log_{10}\Gamma_{0.10}^{\rm int}$",
        "gamma": r"$\log_{10}\Gamma_{0.25}^{\rm int}$",
        "gamma_0p40": r"$\log_{10}\Gamma_{0.40}^{\rm int}$",
    }[macro_predictor]
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.1), sharey=True)
    colors = plt.get_cmap("tab10")
    for index, (configuration, _, _) in enumerate(configurations):
        subset = [result for result in results if result.configuration == configuration]
        if not subset:
            continue
        y = np.asarray([result.pathway for result in subset])
        raw = _log_predictor(np.asarray([result.q / result.alpha for result in subset]))
        gamma = _log_predictor(
            np.asarray([float(getattr(result, macro_predictor)) for result in subset])
        )
        label = DISPLAY_NAMES[configuration]
        axes[0].scatter(raw, y, s=15, alpha=0.65, color=colors(index), label=label)
        axes[1].scatter(gamma, y, s=15, alpha=0.65, color=colors(index), label=label)
    for axis, predictor, xlabel in (
        (axes[0], "rate_ratio", r"$\log_{10}(q/\alpha)$"),
        (axes[1], macro_predictor, macro_label),
    ):
        threshold = float(metrics[predictor]["threshold"])
        if np.isfinite(threshold) and threshold > 0:
            axis.axvline(math.log10(threshold), color="black", linestyle="--")
        axis.set_xlabel(xlabel)
        axis.grid(alpha=0.2)
        axis.set_title(
            rf"BA={float(metrics[predictor]['balanced_accuracy']):.3f}, "
            rf"LOCO={float(metrics[predictor]['loco_balanced_accuracy']):.3f}"
        )
    axes[0].set_ylabel(r"pathway index $I_w$")
    axes[1].legend(frameon=False, fontsize=7, loc="best")
    figure.tight_layout()
    for suffix in ("pdf", "png"):
        figure.savefig(output_dir / f"timescale_collapse.{suffix}", dpi=300)
    plt.close(figure)


def _plot_gamma_heatmaps(
    results: list[TimescaleResult],
    rates: np.ndarray,
    configurations: tuple[tuple[str, str, float], ...],
    output_dir: Path,
) -> None:
    setup_paper_params()
    figure, axes = plt.subplots(
        1,
        len(configurations),
        figsize=(2.9 * len(configurations) + 2.0, 3.5),
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)
    finite_logs = _log_predictor(
        np.asarray([result.gamma for result in results if not np.isnan(result.gamma)])
    )
    limit = max(float(np.max(np.abs(finite_logs))), 1.0)
    image = None
    for axis, (configuration, _, _) in zip(axes, configurations, strict=True):
        matrix = np.full((rates.size, rates.size), np.nan)
        labels = np.full((rates.size, rates.size), "", dtype=object)
        lookup = {float(value): index for index, value in enumerate(rates)}
        for result in results:
            if result.configuration != configuration:
                continue
            index = lookup[result.q], lookup[result.alpha]
            matrix[index] = _log_predictor(np.asarray([result.gamma]))[0]
            labels[index] = "P" if result.path == "PbS" else "S"
        image = axis.imshow(
            matrix,
            origin="lower",
            cmap="coolwarm",
            norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
        )
        for q_index in range(rates.size):
            for alpha_index in range(rates.size):
                axis.text(
                    alpha_index,
                    q_index,
                    labels[q_index, alpha_index],
                    ha="center",
                    va="center",
                    fontsize=5.5,
                    color="black",
                )
        rate_labels = [f"{value:.2g}" for value in rates]
        axis.set_xticks(np.arange(rates.size), rate_labels, rotation=90, fontsize=6)
        axis.set_yticks(np.arange(rates.size), rate_labels, fontsize=6)
        axis.set_xlabel(r"$\alpha$")
        axis.set_title(DISPLAY_NAMES[configuration], fontsize=8)
    axes[0].set_ylabel(r"rewiring $q$")
    assert image is not None
    figure.colorbar(
        image,
        ax=axes.ravel().tolist(),
        label=r"$\log_{10}\Gamma_{0.25}$",
        fraction=0.018,
        pad=0.015,
        shrink=0.8,
    )
    for suffix in ("pdf", "png"):
        figure.savefig(
            output_dir / f"gamma_pathway_heatmaps.{suffix}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(figure)


def _plot_transition_offsets(
    offset_metrics: dict[str, dict[str, object]],
    offset_rows: list[dict[str, object]],
    configurations: tuple[tuple[str, str, float], ...],
    output_dir: Path,
) -> None:
    """Plot each recommendation cloud as one fitted horizontal shift."""

    setup_paper_params()
    figure, axis = plt.subplots(
        figsize=(8.2, 0.62 * len(configurations) + 1.8),
        constrained_layout=True,
    )
    positions = np.arange(len(configurations), dtype=float)
    styles = {
        "rate_ratio": ("tab:blue", "o", -0.12, r"raw $q/\alpha$"),
        "gamma_initial": (
            "tab:red",
            "s",
            0.12,
            r"macro $\Gamma(t=0)$",
        ),
    }
    plotted_values: dict[str, np.ndarray] = {}
    for predictor, (color, marker, displacement, label) in styles.items():
        lookup = {
            str(row["configuration"]): row
            for row in offset_rows
            if row["predictor"] == predictor
        }
        values = np.asarray(
            [float(lookup[item[0]]["offset_log10"]) for item in configurations]
        )
        plotted_values[predictor] = values
        errors = np.asarray(
            [
                1.96 * float(lookup[item[0]]["offset_standard_error"])
                for item in configurations
            ]
        )
        spread = float(offset_metrics[predictor]["offset_log10_range"])
        axis.errorbar(
            values,
            positions + displacement,
            xerr=errors,
            color=color,
            marker=marker,
            linestyle="none",
            markersize=5.5,
            capsize=2.5,
            linewidth=1.0,
            label=f"{label}, range={spread:.3f}",
        )
    for index in range(len(configurations)):
        axis.plot(
            [
                plotted_values["rate_ratio"][index],
                plotted_values["gamma_initial"][index],
            ],
            [positions[index] - 0.12, positions[index] + 0.12],
            color="0.75",
            linewidth=0.8,
            zorder=0,
        )
    axis.axvline(0.0, color="black", linewidth=0.9, linestyle="--")
    axis.set_yticks(
        positions,
        [DISPLAY_NAMES[item[0]] for item in configurations],
        fontsize=8,
    )
    axis.invert_yaxis()
    axis.set_xlabel(r"configuration transition offset $\delta_c$ (log$_{10}$ decades)")
    axis.set_title("Configuration-specific pathway-transition offsets")
    axis.grid(axis="x", alpha=0.2)
    axis.legend(
        frameon=False,
        fontsize=8,
        loc="best",
        title="common-slope logistic fit (95% intervals)",
        title_fontsize=7,
    )
    for suffix in ("pdf", "png"):
        figure.savefig(
            output_dir / f"transition_offsets.{suffix}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(figure)


def _load_results(path: Path) -> list[TimescaleResult]:
    """Load previously completed cells for an incremental scan."""

    results: list[TimescaleResult] = []
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            results.append(
                TimescaleResult(
                    configuration=row["configuration"],
                    recsys=row["recsys"],
                    steepness=float(row["steepness"]),
                    alpha=float(row["alpha"]),
                    q=float(row["q"]),
                    path=row["path"],
                    pathway=float(row["pathway"]),
                    t_polarization=float(row["t_polarization"]),
                    t_homophily=float(row["t_homophily"]),
                    precedence=float(row["precedence"]),
                    opinion_drive=float(row["opinion_drive"]),
                    rewiring_drive=float(row["rewiring_drive"]),
                    gamma=float(row["gamma"]),
                    gamma_0p10=float(row["gamma_0p10"]),
                    gamma_0p40=float(row["gamma_0p40"]),
                    gamma_initial=float(row["gamma_initial"]),
                    gamma_at_window=float(row["gamma_at_window"]),
                    window_end=float(row["window_end"]),
                )
            )
    return results


def _validate_resume_metadata(
    path: Path,
    base: KineticParameters,
    schedule: tuple[int, ...],
    *,
    probe_dt: float,
    progress_threshold: float,
) -> None:
    """Reject incremental reuse across incompatible numerical protocols."""

    metadata_path = path / "run_metadata.json"
    if not metadata_path.exists():
        raise ValueError("--resume requires output-dir/run_metadata.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    recorded_parameters = metadata.get("parameters", {})
    variable_fields = {
        "influence",
        "rewiring",
        "recsys",
        "recommendation_steepness",
    }
    mismatches = [
        name
        for name, expected in asdict(base).items()
        if name not in variable_fields and recorded_parameters.get(name) != expected
    ]
    configuration = metadata.get("configuration", {})
    if configuration.get("record_steps") != list(schedule):
        mismatches.append("record_steps")
    if configuration.get("probe_dt") != probe_dt:
        mismatches.append("probe_dt")
    if configuration.get("progress_threshold") != progress_threshold:
        mismatches.append("progress_threshold")
    if mismatches:
        raise ValueError(
            "--resume metadata is incompatible for: "
            + ", ".join(sorted(set(mismatches)))
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/archive/macroscopic_timescale_ratio_b41"),
    )
    parser.add_argument("--grid-size", type=int, default=41)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--early-until", type=int, default=200)
    parser.add_argument("--early-every", type=int, default=1)
    parser.add_argument("--late-every", type=int, default=20)
    parser.add_argument("--probe-dt", type=float, default=1.0)
    parser.add_argument("--progress-threshold", type=float, default=0.25)
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--noise", type=float, default=1e-5)
    parser.add_argument("--rates", type=float, nargs="+", default=RATES.tolist())
    parser.add_argument(
        "--configurations",
        nargs="+",
        choices=[item[0] for item in CONFIGURATIONS],
        default=[item[0] for item in CONFIGURATIONS],
    )
    parser.add_argument("--jobs", type=int, default=min(os.cpu_count() or 1, 4))
    parser.add_argument(
        "--resume",
        action="store_true",
        help="reuse matching cells already present in output-dir/summary.csv",
    )
    parser.add_argument("--skip-plots", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    rates = np.asarray(sorted(set(args.rates)), dtype=float)
    if np.any(rates <= 0) or np.any(rates > 1):
        raise ValueError("rates must lie in (0, 1]")
    selected_configurations = tuple(
        item for item in CONFIGURATIONS if item[0] in set(args.configurations)
    )
    schedule = record_schedule(
        args.steps,
        early_until=args.early_until,
        early_every=args.early_every,
        late_every=args.late_every,
    )
    base = KineticParameters(
        epsilon=0.45,
        mean_degree=15,
        recsys_count=10,
        recommendation_random_ratio=0.0,
        noise_diffusion=args.noise,
        grid_size=args.grid_size,
        dt=args.dt,
        steps=args.steps,
        record_every=args.late_every,
    )
    output_dir = args.output_dir.expanduser().resolve()
    results: list[TimescaleResult] = []
    if args.resume and (output_dir / "summary.csv").exists():
        _validate_resume_metadata(
            output_dir,
            base,
            schedule,
            probe_dt=args.probe_dt,
            progress_threshold=args.progress_threshold,
        )
        available_rates = {float(value) for value in rates}
        selected_names = {item[0] for item in selected_configurations}
        results = [
            result
            for result in _load_results(output_dir / "summary.csv")
            if result.configuration in selected_names
            and result.alpha in available_rates
            and result.q in available_rates
        ]
        print(f"reusing {len(results)} completed cells", flush=True)
    completed = {(result.configuration, result.alpha, result.q) for result in results}
    cases = [
        (
            configuration,
            recsys,
            steepness,
            float(alpha),
            float(q),
            base,
            schedule,
            args.probe_dt,
            args.progress_threshold,
        )
        for configuration, recsys, steepness in selected_configurations
        for q in rates
        for alpha in rates
        if (configuration, float(alpha), float(q)) not in completed
    ]
    if args.jobs == 1:
        iterator = (_solve_case(*case) for case in cases)
        for count, result in enumerate(iterator, start=1):
            results.append(result)
            print(
                f"[{count:03d}/{len(cases)}] {result.configuration} "
                f"alpha={result.alpha:g}, q={result.q:g}, "
                f"path={result.path}, Gamma={result.gamma:.3g}",
                flush=True,
            )
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as executor:
            futures = [executor.submit(_solve_case, *case) for case in cases]
            for count, future in enumerate(as_completed(futures), start=1):
                result = future.result()
                results.append(result)
                print(
                    f"[{count:03d}/{len(cases)}] {result.configuration} "
                    f"alpha={result.alpha:g}, q={result.q:g}, "
                    f"path={result.path}, Gamma={result.gamma:.3g}",
                    flush=True,
                )
    results.sort(key=lambda item: (item.configuration, item.q, item.alpha))
    metrics, _ = _write_outputs(results, rates, selected_configurations, output_dir)
    offset_metrics, offset_rows = _write_transition_offsets(
        results, selected_configurations, output_dir
    )
    if not args.skip_plots:
        _plot_collapse(results, metrics, selected_configurations, output_dir)
        _plot_gamma_heatmaps(results, rates, selected_configurations, output_dir)
        if offset_rows:
            _plot_transition_offsets(
                offset_metrics,
                offset_rows,
                selected_configurations,
                output_dir,
            )
    write_run_metadata(
        output_dir / "run_metadata.json",
        analysis="operator-resolved macroscopic time-scale ratio collapse",
        command=shlex.join(
            [
                sys.executable,
                "-m",
                "experiments.archive.macroscopic_timescale_ratio_b41.run",
                *sys.argv[1:],
            ]
        ),
        parameters=asdict(base),
        configuration={
            "rates": rates.tolist(),
            "configurations": [list(item) for item in selected_configurations],
            "record_steps": list(schedule),
            "probe_dt": args.probe_dt,
            "progress_threshold": args.progress_threshold,
            "opinion_counterfactual_diffusion": 0.0,
            "l1_higher_steepness_closure": "C * E[S]^zeta from retained M1",
            "reused_cases": len(completed),
            "jobs": args.jobs,
        },
    )
    print(f"wrote macroscopic time-scale experiment to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
