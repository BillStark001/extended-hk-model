"""Fit continuous-pathway transition offsets and plot a completed scan."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from scipy.optimize import minimize
from scipy.special import expit, logit

from ehk.common.plotting import setup_paper_params
from experiments.theory_guided.macroscopic_timescale_ratio.run import (
  TimescaleCell,
)
from experiments.theory_guided.macroscopic_timescale_ratio.scenarios import (
  SCENARIO_BY_KEY,
  RecommendationScenario,
)
from theory.paths import MESOSCOPIC_OUTPUT

PREDICTORS = ("rate_ratio", "gamma_0")
PREDICTOR_LABELS = {
    "rate_ratio": r"raw $q/\alpha$",
    "gamma_0": r"$\Gamma(0)$",
}


def load_cells(path: Path) -> list[TimescaleCell]:
  """Load the canonical summary while ignoring derived convenience columns."""

  cells = []
  with path.open(newline="", encoding="utf-8") as stream:
    for row in csv.DictReader(stream):
      cells.append(
          TimescaleCell(
              configuration=row["configuration"],
              recsys=row["recsys"],
              steepness=float(row["steepness"]),
              closure_note=row["closure_note"],
              alpha_index=int(row["alpha_index"]),
              q_index=int(row["q_index"]),
              alpha=float(row["alpha"]),
              q=float(row["q"]),
              path=row["path"],
              pathway=float(row["pathway"]),
              t_polarization=float(row["t_polarization"]),
              t_homophily=float(row["t_homophily"]),
              precedence=float(row["precedence"]),
              polarization_final=float(row["polarization_final"]),
              homophily_final=float(row["homophily_final"]),
              subjective_final=float(row["subjective_final"]),
              opinion_rate_gamma_0=float(row["opinion_rate_gamma_0"]),
              rewiring_rate_gamma_0=float(row["rewiring_rate_gamma_0"]),
              gamma_0=float(row["gamma_0"]),
          )
      )
  return cells


def _predictor_value(cell: TimescaleCell, predictor: str) -> float:
  if predictor == "rate_ratio":
    return cell.rate_ratio
  if predictor == "gamma_0":
    return cell.gamma_0
  raise KeyError(predictor)


def _log_value(value: float) -> float:
  return float(np.log10(np.clip(value, 1e-12, 1e12)))


def fit_transition_offsets(
    cells: list[TimescaleCell],
    predictor: str,
    scenarios: tuple[RecommendationScenario, ...],
) -> tuple[dict[str, object], list[dict[str, object]]]:
  """Fit one centered transition location per recommendation scenario."""

  names = tuple(scenario.key for scenario in scenarios)
  group_lookup = {name: index for index, name in enumerate(names)}
  selected = [
      cell
      for cell in cells
      if cell.configuration in group_lookup
      and np.isfinite(cell.pathway)
      and not np.isnan(_predictor_value(cell, predictor))
  ]
  if not selected:
    raise ValueError(f"{predictor}: no resolved cells")
  values = np.asarray(
      [_log_value(_predictor_value(cell, predictor)) for cell in selected]
  )
  labels = np.asarray([cell.pathway for cell in selected], dtype=float)
  if np.any((labels < 0.0) | (labels > 1.0)):
    raise ValueError(f"{predictor}: pathway response lies outside [0, 1]")
  groups = np.asarray(
      [group_lookup[cell.configuration] for cell in selected], dtype=int
  )
  design = np.zeros((len(selected), len(names) + 1), dtype=float)
  design[np.arange(len(selected)), groups] = 1.0
  design[:, -1] = values

  slope_initial = 3.0
  initial = np.zeros(len(names) + 1, dtype=float)
  for group in range(len(names)):
    mask = groups == group
    if not np.any(mask):
      raise ValueError(
          f"{predictor}: scenario {names[group]} has no resolved cells"
      )
    probability = float(np.clip(np.mean(labels[mask]), 1e-3, 1 - 1e-3))
    initial[group] = math.log(probability / (1.0 - probability))
    initial[group] -= slope_initial * float(np.mean(values[mask]))
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
        f"{predictor}: common-slope fit failed "
        f"(success={fit.success}, slope={slope}, gradient={gradient_norm})"
    )

  probability = expit(design @ parameters)
  weights = probability * (1.0 - probability)
  hessian = design.T @ (weights[:, None] * design)
  covariance = np.linalg.pinv(hessian, hermitian=True)
  centers = (logit(0.6) - parameters[:-1]) / slope
  equal_center = float(np.mean(centers))
  offsets = centers - equal_center
  rows = []
  for index, scenario in enumerate(scenarios):
    derivative = np.full(len(names) + 1, 0.0)
    derivative[:-1] = 1.0 / (len(names) * slope)
    derivative[index] -= 1.0 / slope
    derivative[-1] = -offsets[index] / slope
    standard_error = float(
        np.sqrt(max(float(derivative @ covariance @ derivative), 0.0))
    )
    rows.append(
        {
            "predictor": predictor,
            "configuration": scenario.key,
            "transition_log10": float(centers[index]),
            "transition": float(10.0 ** centers[index]),
            "offset_log10": float(offsets[index]),
            "offset_standard_error": standard_error,
            "offset_lower_95": float(offsets[index] - 1.96 * standard_error),
            "offset_upper_95": float(offsets[index] + 1.96 * standard_error),
            "count": int(np.sum(groups == index)),
        }
    )
  metrics = {
      "available": True,
      "common_slope": slope,
      "equal_configuration_center_log10": equal_center,
      "equal_configuration_center": float(10.0**equal_center),
      "offset_log10_range": float(np.ptp(offsets)),
      "offset_log10_std": float(np.std(offsets)),
      "negative_log_likelihood": float(fit.fun),
      "count": len(selected),
      "converged": bool(fit.success or gradient_norm <= 1e-4),
      "gradient_infinity_norm": gradient_norm,
      "response": "continuous_I_w",
      "transition_level": 0.6,
  }
  return metrics, rows


def _write_offset_tables(
    cells: list[TimescaleCell],
    scenarios: tuple[RecommendationScenario, ...],
    output_dir: Path,
    *,
    suffix: str = "",
) -> tuple[dict[str, dict[str, object]], list[dict[str, object]]]:
  metrics: dict[str, dict[str, object]] = {}
  rows: list[dict[str, object]] = []
  for predictor in PREDICTORS:
    try:
      predictor_metrics, predictor_rows = fit_transition_offsets(
          cells, predictor, scenarios
      )
    except (ValueError, RuntimeError) as error:
      predictor_metrics = {"available": False, "reason": str(error)}
      predictor_rows = []
    metrics[predictor] = predictor_metrics
    rows.extend(predictor_rows)
  (output_dir / f"transition_offset_metrics{suffix}.json").write_text(
      json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
  )
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
  with (output_dir / f"transition_offsets{suffix}.csv").open(
      "w", newline="", encoding="utf-8"
  ) as stream:
    writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
  return metrics, rows


def _plot_pathway_heatmaps(
    cells: list[TimescaleCell],
    rates: np.ndarray,
    scenarios: tuple[RecommendationScenario, ...],
    output_dir: Path,
) -> None:
  setup_paper_params()
  figure, axes = plt.subplots(
      1,
      len(scenarios),
      figsize=(2.45 * len(scenarios) + 1.3, 3.35),
      constrained_layout=False,
  )
  axes = np.atleast_1d(axes)
  figure.subplots_adjust(
      left=0.055,
      right=0.945,
      bottom=0.24,
      top=0.86,
      wspace=0.16,
  )
  lookup = {
      (cell.configuration, cell.q_index, cell.alpha_index): cell
      for cell in cells
  }
  image = None
  rate_labels = [f"{rate:.2g}" for rate in rates]
  for column, (axis, scenario) in enumerate(
      zip(axes, scenarios, strict=True)
  ):
    values = np.asarray(
        [
            [lookup[scenario.key, q_index, alpha_index].pathway
             for alpha_index in range(rates.size)]
            for q_index in range(rates.size)
        ]
    )
    image = axis.imshow(
        values,
        origin="lower",
        aspect="equal",
        cmap="RdYlBu_r",
        vmin=0.0,
        vmax=1.0,
    )
    axis.set_xticks(np.arange(rates.size), rate_labels, rotation=90, fontsize=6)
    axis.set_yticks(
        np.arange(rates.size), rate_labels if column == 0 else [], fontsize=6
    )
    axis.tick_params(length=2)
    axis.set_xlabel(r"influence $\alpha$", fontsize=8)
    axis.set_title(scenario.display_name, fontsize=8)
    axis.add_patch(
        Rectangle(
            (rates.size - 1.5, -0.5),
            1,
            rates.size,
            fill=False,
            hatch="////",
            edgecolor="0.25",
            linewidth=0.45,
            alpha=0.45,
        )
    )
    axis.add_patch(
        Rectangle(
            (-0.5, rates.size - 1.5),
            rates.size,
            1,
            fill=False,
            hatch="\\\\\\\\",
            edgecolor="0.25",
            linewidth=0.45,
            alpha=0.45,
        )
    )
  axes[0].set_ylabel(r"rewiring $q$", fontsize=8)
  assert image is not None
  color_axis = figure.add_axes((0.96, 0.24, 0.008, 0.62))
  figure.colorbar(image, cax=color_axis, label=r"pathway index $I_w$")
  for suffix in ("pdf", "png"):
    figure.savefig(
        output_dir / f"f_pathway_seven_configurations.{suffix}",
        dpi=300,
        bbox_inches="tight",
    )
  plt.close(figure)


def _plot_transition_offsets(
    metrics: dict[str, dict[str, object]],
    rows: list[dict[str, object]],
    scenarios: tuple[RecommendationScenario, ...],
    output_dir: Path,
) -> None:
  setup_paper_params()
  figure, axis = plt.subplots(
      figsize=(8.6, 0.64 * len(scenarios) + 1.7), constrained_layout=True
  )
  positions = np.arange(len(scenarios), dtype=float)
  styles = {
      "rate_ratio": ("tab:blue", "o", -0.19),
      "gamma_0": ("tab:orange", "s", 0.0),
  }
  plotted: dict[str, np.ndarray] = {}
  for predictor in PREDICTORS:
    color, marker, displacement = styles[predictor]
    lookup = {
        str(row["configuration"]): row
        for row in rows
        if row["predictor"] == predictor
    }
    if len(lookup) != len(scenarios):
      continue
    values = np.asarray(
        [float(lookup[scenario.key]["offset_log10"]) for scenario in scenarios]
    )
    errors = np.asarray(
        [
            1.96 * float(lookup[scenario.key]["offset_standard_error"])
            for scenario in scenarios
        ]
    )
    plotted[predictor] = values
    spread = float(metrics[predictor]["offset_log10_range"])
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
        label=f"{PREDICTOR_LABELS[predictor]}, range={spread:.3f}",
    )
  for index in range(len(scenarios)):
    available = [
        plotted[predictor][index]
        for predictor in PREDICTORS
        if predictor in plotted
    ]
    if len(available) > 1:
      axis.plot(
          available,
          [positions[index] + styles[predictor][2]
           for predictor in PREDICTORS if predictor in plotted],
          color="0.78",
          linewidth=0.75,
          zorder=0,
      )
  axis.axvline(0.0, color="black", linewidth=0.9, linestyle="--")
  axis.set_yticks(
      positions,
      [scenario.display_name for scenario in scenarios],
      fontsize=8,
  )
  axis.invert_yaxis()
  axis.set_xlabel(
      r"configuration transition offset $\delta_c$ (log$_{10}$ decades)"
  )
  axis.set_title("Configuration-specific pathway-transition offsets")
  axis.grid(axis="x", alpha=0.2)
  if plotted:
    axis.legend(
        frameon=False,
        fontsize=8,
        title="continuous-Iw fractional-logit fit (95% fit intervals)",
        title_fontsize=7,
    )
  for suffix in ("pdf", "png"):
    figure.savefig(
        output_dir / f"f_transition_offsets.{suffix}",
        dpi=300,
        bbox_inches="tight",
    )
  plt.close(figure)


def _write_results_note(
    cells: list[TimescaleCell],
    scenarios: tuple[RecommendationScenario, ...],
    metrics: dict[str, dict[str, object]],
    rows: list[dict[str, object]],
    controlled_metrics: dict[str, dict[str, object]],
    output_dir: Path,
) -> None:
  lookup = {
      (str(row["predictor"]), str(row["configuration"])): row
      for row in rows
  }
  lines = [
      "# Macroscopic time-scale result",
      "",
      f"The scan contains {len(cells)} cells across {len(scenarios)} scenarios.",
      "Transition centers fit the same continuous I_w shown in the heatmaps.",
      "",
      "## Transition-offset spread",
      "",
      "| coordinate | full-grid range (decades) | alpha,q<1 range |",
      "|---|---:|---:|",
  ]
  for predictor in PREDICTORS:
    full = metrics[predictor]
    controlled = controlled_metrics[predictor]
    full_value = (
        f"{float(full['offset_log10_range']):.4f}"
        if full.get("available")
        else "unavailable"
    )
    controlled_value = (
        f"{float(controlled['offset_log10_range']):.4f}"
        if controlled.get("available")
        else "unavailable"
    )
    lines.append(
        f"| {PREDICTOR_LABELS[predictor]} | {full_value} | {controlled_value} |"
    )
  lines.extend(
      [
          "",
          "## Full-grid centered offsets",
          "",
          "| scenario | q/alpha | Gamma(0) |",
          "|---|---:|---:|",
      ]
  )
  for scenario in scenarios:
    values = []
    for predictor in PREDICTORS:
      row = lookup.get((predictor, scenario.key))
      values.append(
          f"{float(row['offset_log10']):+.4f}" if row is not None else "n/a"
      )
    lines.append(f"| {scenario.display_name} | {' | '.join(values)} |")
  lines.extend(
      [
          "",
          (
              "The L1, zeta=4 column is a mean-power first-moment closure "
              "diagnostic (E[S]^4), not an exact E[S^4] closure."
          ),
          "",
      ]
  )
  (output_dir / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


def analyze_output(output_dir: Path) -> None:
  protocol = json.loads((output_dir / "protocol.json").read_text(encoding="utf-8"))
  scenarios = tuple(
      SCENARIO_BY_KEY[item["key"]] for item in protocol["scenarios"]
  )
  rates = np.asarray(protocol["rates"], dtype=float)
  cells = load_cells(output_dir / "summary.csv")
  expected = len(scenarios) * rates.size**2
  if len(cells) != expected:
    raise ValueError(f"summary contains {len(cells)} cells, expected {expected}")
  _plot_pathway_heatmaps(cells, rates, scenarios, output_dir)
  metrics, rows = _write_offset_tables(cells, scenarios, output_dir)
  controlled = [cell for cell in cells if cell.alpha < 1.0 and cell.q < 1.0]
  controlled_metrics, _ = _write_offset_tables(
      controlled,
      scenarios,
      output_dir,
      suffix="_rates_below_one",
  )
  _plot_transition_offsets(metrics, rows, scenarios, output_dir)
  _write_results_note(
      cells,
      scenarios,
      metrics,
      rows,
      controlled_metrics,
      output_dir,
  )


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      "--output-dir",
      type=Path,
      default=(
          MESOSCOPIC_OUTPUT.resolve() / "macroscopic_timescale_ratio_b161"
      ),
  )
  return parser.parse_args()


def main() -> None:
  args = parse_args()
  analyze_output(args.output_dir)
  print(f"wrote pathway and transition-offset figures: {args.output_dir}")


if __name__ == "__main__":
  main()
