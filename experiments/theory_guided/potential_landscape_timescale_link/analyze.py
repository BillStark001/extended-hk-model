"""Analyze targeted potential landscapes against the pathway index."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from ehk.common.plotting import setup_paper_params
from experiments.theory_guided.macroscopic_timescale_ratio.scenarios import (
    SCENARIO_BY_KEY,
)
from theory.paths import MESOSCOPIC_OUTPUT

DESIGN_LABELS = {
    "common_rates": r"common rates: $\alpha=q=0.1$",
    "transition_center": r"fitted centers: $\alpha=0.1$",
}
COLORS = tuple(plt.get_cmap("tab10")(index) for index in range(7))


def _read_summary(path: Path) -> list[dict[str, Any]]:
    numeric = {
        "steepness",
        "alpha",
        "q",
        "fitted_transition_ratio",
        "rate_ratio",
        "I_w",
        "t_P_0.5",
        "t_H_0.5",
        "t_u_0.1",
        "t_double_well",
        "t_barrier_0.01",
        "t_barrier_0.05",
        "t_barrier_half_final",
        "barrier_initial",
        "barrier_at_u_0.1",
        "barrier_at_t_P",
        "barrier_at_t_H",
        "barrier_final",
        "barrier_max",
        "well_separation_final",
        "left_curvature_final",
        "right_curvature_final",
        "barrier_curvature_final",
        "t_double_well_minus_t_P",
        "t_double_well_minus_t_H",
    }
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as stream:
        for raw in csv.DictReader(stream):
            rows.append(
                {
                    key: float(value) if key in numeric else value
                    for key, value in raw.items()
                }
            )
    return rows


def _load_case(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        return {
            name: np.asarray(source[name]).copy()
            for name in (
                "x",
                "time",
                "potential",
                "I_p",
                "I_h",
                "barrier_height",
            )
        }


def _save_figure(figure: plt.Figure, output_dir: Path, stem: str) -> None:
    for suffix in ("pdf", "png"):
        figure.savefig(output_dir / f"{stem}.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(figure)


def _design_rows(rows: list[dict[str, Any]], design: str) -> list[dict[str, Any]]:
    return [row for row in rows if row["design"] == design]


def _plot_barrier_evolution(
    rows: list[dict[str, Any]],
    case_data: dict[str, dict[str, np.ndarray]],
    designs: tuple[str, ...],
    output_dir: Path,
) -> None:
    setup_paper_params()
    figure, axes = plt.subplots(
        2,
        len(designs),
        figsize=(5.0 * len(designs), 6.6),
        squeeze=False,
        constrained_layout=True,
    )
    for column, design in enumerate(designs):
        for index, row in enumerate(_design_rows(rows, design)):
            data = case_data[str(row["key"])]
            label = SCENARIO_BY_KEY[str(row["configuration"])].display_name
            axes[0, column].plot(
                data["time"],
                data["barrier_height"],
                color=COLORS[index],
                linewidth=1.35,
                label=label,
            )
            progress = data["I_p"] + data["I_h"]
            keep = np.r_[True, np.diff(progress) > 1e-10]
            axes[1, column].plot(
                progress[keep],
                data["barrier_height"][keep],
                color=COLORS[index],
                linewidth=1.35,
            )
        axes[0, column].set_title(DESIGN_LABELS.get(design, design))
        axes[0, column].set_xlabel("time")
        axes[0, column].set_xscale("symlog", linthresh=10.0)
        axes[0, column].grid(alpha=0.2)
        axes[1, column].set_xlabel(r"macro progress $u=I_p+I_h$")
        axes[1, column].set_xlim(left=0.0)
        axes[1, column].grid(alpha=0.2)
    axes[0, 0].set_ylabel(r"barrier height $\Delta V$")
    axes[1, 0].set_ylabel(r"barrier height $\Delta V$")
    axes[0, -1].legend(
        loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, fontsize=8
    )
    _save_figure(figure, output_dir, "f_barrier_evolution")


def _curve_at_progress(
    progress: np.ndarray,
    curves: np.ndarray,
    threshold: float,
) -> np.ndarray:
    found = np.flatnonzero(progress >= threshold)
    if not found.size:
        return curves[-1]
    upper = int(found[0])
    if upper == 0:
        return curves[0]
    lower = upper - 1
    denominator = float(progress[upper] - progress[lower])
    fraction = (
        (threshold - float(progress[lower])) / denominator if denominator > 0 else 1.0
    )
    return curves[lower] + np.clip(fraction, 0.0, 1.0) * (curves[upper] - curves[lower])


def _plot_potential_snapshots(
    rows: list[dict[str, Any]],
    case_data: dict[str, dict[str, np.ndarray]],
    designs: tuple[str, ...],
    output_dir: Path,
) -> None:
    setup_paper_params()
    configurations = [
        str(row["configuration"]) for row in _design_rows(rows, designs[0])
    ]
    figure, axes = plt.subplots(
        len(designs),
        len(configurations),
        figsize=(2.25 * len(configurations), 2.25 * len(designs)),
        sharex=True,
        squeeze=False,
    )
    figure.subplots_adjust(
        left=0.055,
        right=0.99,
        bottom=0.13,
        top=0.86,
        wspace=0.27,
        hspace=0.43,
    )
    styles = (
        ("initial", "--", "0.55"),
        (r"$u=0.1$", "-", "tab:orange"),
        ("final", "-", "tab:blue"),
    )
    lookup = {(str(row["design"]), str(row["configuration"])): row for row in rows}
    for row_index, design in enumerate(designs):
        y_min = math.inf
        y_max = -math.inf
        selected: list[tuple[np.ndarray, tuple[np.ndarray, ...]]] = []
        for configuration in configurations:
            row = lookup[design, configuration]
            data = case_data[str(row["key"])]
            progress = data["I_p"] + data["I_h"]
            curves = (
                data["potential"][0],
                _curve_at_progress(progress, data["potential"], 0.1),
                data["potential"][-1],
            )
            selected.append((data["x"], curves))
            y_min = min(y_min, *(float(np.min(curve)) for curve in curves))
            y_max = max(y_max, *(float(np.max(curve)) for curve in curves))
        margin = 0.04 * max(y_max - y_min, 1e-8)
        paired = zip(configurations, selected, strict=True)
        for column, (configuration, (x, curves)) in enumerate(paired):
            axis = axes[row_index, column]
            for curve, (label, linestyle, color) in zip(curves, styles, strict=True):
                axis.plot(
                    x,
                    curve,
                    label=label,
                    linestyle=linestyle,
                    color=color,
                    linewidth=1.1,
                )
            axis.set_ylim(y_min - margin, y_max + margin)
            axis.grid(alpha=0.15)
            if row_index == 0:
                axis.set_title(SCENARIO_BY_KEY[configuration].display_name, fontsize=8)
            if row_index == len(designs) - 1:
                axis.set_xlabel("opinion $x$")
        axes[row_index, 0].set_ylabel(DESIGN_LABELS.get(design, design) + "\n$V(x,t)$")
    axes[0, 0].legend(frameon=False, fontsize=7, loc="best")
    _save_figure(figure, output_dir, "f_potential_snapshots")


def _plot_landscape_timing(
    rows: list[dict[str, Any]],
    designs: tuple[str, ...],
    output_dir: Path,
) -> None:
    setup_paper_params()
    figure, axes = plt.subplots(
        1,
        len(designs),
        figsize=(5.2 * len(designs), 4.0),
        squeeze=False,
        constrained_layout=True,
    )
    fields = (
        ("t_H_0.5", r"$t_H$", "s", "tab:red"),
        ("t_P_0.5", r"$t_P$", "o", "tab:blue"),
        ("t_double_well", r"$t_{\rm dw}$", "D", "tab:purple"),
        ("t_barrier_0.05", r"$t_{\Delta V=0.05}$", "^", "tab:green"),
    )
    for column, design in enumerate(designs):
        axis = axes[0, column]
        selected = _design_rows(rows, design)
        positions = np.arange(len(selected), dtype=float)
        for field, label, marker, color in fields:
            values = np.asarray([float(row[field]) for row in selected])
            finite = np.isfinite(values)
            axis.scatter(
                positions[finite],
                values[finite],
                label=label,
                marker=marker,
                color=color,
                s=28,
            )
        axis.set_xticks(
            positions,
            [
                SCENARIO_BY_KEY[str(row["configuration"])].display_name
                for row in selected
            ],
            rotation=38,
            ha="right",
            fontsize=7,
        )
        axis.set_yscale("symlog", linthresh=10.0)
        axis.set_title(DESIGN_LABELS.get(design, design))
        axis.grid(axis="y", alpha=0.2)
    axes[0, 0].set_ylabel("first-passage time")
    axes[0, -1].legend(frameon=False, fontsize=8, loc="best")
    _save_figure(figure, output_dir, "f_landscape_timing")


def _spearman(rows: list[dict[str, Any]], x: str, y: str) -> dict[str, Any]:
    pairs = np.asarray([(float(row[x]), float(row[y])) for row in rows], dtype=float)
    finite = np.all(np.isfinite(pairs), axis=1)
    selected = pairs[finite]
    if (
        selected.shape[0] < 3
        or np.ptp(selected[:, 0]) == 0
        or np.ptp(selected[:, 1]) == 0
    ):
        return {
            "n": int(selected.shape[0]),
            "rho": float("nan"),
            "pvalue": float("nan"),
        }
    result = spearmanr(selected[:, 0], selected[:, 1])
    return {
        "n": int(selected.shape[0]),
        "rho": float(result.statistic),
        "pvalue": float(result.pvalue),
    }


def _metrics(rows: list[dict[str, Any]], designs: tuple[str, ...]) -> dict[str, Any]:
    result: dict[str, Any] = {"by_design": {}}
    outcomes = (
        "t_double_well",
        "t_double_well_minus_t_P",
        "t_double_well_minus_t_H",
        "barrier_at_u_0.1",
        "barrier_final",
        "barrier_max",
    )
    for design in designs:
        selected = _design_rows(rows, design)
        pathways = np.asarray([float(row["I_w"]) for row in selected])
        design_result: dict[str, Any] = {
            "count": len(selected),
            "pathway_mean": float(np.mean(pathways)),
            "pathway_standard_deviation": float(np.std(pathways)),
            "pathway_range": float(np.ptp(pathways)),
            "correlations_with_pathway": {
                outcome: _spearman(selected, "I_w", outcome) for outcome in outcomes
            },
        }
        reference = next(
            (row for row in selected if row["configuration"] == "random"), None
        )
        if reference is not None:
            comparisons = []
            for row in selected:
                comparisons.append(
                    {
                        "configuration": row["configuration"],
                        "delta_I_w": float(row["I_w"]) - float(reference["I_w"]),
                        "delta_t_double_well": float(row["t_double_well"])
                        - float(reference["t_double_well"]),
                        "delta_barrier_at_u_0.1": float(row["barrier_at_u_0.1"])
                        - float(reference["barrier_at_u_0.1"]),
                        "delta_barrier_final": float(row["barrier_final"])
                        - float(reference["barrier_final"]),
                    }
                )
            design_result["relative_to_random"] = comparisons
        result["by_design"][design] = design_result
    return result


def _format(value: Any, digits: int = 4) -> str:
    number = float(value)
    return f"{number:.{digits}g}" if np.isfinite(number) else "n/a"


def _write_results(
    rows: list[dict[str, Any]],
    designs: tuple[str, ...],
    metrics: dict[str, Any],
    output_dir: Path,
) -> None:
    lines = [
        "# Potential-landscape link to the pathway scan",
        "",
        (
            "The experiment contains two controlled seven-scenario designs. "
            "The potential is reconstructed from the coefficient-free force "
            "v/alpha; Delta V is the smaller of the two basin depths."
        ),
    ]
    for design in designs:
        selected = _design_rows(rows, design)
        lines.extend(
            [
                "",
                f"## {DESIGN_LABELS.get(design, design)}",
                "",
                "| scenario | alpha | q | I_w | t_dw | t_dw-t_P | t_dw-t_H | Delta V(u=0.1) | Delta V(final) |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in selected:
            scenario = SCENARIO_BY_KEY[str(row["configuration"])]
            values = (
                scenario.display_name,
                _format(row["alpha"]),
                _format(row["q"]),
                _format(row["I_w"]),
                _format(row["t_double_well"]),
                _format(row["t_double_well_minus_t_P"]),
                _format(row["t_double_well_minus_t_H"]),
                _format(row["barrier_at_u_0.1"]),
                _format(row["barrier_final"]),
            )
            lines.append("| " + " | ".join(values) + " |")
        design_metrics = metrics["by_design"][design]
        lines.extend(
            [
                "",
                (
                    "Pathway dispersion: mean="
                    f"{_format(design_metrics['pathway_mean'])}, "
                    f"range={_format(design_metrics['pathway_range'])}."
                ),
                "",
                "Spearman correlations across the seven scenarios (exploratory):",
                "",
                "| landscape observable | rho with I_w | p | n |",
                "|---|---:|---:|---:|",
            ]
        )
        for field, result in design_metrics["correlations_with_pathway"].items():
            lines.append(
                f"| {field} | {_format(result['rho'], 3)} | "
                f"{_format(result['pvalue'], 3)} | {result['n']} |"
            )
    lines.extend(
        [
            "",
            "## Interpretation guardrails",
            "",
            (
                "These seven-point correlations diagnose consistency with a "
                "mechanism; they are not independent-cell significance tests."
            ),
            (
                "The effective potential is an instantaneous drift integral, "
                "not a thermodynamic potential or a Lyapunov function."
            ),
            (
                "L1-StructureRandom at zeta=4 remains a mean-power first-moment "
                "closure, so its proximity to Random cannot establish microscopic "
                "equivalence."
            ),
            "",
        ]
    )
    (output_dir / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


def analyze_output(output_dir: Path) -> None:
    protocol = json.loads((output_dir / "protocol.json").read_text(encoding="utf-8"))
    designs = tuple(dict.fromkeys(str(case["design"]) for case in protocol["cases"]))
    rows = _read_summary(output_dir / "summary.csv")
    if len(rows) != len(protocol["cases"]):
        raise ValueError(
            f"summary contains {len(rows)} cases, expected {len(protocol['cases'])}"
        )
    case_data = {
        str(row["key"]): _load_case(output_dir / "cases" / f"{row['key']}.npz")
        for row in rows
    }
    _plot_barrier_evolution(rows, case_data, designs, output_dir)
    _plot_potential_snapshots(rows, case_data, designs, output_dir)
    _plot_landscape_timing(rows, designs, output_dir)
    metrics = _metrics(rows, designs)
    (output_dir / "analysis_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    _write_results(rows, designs, metrics, output_dir)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(MESOSCOPIC_OUTPUT.resolve() / "potential_landscape_timescale_link"),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    output_dir = args.output_dir.expanduser().resolve()
    analyze_output(output_dir)
    print(f"wrote potential-landscape pathway analysis: {output_dir}")


if __name__ == "__main__":
    main()
