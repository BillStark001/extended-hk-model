"""Aggregate paired solver and dynamics differences from a completed scan."""

from __future__ import annotations

import argparse
import csv
import math
import os
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ehk.common.plotting import setup_paper_params

PAIR_FIELDS = (
    "epsilon",
    "grid_size",
    "configuration",
    "recsys",
    "steepness",
    "alpha_index",
    "q_index",
    "alpha",
    "q",
    "diagonal_offset",
)


def _read_summary(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _difference(left: dict[str, str], right: dict[str, str], field: str) -> float:
    return abs(float(left[field]) - float(right[field]))


def paired_differences(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    """Reduce four matched dynamics/method rows to one physical-case row."""

    groups: dict[tuple[str, ...], dict[tuple[str, str], dict[str, str]]] = defaultdict(
        dict
    )
    for row in rows:
        key = tuple(row[field] for field in PAIR_FIELDS)
        groups[key][(row["dynamics"], row["opinion_method"])] = row

    expected = {
        ("hk", "measure"),
        ("hk", "fokker_planck"),
        ("deffuant", "measure"),
        ("deffuant", "fokker_planck"),
    }
    result = []
    for key, group in groups.items():
        if set(group) != expected:
            missing = sorted(expected - set(group))
            raise ValueError(f"incomplete four-way comparison for {key}: {missing}")
        hk_jump = group[("hk", "measure")]
        hk_fp = group[("hk", "fokker_planck")]
        deffuant_jump = group[("deffuant", "measure")]
        deffuant_fp = group[("deffuant", "fokker_planck")]
        method_iw_hk = _difference(hk_jump, hk_fp, "I_w")
        method_iw_deffuant = _difference(deffuant_jump, deffuant_fp, "I_w")
        dynamics_iw_jump = _difference(hk_jump, deffuant_jump, "I_w")
        dynamics_iw_fp = _difference(hk_fp, deffuant_fp, "I_w")
        barrier_rows = tuple(group.values())
        overshoot_classes = {row["multiwell_overshoot_class"] for row in barrier_rows}
        output: dict[str, object] = {
            **dict(zip(PAIR_FIELDS, key, strict=True)),
            "method_gap_I_w_hk": method_iw_hk,
            "method_gap_I_w_deffuant": method_iw_deffuant,
            "method_gap_I_w_max": max(method_iw_hk, method_iw_deffuant),
            "dynamics_gap_I_w_measure": dynamics_iw_jump,
            "dynamics_gap_I_w_fokker_planck": dynamics_iw_fp,
            "dynamics_gap_I_w_max": max(dynamics_iw_jump, dynamics_iw_fp),
            "method_gap_barrier_hk": _difference(
                hk_jump, hk_fp, "multiwell_barrier_peak"
            ),
            "method_gap_barrier_deffuant": _difference(
                deffuant_jump, deffuant_fp, "multiwell_barrier_peak"
            ),
            "dynamics_gap_barrier_measure": _difference(
                hk_jump, deffuant_jump, "multiwell_barrier_peak"
            ),
            "dynamics_gap_barrier_fokker_planck": _difference(
                hk_fp, deffuant_fp, "multiwell_barrier_peak"
            ),
            "robust_well_count_max": max(
                int(row["robust_well_count_peak"]) for row in barrier_rows
            ),
            "robust_barrier_count_max": max(
                int(row["robust_barrier_count_peak"]) for row in barrier_rows
            ),
            "dominant_pair_switch_count_max": max(
                int(row["dominant_pair_switch_count"]) for row in barrier_rows
            ),
            "overshoot_class_count": len(overshoot_classes),
            "overshoot_classes": ";".join(sorted(overshoot_classes)),
        }
        output["method_gap_barrier_max"] = max(
            float(output["method_gap_barrier_hk"]),
            float(output["method_gap_barrier_deffuant"]),
        )
        output["dynamics_gap_barrier_max"] = max(
            float(output["dynamics_gap_barrier_measure"]),
            float(output["dynamics_gap_barrier_fokker_planck"]),
        )
        result.append(output)
    return sorted(
        result,
        key=lambda row: (
            float(row["epsilon"]),
            str(row["configuration"]),
            int(row["q_index"]),
            int(row["alpha_index"]),
        ),
    )


def _atomic_csv(path: Path, rows: list[dict[str, object]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _plot_disagreement(rows: list[dict[str, object]], output_dir: Path) -> None:
    setup_paper_params()
    epsilons = sorted({float(row["epsilon"]) for row in rows})
    figure, axes = plt.subplots(
        1,
        len(epsilons),
        figsize=(3.4 * len(epsilons), 3.1),
        sharex=True,
        sharey=True,
        constrained_layout=True,
        squeeze=False,
    )
    for axis, epsilon in zip(axes[0], epsilons, strict=True):
        selected = [row for row in rows if float(row["epsilon"]) == epsilon]
        method_gap = np.asarray([float(row["method_gap_I_w_max"]) for row in selected])
        dynamics_gap = np.asarray(
            [float(row["dynamics_gap_I_w_max"]) for row in selected]
        )
        wells = np.asarray([float(row["robust_well_count_max"]) for row in selected])
        scatter = axis.scatter(
            method_gap,
            dynamics_gap,
            c=wells,
            cmap="viridis",
            vmin=1,
            vmax=max(3, float(np.max(wells))),
            s=18,
            alpha=0.75,
        )
        axis.set_title(rf"$\epsilon={epsilon:g}$")
        axis.set_xlabel(r"method gap $|\Delta I_w|$")
        axis.grid(alpha=0.2)
    axes[0, 0].set_ylabel(r"dynamics gap $|\Delta I_w|$")
    figure.colorbar(scatter, ax=axes, label="maximum robust well count", shrink=0.8)
    for suffix in ("pdf", "png"):
        figure.savefig(output_dir / f"f_method_dynamics_disagreement.{suffix}")
    plt.close(figure)


def _write_results(rows: list[dict[str, object]], output_dir: Path) -> None:
    method_max = max(rows, key=lambda row: float(row["method_gap_I_w_max"]))
    dynamics_max = max(rows, key=lambda row: float(row["dynamics_gap_I_w_max"]))
    class_counts = Counter(
        class_name
        for row in rows
        for class_name in str(row["overshoot_classes"]).split(";")
    )
    lines = [
        "# Multibarrier dynamics comparison",
        "",
        f"Paired physical cases: {len(rows)}.",
        "",
        (
            "Largest pathway-index method gap: "
            f"{float(method_max['method_gap_I_w_max']):.6g} at "
            f"epsilon={method_max['epsilon']}, {method_max['configuration']}, "
            f"q-index={method_max['q_index']}, alpha-index={method_max['alpha_index']}."
        ),
        "",
        (
            "Largest pathway-index dynamics gap: "
            f"{float(dynamics_max['dynamics_gap_I_w_max']):.6g} at "
            f"epsilon={dynamics_max['epsilon']}, {dynamics_max['configuration']}, "
            f"q-index={dynamics_max['q_index']}, alpha-index={dynamics_max['alpha_index']}."
        ),
        "",
        "Overshoot classifications represented: "
        + ", ".join(f"{key} ({value})" for key, value in sorted(class_counts.items()))
        + ".",
        "",
        (
            "The paired table is the canonical source for selecting detailed "
            "four-way trajectory comparisons."
        ),
    ]
    (output_dir / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze_output(output_dir: Path) -> list[dict[str, object]]:
    """Write paired differences, a diagnostic plot, and a compact report."""

    summary = output_dir / "summary.csv"
    if not summary.exists():
        raise FileNotFoundError(f"missing completed scan summary: {summary}")
    rows = paired_differences(_read_summary(summary))
    if not rows:
        raise ValueError("summary contains no paired cases")
    _atomic_csv(output_dir / "paired_differences.csv", rows)
    _plot_disagreement(rows, output_dir)
    _write_results(rows, output_dir)
    return rows


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    output_dir = args.output_dir.expanduser().resolve()
    rows = analyze_output(output_dir)
    finite = sum(math.isfinite(float(row["method_gap_I_w_max"])) for row in rows)
    print(f"analyzed {finite}/{len(rows)} finite paired comparisons: {output_dir}")


if __name__ == "__main__":
    main()
