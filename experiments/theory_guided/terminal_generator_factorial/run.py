"""Run the orthogonal 2x2 terminal-generator experiment.

Examples, from the repository root::

    PYTHONPATH=src:. python -m \
      experiments.theory_guided.terminal_generator_factorial.run smoke
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import scipy

from ehk.modeling.terminal_generator import (
    TerminalGeneratorParameters,
    run_terminal_generator,
)
from experiments.theory_guided.terminal_generator_factorial.paths import (
    OUTPUT_DIR,
)

HERE = Path(__file__).resolve().parent
PRESETS = {
    "smoke": HERE / "configs" / "smoke.json",
    "historical-12": HERE / "configs" / "historical_12.json",
    "paper-figure3": HERE / "configs" / "paper_figure3.json",
}
STATE_LEVELS = ("pair", "score_moments")
TIMESCALES = ("unsplit", "fast_slow")
CATEGORIES = ("k1", "k2", "k3", "k4plus", "censored")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_revision() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != 1:
        raise ValueError("unsupported factorial config schema")
    if int(config["replicates"]) < 1:
        raise ValueError("replicates must be positive")
    if not config["cases"] or not config["configurations"]:
        raise ValueError("cases and configurations must be nonempty")
    return config


def _build_specs(
    config: dict[str, Any],
) -> list[tuple[str, str, str, TerminalGeneratorParameters]]:
    model = config["model"]
    base = TerminalGeneratorParameters(
        population=int(model["population"]),
        grid_size=int(model["grid_size"]),
        mean_degree=int(model["mean_degree"]),
        recsys_count=int(model["recsys_count"]),
        epsilon=float(model["epsilon"]),
        max_steps=int(model["max_steps"]),
        fast_slow_ratio_threshold=float(model["fast_slow_ratio_threshold"]),
        fast_max_steps=int(model["fast_max_steps"]),
        fast_zero_checks=int(model["fast_zero_checks"]),
    )
    specs = []
    serial = 0
    for case in config["cases"]:
        for recommendation in config["configurations"]:
            for state_level in STATE_LEVELS:
                for timescale in TIMESCALES:
                    for replicate in range(int(config["replicates"])):
                        # Common random numbers across the four factorial
                        # cells for one case/configuration/replicate.
                        seed = int(config["base_seed"]) + 1_000_003 * serial + replicate
                        parameters = replace(
                            base,
                            influence=float(case["influence"]),
                            rewiring=float(case["rewiring"]),
                            recsys=str(recommendation["recsys"]),
                            recommendation_steepness=float(recommendation["steepness"]),
                            state_level=state_level,
                            timescale=timescale,
                            seed=seed,
                        )
                        parameters.validate()
                        specs.append(
                            (
                                str(case["key"]),
                                str(recommendation["key"]),
                                str(replicate),
                                parameters,
                            )
                        )
            serial += 1
    return specs


def _run_spec(
    item: tuple[str, str, str, TerminalGeneratorParameters],
) -> dict[str, object]:
    case, configuration, replicate, parameters = item
    result = run_terminal_generator(parameters)
    return {
        "case": case,
        "configuration": configuration,
        "state_level": parameters.state_level,
        "timescale": parameters.timescale,
        "replicate": int(replicate),
        "seed": parameters.seed,
        "alpha": parameters.influence,
        "q": parameters.rewiring,
        "q_over_alpha": parameters.rewiring / parameters.influence,
        "category_final": result.category,
        "raw_category_final": result.raw_category,
        "converged": result.converged,
        "stopped_step": result.stopped_step,
        "rewiring_events": result.rewiring_events,
        "mean_cap_fraction": result.mean_cap_fraction,
        "fast_slow_applied": result.fast_slow_applied,
        "fast_substeps": result.fast_substeps,
        "fast_max_steps_hits": result.fast_max_steps_hits,
        "final_fast_residual": result.final_fast_residual,
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _probabilities(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    keys = sorted(
        {
            (row["case"], row["configuration"], row["state_level"], row["timescale"])
            for row in rows
        }
    )
    output = []
    for case, configuration, state_level, timescale in keys:
        selected = [
            row
            for row in rows
            if (row["case"], row["configuration"], row["state_level"], row["timescale"])
            == (case, configuration, state_level, timescale)
        ]
        counts = {
            category: sum(row["category_final"] == category for row in selected)
            for category in CATEGORIES
        }
        output.append(
            {
                "case": case,
                "configuration": configuration,
                "state_level": state_level,
                "timescale": timescale,
                "n": len(selected),
                **{f"p_{key}": value / len(selected) for key, value in counts.items()},
                "mean_stopped_step": float(
                    np.mean([row["stopped_step"] for row in selected])
                ),
                "fast_slow_applied_fraction": float(
                    np.mean([row["fast_slow_applied"] for row in selected])
                ),
            }
        )
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("preset", nargs="?", choices=tuple(PRESETS), default="smoke")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR.resolve())
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.jobs < 1:
        raise ValueError("jobs must be positive")
    config_path = (
        args.config.expanduser().resolve()
        if args.config is not None
        else PRESETS[args.preset].resolve()
    )
    config = _load_config(config_path)
    specs = _build_specs(config)
    print(
        f"resolved {len(specs)} paths: "
        f"{len(config['cases'])} cases x {len(config['configurations'])} "
        "recommenders x 2 state levels x 2 timescales"
    )
    if args.dry_run:
        return
    if args.jobs == 1:
        rows = [_run_spec(item) for item in specs]
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            rows = list(pool.map(_run_spec, specs))
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "factorial_runs.csv", rows)
    _write_csv(output_dir / "factorial_probabilities.csv", _probabilities(rows))
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "config_path": str(config_path),
        "config_sha256": _sha256(config_path),
        "generator_sha256": _sha256(
            Path(__file__).resolve().parents[3]
            / "src"
            / "ehk"
            / "modeling"
            / "terminal_generator.py"
        ),
        "git_revision": _git_revision(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "factor_axes": {
            "state_level": STATE_LEVELS,
            "timescale": TIMESCALES,
        },
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote factorial experiment to {output_dir}")


if __name__ == "__main__":
    main()
