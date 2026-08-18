"""Classify completed SMP runs and summarize microscopic terminal probabilities."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.stats import gaussian_kde, norm
from tqdm import tqdm

from experiments.theory_guided.terminal_probability.scenarios import (
    PAPER_COMPARISON_CASES,
    RECOMMENDATION_SCENARIOS,
)


CATEGORIES = ("k1", "k2", "k3", "k4plus")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def terminal_peak_count(opinions: np.ndarray, epsilon: float) -> int:
    """Count major KDE peaks separated by more than the confidence radius."""

    values = np.asarray(opinions, dtype=float)
    if values.ndim != 1 or values.size < 2 or not np.all(np.isfinite(values)):
        raise ValueError("terminal opinions must be a finite one-dimensional sample")
    if not 0 < epsilon <= 2:
        raise ValueError("epsilon must lie in (0, 2]")

    axis = np.linspace(-1.0, 1.0, 1001)
    sample_std = float(np.std(values, ddof=1))
    if sample_std <= np.finfo(float).eps:
        density = norm.pdf(axis, values[0], 0.1)
    else:
        def bandwidth(kde: gaussian_kde) -> float:
            return max(kde.scotts_factor(), 0.1 / sample_std)

        density = gaussian_kde(values, bw_method=bandwidth)(axis)
    spacing = float(axis[1] - axis[0])
    peaks, _ = find_peaks(
        density,
        height=float(np.max(density)) * 0.1,
        distance=int(math.floor(epsilon / spacing)) + 1,
    )
    return max(int(peaks.size), 1)


def configuration_key(scenario: dict[str, Any]) -> str:
    factory = str(scenario["RecsysFactoryType"])
    steepness = float(scenario.get("RecSysParams", {}).get("Steepness", 1.0))
    for configuration in RECOMMENDATION_SCENARIOS:
        if (
            factory == configuration.factory
            and math.isclose(steepness, configuration.steepness)
        ):
            return configuration.key
    raise ValueError(
        f"unknown recommender configuration: {factory}, steepness={steepness:g}"
    )


def comparison_case(alpha: float, rewiring: float) -> str:
    for case in PAPER_COMPARISON_CASES:
        if math.isclose(alpha, case.alpha) and math.isclose(rewiring, case.rewiring):
            return case.key
    return ""


def _classify_payload(
    payload: tuple[str, dict[str, Any]],
) -> dict[str, object]:
    workspace_text, scenario = payload
    workspace = Path(workspace_text)
    unique_name = str(scenario["UniqueName"])
    run_dir = workspace / unique_name
    alpha = float(scenario["HKParams"]["Influence"])
    rewiring = float(scenario["HKParams"]["RewiringRate"])
    row: dict[str, object] = {
        "unique_name": unique_name,
        "configuration": configuration_key(scenario),
        "case": comparison_case(alpha, rewiring),
        "alpha": alpha,
        "q": rewiring,
        "status": "incomplete",
        "steps": math.nan,
        "k": math.nan,
    }
    try:
        if not run_dir.is_dir():
            row["status"] = "missing_run"
            return row
        if not any(run_dir.glob("finished-*.msgpack")):
            row["status"] = "unfinished"
            return row
        accumulative = sorted(run_dir.glob("acc-state-*"))
        if not accumulative:
            row["status"] = "missing_accumulative_state"
            return row

        from smp_bindings.model_state import load_accumulative_model_state

        state = load_accumulative_model_state(
            str(accumulative[-1]),
            with_agent_numbers=False,
            with_agent_opinion_sums=False,
        )
        opinions = np.asarray(state["opinions"][-1], dtype=float)
        row["steps"] = int(state["steps"])
        row["k"] = terminal_peak_count(
            opinions, float(scenario["HKParams"]["Tolerance"])
        )
        row["status"] = "complete"
    except Exception as error:  # retain a partial audit instead of losing the batch
        row["status"] = "error"
        row["error"] = f"{type(error).__name__}: {error}"
    return row


def summarize(rows: Iterable[dict[str, object]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    required = {"configuration", "case", "alpha", "q", "status", "k"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError("run table is missing: " + ", ".join(missing))

    summaries: list[dict[str, object]] = []
    grouped = frame.groupby(
        ["configuration", "case", "alpha", "q"],
        sort=True,
        dropna=False,
    )
    for (configuration, case, alpha, rewiring), group in grouped:
        total = len(group)
        complete = group[group["status"] == "complete"]
        result: dict[str, object] = {
            "configuration": configuration,
            "case": case,
            "alpha": alpha,
            "q": rewiring,
            "runs_planned": total,
            "runs_complete": len(complete),
            "p_incomplete": 1 - len(complete) / total,
        }
        counts = {
            "k1": int(np.sum(complete["k"] <= 1)),
            "k2": int(np.sum(complete["k"] == 2)),
            "k3": int(np.sum(complete["k"] == 3)),
            "k4plus": int(np.sum(complete["k"] >= 4)),
        }
        for category in CATEGORIES:
            result[f"count_{category}"] = counts[category]
            result[f"p_{category}"] = counts[category] / total
        probability_mass = float(result["p_incomplete"]) + sum(
            float(result[f"p_{category}"]) for category in CATEGORIES
        )
        if not math.isclose(probability_mass, 1.0, abs_tol=1e-12):
            raise RuntimeError(
                f"probability mass for {configuration}/{case} is {probability_mass}"
            )
        summaries.append(result)
    return pd.DataFrame(summaries)


def _read_manifest(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workspace", type=Path, help="SMP sweep output directory")
    parser.add_argument(
        "--manifest", type=Path,
        help="resolved_scenarios.jsonl (default: WORKSPACE/resolved_scenarios.jsonl)",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        help="analysis directory (default: WORKSPACE/analysis)",
    )
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument(
        "--limit", type=int,
        help="analyze only the first N manifest rows for pipeline debugging",
    )
    parser.add_argument(
        "--require-complete", action="store_true",
        help="return an error if any planned run is absent or unfinished",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.jobs < 1:
        raise ValueError("jobs must be positive")
    workspace = args.workspace.expanduser().resolve()
    manifest = (
        args.manifest.expanduser().resolve()
        if args.manifest is not None
        else workspace / "resolved_scenarios.jsonl"
    )
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else workspace / "analysis"
    )
    scenarios = _read_manifest(manifest)
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("limit must be positive")
        scenarios = scenarios[:args.limit]
    payloads = [(str(workspace), scenario) for scenario in scenarios]

    if args.jobs == 1:
        rows = [_classify_payload(payload) for payload in tqdm(payloads)]
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as executor:
            rows = list(tqdm(
                executor.map(_classify_payload, payloads, chunksize=8),
                total=len(payloads),
            ))

    output_dir.mkdir(parents=True, exist_ok=True)
    run_table = pd.DataFrame(rows)
    summary = summarize(rows)
    run_path = output_dir / "microscopic_terminal_runs.csv"
    summary_path = output_dir / "microscopic_terminal_summary.csv"
    run_table.to_csv(run_path, index=False)
    summary.to_csv(summary_path, index=False)

    incomplete = int(np.sum(run_table["status"] != "complete"))
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, "-m", __package__ + ".analyze", *sys.argv[1:]],
        "workspace": str(workspace),
        "manifest": {"path": str(manifest), "sha256": _sha256(manifest)},
        "runs_planned": len(run_table),
        "runs_complete": len(run_table) - incomplete,
        "runs_incomplete": incomplete,
        "classification": {
            "axis": "1001 points on [-1,1]",
            "minimum_bandwidth": 0.1,
            "relative_height": 0.1,
            "minimum_peak_distance": "strictly greater than epsilon",
        },
        "outputs": [run_path.name, summary_path.name],
    }
    (output_dir / "analysis_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "output_dir": str(output_dir),
        "runs_planned": len(run_table),
        "runs_complete": len(run_table) - incomplete,
        "runs_incomplete": incomplete,
    }, indent=2))
    if args.require_complete and incomplete:
        raise RuntimeError(f"{incomplete} runs are incomplete; see {run_path}")


if __name__ == "__main__":
    main()
