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
from tqdm import tqdm

from ehk.terminal import classify_opinions
from experiments.theory_guided.terminal_probability.scenarios import (
    PAPER_COMPARISON_CASES,
    RECOMMENDATION_SCENARIOS,
)


CATEGORIES = ("k1", "k2", "k3", "k4plus", "censored")
SUCCESS_STATUSES = frozenset({"absorbed", "censored"})


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def terminal_peak_count(opinions: np.ndarray, epsilon: float) -> int:
    """Compatibility helper returning the common classifier's major count.

    Despite the historical name, this no longer performs KDE peak detection.
    Nonabsorbed states have no terminal peak count and raise ``ValueError``.
    """

    result = classify_opinions(opinions, epsilon)
    if result["status"] != "absorbed":
        raise ValueError(f"state is not absorbed: {result['status']}")
    return int(result["k_major"])


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
    payload: tuple[str, dict[str, Any], float],
) -> dict[str, object]:
    workspace_text, scenario, major_cluster_mass = payload
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
        "status": "failure",
        "terminal_status": "",
        "category": "",
        "steps": math.nan,
        "k_all": math.nan,
        "k_major": math.nan,
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
        terminal = classify_opinions(
            opinions,
            float(scenario["HKParams"]["Tolerance"]),
            major_cluster_mass,
            mass_resolution=1.0 / len(opinions),
        )
        row.update({
            "terminal_status": terminal["status"],
            "category": terminal["category"],
            "k_all": terminal["k_all"],
            "k_major": terminal["k_major"],
            "components": json.dumps(terminal["components"], separators=(",", ":")),
            "margins": json.dumps(terminal["margins"], separators=(",", ":")),
            "status": (
                "absorbed" if terminal["status"] == "absorbed" else "censored"
            ),
        })
    except Exception as error:  # retain a partial audit instead of losing the batch
        row["status"] = "error"
        row["error"] = f"{type(error).__name__}: {error}"
    return row


def summarize(rows: Iterable[dict[str, object]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    required = {"configuration", "case", "alpha", "q", "status", "category"}
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
        successful = group[group["status"].isin(SUCCESS_STATUSES)]
        failed = total - len(successful)
        result: dict[str, object] = {
            "configuration": configuration,
            "case": case,
            "alpha": alpha,
            "q": rewiring,
            "runs_planned": total,
            "runs_successful": len(successful),
            "runs_failed": failed,
            "failure_fraction": failed / total,
        }
        counts = {
            category: int(np.sum(successful["category"] == category))
            for category in CATEGORIES
        }
        for category in CATEGORIES:
            result[f"count_{category}"] = counts[category]
            result[f"p_{category}"] = (
                counts[category] / len(successful) if len(successful) else math.nan
            )
        probability_mass = sum(float(result[f"p_{category}"]) for category in CATEGORIES)
        if len(successful) and not math.isclose(probability_mass, 1.0, abs_tol=1e-12):
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
        "--major-cluster-mass", type=float, default=0.02,
        help="minimum component mass counted by the shared terminal classifier",
    )
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
    if not 0 < args.major_cluster_mass <= 1:
        raise ValueError("major-cluster-mass must lie in (0,1]")
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
    payloads = [
        (str(workspace), scenario, args.major_cluster_mass)
        for scenario in scenarios
    ]

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

    failed = int(np.sum(~run_table["status"].isin(SUCCESS_STATUSES)))
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, "-m", __package__ + ".analyze", *sys.argv[1:]],
        "workspace": str(workspace),
        "manifest": {"path": str(manifest), "sha256": _sha256(manifest)},
        "runs_planned": len(run_table),
        "runs_successful": len(run_table) - failed,
        "runs_failed": failed,
        "classification": {
            "protocol": "atomic-measure-confidence-components-v1",
            "occupied_mass": "0.5 / population",
            "major_mass": args.major_cluster_mass,
            "position_resolution": 0.0,
            "mass_resolution": "1 / population",
            "nonabsorbed_or_ambiguous": "censored",
            "data_quality_failures_are_outcomes": False,
        },
        "outputs": [run_path.name, summary_path.name],
    }
    (output_dir / "analysis_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "output_dir": str(output_dir),
        "runs_planned": len(run_table),
        "runs_successful": len(run_table) - failed,
        "runs_failed": failed,
    }, indent=2))
    if args.require_complete and failed:
        raise RuntimeError(f"{failed} runs failed or are incomplete; see {run_path}")


if __name__ == "__main__":
    main()
