"""Run the base-layer terminal-probability experiment with ``smp-lifted``.

The directory name is retained so existing module invocations remain easy to
find, but the former Python pair/score-moment factorial is intentionally not
supported. Every stochastic state update is performed by the external Go
runtime; this module only constructs explicit requests and records responses.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from smp_meso_bindings import print_progress, run_lifted_batch_parallel

from experiments.theory_guided.terminal_generator_factorial.paths import (
    OUTPUT_DIR,
)

HERE = Path(__file__).resolve().parent
REPOSITORY_ROOT = HERE.parents[2]
DEFAULT_BINARY = (
    REPOSITORY_ROOT.parent
    / "social-media-mesoscopic-models"
    / "bin"
    / "smp-lifted"
)
PRESETS = {
    "smoke": HERE / "configs" / "smoke.json",
    "paper-figure3": HERE / "configs" / "paper_figure3.json",
}
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
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != 1:
        raise ValueError("unsupported terminal-probability config schema")
    if int(config["replicates"]) < 1:
        raise ValueError("replicates must be positive")
    if not config["cases"] or not config["configurations"]:
        raise ValueError("cases and configurations must be nonempty")
    return config


def _request_seed(base_seed: int, key: str) -> int:
    suffix = int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16)
    return base_seed + suffix


def _build_requests(
    config: dict[str, Any], workers_per_request: int
) -> list[dict[str, object]]:
    model = config["model"]
    paths = int(config["replicates"])
    requests: list[dict[str, object]] = []
    for case in config["cases"]:
        for recommendation in config["configurations"]:
            request_id = f"{recommendation['key']}/{case['key']}"
            recommender_type = str(recommendation["recsys"])
            requests.append(
                {
                    "request_id": request_id,
                    "layer": "base",
                    "population": int(model["population"]),
                    "opinion_bins": int(model["grid_size"]),
                    "out_degree": int(model["mean_degree"]),
                    "recommendation_count": int(model["recsys_count"]),
                    "max_steps": int(model["max_steps"]),
                    "paths": paths,
                    "interval_paths": 1,
                    "ambiguity_samples": 1,
                    "confidence_level": 0.95,
                    "workers": workers_per_request,
                    "seed": _request_seed(int(config["base_seed"]), request_id),
                    "major_cluster_mass": 0.002,
                    "dynamics": {
                        "type": "hk",
                        "tolerance": float(model["epsilon"]),
                        "influence": float(case["influence"]),
                        "rewiring_rate": float(case["rewiring"]),
                    },
                    "recommender": {
                        "type": recommender_type,
                        "steepness": float(recommendation["steepness"]),
                        "random_ratio": 0.0,
                        "opinion_tolerance": 0.4,
                        "noise_std": 0.0,
                        "noise_quadrature_points": 9,
                    },
                    "initial": {
                        "type": "uniform",
                        "opinion_min": -1.0,
                        "opinion_max": 1.0,
                        "probabilities": [],
                    },
                    "resolution": {
                        "score_max": 45,
                        "availability_bins": 11,
                        "component_size_bins": 12,
                        "opinion_quadrature_points": 7,
                        "opinion_quadrature_rule": "unit_variance_quantile",
                    },
                    "closure": {
                        "motif_relaxation": 0.25,
                        "histogram_relaxation": 0.25,
                        "candidate_relaxation": 0.25,
                        "topology_relaxation": 0.25,
                    },
                    "fast_slow": {
                        "mode": "unsplit",
                        "ratio_threshold": 10.0,
                        "max_substeps": 400,
                        "zero_event_batches": 8,
                        "residual_tolerance": 1e-12,
                        "zero_event_residual": 0.25,
                    },
                    "ambiguity": {
                        "eligibility_correlation_radius": 0.0,
                        "score_availability_radius": 0.0,
                        "motif_persistence_radius": 0.0,
                        "bridge_bias_radius": 0.0,
                        "component_mix_radius": 0.0,
                    },
                }
            )
    return requests


def _write_probability_csv(
    path: Path,
    requests: list[dict[str, object]],
    responses: list[dict[str, Any]],
) -> None:
    rows: list[dict[str, object]] = []
    for request, response in zip(requests, responses, strict=True):
        result = response["result"]
        probabilities = result["point"]["probabilities"]
        configuration, case = str(request["request_id"]).split("/", 1)
        dynamics = request["dynamics"]
        assert isinstance(dynamics, dict)
        rows.append(
            {
                "case": case,
                "configuration": configuration,
                "layer": request["layer"],
                "n": request["paths"],
                "alpha": dynamics["influence"],
                "q": dynamics["rewiring_rate"],
                "q_over_alpha": float(dynamics["rewiring_rate"])
                / float(dynamics["influence"]),
                **dict(zip((f"p_{name}" for name in CATEGORIES), probabilities)),
            }
        )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("preset", nargs="?", choices=tuple(PRESETS), default="smoke")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR.resolve())
    parser.add_argument(
        "--lifted-binary",
        type=Path,
        default=Path(os.environ.get("SMP_LIFTED_BINARY", DEFAULT_BINARY)),
    )
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--workers-per-request", type=int, default=2)
    parser.add_argument("--progress-step-interval", type=int, default=1000)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.jobs < 1 or args.workers_per_request < 1:
        raise ValueError("jobs and workers-per-request must be positive")
    config_path = (
        args.config.expanduser().resolve()
        if args.config is not None
        else PRESETS[args.preset].resolve()
    )
    binary = args.lifted_binary.expanduser().resolve()
    config = _load_config(config_path)
    requests = _build_requests(config, args.workers_per_request)
    print(
        f"resolved {len(requests)} base-layer requests and "
        f"{len(requests) * int(config['replicates'])} Go paths"
    )
    if args.dry_run:
        return
    if not binary.is_file():
        raise FileNotFoundError(f"smp-lifted binary not found: {binary}")
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "base_terminal_requests.jsonl").write_text(
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in requests),
        encoding="utf-8",
    )
    responses = run_lifted_batch_parallel(
        binary,
        requests,
        min(args.jobs, len(requests)),
        progress=None if args.no_progress else print_progress,
        progress_step_interval=0 if args.no_progress else args.progress_step_interval,
    )
    (output_dir / "base_terminal_responses.json").write_text(
        json.dumps(responses, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_probability_csv(
        output_dir / "base_terminal_probabilities.csv", requests, responses
    )
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "config_path": str(config_path),
        "config_sha256": _sha256(config_path),
        "runner_sha256": _sha256(Path(__file__)),
        "binary": str(binary),
        "binary_sha256": _sha256(binary),
        "git_revision": _git_revision(),
        "layer": "base",
        "processes": min(args.jobs, len(requests)),
        "workers_per_request": args.workers_per_request,
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"wrote Go base terminal experiment to {output_dir}")


if __name__ == "__main__":
    main()
