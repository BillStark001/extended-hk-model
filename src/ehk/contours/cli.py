"""Command-line interface for recoverable contour runs."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from .adapters import LandscapeBarrierGapEvaluator, OperatorGapEvaluator
from .protocol import load_protocol
from .runner import SequentialContourEstimator
from .store import EvaluationStore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ehk-contour")
    commands = parser.add_subparsers(dest="command", required=True)

    initialize = commands.add_parser("init", help="initialize an immutable protocol store")
    initialize.add_argument("protocol", type=Path)
    initialize.add_argument("store", type=Path)

    run = commands.add_parser("run", help="resume sequential evaluations")
    run.add_argument("store", type=Path)
    run.add_argument(
        "--adapter",
        choices=("operator-gap", "landscape-barrier-gap"),
        required=True,
    )
    run.add_argument("--binary", type=Path, required=True)
    run.add_argument("--max-evaluations", type=int, required=True)
    run.add_argument("--progress", choices=("none", "human", "jsonl"), default="human")
    run.add_argument("--no-validation", action="store_true")

    status = commands.add_parser("status", help="show journal and pending counts")
    status.add_argument("store", type=Path)

    export = commands.add_parser("export-grid", help="write posterior_grid.npz")
    export.add_argument("store", type=Path)
    export.add_argument("--resolution", type=int, default=201)
    return parser


def _open_store(path: Path) -> EvaluationStore:
    protocol = load_protocol(path / "protocol.json")
    return EvaluationStore(path, protocol)


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "init":
        protocol = load_protocol(args.protocol)
        store = EvaluationStore(args.store, protocol)
        output = {"store": str(store.root), "protocol_sha256": protocol.fingerprint}
    elif args.command == "status":
        store = _open_store(args.store)
        evaluations = store.evaluations()
        output = {
            "protocol_sha256": store.protocol.fingerprint,
            "evaluations": len(evaluations),
            "successful": sum(item.status == "ok" for item in evaluations),
            "failed": sum(item.status != "ok" for item in evaluations),
            "pending": len(store.pending_proposals()),
        }
    elif args.command == "run":
        store = _open_store(args.store)
        estimator = SequentialContourEstimator(
            store.protocol,
            store,
            progress=args.progress,
        )
        evaluator = (
            OperatorGapEvaluator(args.binary)
            if args.adapter == "operator-gap"
            else LandscapeBarrierGapEvaluator(args.binary)
        )
        output = estimator.run(
            evaluator,
            max_evaluations=args.max_evaluations,
            run_validation=not args.no_validation,
        )
    else:
        store = _open_store(args.store)
        estimator = SequentialContourEstimator(store.protocol, store, progress="none")
        output = {"posterior_grid": str(estimator.export_grid(resolution=args.resolution))}
    print(json.dumps(output, indent=2, sort_keys=True))
