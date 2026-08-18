"""Run a preset or JSON-configured terminal-probability sweep."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

from ehk.common.io.files import init_logger
from ehk.micro.runner import run_scenarios
from experiments.paths import SMP_BINARY
from experiments.theory_guided.terminal_probability.paths import OUTPUT_DIR
from experiments.theory_guided.terminal_probability.scenarios import (
    available_presets,
    build_scenarios,
    load_config,
    preset_path,
    scenario_summary,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_config(args: argparse.Namespace, parser: argparse.ArgumentParser) -> Path:
    if args.config is not None and args.preset is not None:
        parser.error("choose either a preset or --config, not both")
    if args.config is not None:
        return args.config.expanduser().resolve()
    if args.preset is not None:
        return preset_path(args.preset).resolve()
    parser.error("provide a preset or --config (use --list-presets to inspect presets)")
    raise AssertionError("argparse exits before this line")


def _write_scenario_manifest(path: Path, scenarios: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for scenario in scenarios:
            stream.write(json.dumps(scenario, sort_keys=True))
            stream.write("\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
    )
    parser.add_argument(
        "preset",
        nargs="?",
        choices=available_presets(),
        help="named scenario preset",
    )
    parser.add_argument("--config", type=Path, help="custom JSON configuration")
    parser.add_argument(
        "--list-presets", action="store_true", help="print preset names and exit"
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--binary", type=Path, default=SMP_BINARY.resolve())
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate and print the resolved sweep without writing or running",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list_presets:
        for name in available_presets():
            print(f"{name}\t{preset_path(name)}")
        return

    config_path = _resolve_config(args, parser)
    config = load_config(config_path)
    scenarios = build_scenarios(config)
    summary = scenario_summary(config, scenarios)
    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else OUTPUT_DIR.resolve() / str(config["name"])
    ).expanduser().resolve()
    binary = args.binary.expanduser().resolve()

    print(json.dumps({
        "config": str(config_path),
        "output_dir": str(output_dir),
        "binary": str(binary),
        **summary,
    }, indent=2))
    if args.dry_run:
        return
    if args.concurrency < 1:
        raise ValueError("concurrency must be positive")
    if not binary.is_file():
        raise FileNotFoundError(f"SMP binary does not exist: {binary}")

    output_dir.mkdir(parents=True, exist_ok=True)
    copied_config = output_dir / "sweep_config.json"
    if copied_config.exists() and json.loads(copied_config.read_text()) != config:
        raise RuntimeError(
            f"output directory already contains a different config: {copied_config}"
        )
    # copyfile avoids cross-filesystem metadata operations on the exFAT run disk.
    shutil.copyfile(config_path, copied_config)
    scenario_manifest = output_dir / "resolved_scenarios.jsonl"
    _write_scenario_manifest(scenario_manifest, scenarios)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config_source": str(config_path),
        "config_sha256": _sha256(config_path),
        "binary": str(binary),
        "concurrency": args.concurrency,
        "command": [sys.executable, "-m", __package__ + ".run", *sys.argv[1:]],
        "scenario_manifest": scenario_manifest.name,
        **summary,
    }
    with (output_dir / "resolved_sweep.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    init_logger(None, str(output_dir / "logfile.log"))
    run_scenarios(binary, output_dir, scenarios, concurrency=args.concurrency)


if __name__ == "__main__":
    main()
