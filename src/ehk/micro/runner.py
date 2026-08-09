"""Thin adapter around the external SMP simulation runtime."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from ehk.micro.scenarios import ScenarioMetadata


def run_scenarios(
    binary: Path,
    workspace: Path,
    scenarios: Sequence[ScenarioMetadata],
    concurrency: int = 4,
) -> None:
    from smp_bindings.simulation import run_simulations

    workspace.mkdir(parents=True, exist_ok=True)
    run_simulations(
        str(binary),
        str(workspace),
        scenarios,
        max_concurrent=concurrency,
        show_position=True,
    )
