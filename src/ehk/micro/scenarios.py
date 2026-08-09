"""Reusable scenario metadata helpers for the external SMP runtime."""

from __future__ import annotations

from typing import Any, TypeAlias


ScenarioMetadata: TypeAlias = dict[str, Any]


def create_scenario_metadata(
    name: str,
    tolerance: float = 0.45,
    influence: float = 0.9,
    rewiring: float = 0.01,
    repost_rate: float = 0.05,
    recsys_type: str = "Random",
    recsys_count: int = 10,
    post_retain_count: int = 3,
    max_sim_step: int = 20_000,
) -> ScenarioMetadata:
    return {
        "UniqueName": name,
        "DynamicsType": "HK",
        "HKParams": {
            "Tolerance": tolerance,
            "Influence": influence,
            "RewiringRate": rewiring,
            "RepostRate": repost_rate,
        },
        "RecsysFactoryType": recsys_type,
        "RecsysCount": recsys_count,
        "PostRetainCount": post_retain_count,
        "MaxSimulationStep": max_sim_step,
    }
