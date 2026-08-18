"""Presets and scenario construction for terminal-probability experiments."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable

from ehk.micro.scenarios import ScenarioMetadata


HERE = Path(__file__).resolve().parent
PRESETS = {
    "paper-figure3": HERE / "configs" / "paper_figure3.json",
    "smoke": HERE / "configs" / "smoke.json",
}

PAPER_RATE_GRID = tuple(10 ** (-3 + index / 3) for index in range(10))


@dataclass(frozen=True)
class RecommendationScenario:
    key: str
    factory: str
    spectrum_rule: str
    steepness: float
    display: str


RECOMMENDATION_SCENARIOS = (
    RecommendationScenario("random", "Random", "Random", 1.0, "Random"),
    RecommendationScenario(
        "opinion_random_zeta1", "OpinionRandom", "opinion_random", 1.0,
        "OpinionRandom\n$\\zeta=1$",
    ),
    RecommendationScenario(
        "opinion_random_zeta4", "OpinionRandom", "opinion_random", 4.0,
        "OpinionRandom\n$\\zeta=4$",
    ),
    RecommendationScenario(
        "structure_random_zeta1", "StructureRandom", "structure_random_l0", 1.0,
        "L0 StructureRandom\n$\\zeta=1$",
    ),
    RecommendationScenario(
        "structure_random_zeta4", "StructureRandom", "structure_random_l0", 4.0,
        "L0 StructureRandom\n$\\zeta=4$",
    ),
)


@dataclass(frozen=True)
class ComparisonCase:
    key: str
    alpha_index: int
    q_index: int
    display: str

    @property
    def alpha(self) -> float:
        return PAPER_RATE_GRID[self.alpha_index]

    @property
    def rewiring(self) -> float:
        return PAPER_RATE_GRID[self.q_index]


PAPER_COMPARISON_CASES = (
    ComparisonCase("low_rewiring", 5, 0, "low $q$"),
    ComparisonCase("balanced", 5, 5, "balanced"),
    ComparisonCase("influence_dominant", 7, 2, "$\\alpha$-dominant"),
    ComparisonCase("rewiring_dominant", 2, 7, "$q$-dominant"),
)


def available_presets() -> tuple[str, ...]:
    return tuple(PRESETS)


def preset_path(name: str) -> Path:
    try:
        return PRESETS[name]
    except KeyError as error:
        choices = ", ".join(available_presets())
        raise ValueError(f"unknown preset {name!r}; choose one of: {choices}") from error


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    required = {"name", "replicates", "model", "configurations"}
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"configuration is missing: {', '.join(missing)}")
    has_rates = "rates" in config
    has_cases = "cases" in config
    if has_rates == has_cases:
        raise ValueError("configuration must define exactly one of rates or cases")
    return config


def rate_values(specification: dict[str, Any]) -> tuple[float, ...]:
    if "values" in specification:
        values = tuple(float(value) for value in specification["values"])
    else:
        minimum = float(specification["minimum"])
        maximum = float(specification["maximum"])
        levels = int(specification["levels"])
        if minimum <= 0 or maximum <= 0 or levels < 2:
            raise ValueError(
                "logarithmic rates require positive bounds and levels >= 2"
            )
        log_minimum = math.log10(minimum)
        log_maximum = math.log10(maximum)
        values = tuple(
            10 ** (
                log_minimum
                + index * (log_maximum - log_minimum) / (levels - 1)
            )
            for index in range(levels)
        )
    if not values or any(not 0 <= value <= 1 for value in values):
        raise ValueError("all influence and rewiring rates must lie in [0, 1]")
    return values


def _safe_key(value: object, *, field: str) -> str:
    key = str(value)
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", key):
        raise ValueError(
            f"{field} {key!r} must use lowercase letters, digits, '_' or '-'"
        )
    return key


def _fixed_rng(seed_namespace: str) -> dict[str, str]:
    digest = hashlib.blake2b(
        seed_namespace.encode("utf-8"), digest_size=16
    ).digest()
    seed1 = int.from_bytes(digest[:8], "big")
    seed2 = int.from_bytes(digest[8:], "big")
    return {
        "Algorithm": "pcg64-dxsm-v1",
        "Seed1": f"0x{seed1:016x}",
        "Seed2": f"0x{seed2:016x}",
    }


def _rate_cells(config: dict[str, Any]) -> list[tuple[str, str, float, float]]:
    if "rates" in config:
        rates = config["rates"]
        influences = rate_values(rates["influence"])
        rewiring_rates = rate_values(rates["rewiring"])
        return [
            (
                f"a{alpha_index:02d}_q{q_index:02d}",
                f"a{alpha_index:02d}_q{q_index:02d}",
                alpha,
                rewiring,
            )
            for alpha_index, alpha in enumerate(influences)
            for q_index, rewiring in enumerate(rewiring_rates)
        ]

    cells: list[tuple[str, str, float, float]] = []
    for case in config["cases"]:
        key = _safe_key(case["key"], field="case key")
        rng_key = _safe_key(case.get("rng_key", key), field="case RNG key")
        alpha = float(case["influence"])
        rewiring = float(case["rewiring"])
        if not 0 <= alpha <= 1 or not 0 <= rewiring <= 1:
            raise ValueError(f"rates for case {key!r} must lie in [0, 1]")
        cells.append((key, rng_key, alpha, rewiring))
    if not cells or len({cell[0] for cell in cells}) != len(cells):
        raise ValueError("case keys must be nonempty and unique")
    return cells


def build_scenarios(config: dict[str, Any]) -> list[ScenarioMetadata]:
    """Resolve a preset into production SMP metadata.

    A cell/replicate uses one root RNG specification for all recommender
    columns. This implements common random numbers without changing a
    column's marginal initial-state ensemble.
    """

    replicates = int(config["replicates"])
    if replicates < 1:
        raise ValueError("replicates must be positive")

    model = config["model"]
    node_count = int(model["node_count"])
    follow_count = int(model["node_follow_count"])
    if node_count < 2 or not 1 <= follow_count < node_count:
        raise ValueError("node_follow_count must lie in [1, node_count)")

    configurations = config["configurations"]
    configuration_keys = [
        _safe_key(item["key"], field="configuration key")
        for item in configurations
    ]
    if not configurations or len(configuration_keys) != len(set(configuration_keys)):
        raise ValueError("configuration keys must be nonempty and unique")

    prefix = _safe_key(config.get("scenario_prefix", "tp"), field="scenario prefix")
    rng_namespace = str(config.get("rng_namespace", config["name"]))
    scenarios: list[ScenarioMetadata] = []
    for cell_key, rng_key, alpha, rewiring in _rate_cells(config):
        for replicate in range(replicates):
            seed_namespace = f"{rng_namespace}|{rng_key}|r={replicate}"
            rng = _fixed_rng(seed_namespace)
            for recommendation, configuration_key in zip(
                configurations, configuration_keys, strict=True
            ):
                steepness = float(recommendation.get("steepness", 1.0))
                if steepness <= 0:
                    raise ValueError("recommendation steepness must be positive")
                scenarios.append(
                    {
                        "DataVersion": 1,
                        "RNG": rng,
                        "UniqueName": (
                            f"{prefix}_{cell_key}_r{replicate:03d}_{configuration_key}"
                        ),
                        "DynamicsType": "HK",
                        "HKParams": {
                            "Tolerance": float(model["confidence_radius"]),
                            "Influence": alpha,
                            "RewiringRate": rewiring,
                            "RepostRate": 0.0,
                        },
                        "RecsysFactoryType": recommendation["factory"],
                        "RecSysParams": {
                            "NoiseStd": float(model["score_noise_std"]),
                            "OpRandomNoiseStd": float(model["score_noise_std"]),
                            "UseCache": True,
                            "Tolerance": float(model["opinion_tolerance"]),
                            "Steepness": steepness,
                            "RandomRatio": float(model["random_ratio"]),
                        },
                        "RecsysCount": int(model["recommendation_count"]),
                        "PostRetainCount": int(model["post_retain_count"]),
                        "NetworkType": "Random",
                        "NodeCount": node_count,
                        "NodeFollowCount": follow_count,
                        "MaxSimulationStep": int(model["maximum_steps"]),
                    }
                )
    return scenarios


def scenario_summary(
    config: dict[str, Any], scenarios: Iterable[ScenarioMetadata]
) -> dict[str, Any]:
    rows = list(scenarios)
    return {
        "name": config["name"],
        "scenario_count": len(rows),
        "rate_cell_count": len(_rate_cells(config)),
        "replicates": int(config["replicates"]),
        "configuration_keys": [
            str(item["key"]) for item in config["configurations"]
        ],
        "first_scenario": rows[0] if rows else None,
        "last_scenario": rows[-1] if rows else None,
    }
