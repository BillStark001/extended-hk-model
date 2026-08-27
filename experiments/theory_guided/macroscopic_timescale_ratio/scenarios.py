"""Scenario and numerical-protocol definitions for the B=81 scan."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from theory.mesoscopic.phase_scan import RATES


@dataclass(frozen=True)
class RecommendationScenario:
  """One recommendation closure represented in the seven-column figure."""

  key: str
  recsys: str
  steepness: float
  display_name: str
  closure_note: str = ""


SCENARIOS = (
    RecommendationScenario("random", "random", 1.0, "Random"),
    RecommendationScenario(
        "opinion_random_zeta1",
        "opinion_random",
        1.0,
        r"OpinionRandom ($\zeta=1$)",
    ),
    RecommendationScenario(
        "opinion_random_zeta4",
        "opinion_random",
        4.0,
        r"OpinionRandom ($\zeta=4$)",
    ),
    RecommendationScenario(
        "structure_random_l0_zeta1",
        "structure_random_l0",
        1.0,
        r"L0-StructureRandom ($\zeta=1$)",
    ),
    RecommendationScenario(
        "structure_random_l0_zeta4",
        "structure_random_l0",
        4.0,
        r"L0-StructureRandom ($\zeta=4$)",
    ),
    RecommendationScenario(
        "structure_random_l1_zeta1",
        "structure_random_l1_mean_power",
        1.0,
        r"L1-StructureRandom ($\zeta=1$)",
        "exact first-score-moment closure",
    ),
    RecommendationScenario(
        "structure_random_l1_zeta4",
        "structure_random_l1_mean_power",
        4.0,
        r"L1-StructureRandom ($\zeta=4$)",
        "mean-power first-score-moment closure",
    ),
)
SCENARIO_BY_KEY = {scenario.key: scenario for scenario in SCENARIOS}
PAPER_RATES = np.asarray(RATES, dtype=float)


def record_schedule(
    steps: int,
    *,
    early_until: int = 200,
    early_every: int = 1,
    late_every: int = 20,
) -> tuple[int, ...]:
  """Combine dense early diagnostics with the paper's 20-step cadence."""

  if steps < 1 or early_until < 0 or early_every < 1 or late_every < 1:
    raise ValueError("invalid record schedule")
  split = min(steps, early_until)
  selected = set(range(0, split + 1, early_every))
  selected.update(range(split, steps + 1, late_every))
  selected.update((0, split, steps))
  return tuple(sorted(selected))


def select_scenarios(keys: list[str] | None) -> tuple[RecommendationScenario, ...]:
  """Resolve an optional ordered subset while rejecting duplicate keys."""

  if keys is None:
    return SCENARIOS
  if len(keys) != len(set(keys)):
    raise ValueError("scenario keys must not be repeated")
  return tuple(SCENARIO_BY_KEY[key] for key in keys)


__all__ = [
    "PAPER_RATES",
    "SCENARIOS",
    "SCENARIO_BY_KEY",
    "RecommendationScenario",
    "record_schedule",
    "select_scenarios",
]
