"""Configuration for the paper's counterfactual landscape comparison."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ehk.common.settings import register_path


NORMALIZED_TIMES = tuple(i / 10 for i in range(11))

# Use common random numbers across scenarios to reduce Monte Carlo noise in
# their comparison. Seeds are strings because the Go/Python protocol preserves
# uint64 values exactly in this representation.
PROBE_RNG = {
    "Algorithm": "pcg64-dxsm-v1",
    "Seed1": "0x4c414e4453434150",
    "Seed2": "0x50524f4245323032",
}

INPUT_DIR = register_path(
    "EHK_SOCIAL_FORCE_INPUT_DIR", "outputs/experiments/paper/mechanism_cases/raw"
)
PROBE_BINARY = register_path(
    "SMP_PROBE_BINARY", "../social-media-models/smp-probe"
)
OUTPUT_DIR = register_path(
    "EHK_SOCIAL_FORCE_OUTPUT_DIR",
    "outputs/experiments/theory_guided/social_force_probe",
)


@dataclass(frozen=True)
class LandscapeScenario:
  key: str
  unique_name: str
  active_step: int
  label: str
  influence: float
  rewiring_rate: float
  repost_rate: float = 0.0
  recsys_factory_type: str = "Random"
  post_retain_count: int = 3

  def metadata(self) -> dict[str, Any]:
    return {
        "UniqueName": self.unique_name,
        "DynamicsType": "HK",
        "HKParams": {
            "Tolerance": 0.45,
            "Influence": self.influence,
            "RewiringRate": self.rewiring_rate,
            "RepostRate": self.repost_rate,
        },
        "RecsysFactoryType": self.recsys_factory_type,
        "RecsysCount": 10,
        "PostRetainCount": self.post_retain_count,
        # The legacy run has no metadata.json, but its events.db contains the
        # rewiring history needed to reconstruct intermediate frozen graphs.
        "RewiringEvent": True,
        "PostEvent": True,
    }


# These are the records used by f_mech_map_baseline.py and
# Preserve the order used by the paper mechanism scenarios.
# active_step uses the existing paper definition with active_threshold=0.98
# and min_inactive_value=0.75. Values already used by the observational
# figures' cache are retained exactly.
SCENARIOS: dict[str, LandscapeScenario] = {
    "baseline": LandscapeScenario(
        key="baseline",
        unique_name="s_mech_baseline",
        active_step=352,
        label="baseline",
        influence=0.05,
        rewiring_rate=0.05,
    ),
    "pbs": LandscapeScenario(
        key="pbs",
        unique_name="s_mech_baseline_pbs",
        active_step=786,
        label="PbS baseline",
        influence=0.05,
        rewiring_rate=0.025,
    ),
    "sbp": LandscapeScenario(
        key="sbp",
        unique_name="s_mech_baseline_sbp",
        active_step=2572,
        label="SbP baseline",
        influence=0.005,
        rewiring_rate=0.025,
    ),
    "influence": LandscapeScenario(
        key="influence",
        unique_name="s_mech_influence",
        active_step=441,
        label="+influence",
        influence=0.1,
        rewiring_rate=0.05,
    ),
    "retweet": LandscapeScenario(
        key="retweet",
        unique_name="s_mech_retweet",
        active_step=3249,
        label="+retweet",
        influence=0.05,
        rewiring_rate=0.05,
        repost_rate=0.1,
    ),
    "op_recsys": LandscapeScenario(
        key="op_recsys",
        unique_name="s_mech_op_recsys",
        active_step=605,
        label="opinion rec.",
        influence=0.05,
        rewiring_rate=0.05,
        recsys_factory_type="OpinionM9",
    ),
    "phase1": LandscapeScenario(
        key="phase1",
        unique_name="s_mech_phase1",
        active_step=2254,
        label="phase 1",
        influence=0.005,
        rewiring_rate=1.0,
        repost_rate=0.5,
        recsys_factory_type="StructureM9",
        post_retain_count=0,
    ),
    "phase2": LandscapeScenario(
        key="phase2",
        unique_name="s_mech_phase2",
        active_step=13192,
        label="phase 2",
        influence=0.005,
        rewiring_rate=1.0,
        recsys_factory_type="StructureM9",
        post_retain_count=0,
    ),
    "phase3": LandscapeScenario(
        key="phase3",
        unique_name="s_mech_phase3",
        active_step=301,
        label="phase 3",
        influence=0.1,
        rewiring_rate=0.1,
        recsys_factory_type="OpinionM9",
        post_retain_count=6,
    ),
    "phase4": LandscapeScenario(
        key="phase4",
        unique_name="s_mech_phase4",
        active_step=679,
        label="phase 4",
        influence=1.0,
        rewiring_rate=0.03,
        recsys_factory_type="OpinionM9",
        post_retain_count=6,
    ),
    "phase5": LandscapeScenario(
        key="phase5",
        unique_name="s_mech_phase5",
        active_step=14,
        label="phase 5",
        influence=1.0,
        rewiring_rate=0.03,
        recsys_factory_type="StructureM9",
        post_retain_count=0,
    ),
}

SCENARIO_ORDER = tuple(SCENARIOS)
