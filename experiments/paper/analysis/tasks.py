import os
from dataclasses import dataclass
from functools import partial
from typing import Callable

import numpy as np

import experiments.paper.scenarios as cfg
from experiments.paper.paths import STAT_INPUT
from experiments.workspaces import instance_name as resolve_instance_name
from experiments.workspaces import workspace_dir
from ehk.micro.pipeline import Context
from ehk.micro.observables import c
from experiments.paper.analysis.execution import generate_stats
from experiments.paper.analysis.records import ScenarioStatistics


def merge_stats_to_context(
    stats: ScenarioStatistics | None,
    ctx: Context,
    ctx_name_to_stat_name: dict[str, str] | None = None,
    include_names: list[str] | None = None,
    exclude_names: list[str] | None = None,
):
  if stats is None:
    return

  sel_names = set(include_names if include_names else ctx.get_state_names())
  if exclude_names:
    sel_names = sel_names - set(exclude_names)

  state_dict = {}
  if ctx_name_to_stat_name is None:
    ctx_name_to_stat_name = {}
  for sel_name in sel_names:
    stat_name = ctx_name_to_stat_name[sel_name] if sel_name in ctx_name_to_stat_name else sel_name
    if hasattr(stats, stat_name):
      value = getattr(stats, stat_name)
      if value is not None:
        state_dict[sel_name] = value
  ctx.set_state(**state_dict)


# These 3 lists control which common statistics are present for each mode.
EPS_COMMON_STATS = [
    "step",
    "active_step",
    "active_step_threshold",
    "g_index_mean_active",
    "x_indices",
    "h_index",
    "p_index",
    "g_index",
    "grad_index",
    "event_count",
    "event_step_mean",
    "triads",
    "x_mean_vars",
    "mean_vars_smpl",
    "last_community_count",
    "last_community_sizes",
    "last_opinion_peak_count",
]

GRAD_COMMON_STATS = [
    "step",
    "active_step",
    "active_step_threshold",
    "g_index_mean_active",
    "x_indices",
    "h_index",
    "p_index",
    "g_index",
    "grad_index",
    "event_count",
    "event_step_mean",
    "triads",
    "x_mean_vars",
    "mean_vars_smpl",
    "last_opinion_peak_count",
]

REP_COMMON_STATS = [
    "step",
    "active_step",
    "active_step_threshold",
    "g_index_mean_active",
    "x_indices",
    "h_index",
    "p_index",
    "g_index",
    "grad_index",
    "event_count",
    "event_step_mean",
    "triads",
    "x_mean_vars",
    "mean_vars_smpl",
    "last_community_count",
    "last_community_sizes",
    "last_opinion_peak_count",
]


def _extract_hk_params(scenario_metadata: "cfg.ScenarioMetadata") -> dict:
  return scenario_metadata["HKParams"]


def _skip_if_peak_exists(exist_stats: ScenarioStatistics | None) -> bool:
  return exist_stats is not None and exist_stats.last_opinion_peak_count is not None


def _skip_gradation(exist_stats: ScenarioStatistics | None) -> bool:
  return exist_stats is not None


def _extra_stats_none() -> dict[str, object]:
  return {}


def _extra_stats_gradation() -> dict[str, object]:
  return {
      "p_backdrop": c.p_backdrop,
      "h_backdrop": c.h_backdrop,
      "g_backdrop": c.g_backdrop,
      "opinion_diff_seg_mean": c.opinion_diff_seg_mean,
      "opinion_diff_seg_std": c.opinion_diff_seg_std,
  }


COMMON_STAT_BUILDERS: dict[str, Callable[[], object]] = {
    "step": lambda: c.total_steps,
    "active_step": lambda: c.active_step,
    "active_step_threshold": lambda: c.active_step_threshold,
    "g_index_mean_active": lambda: c.g_index_mean_active,
    "x_indices": lambda: c.x_indices,
    "h_index": lambda: c.h_index,
    "p_index": lambda: c.p_index,
    "g_index": lambda: c.g_index,
    "grad_index": lambda: c.gradation_index_hp,
    "event_count": lambda: c.event_step.size,
    "event_step_mean": lambda: float(np.mean(c.event_step)),
    "triads": lambda: c.n_triads,
    "x_mean_vars": lambda: c.x_mean_vars,
    "mean_vars_smpl": lambda: c.mean_vars_smpl,
    "last_community_count": lambda: c.last_community_count,
    "last_community_sizes": lambda: c.last_community_sizes,
    "last_opinion_peak_count": lambda: c.last_opinion_peak_count,
}


@dataclass(frozen=True)
class ModeConfig:
  name: str
  scenarios: list[cfg.ScenarioMetadata]
  common_stats: list[str]
  opinion_peak_distance: int
  skip_exist: Callable[[ScenarioStatistics | None], bool]
  extra_stats_builder: Callable[[], dict[str, object]]


MODE_CONFIGS: dict[str, ModeConfig] = {
    "epsilon": ModeConfig(
        name="epsilon",
        scenarios=cfg.all_scenarios_eps,
        common_stats=EPS_COMMON_STATS,
        opinion_peak_distance=20,
        skip_exist=_skip_if_peak_exists,
        extra_stats_builder=_extra_stats_none,
    ),
    "gradation": ModeConfig(
        name="gradation",
        scenarios=cfg.all_scenarios_grad,
        common_stats=GRAD_COMMON_STATS,
        opinion_peak_distance=50,
        skip_exist=_skip_gradation,
        extra_stats_builder=_extra_stats_gradation,
    ),
    "replicate": ModeConfig(
        name="replicate",
        scenarios=cfg.all_scenarios_rep,
        common_stats=REP_COMMON_STATS,
        opinion_peak_distance=50,
        skip_exist=_skip_if_peak_exists,
        extra_stats_builder=_extra_stats_none,
    ),
}


def _build_common_stats(selected_names: list[str]) -> dict[str, object]:
  stats: dict[str, object] = {}
  for name in selected_names:
    if name not in COMMON_STAT_BUILDERS:
      raise ValueError(f"Unknown common stat name: {name}")
    stats[name] = COMMON_STAT_BUILDERS[name]()
  return stats


def get_statistics_for_mode(
    scenario_metadata: "cfg.ScenarioMetadata",
    scenario_base_path: str,
    origin: str,
    exist_stats: ScenarioStatistics | None,
    mode: str,
    active_threshold=0.98,
    min_inactive_value=0.75,
):
  from smp_bindings import RawSimulationRecord

  config = MODE_CONFIGS[mode]
  if config.skip_exist(exist_stats):
    return

  scenario_name = scenario_metadata["UniqueName"]

  scenario_record = RawSimulationRecord(
      scenario_base_path,
      scenario_metadata,
  )

  with scenario_record:

    if not scenario_record.is_finished:
      return None

    assert scenario_record.is_sanitized, "non-sanitized scenario"

    c.set_state(
        scenario_record=scenario_record,
        active_threshold=active_threshold,
        min_inactive_value=min_inactive_value,
        opinion_peak_distance=config.opinion_peak_distance,
    )

    merge_stats_to_context(
        exist_stats,
        c,
        {
            "total_steps": "step",
            "gradation_index_hp": "grad_index",
            "n_triads": "triads",
        },
        exclude_names=["id", "name", "origin"],
    )

    hk_params = _extract_hk_params(scenario_metadata)
    selected_common_stats = _build_common_stats(config.common_stats)
    selected_extra_stats = config.extra_stats_builder()

    return ScenarioStatistics(
        id=exist_stats.id if exist_stats else None,
        name=scenario_name,
        origin=origin,
        tolerance=hk_params["Tolerance"],
        decay=hk_params["Influence"],
        rewiring=hk_params["RewiringRate"],
        retweet=hk_params["RepostRate"],
        recsys_type=scenario_metadata["RecsysFactoryType"],
        tweet_retain_count=scenario_metadata["PostRetainCount"],
        **selected_common_stats,
        **selected_extra_stats,
    )


def get_mode_names() -> list[str]:
  return sorted(MODE_CONFIGS.keys())


def run_mode(mode: str, instance_name: str, concurrency: int):
  if mode not in MODE_CONFIGS:
    valid = ", ".join(get_mode_names())
    raise ValueError(f"Unsupported mode: {mode}. expected one of: {valid}")

  plot_path = str(STAT_INPUT.resolve())
  os.makedirs(plot_path, exist_ok=True)
  stats_db_path = os.path.join(plot_path, "stats.db")

  scenario_base_path = str(workspace_dir(instance_name))
  os.makedirs(scenario_base_path, exist_ok=True)

  c.set_state(active_threshold=0.98, min_inactive_value=0.75)

  mode_config = MODE_CONFIGS[mode]
  get_statistics = partial(get_statistics_for_mode, mode=mode)

  generate_stats(
      get_statistics,
      scenario_base_path,
      stats_db_path,
      resolve_instance_name(instance_name),
      mode_config.scenarios,
      ignore_exist=False,
      concurrency=concurrency,
  )
