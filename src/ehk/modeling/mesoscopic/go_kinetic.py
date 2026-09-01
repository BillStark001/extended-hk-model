"""Thin Python orchestration for the Go mesoscopic kinetic solver.

This module contains request/response translation only.  Numerical evolution is
performed exclusively by ``smp-kinetic`` from social-media-mesoscopic-models.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from smp_meso_bindings.runner_utils import ProgressCallback

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class KineticParameters:
    epsilon: float = 0.45
    influence: float = 0.05
    rewiring: float = 0.025
    dynamics: str = "hk"
    opinion_method: str = "measure"
    mean_degree: int = 15
    recsys_count: int = 10
    recsys: str = "random"
    opinion_tolerance: float = 0.4
    recommendation_steepness: float = 1.0
    recommendation_random_ratio: float = 0.0
    noise_diffusion: float = 0.0
    grid_size: int = 81
    dt: float = 1.0
    steps: int = 1200
    record_every: int = 5
    confidence_mode: str = "cell_average"

    def validate(self) -> None:
        if self.grid_size < 2 or self.steps < 1 or self.record_every < 1:
            raise ValueError("grid_size, steps, and record_every must be positive")
        if not 0 < self.epsilon <= 2:
            raise ValueError("epsilon must lie in (0, 2]")
        if not 0 <= self.influence <= 1 or not 0 <= self.rewiring <= 1:
            raise ValueError("influence and rewiring must lie in [0, 1]")
        if self.dt <= 0 or self.dt * max(self.influence, self.rewiring) > 1 + 1e-12:
            raise ValueError("dt must be positive and both step probabilities <= 1")
        if self.mean_degree < 1 or self.recsys_count < 1:
            raise ValueError("mean_degree and recsys_count must be positive integers")
        if not float(self.mean_degree).is_integer() or not float(self.recsys_count).is_integer():
            raise ValueError("mean_degree and recsys_count must be integer-valued")
        if self.dynamics.casefold() not in {"hk", "deffuant"}:
            raise ValueError(f"unsupported dynamics {self.dynamics!r}")
        if self.opinion_method.casefold() not in {"measure", "fokker_planck"}:
            raise ValueError(f"unsupported opinion method {self.opinion_method!r}")
        if self.recsys.casefold() not in {
            "random", "opinion_random", "structure_random_l0", "structure_random_l1"
        }:
            raise ValueError(f"unsupported Go recommender {self.recsys!r}")


@dataclass(frozen=True)
class KineticResolution:
    population: int = 500
    opinion_min: float = -1.0
    opinion_max: float = 1.0
    opinion_quadrature_points: int = 9
    opinion_quadrature_rule: str = "gauss_hermite"
    confidence_quadrature_points: int = 8
    score_max: int = 64
    distance_grid_size: int = 512
    minimum_bandwidth: float = 0.02
    objective_effective_samples: int = 500


@dataclass(frozen=True)
class ObservableThresholds:
    polarization: float = 0.8
    homophily: float = 0.8


ObservableResolution = KineticResolution


@dataclass(frozen=True)
class ObservableSeries:
    request_id: str
    time: FloatArray
    polarization: FloatArray
    subjective: FloatArray
    homophily: FloatArray
    homophily_raw: FloatArray
    pathway: float
    polarization_first_passage: float
    homophily_first_passage: float


@dataclass
class KineticTrajectory:
    parameters: KineticParameters
    x: FloatArray
    time: FloatArray
    rho: FloatArray
    velocity: FloatArray
    edge: FloatArray
    rewiring_flux: FloatArray

    def metadata(self) -> dict[str, Any]:
        return asdict(self.parameters)


def kinetic_request(
    request_id: str,
    parameters: KineticParameters,
    resolution: KineticResolution,
    thresholds: ObservableThresholds = ObservableThresholds(),
    record_steps: Sequence[int] = (),
) -> dict[str, object]:
    parameters.validate()
    steps = np.asarray(tuple(record_steps), dtype=np.float64)
    snapshots = steps.size > 0
    return {
        "request_id": request_id,
        "population": resolution.population,
        "opinion_bins": parameters.grid_size,
        "out_degree": int(parameters.mean_degree),
        "recommendation_count": int(parameters.recsys_count),
        "steps": parameters.steps,
        "record_every": parameters.record_every,
        "dt": parameters.dt,
        "noise_diffusion": parameters.noise_diffusion,
        "confidence_mode": parameters.confidence_mode,
        "dynamics": {
            "type": parameters.dynamics,
            "opinion_method": parameters.opinion_method,
            "tolerance": parameters.epsilon,
            "influence": parameters.influence,
            "rewiring_rate": parameters.rewiring,
        },
        "recommender": {
            "type": parameters.recsys,
            "steepness": parameters.recommendation_steepness,
            "random_ratio": parameters.recommendation_random_ratio,
            "opinion_tolerance": parameters.opinion_tolerance,
        },
        "initial": {
            "type": "uniform",
            "opinion_min": resolution.opinion_min,
            "opinion_max": resolution.opinion_max,
            "probabilities": np.empty(0, dtype=np.float64),
        },
        "resolution": {
            "opinion_quadrature_points": resolution.opinion_quadrature_points,
            "opinion_quadrature_rule": resolution.opinion_quadrature_rule,
            "confidence_quadrature_points": resolution.confidence_quadrature_points,
            "score_max": resolution.score_max,
            "distance_grid_size": resolution.distance_grid_size,
        },
        "observables": {
            "polarization": True,
            "subjective": True,
            "homophily": True,
            "homophily_raw": True,
            "pathway": True,
            "polarization_first_passage": True,
            "homophily_first_passage": True,
            "polarization_threshold": thresholds.polarization,
            "homophily_threshold": thresholds.homophily,
            "minimum_bandwidth": resolution.minimum_bandwidth,
            "objective_effective_samples": resolution.objective_effective_samples,
        },
        "snapshots": {
            "record_steps": steps,
            "rho": snapshots,
            "edge": snapshots,
            "velocity": snapshots,
            "rewiring_flux": snapshots,
        },
    }


def _passage(summary: dict[str, object], name: str) -> float:
    value = summary[name]
    if not isinstance(value, dict) or not value.get("reached", False):
        return float("nan")
    return float(value["time"])


def observable_series(response: dict[str, object]) -> ObservableSeries:
    result = response["result"]
    if not isinstance(result, dict):
        raise TypeError("kinetic response has no result object")
    series, summary = result["series"], result["summary"]
    if not isinstance(series, dict) or not isinstance(summary, dict):
        raise TypeError("kinetic response has invalid series or summary")
    return ObservableSeries(
        request_id=str(response["request_id"]),
        time=np.asarray(series["time"], dtype=float),
        polarization=np.asarray(series["polarization"], dtype=float),
        subjective=np.asarray(series["subjective"], dtype=float),
        homophily=np.asarray(series["homophily"], dtype=float),
        homophily_raw=np.asarray(series["homophily_raw"], dtype=float),
        pathway=float(summary["pathway"]),
        polarization_first_passage=_passage(summary, "polarization_first_passage"),
        homophily_first_passage=_passage(summary, "homophily_first_passage"),
    )


def trajectory_from_response(
    response: dict[str, object],
    parameters: KineticParameters,
    resolution: KineticResolution,
) -> KineticTrajectory:
    result = response.get("result")
    if not isinstance(result, dict) or not isinstance(result.get("snapshots"), dict):
        raise TypeError("Go response does not contain requested snapshots")
    snapshots = result["snapshots"]
    dx = (resolution.opinion_max - resolution.opinion_min) / parameters.grid_size
    x = resolution.opinion_min + (np.arange(parameters.grid_size) + 0.5) * dx
    return KineticTrajectory(
        parameters=parameters,
        x=x,
        time=np.asarray(snapshots["time"], dtype=float),
        rho=np.asarray(snapshots["rho"], dtype=float),
        velocity=np.asarray(snapshots["velocity"], dtype=float),
        edge=np.asarray(snapshots["edge"], dtype=float),
        rewiring_flux=np.asarray(snapshots["rewiring_flux"], dtype=float),
    )


def solve_trajectory(
    parameters: KineticParameters,
    *,
    record_steps: Sequence[int] | None = None,
    binary_path: PathLike[str] | str | None = None,
    resolution: KineticResolution = KineticResolution(),
) -> KineticTrajectory:
    from smp_meso_bindings import run_kinetic

    if binary_path is None:
        configured = os.environ.get("SMP_KINETIC_BINARY")
        binary_path = (
            Path(configured).expanduser()
            if configured
            else Path(__file__).resolve().parents[5]
            / "social-media-mesoscopic-models"
            / "bin"
            / "smp-kinetic"
        )
    if record_steps is None:
        selected = list(range(0, parameters.steps + 1, parameters.record_every))
        if selected[-1] != parameters.steps:
            selected.append(parameters.steps)
        record_steps = selected
    request = kinetic_request("trajectory", parameters, resolution, record_steps=record_steps)
    response = run_kinetic(binary_path, request)
    return trajectory_from_response(response, parameters, resolution)


# The short public name is intentionally only an external-process adapter; no
# numerical mesoscopic implementation remains in this repository.
solve = solve_trajectory


def solve_trajectory_batch(
    binary_path: PathLike[str] | str,
    cases: list[tuple[str, KineticParameters, Sequence[int]]],
    resolution: KineticResolution,
    processes: int,
    progress: ProgressCallback | None,
) -> list[KineticTrajectory]:
    """Decode requested field snapshots from long-lived Go batch processes."""

    from smp_meso_bindings import run_kinetic_batch, run_kinetic_batch_parallel

    if processes < 1:
        raise ValueError("processes must be positive")
    requests = [
        kinetic_request(key, parameters, resolution, record_steps=record_steps)
        for key, parameters, record_steps in cases
    ]
    if processes == 1:
        responses = run_kinetic_batch(binary_path, requests, progress=progress)
    else:
        responses = run_kinetic_batch_parallel(
            binary_path, requests, min(processes, len(requests)), progress=progress
        )
    return [
        trajectory_from_response(response, parameters, resolution)
        for response, (_, parameters, _) in zip(responses, cases, strict=True)
    ]


def solve_observable_batch(
    binary_path: PathLike[str] | str,
    cases: list[tuple[str, KineticParameters]],
    resolution: KineticResolution,
    thresholds: ObservableThresholds,
    processes: int,
    progress: ProgressCallback | None,
) -> list[ObservableSeries]:
    from smp_meso_bindings import run_kinetic_batch, run_kinetic_batch_parallel

    requests = [kinetic_request(key, params, resolution, thresholds) for key, params in cases]
    if processes == 1:
        responses = run_kinetic_batch(binary_path, requests, progress=progress)
    else:
        responses = run_kinetic_batch_parallel(
            binary_path, requests, processes, progress=progress
        )
    return [observable_series(response) for response in responses]
