"""Thin Python orchestration for the Go mesoscopic kinetic solver.

This module contains request/response translation only.  Numerical evolution is
performed exclusively by ``smp-kinetic`` from social-media-mesoscopic-models.
"""

from __future__ import annotations

import math
import os
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, Any

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


@dataclass(frozen=True)
class KineticStopping:
    """Early-stopping controls; ``steps`` remains the hard safety ceiling."""

    mode: str = "fixed_steps"
    minimum_steps: int = 0
    check_every: int = 1
    patience_steps: int = 1
    state_l1_tolerance: float = 0.0
    energy_absolute_tolerance: float = 0.0
    energy_relative_tolerance: float = 0.0

    def validate(self, maximum_steps: int) -> None:
        if self.mode.casefold() not in {
            "fixed_steps", "state", "energy", "state_and_energy", "state_or_energy"
        }:
            raise ValueError(f"unsupported stopping mode {self.mode!r}")
        if not 0 <= self.minimum_steps <= maximum_steps:
            raise ValueError("minimum_steps must lie in [0, steps]")
        if self.check_every < 1 or self.patience_steps < 1:
            raise ValueError("check_every and patience_steps must be positive")
        if self.mode.casefold() != "fixed_steps" and self.check_every > maximum_steps:
            raise ValueError("adaptive check_every must not exceed steps")
        tolerances = (
            self.state_l1_tolerance,
            self.energy_absolute_tolerance,
            self.energy_relative_tolerance,
        )
        if any(not math.isfinite(value) or value < 0 for value in tolerances):
            raise ValueError("stopping tolerances must be finite and nonnegative")


ObservableResolution = KineticResolution


@dataclass(frozen=True)
class ObservableSeries:
    request_id: str
    time: FloatArray
    polarization: FloatArray
    subjective: FloatArray
    homophily: FloatArray
    homophily_raw: FloatArray
    node_energy: FloatArray
    edge_energy: FloatArray
    pathway: float
    polarization_first_passage: float
    homophily_first_passage: float
    max_node_mass_residual: float
    max_fixed_degree_residual: float


@dataclass(frozen=True)
class DensityVelocitySeries:
    """Only the kinetic fields needed to reconstruct an opinion landscape."""

    request_id: str
    parameters: KineticParameters
    x: FloatArray
    time: FloatArray
    rho: FloatArray
    velocity: FloatArray
    pathway: float
    final_rho: FloatArray | None
    final_edge: FloatArray | None
    max_node_mass_residual: float
    max_fixed_degree_residual: float


@dataclass(frozen=True)
class MultimetricSeries:
    """Scalar energy history plus final fields for method-validity metrics."""

    request_id: str
    parameters: KineticParameters
    time: FloatArray
    node_energy: FloatArray
    edge_energy: FloatArray
    pathway: float
    final_rho: FloatArray
    final_edge: FloatArray
    max_node_mass_residual: float
    max_fixed_degree_residual: float


@dataclass
class KineticTrajectory:
    parameters: KineticParameters
    x: FloatArray
    time: FloatArray
    rho: FloatArray
    velocity: FloatArray
    edge: FloatArray
    rewiring_flux: FloatArray
    node_potential: FloatArray
    edge_potential: FloatArray

    def metadata(self) -> dict[str, Any]:
        return asdict(self.parameters)


def kinetic_request(
    request_id: str,
    parameters: KineticParameters,
    resolution: KineticResolution,
    thresholds: ObservableThresholds | None = None,
    record_steps: Sequence[int] = (),
    snapshot_fields: Sequence[str] | None = None,
    final_snapshot_fields: Sequence[str] = (),
    observable_fields: Sequence[str] | None = None,
    stopping: KineticStopping | None = None,
) -> dict[str, object]:
    parameters.validate()
    thresholds = thresholds or ObservableThresholds()
    stopping = stopping or KineticStopping()
    stopping.validate(parameters.steps)
    steps = np.asarray(tuple(record_steps), dtype=np.float64)
    snapshots = steps.size > 0
    available_snapshot_fields = {
        "rho", "edge", "velocity", "rewiring_flux", "node_potential", "edge_potential",
    }
    selected_snapshot_fields = (
        available_snapshot_fields
        if snapshots and snapshot_fields is None
        else set(snapshot_fields or ())
    )
    unknown_snapshot_fields = selected_snapshot_fields - available_snapshot_fields
    if unknown_snapshot_fields:
        raise ValueError(
            "unknown snapshot fields: " + ", ".join(sorted(unknown_snapshot_fields))
        )
    available_final_fields = {"rho", "edge", "node_potential", "edge_potential"}
    selected_final_fields = set(final_snapshot_fields)
    unknown_final_fields = selected_final_fields - available_final_fields
    if unknown_final_fields:
        raise ValueError(
            "unknown final snapshot fields: " + ", ".join(sorted(unknown_final_fields))
        )
    default_observable_fields = {
        "polarization", "subjective", "homophily", "homophily_raw", "pathway",
        "polarization_first_passage", "homophily_first_passage",
    }
    available_observable_fields = default_observable_fields | {"node_energy", "edge_energy"}
    selected_observable_fields = (
        default_observable_fields
        if observable_fields is None
        else set(observable_fields)
    )
    unknown_observable_fields = selected_observable_fields - available_observable_fields
    if unknown_observable_fields:
        raise ValueError(
            "unknown observable fields: "
            + ", ".join(sorted(unknown_observable_fields))
        )
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
            **{
                field: field in selected_observable_fields
                for field in sorted(available_observable_fields)
            },
            "polarization_threshold": thresholds.polarization,
            "homophily_threshold": thresholds.homophily,
            "minimum_bandwidth": resolution.minimum_bandwidth,
            "objective_effective_samples": resolution.objective_effective_samples,
        },
        "snapshots": {
            "record_steps": steps,
            **{
                field: snapshots and field in selected_snapshot_fields
                for field in sorted(available_snapshot_fields)
            },
            "final_rho": "rho" in selected_final_fields,
            "final_edge": "edge" in selected_final_fields,
            "final_node_potential": "node_potential" in selected_final_fields,
            "final_edge_potential": "edge_potential" in selected_final_fields,
        },
        "stopping": asdict(stopping),
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
    series, summary, diagnostics = (
        result["series"], result["summary"], result.get("diagnostics", {})
    )
    if not isinstance(series, dict) or not isinstance(summary, dict) or not isinstance(diagnostics, dict):
        raise TypeError("kinetic response has invalid series or summary")
    return ObservableSeries(
        request_id=str(response["request_id"]),
        time=np.asarray(series["time"], dtype=float),
        polarization=np.asarray(series["polarization"], dtype=float),
        subjective=np.asarray(series["subjective"], dtype=float),
        homophily=np.asarray(series["homophily"], dtype=float),
        homophily_raw=np.asarray(series["homophily_raw"], dtype=float),
        node_energy=np.asarray(series.get("node_energy", ()), dtype=float),
        edge_energy=np.asarray(series.get("edge_energy", ()), dtype=float),
        pathway=float(summary["pathway"]),
        polarization_first_passage=_passage(summary, "polarization_first_passage"),
        homophily_first_passage=_passage(summary, "homophily_first_passage"),
        max_node_mass_residual=float(diagnostics.get("max_node_mass_residual", math.nan)),
        max_fixed_degree_residual=float(diagnostics.get("max_fixed_degree_residual", math.nan)),
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
        node_potential=np.asarray(snapshots["node_potential"], dtype=float),
        edge_potential=np.asarray(snapshots["edge_potential"], dtype=float),
    )


def density_velocity_series(
    response: dict[str, object],
    parameters: KineticParameters,
    resolution: KineticResolution,
) -> DensityVelocitySeries:
    """Decode the minimal fields and conservation diagnostics for landscapes."""

    result = response.get("result")
    if not isinstance(result, dict) or not isinstance(result.get("snapshots"), dict):
        raise TypeError("Go response does not contain requested snapshots")
    snapshots = result["snapshots"]
    summary = result.get("summary", {})
    diagnostics = result.get("diagnostics", {})
    if not isinstance(summary, dict) or not isinstance(diagnostics, dict):
        raise TypeError("Go response has invalid summary or diagnostics")
    dx = (resolution.opinion_max - resolution.opinion_min) / parameters.grid_size
    x = resolution.opinion_min + (np.arange(parameters.grid_size) + 0.5) * dx
    time_values = np.asarray(snapshots["time"], dtype=float)
    rho = np.asarray(snapshots["rho"], dtype=float)
    velocity = np.asarray(snapshots["velocity"], dtype=float)
    expected_shape = (time_values.size, parameters.grid_size)
    if rho.shape != expected_shape or velocity.shape != expected_shape:
        raise ValueError(
            f"rho and velocity snapshots must have shape {expected_shape}"
        )
    final_rho = (
        np.asarray(snapshots["final_rho"], dtype=float)
        if "final_rho" in snapshots
        else None
    )
    final_edge = (
        np.asarray(snapshots["final_edge"], dtype=float)
        if "final_edge" in snapshots
        else None
    )
    if final_rho is not None and final_rho.shape != (parameters.grid_size,):
        raise ValueError("final rho snapshot has the wrong shape")
    if final_edge is not None and final_edge.shape != (
        parameters.grid_size,
        parameters.grid_size,
    ):
        raise ValueError("final edge snapshot has the wrong shape")
    return DensityVelocitySeries(
        request_id=str(response["request_id"]),
        parameters=parameters,
        x=x,
        time=time_values,
        rho=rho,
        velocity=velocity,
        pathway=float(summary.get("pathway", math.nan)),
        final_rho=final_rho,
        final_edge=final_edge,
        max_node_mass_residual=float(
            diagnostics.get("max_node_mass_residual", math.nan)
        ),
        max_fixed_degree_residual=float(
            diagnostics.get("max_fixed_degree_residual", math.nan)
        ),
    )


def multimetric_series(
    response: dict[str, object],
    parameters: KineticParameters,
) -> MultimetricSeries:
    """Decode energy observables and final density fields from one response."""

    result = response.get("result")
    if not isinstance(result, dict):
        raise TypeError("Go response does not contain a result object")
    series = result.get("series")
    summary = result.get("summary")
    snapshots = result.get("snapshots")
    diagnostics = result.get("diagnostics", {})
    if not isinstance(series, dict) or not isinstance(summary, dict):
        raise TypeError("Go response has invalid scalar series or summary")
    if not isinstance(snapshots, dict) or not isinstance(diagnostics, dict):
        raise TypeError("Go response has invalid final snapshots or diagnostics")

    time_values = np.asarray(series.get("time", ()), dtype=float)
    node_energy = np.asarray(series.get("node_energy", ()), dtype=float)
    edge_energy = np.asarray(series.get("edge_energy", ()), dtype=float)
    expected_shape = (time_values.size,)
    if time_values.size < 2 or node_energy.shape != expected_shape:
        raise ValueError("node energy and time must have the same nontrivial shape")
    if edge_energy.shape != expected_shape:
        raise ValueError("edge energy and time must have the same shape")
    if not np.all(np.isfinite(time_values)) or np.any(np.diff(time_values) <= 0):
        raise ValueError("energy observation times must be finite and increasing")
    if not np.all(np.isfinite(node_energy)) or not np.all(np.isfinite(edge_energy)):
        raise ValueError("energy observations must be finite")

    final_rho = np.asarray(snapshots.get("final_rho", ()), dtype=float)
    final_edge = np.asarray(snapshots.get("final_edge", ()), dtype=float)
    size = parameters.grid_size
    if final_rho.shape != (size,):
        raise ValueError("final rho snapshot has the wrong shape")
    if final_edge.shape != (size, size):
        raise ValueError("final edge snapshot has the wrong shape")
    pathway = float(summary.get("pathway", math.nan))
    if not math.isfinite(pathway):
        raise ValueError("multimetric response is missing a finite pathway")

    return MultimetricSeries(
        request_id=str(response["request_id"]),
        parameters=parameters,
        time=time_values,
        node_energy=node_energy,
        edge_energy=edge_energy,
        pathway=pathway,
        final_rho=final_rho,
        final_edge=final_edge,
        max_node_mass_residual=float(
            diagnostics.get("max_node_mass_residual", math.nan)
        ),
        max_fixed_degree_residual=float(
            diagnostics.get("max_fixed_degree_residual", math.nan)
        ),
    )


def solve_trajectory(
    parameters: KineticParameters,
    *,
    record_steps: Sequence[int] | None = None,
    binary_path: PathLike[str] | str | None = None,
    resolution: KineticResolution | None = None,
) -> KineticTrajectory:
    from smp_meso_bindings import run_kinetic

    resolution = resolution or KineticResolution()
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


def solve_density_velocity_batch(
    binary_path: PathLike[str] | str,
    cases: list[tuple[str, KineticParameters, Sequence[int]]],
    resolution: KineticResolution,
    processes: int,
    progress: ProgressCallback | None,
    *,
    final_snapshot_fields: Sequence[str] = (),
) -> list[DensityVelocitySeries]:
    """Run cases while transferring only fields needed by landscape metrics."""

    from smp_meso_bindings import run_kinetic_batch, run_kinetic_batch_parallel

    if processes < 1:
        raise ValueError("processes must be positive")
    requests = [
        kinetic_request(
            key,
            parameters,
            resolution,
            record_steps=record_steps,
            snapshot_fields=("rho", "velocity"),
            final_snapshot_fields=final_snapshot_fields,
        )
        for key, parameters, record_steps in cases
    ]
    if processes == 1:
        responses = run_kinetic_batch(binary_path, requests, progress=progress)
    else:
        responses = run_kinetic_batch_parallel(
            binary_path, requests, min(processes, len(requests)), progress=progress
        )
    return [
        density_velocity_series(response, parameters, resolution)
        for response, (_, parameters, _) in zip(responses, cases, strict=True)
    ]


def solve_multimetric_batch(
    binary_path: PathLike[str] | str,
    cases: list[tuple[str, KineticParameters]],
    resolution: KineticResolution,
    processes: int,
    progress: ProgressCallback | None,
) -> list[MultimetricSeries]:
    """Run energy histories, online terminal pathway, and final density snapshots."""

    from smp_meso_bindings import run_kinetic_batch, run_kinetic_batch_parallel

    if processes < 1:
        raise ValueError("processes must be positive")
    requests = [
        kinetic_request(
            key,
            parameters,
            resolution,
            observable_fields=("pathway", "node_energy", "edge_energy"),
            final_snapshot_fields=("rho", "edge"),
        )
        for key, parameters in cases
    ]
    if processes == 1:
        responses = run_kinetic_batch(binary_path, requests, progress=progress)
    else:
        responses = run_kinetic_batch_parallel(
            binary_path,
            requests,
            min(processes, len(requests)),
            progress=progress,
        )
    return [
        multimetric_series(response, parameters)
        for response, (_, parameters) in zip(responses, cases, strict=True)
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
