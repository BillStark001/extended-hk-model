"""Scientific response adapters for contour experiments."""

from __future__ import annotations

import hashlib
import math
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ehk.metrics import potential_from_force, quantify_multiwell_series
from ehk.modeling.mesoscopic import (
    KineticParameters,
    KineticResolution,
    ObservableThresholds,
    solve_density_velocity_batch,
    solve_multimetric_batch,
    solve_observable_batch,
)

from .acquisition import Action
from .protocol import ContourProtocol
from .store import Evaluation


def total_variation_from_counts(
    generator_counts: Sequence[int],
    microscopic_counts: Sequence[int],
) -> tuple[float, float]:
    """Return TV and a multinomial delta-method sampling variance."""

    generator = np.asarray(generator_counts, dtype=float)
    microscopic = np.asarray(microscopic_counts, dtype=float)
    if generator.shape != microscopic.shape or generator.ndim != 1 or generator.size < 2:
        raise ValueError("count vectors must have equal one-dimensional shape")
    if np.any(generator < 0) or np.any(microscopic < 0):
        raise ValueError("counts must be non-negative")
    generator_total = float(np.sum(generator))
    microscopic_total = float(np.sum(microscopic))
    if generator_total <= 0 or microscopic_total <= 0:
        raise ValueError("both count vectors need positive totals")
    p = generator / generator_total
    q = microscopic / microscopic_total
    difference = p - q
    value = 0.5 * float(np.sum(np.abs(difference)))
    sign = np.sign(difference)
    generator_variance = float(np.sum(p * sign**2) - np.sum(p * sign) ** 2)
    microscopic_variance = float(np.sum(q * sign**2) - np.sum(q * sign) ** 2)
    variance = 0.25 * (
        generator_variance / generator_total
        + microscopic_variance / microscopic_total
    )
    return value, max(variance, 0.0)


def terminal_density_l2_gaps(
    measure_rho: Sequence[float],
    fokker_planck_rho: Sequence[float],
    measure_edge: np.ndarray,
    fokker_planck_edge: np.ndarray,
    *,
    cell_width: float,
    out_degree: int,
) -> tuple[float, float]:
    """Continuous-L2 gaps for terminal node and normalized edge densities."""

    rho_measure = np.asarray(measure_rho, dtype=float)
    rho_fokker_planck = np.asarray(fokker_planck_rho, dtype=float)
    edge_measure = np.asarray(measure_edge, dtype=float)
    edge_fokker_planck = np.asarray(fokker_planck_edge, dtype=float)
    if (
        rho_measure.ndim != 1
        or rho_fokker_planck.shape != rho_measure.shape
        or edge_measure.shape != (rho_measure.size, rho_measure.size)
        or edge_fokker_planck.shape != edge_measure.shape
    ):
        raise ValueError("terminal density arrays have incompatible shapes")
    if cell_width <= 0 or out_degree < 1:
        raise ValueError("cell width and out degree must be positive")
    node_gap = float(np.linalg.norm(rho_measure - rho_fokker_planck)) / math.sqrt(
        cell_width
    )
    edge_gap = float(np.linalg.norm(edge_measure - edge_fokker_planck)) / (
        out_degree * cell_width
    )
    return node_gap, edge_gap


def hellinger_gap(
    left: Sequence[float] | np.ndarray,
    right: Sequence[float] | np.ndarray,
    *,
    expected_mass: float = 1.0,
    absolute_tolerance: float = 1e-9,
) -> float:
    """Return the dimension-independent Hellinger distance between masses."""

    left_mass = np.asarray(left, dtype=float)
    right_mass = np.asarray(right, dtype=float)
    if left_mass.shape != right_mass.shape or left_mass.size < 2:
        raise ValueError("distribution masses must have equal nontrivial shapes")
    if not np.all(np.isfinite(left_mass)) or not np.all(np.isfinite(right_mass)):
        raise ValueError("distribution masses must be finite")
    if float(np.min(left_mass)) < -absolute_tolerance or float(
        np.min(right_mass)
    ) < -absolute_tolerance:
        raise ValueError("distribution contains material negative mass")
    if expected_mass <= 0 or absolute_tolerance <= 0:
        raise ValueError("expected mass and tolerance must be positive")

    left_mass = np.clip(left_mass, 0.0, None)
    right_mass = np.clip(right_mass, 0.0, None)
    left_total = float(np.sum(left_mass))
    right_total = float(np.sum(right_mass))
    if not math.isclose(
        left_total, expected_mass, rel_tol=0.0, abs_tol=absolute_tolerance
    ):
        raise ValueError(f"left distribution mass is {left_total}, expected {expected_mass}")
    if not math.isclose(
        right_total, expected_mass, rel_tol=0.0, abs_tol=absolute_tolerance
    ):
        raise ValueError(
            f"right distribution mass is {right_total}, expected {expected_mass}"
        )
    left_probability = left_mass / left_total
    right_probability = right_mass / right_total
    value = math.sqrt(
        0.5
        * float(
            np.sum(
                (np.sqrt(left_probability) - np.sqrt(right_probability)) ** 2
            )
        )
    )
    return min(max(value, 0.0), 1.0)


def normalized_time_l1_gap(
    time_values: Sequence[float],
    measure_values: Sequence[float],
    fokker_planck_values: Sequence[float],
) -> tuple[float, float]:
    """Return a baseline-normalized time-average absolute trajectory gap."""

    time_axis = np.asarray(time_values, dtype=float)
    measure = np.asarray(measure_values, dtype=float)
    fokker_planck = np.asarray(fokker_planck_values, dtype=float)
    if time_axis.ndim != 1 or time_axis.size < 2:
        raise ValueError("energy integration requires at least two time points")
    if measure.shape != time_axis.shape or fokker_planck.shape != time_axis.shape:
        raise ValueError("energy series must share one-dimensional time coordinates")
    if not np.all(np.isfinite(time_axis)) or np.any(np.diff(time_axis) <= 0):
        raise ValueError("energy observation times must be finite and increasing")
    if not np.all(np.isfinite(measure)) or not np.all(np.isfinite(fokker_planck)):
        raise ValueError("energy series must be finite")
    initial = 0.5 * float(measure[0] + fokker_planck[0])
    tolerance = 1e-12 * max(1.0, abs(initial))
    if initial <= 0:
        raise ValueError("initial energy scale must be positive")
    if not math.isclose(
        float(measure[0]),
        float(fokker_planck[0]),
        rel_tol=0.0,
        abs_tol=tolerance,
    ):
        raise ValueError("paired initial energies differ")
    duration = float(time_axis[-1] - time_axis[0])
    gap = float(
        np.trapezoid(np.abs(measure - fokker_planck), x=time_axis)
        / (duration * initial)
    )
    return gap, initial


def common_terminal_time(items: Sequence[Any], expected: float) -> float:
    """Require paired trajectories and their final snapshots to share one time."""

    terminal_times: list[float] = []
    for item in items:
        time_values = np.asarray(item.time, dtype=float)
        if time_values.size == 0:
            raise ValueError("terminal comparison requires a recorded terminal time")
        terminal_times.append(float(time_values[-1]))
    if any(not math.isclose(value, expected, rel_tol=0.0, abs_tol=1e-10) for value in terminal_times):
        raise ValueError(
            f"terminal snapshots must be evaluated at T={expected}, got {terminal_times}"
        )
    if any(
        not math.isclose(value, terminal_times[0], rel_tol=0.0, abs_tol=1e-12)
        for value in terminal_times[1:]
    ):
        raise ValueError(f"paired terminal times differ: {terminal_times}")
    return terminal_times[0]


class OperatorGapEvaluator:
    """Evaluate log10 absolute pathway gap for measure versus FPE."""

    def __init__(self, binary: str | Path):
        self.binary = Path(binary)
        if not self.binary.is_file():
            raise FileNotFoundError(self.binary)
        self.binary_sha256 = hashlib.sha256(self.binary.read_bytes()).hexdigest()

    @staticmethod
    def _parameters(
        action: Action,
        protocol: ContourProtocol,
        method: str,
    ) -> KineticParameters:
        simulator = protocol.raw["simulator"]
        group = simulator["group_parameters"][action.group]
        return KineticParameters(
            epsilon=float(simulator["epsilon"]),
            influence=10.0 ** action.x[0],
            rewiring=10.0 ** action.x[1],
            dynamics=str(group["dynamics"]),
            opinion_method=method,
            mean_degree=int(simulator["mean_degree"]),
            recsys_count=int(simulator["recommendation_count"]),
            recsys=str(group["recommender"]["type"]),
            opinion_tolerance=float(group["recommender"]["opinion_tolerance"]),
            recommendation_steepness=float(group["recommender"]["steepness"]),
            recommendation_random_ratio=float(group["recommender"]["random_ratio"]),
            noise_diffusion=float(simulator["noise_diffusion"]),
            grid_size=action.fidelity,
            dt=float(simulator["dt"]),
            steps=int(simulator["steps"]),
            record_every=int(simulator["record_every"]),
            confidence_mode=str(simulator["confidence_mode"]),
        )

    @staticmethod
    def _resolution(protocol: ContourProtocol) -> KineticResolution:
        value = protocol.raw["simulator"]["resolution"]
        return KineticResolution(
            population=int(value["population"]),
            opinion_min=float(value["opinion_min"]),
            opinion_max=float(value["opinion_max"]),
            opinion_quadrature_points=int(value["opinion_quadrature_points"]),
            opinion_quadrature_rule=str(value["opinion_quadrature_rule"]),
            confidence_quadrature_points=int(value["confidence_quadrature_points"]),
            score_max=int(value["score_max"]),
            distance_grid_size=int(value["distance_grid_size"]),
            minimum_bandwidth=float(value["minimum_bandwidth"]),
            objective_effective_samples=int(value["objective_effective_samples"]),
        )

    def __call__(self, action: Action, protocol: ContourProtocol) -> Evaluation:
        expected_digest = str(protocol.raw["simulator"].get("runtime_binary_sha256", ""))
        if not expected_digest or self.binary_sha256 != expected_digest:
            raise ValueError("smp-kinetic binary does not match protocol runtime_binary_sha256")
        if protocol.response.get("observable") != "pathway_integral_completion":
            raise ValueError("OperatorGapEvaluator requires pathway_integral_completion")
        methods = tuple(protocol.response.get("methods", ()))
        if methods != ("measure", "fokker_planck"):
            raise ValueError("operator response methods must be [measure,fokker_planck]")
        cases = [
            (
                f"{action.key(protocol)}/{method}",
                self._parameters(action, protocol, method),
            )
            for method in methods
        ]
        simulator = protocol.raw["simulator"]
        thresholds = ObservableThresholds(
            float(simulator["thresholds"]["polarization"]),
            float(simulator["thresholds"]["homophily"]),
        )
        started = time.monotonic()
        output = solve_observable_batch(
            self.binary,
            cases,
            self._resolution(protocol),
            thresholds,
            1,
            None,
        )
        elapsed = time.monotonic() - started
        values = {method: item.pathway for method, item in zip(methods, output, strict=True)}
        diagnostics = {
            method: {
                "max_node_mass_residual": item.max_node_mass_residual,
                "max_fixed_degree_residual": item.max_fixed_degree_residual,
            }
            for method, item in zip(methods, output, strict=True)
        }
        gap = abs(values["measure"] - values["fokker_planck"])
        floors = protocol.response["numerical_floor_by_fidelity"]
        floor = float(floors[str(action.fidelity)])
        common = {
            "protocol_sha256": protocol.fingerprint,
            "group": action.group,
            "fidelity": action.fidelity,
            "x": action.x,
            "replicate": action.replicate,
            "role": action.role,
            "runtime_seconds": elapsed,
            "payload": {
                "raw": values,
                "absolute_gap": gap,
                "numerical_floor": floor,
                "conservation": diagnostics,
            },
        }
        if gap <= floor:
            return Evaluation(
                **common,
                censoring="left",
                censor_bound=math.log10(floor),
            )
        return Evaluation(**common, value=math.log10(gap))


class LandscapeBarrierGapEvaluator(OperatorGapEvaluator):
    """Evaluate the log10 absolute peak dominant-barrier method gap."""

    @staticmethod
    def _record_steps(landscape: dict[str, object], simulator_steps: int) -> tuple[int, ...]:
        explicit = landscape.get("record_steps")
        schedule = landscape.get("record_schedule")
        if (explicit is None) == (schedule is None):
            raise ValueError(
                "landscape requires exactly one of record_steps and record_schedule"
            )
        if explicit is not None:
            record_steps = tuple(int(item) for item in explicit)
        else:
            if not isinstance(schedule, dict):
                raise ValueError("landscape.record_schedule must be an object")
            dense_through = int(schedule["dense_through"])
            dense_stride = int(schedule["dense_stride"])
            coarse_stride = int(schedule["coarse_stride"])
            if not 0 <= dense_through <= simulator_steps:
                raise ValueError("dense_through must lie within simulator steps")
            if dense_stride < 1 or coarse_stride < 1:
                raise ValueError("record strides must be positive")
            record_steps = tuple(
                sorted(
                    set(range(0, dense_through + 1, dense_stride))
                    | set(range(dense_through, simulator_steps + 1, coarse_stride))
                    | {simulator_steps}
                )
            )
        if (
            not record_steps
            or record_steps[0] < 0
            or record_steps[-1] != simulator_steps
            or tuple(sorted(set(record_steps))) != record_steps
        ):
            raise ValueError(
                "landscape record steps must be unique, increasing, and end at simulator.steps"
            )
        return record_steps

    def __call__(self, action: Action, protocol: ContourProtocol) -> Evaluation:
        expected_digest = str(protocol.raw["simulator"].get("runtime_binary_sha256", ""))
        if not expected_digest or self.binary_sha256 != expected_digest:
            raise ValueError("smp-kinetic binary does not match protocol runtime_binary_sha256")
        if protocol.response.get("observable") != "peak_dominant_barrier_height":
            raise ValueError(
                "LandscapeBarrierGapEvaluator requires peak_dominant_barrier_height"
            )
        methods = tuple(protocol.response.get("methods", ()))
        if methods != ("measure", "fokker_planck"):
            raise ValueError("landscape response methods must be [measure,fokker_planck]")
        landscape = protocol.response.get("landscape", {})
        if not isinstance(landscape, dict):
            raise TypeError("response.landscape must be an object")
        simulator_steps = int(protocol.raw["simulator"]["steps"])
        record_steps = self._record_steps(landscape, simulator_steps)
        cases = [
            (
                f"{action.key(protocol)}/{method}",
                self._parameters(action, protocol, method),
                record_steps,
            )
            for method in methods
        ]
        started = time.monotonic()
        processes = int(protocol.raw["simulator"].get("processes_per_evaluation", 1))
        if processes < 1:
            raise ValueError("simulator.processes_per_evaluation must be positive")
        fields = solve_density_velocity_batch(
            self.binary,
            cases,
            self._resolution(protocol),
            min(processes, len(cases)),
            None,
        )
        elapsed = time.monotonic() - started
        raw: dict[str, float] = {}
        diagnostics: dict[str, dict[str, float | int | str]] = {}
        for method, item in zip(methods, fields, strict=True):
            force = item.velocity / (10.0 ** action.x[0])
            potential = potential_from_force(item.x, force)
            series = quantify_multiwell_series(
                item.x,
                item.time,
                potential,
                item.rho,
                min_basin_mass=float(landscape["min_basin_mass"]),
                min_barrier_height=float(landscape["min_barrier_height"]),
                min_prominence=float(landscape.get("min_prominence", 1e-8)),
                relative_prominence=float(
                    landscape.get("relative_prominence", 1e-3)
                ),
                max_well_displacement=float(landscape["max_well_displacement"]),
                dominant_score_margin=float(landscape["dominant_score_margin"]),
                dominant_switch_persistence=int(
                    landscape["dominant_switch_persistence"]
                ),
                overshoot_tolerance=float(landscape["overshoot_tolerance"]),
            )
            raw[method] = series.peak_barrier_height
            diagnostics[method] = {
                "peak_time": series.peak_time,
                "final_barrier_height": series.final_barrier_height,
                "overshoot_class": series.overshoot_class,
                "dominant_pair_switch_count": int(np.sum(series.dominant_switch)),
                "max_node_mass_residual": item.max_node_mass_residual,
                "max_fixed_degree_residual": item.max_fixed_degree_residual,
            }
        gap = abs(raw["measure"] - raw["fokker_planck"])
        floor = float(
            protocol.response["numerical_floor_by_fidelity"][str(action.fidelity)]
        )
        common = {
            "protocol_sha256": protocol.fingerprint,
            "group": action.group,
            "fidelity": action.fidelity,
            "x": action.x,
            "replicate": action.replicate,
            "role": action.role,
            "runtime_seconds": elapsed,
            "payload": {
                "raw": raw,
                "absolute_gap": gap,
                "numerical_floor": floor,
                "landscape": diagnostics,
            },
        }
        if gap <= floor:
            return Evaluation(
                **common,
                censoring="left",
                censor_bound=math.log10(floor),
            )
        return Evaluation(**common, value=math.log10(gap))


class PairedMultimetricEvaluator(OperatorGapEvaluator):
    """Evaluate five shared measure--FPE discrepancy fields per dynamics."""

    metric_names = (
        "node_hellinger",
        "edge_hellinger",
        "u_rho_time_gap",
        "u_e_time_gap",
        "pathway_gap",
    )

    @staticmethod
    def _parameters_for_dynamics(
        x: tuple[float, ...],
        fidelity: int,
        protocol: ContourProtocol,
        dynamics: str,
        method: str,
    ) -> KineticParameters:
        simulator = protocol.raw["simulator"]
        group = simulator["dynamics_parameters"][dynamics]
        recommender = group["recommender"]
        return KineticParameters(
            epsilon=float(simulator["epsilon"]),
            influence=10.0 ** x[0],
            rewiring=10.0 ** x[1],
            dynamics=dynamics,
            opinion_method=method,
            mean_degree=int(simulator["mean_degree"]),
            recsys_count=int(simulator["recommendation_count"]),
            recsys=str(recommender["type"]),
            opinion_tolerance=float(recommender["opinion_tolerance"]),
            recommendation_steepness=float(recommender["steepness"]),
            recommendation_random_ratio=float(recommender["random_ratio"]),
            noise_diffusion=float(simulator["noise_diffusion"]),
            grid_size=fidelity,
            dt=float(simulator["dt"]),
            steps=int(simulator["steps"]),
            record_every=int(simulator["record_every"]),
            confidence_mode=str(simulator["confidence_mode"]),
        )

    def __call__(
        self,
        x: tuple[float, ...],
        fidelity: int,
        protocol: ContourProtocol,
    ) -> dict[str, Any]:
        expected_digest = str(protocol.raw["simulator"].get("runtime_binary_sha256", ""))
        if not expected_digest or self.binary_sha256 != expected_digest:
            raise ValueError("smp-kinetic binary does not match protocol runtime_binary_sha256")
        if protocol.response.get("observable") != "paired_multimetric_validity":
            raise ValueError(
                "PairedMultimetricEvaluator requires paired_multimetric_validity"
            )
        methods = tuple(protocol.response.get("methods", ()))
        if methods != ("measure", "fokker_planck"):
            raise ValueError("multimetric methods must be [measure,fokker_planck]")
        dynamics_names = tuple(str(item) for item in protocol.response["dynamics"])
        if dynamics_names != ("hk", "deffuant"):
            raise ValueError("multimetric dynamics must be [hk,deffuant]")
        cases = [
            (
                f"multimetric/{dynamics}/{method}/{x}",
                self._parameters_for_dynamics(
                    x, fidelity, protocol, dynamics, method
                ),
            )
            for dynamics in dynamics_names
            for method in methods
        ]
        processes = int(protocol.raw["simulator"].get("processes_per_evaluation", 1))
        if processes < 1:
            raise ValueError("simulator.processes_per_evaluation must be positive")
        started = time.monotonic()
        fields = solve_multimetric_batch(
            self.binary,
            cases,
            self._resolution(protocol),
            min(processes, len(cases)),
            None,
        )
        elapsed = time.monotonic() - started
        case_labels = tuple(
            (dynamics, method)
            for dynamics in dynamics_names
            for method in methods
        )
        by_case = {
            label: item
            for item, label in zip(fields, case_labels, strict=True)
        }
        result: dict[str, Any] = {
            "x": list(x),
            "fidelity": fidelity,
            "runtime_seconds": elapsed,
            "dynamics": {},
        }
        floors = protocol.response["numerical_floor_by_metric"]
        for dynamics in dynamics_names:
            method_data: dict[str, dict[str, Any]] = {}
            for method in methods:
                item = by_case[(dynamics, method)]
                duration = float(item.time[-1] - item.time[0])
                method_data[method] = {
                    "pathway": item.pathway,
                    "u_rho_initial": float(item.node_energy[0]),
                    "u_e_initial": float(item.edge_energy[0]),
                    "u_rho_time_mean": float(
                        np.trapezoid(item.node_energy, x=item.time) / duration
                    ),
                    "u_e_time_mean": float(
                        np.trapezoid(item.edge_energy, x=item.time) / duration
                    ),
                    "max_node_mass_residual": item.max_node_mass_residual,
                    "max_fixed_degree_residual": item.max_fixed_degree_residual,
                }
            measure = by_case[(dynamics, "measure")]
            fokker_planck = by_case[(dynamics, "fokker_planck")]
            terminal_time = common_terminal_time(
                (measure, fokker_planck),
                float(protocol.raw["simulator"]["steps"])
                * float(protocol.raw["simulator"]["dt"]),
            )
            for method in methods:
                method_data[method]["terminal_time"] = terminal_time
            if not np.array_equal(measure.time, fokker_planck.time):
                raise ValueError("paired energy trajectories use different time axes")
            out_degree = int(protocol.raw["simulator"]["mean_degree"])
            node_hellinger = hellinger_gap(
                measure.final_rho, fokker_planck.final_rho
            )
            edge_hellinger = hellinger_gap(
                measure.final_edge / out_degree,
                fokker_planck.final_edge / out_degree,
            )
            u_rho_time_gap, u_rho_initial = normalized_time_l1_gap(
                measure.time,
                measure.node_energy,
                fokker_planck.node_energy,
            )
            u_e_time_gap, u_e_initial = normalized_time_l1_gap(
                measure.time,
                measure.edge_energy,
                fokker_planck.edge_energy,
            )
            raw_metrics = {
                "node_hellinger": node_hellinger,
                "edge_hellinger": edge_hellinger,
                "u_rho_time_gap": u_rho_time_gap,
                "u_e_time_gap": u_e_time_gap,
                "pathway_gap": abs(measure.pathway - fokker_planck.pathway),
            }
            transformed = {
                name: math.log10(max(value, float(floors[name])))
                for name, value in raw_metrics.items()
            }
            censored = {
                name: value <= float(floors[name])
                for name, value in raw_metrics.items()
            }
            result["dynamics"][dynamics] = {
                "methods": method_data,
                "normalization": {
                    "u_rho_initial": u_rho_initial,
                    "u_e_initial": u_e_initial,
                },
                "raw_metrics": raw_metrics,
                "log10_metrics": transformed,
                "left_censored": censored,
                "terminal_time": terminal_time,
            }
        return result
