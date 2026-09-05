"""Cost-aware contour-targeted acquisition over new/promote/replicate actions."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.special import ndtr
from scipy.stats import norm, qmc

from .gp import MaternGP
from .protocol import ContourProtocol
from .store import Evaluation, evaluation_key

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class Action:
    group: str
    fidelity: int
    x: tuple[float, ...]
    replicate: int
    kind: str
    role: str = "train"

    def key(self, protocol: ContourProtocol) -> str:
        return evaluation_key(
            protocol.fingerprint, self.group, self.fidelity, self.x, self.replicate
        )

    def as_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["x"] = list(self.x)
        return value

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> Action:
        fields = dict(value)
        fields["x"] = tuple(float(item) for item in fields["x"])
        return cls(**fields)


@dataclass(frozen=True)
class AcquisitionResult:
    action: Action
    score: float
    expected_reduction: float
    global_exploration: float
    estimated_cost: float


def sobol_points(protocol: ContourProtocol, count: int, seed: int) -> FloatArray:
    sampler = qmc.Sobol(protocol.dimension, scramble=True, seed=seed)
    power = math.ceil(math.log2(count))
    unit = sampler.random_base2(power)[:count]
    domain = np.asarray(protocol.domain, dtype=float)
    return qmc.scale(unit, domain[:, 0], domain[:, 1])


def _same_point(left: Sequence[float], right: Sequence[float]) -> bool:
    return tuple(float(item) for item in left) == tuple(float(item) for item in right)


def candidate_actions(
    protocol: ContourProtocol,
    evaluations: Sequence[Evaluation],
    *,
    seed_offset: int = 0,
) -> list[Action]:
    design = protocol.design
    existing = {
        (item.group, item.fidelity, tuple(item.x), item.replicate)
        for item in evaluations
    }
    points = sobol_points(
        protocol,
        int(design["candidate_count"]),
        int(design["seed"]) + 1009 + seed_offset,
    )
    actions: list[Action] = []
    for group in protocol.groups:
        for point_array in points:
            point = tuple(float(item) for item in point_array)
            for fidelity in protocol.fidelities:
                identity = (group, fidelity, point, 0)
                if identity not in existing:
                    actions.append(Action(group, fidelity, point, 0, "new"))

        successful = [item for item in evaluations if item.group == group and item.status == "ok"]
        for item in successful:
            fidelity_index = protocol.fidelities.index(item.fidelity)
            if fidelity_index + 1 < len(protocol.fidelities):
                promoted = protocol.fidelities[fidelity_index + 1]
                identity = (group, promoted, tuple(item.x), 0)
                if identity not in existing:
                    actions.append(Action(group, promoted, tuple(item.x), 0, "promote"))
            if (
                protocol.response["kind"] != "deterministic_scalar"
                and item.noise_variance > 0
            ):
                replicate = 1 + max(
                    candidate.replicate for candidate in successful
                    if candidate.fidelity == item.fidelity and _same_point(candidate.x, item.x)
                )
                identity = (group, item.fidelity, tuple(item.x), replicate)
                if identity not in existing:
                    actions.append(Action(group, item.fidelity, tuple(item.x), replicate, "replicate"))
    unique: dict[tuple[object, ...], Action] = {}
    for action in actions:
        unique[(action.group, action.fidelity, action.x, action.replicate)] = action
    return list(unique.values())


def estimate_cost(protocol: ContourProtocol, evaluations: Sequence[Evaluation], fidelity: int) -> float:
    observed = [
        item.runtime_seconds for item in evaluations
        if item.fidelity == fidelity and item.status == "ok" and item.runtime_seconds > 0
    ]
    if observed:
        return max(float(np.median(observed)), 1e-6)
    costs = protocol.design.get("fidelity_costs", {})
    if str(fidelity) in costs:
        return float(costs[str(fidelity)])
    return (fidelity / protocol.fidelities[0]) ** 2


def integrated_misclassification_risk(
    prediction_mean: FloatArray,
    prediction_variance: FloatArray,
    levels: Sequence[float],
    weights: Sequence[float],
) -> float:
    standard = np.sqrt(np.maximum(prediction_variance, 1e-15))
    risk = np.zeros_like(standard)
    for level, weight in zip(levels, weights, strict=True):
        risk += weight * ndtr(-np.abs(prediction_mean - level) / standard)
    return float(np.mean(risk) / sum(weights))


def propose_action(
    protocol: ContourProtocol,
    evaluations: Sequence[Evaluation],
    models: dict[str, MaternGP],
    *,
    seed_offset: int = 0,
) -> AcquisitionResult:
    actions = candidate_actions(protocol, evaluations, seed_offset=seed_offset)
    if not actions:
        raise RuntimeError("no unevaluated actions remain")
    integration = sobol_points(
        protocol,
        int(protocol.design["integration_count"]),
        int(protocol.design["seed"]) + 7919,
    )
    target_fidelity = protocol.fidelities[-1]
    exploration_weight = float(protocol.design["exploration_weight"])
    results: list[AcquisitionResult] = []
    for group in protocol.groups:
        model = models.get(group)
        if model is None:
            continue
        target_levels = [target_fidelity] * len(integration)
        prediction = model.predict(integration, target_levels)
        standard = np.sqrt(np.maximum(prediction.variance, 1e-15))
        contour_weight = np.zeros(len(integration))
        for level, weight in zip(protocol.levels, protocol.level_weights, strict=True):
            contour_weight += weight * norm.pdf((prediction.mean - level) / standard) / standard
        group_actions = [action for action in actions if action.group == group]
        action_points = [action.x for action in group_actions]
        action_fidelities = [action.fidelity for action in group_actions]
        candidate_prediction = model.predict(
            action_points,
            action_fidelities,
            include_noise=True,
        )
        covariance = model.posterior_covariance(
            integration,
            target_levels,
            action_points,
            action_fidelities,
        )
        for index, action in enumerate(group_actions):
            action_covariance = covariance[:, index]
            reduction = np.minimum(
                action_covariance * action_covariance
                / max(float(candidate_prediction.variance[index]), 1e-15),
                prediction.variance,
            )
            expected = float(np.mean(contour_weight * reduction))
            global_exploration = float(np.mean(reduction))
            cost = estimate_cost(protocol, evaluations, action.fidelity)
            score = (expected + exploration_weight * global_exploration) / cost
            if math.isfinite(score):
                results.append(AcquisitionResult(action, score, expected, global_exploration, cost))
    if not results:
        raise RuntimeError("no group has a fitted model for acquisition")
    return max(results, key=lambda item: (item.score, item.action.key(protocol)))
